# coding=utf-8
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" PyTorch LLaMA model."""
import copy
import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "5"
import math
from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, DynamicCache, StaticCache
from transformers.modeling_attn_mask_utils import AttentionMaskConverter
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast, SequenceClassifierOutputWithPast
from transformers.modeling_utils import PreTrainedModel
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
from transformers.utils import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    is_flash_attn_2_available,
    is_flash_attn_greater_or_equal_2_10,
    logging,
    replace_return_docstrings,
)
from transformers import LlamaConfig
from transformers import LlamaPreTrainedModel
from transformers.models.llama.modeling_llama import LlamaDecoderLayer, LlamaRMSNorm

if is_flash_attn_2_available():
    from flash_attn import flash_attn_func, flash_attn_varlen_func
    from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input  # noqa

logger = logging.get_logger(__name__)

class LlamaModel(LlamaPreTrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`LlamaDecoderLayer`]

    Args:
        config: LlamaConfig
    """

    def __init__(self, config: LlamaConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [LlamaDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.gradient_checkpointing = False

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
            )

        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
            )
            use_cache = False

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        past_seen_tokens = 0
        if use_cache:  # kept for BC (cache positions)
            if not isinstance(past_key_values, StaticCache):
                past_key_values = DynamicCache.from_legacy_cache(past_key_values)
                past_seen_tokens = past_key_values.get_seq_length()

        if cache_position is None:
            if isinstance(past_key_values, StaticCache):
                raise ValueError("cache_position is a required argument when using StaticCache.")
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = self._update_causal_mask(attention_mask, inputs_embeds, cache_position)

        # embed positions
        hidden_states = inputs_embeds

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None

        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    causal_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                    cache_position,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states_onnorm = hidden_states.clone()
        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = None
        if use_cache:
            next_cache = (
                next_decoder_cache.to_legacy_cache() if isinstance(next_decoder_cache, Cache) else next_decoder_cache
            )
        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        ), hidden_states_onnorm

    # TODO: As of torch==2.2.0, the `attention_mask` passed to the model in `generate` is 2D and of dynamic length even when the static
    # KV cache is used. This is an issue for torch.compile which then recaptures cudagraphs at each decode steps due to the dynamic shapes.
    # (`recording cudagraph tree for symint key 13`, etc.), which is VERY slow. A workaround is `@torch.compiler.disable`, but this prevents using
    # `fullgraph=True`. See more context in https://github.com/huggingface/transformers/pull/29114
    def _update_causal_mask(self, attention_mask, input_tensor, cache_position):
        if self.config._attn_implementation == "flash_attention_2":
            if attention_mask is not None and 0.0 in attention_mask:
                return attention_mask
            return None

        dtype, device = input_tensor.dtype, input_tensor.device
        min_dtype = torch.finfo(dtype).min
        sequence_length = input_tensor.shape[1]
        if hasattr(getattr(self.layers[0], "self_attn", {}), "past_key_value"):  # static cache
            target_length = self.config.max_position_embeddings
        else:  # dynamic cache
            target_length = (
                attention_mask.shape[-1] if isinstance(attention_mask, torch.Tensor) else cache_position[-1] + 1
            )

        causal_mask = torch.full((sequence_length, target_length), fill_value=min_dtype, dtype=dtype, device=device)
        if sequence_length != 1:
            causal_mask = causal_mask.to(torch.float32)#改
            causal_mask = torch.triu(causal_mask, diagonal=1)
            causal_mask = causal_mask.to(device, dtype=dtype)#改
            # causal_mask = torch.triu(causal_mask, diagonal=1)
        causal_mask *= torch.arange(target_length, device=device) > cache_position.reshape(-1, 1)
        causal_mask = causal_mask[None, None, :, :].expand(input_tensor.shape[0], 1, -1, -1)
        if attention_mask is not None:
            causal_mask = causal_mask.clone()  # copy to contiguous memory for in-place edit
            if attention_mask.dim() == 2:
                mask_length = attention_mask.shape[-1]
                padding_mask = causal_mask[..., :mask_length].eq(0.0) * attention_mask[:, None, None, :].eq(0.0)
                causal_mask[..., :mask_length] = causal_mask[..., :mask_length].masked_fill(padding_mask, min_dtype)
            elif attention_mask.dim() == 4:
                # backwards compatibility: we allow passing a 4D attention mask shorter than the input length with
                # cache. In that case, the 4D attention mask attends to the newest tokens only.
                if attention_mask.shape[-2] < cache_position[0] + sequence_length:
                    offset = cache_position[0]
                else:
                    offset = 0
                mask_shape = attention_mask.shape
                mask_slice = (attention_mask.eq(0.0)).to(dtype=dtype) * min_dtype
                causal_mask[
                    : mask_shape[0], : mask_shape[1], offset : mask_shape[2] + offset, : mask_shape[3]
                ] = mask_slice

        if (
            self.config._attn_implementation == "sdpa"
            and attention_mask is not None
            and attention_mask.device.type == "cuda"
        ):
            # TODO: For dynamo, rather use a check on fullgraph=True once this is possible (https://github.com/pytorch/pytorch/pull/120400).
            is_tracing = (
                torch.jit.is_tracing()
                or isinstance(input_tensor, torch.fx.Proxy)
                or (hasattr(torch, "_dynamo") and torch._dynamo.is_compiling())
            )
            if not is_tracing and torch.any(attention_mask != 1):
                # Attend to all tokens in fully masked rows in the causal_mask, for example the relevant first rows when
                # using left padding. This is required by F.scaled_dot_product_attention memory-efficient attention path.
                # Details: https://github.com/pytorch/pytorch/issues/110213
                causal_mask = AttentionMaskConverter._unmask_unattended(causal_mask, min_dtype)

        return causal_mask


class LlamaForCausalLM(LlamaPreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.model = LlamaModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, LlamaForCausalLM

        >>> model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
        >>> tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs, hidden_states_onnorm = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )

        hidden_states = outputs[0]
        if self.config.pretraining_tp > 1:
            lm_head_slices = self.lm_head.weight.split(self.vocab_size // self.config.pretraining_tp, dim=0)
            logits = [F.linear(hidden_states, lm_head_slices[i]) for i in range(self.config.pretraining_tp)]
            logits = torch.cat(logits, dim=-1)
        else:
            logits = self.lm_head(hidden_states)
        logits = logits.float()

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        ), hidden_states_onnorm

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, cache_position=None, **kwargs
    ):
        # With static cache, the `past_key_values` is None
        # TODO joao: standardize interface for the different Cache classes and remove of this if
        has_static_cache = False
        if past_key_values is None:
            past_key_values = getattr(getattr(self.model.layers[0], "self_attn", {}), "past_key_value", None)
            has_static_cache = past_key_values is not None

        past_length = 0
        if past_key_values is not None:
            if isinstance(past_key_values, Cache):
                past_length = cache_position[0] if cache_position is not None else past_key_values.get_seq_length()
                max_cache_length = (
                    torch.tensor(past_key_values.get_max_length(), device=input_ids.device)
                    if past_key_values.get_max_length() is not None
                    else None
                )
                cache_length = past_length if max_cache_length is None else torch.min(max_cache_length, past_length)
            # TODO joao: remove this `else` after `generate` prioritizes `Cache` objects
            else:
                cache_length = past_length = past_key_values[0][0].shape[2]
                max_cache_length = None

            # Keep only the unprocessed tokens:
            # 1 - If the length of the attention_mask exceeds the length of input_ids, then we are in a setting where
            # some of the inputs are exclusively passed as part of the cache (e.g. when passing input_embeds as
            # input)
            if attention_mask is not None and attention_mask.shape[1] > input_ids.shape[1]:
                input_ids = input_ids[:, -(attention_mask.shape[1] - past_length) :]
            # 2 - If the past_length is smaller than input_ids', then input_ids holds all input tokens. We can discard
            # input_ids based on the past_length.
            elif past_length < input_ids.shape[1]:
                input_ids = input_ids[:, past_length:]
            # 3 - Otherwise (past_length >= input_ids.shape[1]), let's assume input_ids only has unprocessed tokens.

            # If we are about to go beyond the maximum cache length, we need to crop the input attention mask.
            if (
                max_cache_length is not None
                and attention_mask is not None
                and cache_length + input_ids.shape[1] > max_cache_length
            ):
                attention_mask = attention_mask[:, -max_cache_length:]

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1] :]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            # The `contiguous()` here is necessary to have a static stride during decoding. torchdynamo otherwise
            # recompiles graphs as the stride of the inputs is a guard. Ref: https://github.com/huggingface/transformers/pull/29114
            # TODO: use `next_tokens` directly instead.
            model_inputs = {"input_ids": input_ids.contiguous()}

        input_length = position_ids.shape[-1] if position_ids is not None else input_ids.shape[-1]
        if cache_position is None:
            cache_position = torch.arange(past_length, past_length + input_length, device=input_ids.device)
        else:
            cache_position = cache_position[-input_length:]

        if has_static_cache:
            past_key_values = None

        model_inputs.update(
            {
                "position_ids": position_ids,
                "cache_position": cache_position,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
            }
        )
        return model_inputs

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past),
            )
        return reordered_past


import argparse
import copy

parser = argparse.ArgumentParser(description='sp')
parser.add_argument('--path', type=str, default="/cpfs01/user/xiaobin/glj/models/vicuna-7b-v1.5/")
parser.add_argument('--start', type=int, default=0)
parser.add_argument('--end', type=int, default=100)
parser.add_argument('--index', type=int, default=1)
parser.add_argument('--gpu_index', type=int, nargs='+', default=[0])
parser.add_argument('--outdir', type=str, default='outdir0')
parser.add_argument('--dataset', type=str, default="ShareGPT_Vicuna_unfiltered/ShareGPT_V4.3_unfiltered_cleaned_split.json")
args = parser.parse_args()
import os

#os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_index)[1:-1]
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import LlamaForCausalLM, AutoModelForCausalLM, AutoTokenizer,BitsAndBytesConfig
from datasets import load_dataset
import json
from fastchat.model.model_adapter import get_conversation_template

bigname=args.path
# bigname = "/home/lyh/weights/hf/llama/7B/"
# smallname = "/home/lyh/weights/hf/llama/7B/"



def longest_common_prefix(list1, list2):
    prefix_length = 0
    min_length = min(len(list1), len(list2))

    for i in range(min_length):
        if list1[i] == list2[i]:
            prefix_length += 1
        else:
            break

    common_prefix = list1[:prefix_length]
    return common_prefix, prefix_length


def build_dataset_rank(
        tokenizer, split="train",
        select=None,
):
    ds = load_dataset('json', data_files=args.dataset)
    # ds = load_dataset('json', data_files="/cpfs01/user/xiaobin/glj/datasets/Spe_eval_merge/Eval_data_split.json")
    ds = ds['train']
    ds = ds.shuffle(seed=42)
    ds1 = ds.select(range(args.start, args.end))
    original_columns1 = ds1.column_names
    num_proc = 4

    def preprocess_function(examples):
        new_examples = {
            "conversation":[],
            "input_ids": [],
            "loss_mask": [],
            "id": []
        }
        for i in range(len(examples['id'])):
            messages = [
                {"role": "system",
                 "content": "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."},
            ]
            convroles=["user","assistant"]
            roles = {"human": "user", "gpt": "assistant"}
            source= examples['conversations'][i]
            if roles[source[0]["from"]] != "user":
                # Skip the first one if it is not from human
                source = source[1:]
            for j, sentence in enumerate(source):
                role = roles[sentence["from"]]
                assert role == convroles[j % 2], f"{i}"
                if sentence["from"]=="gpt":
                    sentence["value"]=" "+sentence["value"]
                messages.append(
                    {"role": role, "content": sentence["value"]}
                )
            conversation=tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )

            if not tokenizer.pad_token_id:
                tokenizer.pad_token_id=tokenizer.unk_token_id

            input_ids = tokenizer(
                conversation,
                return_tensors="pt",
                max_length=2048,
                add_special_tokens=False,
            ).input_ids[0]
            loss_mask=torch.ones_like(input_ids)
            #print(i)

            sep = "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"

            total_len = len(input_ids)

            sep2="<|eot_id|><|start_header_id|>user<|end_header_id|>"
            turns = conversation.split(sep2)

            turns[1]=turns[0]+sep2+turns[1]
            turns=turns[1:]


            cur_len = 1
            loss_mask[:cur_len] = 0
            for i, turn in enumerate(turns):
                if turn == "":
                    break
                turn_len = len(tokenizer(turn).input_ids)

                parts = turn.split(sep)
                if len(parts) != 2:
                    break
                parts[0] += sep
                # "-2" is hardcoded for the Llama tokenizer to make the offset correct.
                instruction_len = len(tokenizer(parts[0]).input_ids) - 1


                # Ignore the user instructions
                if i==0:
                    loss_mask[cur_len: cur_len + instruction_len-2] = 0
                else:
                    loss_mask[cur_len-3: cur_len + instruction_len+1] = 0
                cur_len += turn_len
                if i!=0:
                    cur_len+=3

            loss_mask[cur_len:] = 0



            new_examples["conversation"].append(conversation)
            new_examples["input_ids"].append(input_ids[None,:])
            new_examples["loss_mask"].append(loss_mask[None,:])
            new_examples["id"].append(examples['id'][i])

        return new_examples

    ds1 = ds1.map(
        preprocess_function,
        batched=True,
        # num_proc=num_proc,
        remove_columns=original_columns1,
        load_from_cache_file=False
    )
    ds1.set_format(type="torch")
    return ds1

bigtokenizer = AutoTokenizer.from_pretrained(bigname,use_fast=False)
# quantization_config = BitsAndBytesConfig(
#         load_in_4bit=True,
#         bnb_4bit_compute_dtype=torch.bfloat16,
#         bnb_4bit_use_double_quant=True,
#         bnb_4bit_quant_type="nf4",
#     )
# bigmodel = AutoModelForCausalLM.from_pretrained(bigname, load_in_4bit=True, device_map={"": 0}, )
# smallmodel = AutoModelForCausalLM.from_pretrained(smallname, load_in_4bit=True, device_map={"": 1}, )
bigmodel = LlamaForCausalLM.from_pretrained(bigname,  device_map="auto",torch_dtype=torch.bfloat16)
#bigmodel = AutoModelForCausalLM.from_pretrained(bigname,  device_map="auto",load_in_8bit=True)
bigmodel.eval()
ds = build_dataset_rank(bigtokenizer)
print(ds)

@torch.no_grad()
def ge(data):
    input_ids=data["input_ids"]
    outs_big, hidden_state_onnorm = bigmodel(input_ids.cuda(), output_hidden_states=True)
    hidden_state_big = outs_big.hidden_states[-1]
    td={"id": data["id"], "input_ids":input_ids.cpu()[0],"hidden_state":hidden_state_big.cpu()[0],"hidden_state_onnorm":hidden_state_onnorm.cpu()[0],"loss_mask":data["loss_mask"].cpu()[0]}
    return td

outdir = f'{args.outdir}/{args.index}'
if not os.path.exists(outdir):
    os.makedirs(outdir)

def writedata(name,data_point):
    if not os.path.exists(name):
        os.makedirs(name)
    current_length=len(os.listdir(name))
    idx=current_length
    torch.save(data_point, f'{name}/data_{idx}.ckpt')


for id,data in enumerate(ds):
    if id%100==0:
        print(id,end="\t")
    if id % 1000 == 0:
        print("")
    outdata = ge(data)
    writedata(outdir,outdata)


