import copy
import json
import time

import torch
import torch.nn as nn
from transformers import PreTrainedModel, PretrainedConfig,AutoConfig
from .modeling_llama_kv import LlamaForCausalLM as KVLlamaForCausalLM
from .modeling_mixtral_kv import MixtralForCausalLM as KVMixtralForCausalLM
from .utils import *
from .kv_cache import initialize_past_key_values
from .choices import mc_sim_7b_63
from transformers import AutoTokenizer
import os
from huggingface_hub import hf_hub_download
from .cnets import Model
from .clover2 import Clover2Model, ConfigClover
from .configs import EConfig
from huggingface_hub import hf_hub_download

from safetensors.torch import load_file


class CloModel(nn.Module):

    def __init__(
            self,
            base_model,
            base_model_name_or_path,
            ea_model_path,
    ):

        super().__init__()
        self.base_model = base_model
        self.config = base_model.config
        self.hidden_size = base_model.lm_head.weight.shape[-1]
        self.vocab_size = base_model.lm_head.weight.shape[0]
        self.base_model_name_or_path = base_model_name_or_path
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_name_or_path)
        config = EConfig.from_pretrained(ea_model_path)
        with open(ea_model_path,"r") as f:
            con=json.loads(f.read())
        try:
            bias=con["bias"]
        except:
            bias=True
        
        self.ea_layer = Clover2Model(config, base_model.lm_head, load_emb=True, path = base_model_name_or_path)

        low_memory=False
        
        device = base_model.model.layers[-1].self_attn.q_proj.weight.device
        if device!=base_model.lm_head.weight.device:
            self.ea_layer.diff_device = True
            if not low_memory:
                self.ea_layer.lm_head = base_model.lm_head.to(device)
            else:
                self.ea_layer.layer_device = device

        else:
            self.ea_layer.diff_device = False
        self.ea_layer.to(self.base_model.dtype).to(device)
        self.ea_layer.init_tree()

    def get_tokenizer(self):
        """Get the tokenizer of the base model.

        Returns:
            Tokenizer: The tokenizer of the base model.
        """
        return self.tokenizer

    @classmethod
    def from_pretrained(
            cls,
            Type="LLaMA",
            base_model_path=None,
            ea_model_path=None,
            **kwargs,
    ):
        #assert Type=="LLaMA" or "Mixtral"
        Type=AutoConfig.from_pretrained(base_model_path).architectures[0]
        if Type=='LlamaForCausalLM':
            base_model = KVLlamaForCausalLM.from_pretrained(
                base_model_path, **kwargs
            )
        else:
            base_model = KVMixtralForCausalLM.from_pretrained(
                base_model_path, **kwargs
            )

        configpath=os.path.join(ea_model_path, "config.json")
        if not os.path.exists(configpath):
            configpath = hf_hub_download(ea_model_path, "config.json")
        model = cls(
            base_model,
            base_model_path,
            configpath
        )
        load_model_path = os.path.join(ea_model_path, "model.safetensors")

        ea_layer_state_dict = load_file(load_model_path, device="cuda")
        model.ea_layer.load_state_dict(ea_layer_state_dict, strict=False)
        # model.ea_layer.load_state_dict(ea_layer_state_dict, strict=True)

        return model

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            labels=None,
            past_key_values=None,
            output_orig=False,
            position_ids=None,
            init=True,
            logits_processor=None,
            topk_sample=False
    ):

        with torch.inference_mode():
            # Pass input through the base model
            outputs, hidden_states_onnorm = self.base_model.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                position_ids=position_ids,
            )
            if output_orig:
                orig = self.base_model.lm_head(outputs[0])
            hidden_states = outputs[0].clone()
        if init:
            if logits_processor is not None:
                logits = orig[:, -1]
                logits = logits_processor(None, logits)
                probabilities = torch.nn.functional.softmax(logits, dim=1)
                token = torch.multinomial(probabilities, 1)
            else:
                token = torch.argmax(orig[:, -1])
                token = token[None, None]
            input_ids = torch.cat((input_ids, token.to(input_ids.device)), dim=1)
            if topk_sample:
                ea_logits, tree_buffers = self.ea_layer.beam_genrate(hidden_states_onnorm, input_ids, self.base_model.lm_head, logits_processor)
                if output_orig:
                    return ea_logits, outputs, orig, hidden_states_onnorm, token, tree_buffers
                return ea_logits, hidden_states_onnorm, token, tree_buffers
            else:
                ea_logits = self.ea_layer.topK_genrate(hidden_states_onnorm, input_ids, self.base_model.lm_head, logits_processor)
                if output_orig:
                    return ea_logits, outputs, orig, hidden_states_onnorm, token
                return ea_logits, hidden_states_onnorm, token
        else:
            if output_orig:
                return outputs, orig, hidden_states_onnorm