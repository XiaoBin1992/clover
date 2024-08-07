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
import math
import numpy as np
from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from transformers import AutoModelForCausalLM, AutoConfig
from transformers.activations import ACT2FN
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast, SequenceClassifierOutputWithPast
from transformers.modeling_utils import PreTrainedModel
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
from transformers.utils import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)
try:
    from .configs import EConfig
    from .utils_c import *
    from .choices import *
except:
    from configs import EConfig
    from utils_c import *
    from choices import *
    from utils import prepare_logits_processor
top_k=10
try:
    from .utils import generate_beam_tree_buffers
except:
    from utils import generate_beam_tree_buffers
    
def print_tensor(name, tensor):
    variance = tensor.to(torch.float32).pow(2).mean(-1, keepdim=True)
    print(f"print_tensor {name}: {tensor.size()} {tensor.dtype} {torch.isnan(tensor).any()} {torch.max(tensor)} {torch.min(tensor)} variance:{variance.size()} {torch.min(variance)}")# {tensor} 
    

def print_param(name, param):
    from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled

    if is_deepspeed_zero3_enabled():
        from deepspeed.runtime.zero import GatheredParameters

        with GatheredParameters(param, modifier_rank=0):
            print(f"Gathered print_param {name}: {param.size()} {torch.isnan(param).any()} {torch.max(param)} {torch.min(param)}")
    else:
        print(f"print_param {name}: {param.size()} {torch.isnan(param).any()} {torch.max(param)} {torch.min(param)}")
            
# Copied from transformers.models.bart.modeling_bart._make_causal_mask
def _make_causal_mask(
    input_ids_shape: torch.Size, dtype: torch.dtype, device: torch.device, past_key_values_length: int = 0
):
    """
    Make causal mask used for bi-directional self-attention.
    """
    bsz, tgt_len = input_ids_shape
    mask = torch.full((tgt_len, tgt_len), torch.finfo(dtype).min, device=device)
    mask_cond = torch.arange(mask.size(-1), device=device)
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    mask = mask.to(dtype)

    if past_key_values_length > 0:
        mask = torch.cat([torch.zeros(tgt_len, past_key_values_length, dtype=dtype, device=device), mask], dim=-1)
    return mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len + past_key_values_length)


# Copied from transformers.models.bart.modeling_bart._expand_mask
def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None):
    """
    Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
    """
    bsz, src_len = mask.size()
    tgt_len = tgt_len if tgt_len is not None else src_len

    expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)

    inverted_mask = 1.0 - expanded_mask

    return inverted_mask.masked_fill(inverted_mask.to(torch.bool), torch.finfo(dtype).min)

def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)
def apply_rotary_pos_emb(q, k, cos, sin, position_ids):
    # The first two dimensions of cos and sin are always 1, so we can `squeeze` them.
    cos = cos.squeeze(1).squeeze(0)  # [seq_len, dim]
    sin = sin.squeeze(1).squeeze(0)  # [seq_len, dim]
    cos = cos[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    sin = sin[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

class LlamaRotaryEmbedding(torch.nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Build here to make `torch.jit.trace` work.
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings, device=self.inv_freq.device, dtype=torch.get_default_dtype()
        )

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)

        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :].to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :].to(dtype), persistent=False)

    def forward(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)

        return (
            self.cos_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
            self.sin_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
        )

class LlamaLinearScalingRotaryEmbedding(LlamaRotaryEmbedding):
    """LlamaRotaryEmbedding extended with linear scaling. Credits to the Reddit user /u/kaiokendev"""

    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None, scaling_factor=1.0):
        self.scaling_factor = scaling_factor
        super().__init__(dim, max_position_embeddings, base, device)

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)
        t = t / self.scaling_factor

        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :].to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :].to(dtype), persistent=False)


class LlamaDynamicNTKScalingRotaryEmbedding(LlamaRotaryEmbedding):
    """LlamaRotaryEmbedding extended with Dynamic NTK scaling. Credits to the Reddit users /u/bloc97 and /u/emozilla"""

    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None, scaling_factor=1.0):
        self.scaling_factor = scaling_factor
        super().__init__(dim, max_position_embeddings, base, device)

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len

        if seq_len > self.max_position_embeddings:
            base = self.base * (
                (self.scaling_factor * seq_len / self.max_position_embeddings) - (self.scaling_factor - 1)
            ) ** (self.dim / (self.dim - 2))
            inv_freq = 1.0 / (base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim))
            self.register_buffer("inv_freq", inv_freq, persistent=False)

        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)

        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :].to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :].to(dtype), persistent=False)

class LlamaAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)
        self._init_rope()

    def _init_rope(self):
        if self.config.rope_scaling is None:
            self.rotary_emb = LlamaRotaryEmbedding(self.head_dim, max_position_embeddings=self.max_position_embeddings)
        else:
            scaling_type = self.config.rope_scaling["type"]
            scaling_factor = self.config.rope_scaling["factor"]
            if scaling_type == "linear":
                self.rotary_emb = LlamaLinearScalingRotaryEmbedding(
                    self.head_dim, max_position_embeddings=self.max_position_embeddings, scaling_factor=scaling_factor
                )
            elif scaling_type == "dynamic":
                self.rotary_emb = LlamaDynamicNTKScalingRotaryEmbedding(
                    self.head_dim, max_position_embeddings=self.max_position_embeddings, scaling_factor=scaling_factor
                )
            else:
                raise ValueError(f"Unknown RoPE scaling type {scaling_type}")

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        if self.config.pretraining_tp > 1:
            key_value_slicing = (self.num_key_value_heads * self.head_dim) // self.config.pretraining_tp
            query_slices = self.q_proj.weight.split(
                (self.num_heads * self.head_dim) // self.config.pretraining_tp, dim=0
            )
            key_slices = self.k_proj.weight.split(key_value_slicing, dim=0)
            value_slices = self.v_proj.weight.split(key_value_slicing, dim=0)

            query_states = [F.linear(hidden_states, query_slices[i]) for i in range(self.config.pretraining_tp)]
            query_states = torch.cat(query_states, dim=-1)

            key_states = [F.linear(hidden_states, key_slices[i]) for i in range(self.config.pretraining_tp)]
            key_states = torch.cat(key_states, dim=-1)

            value_states = [F.linear(hidden_states, value_slices[i]) for i in range(self.config.pretraining_tp)]
            value_states = torch.cat(value_states, dim=-1)

        else:
            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2]
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        if past_key_value is not None:
            # reuse k, v, self_attention
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)

        past_key_value = (key_states, value_states) if use_cache else None

        # repeat k/v heads if n_kv_heads < n_heads
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
                f" {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights + attention_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        if self.config.pretraining_tp > 1:
            attn_output = attn_output.split(self.hidden_size // self.config.pretraining_tp, dim=2)
            o_proj_slices = self.o_proj.weight.split(self.hidden_size // self.config.pretraining_tp, dim=1)
            attn_output = sum([F.linear(attn_output[i], o_proj_slices[i]) for i in range(self.config.pretraining_tp)])
        else:
            attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


class LlamaMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        if self.config.pretraining_tp > 1:
            slice = self.intermediate_size // self.config.pretraining_tp
            gate_proj_slices = self.gate_proj.weight.split(slice, dim=0)
            up_proj_slices = self.up_proj.weight.split(slice, dim=0)
            down_proj_slices = self.down_proj.weight.split(slice, dim=1)

            gate_proj = torch.cat(
                [F.linear(x, gate_proj_slices[i]) for i in range(self.config.pretraining_tp)], dim=-1
            )
            up_proj = torch.cat([F.linear(x, up_proj_slices[i]) for i in range(self.config.pretraining_tp)], dim=-1)

            intermediate_states = (self.act_fn(gate_proj) * up_proj).split(slice, dim=2)
            down_proj = [
                F.linear(intermediate_states[i], down_proj_slices[i]) for i in range(self.config.pretraining_tp)
            ]
            down_proj = sum(down_proj)
        else:
            down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

        return down_proj

class LlamaRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        LlamaRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

class LlamaDecoderLayer(nn.Module):
    def __init__(self, config,index):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = LlamaAttention(config=config)
        self.mlp = LlamaMLP(config)
        self.index=index
        if self.index!=0:
            self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
        """

        residual = hidden_states

        if self.index != 0:
            hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs


class ResBlock(nn.Module):
    """
    A Residual Block module.

    This module performs a linear transformation followed by a SiLU activation,
    and then adds the result to the original input, creating a residual connection.

    Args:
        hidden_size (int): The size of the hidden layers in the block.
    """

    def __init__(self, hidden_size):
        super().__init__()
        self.linear = nn.Linear(hidden_size, hidden_size)
        # Initialize as an identity mapping
        nn.init.zeros_(self.linear.weight)
        # Use SiLU activation to keep consistent with the Llama model
        self.act = nn.SiLU()

    def forward(self, x):
        """
        Forward pass of the ResBlock.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output after the residual connection and activation.
        """
        return x + self.act(self.linear(x))

class ResAttentionBlock(nn.Module):
    """
    A Residual Block module.

    This module performs a linear transformation followed by a SiLU activation,
    and then adds the result to the original input, creating a residual connection.

    Args:
        hidden_size (int): The size of the hidden layers in the block.
    """

    def __init__(self, hidden_size, head_size, rms_norm_eps):
        super().__init__()
        self.head_size = head_size
        self.head_dim = hidden_size // head_size
        assert hidden_size % head_size == 0

        self.input_layernorm = LlamaRMSNorm(hidden_size, rms_norm_eps)
        self.q = nn.Linear(hidden_size, hidden_size)
        self.k = nn.Linear(hidden_size, hidden_size)
        self.v = nn.Linear(hidden_size, hidden_size)
        self.rms_norm_eps = rms_norm_eps
        # Use SiLU activation to keep consistent with the Llama model
        # self.act = nn.SiLU()

    def forward(self, x, y, x_replica=None):
        res = x
        x = self.input_layernorm(x)
        x_q = self.q(x)
        if x_replica is not None:
            x_q = x_q.view(-1, self.head_size * self.head_dim)[x_replica]
        y_k = self.k(y)
        att = torch.nn.functional.cosine_similarity(x_q.view(-1, self.head_dim), y_k.view(-1, self.head_dim), eps=self.rms_norm_eps)
        att = att.view(-1, self.head_size, 1)
        v =  self.v(y).view(-1, self.head_size, self.head_dim)
        v = v * att
        v = v.view(x_q.size())
        return res + v # self.act(v)

class I(nn.Module):
    def __init__(self):
        super().__init__()
        self.dummy = nn.Parameter(torch.ones(1, dtype=torch.float32))
    def forward(self,x):
        return x + self.dummy - self.dummy #(also tried x+self.dummy)

def len_list(x,n):
    return [i for i in x if len(i)<=n]


class ConfigClover:
    def __init__(self, config_clover):
        self.num_heads = config_clover["num_heads"]
        self.num_layers = config_clover["num_layers"]
        self.heads_coefficient = config_clover["heads_coefficient"]
        self.decay_coefficient = config_clover["decay_coefficient"]

class Clover2Model(nn.Module):
    '''
    output: [logits_1 (seq_len-1,向后两位), logits_2 (seq_len-2), logits_3 (seq_len-3)] , main_logits
    '''
    def __init__(self, config, head, config_clover=None, load_emb=False, 
                 path='/cpfs01/user/xiaobin/glj/models/vicuna-7b-v1.5', 
                 bias=True):
        super().__init__()
        if config_clover == None:
            self.config_clover = ConfigClover(config["clover"])
        else:
            self.config_clover = config_clover
        self.gradient_checkpointing = True
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.lm_head = head
        for param in self.lm_head.parameters():
            param.requires_grad = False

        def load_tensor(path, name, vocab_truncate=False):
            from safetensors import safe_open
            import json
            try:
                with open(os.path.join(path,"model.safetensors.index.json"),"r") as f:
                    index_json=json.loads(f.read())
                    #print(f"index_json:{index_json} {name}")
                    emb_path=index_json["weight_map"][name]
                    #print(f"emb_path:{emb_path} {name}")
                with safe_open(os.path.join(path,emb_path),
                                framework="pt",
                                device="cpu") as f:
                    tensor_slice = f.get_slice(name)
                    if vocab_truncate:
                        vocab_size, hidden_dim = tensor_slice.get_shape()
                        tensor = tensor_slice[:, :hidden_dim]
                    else:
                        tensor = tensor_slice[:]
            except:
                with open(os.path.join(path, "pytorch_model.bin.index.json"), "r") as f:
                    index_json = json.loads(f.read())
                    emb_path = index_json["weight_map"][name]
                weights=torch.load(os.path.join(path,emb_path))
                tensor=weights[name]
            return tensor#.cuda()
                
        
        self.clover_embed_tokens = nn.Embedding(
                config.vocab_size, config.hidden_size, config.pad_token_id#, dtype=config.torch_dtype
            )
        # self.init_with_data("clover_embed_tokens", load_tensor(path, f"model.embed_tokens.weight", vocab_truncate=True), self.clover_embed_tokens.weight)

        for param in self.clover_embed_tokens.parameters():
            param.requires_grad = False
            
        #self.init_tree()
        self.layers = nn.ModuleList([ LlamaDecoderLayer(config,1+i) for i in range(self.config_clover.num_layers)])

        # base_model = AutoModelForCausalLM.from_pretrained(path).model
        # baseconfig = AutoConfig.from_pretrained(path)
        base_layer = LlamaDecoderLayer(config,1)
        self.init_with_data("base_layer.self_attn.q_proj", load_tensor(path, f"model.layers.{config.num_hidden_layers-1}.self_attn.q_proj.weight"), base_layer.self_attn.q_proj.weight)
        self.init_with_data("base_layer.self_attn.k_proj", load_tensor(path, f"model.layers.{config.num_hidden_layers-1}.self_attn.k_proj.weight"), base_layer.self_attn.k_proj.weight)
        self.init_with_data("base_layer.self_attn.v_proj", load_tensor(path, f"model.layers.{config.num_hidden_layers-1}.self_attn.v_proj.weight"), base_layer.self_attn.v_proj.weight)
        self.init_with_data("base_layer.self_attn.o_proj", load_tensor(path, f"model.layers.{config.num_hidden_layers-1}.self_attn.o_proj.weight"), base_layer.self_attn.o_proj.weight)
        self.init_with_data("base_layer.mlp.gate_proj", load_tensor(path, f"model.layers.{config.num_hidden_layers-1}.mlp.gate_proj.weight"), base_layer.mlp.gate_proj.weight)
        self.init_with_data("base_layer.mlp.down_proj", load_tensor(path, f"model.layers.{config.num_hidden_layers-1}.mlp.down_proj.weight"), base_layer.mlp.down_proj.weight)
        self.init_with_data("base_layer.mlp.up_proj", load_tensor(path, f"model.layers.{config.num_hidden_layers-1}.mlp.up_proj.weight"), base_layer.mlp.up_proj.weight)
        self.init_with_data("base_layer.input_layernorm", load_tensor(path, f"model.layers.{config.num_hidden_layers-1}.input_layernorm.weight"), base_layer.input_layernorm.weight)
        self.init_with_data("base_layer.post_attention_layernorm", load_tensor(path, f"model.layers.{config.num_hidden_layers-1}.post_attention_layernorm.weight"), base_layer.post_attention_layernorm.weight)
        
        base_norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.init_with_data("base_norm", load_tensor(path, f"model.norm.weight"), base_norm.weight)
        
        self.clover_norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        
        self.clover_head_norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.clover_head_mlp = nn.Linear(config.hidden_size, config.hidden_size)
        self.clover_head_mlp2 = nn.Linear(config.hidden_size * 2, config.hidden_size)
        self.clover_head_mlp_rnn = ResAttentionBlock(
            config.hidden_size, config.num_attention_heads, config.rms_norm_eps
        ) #nn.Linear(config.hidden_size * 2, config.hidden_size)
        self.clover_head_mlp_rnn2 = ResAttentionBlock(
            config.hidden_size, config.num_attention_heads, config.rms_norm_eps
        )
        
        
        self._init_clover_head(base_layer, base_norm)#base_model.layers[-1], base_model.norm
        
        # del base_model
        del base_layer
        del base_norm
        torch.cuda.empty_cache()

    def init_tree(self):
        self.tree = mc_sim_7b_63
        self.tree_buffer=generate_tree_buffers(self.tree,self.clover_embed_tokens.weight.device)

    def reset(self):
        self.tree_mask=None

    def _prepare_decoder_attention_mask(self, attention_mask, input_shape, inputs_embeds, past_key_values_length):
        # create causal mask
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        combined_attention_mask = None
        if input_shape[-1] > 1:
            combined_attention_mask = _make_causal_mask(
                input_shape,
                #inputs_embeds.dtype,
                torch.float32, # [MODIFIED] force to cast to float32
                device=inputs_embeds.device,
                past_key_values_length=past_key_values_length,
            )

        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            expanded_attn_mask = _expand_mask(attention_mask, torch.float32, tgt_len=input_shape[-1]).to(
                inputs_embeds.device
            )
            combined_attention_mask = (
                expanded_attn_mask if combined_attention_mask is None else expanded_attn_mask + combined_attention_mask
            )

        # [MODIFIED] add tree mask
        if hasattr(self, "tree_mask") and self.tree_mask is not None:
            tree_mask = self.tree_mask
            tree_len = tree_mask.size(-1)
            combined_attention_mask[:, :, -tree_len:, -tree_len:][
                tree_mask == 0
                ] = torch.finfo(torch.float32).min

        return combined_attention_mask
    
    def _prepare_decoder_attention_mask_clover(self, attention_mask, input_shape, inputs_embeds, past_key_values_length):
        # create causal mask
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        combined_attention_mask = None
        if input_shape[-1] > 1:
            combined_attention_mask = _make_causal_mask(
                input_shape,
                #inputs_embeds.dtype,
                torch.float32, # [MODIFIED] force to cast to float32
                device=inputs_embeds.device,
                past_key_values_length=past_key_values_length,
            )

        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            expanded_attn_mask = _expand_mask(attention_mask, torch.float32, tgt_len=input_shape[-1]).to(
                inputs_embeds.device
            )
            combined_attention_mask = (
                expanded_attn_mask if combined_attention_mask is None else expanded_attn_mask + combined_attention_mask
            )

        return combined_attention_mask

    def _init_clover_head(self, base_layer, base_norm):
        """Initialize the weights of each clover_head using the base model's weights"""

        def init_clover_layer(layer_idx, base_layer, clover_layer):
            self.init_one_param(
                f"clover_layers_{layer_idx}_self_attn.q_proj",
                base_layer.self_attn.q_proj.weight,
                clover_layer.self_attn.q_proj.weight,
            )
            self.init_one_param(
                f"clover_layers_{layer_idx}_self_attn.k_proj",
                base_layer.self_attn.k_proj.weight,
                clover_layer.self_attn.k_proj.weight,
            )
            self.init_one_param(
                f"clover_layers_{layer_idx}_self_attn.v_proj",
                base_layer.self_attn.v_proj.weight,
                clover_layer.self_attn.v_proj.weight,
            )
            self.init_one_param(
                f"clover_layers_{layer_idx}_self_attn.o_proj",
                base_layer.self_attn.o_proj.weight,
                clover_layer.self_attn.o_proj.weight,
            )
            self.init_one_param(
                f"clover_layers_{layer_idx}_mlp.gate_proj",
                base_layer.mlp.gate_proj.weight,
                clover_layer.mlp.gate_proj.weight,
            )
            self.init_one_param(
                f"clover_layers_{layer_idx}_mlp.down_proj",
                base_layer.mlp.down_proj.weight,
                clover_layer.mlp.down_proj.weight,
            )
            self.init_one_param(
                f"clover_layers_{layer_idx}_mlp.up_proj",
                base_layer.mlp.up_proj.weight,
                clover_layer.mlp.up_proj.weight,
            )
            
            self.init_one_param(
                f"clover_layers_{layer_idx}_input_layernorm",
                base_layer.input_layernorm.weight,
                clover_layer.input_layernorm.weight,
            )
            self.init_one_param(
                f"clover_layers_{layer_idx}_post_attention_layernorm",
                base_layer.post_attention_layernorm.weight,
                clover_layer.post_attention_layernorm.weight,
            )
        for idx, layer in enumerate(self.layers):
            init_clover_layer(idx, base_layer, layer)

        self.init_one_param(
            f"clover_norm", base_norm.weight, self.clover_norm.weight
        )
        
        self.init_one_param(
            f"clover_norm", base_norm.weight, self.clover_head_norm.weight
        )


        def init_clover_rnn(clover_head_mlp_rnn):
            self.init_with_method(
                f"clover_head_mlp_rnn_q",
                clover_head_mlp_rnn.q.weight,
                self.eye_with_uniform,
                b=0.01,
                #scale=0.1,
            )
            self.init_with_method(
                f"clover_head_mlp_rnn_q_bias",
                clover_head_mlp_rnn.q.bias,
                nn.init.zeros_,
            )
            self.init_with_method(
                f"clover_head_mlp_rnn_k",
                clover_head_mlp_rnn.k.weight,
                self.eye_with_uniform,
                b=0.01,
                #scale=0.1,
            )
            self.init_with_method(
                f"clover_head_mlp_rnn_k_bias",
                clover_head_mlp_rnn.k.bias,
                nn.init.zeros_,
            )
            self.init_with_method(
                f"clover_head_mlp_rnn_v",
                clover_head_mlp_rnn.v.weight,
                nn.init.uniform_,
                b=0.01
            )
            self.init_with_method(
                f"clover_head_mlp_rnn_v_bias",
                clover_head_mlp_rnn.v.bias,
                nn.init.zeros_,
            )
            # self.init_with_method(
            #     f"clover_head_mlp_rnn_o",
            #     clover_head_mlp_rnn.o.weight,
            #     nn.init.uniform_,
            #     b=0.01
            # )
            # self.init_with_method(
            #     f"clover_head_mlp_rnn_o_bias",
            #     clover_head_mlp_rnn.o.bias,
            #     nn.init.zeros_,
            # )
        init_clover_rnn(self.clover_head_mlp_rnn)
        init_clover_rnn(self.clover_head_mlp_rnn2)
        

        self.init_one_param(
            f"clover_embed_tokens",
            self.lm_head.weight,
            self.clover_embed_tokens.weight,
            # fn=lambda w: nn.functional.normalize(w),
        )
        '''
        for i, head_mlp in enumerate(self.clover_head_mlp):
            for j, one_mlp in enumerate(head_mlp):
                self.init_with_method(
                    f"clover_head_mlp_{i}_{j}",
                    one_mlp.linear.weight,
                    nn.init.zeros_,
                    # one_mlp.weight,
                    # self.eye_with_uniform,
                    # b=0.01,
                )
                if one_mlp.linear.bias is not None:
                # if one_mlp.bias is not None:
                    self.init_with_method(
                        f"clover_head_mlp_{i}_{j}_bais",
                        one_mlp.linear.bias,
                        # one_mlp.bias,
                        nn.init.zeros_,
                    )
        '''
        def part_eye_with_uniform(tensor, b, scale = 1.0):
            nn.init.uniform_(tensor, b=b)
            tensor[:, :tensor.shape[0]] += torch.eye(tensor.shape[0], device=tensor.device) * scale

        if self.clover_head_mlp is not None:
            one_mlp = self.clover_head_mlp
            i = 0
            # for i, one_mlp in enumerate(self.clover_head_mlp):
            #for j, one_mlp in enumerate(head_mlp):
            self.init_with_method(
                f"clover_head_mlp_{i}",#_{j}",
                one_mlp.weight, # one_mlp.linear.weight,
                self.eye_with_uniform,
                b=0.01,
                #nn.init.zeros_,
            )
            if one_mlp.bias is not None: # one_mlp.linear.bias,
                self.init_with_method(
                    f"clover_head_mlp_{i}_bais",#_{j}
                    one_mlp.bias,# one_mlp.linear.bias,
                    nn.init.zeros_,
                )
            one_mlp = self.clover_head_mlp2
            i = 0
            self.init_with_method(
                f"clover_head_mlp_{i}",#_{j}",
                one_mlp.weight, # one_mlp.linear.weight,
                part_eye_with_uniform,#, self.eye_with_uniform,
                b=0.01,
                # scale=0.1,
                #nn.init.zeros_,
            )
            if one_mlp.bias is not None: # one_mlp.linear.bias,
                self.init_with_method(
                    f"clover_head_mlp_{i}_bais",#_{j}
                    one_mlp.bias,# one_mlp.linear.bias,
                    nn.init.zeros_,
                )

    @classmethod
    def init_one_param(cls, name, base, to, fn=None):
        from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled

        if is_deepspeed_zero3_enabled():
            from deepspeed.runtime.zero import GatheredParameters

            with GatheredParameters(base, modifier_rank=None):
                with GatheredParameters(to, modifier_rank=0):
                    if fn is None:
                        to.data[:] = base.data[:].to(to.data.dtype)
                    else:
                        # print(f"init weight {name} {base.size()} {new_base.size()}")
                        to.data[:] = fn(base).data[:].to(to.data.dtype)
                    print(f"Gathered init weight {name} {to.data}")
        else:
            if fn is None:
                to.data[:] = base.data[:].to(to.data.dtype)
            else:
                to.data[:] = fn(base).data[:].to(to.data.dtype)
            print(f"init weight {name} {to.data}")

    @classmethod
    def init_with_data(cls, name, data, to, fn=None):
        from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled

        if is_deepspeed_zero3_enabled():
            from deepspeed.runtime.zero import GatheredParameters
            with GatheredParameters(to, modifier_rank=0):
                to.data[:] = data.to(to.dtype)
                print(f"Gathered init weight {name} {data.dtype} {to.data}")
        else:
            to.data[:] = data.to(to.dtype)
            print(f"init weight {name} {to.data}")
            
    @classmethod
    def init_with_method(cls, name, to, method, **kwargs):
        from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled

        if is_deepspeed_zero3_enabled():
            from deepspeed.runtime.zero import GatheredParameters

            with GatheredParameters(to, modifier_rank=0):
                method(to.data, **kwargs)
                print(f"Gathered init weight {name} {to.data}")
        else:
            method(to.data, **kwargs)
            print(f"init weight {name} {to.data}")

    @classmethod
    def eye_with_uniform(cls, tensor, b):
        nn.init.uniform_(tensor, b=b)
        tensor += torch.eye(tensor.shape[0], device=tensor.device)

    def forward(
        self,
        hidden_states,
        input_ids,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        std=None
    ):
        all_ret = self.forward_clover_h(
            hidden_states,
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            std=std,
        )
        if use_cache:
            clover_hidden_states, token_emb, next_decoder_cache = all_ret
        else:
            clover_hidden_states, token_emb = all_ret
        
        def get_all_head(clover_hidden_states, token_emb):
            hidden_states_seq = []
            
            for i in range(0, self.config_clover.num_heads):
                clover_hidden_states = clover_hidden_states[
                    :, : token_emb.size(1) - i
                ].contiguous()
                lm_head_h, clover_hidden_states = self.forward_rnn(
                    clover_hidden_states, None, i, token_emb=token_emb[:, i:]
                )
                lm_head_h_pad = torch.zeros_like(token_emb)
                lm_head_h_pad[:, : token_emb.size(1) - i] = lm_head_h
                hidden_states_seq.append(lm_head_h_pad)
            
            hidden_states_seq = torch.stack(hidden_states_seq)
            return hidden_states_seq

        if self.gradient_checkpointing and self.training:
            hidden_states_seq = torch.utils.checkpoint.checkpoint(
                get_all_head,
                clover_hidden_states, token_emb
            )
        else:
            hidden_states_seq = get_all_head(clover_hidden_states, token_emb)
        
        return hidden_states_seq
    
    
    def forward_lm_head(self, lm_head_h):
        return self.lm_head(lm_head_h)
    
    def forward_clover_h(
        self,
        hidden_states,
        input_ids,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        std=None
    ):
        token_emb = self.clover_embed_tokens(input_ids)
        token_emb.requires_grad_()
        if self.gradient_checkpointing and self.training:
            hidden_states = torch.utils.checkpoint.checkpoint(
                self.clover_head_mlp_rnn,
                hidden_states, token_emb
            )
        else:
            hidden_states = self.clover_head_mlp_rnn(
                hidden_states, token_emb
        )
    
        batch_size, seq_length, _ = hidden_states.shape
        seq_length_with_past = seq_length
        past_key_values_length = 0
        if past_key_values is not None:
            past_key_values_length = past_key_values[0][0].shape[2]
            seq_length_with_past = seq_length_with_past + past_key_values_length
        if position_ids is None:
            device = hidden_states.device if hidden_states is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
            )
            position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
        else:
            position_ids = position_ids.view(-1, seq_length).long()

        if attention_mask is None:
            attention_mask = torch.ones(
                (batch_size, seq_length_with_past), dtype=torch.bool, device=hidden_states.device
            )
        attention_mask = self._prepare_decoder_attention_mask_clover(
            attention_mask, (batch_size, seq_length), hidden_states, past_key_values_length
        )

        all_hidden_states = () if output_hidden_states else None
        
        next_decoder_cache = () if use_cache else None


        for layer_idx, cur_layer in enumerate(self.layers):
            past_key_value = past_key_values[layer_idx] if past_key_values is not None else None
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            # main_logits = self.lm_head(hidden_states)
            if self.gradient_checkpointing and self.training:
                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        # None for past_key_value
                        return module(*inputs, attention_mask, 
                                    position_ids, past_key_value, output_attentions)
                    return custom_forward

                clover_hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(cur_layer),
                    hidden_states,
                )
                if output_hidden_states:
                    all_hidden_states += (clover_hidden_states,)
                if use_cache:
                    next_decoder_cache += (clover_hidden_states[2 if output_attentions else 1],)
                
                # print_tensor(f"forward_clover_h layer clover_hidden_states", clover_hidden_states[0])
            else:
                clover_hidden_states = cur_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                )
                if output_hidden_states:
                    all_hidden_states += (clover_hidden_states,)
                if use_cache:
                    next_decoder_cache += (clover_hidden_states[2 if output_attentions else 1],)
            hidden_states = clover_hidden_states[0]
        hidden_states = self.clover_norm(hidden_states)
        if use_cache:
            return hidden_states, token_emb, next_decoder_cache
        return hidden_states, token_emb


    def forward_layer(
        self,
        hidden_states,
        input_ids,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        std=None
    ):  
        all_ret = self.forward_clover_h(
            hidden_states,
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            std=std,
        )
        if use_cache:
            hidden_states, token_emb, next_decoder_cache = all_ret
        else:
            hidden_states, token_emb = all_ret

        if use_cache:
            return hidden_states, next_decoder_cache

        return hidden_states

    def forward_rnn(self, hidden_states, input_ids, head_idx, token_emb=None):
        if head_idx > 0:
            if token_emb is None:
                token_emb = self.clover_embed_tokens(input_ids)
            hidden_states = self.clover_head_mlp_rnn2(
                hidden_states, token_emb
            )
            return self.clover_head_norm(self.clover_head_mlp2(torch.concat([hidden_states, token_emb], dim=-1))), hidden_states
        else:
            
            return self.clover_head_norm(self.clover_head_mlp(hidden_states)), hidden_states
    
    @torch.no_grad()
    def repeat_kv(self,kv,numr):
        newkv=[]
        for i in kv:
            newkv.append((i[0].repeat(numr,1,1,1),i[1].repeat(numr,1,1,1)))
        return tuple(newkv)

    @torch.no_grad()
    def reduce_kv(self,kv,numr):
        newkv=[]
        for i in kv:
            newkv.append((i[0][:numr],i[1][:numr]))
        return tuple(newkv)


    def reset_kv(self):
        self.stable_kv=None

    @torch.no_grad()
    def repeat_hidden(self,hidden_state,repeat_num):
        new_hidden=[]
        for id,i in enumerate(repeat_num):
            if i > 0:
                new_hidden.append(hidden_state[:,id:id+1].repeat(1,i,1))
        if new_hidden:
            return torch.cat(new_hidden, dim=1)
        else:
            # Handle case when all elements in repeat_num are 0
            # Returning an empty tensor with the appropriate dimensions
            return torch.zeros((hidden_state.size(0), 0, hidden_state.size(2)), device=hidden_state.device)

    @torch.no_grad()
    def repeat_hidden_dim2(self,hidden_state,repeat_num):
        new_hidden=[]
        for id,i in enumerate(repeat_num):
            if i > 0:
                new_hidden.append(hidden_state[:,:,id:id+1].repeat(1,1,i,1))
        if new_hidden:
            return torch.cat(new_hidden, dim=2)
        else:
            # Handle case when all elements in repeat_num are 0
            # Returning an empty tensor with the appropriate dimensions
            return torch.zeros((hidden_state.size(0), 0, hidden_state.size(2)), device=hidden_state.device)
        
    
    def sample(self,logits, logits_processor,k=1, replacement=False):
        logits = logits_processor(None, logits)
        probabilities = torch.nn.functional.softmax(logits, dim=1)
        sampled_indices = torch.multinomial(probabilities, k, replacement=False)
        sampled_probs = torch.gather(probabilities, 1, sampled_indices)

        cumulative_sum = torch.cumsum(sampled_probs, dim=1)
        cumulative_sum = torch.cat(
            (torch.zeros(cumulative_sum.shape[0], 1, device=cumulative_sum.device), cumulative_sum[:, :-1]), dim=-1)

        sampled_probs = sampled_probs / (1 - cumulative_sum)
        sampled_probs[torch.isinf(sampled_probs)] = -1
        sampled_probs[torch.isnan(sampled_probs)] = -1

        sampled_probs = torch.clamp(sampled_probs, min=0.0, max=1.0)

        return sampled_indices, sampled_probs,probabilities

    
    def filter_arrays(self, prob, index, n):
        # Ensure prob and index are numpy arrays for easy manipulation
        prob = np.array(prob)
        index = np.array(index)

        # Get the indices that would sort the prob array in ascending order
        sorted_indices = np.argsort(prob)

        # Get the indices to keep (remove the smallest n elements)
        indices_to_keep = sorted_indices[n:]

        # Filter prob and index arrays
        filtered_prob = prob[indices_to_keep]
        filtered_index = index[indices_to_keep]

        return filtered_prob.tolist(), filtered_index.tolist()

    @torch.no_grad()
    def beam_genrate(self, hidden_states, input_ids, head, logits_processor,max_length=3, use_cache=True, 
                     token_count = 20, 
                     token_score_threldhold = 0.3, 
                     top_p_threldhold = 0.95):
        input_ids = input_ids[:, 1:]
        input_ids = input_ids.to(hidden_states.device)
        ss_token,ss_prob,ss_op = [],[],[]
        self.reset()
        if use_cache:

            if hasattr(self, "stable_kv") and self.stable_kv is not None:
                # kv_len=self.stable_kv[0][0].shape[2]
                out_hidden, past_key_values = self.forward_layer(hidden_states, past_key_values=self.stable_kv,use_cache=True)
            else:
                out_hidden, past_key_values = self.forward_layer(hidden_states, use_cache=True)
            self.stable_kv=past_key_values
            hidden_states = out_hidden[:, -1:]
            input_ids = input_ids[:, -1:]
            out_hidden, out_head = self.forward_rnn(hidden_states, input_ids=input_ids, head_idx=0)

            if not self.diff_device:
                last_headout = head(out_hidden[0])
            else:
                if hasattr(self, "layer_device"):
                    last_headout = head(out_hidden[0])
                    last_headout = last_headout.to(self.layer_device)
                else:
                    last_headout = F.linear(out_hidden[0], self.lm_head.weight)

            token_left = token_count
            repeat_nums = []
            tree_indices = []
            position_ids_list = []
            attn_mask = []
            sort_prob = []
            mc_sim_t = []
            flage_qiut = 0
            
            for i in range(self.config_clover.num_heads - 1):
                # sample update ss_token, ss_prob, ss_op
                if logits_processor is not None:
                    topk_index,topk_prob,op=self.sample(last_headout,logits_processor,k=token_count,)
                else:
                    top=torch.topk(last_headout, token_count, dim=-1)
                    topk_index,topk_prob = top.indices,top.values
                    topk_prob = nn.functional.softmax(topk_prob, dim=-1)
                    op=None

                ss_token.append(topk_index)
                ss_prob.append(topk_prob)
                ss_op.append(op)
                repeat_nums.append([])
                tree_indices.append([])
                position_ids_list.append([])
                sort_prob.append([])
                mc_sim_t.append([])
                for seq_len in range(len(topk_index)):
                    # prob_min = topk_prob[seq_len][0] * token_score_threldhold
                    sum_prob = 0
                    repeat_nums[i].append(0)
                    for k in range(token_count):
                        # if topk_prob[seq_len][k] < prob_min:
                        #     break
                        sum_prob += topk_prob[seq_len][k]
                        if sum_prob > top_p_threldhold and k > 0:
                            break
                        repeat_nums[i][seq_len] += 1
                        tree_indices[i].append(seq_len * token_count + k)
                        position_ids_list[i].append(0)
                        if i == 0:
                            sort_prob[i].append(topk_prob[seq_len][k].cpu().item())
                            mc_sim_t[i].append([k])
                        else:
                            # sort_prob_ = self.repeat_hidden_list(sort_prob[i-1],repeat_nums[i])
                            sort_prob[i].append(topk_prob[seq_len][k].cpu().item()*sort_prob[i-1][seq_len])
                            mc_sim_t[i].append(mc_sim_t[i-1][seq_len]+[k])
                
                if len(position_ids_list[i]) >= token_left:
                    # update tree_buffer sort+remake
                    sort_prob[i], mc_sim_t[i] = self.filter_arrays(sort_prob[i], mc_sim_t[i], len(position_ids_list[i]) - token_left)
                    mc_sim_list = [mc_sim for mc_sim_ in mc_sim_t for mc_sim in mc_sim_]
                    tree_buffer = generate_beam_tree_buffers(mc_sim_list, device=self.clover_head_mlp_rnn.q.weight.device, topk=token_count)
                    tree_buffer["retrieve_indices_head"] = tree_buffer["retrieve_indices"].to(
                                self.clover_head_mlp_rnn.q.weight.device)
                    flage_qiut = 1
                    break
                else:
                    token_left -= len(position_ids_list[i])
                # gen attn_mask
                if i == 0:
                    attn_mask.append(torch.eye(sum(repeat_nums[i]), sum(repeat_nums[i]), device=self.clover_embed_tokens.weight.device).unsqueeze(0).unsqueeze(0))
                else:
                    attn_mask.append(torch.cat([self.repeat_hidden_dim2(attn_mask[i-1],repeat_nums[i]),
                                               torch.eye(sum(repeat_nums[i]), sum(repeat_nums[i]), device=self.clover_embed_tokens.weight.device).unsqueeze(0).unsqueeze(0)], dim=-1))
                
                
                topk_index = topk_index.view(-1)
                select_index=topk_index[tree_indices[i]]
                # print(select_index)
                #len_sq=select_index.shape[0]
                input_ids=select_index[None,:]
                hidden_states=self.repeat_hidden(out_head, repeat_nums[i])

                out_hidden, out_head = self.forward_rnn(hidden_states, input_ids=input_ids, head_idx=i+1)
                # print('i: ', i, ' ', out_hidden.shape)
                if not self.diff_device:
                    last_headout = head(out_hidden[0])
                else:
                    if hasattr(self, "layer_device"):
                        last_headout = head(out_hidden[0])
                        last_headout = last_headout.to(self.layer_device)
                    else:
                        last_headout = F.linear(out_hidden[0], self.lm_head.weight)
            if flage_qiut == 0:
                if logits_processor is not None:
                    topk_index,topk_prob,op=self.sample(last_headout,logits_processor,k=token_count,)
                else:
                    top = torch.topk(last_headout, token_count, dim=-1)
                    topk_index, topk_prob = top.indices, top.values
                    topk_prob = nn.functional.softmax(topk_prob, dim=-1)
                    op=None
                ss_token.append(topk_index)
                ss_prob.append(topk_prob)
                ss_op.append(op)
                
                repeat_nums.append([])
                tree_indices.append([])
                position_ids_list.append([])
                sort_prob.append([])
                mc_sim_t.append([])
                i = self.config_clover.num_heads - 1
                for seq_len in range(len(topk_index)):
                    # prob_min = topk_prob[seq_len][0] * token_score_threldhold
                    sum_prob = 0
                    repeat_nums[i].append(0)
                    for k in range(token_count):
                        # if topk_prob[seq_len][k] < prob_min:
                        #     break
                        sum_prob += topk_prob[seq_len][k]
                        if sum_prob > top_p_threldhold:
                            break
                        repeat_nums[i][seq_len] += 1
                        tree_indices[i].append([seq_len * token_count + k])
                        position_ids_list[i].append(0)
                        if i == 0:
                            sort_prob[i].append(topk_prob[seq_len][k].cpu().item())
                            mc_sim_t[i].append([k])
                        else:
                            sort_prob[i].append(topk_prob[seq_len][k].cpu().item()*sort_prob[i-1][seq_len])
                            mc_sim_t[i].append(mc_sim_t[i-1][seq_len]+[k])
                
                if len(position_ids_list[i]) >= token_left:
                    # update tree_buffer sort+remake
                    sort_prob[i], mc_sim_t[i] = self.filter_arrays(sort_prob[i], mc_sim_t[i], len(position_ids_list[i]) - token_left)
                    mc_sim_list = [mc_sim for mc_sim_ in mc_sim_t for mc_sim in mc_sim_]
                    tree_buffer = generate_beam_tree_buffers(mc_sim_list, device=self.clover_head_mlp_rnn.q.weight.device, topk=token_count)
                    tree_buffer["retrieve_indices_head"] = tree_buffer["retrieve_indices"].to(
                                self.clover_head_mlp_rnn.q.weight.device)
                else:
                    mc_sim_list = [mc_sim for mc_sim_ in mc_sim_t for mc_sim in mc_sim_]
                    tree_buffer = generate_beam_tree_buffers(mc_sim_list, device=self.clover_head_mlp_rnn.q.weight.device, topk=token_count)
                    tree_buffer["retrieve_indices_head"] = tree_buffer["retrieve_indices"].to(
                                self.clover_head_mlp_rnn.q.weight.device)

            else:
                pass
        else:
            # TODO
            pass
        return (torch.cat(ss_token),torch.cat(ss_prob),ss_op), tree_buffer

    
    @torch.no_grad()
    def topK_genrate(self, hidden_states, input_ids, head, logits_processor,max_length=3, use_cache=True):
        input_ids = input_ids[:, 1:]
        input_ids = input_ids.to(hidden_states.device)
        ss_token,ss_prob,ss_op = [],[],[]
        self.reset()
        if use_cache:

            if hasattr(self, "stable_kv") and self.stable_kv is not None:
                out_hidden, past_key_values = self.forward_layer(hidden_states, input_ids[:, -hidden_states.shape[1]:], past_key_values=self.stable_kv,use_cache=True)
            else:
                out_hidden, past_key_values = self.forward_layer(hidden_states, input_ids, use_cache=True)
            self.stable_kv=past_key_values
            hidden_states = out_hidden[:, -1:]
            input_ids = input_ids[:, -1:]
            out_hidden, out_head = self.forward_rnn(hidden_states, input_ids=input_ids, head_idx=0)

            if not self.diff_device:
                last_headout = head(out_hidden[0])
            else:
                if hasattr(self, "layer_device"):
                    last_headout = head(out_hidden[0])
                    last_headout = last_headout.to(self.layer_device)
                else:
                    last_headout = F.linear(out_hidden[0], self.lm_head.weight)

            for i in range(len(self.tree_buffer['tree_indices'])):
                # sample update ss_token, ss_prob, ss_op
                if logits_processor is not None:
                    topk_index,topk_prob,op=self.sample(last_headout,logits_processor,k=top_k,)
                else:
                    top=torch.topk(last_headout, top_k, dim=-1)
                    topk_index,topk_prob = top.indices,top.values
                    topk_prob = nn.functional.softmax(topk_prob, dim=-1)
                    op=None

                ss_token.append(topk_index)
                ss_prob.append(topk_prob)
                ss_op.append(op)

                topk_index = topk_index.view(-1)
                select_index=topk_index[self.tree_buffer['tree_indices'][i]]
                input_ids=select_index[None,:]
                hidden_states=self.repeat_hidden(out_head, self.tree_buffer["repeat_nums"][i])

                out_hidden, out_head = self.forward_rnn(hidden_states, input_ids=input_ids, head_idx=i+1)
                if not self.diff_device:
                    last_headout = head(out_hidden[0])
                else:
                    if hasattr(self, "layer_device"):
                        last_headout = head(out_hidden[0])
                        last_headout = last_headout.to(self.layer_device)
                    else:
                        last_headout = F.linear(out_hidden[0], self.lm_head.weight)

            if logits_processor is not None:
                topk_index,topk_prob,op=self.sample(last_headout,logits_processor,k=top_k,)
            else:
                top = torch.topk(last_headout, top_k, dim=-1)
                topk_index, topk_prob = top.indices, top.values
                topk_prob = nn.functional.softmax(topk_prob, dim=-1)
                op=None
            ss_token.append(topk_index)
            ss_prob.append(topk_prob)
            ss_op.append(op)
            
        else:
            # TODO
            pass
        
        return (torch.cat(ss_token),torch.cat(ss_prob),ss_op)



class Vhead(nn.Module):
    def __init__(self,ins=6566,outs=32000):
        super().__init__()
        self.fc = nn.Linear(ins,outs,bias=False)
    def forward(self,x):
        return self.fc(x)



import torch

def count_parameters(model):
    return sum(p.numel() for p in model.parameters())
