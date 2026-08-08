# -*- coding: utf-8 -*-
# 调用官方mamba-ssm2.2.6中的mamba layer，提升mamba速度
from torch import Tensor, nn
import torch
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Union
import math
from mamba_ssm.mamba_ssm.modules.mamba_simple import Mamba
from mamba_ssm.mamba_ssm.ops.triton.layer_norm import RMSNorm, layer_norm_fn, rms_norm_fn
#from mambapy.mamba import MambaConfig, RMSNorm
from uer.layers.transformer import TransformerLayer
import pudb

"""

This file implements the Jamba architecture, as proposed by AI21labs (altough others have proposed blending Mamba & attention in the past).
A Jamba model combines Mamba and attention layers, as well as MoE for the MLP blocks.

This file closely follows the official Jamba implementation (https://huggingface.co/ai21labs/Jamba-v0.1).
But it is quite shorter (800 lines vs 2100 lines), because it has been stripped of all optional features that come with transformers.
It is thus easier to read, while keeping the same performances.
It supports training (using official CUDA mamba backend or mamba.py) & inference.
You can also load pretrained Jamba models from HF using the from_pretrained function.

Architecture of the torch modules found in this file :
- JambaLM: the final object, used for language modeling. has an embedding layer, an lm head...
- Jamba: core model. inputs (B, L, D), outputs (B, L, D). (B=batch size, L=seq len, D=d_model).
  It is composed of two types of layers : MambaLayer and AttentionLayer.
- AttentionLayer: standard GQA attention layer + MoE MLP (the attn computations are located in the AttentionSDPA module)
- MambaLayer : standard Mamba layer + MoE MLP. (the Mamba computations are located in the mamba.py file)
- SparseMoEBlock and MLP : Moe MLP

Notes :
-if using use_cuda, you must train in float32. If not, the following error is triggered : 
"Expected B.scalar_type() == (!is_variable_B ? weight_type : input_type) to be true, but got false."
when calling the selective_scan_fn function. Not clear why this error shows up when in (b)float16. TODO: investigate.

"""


@dataclass
class JambaLMConfig():
    def __init__(self, args):
        self.d_model: int = args.hidden_size

        self.mlp_size: int = args.feedforward_size

        self.initializer_range: float = 0.02
        self.rms_norm_eps: float = 1e-5

        # mamba related
        self.d_state: int = args.mamba_d_state  # N in paper
        self.expand_factor: int = args.mamba_expand_factor  # N in paper
        self.d_conv: int = args.mamba_d_conv
        self.dt_rank: Union[int, str] = 'auto'

        self.dt_min: float = 0.001
        self.dt_max: float = 0.1
        self.dt_init: str = "random"  #  "random" or "constant"
        self.dt_scale: float = 1.0
        self.dt_init_floor = 1e-4
        self.bias: bool = False
        self.conv_bias: bool = True
        self.inner_layernorms: bool = True
        self.use_cuda: bool = True
        self.pscan: bool = True  #  use parallel scan mode or sequential mode when training

        # MoE related
        self.num_experts: int = args.moe_num_experts
        self.num_experts_per_tok: int = args.moe_num_experts_per_token

        # language modeling
        self.vocab_size: int = len(args.vocab)
        self.pad_token_id: int = args.moe_pad_token_id
        self.tie_lm_weights: bool = True

        self.d_inner = self.expand_factor * self.d_model  # E*D = ED in comments

        if self.dt_rank == 'auto':
            self.dt_rank = math.ceil(self.d_model / 16)


# 先过transformer，再分别过mamba
class Transformer_Mamba_block(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.config = JambaLMConfig(args)

        if args.jamba_inner_parameter_sharing == "True":
            self.transformer_layer = TransformerLayer(args)
            transformer_layers = [self.transformer_layer]
            self.mamba_layer1 = MambaLayer(self.config)
            self.mamba_layer2 = MambaLayer(self.config)
            # 多层叠加: [self.mamba_layer1,self.mamba_layer1,...]
            mamba_layers1 = [self.mamba_layer1, self.mamba_layer1]
            mamba_layers2 = [self.mamba_layer2, self.mamba_layer2]
        else:
            transformer_layers = [TransformerLayer(args)]
            # 多层叠加: [MambaLayer(args),MambaLayer(args),...]
            mamba_layers1 = [MambaLayer(self.config), MambaLayer(self.config)]
            mamba_layers2 = [MambaLayer(self.config), MambaLayer(self.config)]

        self.transformer_layers = nn.ModuleList(transformer_layers)
        self.mamba_layer1 = nn.ModuleList(mamba_layers1)
        self.mamba_layer2 = nn.ModuleList(mamba_layers2)

    def forward(self, x, transformer_mask,  byte_mask, transmission_mask, byte_transmission_idx,
                byte_max_len=192, transmission_max_len=32, position_bias=None):
        # x: (B, L, D)
        # seg: segment
        # byte_max_len: Byte max len 实际上是165
        # transmission_max_len: Transmission max len 实际上是21

        for layer_idx, layer in enumerate(self.transformer_layers):
            x, _ = layer(x, transformer_mask, position_bias=position_bias)

        bs = len(x)  # batch size
        d = x[0].size(-1)  # dimension
        x_byte = torch.zeros(bs, byte_max_len, d, device=x.device)  # .cuda(gpu_id)
        x_transmission = torch.zeros(bs, transmission_max_len, d, device=x.device)  # .cuda(gpu_id)

        #import pdb
        #pdb.set_trace()

        for i, sequence in enumerate(x):
            byte_end = byte_transmission_idx[i][0]
            transmission_end = byte_transmission_idx[i][1]
            x_byte[i, :byte_end, :] = sequence[:byte_end, :]
            x_transmission[i, :transmission_end-byte_end, :] = \
                sequence[byte_end:transmission_end, :]

        for layer_idx, layer in enumerate(self.mamba_layer1):
            x_byte = layer(x_byte, byte_mask)

        for layer_idx, layer in enumerate(self.mamba_layer2):
            x_transmission = layer(x_transmission, transmission_mask)

        return x_byte, x_transmission


# 先分别过mamba，再过transformer
class Mamba_Transformer_block(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.config = JambaLMConfig(args)
        self.max_seq_length = args.max_seq_length

        # init each model layer, attention-to-Mamba=1:3
        if args.jamba_inner_parameter_sharing == "True":
            self.transformer_layer = TransformerLayer(args)
            transformer_layers = [self.transformer_layer]
            self.mamba_layer1 = MambaLayer(self.config)
            self.mamba_layer2 = MambaLayer(self.config)
            mamba_layers1 = [self.mamba_layer1]  # 多层叠加: [self.mamba_layer1,self.mamba_layer1,...]
            mamba_layers2 = [self.mamba_layer2]
        else:
            transformer_layers = [TransformerLayer(args)]
            mamba_layers1 = [MambaLayer(self.config)]  # 多层叠加: [MambaLayer(args),MambaLayer(args),...]
            mamba_layers2 = [MambaLayer(self.config)]

        self.transformer_layers = nn.ModuleList(transformer_layers)
        self.mamba_layer1 = nn.ModuleList(mamba_layers1)
        self.mamba_layer2 = nn.ModuleList(mamba_layers2)

    def forward(self, x, transformer_mask,  byte_mask, transmission_mask, byte_transmission_idx,
                byte_max_len=192, transmission_max_len=32, position_bias=None):
        # x: (B, L, D)
        # seg: segment
        # byte_max_len: Byte max len 实际上是165
        # transmission_max_len: Transmission max len 实际上是21

        bs = len(x)  # batch size
        d = x[0].size(-1)  # dimension
        x_byte = torch.zeros(bs, byte_max_len, d, device=x.device)  # .cuda(gpu_id)
        x_transmission = torch.zeros(bs, transmission_max_len, d, device=x.device)  # .cuda(gpu_id)

        for i, sequence in enumerate(x):
            byte_end = byte_transmission_idx[i][0]
            transmission_end = byte_transmission_idx[i][1]
            x_byte[i, :byte_end, :] = sequence[:byte_end, :]
            x_transmission[i, :transmission_end-byte_end, :] = \
                sequence[byte_end:transmission_end, :]

        for layer_idx, layer in enumerate(self.mamba_layer1):
            x_byte = layer(x_byte, byte_mask)

        for layer_idx, layer in enumerate(self.mamba_layer2):
            x_transmission = layer(x_transmission, transmission_mask)

        output = torch.zeros(bs, self.max_seq_length, d, device=x_byte.device)
        for i in range(bs):
            byte_end = byte_transmission_idx[i][0]
            transmission_end = byte_transmission_idx[i][1]
            output[i, :byte_end, :] = x_byte[i][:byte_end, :]
            output[i, byte_end:transmission_end, :] = \
                x_transmission[i][:transmission_end-byte_end, :]

        for layer_idx, layer in enumerate(self.transformer_layers):
            output, _ = layer(output, transformer_mask, position_bias=position_bias)

        return output


class MambaLayer(nn.Module):
    def __init__(self, config: JambaLMConfig):
        super().__init__()

        self.config = config
        self.mamba = Mamba(
            d_model=config.d_model,
            d_state=config.d_state,
            d_conv=config.d_conv,
            expand=config.expand_factor,
        )

        self.mlp = MLP(config)
        self.input_layernorm = RMSNorm(config.d_model, eps=config.rms_norm_eps)
        self.pre_mlp_layernorm = RMSNorm(config.d_model, eps=config.rms_norm_eps)

    def forward(self, x, padding_mask=None):
        # x: (B, L, D)

        # outputs: (B, L, D)

        # mamba
        residual = x
        x = self.input_layernorm(x)

        x = self.mamba(x)
        # -------- post-mask --------
        if padding_mask is not None:
            x = x * padding_mask.unsqueeze(-1)
        x = residual + x

        # FFN
        residual = x
        x = self.pre_mlp_layernorm(x)
        x = self.mlp(x)
        outputs = residual + x

        return outputs

    def get_empty_cache(self, batch_size, device):
        """
        初始化 cache，用于 autoregressive inference
        """
        # 官方 Mamba2 step cache 只需要 x 的 conv history
        conv_cache = torch.zeros(
            batch_size,
            self.config.mamba_config.mamba_d_state,
            self.config.mamba_config.mamba_d_conv - 1,
            device=device
        )
        return (None, conv_cache)


class MLP(nn.Module):
    def __init__(self, config: JambaLMConfig):
        super().__init__()

        self.hidden_dim = config.d_model
        self.ffn_dim = config.mlp_size

        self.gate_proj = nn.Linear(self.hidden_dim, self.ffn_dim, bias=False)
        self.down_proj = nn.Linear(self.ffn_dim, self.hidden_dim, bias=False)
        self.up_proj = nn.Linear(self.hidden_dim, self.ffn_dim, bias=False)

    def forward(self, x):
        # x : (B, L, D)

        # y : (B, L, D)

        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


def load_balancing_loss(router_logits, num_experts, num_experts_per_tok):
    # router_logits: list of router_logit, one per layer, each (B*D, n_experts)

    # moe_aux_loss : scalar

    router_logits = torch.cat([r for r in router_logits if r.shape[1] > 1], dim=0)

    routing_weights = torch.nn.functional.softmax(router_logits, dim=-1)
    _, selected_experts = torch.topk(routing_weights, num_experts_per_tok, dim=-1)
    expert_mask = torch.nn.functional.one_hot(selected_experts, num_experts)

    # percentage of tokens routed to each experts
    tokens_per_expert = torch.mean(expert_mask.float(), dim=0)

    # average probability of routing to these experts
    router_prob_per_expert = torch.mean(routing_weights, dim=0)

    moe_aux_loss = torch.sum(tokens_per_expert * router_prob_per_expert.unsqueeze(0))
    return moe_aux_loss * num_experts


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

