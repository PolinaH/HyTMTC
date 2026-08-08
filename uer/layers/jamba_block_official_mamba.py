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
from uer.layers.multi_headed_attn import MultiHeadedAttention
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

        '''
        self.mamba_config = MambaConfig(d_model=self.d_model, n_layers=0, dt_rank=self.dt_rank, d_state=self.d_state,
                                        expand_factor=self.expand_factor, d_conv=self.d_conv, dt_min=self.dt_min,
                                        dt_max=self.dt_max,
                                        dt_init=self.dt_init, dt_scale=self.dt_scale, rms_norm_eps=self.rms_norm_eps,
                                        bias=self.bias, conv_bias=self.conv_bias,
                                        inner_layernorms=self.inner_layernorms,
                                        pscan=self.pscan, use_cuda=self.use_cuda)
        '''

class JambaBlock17(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.config = JambaLMConfig(args)

        # init each model layer, attention-to-Mamba=1:7
        if args.jamba_inner_parameter_sharing == "True":
            self.mamba_layer = MambaLayer(self.config, num_experts=1)
            self.mambamoe_layer = MambaLayer(self.config, num_experts=self.config.num_experts)
            self.transformer_layer = TransformerLayer(args)
            layers = [self.mamba_layer,
                      self.mambamoe_layer,
                      self.mamba_layer,
                      self.mambamoe_layer,
                      self.transformer_layer,
                      self.mambamoe_layer,
                      self.mamba_layer,
                      self.mambamoe_layer]
        else:
            layers = [MambaLayer(self.config, num_experts=1),
                      MambaLayer(self.config, num_experts=self.config.num_experts),
                      MambaLayer(self.config, num_experts=1),
                      MambaLayer(self.config, num_experts=self.config.num_experts),
                      TransformerLayer(args),
                      MambaLayer(self.config, num_experts=self.config.num_experts),
                      MambaLayer(self.config, num_experts=1),
                      MambaLayer(self.config, num_experts=self.config.num_experts)]

        self.layers = nn.ModuleList(layers)

        # here you may want to init the weights in a particular manner if you don't use this jamba inside a JambaLM (see JambaLM)

    def forward(self, x, mask, position_bias):
        # x: (B, L, D)

        # logits: (B, L, D)
        # router_logits : (B*L, n_experts)
        router_logits = []
        for layer_idx, layer in enumerate(self.layers):
            if layer_idx == 4:  # transformer layer
                x, _ = layer(x, mask, position_bias=position_bias)
                batch_size, sequence_length, hidden_dim = x.shape
                tmp_router_logits = torch.ones(
                    (batch_size * sequence_length, 1),
                    device=x.device,
                    dtype=x.dtype,
                    requires_grad=x.requires_grad,
                )
                router_logits.append(tmp_router_logits)
            else:  # mamba/mamba moe layer
                layer_output, _ = layer(x)
                x = layer_output[0]
                router_logits.append(layer_output[1])

        return x, router_logits

    def step(self, x, caches):
        # x: (B, L, D)

        # logits: (B, L, D)
        # caches

        for i, decoder_layer in enumerate(self.layers):
            layer_output, caches[i] = decoder_layer(x, caches[i])
            x = layer_output[0]

        return x, caches


class JambaBlock15(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.config = JambaLMConfig(args)

        # init each model layer, attention-to-Mamba=1:5
        if args.jamba_inner_parameter_sharing == "True":
            self.mamba_layer = MambaLayer(self.config, num_experts=1)
            self.mambamoe_layer = MambaLayer(self.config, num_experts=self.config.num_experts)
            self.transformer_layer = TransformerLayer(args)
            layers = [self.mamba_layer,
                      self.mambamoe_layer,
                      self.transformer_layer,
                      self.mambamoe_layer,
                      self.mamba_layer,
                      self.mambamoe_layer]
        else:
            layers = [MambaLayer(self.config, num_experts=1),
                      MambaLayer(self.config, num_experts=self.config.num_experts),
                      TransformerLayer(args),
                      MambaLayer(self.config, num_experts=self.config.num_experts),
                      MambaLayer(self.config, num_experts=1),
                      MambaLayer(self.config, num_experts=self.config.num_experts)]

        self.layers = nn.ModuleList(layers)

        # here you may want to init the weights in a particular manner if you don't use this jamba inside a JambaLM (see JambaLM)

    def forward(self, x, mask, position_bias):
        # x: (B, L, D)

        # logits: (B, L, D)
        # router_logits : (B*L, n_experts)
        router_logits = []
        for layer_idx, layer in enumerate(self.layers):
            if layer_idx == 2:  # transformer layer
                x, _ = layer(x, mask, position_bias=position_bias)
                batch_size, sequence_length, hidden_dim = x.shape
                tmp_router_logits = torch.ones(
                    (batch_size * sequence_length, 1),
                    device=x.device,
                    dtype=x.dtype,
                    requires_grad=x.requires_grad,
                )
                router_logits.append(tmp_router_logits)
            else:  # mamba/mamba moe layer
                layer_output, _ = layer(x)
                x = layer_output[0]
                router_logits.append(layer_output[1])

        return x, router_logits

    def step(self, x, caches):
        # x: (B, L, D)

        # logits: (B, L, D)
        # caches

        for i, decoder_layer in enumerate(self.layers):
            layer_output, caches[i] = decoder_layer(x, caches[i])
            x = layer_output[0]

        return x, caches


class JambaBlock13(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.config = JambaLMConfig(args)

        # init each model layer, attention-to-Mamba=1:3
        if args.jamba_inner_parameter_sharing == "True":
            #mamba_layer = MambaLayer(self.config, num_experts=1)
            mamba_layer = MambaLayer_bidirection(self.config, num_experts=1)
            #mambamoe_layer = MambaLayer(self.config, num_experts=self.config.num_experts)
            mambamoe_layer = MambaLayer_bidirection(self.config, num_experts=self.config.num_experts)
            transformer_layer = TransformerLayer(self.config, args, num_experts=1)
            layers = [mamba_layer,
                      mambamoe_layer,
                      transformer_layer,
                      mambamoe_layer]
        else:
            layers = [MambaLayer(self.config, num_experts=1),
                      MambaLayer(self.config, num_experts=self.config.num_experts),
                      TransformerLayer(self.config, args, num_experts=1),
                      MambaLayer(self.config, num_experts=self.config.num_experts)]

        self.layers = nn.ModuleList(layers)

        # here you may want to init the weights in a particular manner if you don't use this jamba inside a JambaLM (see JambaLM)

    def forward(self, x, mask, mamba_mask, position_bias):
        # x: (B, L, D)

        # logits: (B, L, D)
        # router_logits : (B*L, n_experts)

        router_logits = []
        for layer_idx, layer in enumerate(self.layers):
            if layer_idx == 2:  # transformer layer
                x, _, _ = layer(x, mask, position_bias=position_bias)
                batch_size, sequence_length, hidden_dim = x.shape
                tmp_router_logits = torch.ones(
                    (batch_size * sequence_length, 1),
                    device=x.device,
                    dtype=x.dtype,
                    requires_grad=x.requires_grad,
                )
                router_logits.append(tmp_router_logits)
            else:  # mamba/mamba moe layer
                layer_output, _, _ = layer(x, mamba_mask)
                x = layer_output[0]
                router_logits.append(layer_output[1])

        return x, router_logits, None

    def step(self, x, caches):
        # x: (B, L, D)

        # logits: (B, L, D)
        # caches

        for i, decoder_layer in enumerate(self.layers):
            layer_output, caches[i] = decoder_layer(x, caches[i])
            x = layer_output[0]

        return x, caches


class JambaBlock_LSTM_Transformer(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.config = JambaLMConfig(args)

        # init each model layer, attention-to-Mamba=1:3
        if args.jamba_inner_parameter_sharing == "True":
            self.lstm_layer = LSTMLayer(self.config, num_experts=1)
            self.lstmmoe_layer = LSTMLayer(self.config, num_experts=self.config.num_experts)
            self.transformer_layer = TransformerLayer(self.config, args, num_experts=1)
            layers = [self.lstm_layer,
                      self.lstmmoe_layer,
                      self.transformer_layer,
                      self.lstmmoe_layer]
        else:
            layers = [LSTMLayer(self.config, num_experts=1),
                      LSTMLayer(self.config, num_experts=self.config.num_experts),
                      TransformerLayer(self.config, args, num_experts=1),
                      LSTMLayer(self.config, num_experts=self.config.num_experts)]

        self.layers = nn.ModuleList(layers)

        # here you may want to init the weights in a particular manner if you don't use this jamba inside a JambaLM (see JambaLM)

    def forward(self, x, mask, mamba_mask, position_bias):
        # x: (B, L, D)

        # logits: (B, L, D)
        # router_logits : (B*L, n_experts)

        router_logits = []
        for layer_idx, layer in enumerate(self.layers):
            if layer_idx == 2:  # transformer layer
                x, _, _ = layer(x, mask, position_bias=position_bias)
                batch_size, sequence_length, hidden_dim = x.shape
                tmp_router_logits = torch.ones(
                    (batch_size * sequence_length, 1),
                    device=x.device,
                    dtype=x.dtype,
                    requires_grad=x.requires_grad,
                )
                router_logits.append(tmp_router_logits)
            else:  # mamba/mamba moe layer
                layer_output, _, _ = layer(x, mamba_mask)
                x = layer_output[0]
                router_logits.append(layer_output[1])

        return x, router_logits, None

    def step(self, x, caches):
        # x: (B, L, D)

        # logits: (B, L, D)
        # caches

        for i, decoder_layer in enumerate(self.layers):
            layer_output, caches[i] = decoder_layer(x, caches[i])
            x = layer_output[0]

        return x, caches


class JambaEncoder_onlyMamba_block(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.config = JambaLMConfig(args)

        # init each model layer, attention-to-Mamba=1:3
        if args.jamba_inner_parameter_sharing == "True":
            self.mamba_layer = MambaLayer(self.config, num_experts=1)
            layers = [self.mamba_layer,
                      self.mamba_layer,
                      self.mamba_layer,
                      self.mamba_layer
                      ]
        else:
            layers = [MambaLayer(self.config, num_experts=1),
                      MambaLayer(self.config, num_experts=1),
                      MambaLayer(self.config, num_experts=1),
                      MambaLayer(self.config, num_experts=1)
                      ]

        self.layers = nn.ModuleList(layers)

        # here you may want to init the weights in a particular manner if you don't use this jamba inside a JambaLM (see JambaLM)
    def forward(self, x, mamba_mask, position_bias):
        # x: (B, L, D)
        # logits: (B, L, D)
        # router_logits : (B*L, n_experts)

        router_logits = []
        for layer_idx, layer in enumerate(self.layers):
            layer_output, _, _ = layer(x, mamba_mask)
            x = layer_output[0]
            router_logits.append(layer_output[1])

        return x, router_logits, None


class JambaEncoder_onlyMambaMOE_block(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.config = JambaLMConfig(args)

        # init each model layer, attention-to-Mamba=1:3
        if args.jamba_inner_parameter_sharing == "True":
            self.mambamoe_layer = MambaLayer(self.config, num_experts=self.config.num_experts)
            layers = [self.mambamoe_layer,
                      self.mambamoe_layer,
                      self.mambamoe_layer,
                      self.mambamoe_layer
                      ]
        else:
            layers = [MambaLayer(self.config, num_experts=self.config.num_experts),
                      MambaLayer(self.config, num_experts=self.config.num_experts),
                      MambaLayer(self.config, num_experts=self.config.num_experts),
                      MambaLayer(self.config, num_experts=self.config.num_experts)
                      ]

        self.layers = nn.ModuleList(layers)

        # here you may want to init the weights in a particular manner if you don't use this jamba inside a JambaLM (see JambaLM)

    def forward(self, x, mamba_mask, position_bias):
        # x: (B, L, D)

        # logits: (B, L, D)
        # router_logits : (B*L, n_experts)

        router_logits = []
        for layer_idx, layer in enumerate(self.layers):
            layer_output, _, _ = layer(x, mamba_mask)
            x = layer_output[0]
            router_logits.append(layer_output[1])

        return x, router_logits, None


class JambaEncoder_onlyLSTM_block(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.config = JambaLMConfig(args)

        # init each model layer, attention-to-Mamba=1:3
        if args.jamba_inner_parameter_sharing == "True":
            self.lstm_layer = LSTMLayer(self.config, num_experts=1)
            layers = [self.lstm_layer,
                      self.lstm_layer,
                      self.lstm_layer,
                      self.lstm_layer
                      ]
        else:
            layers = [LSTMLayer(self.config, num_experts=1),
                      LSTMLayer(self.config, num_experts=1),
                      LSTMLayer(self.config, num_experts=1),
                      LSTMLayer(self.config, num_experts=1)
                      ]

        self.layers = nn.ModuleList(layers)

        # here you may want to init the weights in a particular manner if you don't use this jamba inside a JambaLM (see JambaLM)
    def forward(self, x, mamba_mask, position_bias):
        # x: (B, L, D)
        # logits: (B, L, D)
        # router_logits : (B*L, n_experts)

        router_logits = []
        for layer_idx, layer in enumerate(self.layers):
            layer_output, _, _ = layer(x, mamba_mask)
            x = layer_output[0]
            router_logits.append(layer_output[1])

        return x, router_logits, None


class JambaEncoder_onlyLSTMMOE_block(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.config = JambaLMConfig(args)

        # init each model layer, attention-to-Mamba=1:3
        if args.jamba_inner_parameter_sharing == "True":
            self.lstmmoe_layer = LSTMLayer(self.config, num_experts=self.config.num_experts)
            layers = [self.lstmmoe_layer,
                      self.lstmmoe_layer,
                      self.lstmmoe_layer,
                      self.lstmmoe_layer
                      ]
        else:
            layers = [LSTMLayer(self.config, num_experts=self.config.num_experts),
                      LSTMLayer(self.config, num_experts=self.config.num_experts),
                      LSTMLayer(self.config, num_experts=self.config.num_experts),
                      LSTMLayer(self.config, num_experts=self.config.num_experts)
                      ]

        self.layers = nn.ModuleList(layers)

        # here you may want to init the weights in a particular manner if you don't use this jamba inside a JambaLM (see JambaLM)

    def forward(self, x, mamba_mask, position_bias):
        # x: (B, L, D)

        # logits: (B, L, D)
        # router_logits : (B*L, n_experts)

        router_logits = []
        for layer_idx, layer in enumerate(self.layers):
            layer_output, _, _ = layer(x, mamba_mask)
            x = layer_output[0]
            router_logits.append(layer_output[1])

        return x, router_logits, None


class JambaEncoder_onlyTransformer_block(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.config = JambaLMConfig(args)

        # init each model layer, attention-to-Mamba=1:3
        if args.jamba_inner_parameter_sharing == "True":
            self.transformer_layer = TransformerLayer(self.config, args, num_experts=1)
            layers = [self.transformer_layer,
                      self.transformer_layer,
                      self.transformer_layer,
                      self.transformer_layer
                      ]
        else:
            layers = [TransformerLayer(self.config, args, num_experts=1),
                      TransformerLayer(self.config, args, num_experts=1),
                      TransformerLayer(self.config, args, num_experts=1),
                      TransformerLayer(self.config, args, num_experts=1)
                      ]

        self.layers = nn.ModuleList(layers)

        # here you may want to init the weights in a particular manner if you don't use this jamba inside a JambaLM (see JambaLM)

    def forward(self, x, mask, position_bias):
        # x: (B, L, D)

        # logits: (B, L, D)
        # router_logits : (B*L, n_experts)

        router_logits = []
        for layer_idx, layer in enumerate(self.layers):
            x, _, attn = layer(x, mask, position_bias=position_bias)
            batch_size, sequence_length, hidden_dim = x.shape
            tmp_router_logits = torch.ones(
                (batch_size * sequence_length, 1),
                device=x.device,
                dtype=x.dtype,
                requires_grad=x.requires_grad,
            )
            router_logits.append(tmp_router_logits)

        return x, router_logits, attn


class JambaEncoder_onlyTransformerMoE_block(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.config = JambaLMConfig(args)

        # init each model layer, attention-to-Mamba=1:3
        if args.jamba_inner_parameter_sharing == "True":
            self.transformer_layer = TransformerLayer(self.config, args, num_experts=self.config.num_experts)
            layers = [self.transformer_layer,
                      self.transformer_layer,
                      self.transformer_layer,
                      self.transformer_layer
                      ]
        else:
            layers = [TransformerLayer(self.config, args, num_experts=self.config.num_experts),
                      TransformerLayer(self.config, args, num_experts=self.config.num_experts),
                      TransformerLayer(self.config, args, num_experts=self.config.num_experts),
                      TransformerLayer(self.config, args, num_experts=self.config.num_experts)
                      ]

        self.layers = nn.ModuleList(layers)

        # here you may want to init the weights in a particular manner if you don't use this jamba inside a JambaLM (see JambaLM)

    def forward(self, x, mask, position_bias):
        # x: (B, L, D)

        # logits: (B, L, D)
        # router_logits : (B*L, n_experts)

        router_logits = []
        for layer_idx, layer in enumerate(self.layers):
            x, _, attn = layer(x, mask, position_bias=position_bias)
            batch_size, sequence_length, hidden_dim = x.shape
            tmp_router_logits = torch.ones(
                (batch_size * sequence_length, 1),
                device=x.device,
                dtype=x.dtype,
                requires_grad=x.requires_grad,
            )
            router_logits.append(tmp_router_logits)

        return x, router_logits, attn


class MambaLayer(nn.Module):
    def __init__(self, config: JambaLMConfig, num_experts: int):
        super().__init__()

        self.config = config

        #self.mamba = MambaBlock(config=config.mamba_config)

        self.mamba = Mamba(
            d_model=config.d_model,
            d_state=config.d_state,
            d_conv=config.d_conv,
            expand=config.expand_factor,
        )

        num_experts_per_tok = config.num_experts_per_tok if num_experts > 1 else 1
        self.moe = SparseMoEBlock(config, num_experts=num_experts, num_experts_per_tok=num_experts_per_tok)
        self.input_layernorm = RMSNorm(config.d_model, eps=config.rms_norm_eps)
        self.pre_moe_layernorm = RMSNorm(config.d_model, eps=config.rms_norm_eps)

    def forward(self, x, mamba_mask, cache=None):
        # x: (B, L, D)

        # outputs: (B, L, D)

        # mamba
        residual = x
        x = self.input_layernorm(x)
        x = self.mamba(x)
        # -------- post-mask --------
        if mamba_mask is not None:
            x = x * mamba_mask.unsqueeze(-1)
        x = residual + x

        # FFN
        residual = x
        x = self.pre_moe_layernorm(x)
        x, router_logits = self.moe(x)
        x = residual + x

        outputs = (x, router_logits)

        return outputs, cache, None

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


class MambaLayer_bidirection(nn.Module):
    def __init__(self, config: JambaLMConfig, num_experts: int):
        super().__init__()

        self.config = config

        # Forward Mamba: processes sequence in original order
        self.mamba_forward = Mamba(
            d_model=config.d_model,
            d_state=config.d_state,
            d_conv=config.d_conv,
            expand=config.expand_factor,
        )

        # Backward Mamba: processes sequence in reversed order
        self.mamba_backward = Mamba(
            d_model=config.d_model,
            d_state=config.d_state,
            d_conv=config.d_conv,
            expand=config.expand_factor,
        )

        # Linear projection to merge forward and backward outputs: 2*D -> D
        self.merge_proj = nn.Linear(config.d_model * 2, config.d_model, bias=False)

        num_experts_per_tok = config.num_experts_per_tok if num_experts > 1 else 1
        self.moe = SparseMoEBlock(config, num_experts=num_experts, num_experts_per_tok=num_experts_per_tok)
        self.input_layernorm = RMSNorm(config.d_model, eps=config.rms_norm_eps)
        self.pre_moe_layernorm = RMSNorm(config.d_model, eps=config.rms_norm_eps)

    def forward(self, x, mamba_mask, cache=None):
        # x: (B, L, D)

        # outputs: (B, L, D)

        # mamba
        residual = x
        x = self.input_layernorm(x)

        # Forward direction: process sequence as-is
        x_forward = self.mamba_forward(x)

        # Backward direction: flip sequence along time dim, process, flip back
        x_backward = torch.flip(x, [1])
        x_backward = self.mamba_backward(x_backward)
        x_backward = torch.flip(x_backward, [1])

        # Merge forward and backward outputs via concatenation + linear projection
        x = self.merge_proj(torch.cat([x_forward, x_backward], dim=-1))

        # -------- post-mask --------
        if mamba_mask is not None:
            x = x * mamba_mask.unsqueeze(-1)
        x = residual + x

        # FFN
        residual = x
        x = self.pre_moe_layernorm(x)
        x, router_logits = self.moe(x)
        x = residual + x

        outputs = (x, router_logits)

        return outputs, cache, None

    def get_empty_cache(self, batch_size, device):
        """
        初始化 cache，用于 autoregressive inference
        （双向 Mamba 不适用于自回归推理，保留接口兼容）
        """
        conv_cache = torch.zeros(
            batch_size,
            self.config.d_model,
            self.config.d_conv - 1,
            device=device
        )
        return (None, conv_cache)


class LSTMLayer(nn.Module):
    def __init__(self, config: JambaLMConfig, num_experts: int):
        super().__init__()

        self.config = config

        #self.mamba = MambaBlock(config=config.mamba_config)

        self.lstm = nn.LSTM(
            input_size=config.d_model,
            hidden_size=config.d_model,
            num_layers=1,
            batch_first=True,
            bidirectional=False
        )

        num_experts_per_tok = config.num_experts_per_tok if num_experts > 1 else 1
        self.moe = SparseMoEBlock(config, num_experts=num_experts, num_experts_per_tok=num_experts_per_tok)
        self.input_layernorm = RMSNorm(config.d_model, eps=config.rms_norm_eps)
        self.pre_moe_layernorm = RMSNorm(config.d_model, eps=config.rms_norm_eps)

    def forward(self, x, mamba_mask, cache=None):
        # x: (B, L, D)

        # outputs: (B, L, D)

        # mamba
        residual = x
        x = self.input_layernorm(x)
        x, (h_n, c_n) = self.lstm(x)

        # -------- post-mask --------
        if mamba_mask is not None:
            x = x * mamba_mask.unsqueeze(-1)
        x = residual + x

        # FFN
        residual = x
        x = self.pre_moe_layernorm(x)
        x, router_logits = self.moe(x)
        x = residual + x

        outputs = (x, router_logits)

        return outputs, cache, None


class TransformerLayer(nn.Module):
    def __init__(self, config: JambaLMConfig, args, num_experts: int):
        super().__init__()

        self.config = config

        attention_head_size = args.hidden_size // args.output_self_attn_heads_num
        self.self_attn = MultiHeadedAttention(
            args.hidden_size, args.heads_num, attention_head_size, args.output_self_attn_dropout)

        num_experts_per_tok = config.num_experts_per_tok if num_experts > 1 else 1
        self.moe = SparseMoEBlock(config, num_experts=num_experts, num_experts_per_tok=num_experts_per_tok)
        self.input_layernorm = RMSNorm(config.d_model, eps=config.rms_norm_eps)
        self.pre_moe_layernorm = RMSNorm(config.d_model, eps=config.rms_norm_eps)

    def forward(self, x, mask, position_bias):
        # x: (B, L, D)
        # outputs: (B, L, D)

        # transformer
        residual = x
        x = self.input_layernorm(x)
        x, attn = self.self_attn(x, x, x, mask, position_bias)
        x = residual + x

        # FFN
        residual = x
        x = self.pre_moe_layernorm(x)
        x, router_logits = self.moe(x)
        x = residual + x

        return x, router_logits, attn


class SparseMoEBlock(nn.Module):
    def __init__(self, config: JambaLMConfig, num_experts: int, num_experts_per_tok: int):
        super().__init__()

        self.hidden_dim = config.d_model
        self.ffn_dim = config.mlp_size

        self.num_experts = num_experts
        self.top_k = num_experts_per_tok

        if num_experts > 1:
            self.router = nn.Linear(self.hidden_dim, self.num_experts, bias=False)
        else:
            self.router = None

        self.experts = nn.ModuleList([MLP(config) for _ in range(self.num_experts)])

    def forward(self, x):
        # x: (B, L, D)

        # final_hidden_states: (B, L, D)
        # router_logits: (B*L, n_experts)

        # note : it is not clear why we work with shape (B*L, D) here.
        # I copied this code from the official jamba imple, and did not have time to think it through.

        batch_size, sequence_length, hidden_dim = x.shape

        # no routing
        if self.num_experts == 1:
            final_hidden_states = self.experts[0](x)
            router_logits = torch.ones(
                (batch_size * sequence_length, 1),
                device=x.device,
                dtype=x.dtype,
                requires_grad=x.requires_grad,
            )
            return final_hidden_states, router_logits

        # routing
        x = x.view(-1, hidden_dim)  # (B*L, D)

        router_logits = self.router(x)  # (B*L, n_experts)
        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
        routing_weights = routing_weights.to(x.dtype)

        final_hidden_states = torch.zeros((batch_size * sequence_length, hidden_dim), dtype=x.dtype, device=x.device)

        # One hot encode the selected experts to create an expert mask
        # this will be used to easily index which expert is going to be sollicitated
        expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)

        # loop over all available experts in the model and perform the computation on each expert
        for expert_idx in range(self.num_experts):
            expert_layer = self.experts[expert_idx]
            idx, top_x = torch.where(expert_mask[expert_idx])

            if top_x.shape[0] == 0:
                continue

            # in torch it is faster to index using lists than torch tensors
            top_x_list = top_x.tolist()
            idx_list = idx.tolist()

            # Index the correct hidden states and compute the expert hidden state for
            # the current expert. We need to make sure to multiply the output hidden
            # states by `routing_weights` on the corresponding tokens (top-1 and top-2)
            current_state = x[None, top_x_list].reshape(-1, hidden_dim)
            current_hidden_states = expert_layer(current_state) * routing_weights[top_x_list, idx_list, None]

            # However `index_add_` only support torch tensors for indexing so we'll use
            # the `top_x` tensor here.
            final_hidden_states.index_add_(0, top_x, current_hidden_states.to(x.dtype))
        final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)

        return final_hidden_states, router_logits


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


def from_pretrained(name: str):
    """
    Returns a model loaded with pretrained weights pulled from HuggingFace.
    The model has to follow the same structure as the original Jamba model on HF (ai21labs/Jamba-v0.1).
    You can easily adapt this function.

    Args:
        name: for example:
            * 'TechxGenus/Mini-Jamba'
            * 'ai21labs/Jamba-v0.1'

    Returns:
        model: a Jamba model configured with the proper parameters and initialized with the proper weights
    """

    try:
        from transformers import AutoModelForCausalLM
    except ImportError:
        print(
            "The from_pretrained function pulls weights from HuggingFace and thus needs transformers to be installed (pip install transformers)")
        return

    model_hf = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.float32, use_mamba_kernels=False,
                                                    device_map="auto", trust_remote_code=True)

    #  copy config data
    config = JambaLMConfig(vocab_size=model_hf.config.vocab_size, d_model=model_hf.config.hidden_size,
                           n_layers=model_hf.config.num_hidden_layers,
                           rms_norm_eps=model_hf.config.rms_norm_eps, mlp_size=model_hf.config.intermediate_size,
                           inner_layernorms=model_hf.config.mamba_inner_layernorms,
                           expand_factor=model_hf.config.mamba_expand, dt_rank=model_hf.config.mamba_dt_rank,
                           d_state=model_hf.config.mamba_d_state,
                           d_conv=model_hf.config.mamba_d_conv, conv_bias=model_hf.config.mamba_conv_bias,
                           initializer_range=model_hf.config.initializer_range,
                           num_experts=model_hf.config.num_experts,
                           num_experts_per_tok=model_hf.config.num_experts_per_tok,
                           attn_layer_offset=model_hf.config.attn_layer_offset,
                           attn_layer_period=model_hf.config.attn_layer_period,
                           expert_layer_offset=model_hf.config.expert_layer_offset,
                           expert_layer_period=model_hf.config.expert_layer_period,
                           num_key_value_heads=model_hf.config.num_key_value_heads,
                           num_attention_heads=model_hf.config.num_attention_heads,
                           pad_token_id=model_hf.config.pad_token_id, bias=model_hf.config.mamba_proj_bias,
                           attention_dropout=model_hf.config.attention_dropout,
                           tie_lm_weights=model_hf.config.tie_word_embeddings)

    model = JambaLM(config)

    #  copy weights
    for name, param in model_hf.named_parameters():
        name = name.replace("model.", "jamba.")

        if "embed_tokens" in name:
            name = "embedding.weight"

        if "final_layernorm" in name:
            name = name.replace("jamba.", "")

        counterpart_param = model.get_parameter(name)
        if counterpart_param is not None:
            counterpart_param.data.copy_(param.data)

    del model_hf

    return model


class AttentionLayer(nn.Module):
    def __init__(self, config: JambaLMConfig, num_experts: int):
        super().__init__()

        self.self_attn = AttentionSDPA(config)

        num_experts_per_tok = config.num_experts_per_tok if num_experts > 1 else 1
        self.moe = SparseMoEBlock(config, num_experts=num_experts, num_experts_per_tok=num_experts_per_tok)
        self.input_layernorm = RMSNorm(config.d_model, eps=config.rms_norm_eps)
        self.pre_moe_layernorm = RMSNorm(config.d_model, eps=config.rms_norm_eps)

    def forward(self, x, cache=None):
        # x: (B, L, D)

        # outputs: (B, L, D)

        # attention
        residual = x
        x = self.input_layernorm(x)
        x, cache = self.self_attn(x, cache)
        x = residual + x

        # FFN
        residual = x
        x = self.pre_moe_layernorm(x)
        x, router_logits = self.moe(x)
        x = residual + x

        outputs = (x, router_logits)
        return outputs, cache

    def get_empty_cache(self, batch_size, device):
        return (None, None)


class AttentionSDPA(nn.Module):
    def __init__(self, config: JambaLMConfig):
        super().__init__()

        self.config = config

        self.hidden_size = config.d_model
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.attention_dropout = config.attention_dropout

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

    def forward(self, x, cache=None):
        # x: (B, L, D)

        # attn_output: (B, L, D)

        B, L, _ = x.size()

        queries = self.q_proj(x)
        keys = self.k_proj(x)
        values = self.v_proj(x)

        queries = queries.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        keys = keys.view(B, L, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        values = values.view(B, L, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        # kv cache implementation
        if cache is not None:
            past_keys, past_values = cache

            # not first in the sequence
            if past_keys is not None:
                keys = torch.cat([past_keys, keys], dim=2)
                values = torch.cat([past_values, values], dim=2)

            cache = (keys, values)  # prepare cache for next token

        # GQA related
        keys = repeat_kv(keys, self.num_key_value_groups)
        values = repeat_kv(values, self.num_key_value_groups)

        attn_output = F.scaled_dot_product_attention(queries, keys, values,
                                                     dropout_p=self.attention_dropout if self.training else 0.0,
                                                     is_causal=(cache is None))
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(B, L, self.hidden_size)

        attn_output = self.o_proj(attn_output)

        return attn_output, cache
