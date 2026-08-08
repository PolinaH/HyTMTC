# -*- coding: utf-8 -*-
from torch import Tensor, nn
from zeta import MambaBlock
from zeta.nn import FeedForward
from zeta import MultiQueryAttention
from zeta.nn.modules.simple_rmsnorm import SimpleRMSNorm
from zeta.moe import MoE
from zeta.nn import OutputHead
from uer.layers.transformer import TransformerLayer, TransformerMOELayer
import pudb


class MambaMoELayer(nn.Module):
    def __init__(
        self,
        dim: int,
        depth: int,
        d_state: int,
        d_conv: int,
        num_experts: int = 8,
        num_experts_per_token: int = 2,
        *args,
        **kwargs,
    ):
        """
        Initialize the MambaMoELayer.

        Args:
            dim (int): Dimension of the input tensor.
            d_state (int): Dimension of the state tensor.
            d_conv (int): Dimension of the convolutional tensor.
            num_experts (int, optional): Number of experts. Defaults to 8.
            num_experts_per_token (int, optional): Number of experts per token. Defaults to 2.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """
        super().__init__()
        self.dim = dim
        self.depth = depth
        self.d_state = d_state
        self.d_conv = d_conv
        self.num_experts = num_experts
        self.num_experts_per_token = num_experts_per_token

        # Mamba
        self.mamba = MambaBlock(
            dim=self.dim,
            depth=self.depth,
            d_state=self.d_state,
            d_conv=self.d_conv,
        )

        # MoE
        self.moe = MoE(
            dim=self.dim,
            num_experts=self.num_experts,
            hidden_dim=self.dim * 4,
        )

    def forward(self, x: Tensor):
        """
        Forward pass of the MambaMoELayer.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Output tensor after applying the MambaMoELayer.
        """
        skip = x

        x = SimpleRMSNorm(self.dim)(x)
        x = self.mamba(x) + x

        x = SimpleRMSNorm(self.dim)(x)
        moe_out, _ = self.moe(x)
        x = moe_out + skip
        return x


class JambaZetaBlock13(nn.Module):
    """
    JambaBlock is a module that combines MambaBlock, MambaMoELayer, and TransformerBlock
    to process input tensors.

    Args:
        dim (int): The input dimension.
        d_state (int): The dimension of the state in MambaBlock and MambaMoELayer.
        d_conv (int): The dimension of the convolutional output in MambaBlock and MambaMoELayer.
        heads (int): The number of attention heads in TransformerBlock.
        num_experts (int, optional): The number of experts in MambaMoELayer. Defaults to 8.
        num_experts_per_token (int, optional): The number of experts per token in MambaMoELayer. Defaults to 2.

    Attributes:
        dim (int): The input dimension.
        d_state (int): The dimension of the state in MambaBlock and MambaMoELayer.
        d_conv (int): The dimension of the convolutional output in MambaBlock and MambaMoELayer.
        heads (int): The number of attention heads in TransformerBlock.
        num_experts (int): The number of experts in MambaMoELayer.
        num_experts_per_tok (int): The number of experts per token in MambaMoELayer.
        mamba_layer (MambaBlock): The MambaBlock layer.
        mamba_moe_layer (MambaMoELayer): The MambaMoELayer layer.
        transformer (TransformerBlock): The TransformerBlock layer.

    """

    def __init__(self,args):
        super().__init__()
        self.dim = args.emb_size
        self.mamba_depth = args.mamba_depth
        self.d_state = args.mamba_d_state
        self.d_conv = args.mamba_d_conv
        self.num_experts = args.moe_num_experts
        self.num_experts_per_token = args.moe_num_experts_per_token

        # Mamba
        self.mamba_layer = MambaBlock(
            dim=self.dim,
            depth=self.mamba_depth,
            d_state=self.d_state,
            d_conv=self.d_conv,
        )

        # Mamba MoE layer
        self.mamba_moe_layer = MambaMoELayer(
            dim=self.dim,
            depth=self.mamba_depth,
            d_state=self.d_state,
            d_conv=self.d_conv,
            num_experts=self.num_experts,
            num_experts_per_token=self.num_experts_per_token,
        )

        # Transformer
        self.transformer = TransformerLayer(args)


    def forward(self, hidden, mask, position_bias) -> Tensor:
        #  l=4, a:m=1:3 ratio of attention-to-Mamba layers
        #  MoE applied every e = 2 layers.
        hidden = self.mamba_layer(hidden)
        hidden = self.mamba_moe_layer(hidden)
        hidden,_ = self.transformer(hidden, mask, position_bias=position_bias)
        hidden = self.mamba_moe_layer(hidden)
        return hidden


class JambaZetaBlock17(nn.Module):
    """
    JambaBlock is a module that combines MambaBlock, MambaMoELayer, and TransformerBlock
    to process input tensors.

    Args:
        dim (int): The input dimension.
        d_state (int): The dimension of the state in MambaBlock and MambaMoELayer.
        d_conv (int): The dimension of the convolutional output in MambaBlock and MambaMoELayer.
        heads (int): The number of attention heads in TransformerBlock.
        num_experts (int, optional): The number of experts in MambaMoELayer. Defaults to 8.
        num_experts_per_token (int, optional): The number of experts per token in MambaMoELayer. Defaults to 2.

    Attributes:
        dim (int): The input dimension.
        d_state (int): The dimension of the state in MambaBlock and MambaMoELayer.
        d_conv (int): The dimension of the convolutional output in MambaBlock and MambaMoELayer.
        heads (int): The number of attention heads in TransformerBlock.
        num_experts (int): The number of experts in MambaMoELayer.
        num_experts_per_tok (int): The number of experts per token in MambaMoELayer.
        mamba_layer (MambaBlock): The MambaBlock layer.
        mamba_moe_layer (MambaMoELayer): The MambaMoELayer layer.
        transformer (TransformerBlock): The TransformerBlock layer.

    """

    def __init__(self,args):
        super().__init__()
        self.dim = args.emb_size
        self.mamba_depth = args.mamba_depth
        self.d_state = args.mamba_d_state
        self.d_conv = args.mamba_d_conv
        self.num_experts = args.moe_num_experts
        self.num_experts_per_token = args.moe_num_experts_per_token

        # Mamba
        self.mamba_layer = MambaBlock(
            dim=self.dim,
            depth=self.mamba_depth,
            d_state=self.d_state,
            d_conv=self.d_conv,
        )

        # Mamba MoE layer
        self.mamba_moe_layer = MambaMoELayer(
            dim=self.dim,
            depth=self.mamba_depth,
            d_state=self.d_state,
            d_conv=self.d_conv,
            num_experts=self.num_experts,
            num_experts_per_token=self.num_experts_per_token,
        )

        # Transformer
        self.transformer = TransformerLayer(args)


    def forward(self, hidden, mask, position_bias) -> Tensor:
        #  l=8, a:m=1:7 ratio of attention-to-Mamba layers
        #  MoE applied every e = 2 layers.
        pudb.set_trace()
        hidden = self.mamba_layer(hidden)
        hidden = self.mamba_moe_layer(hidden)
        hidden = self.mamba_layer(hidden)
        hidden = self.mamba_moe_layer(hidden)
        pudb.set_trace()
        hidden,_ = self.transformer(hidden, mask, position_bias=position_bias)
        pudb.set_trace()
        hidden = self.mamba_moe_layer(hidden)
        hidden = self.mamba_layer(hidden)
        hidden = self.mamba_moe_layer(hidden)
        return hidden