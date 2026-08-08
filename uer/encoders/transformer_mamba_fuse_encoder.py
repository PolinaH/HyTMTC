import torch
import torch.nn as nn
from uer.layers.layer_norm import LayerNorm, T5LayerNorm
from uer.layers.transformer_mamba_fuse_block import Transformer_Mamba_block, Mamba_Transformer_block
import pudb


class TransformerMambaEncoder(nn.Module):
    """
    BERT encoder exploits 12 or 24 transformer layers to extract features.
    """

    def __init__(self, args):
        super(TransformerMambaEncoder, self).__init__()
        self.mask = args.mask
        self.layers_num = args.layers_num
        self.parameter_sharing = args.parameter_sharing
        self.factorized_embedding_parameterization = args.factorized_embedding_parameterization
        self.layernorm_positioning = args.layernorm_positioning
        self.relative_position_embedding = args.relative_position_embedding
        self.is_moe = args.is_moe

        if self.factorized_embedding_parameterization:
            self.linear = nn.Linear(args.emb_size, args.hidden_size)

        self.transformer_mamba_encoder = Transformer_Mamba_block(args)

        if self.layernorm_positioning == "pre":
            if args.layernorm == "t5":
                self.layer_norm = T5LayerNorm(args.hidden_size)
            else:
                self.layer_norm = LayerNorm(args.hidden_size)

    def forward(self, emb, seg, byte_transmission_idx, byte_max_len=165, transmission_max_len=21,
                input_ids=None, proto=None):
        """
        Args:
            emb: [batch_size x seq_length x emb_size]
            seg: [batch_size x seq_length]
            byte_transmission_idx: [batch_size x 2]
            byte_max_len: [batch_size] mamba运算时字节信息最大长度
            transmission_max_len: [batch_size] mamba运算时传输模式信息最大长度
        Returns:
            hidden: [batch_size x seq_length x hidden_size]
        """

        batch_size, seq_length, _ = emb.size()
        # Generate mask according to segment indicators.
        # transformer mask: [batch_size x 1 x seq_length x seq_length]
        if self.mask == "fully_visible":
            mask = (seg > 0). \
                unsqueeze(1). \
                repeat(1, seq_length, 1). \
                unsqueeze(1)
            mask = mask.float()
            # mask:sequence中有值的地方在mask中是0，后面没有值的地方在mask中是-10000
            mask = (1.0 - mask) * -10000.0  # shape: [batch_size, 1, seq_length(512), seq_length(512)]
        elif self.mask == "causal":
            mask = torch.ones(seq_length, seq_length, device=emb.device)
            mask = torch.tril(mask)
            mask = (1.0 - mask) * -10000
            mask = mask.repeat(batch_size, 1, 1, 1)
        else:
            mask_a = (seg == 1). \
                unsqueeze(1). \
                repeat(1, seq_length, 1). \
                unsqueeze(1).float()

            mask_b = (seg > 0). \
                unsqueeze(1). \
                repeat(1, seq_length, 1). \
                unsqueeze(1).float()

            mask_tril = torch.ones(seq_length, seq_length, device=emb.device)
            mask_tril = torch.tril(mask_tril)
            mask_tril = mask_tril.repeat(batch_size, 1, 1, 1)

            mask = (mask_a + mask_b + mask_tril >= 2).float()
            mask = (1.0 - mask) * -10000.0

        # mamba mask
        byte_len = torch.tensor([row[0] for row in byte_transmission_idx], device=emb.device)
        transmission_len = torch.tensor([row[1]-row[0] for row in byte_transmission_idx], device=emb.device)
        byte_mask = \
            torch.arange(byte_max_len, device=emb.device).\
                expand(batch_size, byte_max_len) < byte_len.unsqueeze(1)
        byte_mask = byte_mask.float()  # 变成 0/1

        transmission_mask = \
            torch.arange(transmission_max_len, device=emb.device).\
                expand(batch_size, transmission_max_len) < transmission_len.unsqueeze(1)
        transmission_mask = transmission_mask.float()  # 变成 0/1

        #import pdb
        #pdb.set_trace()

        # ---------------------------------------------------------------
        # transformer_mamba_encoder
        # x_byte: [batch_size, length(192), 256]
        # x_transmission: [batch_size, length(32), 256]
        x_byte, x_transmission = \
            self.transformer_mamba_encoder(emb, mask, byte_mask, transmission_mask, byte_transmission_idx,
                                           byte_max_len, transmission_max_len)
        #---------------------------------------------------------------

        return x_byte, x_transmission


class MambaTransformerEncoder(nn.Module):
    """
    BERT encoder exploits 12 or 24 transformer layers to extract features.
    """

    def __init__(self, args):
        super(MambaTransformerEncoder, self).__init__()
        self.mask = args.mask
        self.layers_num = args.layers_num
        self.parameter_sharing = args.parameter_sharing
        self.factorized_embedding_parameterization = args.factorized_embedding_parameterization
        self.layernorm_positioning = args.layernorm_positioning
        self.relative_position_embedding = args.relative_position_embedding
        self.is_moe = args.is_moe

        if self.factorized_embedding_parameterization:
            self.linear = nn.Linear(args.emb_size, args.hidden_size)

        self.mamba_transformer_encoder = Mamba_Transformer_block(args)

        if self.layernorm_positioning == "pre":
            if args.layernorm == "t5":
                self.layer_norm = T5LayerNorm(args.hidden_size)
            else:
                self.layer_norm = LayerNorm(args.hidden_size)

    def forward(self, emb, seg, byte_transmission_idx, byte_max_len=165, transmission_max_len=21,
                input_ids=None, proto=None):
        """
        Args:
            emb: [batch_size x seq_length x emb_size]
            seg: [batch_size x seq_length]
            byte_transmission_idx: [batch_size x 2]
            byte_max_len: [batch_size] mamba运算时字节信息最大长度
            transmission_max_len: [batch_size] mamba运算时传输模式信息最大长度
        Returns:
            hidden: [batch_size x seq_length x hidden_size]
        """

        batch_size, seq_length, _ = emb.size()
        # Generate mask according to segment indicators.
        # transformer mask: [batch_size x 1 x seq_length x seq_length]
        if self.mask == "fully_visible":
            mask = (seg > 0). \
                unsqueeze(1). \
                repeat(1, seq_length, 1). \
                unsqueeze(1)
            mask = mask.float()
            # mask:sequence中有值的地方在mask中是0，后面没有值的地方在mask中是-10000
            mask = (1.0 - mask) * -10000.0  # shape: [batch_size, 1, seq_length(512), seq_length(512)]
        elif self.mask == "causal":
            mask = torch.ones(seq_length, seq_length, device=emb.device)
            mask = torch.tril(mask)
            mask = (1.0 - mask) * -10000
            mask = mask.repeat(batch_size, 1, 1, 1)
        else:
            mask_a = (seg == 1). \
                unsqueeze(1). \
                repeat(1, seq_length, 1). \
                unsqueeze(1).float()

            mask_b = (seg > 0). \
                unsqueeze(1). \
                repeat(1, seq_length, 1). \
                unsqueeze(1).float()

            mask_tril = torch.ones(seq_length, seq_length, device=emb.device)
            mask_tril = torch.tril(mask_tril)
            mask_tril = mask_tril.repeat(batch_size, 1, 1, 1)

            mask = (mask_a + mask_b + mask_tril >= 2).float()
            mask = (1.0 - mask) * -10000.0

        # mamba mask
        byte_len = torch.tensor([row[0] for row in byte_transmission_idx], device=emb.device)
        transmission_len = torch.tensor([row[1]-row[0] for row in byte_transmission_idx], device=emb.device)
        byte_mask = \
            torch.arange(byte_max_len, device=emb.device).\
                expand(batch_size, byte_max_len) < byte_len.unsqueeze(1)
        byte_mask = byte_mask.float()  # 变成 0/1

        transmission_mask = \
            torch.arange(transmission_max_len, device=emb.device).\
                expand(batch_size, transmission_max_len) < transmission_len.unsqueeze(1)
        transmission_mask = transmission_mask.float()  # 变成 0/1

        # ---------------------------------------------------------------
        # transformer_mamba_encoder
        # x_byte: [batch_size, length(192), 256]
        # x_transmission: [batch_size, length(32), 256]
        output = \
            self.mamba_transformer_encoder(emb, mask, byte_mask, transmission_mask, byte_transmission_idx,
                                           byte_max_len, transmission_max_len)
        #---------------------------------------------------------------

        return output
