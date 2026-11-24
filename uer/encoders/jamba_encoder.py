import torch
import torch.nn as nn
from uer.layers.layer_norm import LayerNorm, T5LayerNorm
from uer.layers.relative_position_embedding import RelativePositionEmbedding
from uer.layers.jamba_block_official_mamba import JambaBlock17, JambaBlock15, JambaBlock13, JambaEncoder_onlyMamba_block, \
    JambaEncoder_onlyMambaMOE_block, JambaEncoder_onlyTransformer_block, JambaEncoder_onlyTransformerMoE_block, \
    JambaEncoder_onlyLSTM_block, JambaEncoder_onlyLSTMMOE_block, JambaBlock_LSTM_Transformer
import pudb


class JambaEncoder(nn.Module):
    """
    BERT encoder exploits 12 or 24 transformer layers to extract features.
    """

    def __init__(self, args):
        super(JambaEncoder, self).__init__()
        self.mask = args.mask
        self.layers_num = args.layers_num
        self.parameter_sharing = args.parameter_sharing
        self.factorized_embedding_parameterization = args.factorized_embedding_parameterization
        self.layernorm_positioning = args.layernorm_positioning
        self.relative_position_embedding = args.relative_position_embedding
        self.is_moe = args.is_moe
        has_bias = bool(1 - args.remove_transformer_bias)
        self.jamba_atm_ratio = args.jamba_atm_ratio

        if self.factorized_embedding_parameterization:
            self.linear = nn.Linear(args.emb_size, args.hidden_size)

        if self.jamba_atm_ratio == 13:
            JambaChoice = JambaBlock13
        else:
            raise ValueError(f"The ratio of attention-to-Mamba layers "
                             f"a:m={self.atm_ratio} is not implemented now")

        if self.parameter_sharing:
            self.jamba = JambaChoice(args)
        else:
            self.jamba = nn.ModuleList(
                [JambaChoice(args) for _ in range(self.layers_num)]
            )

        if self.layernorm_positioning == "pre":
            if args.layernorm == "t5":
                self.layer_norm = T5LayerNorm(args.hidden_size)
            else:
                self.layer_norm = LayerNorm(args.hidden_size)

        if self.relative_position_embedding:
            self.relative_pos_emb = RelativePositionEmbedding(bidirectional=True, heads_num=args.heads_num,
                                                              num_buckets=args.relative_attention_buckets_num)


    def forward(self, emb, seg, byte_transmission_idx=None, input_ids=None, proto=None):
        """
        Args:
            emb: [batch_size x seq_length x emb_size]
            seg: [batch_size x seq_length]
        Returns:
            hidden: [batch_size x seq_length x hidden_size]
        """

        if self.factorized_embedding_parameterization:
            emb = self.linear(emb)

        batch_size, seq_length, _ = emb.size()
        # Generate mask according to segment indicators.
        # mask: [batch_size x 1 x seq_length x seq_length]
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
        seq_len = torch.tensor([row[1] for row in byte_transmission_idx], device=emb.device)
        seq_mask = \
            torch.arange(seq_length, device=emb.device).\
                expand(batch_size, seq_length) < seq_len.unsqueeze(1)
        seq_mask = seq_mask.float()  # 变成 0/1

        hidden = emb  # shape: [batch_size, seq_length(512), 768]

        if self.relative_position_embedding:
            position_bias = self.relative_pos_emb(hidden, hidden)
        else:
            position_bias = None

        # ---------------------------------------------------------------
        # JambaEncoder processing
        for i in range(self.layers_num):
            if self.parameter_sharing:
                hidden, _, _ = self.jamba(hidden, mask, seq_mask, position_bias=position_bias)
            else:
                hidden, _, _ = self.jamba[i](hidden, mask, seq_mask, position_bias=position_bias)
        #---------------------------------------------------------------

        if self.layernorm_positioning == "pre":
            return self.layer_norm(hidden), _
        else:
            return hidden, _  # hidden shape: [batch_size, seq_length(512), 768]


class JambaEncoder_LSTM_Transformer(nn.Module):
    """
    BERT encoder exploits 12 or 24 transformer layers to extract features.
    """

    def __init__(self, args):
        super(JambaEncoder_LSTM_Transformer, self).__init__()
        self.mask = args.mask
        self.layers_num = args.layers_num
        self.parameter_sharing = args.parameter_sharing
        self.factorized_embedding_parameterization = args.factorized_embedding_parameterization
        self.layernorm_positioning = args.layernorm_positioning
        self.relative_position_embedding = args.relative_position_embedding
        self.is_moe = args.is_moe
        has_bias = bool(1 - args.remove_transformer_bias)
        self.jamba_atm_ratio = args.jamba_atm_ratio

        if self.factorized_embedding_parameterization:
            self.linear = nn.Linear(args.emb_size, args.hidden_size)

        JambaChoice = JambaBlock_LSTM_Transformer

        if self.parameter_sharing:
            self.jamba = JambaChoice(args)
        else:
            self.jamba = nn.ModuleList(
                [JambaChoice(args) for _ in range(self.layers_num)]
            )

        if self.layernorm_positioning == "pre":
            if args.layernorm == "t5":
                self.layer_norm = T5LayerNorm(args.hidden_size)
            else:
                self.layer_norm = LayerNorm(args.hidden_size)

        if self.relative_position_embedding:
            self.relative_pos_emb = RelativePositionEmbedding(bidirectional=True, heads_num=args.heads_num,
                                                              num_buckets=args.relative_attention_buckets_num)


    def forward(self, emb, seg, byte_transmission_idx=None, input_ids=None, proto=None):
        """
        Args:
            emb: [batch_size x seq_length x emb_size]
            seg: [batch_size x seq_length]
        Returns:
            hidden: [batch_size x seq_length x hidden_size]
        """

        if self.factorized_embedding_parameterization:
            emb = self.linear(emb)

        batch_size, seq_length, _ = emb.size()
        # Generate mask according to segment indicators.
        # mask: [batch_size x 1 x seq_length x seq_length]
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
        seq_len = torch.tensor([row[1] for row in byte_transmission_idx], device=emb.device)
        seq_mask = \
            torch.arange(seq_length, device=emb.device).\
                expand(batch_size, seq_length) < seq_len.unsqueeze(1)
        seq_mask = seq_mask.float()  # 变成 0/1

        hidden = emb  # shape: [batch_size, seq_length(512), 768]

        if self.relative_position_embedding:
            position_bias = self.relative_pos_emb(hidden, hidden)
        else:
            position_bias = None

        # ---------------------------------------------------------------
        # JambaEncoder processing
        for i in range(self.layers_num):
            if self.parameter_sharing:
                hidden, _, _ = self.jamba(hidden, mask, seq_mask, position_bias=position_bias)
            else:
                hidden, _, _ = self.jamba[i](hidden, mask, seq_mask, position_bias=position_bias)
        #---------------------------------------------------------------

        if self.layernorm_positioning == "pre":
            return self.layer_norm(hidden), _
        else:
            return hidden, _  # hidden shape: [batch_size, seq_length(512), 768]


class JambaEncoder_onlyMamba(nn.Module):
    """
    BERT encoder exploits 12 or 24 transformer layers to extract features.
    """

    def __init__(self, args):
        super(JambaEncoder_onlyMamba, self).__init__()
        self.mask = args.mask
        self.layers_num = args.layers_num
        self.parameter_sharing = args.parameter_sharing
        self.factorized_embedding_parameterization = args.factorized_embedding_parameterization
        self.layernorm_positioning = args.layernorm_positioning
        self.relative_position_embedding = args.relative_position_embedding
        self.is_moe = args.is_moe
        has_bias = bool(1 - args.remove_transformer_bias)
        self.jamba_atm_ratio = args.jamba_atm_ratio

        if self.factorized_embedding_parameterization:
            self.linear = nn.Linear(args.emb_size, args.hidden_size)

        JambaChoice = JambaEncoder_onlyMamba_block

        if self.parameter_sharing:
            self.jamba = JambaChoice(args)
        else:
            self.jamba = nn.ModuleList(
                [JambaChoice(args) for _ in range(self.layers_num)]
            )

        if self.layernorm_positioning == "pre":
            if args.layernorm == "t5":
                self.layer_norm = T5LayerNorm(args.hidden_size)
            else:
                self.layer_norm = LayerNorm(args.hidden_size)

        if self.relative_position_embedding:
            self.relative_pos_emb = RelativePositionEmbedding(bidirectional=True, heads_num=args.heads_num,
                                                              num_buckets=args.relative_attention_buckets_num)


    def forward(self, emb, seg, byte_transmission_idx=None, input_ids=None, proto=None):
        """
        Args:
            emb: [batch_size x seq_length x emb_size]
            seg: [batch_size x seq_length]
        Returns:
            hidden: [batch_size x seq_length x hidden_size]
        """

        if self.factorized_embedding_parameterization:
            emb = self.linear(emb)

        batch_size, seq_length, _ = emb.size()

        # mamba mask
        seq_len = torch.tensor([row[1] for row in byte_transmission_idx], device=emb.device)
        seq_mask = \
            torch.arange(seq_length, device=emb.device).\
                expand(batch_size, seq_length) < seq_len.unsqueeze(1)
        seq_mask = seq_mask.float()  # 变成 0/1

        hidden = emb  # shape: [batch_size, seq_length(512), 768]

        if self.relative_position_embedding:
            position_bias = self.relative_pos_emb(hidden, hidden)
        else:
            position_bias = None

        # ---------------------------------------------------------------
        # JambaEncoder processing
        for i in range(self.layers_num):
            if self.parameter_sharing:
                hidden, _, _ = self.jamba(hidden, seq_mask, position_bias=position_bias)
            else:
                hidden, _, _ = self.jamba[i](hidden, seq_mask, position_bias=position_bias)
        #---------------------------------------------------------------

        if self.layernorm_positioning == "pre":
            return self.layer_norm(hidden), _
        else:
            return hidden, _  # hidden shape: [batch_size, seq_length(512), 768]


class JambaEncoder_onlyMambaMOE(nn.Module):
    """
    BERT encoder exploits 12 or 24 transformer layers to extract features.
    """

    def __init__(self, args):
        super(JambaEncoder_onlyMambaMOE, self).__init__()
        self.mask = args.mask
        self.layers_num = args.layers_num
        self.parameter_sharing = args.parameter_sharing
        self.factorized_embedding_parameterization = args.factorized_embedding_parameterization
        self.layernorm_positioning = args.layernorm_positioning
        self.relative_position_embedding = args.relative_position_embedding
        self.is_moe = args.is_moe
        has_bias = bool(1 - args.remove_transformer_bias)
        self.jamba_atm_ratio = args.jamba_atm_ratio

        if self.factorized_embedding_parameterization:
            self.linear = nn.Linear(args.emb_size, args.hidden_size)

        JambaChoice = JambaEncoder_onlyMambaMOE_block

        if self.parameter_sharing:
            self.jamba = JambaChoice(args)
        else:
            self.jamba = nn.ModuleList(
                [JambaChoice(args) for _ in range(self.layers_num)]
            )

        if self.layernorm_positioning == "pre":
            if args.layernorm == "t5":
                self.layer_norm = T5LayerNorm(args.hidden_size)
            else:
                self.layer_norm = LayerNorm(args.hidden_size)

        if self.relative_position_embedding:
            self.relative_pos_emb = RelativePositionEmbedding(bidirectional=True, heads_num=args.heads_num,
                                                              num_buckets=args.relative_attention_buckets_num)


    def forward(self, emb, seg, byte_transmission_idx=None, input_ids=None, proto=None):
        """
        Args:
            emb: [batch_size x seq_length x emb_size]
            seg: [batch_size x seq_length]
        Returns:
            hidden: [batch_size x seq_length x hidden_size]
        """

        if self.factorized_embedding_parameterization:
            emb = self.linear(emb)

        batch_size, seq_length, _ = emb.size()

        # mamba mask
        seq_len = torch.tensor([row[1] for row in byte_transmission_idx], device=emb.device)
        seq_mask = \
            torch.arange(seq_length, device=emb.device).\
                expand(batch_size, seq_length) < seq_len.unsqueeze(1)
        seq_mask = seq_mask.float()  # 变成 0/1

        hidden = emb  # shape: [batch_size, seq_length(512), 768]

        if self.relative_position_embedding:
            position_bias = self.relative_pos_emb(hidden, hidden)
        else:
            position_bias = None

        # ---------------------------------------------------------------
        # JambaEncoder processing
        for i in range(self.layers_num):
            if self.parameter_sharing:
                hidden, _, _ = self.jamba(hidden, seq_mask, position_bias=position_bias)
            else:
                hidden, _, _ = self.jamba[i](hidden, seq_mask, position_bias=position_bias)
        #---------------------------------------------------------------

        if self.layernorm_positioning == "pre":
            return self.layer_norm(hidden), _
        else:
            return hidden, _  # hidden shape: [batch_size, seq_length(512), 768]


class JambaEncoder_onlyTransformer(nn.Module):
    """
    BERT encoder exploits 12 or 24 transformer layers to extract features.
    """

    def __init__(self, args):
        super(JambaEncoder_onlyTransformer, self).__init__()
        self.mask = args.mask
        self.layers_num = args.layers_num
        self.parameter_sharing = args.parameter_sharing
        self.factorized_embedding_parameterization = args.factorized_embedding_parameterization
        self.layernorm_positioning = args.layernorm_positioning
        self.relative_position_embedding = args.relative_position_embedding
        self.is_moe = args.is_moe

        if self.factorized_embedding_parameterization:
            self.linear = nn.Linear(args.emb_size, args.hidden_size)

        JambaChoice = JambaEncoder_onlyTransformer_block

        if self.parameter_sharing:
            self.jamba = JambaChoice(args)
        else:
            self.jamba = nn.ModuleList(
                [JambaChoice(args) for _ in range(self.layers_num)]
            )

        if self.layernorm_positioning == "pre":
            if args.layernorm == "t5":
                self.layer_norm = T5LayerNorm(args.hidden_size)
            else:
                self.layer_norm = LayerNorm(args.hidden_size)

        if self.relative_position_embedding:
            self.relative_pos_emb = RelativePositionEmbedding(bidirectional=True, heads_num=args.heads_num,
                                                              num_buckets=args.relative_attention_buckets_num)


    def forward(self, emb, seg, byte_transmission_idx=None, input_ids=None, proto=None):
        """
        Args:
            emb: [batch_size x seq_length x emb_size]
            seg: [batch_size x seq_length]
        Returns:
            hidden: [batch_size x seq_length x hidden_size]
        """

        if self.factorized_embedding_parameterization:
            emb = self.linear(emb)

        batch_size, seq_length, _ = emb.size()
        # Generate mask according to segment indicators.
        # mask: [batch_size x 1 x seq_length x seq_length]
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

        hidden = emb  # shape: [batch_size, seq_length(512), 768]

        if self.relative_position_embedding:
            position_bias = self.relative_pos_emb(hidden, hidden)
        else:
            position_bias = None

        # ---------------------------------------------------------------
        # JambaEncoder processing
        for i in range(self.layers_num):
            if self.parameter_sharing:
                hidden, _, attn = self.jamba(hidden, mask, position_bias=position_bias)
            else:
                hidden, _, attn = self.jamba[i](hidden, mask, position_bias=position_bias)
        #---------------------------------------------------------------

        if self.layernorm_positioning == "pre":
            return self.layer_norm(hidden), attn
        else:
            return hidden, attn  # hidden shape: [batch_size, seq_length(512), 768]


class JambaEncoder_onlyTransformerMoE(nn.Module):
    """
    BERT encoder exploits 12 or 24 transformer layers to extract features.
    """

    def __init__(self, args):
        super(JambaEncoder_onlyTransformerMoE, self).__init__()
        self.mask = args.mask
        self.layers_num = args.layers_num
        self.parameter_sharing = args.parameter_sharing
        self.factorized_embedding_parameterization = args.factorized_embedding_parameterization
        self.layernorm_positioning = args.layernorm_positioning
        self.relative_position_embedding = args.relative_position_embedding
        self.is_moe = args.is_moe

        if self.factorized_embedding_parameterization:
            self.linear = nn.Linear(args.emb_size, args.hidden_size)

        JambaChoice = JambaEncoder_onlyTransformerMoE_block

        if self.parameter_sharing:
            self.jamba = JambaChoice(args)
        else:
            self.jamba = nn.ModuleList(
                [JambaChoice(args) for _ in range(self.layers_num)]
            )

        if self.layernorm_positioning == "pre":
            if args.layernorm == "t5":
                self.layer_norm = T5LayerNorm(args.hidden_size)
            else:
                self.layer_norm = LayerNorm(args.hidden_size)

        if self.relative_position_embedding:
            self.relative_pos_emb = RelativePositionEmbedding(bidirectional=True, heads_num=args.heads_num,
                                                              num_buckets=args.relative_attention_buckets_num)


    def forward(self, emb, seg, byte_transmission_idx=None, input_ids=None, proto=None):
        """
        Args:
            emb: [batch_size x seq_length x emb_size]
            seg: [batch_size x seq_length]
        Returns:
            hidden: [batch_size x seq_length x hidden_size]
        """

        if self.factorized_embedding_parameterization:
            emb = self.linear(emb)

        batch_size, seq_length, _ = emb.size()
        # Generate mask according to segment indicators.
        # mask: [batch_size x 1 x seq_length x seq_length]
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

        hidden = emb  # shape: [batch_size, seq_length(512), 768]

        if self.relative_position_embedding:
            position_bias = self.relative_pos_emb(hidden, hidden)
        else:
            position_bias = None

        # ---------------------------------------------------------------
        # JambaEncoder processing
        for i in range(self.layers_num):
            if self.parameter_sharing:
                hidden, _, attn = self.jamba(hidden, mask, position_bias=position_bias)
            else:
                hidden, _, attn = self.jamba[i](hidden, mask, position_bias=position_bias)
        #---------------------------------------------------------------

        if self.layernorm_positioning == "pre":
            return self.layer_norm(hidden), attn
        else:
            return hidden, attn  # hidden shape: [batch_size, seq_length(512), 768]


class JambaEncoder_onlyLSTM(nn.Module):
    """
    BERT encoder exploits 12 or 24 transformer layers to extract features.
    """

    def __init__(self, args):
        super(JambaEncoder_onlyLSTM, self).__init__()
        self.mask = args.mask
        self.layers_num = args.layers_num
        self.parameter_sharing = args.parameter_sharing
        self.factorized_embedding_parameterization = args.factorized_embedding_parameterization
        self.layernorm_positioning = args.layernorm_positioning
        self.relative_position_embedding = args.relative_position_embedding
        self.is_moe = args.is_moe
        has_bias = bool(1 - args.remove_transformer_bias)
        self.jamba_atm_ratio = args.jamba_atm_ratio

        if self.factorized_embedding_parameterization:
            self.linear = nn.Linear(args.emb_size, args.hidden_size)

        JambaChoice = JambaEncoder_onlyLSTM_block

        if self.parameter_sharing:
            self.jamba = JambaChoice(args)
        else:
            self.jamba = nn.ModuleList(
                [JambaChoice(args) for _ in range(self.layers_num)]
            )

        if self.layernorm_positioning == "pre":
            if args.layernorm == "t5":
                self.layer_norm = T5LayerNorm(args.hidden_size)
            else:
                self.layer_norm = LayerNorm(args.hidden_size)

        if self.relative_position_embedding:
            self.relative_pos_emb = RelativePositionEmbedding(bidirectional=True, heads_num=args.heads_num,
                                                              num_buckets=args.relative_attention_buckets_num)


    def forward(self, emb, seg, byte_transmission_idx=None, input_ids=None, proto=None):
        """
        Args:
            emb: [batch_size x seq_length x emb_size]
            seg: [batch_size x seq_length]
        Returns:
            hidden: [batch_size x seq_length x hidden_size]
        """

        if self.factorized_embedding_parameterization:
            emb = self.linear(emb)

        batch_size, seq_length, _ = emb.size()

        # mamba mask
        seq_len = torch.tensor([row[1] for row in byte_transmission_idx], device=emb.device)
        seq_mask = \
            torch.arange(seq_length, device=emb.device).\
                expand(batch_size, seq_length) < seq_len.unsqueeze(1)
        seq_mask = seq_mask.float()  # 变成 0/1

        hidden = emb  # shape: [batch_size, seq_length(512), 768]

        if self.relative_position_embedding:
            position_bias = self.relative_pos_emb(hidden, hidden)
        else:
            position_bias = None

        # ---------------------------------------------------------------
        # JambaEncoder processing
        for i in range(self.layers_num):
            if self.parameter_sharing:
                hidden, _, _ = self.jamba(hidden, seq_mask, position_bias=position_bias)
            else:
                hidden, _, _ = self.jamba[i](hidden, seq_mask, position_bias=position_bias)
        #---------------------------------------------------------------

        if self.layernorm_positioning == "pre":
            return self.layer_norm(hidden), _
        else:
            return hidden, _  # hidden shape: [batch_size, seq_length(512), 768]


class JambaEncoder_onlyLSTMMoE(nn.Module):
    """
    BERT encoder exploits 12 or 24 transformer layers to extract features.
    """

    def __init__(self, args):
        super(JambaEncoder_onlyLSTMMoE, self).__init__()
        self.mask = args.mask
        self.layers_num = args.layers_num
        self.parameter_sharing = args.parameter_sharing
        self.factorized_embedding_parameterization = args.factorized_embedding_parameterization
        self.layernorm_positioning = args.layernorm_positioning
        self.relative_position_embedding = args.relative_position_embedding
        self.is_moe = args.is_moe
        has_bias = bool(1 - args.remove_transformer_bias)
        self.jamba_atm_ratio = args.jamba_atm_ratio

        if self.factorized_embedding_parameterization:
            self.linear = nn.Linear(args.emb_size, args.hidden_size)

        JambaChoice = JambaEncoder_onlyLSTMMOE_block

        if self.parameter_sharing:
            self.jamba = JambaChoice(args)
        else:
            self.jamba = nn.ModuleList(
                [JambaChoice(args) for _ in range(self.layers_num)]
            )

        if self.layernorm_positioning == "pre":
            if args.layernorm == "t5":
                self.layer_norm = T5LayerNorm(args.hidden_size)
            else:
                self.layer_norm = LayerNorm(args.hidden_size)

        if self.relative_position_embedding:
            self.relative_pos_emb = RelativePositionEmbedding(bidirectional=True, heads_num=args.heads_num,
                                                              num_buckets=args.relative_attention_buckets_num)


    def forward(self, emb, seg, byte_transmission_idx=None, input_ids=None, proto=None):
        """
        Args:
            emb: [batch_size x seq_length x emb_size]
            seg: [batch_size x seq_length]
        Returns:
            hidden: [batch_size x seq_length x hidden_size]
        """

        if self.factorized_embedding_parameterization:
            emb = self.linear(emb)

        batch_size, seq_length, _ = emb.size()

        # mamba mask
        seq_len = torch.tensor([row[1] for row in byte_transmission_idx], device=emb.device)
        seq_mask = \
            torch.arange(seq_length, device=emb.device).\
                expand(batch_size, seq_length) < seq_len.unsqueeze(1)
        seq_mask = seq_mask.float()  # 变成 0/1

        hidden = emb  # shape: [batch_size, seq_length(512), 768]

        if self.relative_position_embedding:
            position_bias = self.relative_pos_emb(hidden, hidden)
        else:
            position_bias = None

        # ---------------------------------------------------------------
        # JambaEncoder processing
        for i in range(self.layers_num):
            if self.parameter_sharing:
                hidden, _, _ = self.jamba(hidden, seq_mask, position_bias=position_bias)
            else:
                hidden, _, _ = self.jamba[i](hidden, seq_mask, position_bias=position_bias)
        #---------------------------------------------------------------

        if self.layernorm_positioning == "pre":
            return self.layer_norm(hidden), _
        else:
            return hidden, _  # hidden shape: [batch_size, seq_length(512), 768]