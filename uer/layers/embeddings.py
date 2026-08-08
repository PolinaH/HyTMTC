import torch
import math
import torch.nn as nn
from uer.layers.layer_norm import LayerNorm
import pudb


class JambaEmbedding(nn.Module):
    """
    BERT embedding consists of three parts:
    word embedding, position embedding, and segment embedding.
    """
    def __init__(self, args, vocab_size):
        super(JambaEmbedding, self).__init__()
        self.remove_embedding_layernorm = args.remove_embedding_layernorm
        self.dropout = nn.Dropout(args.dropout)
        self.max_seq_length = args.max_seq_length
        self.word_embedding = nn.Embedding(vocab_size, args.emb_size)
        self.position_embedding = nn.Embedding(self.max_seq_length, args.emb_size)
        self.segment_embedding = nn.Embedding(11, args.emb_size)
        self.coarse_embedding = nn.Embedding(8, args.emb_size)
        self.fine_embedding = nn.Embedding(15, args.emb_size)
        if not self.remove_embedding_layernorm:
            self.layer_norm = LayerNorm(args.emb_size)

    def forward(self, src, seg, coarse, fine, sample_len):
        #pudb.set_trace()

        sequence_len = src.size(1)
        set_0_len = sequence_len-sample_len

        # src: word, shape:[batch_size,seq_length]
        # seg: segment, shape:[batch_size,seq_length(512)], e.g.[[1,1,1,1,...,2,2,2,2,...,0,0],...]
        word_emb = self.word_embedding(src)  # shape:[[batch_size, seq_length, 768]]

        # torch.arange(0, word_emb.size(1), device=word_emb.device, dtype=torch.long).unsqueeze(0).repeat(word_emb.size(0), 1)
        # above shape:[batch_size,seq_length(512)]
        pos_emb = self.position_embedding(
            torch.arange(0, word_emb.size(1), device=word_emb.device, dtype=torch.long)
            .unsqueeze(0)
            .repeat(word_emb.size(0), 1)
        )  # shape:[batch_size,seq_length(512), 768], e.g.[[  0,   1,   2,  ..., 509, 510, 511],...]

        seg_emb = self.segment_embedding(seg)  # shape:[[batch_size, seq_length, 768]]

        coarse_emb = self.coarse_embedding(coarse)
        fine_emb = self.fine_embedding(fine)

        # 五个embedding相加得到输入的最终embedding
        # shape:[[batch_size, seq_length, 768]]
        emb = word_emb + pos_emb + seg_emb + coarse_emb + fine_emb
        # 样本结束后的tokens的embedding置0
        for itm_idx in range(len(src)):
            emb[itm_idx, -set_0_len[itm_idx]:, :] = 0

        if not self.remove_embedding_layernorm:
            emb = self.layer_norm(emb)
        emb = self.dropout(emb)
        return emb


class JambaDirectionLengthEmbedding(nn.Module):
    """
    BERT embedding consists of three parts:
    word embedding, position embedding, and segment embedding.
    """
    def __init__(self, args, vocab_size):
        super(JambaDirectionLengthEmbedding, self).__init__()
        self.remove_embedding_layernorm = args.remove_embedding_layernorm
        self.dropout = nn.Dropout(args.dropout)
        self.max_seq_length = args.max_seq_length
        self.word_embedding = nn.Embedding(vocab_size, args.emb_size)
        self.position_embedding = nn.Embedding(self.max_seq_length, args.emb_size)
        self.segment_embedding = nn.Embedding(7, args.emb_size)
        self.coarse_embedding = nn.Embedding(9, args.emb_size)
        self.fine_embedding = nn.Embedding(16, args.emb_size)
        if not self.remove_embedding_layernorm:
            self.layer_norm = LayerNorm(args.emb_size)

    def forward(self, src, seg, coarse, fine, sample_len):
        #pudb.set_trace()

        sequence_len = src.size(1)
        set_0_len = sequence_len-sample_len

        # src: word, shape:[batch_size,seq_length]
        # seg: segment, shape:[batch_size,seq_length(512)], e.g.[[1,1,1,1,...,2,2,2,2,...,0,0],...]
        word_emb = self.word_embedding(src)  # shape:[[batch_size, seq_length, 768]]

        # torch.arange(0, word_emb.size(1), device=word_emb.device, dtype=torch.long).unsqueeze(0).repeat(word_emb.size(0), 1)
        # above shape:[batch_size,seq_length(512)]
        pos_emb = self.position_embedding(
            torch.arange(0, word_emb.size(1), device=word_emb.device, dtype=torch.long)
            .unsqueeze(0)
            .repeat(word_emb.size(0), 1)
        )  # shape:[batch_size,seq_length(512), 768], e.g.[[  0,   1,   2,  ..., 509, 510, 511],...]

        seg_emb = self.segment_embedding(seg)  # shape:[[batch_size, seq_length, 768]]

        coarse_emb = self.coarse_embedding(coarse)
        fine_emb = self.fine_embedding(fine)

        # 五个embedding相加得到输入的最终embedding
        # shape:[[batch_size, seq_length, 768]]
        emb = word_emb + pos_emb + seg_emb + coarse_emb + fine_emb
        # 样本结束后的tokens的embedding置0
        for itm_idx in range(len(src)):
            emb[itm_idx, -set_0_len[itm_idx]:, :] = 0

        if not self.remove_embedding_layernorm:
            emb = self.layer_norm(emb)
        emb = self.dropout(emb)
        return emb


class JambaNoCoarseEmbedding(nn.Module):
    """
    BERT embedding consists of three parts:
    word embedding, position embedding, and segment embedding.
    """
    def __init__(self, args, vocab_size):
        super(JambaNoCoarseEmbedding, self).__init__()
        self.remove_embedding_layernorm = args.remove_embedding_layernorm
        self.dropout = nn.Dropout(args.dropout)
        self.max_seq_length = args.max_seq_length
        self.word_embedding = nn.Embedding(vocab_size, args.emb_size)
        self.position_embedding = nn.Embedding(self.max_seq_length, args.emb_size)
        self.segment_embedding = nn.Embedding(7, args.emb_size)
        self.fine_embedding = nn.Embedding(16, args.emb_size)
        if not self.remove_embedding_layernorm:
            self.layer_norm = LayerNorm(args.emb_size)

    def forward(self, src, seg, coarse, fine, sample_len):
        #pudb.set_trace()

        sequence_len = src.size(1)
        set_0_len = sequence_len-sample_len

        # src: word, shape:[batch_size,seq_length]
        # seg: segment, shape:[batch_size,seq_length(512)], e.g.[[1,1,1,1,...,2,2,2,2,...,0,0],...]
        word_emb = self.word_embedding(src)  # shape:[[batch_size, seq_length, 768]]

        # torch.arange(0, word_emb.size(1), device=word_emb.device, dtype=torch.long).unsqueeze(0).repeat(word_emb.size(0), 1)
        # above shape:[batch_size,seq_length(512)]
        pos_emb = self.position_embedding(
            torch.arange(0, word_emb.size(1), device=word_emb.device, dtype=torch.long)
            .unsqueeze(0)
            .repeat(word_emb.size(0), 1)
        )  # shape:[batch_size,seq_length(512), 768], e.g.[[  0,   1,   2,  ..., 509, 510, 511],...]

        seg_emb = self.segment_embedding(seg)  # shape:[[batch_size, seq_length, 768]]

        fine_emb = self.fine_embedding(fine)

        # 五个embedding相加得到输入的最终embedding
        # shape:[[batch_size, seq_length, 768]]
        emb = word_emb + pos_emb + seg_emb + fine_emb
        # 样本结束后的tokens的embedding置0
        for itm_idx in range(len(src)):
            emb[itm_idx, -set_0_len[itm_idx]:, :] = 0

        if not self.remove_embedding_layernorm:
            emb = self.layer_norm(emb)
        emb = self.dropout(emb)
        return emb


class JambaNoFineEmbedding(nn.Module):
    """
    BERT embedding consists of three parts:
    word embedding, position embedding, and segment embedding.
    """
    def __init__(self, args, vocab_size):
        super(JambaNoFineEmbedding, self).__init__()
        self.remove_embedding_layernorm = args.remove_embedding_layernorm
        self.dropout = nn.Dropout(args.dropout)
        self.max_seq_length = args.max_seq_length
        self.word_embedding = nn.Embedding(vocab_size, args.emb_size)
        self.position_embedding = nn.Embedding(self.max_seq_length, args.emb_size)
        self.segment_embedding = nn.Embedding(7, args.emb_size)
        self.coarse_embedding = nn.Embedding(9, args.emb_size)
        if not self.remove_embedding_layernorm:
            self.layer_norm = LayerNorm(args.emb_size)

    def forward(self, src, seg, coarse, fine, sample_len):
        #pudb.set_trace()

        sequence_len = src.size(1)
        set_0_len = sequence_len-sample_len

        # src: word, shape:[batch_size,seq_length]
        # seg: segment, shape:[batch_size,seq_length(512)], e.g.[[1,1,1,1,...,2,2,2,2,...,0,0],...]
        word_emb = self.word_embedding(src)  # shape:[[batch_size, seq_length, 768]]

        # torch.arange(0, word_emb.size(1), device=word_emb.device, dtype=torch.long).unsqueeze(0).repeat(word_emb.size(0), 1)
        # above shape:[batch_size,seq_length(512)]
        pos_emb = self.position_embedding(
            torch.arange(0, word_emb.size(1), device=word_emb.device, dtype=torch.long)
            .unsqueeze(0)
            .repeat(word_emb.size(0), 1)
        )  # shape:[batch_size,seq_length(512), 768], e.g.[[  0,   1,   2,  ..., 509, 510, 511],...]

        seg_emb = self.segment_embedding(seg)  # shape:[[batch_size, seq_length, 768]]

        coarse_emb = self.coarse_embedding(coarse)

        # 五个embedding相加得到输入的最终embedding
        # shape:[[batch_size, seq_length, 768]]
        emb = word_emb + pos_emb + seg_emb + coarse_emb
        # 样本结束后的tokens的embedding置0
        for itm_idx in range(len(src)):
            emb[itm_idx, -set_0_len[itm_idx]:, :] = 0

        if not self.remove_embedding_layernorm:
            emb = self.layer_norm(emb)
        emb = self.dropout(emb)
        return emb


class JambaNoCoarseFineEmbedding(nn.Module):
    """
    BERT embedding consists of three parts:
    word embedding, position embedding, and segment embedding.
    """
    def __init__(self, args, vocab_size):
        super(JambaNoCoarseFineEmbedding, self).__init__()
        self.remove_embedding_layernorm = args.remove_embedding_layernorm
        self.dropout = nn.Dropout(args.dropout)
        self.max_seq_length = args.max_seq_length
        self.word_embedding = nn.Embedding(vocab_size, args.emb_size)
        self.position_embedding = nn.Embedding(self.max_seq_length, args.emb_size)
        self.segment_embedding = nn.Embedding(7, args.emb_size)
        if not self.remove_embedding_layernorm:
            self.layer_norm = LayerNorm(args.emb_size)

    def forward(self, src, seg, coarse, fine, sample_len):
        #pudb.set_trace()

        sequence_len = src.size(1)
        set_0_len = sequence_len-sample_len

        # src: word, shape:[batch_size,seq_length]
        # seg: segment, shape:[batch_size,seq_length(512)], e.g.[[1,1,1,1,...,2,2,2,2,...,0,0],...]
        word_emb = self.word_embedding(src)  # shape:[[batch_size, seq_length, 768]]

        # torch.arange(0, word_emb.size(1), device=word_emb.device, dtype=torch.long).unsqueeze(0).repeat(word_emb.size(0), 1)
        # above shape:[batch_size,seq_length(512)]
        pos_emb = self.position_embedding(
            torch.arange(0, word_emb.size(1), device=word_emb.device, dtype=torch.long)
            .unsqueeze(0)
            .repeat(word_emb.size(0), 1)
        )  # shape:[batch_size,seq_length(512), 768], e.g.[[  0,   1,   2,  ..., 509, 510, 511],...]

        seg_emb = self.segment_embedding(seg)  # shape:[[batch_size, seq_length, 768]]

        # 五个embedding相加得到输入的最终embedding
        # shape:[[batch_size, seq_length, 768]]
        emb = word_emb + pos_emb + seg_emb
        # 样本结束后的tokens的embedding置0
        for itm_idx in range(len(src)):
            emb[itm_idx, -set_0_len[itm_idx]:, :] = 0

        if not self.remove_embedding_layernorm:
            emb = self.layer_norm(emb)
        emb = self.dropout(emb)
        return emb


class WordEmbedding(nn.Module):
    """
    """

    def __init__(self, args, vocab_size):
        super(WordEmbedding, self).__init__()
        self.remove_embedding_layernorm = args.remove_embedding_layernorm
        self.dropout = nn.Dropout(args.dropout)
        self.word_embedding = nn.Embedding(vocab_size, args.emb_size)
        if not self.remove_embedding_layernorm:
            self.layer_norm = LayerNorm(args.emb_size)

    def forward(self, src, _):
        emb = self.word_embedding(src)
        if not self.remove_embedding_layernorm:
            emb = self.layer_norm(emb)
        emb = self.dropout(emb)
        return emb


class WordPosEmbedding(nn.Module):
    """
    GPT embedding consists of two parts:
    word embedding and position embedding.
    """

    def __init__(self, args, vocab_size):
        super(WordPosEmbedding, self).__init__()
        self.remove_embedding_layernorm = args.remove_embedding_layernorm
        self.dropout = nn.Dropout(args.dropout)
        self.max_seq_length = args.max_seq_length
        self.word_embedding = nn.Embedding(vocab_size, args.emb_size)
        self.position_embedding = nn.Embedding(self.max_seq_length, args.emb_size)
        if not self.remove_embedding_layernorm:
            self.layer_norm = LayerNorm(args.emb_size)

    def forward(self, src, _):
        word_emb = self.word_embedding(src)
        pos_emb = self.position_embedding(
            torch.arange(0, word_emb.size(1), device=word_emb.device, dtype=torch.long)
            .unsqueeze(0)
            .repeat(word_emb.size(0), 1)
        )

        emb = word_emb + pos_emb
        if not self.remove_embedding_layernorm:
            emb = self.layer_norm(emb)
        emb = self.dropout(emb)
        return emb


class WordPosSegEmbedding(nn.Module):
    """
    BERT embedding consists of three parts:
    word embedding, position embedding, and segment embedding.
    """
    def __init__(self, args, vocab_size):
        super(WordPosSegEmbedding, self).__init__()
        self.remove_embedding_layernorm = args.remove_embedding_layernorm
        self.dropout = nn.Dropout(args.dropout)
        self.max_seq_length = args.max_seq_length
        self.word_embedding = nn.Embedding(vocab_size, args.emb_size)
        self.position_embedding = nn.Embedding(self.max_seq_length, args.emb_size)
        self.segment_embedding = nn.Embedding(3, args.emb_size)
        if not self.remove_embedding_layernorm:
            self.layer_norm = LayerNorm(args.emb_size)

    def forward(self, src, seg):
        # src: word, shape:[batch_size,seq_length(512)]
        # seg: segment, shape:[batch_size,seq_length(512)], e.g.[[1,1,1,1,...,2,2,2,2,...,0,0],...]
        word_emb = self.word_embedding(src)  # shape:[[batch_size, seq_length(512), 768]]

        # torch.arange(0, word_emb.size(1), device=word_emb.device, dtype=torch.long).unsqueeze(0).repeat(word_emb.size(0), 1)
        # above shape:[batch_size,seq_length(512)]
        pos_emb = self.position_embedding(
            torch.arange(0, word_emb.size(1), device=word_emb.device, dtype=torch.long)
            .unsqueeze(0)
            .repeat(word_emb.size(0), 1)
        )  # shape:[batch_size,seq_length(512), 768], e.g.[[  0,   1,   2,  ..., 509, 510, 511],...]
        seg_emb = self.segment_embedding(seg)  # shape:[[batch_size, seq_length(512), 768]]

        # 三个embedding相加得到输入的最终embedding
        emb = word_emb + pos_emb + seg_emb  # shape:[[batch_size, seq_length(512), 768]]
        if not self.remove_embedding_layernorm:
            emb = self.layer_norm(emb)
        emb = self.dropout(emb)
        return emb


class WordSinusoidalposEmbedding(nn.Module):
    """Sinusoidal positional encoding for non-recurrent neural networks.
    Implementation based on "Attention Is All You Need"
    :cite:`DBLP:journals/corr/VaswaniSPUJGKP17`
    Args:
       dropout (float): dropout parameter
       dim (int): embedding size
    """

    def __init__(self, args, vocab_size):
        super(WordSinusoidalposEmbedding, self).__init__()
        if args.emb_size % 2 != 0:
            raise ValueError("Cannot use sin/cos positional encoding with "
                             "odd dim (got dim={:d})".format(args.emb_size))
        self.max_seq_length = args.max_seq_length
        pe = torch.zeros(self.max_seq_length, args.emb_size)
        position = torch.arange(0, self.max_seq_length).unsqueeze(1)
        div_term = torch.exp(
            (
                torch.arange(0, args.emb_size, 2, dtype=torch.float)
                *- (math.log(10000.0) / args.emb_size)
            )
        )
        pe[:, 0::2] = torch.sin(position.float() * div_term)
        pe[:, 1::2] = torch.cos(position.float() * div_term)
        pe = pe.unsqueeze(1)
        self.register_buffer("pe", pe)

        self.word_embedding = nn.Embedding(vocab_size, args.emb_size)

        self.dropout = nn.Dropout(args.dropout)

    def forward(self, src, _):
        """Embed inputs.
        Args:
            emb (FloatTensor): Sequence of word vectors
                ``(batch_size, seq_len, self.dim)``
            step (int or NoneType): If stepwise (``seq_len = 1``), use
                the encoding for this position.
        """
        word_emb = self.word_embedding(src)
        emb = word_emb * math.sqrt(word_emb.size(-1))
        emb = emb + self.pe[: emb.size(1)].transpose(0, 1)
        emb = self.dropout(emb)
        return emb
