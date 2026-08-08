import torch
import torch.nn as nn
from uer.targets import *
import pudb

class TransformerMambaTarget(MlmTarget):
    """
    BERT exploits masked language modeling (MLM)
    and mixed sentence prediction (MSP) for pretraining.
    """

    def __init__(self, args, vocab_size):
        super(TransformerMambaTarget, self).__init__(args, vocab_size)
        self.max_seq_length = args.max_seq_length

    def forward(self, memory_bank, tgt, byte_transmission_idx=None, *args, **kwargs):
        """
        Args:
            memory_bank: [batch_size x seq_length x hidden_size]
            tgt: tgt_mlm [batch_size x seq_length]
            seg: segment for input

        Returns:
            loss_mlm: Masked language model loss.
            correct_mlm: Number of words that are predicted correctly.
            denominator: Number of masked words.
        """
        # Masked language model (MLM).
        bs = len(memory_bank[0])
        d = memory_bank[0][0].size(-1)  # dimension

        #import pdb
        #pdb.set_trace()

        pred_emb = torch.zeros(bs, self.max_seq_length, d, device=memory_bank[0].device)
        for i in range(bs):
            byte_end = byte_transmission_idx[i][0]
            transmission_end = byte_transmission_idx[i][1]
            pred_emb[i, :byte_end, :] = memory_bank[0][i][:byte_end, :]
            pred_emb[i, byte_end:transmission_end, :] = \
                memory_bank[1][i][:transmission_end-byte_end, :]

        loss_mlm, correct_mlm, denominator = self.mlm(pred_emb, tgt)

        return loss_mlm, correct_mlm, denominator


class MambaTransformerTarget(MlmTarget):
    """
    BERT exploits masked language modeling (MLM)
    and mixed sentence prediction (MSP) for pretraining.
    """

    def __init__(self, args, vocab_size):
        super(MambaTransformerTarget, self).__init__(args, vocab_size)
        self.max_seq_length = args.max_seq_length

    def forward(self, memory_bank, tgt, byte_transmission_idx=None, *args, **kwargs):
        """
        Args:
            memory_bank: [batch_size x seq_length x hidden_size]
            tgt: tgt_mlm [batch_size x seq_length]
            seg: segment for input

        Returns:
            loss_mlm: Masked language model loss.
            correct_mlm: Number of words that are predicted correctly.
            denominator: Number of masked words.
        """
        # Masked language model (MLM).
        loss_mlm, correct_mlm, denominator = self.mlm(memory_bank, tgt)

        return loss_mlm, correct_mlm, denominator
