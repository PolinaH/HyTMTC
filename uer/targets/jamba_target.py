import torch
import torch.nn as nn
from uer.targets import *
import pudb


class JambaTarget(MlmTarget):
    """
    BERT exploits masked language modeling (MLM)
    and mixed sentence prediction (MSP) for pretraining.
    """

    def __init__(self, args, vocab_size):
        super(JambaTarget, self).__init__(args, vocab_size)
        # PP.
        self.pp_linear_1 = nn.Linear(args.hidden_size, args.hidden_size)
        self.pp_linear_2 = nn.Linear(args.hidden_size, args.class_num)
        weights = torch.tensor([1, 1, 1, 1]).float()
        self.pp_entroy = nn.CrossEntropyLoss(weight=weights)

    def forward(self, memory_bank, tgt, byte_transmission_idx=None, gpu_id=None):
        """
        Args:
            memory_bank: [batch_size x seq_length x hidden_size]
            tgt: tuple with tgt_mlm [batch_size x seq_length] and tgt_nsp [batch_size]

        Returns:
            loss_mlm: Masked language model loss.
            loss_msp: Mixed sentence prediction loss.
            correct_mlm: Number of words that are predicted correctly.
            correct_msp: Number of sentences that are predicted correctly.
            denominator: Number of masked words.
        """
        # Masked language model (MLM).
        assert type(tgt) == tuple
        tgt_mlm, tgt_pp = tgt[0], tgt[1]
        loss_mlm, correct_mlm, denominator = self.mlm(memory_bank, tgt_mlm)

        # packet prediction (PP).
        output_pp = torch.tanh(self.pp_linear_1(memory_bank[:, 0, :]))
        output_pp = self.pp_linear_2(output_pp)
        loss_pp = self.pp_entroy(output_pp, tgt_pp)
        #loss_pp = self.criterion(self.softmax(output_pp), tgt_pp)
        correct_pp = self.softmax(output_pp).argmax(dim=-1).eq(tgt_pp).sum()

        return loss_mlm, loss_pp, correct_mlm, correct_pp, denominator


class JambaMaskOnlyTarget(MlmTarget):
    """
    BERT exploits masked language modeling (MLM)
    and mixed sentence prediction (MSP) for pretraining.
    """

    def __init__(self, args, vocab_size):
        super(JambaMaskOnlyTarget, self).__init__(args, vocab_size)

    def forward(self, memory_bank, tgt, byte_transmission_idx=None, gpu_id=None):
        """
        Args:
            memory_bank: [batch_size x seq_length x hidden_size]
            tgt: tgt_mlm [batch_size x seq_length]

        Returns:
            loss_mlm: Masked language model loss.
            correct_mlm: Number of words that are predicted correctly.
            denominator: Number of masked words.
        """
        # Masked language model (MLM).
        tgt_mlm = tgt
        loss_mlm, correct_mlm, denominator = self.mlm(memory_bank, tgt_mlm)

        return loss_mlm, correct_mlm, denominator
