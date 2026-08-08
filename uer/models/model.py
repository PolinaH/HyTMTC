import torch
import torch.nn as nn
import pudb


class Model(nn.Module):
    """
    Pretraining models consist of three parts:
        - embedding
        - encoder
        - target
    """
    def __init__(self, args, embedding, encoder, target):
        super(Model, self).__init__()
        self.embedding = embedding
        self.encoder = encoder
        self.target = target
        
        if args.target in ['bert','bertflow','mlm', 'jamba'] and args.tie_weights:
            self.target.mlm_linear_2.weight = self.embedding.word_embedding.weight
        elif args.target in ['lm','t5'] and args.tie_weights:
            self.target.output_layer.weight = self.embedding.word_embedding.weight

        if args.target == 't5' and args.share_embedding:
            self.target.embedding.word_embedding.weight = self.embedding.word_embedding.weight
        
        self.is_moe = args.is_moe

    def forward(self, src, seg, coarse, fine, tgt, sample_len, byte_transmission_idx):
        emb = self.embedding(src, seg, coarse, fine, sample_len)
        output, _ = self.encoder(emb, seg, byte_transmission_idx)
        loss_info = self.target(output, tgt, byte_transmission_idx)
        return loss_info
