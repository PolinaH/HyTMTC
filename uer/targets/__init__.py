from uer.targets.mlm_target import MlmTarget
from uer.targets.lm_target import LmTarget
from uer.targets.bert_target import BertTarget
from uer.targets.bertflow_target import BertFlowTarget
from uer.targets.cls_target import ClsTarget
from uer.targets.bilm_target import BilmTarget
from uer.targets.albert_target import AlbertTarget
from uer.targets.seq2seq_target import Seq2seqTarget
from uer.targets.t5_target import T5Target
from uer.targets.prefixlm_target import PrefixlmTarget
from uer.targets.jamba_target import JambaTarget, JambaMaskOnlyTarget
from uer.targets.transformer_mamba_target import TransformerMambaTarget, MambaTransformerTarget


str2target = {"bert": BertTarget, "bertflow": BertFlowTarget, "mlm": MlmTarget, "lm": LmTarget,
              "bilm": BilmTarget, "albert": AlbertTarget, "seq2seq": Seq2seqTarget,
              "t5": T5Target, "cls": ClsTarget, "prefixlm": PrefixlmTarget, "jamba": JambaTarget,
              "jamba_mask_only": JambaMaskOnlyTarget, "jamba_mask_only_direction": JambaMaskOnlyTarget,
              "transformer_mamba": TransformerMambaTarget, "mamba_transformer": MambaTransformerTarget}

__all__ = ["BertTarget", "BertFlowTarget", "MlmTarget", "LmTarget", "BilmTarget", "AlbertTarget",
           "Seq2seqTarget", "T5Target", "ClsTarget", "PrefixlmTarget", "JambaTarget", "JambaMaskOnlyTarget",
           "JambaMaskOnlyTarget", "TransformerMambaTarget", "MambaTransformerTarget", "str2target"]
