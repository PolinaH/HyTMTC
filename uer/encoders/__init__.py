from uer.encoders.transformer_encoder import TransformerEncoder
from uer.encoders.rnn_encoder import RnnEncoder
from uer.encoders.rnn_encoder import LstmEncoder
from uer.encoders.rnn_encoder import GruEncoder
from uer.encoders.rnn_encoder import BirnnEncoder
from uer.encoders.rnn_encoder import BilstmEncoder
from uer.encoders.rnn_encoder import BigruEncoder
from uer.encoders.cnn_encoder import GatedcnnEncoder
from uer.encoders.jamba_encoder import JambaEncoder, JambaEncoder_onlyMamba, JambaEncoder_onlyMambaMOE, \
    JambaEncoder_onlyTransformer, JambaEncoder_onlyTransformerMoE, JambaEncoder_onlyLSTM, \
    JambaEncoder_onlyLSTMMoE, JambaEncoder_LSTM_Transformer
from uer.encoders.transformer_mamba_fuse_encoder import TransformerMambaEncoder, MambaTransformerEncoder


str2encoder = {"transformer": TransformerEncoder, "rnn": RnnEncoder, "lstm": LstmEncoder,
               "gru": GruEncoder, "birnn": BirnnEncoder, "bilstm": BilstmEncoder, "bigru": BigruEncoder,
               "gatedcnn": GatedcnnEncoder, "jamba": JambaEncoder,
               "jamba_only_mamba": JambaEncoder_onlyMamba,
               "jamba_only_mambamoe": JambaEncoder_onlyMambaMOE,
               "jamba_only_transformer": JambaEncoder_onlyTransformer,
               "jamba_only_transformermoe": JambaEncoder_onlyTransformerMoE,
               "jamba_only_lstm": JambaEncoder_onlyLSTM,
               "jamba_only_lstmmoe": JambaEncoder_onlyLSTMMoE,
               "jamba_lstm_transformer": JambaEncoder_LSTM_Transformer,
               "transformer_mamba": TransformerMambaEncoder,
               "mamba_transformer": MambaTransformerEncoder}

__all__ = ["TransformerEncoder", "RnnEncoder", "LstmEncoder", "GruEncoder", "BirnnEncoder",
           "BilstmEncoder", "BigruEncoder", "GatedcnnEncoder", "JambaEncoder", "JambaEncoder_onlyMamba",
           "JambaEncoder_onlyMambaMOE", "JambaEncoder_onlyTransformer",
           "JambaEncoder_onlyTransformerMoE", "JambaEncoder_onlyLSTM", "JambaEncoder_onlyLSTMMoE",
           "TransformerMambaEncoder", "JambaEncoder_LSTM_Transformer",
           "MambaTransformerEncoder", "str2encoder"]

