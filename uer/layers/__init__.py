from uer.layers.embeddings import WordEmbedding
from uer.layers.embeddings import WordPosEmbedding
from uer.layers.embeddings import WordPosSegEmbedding
from uer.layers.embeddings import WordSinusoidalposEmbedding
from uer.layers.embeddings import JambaEmbedding
from uer.layers.embeddings import JambaNoCoarseEmbedding
from uer.layers.embeddings import JambaNoFineEmbedding
from uer.layers.embeddings import JambaNoCoarseFineEmbedding
from uer.layers.embeddings import JambaDirectionLengthEmbedding


str2embedding = {"word": WordEmbedding, "word_pos": WordPosEmbedding, "word_pos_seg": WordPosSegEmbedding,
                 "word_sinusoidalpos": WordSinusoidalposEmbedding, "jamba": JambaEmbedding,
                 "jamba_direction_length": JambaDirectionLengthEmbedding,
                 "jamba_direction_length_no_coarse": JambaNoCoarseEmbedding,
                 "jamba_direction_length_no_fine": JambaNoFineEmbedding,
                 "jamba_direction_length_no_coarse_fine": JambaNoCoarseFineEmbedding}

__all__ = ["WordEmbedding", "WordPosEmbedding", "WordPosSegEmbedding",
           "WordSinusoidalposEmbedding", "JambaEmbedding", "JambaNoCoarseEmbedding",
           "JambaNoFineEmbedding", "JambaNoCoarseFineEmbedding", "JambaDirectionLengthEmbedding",
           "str2embedding"]
