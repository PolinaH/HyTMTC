from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers, processors
import json
import os
import pudb


def build_tokenizer(corpora_path, trained_tokenizer_path):
    # generate source dictionary,0-65535 
    num_count = 65536
    not_change_string_count = 5
    i = 0
    source_dictionary = {}  # 提供一个初始的词汇表
    tuple_sep = ()
    tuple_cls = ()

    #pudb.set_trace()
    while i < num_count:
        temp_string = '{:04x}'.format(i) 
        source_dictionary[temp_string] = i
        i += 1
    # Initialize a tokenizer
    # max_input_chars_per_word: 设置输入分词时每个词允许的最大字符数量
    tokenizer = Tokenizer(models.WordPiece(vocab=source_dictionary,unk_token="[UNK]",max_input_chars_per_word=4))

    # Customize pre-tokenization and decoding
    tokenizer.pre_tokenizer = pre_tokenizers.BertPreTokenizer()
    tokenizer.decoder = decoders.WordPiece()
    tokenizer.post_processor = processors.BertProcessing(sep=("[SEP]",1),cls=('[CLS]',2))

    # And then train
    trainer = trainers.WordPieceTrainer(vocab_size=65536, min_frequency=2)
    tokenizer.train([corpora_path, corpora_path], trainer=trainer)

    # And Save it
    tokenizer.save(trained_tokenizer_path, pretty=True)
    return 0

def build_vocab(trained_tokenizer_path, vocab_path):
    json_file = open(trained_tokenizer_path,'r')
    json_content = json_file.read()
    json_file.close()
    vocab_json = json.loads(json_content)
    # 'PAD':0,'SEP':1,'CLS':2,'UNK':3,'MASK':4
    vocab_txt = ["[PAD]","[SEP]","[CLS]","[UNK]","[MASK]"]
    for item in vocab_json['model']['vocab']:
        vocab_txt.append(item) # append key of vocab_json
    with open(vocab_path,'w') as f:
        for word in vocab_txt:
            f.write(word+"\n")
    return 0


def build_vocab_no_training(vocab_path):
    num_count = 65536
    i = 0
    # 'PAD':0,'SEP':1,'CLS':2,'UNK':3,'MASK':4
    vocab_txt = ["[PAD]","[SEP]","[CLS]","[UNK]","[MASK]"]
    while i < num_count:
        temp_string = '{:04x}'.format(i)
        vocab_txt.append(temp_string)
        i += 1
    with open(vocab_path,'w') as f:
        for word in vocab_txt:
            f.write(word+"\n")


if __name__ == "__main__":
    root = '/workspace/data1/hym/HyTMTC'
    corpora_bigram_path = f"{root}/data_generation/test_corpora_bigram.txt"
    trained_tokenizer_path = f"{root}/data_generation/test_corpora_bigram_trained_tokenizer.json"
    vocab_path = f"{root}/data_generation/test_corpora_bigram_vocab.txt"

    #build_tokenizer(corpora_bigram_path, trained_tokenizer_path)

    build_vocab(trained_tokenizer_path, vocab_path)