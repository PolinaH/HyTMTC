#!/usr/bin/python3
#-*- coding:utf-8 -*-
import argparse
import os
import sys
import six
from packaging import version
from uer.utils.data import *
from uer.utils import *
import logging
from pprint import pformat
import datetime
import pudb

sys.path.append(os.getcwd())
assert version.parse(six.__version__) >= version.parse("1.12.0")

def logging_definition(args,root):
    mdate = datetime.date.today()
    log_file_folder = f"{root}/log/model_pretrain_data_generation"
    log_file_path = f"{log_file_folder}/{mdate}_{len(os.listdir(log_file_folder))}.txt"
    if args.verbose == False:
        logging.basicConfig(filename=log_file_path,
                            filemode='w',
                            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                            level=logging.INFO)
    else:
        logging.basicConfig(filename=log_file_path,
                            filemode='w',
                            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                            level=logging.DEBUG)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    # 设置日志打印格式
    formatter = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)
    return logging


def args_definition():
    parser = argparse.ArgumentParser()

    # Path options.
    parser.add_argument("--root", default=None)
    parser.add_argument('--verbose', action='store_true',help='whether to print the debugging logs.')
    parser.add_argument("--dataset", required=True, help="Dataset Choice")
    parser.add_argument("--corpus_path", type=str, required=True,
                        help="Path of the corpus bigram/gram files for pretraining.")
    parser.add_argument("--vocab_path", default=None, type=str,
                        help="Path of the vocabulary file.")
    parser.add_argument("--spm_model_path", default=None, type=str,
                        help="Path of the sentence piece model.")
    parser.add_argument("--tgt_vocab_path", default=None, type=str,
                        help="Path of the target vocabulary file.")
    parser.add_argument("--tgt_spm_model_path", default=None, type=str,
                        help="Path of the target sentence piece model.")
    parser.add_argument("--datasets_folder", type=str, default="dataset",
                        help="Folder path of the pretrain datasets.")
    parser.add_argument("--splited_dataset_pt_folder", type=str, default="dataset/splited_dataset_pt",
                        help="Folder path of the splited dataset.pt files.")
    parser.add_argument("--dataset_path", type=str, default="dataset.pt",
                        help="Path of the preprocessed dataset.")

    # corpora file generation
    parser.add_argument("--start_index", default=14, type=int, help="the index of start byte (i.e., If start from IP header, start_index = 14)")
    parser.add_argument("--select_packet_len", default=64, type=int, help="the bytes of used")

    # Preprocess options.
    parser.add_argument("--pkt_select_num", type=int, default=5, help="packets selected for each pretrain sample")
    parser.add_argument("--pkt_interval", type=int, default=15, help="packets interval for pretrain sample generation")
    parser.add_argument("--tokenizer", choices=["bert", "char", "space", "jamba"], default="bert",
                        help="Specify the tokenizer." 
                             "Original Google BERT uses bert tokenizer on Chinese corpus."
                             "Char tokenizer segments sentences into characters."
                             "Space tokenizer segments sentences into words according to space.")
    parser.add_argument("--tgt_tokenizer", choices=["bert", "char", "space"], default="bert",
                        help="Specify the tokenizer.")
    parser.add_argument("--max_process", type=int, default=0,
                        help="Split the whole dataset into `max_process` parts, "
                             "and each part is fed to a single process in training step. "
                             "0: no use")
    parser.add_argument("--target", choices=["bert", "bertflow","lm", "mlm", "bilm", "albert", "seq2seq",
                                             "t5", "cls", "prefixlm", "jamba", "jamba_mask_only", "jamba_mask_only_direction"],
                        default="bert",
                        help="The training dataset target.")
    parser.add_argument("--flows_size", type=int, default=100000,
                        help="The buffer size of flows in memory, specific to targets that require negative sampling.")
    parser.add_argument("--seq_length", type=int, default=512, help="Sequence length of instances.")
    parser.add_argument("--tgt_seq_length", type=int, default=512, help="Target sequence length of instances.")
    parser.add_argument("--dup_factor", type=int, default=5,
                        help="Duplicate instances multiple times.")
    parser.add_argument("--short_seq_prob", type=float, default=0.1,
                        help="Probability of truncating sequence."
                             "The larger value, the higher probability of using short (truncated) sequence.")
    parser.add_argument("--full_sentences", action="store_true", help="Full sentences.")
    parser.add_argument("--seed", type=int, default=7, help="Random seed.")

    # Masking options.
    parser.add_argument("--dynamic_masking", action="store_true", help="Dynamic masking.")
    parser.add_argument("--whole_word_masking", action="store_true", help="Whole word masking.")
    parser.add_argument("--span_masking", action="store_true", help="Span masking.")
    parser.add_argument("--span_geo_prob", type=float, default=0.2,
                        help="Hyperparameter of geometric distribution for span masking.")
    parser.add_argument("--span_max_length", type=int, default=10,
                        help="Max length for span masking.")

    return parser.parse_args()


if __name__ == "__main__":
    root = '/workspace/data1/hym/HyTMTC'
    args = args_definition()

    if os.path.exists(args.dataset_path):
        logging.info(f"{args.dataset_path} already exists")
        raise FileExistsError(f"{args.dataset_path} already exists, please check it before generating")
    if not os.path.exists(args.splited_dataset_pt_folder):
        os.makedirs(args.splited_dataset_pt_folder)
    else:
        os.system(f"rm {args.splited_dataset_pt_folder}/*")
    # Dynamic masking.
    if args.dynamic_masking:
        args.dup_factor = 1

    logging = logging_definition(args, root)
    logging.info('Running with configuration:\n' + pformat(vars(args)))


    # Build tokenizer.
    tokenizer = str2tokenizer[args.tokenizer](args)
    if args.target == "seq2seq":
        args.tgt_tokenizer = str2tokenizer[args.tgt_tokenizer](args, False)

    args.logging = logging

    # Build and save dataset.
    #dataset = str2dataset[args.target](args, tokenizer.vocab, tokenizer)
    #pudb.set_trace()

    if args.target == 'jamba':
        jamba_build_and_save(args)
    elif args.target == 'jamba_mask_only':
        jamba_build_and_save_mask_only(args)
    elif args.target == 'jamba_mask_only_direction':
        jamba_build_and_save_mask_only_direction(args)
    else:
        pass
        #dataset.build_and_save(args.max_process,split_by_flow=True)
