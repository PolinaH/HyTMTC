# -*- coding: utf-8 -*-
# 用于根据独立数据集的corpora files等生成merged pretrain dataset
import logging
import argparse
import os
import datetime
from pprint import pformat
from pretrain_data_gen import pretrain_dataset_generation,corpora_to_bigram, corpora_to_gram, merge
from vocab_gen import build_tokenizer,build_vocab
from tqdm import tqdm
import multiprocessing as mp
import pudb


def logging_definition(args,root):
    mdate = datetime.date.today()
    log_file_folder = f"{root}/log/pretrain_corpus_generation"
    log_file_path = f"{log_file_folder}/{mdate}_{str(len(os.listdir(log_file_folder)))}.txt"
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
    p = argparse.ArgumentParser()
    p.add_argument('--verbose', action='store_true', help='whether to print the debugging logs.')
    p.add_argument("--corpora_path", required=True, type=str)

    return p.parse_args()


if __name__ == '__main__':
    root = '/workspace/data1/hym/TrafficFormer'

    args = args_definition()
    logging = logging_definition(args, root)
    logging.info('Running with configuration:\n' + pformat(vars(args)))

    dataset = 'Browser_CICMalAnal2017_Benign_CICMalAnal2017_Malware_CICIOT2023_Benign_CICIOT2023_Malware_ISCX_NonVPN_2016'

    corpora_datasets_path = f"{root}/{args.corpora_path}"
    corpora_folder = f"{corpora_datasets_path}/{dataset}"
    corpora_gram_folder = f"{corpora_folder}/corpora_gram_{dataset}"

    merge(corpora_gram_folder)


    # PYTHONPATH='.' python data_generation/pretrain_merged_dataset_genenration_direction.py --corpora_path
