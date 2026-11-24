# -*- coding: utf-8 -*-
import logging
import argparse
import os
import datetime
from pprint import pformat
from pretrain_data_gen_direction import pretrain_dataset_generation,corpora_to_bigram, corpora_to_gram, merge
from vocab_gen import build_tokenizer,build_vocab,build_vocab_no_training
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
    p.add_argument('--verbose', action='store_true',help='whether to print the debugging logs.')
    p.add_argument("--dataset", required=True, help="Dataset Choice")
    p.add_argument("--pcap_path", required=False, help="if .pcap doesn't exists, specify pcap path")
    p.add_argument("--max_process", required=False, default=46, type=int, help="Max process num. 0: not use")
    p.add_argument("--target",
                        choices=["bert", "bertflow", "lm", "mlm", "bilm", "albert", "seq2seq", "t5", "cls", "prefixlm",
                                 "jamba"], default="jamba",
                        help="The training dataset target.")

    # corpora file generation
    p.add_argument("--start_index", default=14, type=int, help="the index of start byte (i.e., If start from IP header, start_index = 14)")
    p.add_argument("--select_packet_len", default=64, type=int, help="the bytes of used")

    # enhance
    p.add_argument("--add_network_err", default="False", help="whether to add network err")

    # use corpora bigram or not
    p.add_argument("--corpora_bigram", action="store_true", help="use corpora bigram or not")

    return p.parse_args()


if __name__ == "__main__":
    root = '/workspace/data1/hym/HyTMTC'
    datasetInfo = {'USTC_TFC2016':{'pcap_exist':True,'is_dir_refer_class':True,'num_category':2},
                   'USTC_TFC2016_small':{'pcap_exist':True,'is_dir_refer_class':True,'num_category':2},
                   'ISCX_VPN_2016':{'pcap_exist':True,'is_dir_refer_class':True,'num_category':14},
                   'ISCX_VPN_2016_small':{'pcap_exist':True,'is_dir_refer_class':True,'num_category':2},
                   'Browser':{'pcap_exist':True,'is_dir_refer_class':False,'num_category':4},
                   'CICMalAnal2017_Benign':{'pcap_exist':False,'is_dir_refer_class':True,'num_category':1},
                   'medbiot_normal':{'pcap_exist':True,'is_dir_refer_class':True,'num_category':3},
                   'CICIOT2023_Benign':{'pcap_exist':False,'is_dir_refer_class':True,'num_category':1},
                   'CICIOT2023_Malware_small':{'pcap_exist':False,'is_dir_refer_class':True,'num_category':7},
                   'CICMalAnal2017_Malware':{'pcap_exist':False,'is_dir_refer_class':True,'num_category':4},
                   'ISCX_NonVPN_2016':{'pcap_exist':False,'is_dir_refer_class':True,'num_category':3},
                   'medbiot':{'pcap_exist':False,'is_dir_refer_class':True,'num_category':4},
                   'CICIOT2023_Malware_whole':{'pcap_exist':False,'is_dir_refer_class':True,'num_category':1},
                   'CICIOT2023_Malware':{'pcap_exist':False,'is_dir_refer_class':True,'num_category':1}}

    args = args_definition()
    logging = logging_definition(args, root)
    logging.info('Running with configuration:\n' + pformat(vars(args)))
    raw_dataset_path = f'{root}/raw_datasets/{args.dataset}'  # raw pcap/pcapng dataset path
    pcapng_path = f"{raw_dataset_path}/pcapng"
    pcap_path = f"{raw_dataset_path}/pcap"
    split_dataset_path = f'{root}/split_datasets/{args.dataset}'  # split datasets path
    if not os.path.exists(split_dataset_path):
        os.makedirs(split_dataset_path)

    corpora_folder = f"{root}/corpora_direction_datasets_1/{args.dataset}"
    # corpora file path: f"{corpora_folder}/corpora_{args.dataset}.txt"
    # corpora bigram path: f"{corpora_folder}/corpora_bigram_{args.dataset}.txt"
    # vocab path: f"{corpora_folder}/corpora_bigram_vocab_{args.dataset}.txt"
    corpora_file_folder = f"{corpora_folder}/corpora_{args.dataset}"
    corpora_bigram_folder = f"{corpora_folder}/corpora_bigram_{args.dataset}"
    corpora_gram_folder = f"{corpora_folder}/corpora_gram_{args.dataset}"

    # generate corpora file
    if os.path.exists(corpora_file_folder) and len(os.listdir(corpora_file_folder))!=0:
        logging.info(f"corpora files {corpora_file_folder}/* already exists, skip generating")
    else:
        pretrain_dataset_generation(logging, args.dataset, pcapng_path, pcap_path, split_dataset_path, corpora_file_folder,
                                    folder_is_pcap_label=datasetInfo[args.dataset]['is_dir_refer_class'],
                                    start_index=args.start_index,
                                    select_packet_len=args.select_packet_len,
                                    add_network_err=args.add_network_err,
                                    max_process=args.max_process)

        logging.info(f"corpora files is saved in {corpora_file_folder}")


    # generate corpora gram file
    if os.path.exists(corpora_gram_folder) and len(os.listdir(corpora_gram_folder))!=0:
        logging.info(f"{corpora_gram_folder}/* already exists")
        raise FileExistsError(f"corpora gram files {corpora_gram_folder}/* already exists, "
                              f"please check it before generating")
    elif not os.path.exists(corpora_gram_folder):
        os.makedirs(corpora_gram_folder)

    trained_tokenizer_path = f"{corpora_folder}/corpora_gram_trained_tokenizer.json"
    vocab_path = f"{corpora_folder}/corpora_gram_vocab_{args.dataset}.txt"
    # generate corpora gram file
    logging.info("Begin generating corpora gram file...")
    all_corpora_files = os.listdir(corpora_file_folder)
    if args.max_process > 0:
        pbar = tqdm(total=len(all_corpora_files))
        pbar.set_description('generate corpora gram files')
        update = lambda *args: pbar.update()
        pool = mp.Pool(args.max_process)
        for file in all_corpora_files:
            pool.apply_async(corpora_to_gram, (f"{corpora_file_folder}/{file}",
                                               f"{corpora_gram_folder}/{file}"),callback=update)
        pool.close()
        pool.join()
    else:  # args.max_process==0,不使用多进程
        for file in tqdm(all_corpora_files):
            corpora_to_gram(f"{corpora_file_folder}/{file}", f"{corpora_gram_folder}/{file}")

    logging.info(f"corpora gram files is saved in {corpora_gram_folder}")


    # merge corpora bigram/gram files
    if args.corpora_bigram:
        logging.info("start merge corpora bigram files...")
        merge(corpora_bigram_folder)
        logging.info(f"merged corpora bigram file is saved in {corpora_bigram_folder}_all.txt")
    else:
        logging.info("start merge corpora gram files...")
        merge(corpora_gram_folder)
        logging.info(f"merged corpora gram file is saved in {corpora_gram_folder}_all.txt")


    # generate trained tokenizer file
    if args.corpora_bigram:
        logging.info("Begin generating trained tokenizer file based on corpora bigram file...")
        build_tokenizer(f"{corpora_bigram_folder}_all.txt", trained_tokenizer_path)
        logging.info(f"trained tokenizer file is saved in {trained_tokenizer_path}")
    else:
        logging.info("Do not use corpora bigram file")
        logging.info("Begin generating trained tokenizer file based on corpora gram file...")
        build_tokenizer(f"{corpora_gram_folder}_all.txt", trained_tokenizer_path)
        logging.info(f"trained tokenizer file is saved in {trained_tokenizer_path}")


    # generate vocab file
    logging.info("Begin generating vocab file...")
    if args.target == "jamba":
        build_vocab_no_training(vocab_path)  # 不进行分词的vocab
    else:
        build_vocab(trained_tokenizer_path, vocab_path)
    logging.info(f"vocab file is saved in {vocab_path}")


    # PYTHONPATH='.' python data_generation/run_pretrain_data_gen_direction.py --dataset