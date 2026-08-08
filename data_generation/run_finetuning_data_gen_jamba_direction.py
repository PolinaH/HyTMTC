# -*- coding: utf-8 -*-
import argparse
import os
from uer.utils.tokenizers import JambaTokenizer
from finetuning_data_gen_jamba_direction import convert_splitcap,generation_multiP,dataset_extract,\
    small_dataset_extract,mini_dataset_extract
import logging
from pprint import pformat
import datetime
import pudb


def logging_definition(args,root):
    mdate = datetime.date.today()
    log_file_folder = f"{root}/log/finetuning_data_generation"
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
    return logging, log_file_path


def args_definition():
    p = argparse.ArgumentParser()
    p.add_argument('--verbose', action='store_true',help='whether to print the debugging logs.')
    p.add_argument("--dataset", required=True, help="Dataset Choice")
    p.add_argument("--pcap_path", required=False, help="if .pcap doesn't exists, specify pcap path")
    p.add_argument("--max_process", required=False, default=44, help="Max process num")

    p.add_argument("--start_index", default=14, type=int,
                   help="the index of start byte (i.e., If start from IP header, start_index = 14)")
    p.add_argument("--payload_packet_num", default=5, type=int,
                   help="Number of the used packets from each flow.")
    p.add_argument("--payload_length", default=64, type=int, help="Number of the used bytes of data.")

    p.add_argument("--tokenizer", choices=["bert", "char", "space", "jamba"], default="jamba",
                        help="Specify the tokenizer.")
    p.add_argument("--vocab_path", default=None, type=str, help="Path of the vocabulary file.")
    p.add_argument("--spm_model_path", default=None, type=str, help="Path of the sentence piece model.")
    p.add_argument("--tgt_vocab_path", default=None, type=str, help="Path of the target vocabulary file.")
    p.add_argument("--tgt_spm_model_path", default=None, type=str, help="Path of the target sentence piece model.")

    # dataset generation
    p.add_argument("--process_samples_num", default=15000, type=int, help="process sample num per class.")
    p.add_argument("--max_samples_num", default=10000, type=int, help="max sample num per class.")

    # small dataset generation
    p.add_argument("--small_dataset", action="store_true", help="Whether to generate small dataset.")
    p.add_argument("--mini_dataset", action="store_true", help="Whether to generate mini train dataset.")
    p.add_argument("--selected_sample_num", default=500, type=int,
                   help="Number of samples per class in small dataset.")

    # enhance
    p.add_argument("--enhance_setup", required=False, default=False, type=bool, help="whether to enhance file")
    p.add_argument("--enhance_file", required=False, default='train', choices=['train','valid','test'], help="choose enhance file")
    p.add_argument("--enhance_factor", required=False, default=1, type=int, help="enhance factor")

    return p.parse_args()


if __name__ == "__main__":
    # if the traffic is the pacp type, pcapng_path == pcap_path
    root = '/workspace/data1/hym/TrafficFormer'
    # 如果is_dir_refer_class设置为False，
    # 则在第一遍执行程序后(已经变成了一个目录代表1类)，需要将is_dir_refer_class的值改为True
    datasetInfo = {'USTC_TFC2016':{'pcap_exist':True,'is_dir_refer_class':True,'num_category':18},
                   'USTC_TFC2016_small':{'pcap_exist':True,'is_dir_refer_class':True,'num_category':2},
                   'ISCX_VPN_2016_service':{'pcap_exist':True,'is_dir_refer_class':True,'num_category':6},
                   'ISCX_VPN_2016_app':{'pcap_exist':True,'is_dir_refer_class':True,'num_category':14},
                   'CICIOT2023':{'pcap_exist':True,'is_dir_refer_class':True,'num_category':8},
                   'mqttiot':{'pcap_exist':True,'is_dir_refer_class':True,'num_category':5},
                   'toniot':{'pcap_exist':True,'is_dir_refer_class':True,'num_category':10},
                   'CICMalAnal2017_Malware':{'pcap_exist':True,'is_dir_refer_class':True,'num_category':4},
                   'botiot':{'pcap_exist':True,'is_dir_refer_class':True,'num_category':4},
                   'medbiot':{'pcap_exist':True,'is_dir_refer_class':True,'num_category':4},
                   'ISCX_Tor_2016_app':{'pcap_exist':True,'is_dir_refer_class':True,'num_category':8},
                   'ISCX_Tor_2016_service':{'pcap_exist':True,'is_dir_refer_class':True,'num_category':7},
                   'CICIoMT2024':{'pcap_exist':True,'is_dir_refer_class':True,'num_category':4},
                   'DoHBrw2020':{'pcap_exist':True,'is_dir_refer_class':True,'num_category':2},
                   'Edge_IIoT':{'pcap_exist':True,'is_dir_refer_class':True,'num_category':7},
                   'cstnet_tls': {'pcap_exist': True, 'is_dir_refer_class': True, 'num_category': 19},
                   'TII_SSRC_23': {'pcap_exist': True, 'is_dir_refer_class': True, 'num_category': 2},
                   'ACI_IoT_23': {'pcap_exist': True, 'is_dir_refer_class': True, 'num_category': 5},
                   'TII_SSRC_23_multi': {'pcap_exist': True, 'is_dir_refer_class': True, 'num_category': 5},
                   'NUDT_MobileTraffic': {'pcap_exist': True, 'is_dir_refer_class': True, 'num_category': 20},}

    args = args_definition()

    raw_dataset_path = f'{root}/raw_datasets/{args.dataset}'  # raw pcap/pcapng dataset path
    pcapng_path = f"{raw_dataset_path}/pcapng"
    pcap_path = f"{raw_dataset_path}/pcap"

    split_dataset_path = f'{root}/split_datasets/{args.dataset}'  # split datasets path
    if not os.path.exists(split_dataset_path):
        os.makedirs(split_dataset_path)

    # ！！！！！！！！注意修改！！！！！！！！
    # 同时修改feature_dataset_path和processed_dataset_path
    # pretrain_dataset是模型的预训练数据集
    # 顺序vocab文件前缀：corpora_gram_vocab
    # 出现次数降序vocab文件前缀：sorted_corpora_gram_vocab
    pretrain_dataset = 'Browser_CICMalAnal2017_Benign_CICMalAnal2017_Malware_CICIOT2023_Benign_CICIOT2023_Malware_ISCX_NonVPN_2016'
    # 按照token出现次数降序排列后打标签的vocab
    args.vocab_path = f"{root}/corpora_direction_datasets_1/{pretrain_dataset}/" \
                      f"sorted_corpora_gram_vocab_{pretrain_dataset}_length_direction.txt"

    feature_dataset_path = f'{root}/feature_datasets_jamba_sorted_vocab_direction_1/{args.dataset}'
    if not os.path.exists(feature_dataset_path):
        os.makedirs(feature_dataset_path)

    processed_dataset_path = f'{root}/processed_datasets_jamba_sorted_vocab_direction_1/{args.dataset}'  # processed datasets path
    if not os.path.exists(processed_dataset_path):
        os.makedirs(processed_dataset_path)
    ######################################################

    tsv_dataset_folder = f"{processed_dataset_path}/dataset"

    tokenizer = JambaTokenizer(args)

    logging, log_file_path = logging_definition(args, root)
    logging.info('Running with configuration:\n' + pformat(vars(args)))

    #######################################################
    if args.mini_dataset:  # 在训练集中继续分割小训练集，验证集和测试集不动
        if os.path.exists(f"{processed_dataset_path}/dataset/train_dataset_5.tsv"):
            logging.info(f"{processed_dataset_path}/dataset/mini tsv files already exist")
            raise FileExistsError(f"{processed_dataset_path}/dataset/mini tsv files already exist, "
                                  f"please check it before generating")
        if not os.path.exists(f"{processed_dataset_path}/dataset/train_dataset_{int(args.selected_sample_num*0.8)}.tsv"):
            logging.info(
                f"{processed_dataset_path}/dataset/*_{int(args.selected_sample_num*0.8)}.tsv files don't exist, please generate first")
            raise FileNotFoundError(
                f"{processed_dataset_path}/dataset/*_{int(args.selected_sample_num*0.8)}.tsv files don't exist, please generate first")
        logging.info("Generating small tsv dataset...")

        logging.info("Generating mini train tsv dataset...")

        num_category = datasetInfo[args.dataset]['num_category']

        train_tsv = f"train_dataset_{int(args.selected_sample_num*0.8)}.tsv"
        mini_dataset_extract(logging, f"{processed_dataset_path}/dataset", train_tsv,
                             num_category, args.selected_sample_num)

    elif args.small_dataset:
        if os.path.exists(f"{processed_dataset_path}/dataset/train_dataset_{int(args.selected_sample_num*0.8)}.tsv"):
            logging.info(f"{processed_dataset_path}/dataset/*_{int(args.selected_sample_num*0.8)}.tsv files already exists")
            raise FileExistsError(
                f"{processed_dataset_path}/dataset/*_{int(args.selected_sample_num*0.8)}.tsv files already exists, "
                f"please check it before generating")
        if not os.path.exists(f"{processed_dataset_path}/dataset/train_dataset.tsv"):
            logging.info(f"{processed_dataset_path}/dataset/*.tsv files doesn't exist, please generate first")
            raise FileNotFoundError(
                f"{processed_dataset_path}/dataset/*.tsv files doesn't exist, please generate first")
        logging.info("Generating small tsv dataset...")

        # max_samples is the maximum samples of each class
        logging.info(f"samples per class: {args.selected_sample_num}")
        # num_category is the number of class
        num_category = datasetInfo[args.dataset]['num_category']

        small_dataset_extract(logging, processed_dataset_path, num_category, args.selected_sample_num,
                              features=['datagram', "tokens_list", "coarse_semantic_list",
                                        "fine_semantic_list", "seg_list"])

    else:
        if os.path.exists(f"{processed_dataset_path}/dataset/train_dataset.tsv"):
            logging.info(f"{processed_dataset_path}/dataset/*.tsv files already exists")
            raise FileExistsError(f"{processed_dataset_path}/dataset/*.tsv files already exists, "
                                  f"please check it before generating")
        logging.info("Generating normal tsv dataset...")

        # max_samples is the maximum samples of each class
        logging.info(f"process samples per class: {args.process_samples_num}")
        logging.info(f"max samples per class: {args.max_samples_num}")
        # num_category is the number of class
        num_category = datasetInfo[args.dataset]['num_category']

        # begin generation
        logging.info("begin generation")
        convert_splitcap(logging, pcapng_path, pcap_path, split_dataset_path,
                         folder_is_pcap_label=datasetInfo[args.dataset]['is_dir_refer_class'])

        if os.path.exists(f"{processed_dataset_path}/dataset.json"):
            logging.info(f"{processed_dataset_path}/dataset.json already exists, skip feature data generation")
        else:
            generation_multiP(logging, tokenizer, f'{split_dataset_path}/splitcap', args.process_samples_num,
                              args.max_samples_num, processed_dataset_path, feature_dataset_path,
                              payload_packet_num=args.payload_packet_num, start_index=args.start_index,
                              payload_length=args.payload_length, max_process=args.max_process)

        dataset_extract(logging, processed_dataset_path, num_category,
                        features=['datagram', "tokens_list", "coarse_semantic_list", "fine_semantic_list",
                                  "seg_list"])

    logging.info(f"log file: {log_file_path}")

    # cd data_generation
    # 生成jamba数据集（按照每类最多10000个样本生成的数据集，命名train_dataset.tsv）
    # 单进程可按照此设置控制，多进程时每类生成的样本数量还是process_samples_num
    # PYTHONPATH='.' python data_generation/run_finetuning_data_gen_jamba_direction.py --max_samples_num 1000 --dataset ISCX_VPN_2016_app
    # PYTHONPATH='.' python data_generation/run_finetuning_data_gen_jamba_direction.py --process_samples_num 15000 --dataset ISCX_VPN_2016_app

    # 生成jamba小样本数据集（按照每类最多selected_sample_num个样本生成的数据集，命名train_dataset_{selected_sample_num}.tsv）
    # PYTHONPATH='.' python data_generation/run_finetuning_data_gen_jamba_direction.py --small_dataset --selected_sample_num 500 --dataset ISCX_VPN_2016_service

    # 生成jamba mini数据集（按照每类最多selected_sample_num个样本生成的数据集，命名train_dataset_{selected_sample_num}.tsv）
    # PYTHONPATH='.' python data_generation/run_finetuning_data_gen_jamba_direction.py --mini_dataset --selected_sample_num 500 --dataset ISCX_VPN_2016_service