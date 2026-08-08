# -*- coding: utf-8 -*-
import os
import datetime
import argparse


def config_run_classifier_jamba_args(parser):
    # Path options.
    parser.add_argument("--dataset", required=True, type=str)
    parser.add_argument("--bin", action='store_true', help="Whether to use binary dataset.")
    parser.add_argument("--no_pretrain", action='store_true', help="Whether to use pretrain model.")
    parser.add_argument("--only_val_test", action='store_true', help="Whether to train the model.")
    parser.add_argument("--no_use_self_attn", action='store_true',
                        help="Whether to NO USE self attention layer.")

    parser.add_argument("--warm_up_just_unfreeze_fc", default="False", type=str, help="Whether to unfreeze specific layer.")
    parser.add_argument("--jamba_inner_parameter_sharing", default='True', type=str)
    parser.add_argument("--jamba_atm_ratio", default=13, type=int)
    parser.add_argument("--embedding", required=True, type=str)
    parser.add_argument("--output_self_attn_heads_num", default=8, type=int)
    parser.add_argument("--pretrained_model_file", default='', type=str)
    parser.add_argument("--part", type=str)
    parser.add_argument("--train_num", default=400, type=int)
    parser.add_argument("--epoch", default=-1, type=int)

    return parser


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser = config_run_classifier_jamba_args(parser)
    config_args = parser.parse_args()

    file = "fine_tuning/run_classifier_direction_jamba.py"
    CUDA_VISIBLE_DEVICES = '0'  # '0'

    # 10.112.173.89
    root = '/workspace/data1/hym/TrafficFormer'
    # 10.112.143.54
    #root = '/workspace/hym/TrafficFormer'

    # !!!!!!!! model_config_path和encoder需要同步修改 !!!!!!!!
    # model_config_path                      embedding       encoder        target
    # 'models/bert/base_config.json'         'word_pos_seg'  'transformer'  'bertflow'
    # 'models/jamba/jamba_zeta_config.json'  'jamba'         'jamba_zeta'    'jamba'
    # 'models/jamba/jamba_config.json'       'jamba'         'jamba'         'jamba'
    model_config_path = 'models/jamba/jamba_config.json'

    # !!!!!!!! 修改pretrain_dataset !!!!!!!!
    corpora_path = "corpora_direction_datasets_1"
    pretrain_dataset = 'Browser_CICMalAnal2017_Benign_CICMalAnal2017_Malware_CICIOT2023_Benign_CICIOT2023_Malware_ISCX_NonVPN_2016'

    vocab_file = f"{root}/{corpora_path}/{pretrain_dataset}/" \
                 f"sorted_corpora_gram_vocab_{pretrain_dataset}_length_direction.txt"

    # !!!!!!!! 修改processed_datasets路径 !!!!!!!!
    # processed_datasets路径
    # processed_datasets_jamba_sorted_vocab_test_2:测试数据集，前5个包+方向长度信息
    # processed_datasets_jamba_sorted_vocab_direction_1:前5个包+方向长度信息
    processed_datasets = f"{root}/processed_datasets_jamba_sorted_vocab_direction_1"

    # training
    if config_args.dataset == "DataCon2021":
        train_file = "train_dataset.tsv"
        valid_file = "valid_dataset.tsv"
        test_file = "test_dataset.tsv"
    elif config_args.dataset == "DataCon2020":
        if config_args.train_num == 3000:
            train_file = f"train_dataset.tsv"
        else:
            train_file = f"train_dataset_{config_args.train_num}.tsv"
        valid_file = "valid_dataset.tsv"
        test_file = "test_dataset.tsv"
    elif config_args.bin:
        train_file = "train_bin_dataset_500.tsv"
        valid_file = "valid_bin_dataset_50.tsv"
        test_file = "test_bin_dataset_50.tsv"
    else:
        train_file = f"train_dataset_{config_args.train_num}.tsv"
        valid_file = "valid_dataset_50.tsv"
        test_file = "test_dataset_50.tsv"

    if config_args.only_val_test:
        if config_args.dataset == "DataCon2021":
            train_path = None
            valid_path = f"{processed_datasets}/{config_args.dataset}/" \
                         f"{config_args.part}/dataset/{valid_file}"
            test_path = f"{processed_datasets}/{config_args.dataset}/" \
                        f"{config_args.part}/dataset/{test_file}"
        else:
            train_path = None
            valid_path = f"{processed_datasets}/{config_args.dataset}/dataset/{valid_file}"
            test_path = f"{processed_datasets}/{config_args.dataset}/dataset/{test_file}"
    else:
        if config_args.dataset == "DataCon2021":
            train_path = f"{processed_datasets}/{config_args.dataset}/" \
                         f"{config_args.part}/dataset/{train_file}"
            valid_path = f"{processed_datasets}/{config_args.dataset}/" \
                         f"{config_args.part}/dataset/{valid_file}"
            test_path = f"{processed_datasets}/{config_args.dataset}/" \
                        f"{config_args.part}/dataset/{test_file}"
        else:
            train_path = f"{processed_datasets}/{config_args.dataset}/dataset/{train_file}"
            valid_path = f"{processed_datasets}/{config_args.dataset}/dataset/{valid_file}"
            test_path = f"{processed_datasets}/{config_args.dataset}/dataset/{test_file}"

    pretrained_model_folder = f"{root}/models/pre_trained_models"
    if config_args.no_use_self_attn:
        no_use_self_attn = 'True'
    else:
        no_use_self_attn = 'False'

    if config_args.pretrained_model_file == '':
        # --jamba_atm_ratio 13 python慢速版
        #pretrained_model_file = "2025-06-12_22/2025-06-12_22_pre-trained_model_Browser_CICMalAnal2017_Benign_CICMalAnal2017_Malware_CICIOT2023_Benign_CICIOT2023_Malware_ISCX_NonVPN_2016-12000.bin"
        # 官方mamba快速版
        #pretrained_model_file = "2025-12-01_7/2025-12-01_7_pre-trained_model_Browser_CICMalAnal2017_Benign_CICMalAnal2017_Malware_CICIOT2023_Benign_CICIOT2023_Malware_ISCX_NonVPN_2016-42000.bin"
        # 251210安装好causal-conv1d后版本
        pretrained_model_file = '2025-12-11_14/2025-12-11_14_pre-trained_model_Browser_CICMalAnal2017_Benign_CICMalAnal2017_Malware_CICIOT2023_Benign_CICIOT2023_Malware_ISCX_NonVPN_2016-42000.bin'
    else:
        pretrained_model_file = config_args.pretrained_model_file

    if config_args.no_pretrain:
        pretrained_model_path = None
    else:
        pretrained_model_path = f"{pretrained_model_folder}/{pretrained_model_file}"

    finetuned_model_folder = f"{root}/models/fine_tuning_models"

    finetuned_model_path = f"{pretrained_model_file.split('/')[0]}/{pretrained_model_file.split('-')[-1][:-4]}"

    if not os.path.exists(f"{finetuned_model_folder}/{finetuned_model_path}"):
        os.makedirs(f"{finetuned_model_folder}/{finetuned_model_path}")
    exist_output_model_num = len(os.listdir(f"{finetuned_model_folder}/{finetuned_model_path}"))

    output_model_file = f"finetuned_model_{exist_output_model_num}.bin"
    if config_args.epoch != -1:
        epochs_num = config_args.epoch
    else:
        epochs_num = 50  # 50
    earlystop = 20
    report_steps = 1
    # batch_size是总batch size，若是多卡训练，每个卡的batch size=batch_size/卡数量
    batch_size = 128  # 128
    learning_rate = 2.5e-4  # 2.5e-4
    '''
    if config_args.train_num >= 50:
        batch_size = 128  # 128
        learning_rate = 2.5e-4
    elif config_args.train_num == 20 or config_args.train_num == 30:
        batch_size = 64  # 128
        learning_rate = 1.2e-4
    elif config_args.train_num == 10:
        batch_size = 32  # 128
        learning_rate = 0.5e-4
    elif config_args.train_num == 5:
        batch_size = 16  # 128
        learning_rate = 0.25e-4
    elif config_args.train_num == 1:
        batch_size = 4  # 128
        learning_rate = 0.06e-4
    else:
        batch_size = 128  # 128
        learning_rate = 2.5e-4
    '''

    embedding = config_args.embedding  # 'jamba_direction_length' 'jamba'
    encoder = 'jamba'  # 'transformer'
    mask = 'fully_visible'

    tokenizer = 'jamba'
    seq_length = 192  # 176 352

    output_self_attn_heads_num = config_args.output_self_attn_heads_num
    output_self_attn_dropout = 0.1

    cmd = f"python {file} " \
          f"--CUDA_VISIBLE_DEVICES {CUDA_VISIBLE_DEVICES} " \
          f"--root {root} " \
          f"--dataset {config_args.dataset} " \
          f"--vocab_path {vocab_file } " \
          f"--train_path {train_path} " \
          f"--valid_path {valid_path} " \
          f"--test_path {test_path} " \
          f"--model_config_path {model_config_path} " \
          f"--no_use_self_attn {no_use_self_attn} " \
          f"--warm_up_just_unfreeze_fc {config_args.warm_up_just_unfreeze_fc} " \
          f"--jamba_inner_parameter_sharing {config_args.jamba_inner_parameter_sharing} " \
          f"--jamba_atm_ratio {config_args.jamba_atm_ratio} " \
          f"--pretrained_model_path {pretrained_model_path} " \
          f"--output_model_path {finetuned_model_folder}/{finetuned_model_path}/{output_model_file} " \
          f"--epochs_num {epochs_num} " \
          f"--earlystop {earlystop} " \
          f"--batch_size {batch_size} " \
          f"--report_steps {report_steps} " \
          f"--embedding {embedding} " \
          f"--encoder {encoder} " \
          f"--mask {mask} " \
          f"--tokenizer {tokenizer} " \
          f"--seq_length {seq_length} " \
          f"--output_self_attn_heads_num {output_self_attn_heads_num} " \
          f"--output_self_attn_dropout {output_self_attn_dropout} " \
          f"--learning_rate {learning_rate}"
    os.system(cmd)

    # 如果需要args指定pretrained_model_file，使用--pretrained_model_file
    # 如果warm up阶段（epoch前百分之20）只解冻fc层，使用--warm_up_just_unfreeze_fc True
    # 指定训练集样本数量--train_num，选择DataCon2020时，全部训练集数量是3000（每类）
    # 指定DataCon2021选择part1或者part2 --part

    # PYTHONPATH='.' python fine_tuning/config_run_classifier_direction_jamba.py --jamba_atm_ratio --embedding --dataset --train_num

    # 不使用预训练参数
    # PYTHONPATH='.' python fine_tuning/config_run_classifier_direction_jamba.py --no_pretrain --jamba_atm_ratio --embedding --dataset --train_num