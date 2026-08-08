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

    parser.add_argument("--warm_up_just_unfreeze_fc", default="False", type=str, help="Whether to unfreeze specific layer.")
    parser.add_argument("--jamba_inner_parameter_sharing", default='True', type=str)
    parser.add_argument("--jamba_atm_ratio", required=True, type=int)
    parser.add_argument("--embedding", required=True, type=str)
    parser.add_argument("--output_self_attn_heads_num", default=4, type=int)
    parser.add_argument("--pretrained_model_file", default='', type=str)

    return parser


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser = config_run_classifier_jamba_args(parser)
    config_args = parser.parse_args()

    file = "fine_tuning/run_classifier_jamba.py"
    CUDA_VISIBLE_DEVICES = '0,1'

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
    pretrain_dataset = 'Browser_CICMalAnal2017_Benign_CICMalAnal2017_Malware_CICIOT2023_Benign_CICIOT2023_Malware_small_ISCX_NonVPN_2016'

    # tokenizer
    # original vocab file
    #vocab_file = f"{pretrain_dataset}/corpora_gram_vocab_{pretrain_dataset}.txt"
    # new vocab file
    vocab_file = f"{pretrain_dataset}/sorted_corpora_gram_vocab_{pretrain_dataset}.txt"

    # !!!!!!!! 修改processed_datasets路径 !!!!!!!!
    # processed_datasets路径
    # processed_datasets_jamba_sorted_vocab_test_1:测试数据集，前5个包
    # processed_datasets_jamba_sorted_vocab_test:测试数据集，前10个包
    processed_datasets = f"{root}/processed_datasets_jamba_sorted_vocab_test_1"

    # training
    if config_args.dataset == "DataCon2020":
        train_file = "train_dataset.tsv"
        valid_file = "valid_dataset.tsv"
        test_file = "test_dataset.tsv"
    elif config_args.bin:
        train_file = "train_bin_dataset_500.tsv"
        valid_file = "valid_bin_dataset_500.tsv"
        test_file = "test_bin_dataset_500.tsv"
    else:
        train_file = "train_dataset_500.tsv"
        valid_file = "valid_dataset_500.tsv"
        test_file = "test_dataset_500.tsv"

    if config_args.only_val_test:
        train_path = None
        valid_path = f"{processed_datasets}/{config_args.dataset}/dataset/{valid_file}"
        test_path = f"{processed_datasets}/{config_args.dataset}/dataset/{test_file}"
    else:
        train_path = f"{processed_datasets}/{config_args.dataset}/dataset/{train_file}"
        valid_path = f"{processed_datasets}/{config_args.dataset}/dataset/{valid_file}"
        test_path = f"{processed_datasets}/{config_args.dataset}/dataset/{test_file}"

    pretrained_model_folder = f"{root}/models/pre_trained_models"

    if config_args.pretrained_model_file == '':
        # --jamba_atm_ratio 13
        pretrained_model_file = "2025-05-27_2/2025-05-27_2_pre-trained_model_Browser_CICMalAnal2017_Benign_CICMalAnal2017_Malware_CICIOT2023_Benign_CICIOT2023_Malware_small_ISCX_NonVPN_2016-3000.bin"
        # --jamba_atm_ratio 15
        #pretrained_model_file = "2025-05-27_3/2025-05-27_3_pre-trained_model_Browser_CICMalAnal2017_Benign_CICMalAnal2017_Malware_CICIOT2023_Benign_CICIOT2023_Malware_small_ISCX_NonVPN_2016-3000.bin"
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
    epochs_num = 10
    earlystop = 20
    report_steps = 1
    # batch_size是总batch size，若是多卡训练，每个卡的batch size=batch_size/卡数量
    batch_size = 128  # 128
    learning_rate = 2.5e-4

    embedding = config_args.embedding
    encoder = 'jamba'  # 'transformer'
    mask = 'fully_visible'

    tokenizer = 'jamba'
    seq_length = 176  # 352

    output_self_attn_heads_num = config_args.output_self_attn_heads_num
    output_self_attn_dropout = 0.1

    cmd = f"python {file} " \
          f"--CUDA_VISIBLE_DEVICES {CUDA_VISIBLE_DEVICES} " \
          f"--root {root} " \
          f"--dataset {config_args.dataset} " \
          f"--vocab_path corpora_datasets/{vocab_file} " \
          f"--train_path {train_path} " \
          f"--valid_path {valid_path} " \
          f"--test_path {test_path} " \
          f"--model_config_path {model_config_path} " \
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

    # PYTHONPATH='.' python fine_tuning/config_run_classifier_jamba.py --jamba_atm_ratio --embedding --dataset

    # 不使用预训练参数
    # PYTHONPATH='.' python fine_tuning/config_run_classifier_jamba.py --no_pretrain --jamba_atm_ratio --embedding --dataset

    # 2分类
    # PYTHONPATH='.' python fine_tuning/config_run_classifier_jamba.py --bin --jamba_atm_ratio --embedding --dataset

    # 2分类且不使用预训练参数
    # PYTHONPATH='.' python fine_tuning/config_run_classifier_jamba.py --no_pretrain --bin --jamba_atm_ratio --embedding --dataset

    # 只测试，不训练，示例：
    # PYTHONPATH='.' python fine_tuning/config_run_classifier_jamba.py --only_val_test --jamba_atm_ratio --embedding --dataset