# -*- coding: utf-8 -*-
import os
import datetime
import argparse


def config_run_pretrain_jamba_args(parser):
    # Path options.
    parser.add_argument("--pretrain_folder", required=True, type=str)
    parser.add_argument("--corpora_folder", required=True, type=str)
    parser.add_argument("--jamba_inner_parameter_sharing", default="True", type=str)
    parser.add_argument("--jamba_atm_ratio", default=13, type=int)
    parser.add_argument("--embedding", required=True, type=str)
    parser.add_argument("--encoder", required=True, type=str)
    parser.add_argument("--target", required=True, type=str)

    return parser


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser = config_run_pretrain_jamba_args(parser)
    config_args = parser.parse_args()

    # 10.112.173.89
    root = '/workspace/data1/hym/HyTMTC'

    file = f"{root}/pre_training/pretrain.py"
    dataset = 'Browser_CICMalAnal2017_Benign_CICMalAnal2017_Malware_CICIOT2023_Benign_CICIOT2023_Malware_ISCX_NonVPN_2016'  # 'medbiot_normal'

    #-------------------------------------------------------------------------
    # 全部样本
    pretrain_folder = config_args.pretrain_folder
    dataset_path = f"{root}/{pretrain_folder}/{dataset}/dataset_{dataset}.pt"

    # 320个样本: '320'
    # 640个样本: '640'
    # 6400个样本: '6400'
    #sample_cnt = '320'
    #dataset_path = f"{root}/pretrain_datasets/{dataset}/dataset_{dataset}_{sample_cnt}.pt"
    # -------------------------------------------------------------------------

    corpora_folder = config_args.corpora_folder
    # corpora gram vocab file
    vocab_path = f"{root}/{corpora_folder}/{dataset}/" \
                 f"sorted_corpora_gram_vocab_{dataset}_length_direction.txt"

    # 10.112.173.89
    CUDA_VISIBLE_DEVICES = '0,1'  # '0,1'
    world_size = 2  # 2
    gpu_ranks = '0 1'  # '0 1'
    learning_rate = 6e-4
    total_steps = 42000
    save_checkpoint_steps = 6000  # 10000
    report_steps = 20
    batch_size = 128  # 64

    scheduler = 'cosine'

    # packet prediction classes num
    class_num = 4  # 4分类

    jamba_atm_ratio = config_args.jamba_atm_ratio
    embedding = config_args.embedding  # 'jamba_direction_length'
    encoder = config_args.encoder  # 'transformer'
    mask = 'fully_visible'
    target = config_args.target  # 'jamba_mask_only_direction'

    mdate = datetime.date.today()
    log_file_folder = f"{root}/log/model_pretrain"
    log_file_idx = len(os.listdir(log_file_folder))

    output_model_folder = f"{root}/models/pre_trained_models/{mdate}_{log_file_idx}"
    if not os.path.exists(output_model_folder):
        os.makedirs(output_model_folder)
    output_model_path = \
        f"{output_model_folder}/{mdate}_{log_file_idx}_pre-trained_model_{dataset}.bin"


    cmd = f"python3 -u {file} " \
          f"--CUDA_VISIBLE_DEVICES {CUDA_VISIBLE_DEVICES} " \
          f"--root {root} " \
          f"--model_config_path {model_config_path} " \
          f"--dataset {dataset} " \
          f"--dataset_path {dataset_path} " \
          f"--vocab_path {vocab_path} " \
          f"--output_model_path {output_model_path} " \
          f"--world_size {world_size} " \
          f"--gpu_ranks {gpu_ranks} " \
          f"--learning_rate {learning_rate} " \
          f"--master_ip tcp://localhost:12345 " \
          f"--total_steps {total_steps} " \
          f"--save_checkpoint_steps {save_checkpoint_steps} " \
          f"--report_steps {report_steps} " \
          f"--batch_size {batch_size} " \
          f"--scheduler {scheduler} " \
          f"--class_num {class_num} " \
          f"--jamba_atm_ratio {jamba_atm_ratio} " \
          f"--embedding {embedding} " \
          f"--encoder {encoder} " \
          f"--mask {mask} " \
          f"--target {target} " \
          f"--jamba_inner_parameter_sharing {config_args.jamba_inner_parameter_sharing} " \
          f"| tee {root}/log/model_pretrain/{mdate}_{log_file_idx}.txt"
    os.system(cmd)

    # PYTHONPATH='.' python pre_training/run_pretrain.py --corpora_folder --pretrain_folder --embedding --encoder --target