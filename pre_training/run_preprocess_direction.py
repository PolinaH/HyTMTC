# -*- coding: utf-8 -*-
import os


if __name__ == "__main__":
    file = "pre_training/preprocess_direction.py"
    root = '/workspace/data1/hym/TrafficFormer'

    # ！！！！！！！注意修改参数和路径！！！！！！！
    dataset = 'Browser_CICMalAnal2017_Benign_CICMalAnal2017_Malware_CICIOT2023_Benign_CICIOT2023_Malware_ISCX_NonVPN_2016'

    corpora_folder = f"{root}/corpora_direction_datasets_1/{dataset}"
    #corpora_bigram_folder = f"{corpora_folder}/corpora_bigram_{dataset}"
    #vocab_bigram_path = f"{corpora_folder}/corpora_gram_vocab_{dataset}.txt"
    corpora_gram_folder = f"{corpora_folder}/corpora_gram_{dataset}"
    vocab_gram_path = f"{corpora_folder}/sorted_corpora_gram_vocab_{dataset}_length_direction.txt"
    datasets_folder = f"{root}/pretrain_datasets_sorted_vocab_direction_1/{dataset}"
    splited_dataset_pt_folder = f"{datasets_folder}/splited_dataset_pt_{dataset}"
    dataset_path = f"{datasets_folder}/dataset_{dataset}.pt"

    # 注意和run_pretrain_data_gen.py中args_definition()函数中的select_packet_len值保持一致
    select_packet_len = 64  # the bytes of used
    pkt_select_num = 5  # 每条流选择前pkt_select_num个数据包
    seq_length = 192  # 176
    max_process = 44  # 0：不使用多进程

    tokenizer = 'jamba'
    target = 'jamba_mask_only_direction'  # 'jamba'

    cmd = f"python3 {file} " \
          f"--root {root} " \
          f"--dataset {dataset} " \
          f"--corpus_path {corpora_gram_folder} " \
          f"--vocab_path {vocab_gram_path} " \
          f"--dataset_path {dataset_path} " \
          f"--datasets_folder {datasets_folder} " \
          f"--splited_dataset_pt_folder {splited_dataset_pt_folder} " \
          f"--pkt_select_num {pkt_select_num} " \
          f"--select_packet_len {select_packet_len} " \
          f"--seq_length {seq_length} " \
          f"--max_process {max_process} " \
          f"--tokenizer {tokenizer} " \
          f"--target {target}"
    os.system(cmd)

    # PYTHONPATH='.' python pre_training/run_preprocess_direction.py