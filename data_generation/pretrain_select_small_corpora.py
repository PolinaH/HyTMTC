# -*- coding: utf-8 -*-
import os
import random
import argparse


def config_args(parser):
    # Path options.
    parser.add_argument("--corpora_path", required=True, type=str)

    return parser


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser = config_args(parser)
    config_args = parser.parse_args()

    root = '/workspace/data1/hym/HyTMTC'
    dataset = 'ISCX_NonVPN_2016'
    cnt_max = 100000
    select_interval =3  # 每select_interval条流中选取1条流

    corpora_folder = f"{root}/{config_args.corpora_path}/{dataset}"
    src_file = f'{corpora_folder}/corpora_gram_{dataset}_all.txt'
    dst_file = f'{corpora_folder}/corpora_gram_{dataset}_all_small.txt'

    cnt = 0
    with open(dst_file, 'w') as fw:
        with open(src_file, 'r') as fr:
            while cnt < cnt_max * select_interval:
                line = fr.readline()
                if not line:
                    break
                if line[:2] == "||":
                    cnt += 1
                if cnt % select_interval == 0:
                    fw.write(line)

    # PYTHONPATH='.' python data_generation/pretrain_select_small_corpora.py --corpora_path