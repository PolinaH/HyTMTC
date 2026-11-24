# 相比jamba_vocab_generation.py，在末尾加入了direction length的编码，从-1454至1454
# -*- coding: utf-8 -*-
from collections import defaultdict
import argparse


def config_args(parser):
    # Path options.
    parser.add_argument("--corpora_path", required=True, type=str)

    return parser


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser = config_args(parser)
    config_args = parser.parse_args()

    root = "/workspace/data1/hym/HyTMTC"
    corpora_folder = f"{root}/{config_args.corpora_path}"
    datasetname = "Browser_CICMalAnal2017_Benign_CICMalAnal2017_Malware_CICIOT2023_Benign_CICIOT2023_Malware_ISCX_NonVPN_2016"
    corpora_file_path = f"{corpora_folder}/{datasetname}/corpora_gram_{datasetname}_all.txt"
    vocab_path = f"{corpora_folder}/{datasetname}/sorted_corpora_gram_vocab_{datasetname}_length_direction.txt"

    counter = defaultdict(int)
    cnt = 0
    with open(corpora_file_path, 'r') as f:
        while True:
            line = f.readline()

            if not line:
                break

            packet = line.strip()
            if packet.startswith("||"):
                cnt += 1
                #print(cnt)
                packet = packet[2:]
                packet_tokens = packet.split()
                for token in packet_tokens:
                    if len(token) == 4:
                        counter[token] += 1
            elif packet == "":
                continue
            else:
                cnt += 1
                #print(cnt)
                packet_tokens = packet.split()
                for token in packet_tokens:
                    if len(token) == 4:
                        counter[token] += 1
    print(f"Finish counting")

    # 2. 确保包含全部65536个可能的token（0000-FFFF）和-1454至+1455的方向长度信息
    flow_tokens = [f"{i:04x}" for i in range(65536)]
    direction_length_tokens = []
    for item in range(-1500, 1501):  # 1454 1500
        if item < 0:
            direction_length_tokens.append(str(item))
        else:
            direction_length_tokens.append('+' + str(item))

    all_tokens = flow_tokens + direction_length_tokens
    missing_tokens = set(all_tokens) - set(counter.keys())
    for token in missing_tokens:
        counter[token] = 0

    # 3. 按词频降序排序，词频相同时按token升序
    sorted_tokens = sorted(
        counter.items(),
        key=lambda x: (-x[1], x[0])  # 先按词频降序，再按token升序
    )
    print(f"Finish sorting")

    # 4. 写入文件
    with open(vocab_path, 'w') as f:
        f.write("[PAD]\n")
        f.write("[SEP]\n")
        f.write("[CLS]\n")
        f.write("[UNK]\n")
        f.write("[MASK]\n")
        for token, count in sorted_tokens:
            f.write(f"{token}\n")  # 每行一个token

    print(f"成功输出 bigram token 和 方向长度 token 到 {vocab_path}")

    # python data_generation/jamba_vocab_direction_generation.py --corpora_path