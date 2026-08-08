# -*- coding: utf-8 -*-
import os
import random


if __name__ == "__main__":
    default_cmd = "PYTHONPATH='.' python data_generation/run_pretrain_data_gen_direction.py"

    cmd1 = f"{default_cmd} --dataset Browser"
    os.system(cmd1)

    cmd1 = f"{default_cmd} --dataset CICIOT2023_Benign"
    os.system(cmd1)

    cmd1 = f"{default_cmd} --dataset CICIOT2023_Malware"
    os.system(cmd1)

    cmd1 = f"{default_cmd} --dataset CICMalAnal2017_Benign"
    os.system(cmd1)

    cmd1 = f"{default_cmd} --dataset CICMalAnal2017_Malware"
    os.system(cmd1)

    cmd1 = f"{default_cmd} --dataset ISCX_NonVPN_2016"
    os.system(cmd1)

    corpora_path = "corpora_direction_datasets"
    os.system(f"PYTHONPATH='.' python data_generation/pretrain_select_small_corpora.py "
              f"--corpora_path {corpora_path}")


    dst_dataset = "Browser_CICMalAnal2017_Benign_CICMalAnal2017_Malware_CICIOT2023_Benign_CICIOT2023_Malware_ISCX_NonVPN_2016"
    os.system(f"cp {corpora_path}/Browser/corpora_gram_Browser_all.txt "
              f"{corpora_path}/{dst_dataset}/corpora_gram_{dst_dataset}/")

    os.system(f"cp {corpora_path}/CICIOT2023_Benign/corpora_gram_CICIOT2023_Benign_all.txt "
              f"{corpora_path}/{dst_dataset}/corpora_gram_{dst_dataset}/")

    os.system(f"cp {corpora_path}/CICIOT2023_Malware/corpora_gram_CICIOT2023_Malware_all.txt "
              f"{corpora_path}/{dst_dataset}/corpora_gram_{dst_dataset}/")

    os.system(f"cp {corpora_path}/CICMalAnal2017_Benign/corpora_gram_CICMalAnal2017_Benign_all.txt "
              f"{corpora_path}/{dst_dataset}/corpora_gram_{dst_dataset}/")

    os.system(f"cp {corpora_path}/CICMalAnal2017_Malware/corpora_gram_CICMalAnal2017_Malware_all.txt "
              f"{corpora_path}/{dst_dataset}/corpora_gram_{dst_dataset}/")

    os.system(f"cp {corpora_path}/ISCX_NonVPN_2016/corpora_gram_ISCX_NonVPN_2016_all_small.txt "
              f"{corpora_path}/{dst_dataset}/corpora_gram_{dst_dataset}/")

    os.system(f"PYTHONPATH='.' python data_generation/pretrain_merged_dataset_genenration_direction.py "
              f"--corpora_path {corpora_path}")

    os.system(f"python data_generation/jamba_vocab_direction_generation.py --corpora_path {corpora_path}")

    os.system("PYTHONPATH='.' python pre_training/run_preprocess_direction.py")
