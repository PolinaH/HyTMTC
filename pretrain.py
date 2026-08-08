# -*- coding: utf-8 -*-
import os


if __name__ == "__main__":
    os.system(f"PYTHONPATH='.' python pre_training/run_pretrain.py "
              f"--corpora_folder corpora_direction_datasets_1 "
              f"--pretrain_folder pretrain_datasets_sorted_vocab_direction_1 "
              f"--jamba_atm_ratio 13 "
              f"--embedding jamba_direction_length "
              f"--target jamba")
