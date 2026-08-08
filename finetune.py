# -*- coding: utf-8 -*-
import os

if __name__ == "__main__":
    default_cmd = f"PYTHONPATH='.' python fine_tuning/config_run_classifier_direction_jamba.py " \
                  f"--jamba_atm_ratio 13 --embedding jamba_direction_length " \
                  f"--train_num 400 "

    cmd1 = f"{default_cmd} --dataset ISCX_VPN_2016_service"
    os.system(cmd1)

    cmd1 = f"{default_cmd} --dataset ISCX_VPN_2016_app"
    os.system(cmd1)

    cmd1 = f"{default_cmd} --dataset USTC_TFC2016"
    os.system(cmd1)

    cmd1 = f"{default_cmd} --dataset medbiot"
    os.system(cmd1)
