# -*- coding: utf-8 -*-
import os


if __name__ == "__main__":
    default_cmd = f"PYTHONPATH='.' python data_generation/run_finetuning_data_gen_jamba_direction.py " \
                  f"--process_samples_num 15000"

    cmd1 = f"{default_cmd} --dataset ISCX_VPN_2016_service"
    os.system(cmd1)

    cmd1 = f"{default_cmd} --dataset ISCX_VPN_2016_app"
    os.system(cmd1)

    cmd1 = f"{default_cmd} --dataset USTC_TFC2016"
    os.system(cmd1)

    cmd1 = f"{default_cmd} --dataset medbiot"
    os.system(cmd1)


    default_cmd = f"PYTHONPATH='.' python data_generation/run_finetuning_data_gen_jamba_direction.py " \
                  f"--small_dataset --selected_sample_num 500"

    cmd1 = f"{default_cmd} --dataset ISCX_VPN_2016_service"
    os.system(cmd1)

    cmd1 = f"{default_cmd} --dataset ISCX_VPN_2016_app"
    os.system(cmd1)

    cmd1 = f"{default_cmd} --dataset USTC_TFC2016"
    os.system(cmd1)

    cmd1 = f"{default_cmd} --dataset medbiot"
    os.system(cmd1)


    default_cmd = f"PYTHONPATH='.' python data_generation/run_finetuning_data_gen_jamba_direction.py " \
                  f"--mini_dataset --selected_sample_num 500"

    cmd1 = f"{default_cmd} --dataset ISCX_VPN_2016_service"
    os.system(cmd1)

    cmd1 = f"{default_cmd} --dataset ISCX_VPN_2016_app"
    os.system(cmd1)

    cmd1 = f"{default_cmd} --dataset USTC_TFC2016"
    os.system(cmd1)

    cmd1 = f"{default_cmd} --dataset medbiot"
    os.system(cmd1)

