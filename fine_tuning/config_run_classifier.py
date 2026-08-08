# -*- coding: utf-8 -*-
import os


if __name__ == "__main__":
    file = "fine_tuning/run_classifier.py"
    CUDA_VISIBLE_DEVICES = '0,1'
    root = '/workspace/data1/hym/TrafficFormer'
    # !!!!!!!! model_config_path和encoder需要同步修改 !!!!!!!!
    # model_config_path                      embedding       encoder        target
    # 'models/bert/base_config.json'         'word_pos_seg'  'transformer'  'bertflow'
    # 'models/jamba/jamba_zeta_config.json'  'jamba'         'jamba_zeta'    'jamba'
    # 'models/jamba/jamba_config.json'       'jamba'         'jamba'         'jamba'
    model_config_path = 'models/bert/base_config.json'
    dataset = 'USTC_TFC2016'

    # tokenizer
    vocab_file = "encryptd_vocab.txt"

    # training
    train_file = "train_dataset.tsv"
    valid_file = "valid_dataset.tsv"
    test_file = "test_dataset.tsv"

    pretrained_model_file = "nomoe_bertflow_pre-trained_model.bin"
    exist_output_model_num = len(os.listdir(f'{root}/models/fine_tuning_models'))
    output_model_file = f"finetuned_model_{str(exist_output_model_num)}.bin"
    epochs_num = 4
    earlystop = 4
    batch_size = 64  # 128
    learning_rate = 6e-5

    embedding = 'word_pos_seg'  # 'word_pos_seg'
    encoder = 'transformer'  # 'transformer'
    mask = 'fully_visible'
    seq_length = 320

    cmd = f"python {file} " \
          f"--CUDA_VISIBLE_DEVICES {CUDA_VISIBLE_DEVICES} " \
          f"--root {root} " \
          f"--dataset {dataset} " \
          f"--vocab_path models/{vocab_file} " \
          f"--train_path {root}/processed_datasets/{dataset}/dataset/{train_file} " \
          f"--valid_path {root}/processed_datasets/{dataset}/dataset/{valid_file} " \
          f"--test_path {root}/processed_datasets/{dataset}/dataset/{test_file} " \
          f"--model_config_path {model_config_path} " \
          f"--pretrained_model_path {root}/models/pre_trained_models/{pretrained_model_file} " \
          f"--output_model_path {root}/models/fine_tuning_models/{output_model_file} " \
          f"--epochs_num {epochs_num} " \
          f"--earlystop {earlystop} " \
          f"--batch_size {batch_size} " \
          f"--embedding {embedding} " \
          f"--encoder {encoder} " \
          f"--mask {mask} " \
          f"--seq_length {seq_length} " \
          f"--learning_rate {learning_rate}"
    os.system(cmd)

    # PYTHONPATH='.' python fine_tuning/config_run_classifier.py