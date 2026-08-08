import os
import sys
sys.path.append(os.getcwd())
import argparse
import torch
import uer.trainer as trainer
from uer.utils.config import load_hyperparam
from args_options.args_options import model_opts,optimization_opts
import warnings
warnings.filterwarnings("ignore")
import logging
from pprint import pformat
import datetime
import pudb


def logging_definition(args, root):
    mdate = datetime.date.today()
    log_file_folder = f"{root}/log/model_pretrain"
    log_file_path = f"{log_file_folder}/{mdate}_{args.encoder}_{len(os.listdir(log_file_folder))}.txt"
    if args.verbose == False:
        logging.basicConfig(filename=log_file_path,
                            filemode='w',
                            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                            level=logging.INFO)
    else:
        logging.basicConfig(filename=log_file_path,
                            filemode='w',
                            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                            level=logging.DEBUG)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    # 设置日志打印格式
    formatter = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)
    return logging


def args_definition():
    parser = argparse.ArgumentParser()

    # Path options.
    parser.add_argument("--root", default=None)
    parser.add_argument('--verbose', action='store_true', help='whether to print the debugging logs.')
    parser.add_argument("--dataset_path", type=str, default="dataset.pt",
                        help="Path of the preprocessed dataset.")
    parser.add_argument("--vocab_path", default=None, type=str,
                        help="Path of the vocabulary file.")
    parser.add_argument("--spm_model_path", default=None, type=str,
                        help="Path of the sentence piece model.")
    parser.add_argument("--tgt_vocab_path", default=None, type=str,
                        help="Path of the target vocabulary file.")
    parser.add_argument("--tgt_spm_model_path", default=None, type=str,
                        help="Path of the target sentence piece model.")
    parser.add_argument("--pretrained_model_path", type=str, default=None,
                        help="Path of the pretrained model.")
    parser.add_argument("--output_model_path", type=str, required=True,
                        help="Path of the output model.")
    parser.add_argument("--model_config_path", type=str, default="models/bert/base_config.json", #define the model
                        help="Config file of model hyper-parameters.")

    # Training and saving options.
    parser.add_argument("--CUDA_VISIBLE_DEVICES", required=True, help="CUDA_VISIBLE_DEVICES")
    parser.add_argument("--total_steps", type=int, default=100000, help="Total training steps.")
    parser.add_argument("--save_checkpoint_steps", type=int, default=10000,
                        help="Specific steps to save model checkpoint.")
    parser.add_argument("--report_steps", type=int, default=100, help="Specific steps to print prompt.")
    parser.add_argument("--accumulation_steps", type=int, default=1,
                        help="Specific steps to accumulate gradient.")
    parser.add_argument("--class_num", type=int, default=4, help="Number of classes")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Training batch size. The actual batch_size is [batch_size x world_size x accumulation_steps].")
    parser.add_argument("--instances_buffer_size", type=int, default=25600,
                        help="The buffer size of instances in memory.")
    parser.add_argument("--labels_num", type=int, required=False,
                        help="Number of prediction labels.")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout value.")
    parser.add_argument("--seed", type=int, default=7, help="Random seed.")

    # Preprocess options.
    parser.add_argument("--tokenizer", choices=["bert", "char", "space"], default="bert",
                        help="Specify the tokenizer."
                             "Original Google BERT uses bert tokenizer on Chinese corpus."
                             "Char tokenizer segments sentences into characters."
                             "Space tokenizer segments sentences into words according to space.")

    # Model options.
    model_opts(parser)
    parser.add_argument("--tgt_embedding", choices=["word", "word_pos", "word_pos_seg", "word_sinusoidalpos"], default="word_pos_seg",
                        help="Target embedding type.")
    parser.add_argument("--decoder", choices=["transformer"], default="transformer", help="Decoder type.")
    parser.add_argument("--pooling", choices=["mean", "max", "first", "last"], default="first",
                        help="Pooling type.")
    parser.add_argument("--target", choices=["bert","bertflow","lm", "mlm", "bilm", "albert", "seq2seq",
                                             "t5", "cls", "prefixlm", "jamba", "jamba_mask_only",
                                             "jamba_mask_only_direction", "transformer_mamba",
                                             "mamba_transformer"],
                        default="bert",
                        help="The training target of the pretraining model.")
    parser.add_argument("--tie_weights", action="store_true",
                        help="Tie the word embedding and softmax weights.")
    parser.add_argument("--has_lmtarget_bias", action="store_true",
                        help="Add bias on output_layer for lm target.")
    parser.add_argument("--jamba_atm_ratio", type=int,
                        help="The ratio of attention-to-Mamba layers a:m.")

    # Masking options.
    parser.add_argument("--whole_word_masking", action="store_true", help="Whole word masking.")
    parser.add_argument("--span_masking", action="store_true", help="Span masking.")
    parser.add_argument("--span_geo_prob", type=float, default=0.2,
                        help="Hyperparameter of geometric distribution for span masking.")
    parser.add_argument("--span_max_length", type=int, default=10,
                        help="Max length for span masking.")

    # Optimizer options.
    optimization_opts(parser)

    # GPU options.
    parser.add_argument("--world_size", type=int, default=1, help="Total number of processes (GPUs) for training.")
    parser.add_argument("--gpu_ranks", default=[], nargs='+', type=int, help="List of ranks of each process."
                        " Each process has a unique integer rank whose value is in the interval [0, world_size), and runs in a single GPU.")
    parser.add_argument("--master_ip", default="tcp://localhost:12345", type=str, help="IP-Port of master for training.")
    parser.add_argument("--backend", choices=["nccl", "gloo"], default="nccl", type=str, help="Distributed backend.")


    args = parser.parse_args()

    if args.target == "cls":
        assert args.labels_num is not None, "Cls target needs the denotation of the number of labels."

    # Load hyper-parameters from config file. 
    if args.model_config_path:
        load_hyperparam(args)

    return args


if __name__ == "__main__":
    root = '/workspace/data1/hym/TrafficFormer'
    datasetInfo = {'USTC_TFC2016':{'pcap_exist':True,'is_dir_refer_class':True,'num_category':2},
                   'USTC_TFC2016_small':{'pcap_exist':True,'is_dir_refer_class':True,'num_category':2},
                   'ISCX_VPN_2016': {'pcap_exist': True, 'is_dir_refer_class': True, 'num_category': 14},
                   'ISCX_VPN_2016_small': {'pcap_exist': True, 'is_dir_refer_class': True, 'num_category': 2}}

    args = args_definition()

    #logging = logging_definition(args, root)
    print('Running with configuration:\n' + pformat(vars(args)))

    # CUDA configuration
    os.environ['CUDA_VISIBLE_DEVICES'] = args.CUDA_VISIBLE_DEVICES

    ranks_num = len(args.gpu_ranks)
    if args.world_size > 1:
        # Multiprocessing distributed mode.
        assert torch.cuda.is_available(), "No available GPUs."
        assert ranks_num <= args.world_size, "Started processes exceed `world_size` upper limit."
        assert ranks_num <= torch.cuda.device_count(), "Started processes exceeds the available GPUs."
        args.dist_train = True
        args.ranks_num = ranks_num
        print("Using distributed mode for training.")
        print(f"Using GPUs: {args.CUDA_VISIBLE_DEVICES}.")
    elif args.world_size == 1 and ranks_num == 1:
        # Single GPU mode.
        assert torch.cuda.is_available(), "No available GPUs."
        args.gpu_id = args.gpu_ranks[0]
        print(f"args.gpu_id: {args.gpu_id}")
        print(f"torch.cuda.device_count: {torch.cuda.device_count()}")
        assert args.gpu_id < torch.cuda.device_count(), "Invalid specified GPU device."
        args.dist_train = False
        args.single_gpu = True
        print(f"Using GPU {args.gpu_id} for training.")
    else:
        # CPU mode.
        assert ranks_num == 0, "GPUs are specified, please check the arguments."
        args.dist_train = False
        args.single_gpu = False
        print("Using CPU mode for training.")

    trainer.train_and_validate(args)

    #  CUDA_VISIBLE_DEVICES=2,3,4 python3 pre_training/pretrain.py --dataset_path dataset.pt \
    #                      --vocab_path models/encryptd_vocab.txt \
    #                      --output_model_path model.bin \
    #                      --world_size 3 --gpu_ranks 0 1 2 --master_ip tcp://localhost:12345 \
    #                      --total_steps 90000 --save_checkpoint_steps 10000 --batch_size 64 \
    #                      --embedding word_pos_seg --encoder transformer --mask fully_visible --target bertflow
