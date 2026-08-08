# 文件原始位置：uer/opts.py


def default_opts(parser):
    # logger
    parser.add_argument('--verbose', action='store_true',help='whether to print the debugging logs.')
    parser.add_argument('--logging', required=False, help='logger')
    # 全局设置
    parser.add_argument("--root", required=True, help="root path")
    parser.add_argument("--dataset", required=True, help="dataset name")

    return parser


def model_opts(parser):
    parser.add_argument("--embedding", choices=["word", "word_pos", "word_pos_seg", "word_sinusoidalpos",
                                                "jamba", "jamba_direction_length_no_coarse", "jamba_direction_length_no_fine",
                                                "jamba_direction_length_no_coarse_fine", "jamba_direction_length"], default="word_pos_seg",
                        help="Emebdding type.")
    parser.add_argument("--max_seq_length", type=int, default=512,
                        help="Max sequence length for word embedding.")
    parser.add_argument("--relative_position_embedding", action="store_true",
                        help="Use relative position embedding.")
    parser.add_argument("--relative_attention_buckets_num", type=int, default=32,
                        help="Buckets num of relative position embedding.")
    parser.add_argument("--remove_embedding_layernorm", action="store_true",
                        help="Remove layernorm on embedding.")
    parser.add_argument("--remove_attention_scale", action="store_true",
                        help="Remove attention scale.")
    parser.add_argument("--encoder", choices=["transformer", "rnn", "lstm", "gru",
                                              "birnn", "bilstm", "bigru",
                                              "gatedcnn", "jamba_zeta", "jamba", "jamba_only_mamba",
                                              "jamba_only_mambamoe", "jamba_only_transformer",
                                              "jamba_only_transformermoe", "jamba_only_lstm",
                                              "jamba_only_lstmmoe", "jamba_lstm_transformer",
                                              "transformer_mamba", "mamba_transformer"],
                        default="transformer", help="Encoder type.")
    parser.add_argument("--mask", choices=["fully_visible", "causal", "causal_with_prefix"], default="fully_visible",
                        help="Mask type.")

    parser.add_argument("--output_self_attn_heads_num", default=8, type=int)
    parser.add_argument("--output_self_attn_dropout", default=0.1, type=float)

    parser.add_argument("--layernorm_positioning", choices=["pre", "post"], default="post",
                        help="Layernorm positioning.")
    parser.add_argument("--feed_forward", choices=["dense", "gated"], default="dense",
                        help="Feed forward type, specific to transformer model.")
    parser.add_argument("--remove_transformer_bias", action="store_true",
                        help="Remove bias on transformer layers.")
    parser.add_argument("--layernorm", choices=["normal", "t5"], default="normal",
                        help="Layernorm type.")
    parser.add_argument("--bidirectional", action="store_true", help="Specific to recurrent model.")
    parser.add_argument("--factorized_embedding_parameterization", action="store_true", help="Factorized embedding parameterization.")
    parser.add_argument("--warm_up_just_unfreeze_fc", default="False", type=str, help="Whether to unfreeze specific layer.")
    parser.add_argument("--parameter_sharing", action="store_true", help="Parameter sharing for every block")
    parser.add_argument("--jamba_inner_parameter_sharing", default="True", type=str, help="Parameter sharing for layers within jamba block")
    parser.add_argument("--soft_targets", action='store_true', help="Train model with logits.")
    parser.add_argument("--soft_alpha", type=float, default=0.5, help="Weight of the soft targets loss.")
    # MOE Model Options
    parser.add_argument("--is_moe", action="store_true", help="adopt moe layer.")
    parser.add_argument("--vocab_size", type=int, required=False, help="Number of vocab.")
    parser.add_argument("--moebert_route_method",
                        choices=["gate-token", "gate-sentence", "hash-random", "hash-balance", "proto"],
                        default="gate-token",help="moebert route method.")
    parser.add_argument("--moebert_route_hash_list", default=None, type=str, help="Path of moebert hash list file.")
    parser.add_argument("--moebert_load_balance", type=float, default=0.0, help="gate loss weight.")

    return parser


def optimization_opts(parser):
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate.")
    parser.add_argument("--warmup", type=float, default=0.1, help="Warm up value.")
    parser.add_argument("--fp16", action='store_true',
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit.")
    parser.add_argument("--fp16_opt_level", choices=["O0", "O1", "O2", "O3" ], default='O1',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument("--optimizer", choices=["adamw", "adafactor"], default="adamw",
                        help="Optimizer type.")
    parser.add_argument("--scheduler", choices=["linear", "cosine", "cosine_with_restarts", "polynomial",
                                                "constant", "constant_with_warmup"],
                        default="linear", help="Scheduler type.")

    return parser


def training_opts(parser):
    parser.add_argument("--CUDA_VISIBLE_DEVICES", required=True, help="CUDA_VISIBLE_DEVICES")
    parser.add_argument("--device", help="GPU device")
    parser.add_argument("--labels_num", help="labels num")
    parser.add_argument("--tokenizer", choices=["bert", "char", "space", "jamba"], default="bert",
                        help="Specify the tokenizer."
                             "Original Google BERT uses bert tokenizer on Chinese corpus."
                             "Char tokenizer segments sentences into characters."
                             "Space tokenizer segments sentences into words according to space.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size.")
    parser.add_argument("--seq_length", type=int, default=128, help="Sequence length.")
    parser.add_argument("--pooling", choices=['mean', 'max', 'first', 'last'], default="first", help="Pooling type.")
    parser.add_argument("--dropout", type=float, default=0.5, help="Dropout.")
    parser.add_argument("--epochs_num", type=int, default=3, help="Number of epochs.")
    parser.add_argument("--report_steps", type=int, default=100, help="Specific steps to print prompt.")
    parser.add_argument("--seed", type=int, default=7, help="Random seed.")
    parser.add_argument("--earlystop", required=True, type=int, default=5, help="early stop rounds")

    return parser


def finetune_opts(parser):
    # Path options.
    parser.add_argument("--pretrained_model_path", default=None, type=str,
                        help="Path of the pretrained model.")
    parser.add_argument("--output_model_path", default="models/finetuned_model.bin", type=str,
                        help="Path of the output model.")
    parser.add_argument("--vocab_path", default=None, type=str, help="Path of the vocabulary file.")
    parser.add_argument("--spm_model_path", default=None, type=str,
                        help="Path of the sentence piece model.")
    parser.add_argument("--train_path", type=str, help="Path of the trainset.")
    parser.add_argument("--valid_path", type=str, help="Path of the validset.")
    parser.add_argument("--test_path", default=None, type=str, help="Path of the testset.")
    parser.add_argument("--model_config_path", default="models/bert/base_config.json", type=str,
                        help="Path of the config file.")

    # default options
    parser = default_opts(parser)

    # Model options
    parser = model_opts(parser)

    # Optimization options
    parser = optimization_opts(parser)

    # Training options
    parser = training_opts(parser)

    return parser


def infer_opts(parser):
    # Path options.
    parser.add_argument("--load_model_path", default=None, type=str, help="Path of the input model.")
    parser.add_argument("--vocab_path", default=None, type=str, help="Path of the vocabulary file.")
    parser.add_argument("--spm_model_path", default=None, type=str, help="Path of the sentence piece model.")
    parser.add_argument("--test_path", type=str, required=True, help="Path of the testset.")
    parser.add_argument("--prediction_path", type=str, required=True, help="Path of the prediction file.")
    parser.add_argument("--model_config_path", default="models/bert/base_config.json", type=str,
                        help="Path of the config file.")
    # Inference options
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size.")
    parser.add_argument("--seq_length", type=int, default=128, help="Sequence length.")

    # default options
    parser = default_opts(parser)

    # Model options
    parser = model_opts(parser)

    return parser