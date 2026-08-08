import os
import random
import pickle
import torch
from multiprocessing import Pool
from uer.utils import *
from typing import List
from uer.utils.constants import *
from uer.utils.tokenizers import *
from uer.utils.misc import count_lines
from uer.utils.seed import set_seed
from tqdm import tqdm
import multiprocessing as mp
import copy
import pudb


def mask_seq_jamba(src, tokenizer, whole_word_masking, span_masking, span_geo_prob, span_max_length):
    vocab = tokenizer.vocab

    src_no_pad = src
    tokens_index, src_no_pad = create_index(src_no_pad, tokenizer, whole_word_masking, span_masking, span_geo_prob, span_max_length)
    if len(src_no_pad) < len(src):
        src = src_no_pad + (len(src) - len(src_no_pad)) * [PAD_ID]
    else:
        src = src_no_pad

    random.shuffle(tokens_index)
    num_to_predict = max(1, int(round(len(src_no_pad) * 0.15)))
    tgt_mlm = []
    for index_set in tokens_index:
        if len(tgt_mlm) >= num_to_predict:
            break
        if whole_word_masking:
            i = index_set[0]
            mask_len = index_set[1]
            if len(tgt_mlm) + mask_len > num_to_predict:
                continue

            for j in range(mask_len):
                token = src[i + j]
                tgt_mlm.append((i + j, token))
                prob = random.random()
                if prob < 0.8:
                    src[i + j] = vocab.get(MASK_TOKEN)
                elif prob < 0.9:
                    while True:
                        rdi = random.randint(1, len(vocab) - 1)
                        if rdi not in [vocab.get(CLS_TOKEN), vocab.get(SEP_TOKEN), vocab.get(MASK_TOKEN), PAD_ID]:
                            break
                    src[i + j] = rdi
        elif span_masking:
            i = index_set[0]
            span_len = index_set[1]
            if len(tgt_mlm) + span_len > num_to_predict:
                continue

            for j in range(span_len):
                token = src[i + j]
                tgt_mlm.append((i + j, token))
            prob = random.random()
            if prob < 0.8:
                for j in range(span_len):
                    src[i + j] = vocab.get(MASK_TOKEN)
            elif prob < 0.9:
                for j in range(span_len):
                    while True:
                        rdi = random.randint(1, len(vocab) - 1)
                        if rdi not in [vocab.get(CLS_TOKEN), vocab.get(SEP_TOKEN), vocab.get(MASK_TOKEN), PAD_ID]:
                            break
                    src[i + j] = rdi
        else:
            i = index_set[0]
            token = src[i]
            tgt_mlm.append((i, token))
            prob = random.random()
            if prob < 0.8:
                src[i] = vocab.get(MASK_TOKEN)
            elif prob < 0.9:
                while True:
                    rdi = random.randint(1, len(vocab) - 1)
                    if rdi not in [vocab.get(CLS_TOKEN), vocab.get(SEP_TOKEN), vocab.get(MASK_TOKEN), PAD_ID]:
                        break
                src[i] = rdi
    tgt_mlm = sorted(tgt_mlm, key=lambda x: x[0])
    return src, tgt_mlm


def mask_seq(src, tokenizer, whole_word_masking, span_masking, span_geo_prob, span_max_length):
    vocab = tokenizer.vocab

    for i in range(len(src) - 1, -1, -1):
        if src[i] != PAD_ID:
            break
    src_no_pad = src[:i + 1]
    tokens_index, src_no_pad = create_index(src_no_pad, tokenizer, whole_word_masking, span_masking, span_geo_prob, span_max_length)
    if len(src_no_pad) < len(src):
        src = src_no_pad + (len(src) - len(src_no_pad)) * [PAD_ID]
    else:
        src = src_no_pad

    random.shuffle(tokens_index)
    num_to_predict = max(1, int(round(len(src_no_pad) * 0.15))) 
    tgt_mlm = []
    for index_set in tokens_index:
        if len(tgt_mlm) >= num_to_predict:
            break
        if whole_word_masking:
            i = index_set[0]
            mask_len = index_set[1]
            if len(tgt_mlm) + mask_len > num_to_predict:
                continue

            for j in range(mask_len):
                token = src[i + j]
                tgt_mlm.append((i + j, token))
                prob = random.random()
                if prob < 0.8:
                    src[i + j] = vocab.get(MASK_TOKEN)
                elif prob < 0.9:
                    while True:
                        rdi = random.randint(1, len(vocab) - 1)
                        if rdi not in [vocab.get(CLS_TOKEN), vocab.get(SEP_TOKEN), vocab.get(MASK_TOKEN), PAD_ID]:
                            break
                    src[i + j] = rdi
        elif span_masking:
            i = index_set[0]
            span_len = index_set[1]
            if len(tgt_mlm) + span_len > num_to_predict:
                continue

            for j in range(span_len):
                token = src[i + j]
                tgt_mlm.append((i + j, token))
            prob = random.random()
            if prob < 0.8:
                for j in range(span_len):
                    src[i + j] = vocab.get(MASK_TOKEN)
            elif prob < 0.9:
                for j in range(span_len):
                    while True:
                        rdi = random.randint(1, len(vocab) - 1)
                        if rdi not in [vocab.get(CLS_TOKEN), vocab.get(SEP_TOKEN), vocab.get(MASK_TOKEN), PAD_ID]:
                            break
                    src[i + j] = rdi
        else:
            i = index_set[0]
            token = src[i]
            tgt_mlm.append((i, token))
            prob = random.random()
            if prob < 0.8: 
                src[i] = vocab.get(MASK_TOKEN)
            elif prob < 0.9: 
                while True:
                    rdi = random.randint(1, len(vocab) - 1)
                    if rdi not in [vocab.get(CLS_TOKEN), vocab.get(SEP_TOKEN), vocab.get(MASK_TOKEN), PAD_ID]:
                        break
                src[i] = rdi
    tgt_mlm = sorted(tgt_mlm, key=lambda x: x[0])
    return src, tgt_mlm


def create_index(src, tokenizer, whole_word_masking, span_masking, span_geo_prob, span_max_length):
    tokens_index = []
    span_end_position = -1
    vocab = tokenizer.vocab
    if whole_word_masking: 
        src_wwm = []
        src_length = len(src)
        has_cls, has_sep = False, False
        if src[0] == vocab.get(CLS_TOKEN):
            src = src[1:]
            has_cls = True
        if src[-1] == vocab.get(SEP_TOKEN):
            src = src[:-1]
            has_sep = True
        sentence = "".join(tokenizer.convert_ids_to_tokens(src)).replace('[UNK]', '').replace('##', '')
        import jieba
        wordlist = jieba.cut(sentence)
        if has_cls:
            src_wwm += [vocab.get(CLS_TOKEN)]
        for word in wordlist:
            position = len(src_wwm)
            src_wwm += tokenizer.convert_tokens_to_ids(tokenizer.tokenize(word))
            if len(src_wwm) < src_length:
                tokens_index.append([position, len(src_wwm)-position])
        if has_sep:
            src_wwm += [vocab.get(SEP_TOKEN)]
        if len(src_wwm) > src_length:
            src = src_wwm[:src_length]
        else:
            src = src_wwm
    else:
        for (i, token) in enumerate(src):
            if token == vocab.get(CLS_TOKEN) or token == vocab.get(SEP_TOKEN) or token == PAD_ID:
                continue
            if not span_masking:
                tokens_index.append([i])
            else:
                if i < span_end_position:
                    continue
                span_len = get_span_len(span_max_length, span_geo_prob)
                span_end_position = i + span_len
                if span_end_position > len(src):
                    span_len = len(src) - i
                tokens_index.append([i, span_len])
    return tokens_index, src


def get_span_len(max_span_len, p):
    geo_prob_cum = [0.0]
    geo_prob = 1.0
    for i in range(max_span_len + 1):
        if i == 0:
            continue
        if i == 1:
            geo_prob *= p
            geo_prob_cum.append(geo_prob_cum[-1] + geo_prob)
        else:
            geo_prob *= (1 - p)
            geo_prob_cum.append(geo_prob_cum[-1] + geo_prob)

    prob = geo_prob_cum[-1] * random.random()
    for i in range(len(geo_prob_cum) - 1):
        if prob >= geo_prob_cum[i] and prob < geo_prob_cum[i + 1]:
            current_span_len = i + 1
    return current_span_len


def merge_dataset(args, dataset_path, workers_num):
    # Merge datasets.
    dataset_writer = open(dataset_path, "ab")
    for file in os.listdir(args.splited_dataset_pt_folder):
        tmp_dataset_reader = open(f"{args.splited_dataset_pt_folder}/{file}", "rb")
        while True:
            tmp_data = tmp_dataset_reader.read(2**20)
            if tmp_data:
                dataset_writer.write(tmp_data)
            else:
                break
        tmp_dataset_reader.close()
        #os.remove(f"{args.root}/pretrain_datasets/{args.dataset}/tmp/dataset-tmp-" + str(i) + ".pt")
    dataset_writer.close()


def truncate_seq_pair(tokens_a, tokens_b, max_num_tokens):
    """ truncate sequence pair to specific length """
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_num_tokens:
            break
        trunc_tokens = tokens_a if len(tokens_a) > len(tokens_b) else tokens_b
        if random.random() < 0.5: 
            del trunc_tokens[0]
        else:
            trunc_tokens.pop()

def record_flow_start(corpus_path):
    starts = []
    with open(corpus_path, mode="r", encoding="utf-8") as f:
        i = 0
        while True:
            line = f.readline()
            if not line:
                break
            if line[:2] == "||":
                starts.append(i)
            i+=1
    starts.append(i)
    return starts

class Dataset(object):
    def __init__(self, args, vocab, tokenizer):
        self.args = args
        self.vocab = vocab
        self.tokenizer = tokenizer
        self.corpus_path = args.corpus_path
        self.dataset_path = args.dataset_path
        self.seq_length = args.seq_length
        self.seed = args.seed
        self.dynamic_masking = args.dynamic_masking
        self.whole_word_masking = args.whole_word_masking
        self.span_masking = args.span_masking
        self.span_geo_prob = args.span_geo_prob
        self.span_max_length = args.span_max_length
        self.flows_size = args.flows_size
        self.dup_factor = args.dup_factor

    def build_and_save(self, workers_num, split_by_flow=False):
        """
        Build dataset from the given corpus.
        Start workers_num processes and each process deals with a part of data.
        """
        lines_num = count_lines(self.corpus_path)
        print(f"Starting {workers_num} workers for building datasets ... ")
        assert (workers_num >= 1)
        if workers_num == 1:
            self.worker(0, 0, lines_num)
        else:
            pool = Pool(workers_num)
            if split_by_flow:
                starts = record_flow_start(self.corpus_path)
            current_index = 0
            for i in range(workers_num):
                if split_by_flow:
                    # start = starts[current_index]
                    # for j in range(len(starts))[current_index:]:
                    #     if starts[j]-starts[current_index]>perburst:
                    #         current_index = j
                    #         break
                    # if i==workers_num-1:
                    #     current_index = len(starts)-1
                    # end = starts[current_index]
                    start = starts[i*(len(starts)-1)//workers_num]
                    end = starts[(i+1)*(len(starts)-1)//workers_num]
                else:
                    start = i * lines_num // workers_num
                    end = (i + 1) * lines_num // workers_num
                pool.apply_async(func=self.worker, args=[i, start, end])
            pool.close()
            pool.join()

        # Merge datasets.
        merge_dataset(self.args, self.dataset_path, workers_num)
        
    def worker(self, proc_id, start, end):
        raise NotImplementedError()


class DataLoader(object):
    def __init__(self, args, dataset_path, batch_size, proc_id, proc_num, shuffle=False):
        self.args = args
        self.tokenizer = args.tokenizer
        self.batch_size = batch_size
        self.instances_buffer_size = args.instances_buffer_size
        self.proc_id = proc_id
        self.proc_num = proc_num
        self.shuffle = shuffle
        self.dataset_reader = open(dataset_path, "rb")
        self.read_count = 0
        self.start = 0
        self.end = 0
        self.buffer = []
        self.vocab = args.vocab
        self.whole_word_masking = args.whole_word_masking
        self.span_masking = args.span_masking
        self.span_geo_prob = args.span_geo_prob
        self.span_max_length = args.span_max_length

    def _fill_buf(self):
        try:
            self.buffer = []
            while True:
                instance = pickle.load(self.dataset_reader)  # 连续加载多个对象
                self.read_count += 1
                if (self.read_count - 1) % self.proc_num == self.proc_id: 
                    self.buffer.append(instance)
                    if len(self.buffer) >= self.instances_buffer_size:
                        break
        except EOFError:
            # Reach file end.
            self.dataset_reader.seek(0)

        if self.shuffle:
            random.shuffle(self.buffer)
        self.start = 0
        self.end = len(self.buffer)

    def _empty(self):
        return self.start >= self.end

    def __del__(self):
        self.dataset_reader.close()


def jamba_worker_wrapper(args):
    obj, corpus_file, max_process = args
    return obj.worker(corpus_file, max_process)


def jamba_build_and_save(args, split_by_flow=False):
    """
    Build dataset from the given corpus.
    Start max_process processes and each process deals with a part of data.
    max_process:
        0 no use multiprocessing
        >=1 use multiprocessing
    split_by_flow: no use in JambaDataset
    """

    worker_args = {
        'max_process': args.max_process,
        'seed': args.seed,
        'splited_dataset_pt_folder': args.splited_dataset_pt_folder,
        'dataset': args.dataset,
        'pkt_select_num': args.pkt_select_num,
        'pkt_interval': args.pkt_interval,
        'whole_word_masking': args.whole_word_masking,
        'span_masking': args.whole_word_masking,
        'span_geo_prob': args.whole_word_masking,
        'span_max_length': args.whole_word_masking,
        'spm_model_path': args.spm_model_path,
        'vocab_path': args.vocab_path,
        'tgt_spm_model_path': args.tgt_spm_model_path,
        'tgt_vocab_path': args.tgt_vocab_path,
        'select_packet_len': args.select_packet_len,
    }

    if args.max_process == 0:
        args.logging.info(f"Do not use multiprocessing, start building datasets...")
        for file in os.listdir(args.corpus_path):
            args.logging.info(f"Start processing {file}...")
            jamba_worker(worker_args, f"{args.corpus_path}/{file}", args.max_process)
            args.logging.info(f"Finish processing {file}...")
        args.logging.info(f"datasets are saved in {args.splited_dataset_pt_folder}")

    else:  # multiprocessing
        args.logging.info(f"Starting {args.max_process} workers for building datasets...")
        all_files = os.listdir(args.corpus_path)
        pbar = tqdm(total=len(all_files))
        pbar.set_description('generate dataset files')
        update = lambda *args: pbar.update()
        pool = mp.Pool(args.max_process)
        for file in all_files:
            pool.apply_async(jamba_worker,
                             (worker_args, f"{args.corpus_path}/{file}"),
                             callback=update,error_callback=jamba_worker_error_callback)
        pool.close()
        pool.join()
        args.logging.info(f"datasets are saved in {args.splited_dataset_pt_folder}")

    # Merge datasets.
    args.logging.info(f"start merging dataset files...")
    merge_dataset(args, args.dataset_path, args.max_process)
    args.logging.info(f"whole dataset file is saved in {args.dataset_path}")


def jamba_build_and_save_mask_only(args, split_by_flow=False):
    """
    Build dataset from the given corpus.
    Start max_process processes and each process deals with a part of data.
    max_process:
        0 no use multiprocessing
        >=1 use multiprocessing
    split_by_flow: no use in JambaDataset
    """

    worker_args = {
        'max_process': args.max_process,
        'seed': args.seed,
        'splited_dataset_pt_folder': args.splited_dataset_pt_folder,
        'dataset': args.dataset,
        'pkt_select_num': args.pkt_select_num,
        'pkt_interval': args.pkt_interval,
        'whole_word_masking': args.whole_word_masking,
        'dynamic_masking': args.dynamic_masking,
        'span_masking': args.whole_word_masking,
        'span_geo_prob': args.whole_word_masking,
        'span_max_length': args.whole_word_masking,
        'spm_model_path': args.spm_model_path,
        'vocab_path': args.vocab_path,
        'tgt_spm_model_path': args.tgt_spm_model_path,
        'tgt_vocab_path': args.tgt_vocab_path,
        'select_packet_len': args.select_packet_len,
    }

    if args.max_process == 0:
        args.logging.info(f"Do not use multiprocessing, start building datasets...")
        for file in os.listdir(args.corpus_path):
            args.logging.info(f"Start processing {file}...")
            jamba_worker_mask_only(worker_args, f"{args.corpus_path}/{file}", args.max_process)
            args.logging.info(f"Finish processing {file}...")
        args.logging.info(f"datasets are saved in {args.splited_dataset_pt_folder}")

    else:  # multiprocessing
        args.logging.info(f"Starting {args.max_process} workers for building datasets...")
        all_files = os.listdir(args.corpus_path)
        pbar = tqdm(total=len(all_files))
        pbar.set_description('generate dataset files')
        update = lambda *args: pbar.update()
        pool = mp.Pool(args.max_process)
        for file in all_files:
            pool.apply_async(jamba_worker_mask_only,
                             (worker_args, f"{args.corpus_path}/{file}"),
                             callback=update,error_callback=jamba_worker_error_callback)
        pool.close()
        pool.join()
        args.logging.info(f"datasets are saved in {args.splited_dataset_pt_folder}")

    # Merge datasets.
    args.logging.info(f"start merging dataset files...")
    merge_dataset(args, args.dataset_path, args.max_process)
    args.logging.info(f"whole dataset file is saved in {args.dataset_path}")


def jamba_build_and_save_mask_only_direction(args, split_by_flow=False):
    """
    Build dataset from the given corpus.
    Start max_process processes and each process deals with a part of data.
    max_process:
        0 no use multiprocessing
        >=1 use multiprocessing
    split_by_flow: no use in JambaDataset
    """

    worker_args = {
        'max_process': args.max_process,
        'seed': args.seed,
        'splited_dataset_pt_folder': args.splited_dataset_pt_folder,
        'dataset': args.dataset,
        'pkt_select_num': args.pkt_select_num,
        'whole_word_masking': args.whole_word_masking,
        'dynamic_masking': args.dynamic_masking,
        'span_masking': args.whole_word_masking,
        'span_geo_prob': args.whole_word_masking,
        'span_max_length': args.whole_word_masking,
        'spm_model_path': args.spm_model_path,
        'vocab_path': args.vocab_path,
        'tgt_spm_model_path': args.tgt_spm_model_path,
        'tgt_vocab_path': args.tgt_vocab_path,
        'select_packet_len': args.select_packet_len,
    }

    if args.max_process == 0:
        args.logging.info(f"Do not use multiprocessing, start building datasets...")
        for file in os.listdir(args.corpus_path):
            args.logging.info(f"Start processing {file}...")
            jamba_worker_mask_only_direction(worker_args, f"{args.corpus_path}/{file}",
                                             args.max_process)
            args.logging.info(f"Finish processing {file}...")
        args.logging.info(f"datasets are saved in {args.splited_dataset_pt_folder}")

    else:  # multiprocessing
        args.logging.info(f"Starting {args.max_process} workers for building datasets...")
        all_files = os.listdir(args.corpus_path)
        pbar = tqdm(total=len(all_files))
        pbar.set_description('generate dataset files')
        update = lambda *args: pbar.update()
        pool = mp.Pool(args.max_process)
        for file in all_files:
            pool.apply_async(jamba_worker_mask_only_direction,
                             (worker_args, f"{args.corpus_path}/{file}"),
                             callback=update,error_callback=jamba_worker_error_callback)
        pool.close()
        pool.join()
        args.logging.info(f"datasets are saved in {args.splited_dataset_pt_folder}")

    # Merge datasets.
    args.logging.info(f"start merging dataset files...")
    merge_dataset(args, args.dataset_path, args.max_process)
    args.logging.info(f"whole dataset file is saved in {args.dataset_path}")


def jamba_build_and_save_mask_only_direction_small(args, split_by_flow=False):
    """
    Build dataset from the given corpus.
    Start max_process processes and each process deals with a part of data.
    max_process:
        0 no use multiprocessing
        >=1 use multiprocessing
    split_by_flow: no use in JambaDataset
    """

    worker_args = {
        'max_process': args.max_process,
        'seed': args.seed,
        'ratio': args.ratio,
        'data_choice': args.data_choice,
        'datasets_folder': args.datasets_folder,
        'dataset': args.dataset,
        'pkt_select_num': args.pkt_select_num,
        'whole_word_masking': args.whole_word_masking,
        'dynamic_masking': args.dynamic_masking,
        'span_masking': args.whole_word_masking,
        'span_geo_prob': args.whole_word_masking,
        'span_max_length': args.whole_word_masking,
        'spm_model_path': args.spm_model_path,
        'vocab_path': args.vocab_path,
        'tgt_spm_model_path': args.tgt_spm_model_path,
        'tgt_vocab_path': args.tgt_vocab_path,
        'select_packet_len': args.select_packet_len,
    }

    args.logging.info(f"Do not use multiprocessing, start building datasets...")
    for file in os.listdir(args.corpus_path):
        args.logging.info(f"Start processing {file}...")
        jamba_worker_mask_only_direction_small(worker_args, f"{args.corpus_path}/{file}",
                                               args.max_process)
        args.logging.info(f"Finish processing {file}...")
    args.logging.info(f"datasets are saved in {args.datasets_folder}")


def jamba_build_and_save_mask_only_direction_subsequence(args, split_by_flow=False):
    """
    Build dataset from the given corpus.
    Start max_process processes and each process deals with a part of data.
    max_process:
        0 no use multiprocessing
        >=1 use multiprocessing
    split_by_flow: no use in JambaDataset
    """

    worker_args = {
        'max_process': args.max_process,
        'seed': args.seed,
        'data_choice': args.data_choice,
        'datasets_folder': args.datasets_folder,
        'dataset': args.dataset,
        'pkt_select_num': args.pkt_select_num,
        'whole_word_masking': args.whole_word_masking,
        'dynamic_masking': args.dynamic_masking,
        'span_masking': args.whole_word_masking,
        'span_geo_prob': args.whole_word_masking,
        'span_max_length': args.whole_word_masking,
        'spm_model_path': args.spm_model_path,
        'vocab_path': args.vocab_path,
        'tgt_spm_model_path': args.tgt_spm_model_path,
        'tgt_vocab_path': args.tgt_vocab_path,
        'select_packet_len': args.select_packet_len,
    }

    args.logging.info(f"Do not use multiprocessing, start building datasets...")
    for file in os.listdir(args.corpus_path):
        args.logging.info(f"Start processing {file}...")
        jamba_worker_mask_only_direction_subsequence(worker_args, f"{args.corpus_path}/{file}",
                                               args.max_process)
        args.logging.info(f"Finish processing {file}...")
    args.logging.info(f"datasets are saved in {args.datasets_folder}")



def jamba_worker_error_callback(e):
    print(e)


def jamba_worker_mask_only(worker_args, corpus_file, proc_id=None, start=None, end=None):
    tokenizer = JambaTokenizer1(worker_args)
    set_seed(worker_args['seed'])
    if worker_args['max_process'] > 0:  # multiprocessing
        file_id = os.getpid()
    else:
        file_id = 0

    dataset_writer = open(f"{worker_args['splited_dataset_pt_folder']}/"
                          f"dataset_{worker_args['dataset']}_{file_id}.pt", "ab")
    for flow, coarse_semantic_list, fine_semantic_list, direction_list, stop_flag \
            in jamba_parse_corpora_files(worker_args, tokenizer, corpus_file):
        #pudb.set_trace()
        if stop_flag:  # 文件读取结束
            break
        # sample: 一个列表[src, segment, coarse_semantic, fine_semantic, tgt_mlm, label]
        for sample in jamba_build_a_pretrain_sample_mask_only(worker_args, tokenizer,
                    flow, coarse_semantic_list, fine_semantic_list, direction_list, worker_args['pkt_select_num']):
            pickle.dump(sample, dataset_writer)
    dataset_writer.close()


def jamba_worker_mask_only_direction(worker_args, corpus_file, proc_id=None, start=None, end=None):
    tokenizer = JambaTokenizer1(worker_args)
    set_seed(worker_args['seed'])
    if worker_args['max_process'] > 0:  # multiprocessing
        file_id = os.getpid()
    else:
        file_id = 0

    dataset_writer = open(f"{worker_args['splited_dataset_pt_folder']}/"
                          f"dataset_{worker_args['dataset']}_{file_id}.pt", "ab")
    for flow, coarse_semantic_list, fine_semantic_list, direction_length_list, stop_flag \
            in jamba_parse_corpora_files_direction(worker_args, tokenizer, corpus_file):
        #pudb.set_trace()
        if stop_flag:  # 文件读取结束
            break
        # sample: 一个列表[src, segment, coarse_semantic, fine_semantic, tgt_mlm, label]
        for sample in jamba_build_a_pretrain_sample_mask_only_direction(worker_args, tokenizer,
                    flow, coarse_semantic_list, fine_semantic_list, direction_length_list,
                    worker_args['pkt_select_num']):
            pickle.dump(sample, dataset_writer)
    dataset_writer.close()


def jamba_worker_mask_only_direction_small(worker_args, corpus_file, proc_id=None, start=None, end=None):
    tokenizer = JambaTokenizer1(worker_args)
    set_seed(worker_args['seed'])

    # (1)字节bibyte：jamba_parse_corpora_files_direction_small不动
    # (2)长度length
    # (3)方向direction
    # (4)字节+长度bibyte_length
    # (5)字节+方向bibyte_direction
    # (6)长度+方向length_direction：jamba_parse_corpora_files_direction_small不动
    # (7)字节+长度+方向bibyte_length_direction：jamba_parse_corpora_files_direction_small不动

    dataset_writer = open(f"{worker_args['datasets_folder']}/"
                          f"dataset_small_{worker_args['ratio']}_{worker_args['data_choice']}.pt", "ab")
    for flow, coarse_semantic_list, fine_semantic_list, direction_length_list, stop_flag \
            in jamba_parse_corpora_files_direction_small(worker_args, tokenizer, corpus_file, worker_args['data_choice']):
        #pudb.set_trace()
        if stop_flag:  # 文件读取结束
            break
        # sample: 一个列表[src, segment, coarse_semantic, fine_semantic, tgt_mlm, label]
        for sample in jamba_build_a_pretrain_sample_mask_only_direction_small(worker_args, tokenizer,
                    flow, coarse_semantic_list, fine_semantic_list, direction_length_list, worker_args['data_choice'],
                    worker_args['pkt_select_num']):
            pickle.dump(sample, dataset_writer)
    dataset_writer.close()


def jamba_worker_mask_only_direction_subsequence(worker_args, corpus_file, proc_id=None, start=None, end=None):
    tokenizer = JambaTokenizer1(worker_args)
    set_seed(worker_args['seed'])

    # (1)字节bibyte：jamba_parse_corpora_files_direction_small不动
    # (2)长度length
    # (3)方向direction
    # (4)字节+长度bibyte_length
    # (5)字节+方向bibyte_direction
    # (6)长度+方向length_direction：jamba_parse_corpora_files_direction_small不动
    # (7)字节+长度+方向bibyte_length_direction：jamba_parse_corpora_files_direction_small不动

    dataset_writer = open(f"{worker_args['datasets_folder']}/"
                          f"dataset_{worker_args['data_choice']}.pt", "ab")
    for flow, coarse_semantic_list, fine_semantic_list, direction_length_list, stop_flag \
            in jamba_parse_corpora_files_direction_small(worker_args, tokenizer, corpus_file, worker_args['data_choice']):
        #pudb.set_trace()
        if stop_flag:  # 文件读取结束
            break
        # sample: 一个列表[src, segment, coarse_semantic, fine_semantic, tgt_mlm, label]
        for sample in jamba_build_a_pretrain_sample_mask_only_direction_small(worker_args, tokenizer,
                    flow, coarse_semantic_list, fine_semantic_list, direction_length_list, worker_args['data_choice'],
                    worker_args['pkt_select_num']):
            pickle.dump(sample, dataset_writer)
    dataset_writer.close()


# 生成一个预训练样本, 返回值是列表[src, segment, coarse_semantic, fine_semantic, tgt_mlm, label]
# 只考虑mask任务，每条流选前5个包
def jamba_build_a_pretrain_sample_mask_only(worker_args, tokenizer, flow, coarse_semantic_list, fine_semantic_list, direction_list, pkt_select_num:int=5):
    #pudb.set_trace()

    src,coarse_semantic,fine_semantic,segment = [],[],[],[]
    src.append(tokenizer.vocab.get(CLS_TOKEN))
    coarse_semantic.append(tokenizer.vocab.get(CLS_TOKEN))
    fine_semantic.append(tokenizer.vocab.get(CLS_TOKEN))
    for pkt_idx in range(0, min(len(flow), pkt_select_num)):
        # 数据包tokens
        src.extend(flow[pkt_idx])
        src.append(tokenizer.vocab.get(SEP_TOKEN))
        # segment和position
        segment.append(len(src))  # [30,60,89]代表第1个segment长度30，第2个segment长度60-30...
        # coarse semantic和fine semantic
        coarse_semantic.extend(coarse_semantic_list[pkt_idx])
        coarse_semantic.append(tokenizer.vocab.get(SEP_TOKEN))
        fine_semantic.extend(fine_semantic_list[pkt_idx])
        fine_semantic.append(tokenizer.vocab.get(SEP_TOKEN))

    if worker_args['dynamic_masking']:
        yield [src, segment, coarse_semantic, fine_semantic]
    else:
        src, tgt_mlm = mask_seq_jamba(src, tokenizer, worker_args['whole_word_masking'],
                                worker_args['span_masking'], worker_args['span_geo_prob'],
                                worker_args['span_max_length'])
        yield [src, segment, coarse_semantic, fine_semantic, tgt_mlm]


def jamba_build_a_pretrain_sample_mask_only_direction_small(worker_args, tokenizer, flow,
                                                      coarse_semantic_list, fine_semantic_list,
                                                      direction_length_list, data_choice,
                                                      pkt_select_num:int=5):
    #pudb.set_trace()

    src,coarse_semantic,fine_semantic,segment = [],[],[],[]

    if data_choice == 'bibyte' or data_choice == 'bibyte_length' or data_choice == 'bibyte_direction' \
            or data_choice == 'bibyte_length_direction':
        for pkt_idx in range(0, min(len(flow), pkt_select_num)):
            if pkt_idx == 0:
                src.append(tokenizer.vocab.get(CLS_TOKEN))
                coarse_semantic.append(tokenizer.vocab.get(CLS_TOKEN))
                fine_semantic.append(tokenizer.vocab.get(CLS_TOKEN))
            else:
                src.append(tokenizer.vocab.get(SEP_TOKEN))
                coarse_semantic.append(tokenizer.vocab.get(SEP_TOKEN))
                fine_semantic.append(tokenizer.vocab.get(SEP_TOKEN))

            # 数据包tokens
            src.extend(flow[pkt_idx])
            # segment和position
            segment.append(len(src))  # [30,60,89]代表第1个segment长度30，第2个segment长度60-30...
            # coarse semantic和fine semantic
            coarse_semantic.extend(coarse_semantic_list[pkt_idx])
            fine_semantic.extend(fine_semantic_list[pkt_idx])

    if data_choice == 'bibyte_length' or data_choice == 'bibyte_direction' \
            or data_choice == 'bibyte_length_direction':
        src.append(tokenizer.vocab.get(SEP_TOKEN))
        coarse_semantic.append(tokenizer.vocab.get(SEP_TOKEN))
        fine_semantic.append(tokenizer.vocab.get(SEP_TOKEN))
        src.extend(direction_length_list)
        coarse_semantic.extend([8] * len(direction_length_list))
        fine_semantic.extend([15] * len(direction_length_list))
        segment.append(len(src))
    elif data_choice == 'length' or data_choice == 'direction' or data_choice == 'length_direction':
        src.append(tokenizer.vocab.get(CLS_TOKEN))
        coarse_semantic.append(tokenizer.vocab.get(CLS_TOKEN))
        fine_semantic.append(tokenizer.vocab.get(CLS_TOKEN))
        src.extend(direction_length_list)
        coarse_semantic.extend([8] * len(direction_length_list))
        fine_semantic.extend([15] * len(direction_length_list))
        segment.append(len(src))
    else:
        pass

    if worker_args['dynamic_masking']:
        yield [src, segment, coarse_semantic, fine_semantic]
    else:
        src, tgt_mlm = mask_seq_jamba(src, tokenizer, worker_args['whole_word_masking'],
                                worker_args['span_masking'], worker_args['span_geo_prob'],
                                worker_args['span_max_length'])
        yield [src, segment, coarse_semantic, fine_semantic, tgt_mlm]


def jamba_build_a_pretrain_sample_mask_only_direction_subsequence(worker_args, tokenizer, flow,
                                                      coarse_semantic_list, fine_semantic_list,
                                                      direction_length_list, data_choice,
                                                      pkt_select_num:int=5):
    #pudb.set_trace()

    src,coarse_semantic,fine_semantic,segment = [],[],[],[]

    if data_choice == 'bibyte' or data_choice == 'bibyte_length' or data_choice == 'bibyte_direction' \
            or data_choice == 'bibyte_length_direction':
        for pkt_idx in range(0, min(len(flow), pkt_select_num)):
            if pkt_idx == 0:
                src.append(tokenizer.vocab.get(CLS_TOKEN))
                coarse_semantic.append(tokenizer.vocab.get(CLS_TOKEN))
                fine_semantic.append(tokenizer.vocab.get(CLS_TOKEN))
            else:
                src.append(tokenizer.vocab.get(SEP_TOKEN))
                coarse_semantic.append(tokenizer.vocab.get(SEP_TOKEN))
                fine_semantic.append(tokenizer.vocab.get(SEP_TOKEN))

            # 数据包tokens
            src.extend(flow[pkt_idx])
            # segment和position
            segment.append(len(src))  # [30,60,89]代表第1个segment长度30，第2个segment长度60-30...
            # coarse semantic和fine semantic
            coarse_semantic.extend(coarse_semantic_list[pkt_idx])
            fine_semantic.extend(fine_semantic_list[pkt_idx])

    if data_choice == 'bibyte_length' or data_choice == 'bibyte_direction' \
            or data_choice == 'bibyte_length_direction':
        src.append(tokenizer.vocab.get(SEP_TOKEN))
        coarse_semantic.append(tokenizer.vocab.get(SEP_TOKEN))
        fine_semantic.append(tokenizer.vocab.get(SEP_TOKEN))
        src.extend(direction_length_list)
        coarse_semantic.extend([8] * len(direction_length_list))
        fine_semantic.extend([15] * len(direction_length_list))
        segment.append(len(src))
    elif data_choice == 'length' or data_choice == 'direction' or data_choice == 'length_direction':
        src.append(tokenizer.vocab.get(CLS_TOKEN))
        coarse_semantic.append(tokenizer.vocab.get(CLS_TOKEN))
        fine_semantic.append(tokenizer.vocab.get(CLS_TOKEN))
        src.extend(direction_length_list)
        coarse_semantic.extend([8] * len(direction_length_list))
        fine_semantic.extend([15] * len(direction_length_list))
        segment.append(len(src))
    else:
        pass

    if worker_args['dynamic_masking']:
        yield [src, segment, coarse_semantic, fine_semantic]
    else:
        src, tgt_mlm = mask_seq_jamba(src, tokenizer, worker_args['whole_word_masking'],
                                worker_args['span_masking'], worker_args['span_geo_prob'],
                                worker_args['span_max_length'])
        yield [src, segment, coarse_semantic, fine_semantic, tgt_mlm]


def jamba_build_a_pretrain_sample_mask_only_direction(worker_args, tokenizer, flow,
                                                      coarse_semantic_list, fine_semantic_list,
                                                      direction_length_list, pkt_select_num:int=5):
    #pudb.set_trace()

    src,coarse_semantic,fine_semantic,segment = [],[],[],[]
    for pkt_idx in range(0, min(len(flow), pkt_select_num)):
        if pkt_idx == 0:
            src.append(tokenizer.vocab.get(CLS_TOKEN))
            coarse_semantic.append(tokenizer.vocab.get(CLS_TOKEN))
            fine_semantic.append(tokenizer.vocab.get(CLS_TOKEN))
        else:
            src.append(tokenizer.vocab.get(SEP_TOKEN))
            coarse_semantic.append(tokenizer.vocab.get(SEP_TOKEN))
            fine_semantic.append(tokenizer.vocab.get(SEP_TOKEN))

        # 数据包tokens
        src.extend(flow[pkt_idx])
        # segment和position
        segment.append(len(src))  # [30,60,89]代表第1个segment长度30，第2个segment长度60-30...
        # coarse semantic和fine semantic
        coarse_semantic.extend(coarse_semantic_list[pkt_idx])
        fine_semantic.extend(fine_semantic_list[pkt_idx])

    src.append(tokenizer.vocab.get(SEP_TOKEN))
    coarse_semantic.append(tokenizer.vocab.get(SEP_TOKEN))
    fine_semantic.append(tokenizer.vocab.get(SEP_TOKEN))
    src.extend(direction_length_list)
    coarse_semantic.extend([8] * len(direction_length_list))
    fine_semantic.extend([15] * len(direction_length_list))
    segment.append(len(src))

    if worker_args['dynamic_masking']:
        yield [src, segment, coarse_semantic, fine_semantic]
    else:
        src, tgt_mlm = mask_seq_jamba(src, tokenizer, worker_args['whole_word_masking'],
                                worker_args['span_masking'], worker_args['span_geo_prob'],
                                worker_args['span_max_length'])
        yield [src, segment, coarse_semantic, fine_semantic, tgt_mlm]


def jamba_worker(worker_args, corpus_file, proc_id=None, start=None, end=None):
    tokenizer = JambaTokenizer1(worker_args)
    set_seed(worker_args['seed'])
    if worker_args['max_process'] > 0:  # multiprocessing
        file_id = os.getpid()
    else:
        file_id = 0

    dataset_writer = open(f"{worker_args['splited_dataset_pt_folder']}/"
                          f"dataset_{worker_args['dataset']}_{file_id}.pt", "ab")
    for flow, coarse_semantic_list, fine_semantic_list, direction_list, stop_flag \
            in jamba_parse_corpora_files(worker_args, tokenizer, corpus_file):
        #pudb.set_trace()
        if stop_flag:  # 文件读取结束
            break
        # sample: 一个列表[src, segment, coarse_semantic, fine_semantic, tgt_mlm, label]
        for sample in jamba_build_a_pretrain_sample(worker_args, tokenizer,
                    flow, coarse_semantic_list, fine_semantic_list, direction_list, worker_args['pkt_interval']):
            pickle.dump(sample, dataset_writer)
    dataset_writer.close()


# 生成一个预训练样本, 返回值是列表[src, segment, coarse_semantic, fine_semantic, tgt_mlm, label]
def jamba_build_a_pretrain_sample(worker_args, tokenizer, flow, coarse_semantic_list, fine_semantic_list, direction_list, pkt_interval:int=15):
    cnt = int((len(flow)-1)/pkt_interval) + 1

    for round_idx in range(cnt):
        # 根据概率选择扰动类别
        label_prob = random.random()
        if label_prob < 0.6:
            label = 0  # 正常
        elif label_prob < 0.8:
            label = 1  # 丢包
        elif label_prob < 0.9:
            label = 2  # 重传
        else:
            label = 3  # 乱序

        #pudb.set_trace()

        # 判断是否能构建样本
        if round_idx != cnt-1:
            num_packets = pkt_interval
        else:
            num_packets = len(flow) - round_idx*pkt_interval
            if (num_packets < 2) or (num_packets < 4 and label == 1):
                continue

        if label == 0:  # 正常类样本：连续、顺序不变
            if num_packets >= 10:
                random_start = random.randint(0, num_packets - 10)
                start = round_idx * pkt_interval + random_start
                stop = start + 10
            else:  # 2~9 个包
                start = round_idx * pkt_interval
                stop = round_idx * pkt_interval + num_packets

            src,coarse_semantic,fine_semantic,segment = [],[],[],[]
            src.append(tokenizer.vocab.get(CLS_TOKEN))
            coarse_semantic.append(tokenizer.vocab.get(CLS_TOKEN))
            fine_semantic.append(tokenizer.vocab.get(CLS_TOKEN))
            for pkt_idx in range(start, stop):
                # 数据包tokens
                src.extend(flow[pkt_idx])
                src.append(tokenizer.vocab.get(SEP_TOKEN))
                # segment和position
                segment.append(len(src))  # [30,60,89]代表第1个segment长度30，第2个segment长度60-30...
                # coarse semantic和fine semantic
                coarse_semantic.extend(coarse_semantic_list[pkt_idx])
                coarse_semantic.append(tokenizer.vocab.get(SEP_TOKEN))
                fine_semantic.extend(fine_semantic_list[pkt_idx])
                fine_semantic.append(tokenizer.vocab.get(SEP_TOKEN))

            if worker_args['dynamic_masking']:
                yield [src, segment, coarse_semantic, fine_semantic, label]
            else:
                src, tgt_mlm = mask_seq_jamba(src, tokenizer, worker_args['whole_word_masking'],
                                              worker_args['span_masking'], worker_args['span_geo_prob'],
                                              worker_args['span_max_length'])
                yield [src, segment, coarse_semantic, fine_semantic, tgt_mlm, label]

        elif label == 1:  # 丢包：删除2个包
            if num_packets >= 12:
                random_start = random.randint(0, num_packets - 12)
                start = round_idx * pkt_interval + random_start
                stop = start + 12
            else:  # 4~11 个包
                start = round_idx * pkt_interval
                stop = round_idx * pkt_interval + num_packets

            possible_numbers = list(range(start, stop))
            drop_idx1, drop_idx2 = random.sample(possible_numbers, 2)  # 删除的数据包的idx

            src,coarse_semantic,fine_semantic,segment = [],[],[],[]
            src.append(tokenizer.vocab.get(CLS_TOKEN))
            coarse_semantic.append(tokenizer.vocab.get(CLS_TOKEN))
            fine_semantic.append(tokenizer.vocab.get(CLS_TOKEN))
            for pkt_idx in range(start, stop):
                if (pkt_idx != drop_idx1) and (pkt_idx != drop_idx2):
                    # 数据包tokens
                    src.extend(flow[pkt_idx])
                    src.append(tokenizer.vocab.get(SEP_TOKEN))
                    # segment和position
                    segment.append(len(src))  # [30,60,89]代表第1个segment长度30，第2个segment长度60-30...
                    # coarse semantic和fine semantic
                    coarse_semantic.extend(coarse_semantic_list[pkt_idx])
                    coarse_semantic.append(tokenizer.vocab.get(SEP_TOKEN))
                    fine_semantic.extend(fine_semantic_list[pkt_idx])
                    fine_semantic.append(tokenizer.vocab.get(SEP_TOKEN))

            if worker_args['dynamic_masking']:
                yield [src, segment, coarse_semantic, fine_semantic, label]
            else:
                src, tgt_mlm = mask_seq_jamba(src, tokenizer, worker_args['whole_word_masking'],
                                              worker_args['span_masking'], worker_args['span_geo_prob'],
                                              worker_args['span_max_length'])
                yield [src, segment, coarse_semantic, fine_semantic, tgt_mlm, label]

        elif label == 2:  # 重传：插入2个已有包
            if num_packets >= 8:
                random_start = random.randint(0, num_packets - 8)
                start = round_idx * pkt_interval + random_start
                stop = start + 8
            else:  # 2~7 个包
                start = round_idx * pkt_interval
                stop = round_idx * pkt_interval + num_packets

            possible_numbers = list(range(start, stop))
            repeat_idx = random.sample(possible_numbers, 2)  # 重传的数据包的idx列表
            repeat_idx.sort()

            repeat_pkts, repeat_coarse, repeat_fine = [],[],[]
            repeat_pkts.append(flow[repeat_idx[0]])
            repeat_pkts.append(flow[repeat_idx[1]])
            repeat_coarse.append(coarse_semantic_list[repeat_idx[0]])
            repeat_coarse.append(coarse_semantic_list[repeat_idx[1]])
            repeat_fine.append(fine_semantic_list[repeat_idx[0]])
            repeat_fine.append(fine_semantic_list[repeat_idx[1]])

            tmp_packets = copy.deepcopy(flow[start:stop])
            tmp_coarse = copy.deepcopy(coarse_semantic_list[start:stop])
            tmp_fine = copy.deepcopy(fine_semantic_list[start:stop])

            insert_idx1 = random.randint(repeat_idx[0]+1-start, stop-start)
            tmp_packets.insert(insert_idx1, repeat_pkts[0])
            tmp_coarse.insert(insert_idx1, repeat_coarse[0])
            tmp_fine.insert(insert_idx1, repeat_fine[0])
            if insert_idx1+start > repeat_idx[1]:
                insert_idx2 = random.randint(repeat_idx[1]+1-start, stop+1-start)
            else:
                insert_idx2 = random.randint(repeat_idx[1]+2-start, stop+1-start)
            tmp_packets.insert(insert_idx2, repeat_pkts[1])
            tmp_coarse.insert(insert_idx2, repeat_coarse[1])
            tmp_fine.insert(insert_idx2, repeat_fine[1])

            src,coarse_semantic,fine_semantic,segment = [],[],[],[]
            src.append(tokenizer.vocab.get(CLS_TOKEN))
            coarse_semantic.append(tokenizer.vocab.get(CLS_TOKEN))
            fine_semantic.append(tokenizer.vocab.get(CLS_TOKEN))
            for pkt_idx in range(0, stop-start+2):
                # 数据包tokens
                src.extend(tmp_packets[pkt_idx])
                src.append(tokenizer.vocab.get(SEP_TOKEN))
                # segment和position
                segment.append(len(src))  # [30,60,89]代表第1个segment长度30，第2个segment长度60-30...
                # coarse semantic和fine semantic
                coarse_semantic.extend(tmp_coarse[pkt_idx])
                coarse_semantic.append(tokenizer.vocab.get(SEP_TOKEN))
                fine_semantic.extend(tmp_fine[pkt_idx])
                fine_semantic.append(tokenizer.vocab.get(SEP_TOKEN))

            if worker_args['dynamic_masking']:
                yield [src, segment, coarse_semantic, fine_semantic, label]
            else:
                src, tgt_mlm = mask_seq_jamba(src, tokenizer, worker_args['whole_word_masking'],
                                              worker_args['span_masking'], worker_args['span_geo_prob'],
                                              worker_args['span_max_length'])
                yield [src, segment, coarse_semantic, fine_semantic, tgt_mlm, label]

        elif label == 3:  # 乱序：2次交换前后两个包的顺序，若flow中只有2个数据包，则只交换一次
            if num_packets >= 10:
                random_start = random.randint(0, num_packets - 10)
                start = round_idx * pkt_interval + random_start
                stop = start + 10
            else:  # 2~9 个包
                start = round_idx * pkt_interval
                stop = round_idx * pkt_interval + num_packets

            src,coarse_semantic,fine_semantic,segment = [],[],[],[]
            src.append(tokenizer.vocab.get(CLS_TOKEN))
            coarse_semantic.append(tokenizer.vocab.get(CLS_TOKEN))
            fine_semantic.append(tokenizer.vocab.get(CLS_TOKEN))

            if num_packets == 2:
                # 数据包tokens
                src.extend(flow[start+1])
                src.append(tokenizer.vocab.get(SEP_TOKEN))
                # segment和position
                segment.append(len(src))  # [30,60,89]代表第1个segment长度30，第2个segment长度60-30...
                # coarse semantic和fine semantic
                coarse_semantic.extend(coarse_semantic_list[start+1])
                coarse_semantic.append(tokenizer.vocab.get(SEP_TOKEN))
                fine_semantic.extend(fine_semantic_list[start+1])
                fine_semantic.append(tokenizer.vocab.get(SEP_TOKEN))

                # 数据包tokens
                src.extend(flow[start])
                src.append(tokenizer.vocab.get(SEP_TOKEN))
                # segment和position
                segment.append(len(src))  # [30,60,89]代表第1个segment长度30，第2个segment长度60-30...
                # coarse semantic和fine semantic
                coarse_semantic.extend(coarse_semantic_list[start])
                coarse_semantic.append(tokenizer.vocab.get(SEP_TOKEN))
                fine_semantic.extend(fine_semantic_list[start])
                fine_semantic.append(tokenizer.vocab.get(SEP_TOKEN))

            else:  # num_packets > 2
                possible_numbers = list(range(start, stop - 1))
                switch_idx = random.sample(possible_numbers, 2)  # 交换的数据包的idx列表，第idx数据包与第idx+1数据包交换
                switch_idx.sort()

                for pkt_idx in range(start, stop):
                    if (pkt_idx == switch_idx[0]) or (pkt_idx == switch_idx[1]):
                        # 数据包tokens
                        src.extend(flow[pkt_idx+1])
                        src.append(tokenizer.vocab.get(SEP_TOKEN))
                        # segment和position
                        segment.append(len(src))  # [30,60,89]代表第1个segment长度30，第2个segment长度60-30...
                        # coarse semantic和fine semantic
                        coarse_semantic.extend(coarse_semantic_list[pkt_idx+1])
                        coarse_semantic.append(tokenizer.vocab.get(SEP_TOKEN))
                        fine_semantic.extend(fine_semantic_list[pkt_idx+1])
                        fine_semantic.append(tokenizer.vocab.get(SEP_TOKEN))
                    elif (pkt_idx == (switch_idx[0]+1)) or (pkt_idx == (switch_idx[1]+1)):
                        # 数据包tokens
                        src.extend(flow[pkt_idx-1])
                        src.append(tokenizer.vocab.get(SEP_TOKEN))
                        # segment和position
                        segment.append(len(src))  # [30,60,89]代表第1个segment长度30，第2个segment长度60-30...
                        # coarse semantic和fine semantic
                        coarse_semantic.extend(coarse_semantic_list[pkt_idx-1])
                        coarse_semantic.append(tokenizer.vocab.get(SEP_TOKEN))
                        fine_semantic.extend(fine_semantic_list[pkt_idx-1])
                        fine_semantic.append(tokenizer.vocab.get(SEP_TOKEN))
                    else:
                        # 数据包tokens
                        src.extend(flow[pkt_idx])
                        src.append(tokenizer.vocab.get(SEP_TOKEN))
                        # segment和position
                        segment.append(len(src))  # [30,60,89]代表第1个segment长度30，第2个segment长度60-30...
                        # coarse semantic和fine semantic
                        coarse_semantic.extend(coarse_semantic_list[pkt_idx])
                        coarse_semantic.append(tokenizer.vocab.get(SEP_TOKEN))
                        fine_semantic.extend(fine_semantic_list[pkt_idx])
                        fine_semantic.append(tokenizer.vocab.get(SEP_TOKEN))
            if worker_args['dynamic_masking']:
                yield [src, segment, coarse_semantic, fine_semantic, label]
            else:
                src, tgt_mlm = mask_seq_jamba(src, tokenizer, worker_args['whole_word_masking'],
                                              worker_args['span_masking'], worker_args['span_geo_prob'],
                                              worker_args['span_max_length'])
                yield [src, segment, coarse_semantic, fine_semantic, tgt_mlm, label]


# 从corpora bigram/gram file中顺序读入每条流，遍历使用，每次返回5个值
# 流：二维数组, 粗粒度语义：二维数组, 细粒度语义：二维数组, 方向：一维数组, 文件结束标值：bool
def jamba_parse_corpora_files(worker_args, tokenizer, file_path):
    flow = []  # 流列表，里面包含这条流的所有数据包
    coarse_semantic_list = []  # 粗粒度语义编码列表，保存这条流中所有数据包的粗粒度语义编码
    fine_semantic_list = []  # 细粒度语义编码列表，保存这条流中所有数据包的细粒度语义编码
    direction_list = []  # 保存这条流中所有数据包的方向
    direction_flag = 1  # 初始方向为1
    stop_flag = False
    with open(file_path, 'r') as f:
        while True:
            line = f.readline()

            # 文件读取结束
            if not line:
                stop_flag = True
                yield flow, coarse_semantic_list, fine_semantic_list, direction_list, stop_flag

            packet = line.strip()

            # 新的一条流开始
            if packet.startswith("||"):
                # 上一条流已读完，返回它
                if flow != []:
                    yield flow, coarse_semantic_list, fine_semantic_list, direction_list, stop_flag
                flow = []
                coarse_semantic_list = []
                fine_semantic_list = []
                direction_list = []
                direction_flag = 1  # 每条新流开始默认方向1

                packet = packet[2:]
                # packet_tokens = tokenizer.tokenize(packet)
                packet_tokens = packet.split(' ')
                if len(packet_tokens[-1]) != 4:
                    packet_tokens = packet_tokens[:-1]
                coarse_tokens, fine_tokens, error_flag = \
                    jamba_semantic_embedding(worker_args, packet_tokens)
                if error_flag:
                    continue
                flow.append(tokenizer.convert_tokens_to_ids(packet_tokens))
                coarse_semantic_list.append(coarse_tokens)
                fine_semantic_list.append(fine_tokens)
                direction_list.append(direction_flag)
            elif packet == "":
                # 空行，表示方向切换
                direction_flag = 2 if direction_flag == 1 else 1
            else:
                # 正常数据包行
                #packet_tokens = tokenizer.tokenize(packet)
                packet_tokens = packet.split(' ')
                if len(packet_tokens[-1]) != 4:
                    packet_tokens = packet_tokens[:-1]
                coarse_tokens, fine_tokens, error_flag = \
                    jamba_semantic_embedding(worker_args, packet_tokens)
                if error_flag:
                    continue
                flow.append(tokenizer.convert_tokens_to_ids(packet_tokens))
                coarse_semantic_list.append(coarse_tokens)
                fine_semantic_list.append(fine_tokens)
                direction_list.append(direction_flag)


def jamba_parse_corpora_files_direction_small(worker_args, tokenizer, file_path, data_choice):
    flow = []  # 流列表，里面包含这条流的所有数据包
    coarse_semantic_list = []  # 粗粒度语义编码列表，保存这条流中所有数据包的粗粒度语义编码
    fine_semantic_list = []  # 细粒度语义编码列表，保存这条流中所有数据包的细粒度语义编码
    direction_length_list = []  # 该条流的方向长度列表
    stop_flag = False
    with open(file_path, 'r') as f:
        while True:
            line = f.readline()

            # 文件读取结束
            if not line:
                stop_flag = True

                for pop_idx in pop_pkt_idx[::-1]:
                    direction_length_tokens.pop(pop_idx)
                direction_length_list = tokenizer.convert_tokens_to_ids(direction_length_tokens)

                yield flow, coarse_semantic_list, fine_semantic_list, direction_length_list, \
                    stop_flag

            packet = line.strip()

            # 新的一条流开始
            if packet.startswith("||"):
                # 上一条流已读完，返回它
                if flow != []:
                    for pop_idx in pop_pkt_idx[::-1]:
                        direction_length_tokens.pop(pop_idx)
                    direction_length_list = tokenizer.convert_tokens_to_ids(direction_length_tokens)

                    yield flow, coarse_semantic_list, fine_semantic_list, direction_length_list, \
                        stop_flag

                pkt_cnt = 0
                pop_pkt_idx = []
                flow = []
                coarse_semantic_list = []
                fine_semantic_list = []

                direction_length_string = packet[2:]
                direction_length_tokens = direction_length_string.split(' ')
                direction_length_tokens = direction_length_tokens[:20]
                for idx_item, item in enumerate(direction_length_tokens):
                    if data_choice == 'length' or data_choice == 'bibyte_length':
                        tmp_item = abs(int(item))
                        if tmp_item > 1500:
                            direction_length_tokens[idx_item] = '+1500'
                        else:
                            direction_length_tokens[idx_item] = '+'+str(tmp_item)

                    elif data_choice == 'direction' or data_choice == 'bibyte_direction':
                        tmp_item = int(item)
                        if tmp_item > 0:
                            direction_length_tokens[idx_item] = '+1'
                        elif tmp_item <= 0:
                            direction_length_tokens[idx_item] = '-1'
                        else:
                            pass

                    else:
                        tmp_item = int(item)
                        if tmp_item > 1500:
                            direction_length_tokens[idx_item] = '+1500'
                        elif tmp_item < -1500:
                            direction_length_tokens[idx_item] = '-1500'
                        else:
                            pass

                line = f.readline()
                packet = line.strip()
                packet_tokens = packet.split(' ')
                if len(packet_tokens[-1]) != 4:
                    packet_tokens = packet_tokens[:-1]
                coarse_tokens, fine_tokens, error_flag = \
                    jamba_semantic_embedding(worker_args, packet_tokens)
                if error_flag:
                    pop_pkt_idx.append(pkt_cnt)
                    continue
                flow.append(tokenizer.convert_tokens_to_ids(packet_tokens))
                coarse_semantic_list.append(coarse_tokens)
                fine_semantic_list.append(fine_tokens)
            elif packet == "":
                pass
            else:
                # 正常数据包行
                pkt_cnt += 1
                packet_tokens = packet.split(' ')
                if len(packet_tokens[-1]) != 4:
                    packet_tokens = packet_tokens[:-1]
                coarse_tokens, fine_tokens, error_flag = \
                    jamba_semantic_embedding(worker_args, packet_tokens)
                if error_flag:
                    pop_pkt_idx.append(pkt_cnt)
                    continue
                flow.append(tokenizer.convert_tokens_to_ids(packet_tokens))
                coarse_semantic_list.append(coarse_tokens)
                fine_semantic_list.append(fine_tokens)


# 从corpora gram file中顺序读入每条流，遍历使用，每次返回5个值
# 流：二维数组, 粗粒度语义：二维数组, 细粒度语义：二维数组, 方向：一维数组, 文件结束标值：bool
def jamba_parse_corpora_files_direction(worker_args, tokenizer, file_path):
    flow = []  # 流列表，里面包含这条流的所有数据包
    coarse_semantic_list = []  # 粗粒度语义编码列表，保存这条流中所有数据包的粗粒度语义编码
    fine_semantic_list = []  # 细粒度语义编码列表，保存这条流中所有数据包的细粒度语义编码
    direction_length_list = []  # 该条流的方向长度列表
    stop_flag = False
    with open(file_path, 'r') as f:
        while True:
            line = f.readline()

            # 文件读取结束
            if not line:
                stop_flag = True

                for pop_idx in pop_pkt_idx[::-1]:
                    direction_length_tokens.pop(pop_idx)
                direction_length_list = tokenizer.convert_tokens_to_ids(direction_length_tokens)

                yield flow, coarse_semantic_list, fine_semantic_list, direction_length_list, \
                    stop_flag

            packet = line.strip()

            # 新的一条流开始
            if packet.startswith("||"):
                # 上一条流已读完，返回它
                if flow != []:
                    for pop_idx in pop_pkt_idx[::-1]:
                        direction_length_tokens.pop(pop_idx)
                    direction_length_list = tokenizer.convert_tokens_to_ids(direction_length_tokens)

                    yield flow, coarse_semantic_list, fine_semantic_list, direction_length_list, \
                        stop_flag

                pkt_cnt = 0
                pop_pkt_idx = []
                flow = []
                coarse_semantic_list = []
                fine_semantic_list = []

                direction_length_string = packet[2:]
                direction_length_tokens = direction_length_string.split(' ')
                direction_length_tokens = direction_length_tokens[:20]
                for idx_item, item in enumerate(direction_length_tokens):
                    tmp_item = int(item)
                    if tmp_item > 1500:
                        direction_length_tokens[idx_item] = '+1500'
                    elif tmp_item < -1500:
                        direction_length_tokens[idx_item] = '-1500'
                    else:
                        pass

                line = f.readline()
                packet = line.strip()
                packet_tokens = packet.split(' ')
                if len(packet_tokens[-1]) != 4:
                    packet_tokens = packet_tokens[:-1]
                coarse_tokens, fine_tokens, error_flag = \
                    jamba_semantic_embedding(worker_args, packet_tokens)
                if error_flag:
                    pop_pkt_idx.append(pkt_cnt)
                    continue
                flow.append(tokenizer.convert_tokens_to_ids(packet_tokens))
                coarse_semantic_list.append(coarse_tokens)
                fine_semantic_list.append(fine_tokens)
            elif packet == "":
                pass
            else:
                # 正常数据包行
                pkt_cnt += 1
                packet_tokens = packet.split(' ')
                if len(packet_tokens[-1]) != 4:
                    packet_tokens = packet_tokens[:-1]
                coarse_tokens, fine_tokens, error_flag = \
                    jamba_semantic_embedding(worker_args, packet_tokens)
                if error_flag:
                    pop_pkt_idx.append(pkt_cnt)
                    continue
                flow.append(tokenizer.convert_tokens_to_ids(packet_tokens))
                coarse_semantic_list.append(coarse_tokens)
                fine_semantic_list.append(fine_tokens)


# 生成coarse semantic embedding和fine semantic embedding
def jamba_semantic_embedding(worker_args, tokens: List[str]):
    error_flag = False  # 若返回时error_flag==True，则舍弃这个数据包

    tokens_len = len(tokens)
    # 先默认数据包均包含(worker_args['select_packet_len']/2)个tokens，即均包含32个tokens(64字节)
    coarse_tokens = [5] * int(worker_args['select_packet_len']/2)  # 除1-4编号外的全是payload
    fine_tokens = [12] * int(worker_args['select_packet_len']/2)  # 11个字段之外的packet字段赋值12

    # Step 1: IP header
    ihl_token = tokens[0]
    try:
        ihl = int(ihl_token[1], 16)
    except (IndexError, ValueError):
        error_flag = True
        return coarse_tokens, fine_tokens, error_flag

    ip_header_len_tokens = (ihl << 2) >> 1  # ihl * 4 / 2
    if ip_header_len_tokens != 10:
        error_flag = True
        return coarse_tokens, fine_tokens, error_flag
    end_ip = ip_header_len_tokens

    coarse_tokens[0:end_ip] = [1] * end_ip
    fine_tokens[1] = 1
    fine_tokens[4] = 2
    fine_tokens[6:8] = [3]*2
    fine_tokens[8:10] = [4]*2

    # Step 2: L4 protocol header
    try:
        protocol = int(tokens[4][2:], 16)  # 获取协议十进制编码tcp：6，udp：17
    except:
        error_flag = True
        return coarse_tokens, fine_tokens, error_flag

    l4_start = end_ip
    if protocol == 6:  # TCP
        l4_label = 2
        l4_len_tokens = 10  # 只考虑tcp标准情况
        fine_tokens[0 + l4_start] = 5
        fine_tokens[1 + l4_start] = 6
        fine_tokens[2 + l4_start:4 + l4_start] = [7] * 2
        fine_tokens[4 + l4_start:6 + l4_start] = [8] * 2
        fine_tokens[6 + l4_start] = 9
    elif protocol == 17:  # UDP
        l4_label = 3
        l4_len_tokens = 4  # 8 bytes / 2
        fine_tokens[0 + l4_start] = 10
        fine_tokens[1 + l4_start] = 11
    else:
        l4_label = 4
        l4_len_tokens = 4  # 8 bytes / 2

    end_l4 = l4_start + l4_len_tokens
    coarse_tokens[l4_start:end_l4] = [l4_label] * l4_len_tokens

    coarse_tokens = coarse_tokens[:tokens_len]
    fine_tokens = fine_tokens[:tokens_len]

    coarse_tokens = [ctx+2 for ctx in coarse_tokens]  # 从3开始编码，给PAD SEP CLS保留编码号
    fine_tokens = [ftx+2 for ftx in fine_tokens]

    return coarse_tokens, fine_tokens, error_flag


class JambaDataset(Dataset):
    """
    Construct dataset for MLM and MIX tasks from the given corpus.
    Each document consists of multiple paragraphs,
    Each paragraph consists of multiple sentences,
    and each sentence occupies a single line.
    Paragraphs in corpus must be separated by empty lines.
    Documents in corpus must be separated by empty lines.
    """

    def __init__(self, args, vocab, tokenizer):
        super(JambaDataset, self).__init__(args, vocab, tokenizer)
        self.args = args
        self.short_seq_prob = args.short_seq_prob
        self.logging = args.logging

    def build_and_save(self, max_process, split_by_flow=False):
        """
        Build dataset from the given corpus.
        Start max_process processes and each process deals with a part of data.
        max_process:
            0 no use multiprocessing
            >=1 use multiprocessing
        split_by_flow: no use in JambaDataset
        """

        if max_process == 0:
            self.logging.info(f"Do not use multiprocessing, start building datasets...")
            for file in os.listdir(self.args.corpus_path):
                self.logging.info(f"Start processing {file}...")
                self.worker(f"{self.args.corpus_path}/{file}", max_process)
                self.logging.info(f"Finish processing {file}...")
            self.logging.info(f"datasets are saved in {self.args.splited_dataset_pt_folder}")

        else:  # multiprocessing
            self.logging.info(f"Starting {max_process} workers for building datasets...")
            all_files = os.listdir(self.args.corpus_path)
            pbar = tqdm(total=len(all_files))
            pbar.set_description('generate dataset files')
            update = lambda *args: pbar.update()
            pool = mp.Pool(max_process)
            for file in all_files:
                pool.apply_async(jamba_worker_wrapper,
                                 (self, f"{self.args.corpus_path}/{file}", max_process),
                                 callback=update)
            pool.close()
            pool.join()
            self.logging.info(f"datasets are saved in {self.args.splited_dataset_pt_folder}")

        # Merge datasets.
        self.logging.info(f"start merging dataset files...")
        merge_dataset(self.args, self.dataset_path, max_process)
        self.logging.info(f"whole dataset file is saved in {self.args.dataset_path}")

    # 在JambaDataset中worker()函数只处理单个文件，多进程在build_and_save()函数中实现
    def worker(self, corpus_file, max_process, proc_id=None, start=None, end=None):
        set_seed(self.seed)
        if max_process > 0:  # multiprocessing
            file_id = os.getpid()
        else:
            file_id = 0
        dataset_writer = open(f"{self.args.splited_dataset_pt_folder}/"
                              f"dataset_{self.args.dataset}_{file_id}.pt", "ab")

        for flow, coarse_semantic_list, fine_semantic_list, direction_list, stop_flag \
                in self.parse_corpora_files(corpus_file):
            #pudb.set_trace()

            if stop_flag:  # 文件读取结束
                break
            # sample: 一个列表[src, segment, coarse_semantic, fine_semantic, tgt_mlm, label]
            for sample in self.build_a_pretrain_sample(
                        flow, coarse_semantic_list, fine_semantic_list, direction_list, self.args.pkt_interval):
                pickle.dump(sample, dataset_writer)

        dataset_writer.close()


    # 生成一个预训练样本, 返回值是列表[src, segment, coarse_semantic, fine_semantic, tgt_mlm, label]
    def build_a_pretrain_sample(self, flow, coarse_semantic_list, fine_semantic_list, direction_list, pkt_interval:int=15):
        cnt = int((len(flow)-1)/pkt_interval) + 1

        for round_idx in range(cnt):
            # 根据概率选择扰动类别
            label_prob = random.random()
            if label_prob < 0.6:
                label = 0  # 正常
            elif label_prob < 0.8:
                label = 1  # 丢包
            elif label_prob < 0.9:
                label = 2  # 重传
            else:
                label = 3  # 乱序

            #pudb.set_trace()

            # 判断是否能构建样本
            if round_idx != cnt-1:
                num_packets = pkt_interval
            else:
                num_packets = len(flow) - round_idx*pkt_interval
                if (num_packets < 2) or (num_packets < 4 and label == 1):
                    continue

            if label == 0:  # 正常类样本：连续、顺序不变
                if num_packets >= 10:
                    random_start = random.randint(0, num_packets - 10)
                    start = round_idx * pkt_interval + random_start
                    stop = start + 10
                else:  # 2~9 个包
                    start = round_idx * pkt_interval
                    stop = round_idx * pkt_interval + num_packets

                src,coarse_semantic,fine_semantic,segment = [],[],[],[]
                src.append(self.vocab.get(CLS_TOKEN))
                coarse_semantic.append(self.vocab.get(CLS_TOKEN))
                fine_semantic.append(self.vocab.get(CLS_TOKEN))
                for pkt_idx in range(start, stop):
                    # 数据包tokens
                    src.extend(flow[pkt_idx])
                    src.append(self.vocab.get(SEP_TOKEN))
                    # segment和position
                    segment.append(len(src))  # [30,60,89]代表第1个segment长度30，第2个segment长度60-30...
                    # coarse semantic和fine semantic
                    coarse_semantic.extend(coarse_semantic_list[pkt_idx])
                    coarse_semantic.append(self.vocab.get(SEP_TOKEN))
                    fine_semantic.extend(fine_semantic_list[pkt_idx])
                    fine_semantic.append(self.vocab.get(SEP_TOKEN))

                src, tgt_mlm = mask_seq_jamba(src, self.tokenizer, self.whole_word_masking, self.span_masking, self.span_geo_prob, self.span_max_length)
                yield [src, segment, coarse_semantic, fine_semantic, tgt_mlm, label]

            elif label == 1:  # 丢包：删除2个包
                if num_packets >= 12:
                    random_start = random.randint(0, num_packets - 12)
                    start = round_idx * pkt_interval + random_start
                    stop = start + 12
                else:  # 4~11 个包
                    start = round_idx * pkt_interval
                    stop = round_idx * pkt_interval + num_packets

                possible_numbers = list(range(start, stop))
                drop_idx1, drop_idx2 = random.sample(possible_numbers, 2)  # 删除的数据包的idx

                src,coarse_semantic,fine_semantic,segment = [],[],[],[]
                src.append(self.vocab.get(CLS_TOKEN))
                coarse_semantic.append(self.vocab.get(CLS_TOKEN))
                fine_semantic.append(self.vocab.get(CLS_TOKEN))
                for pkt_idx in range(start, stop):
                    if (pkt_idx != drop_idx1) and (pkt_idx != drop_idx2):
                        # 数据包tokens
                        src.extend(flow[pkt_idx])
                        src.append(self.vocab.get(SEP_TOKEN))
                        # segment和position
                        segment.append(len(src))  # [30,60,89]代表第1个segment长度30，第2个segment长度60-30...
                        # coarse semantic和fine semantic
                        coarse_semantic.extend(coarse_semantic_list[pkt_idx])
                        coarse_semantic.append(self.vocab.get(SEP_TOKEN))
                        fine_semantic.extend(fine_semantic_list[pkt_idx])
                        fine_semantic.append(self.vocab.get(SEP_TOKEN))

                src, tgt_mlm = mask_seq_jamba(src, self.tokenizer, self.whole_word_masking, self.span_masking, self.span_geo_prob, self.span_max_length)
                yield [src, segment, coarse_semantic, fine_semantic, tgt_mlm, label]

            elif label == 2:  # 重传：插入2个已有包
                if num_packets >= 8:
                    random_start = random.randint(0, num_packets - 8)
                    start = round_idx * pkt_interval + random_start
                    stop = start + 8
                else:  # 2~7 个包
                    start = round_idx * pkt_interval
                    stop = round_idx * pkt_interval + num_packets

                possible_numbers = list(range(start, stop))
                repeat_idx = random.sample(possible_numbers, 2)  # 重传的数据包的idx列表
                repeat_idx.sort()

                repeat_pkts, repeat_coarse, repeat_fine = [],[],[]
                repeat_pkts.append(flow[repeat_idx[0]])
                repeat_pkts.append(flow[repeat_idx[1]])
                repeat_coarse.append(coarse_semantic_list[repeat_idx[0]])
                repeat_coarse.append(coarse_semantic_list[repeat_idx[1]])
                repeat_fine.append(fine_semantic_list[repeat_idx[0]])
                repeat_fine.append(fine_semantic_list[repeat_idx[1]])

                tmp_packets = copy.deepcopy(flow[start:stop])
                tmp_coarse = copy.deepcopy(coarse_semantic_list[start:stop])
                tmp_fine = copy.deepcopy(fine_semantic_list[start:stop])

                insert_idx1 = random.randint(repeat_idx[0]+1-start, stop-start)
                tmp_packets.insert(insert_idx1, repeat_pkts[0])
                tmp_coarse.insert(insert_idx1, repeat_coarse[0])
                tmp_fine.insert(insert_idx1, repeat_fine[0])
                if insert_idx1+start > repeat_idx[1]:
                    insert_idx2 = random.randint(repeat_idx[1]+1-start, stop+1-start)
                else:
                    insert_idx2 = random.randint(repeat_idx[1]+2-start, stop+1-start)
                tmp_packets.insert(insert_idx2, repeat_pkts[1])
                tmp_coarse.insert(insert_idx2, repeat_coarse[1])
                tmp_fine.insert(insert_idx2, repeat_fine[1])

                src,coarse_semantic,fine_semantic,segment = [],[],[],[]
                src.append(self.vocab.get(CLS_TOKEN))
                coarse_semantic.append(self.vocab.get(CLS_TOKEN))
                fine_semantic.append(self.vocab.get(CLS_TOKEN))
                for pkt_idx in range(0, stop-start+2):
                    # 数据包tokens
                    src.extend(tmp_packets[pkt_idx])
                    src.append(self.vocab.get(SEP_TOKEN))
                    # segment和position
                    segment.append(len(src))  # [30,60,89]代表第1个segment长度30，第2个segment长度60-30...
                    # coarse semantic和fine semantic
                    coarse_semantic.extend(tmp_coarse[pkt_idx])
                    coarse_semantic.append(self.vocab.get(SEP_TOKEN))
                    fine_semantic.extend(tmp_fine[pkt_idx])
                    fine_semantic.append(self.vocab.get(SEP_TOKEN))

                src, tgt_mlm = mask_seq_jamba(src, self.tokenizer, self.whole_word_masking, self.span_masking, self.span_geo_prob, self.span_max_length)
                yield [src, segment, coarse_semantic, fine_semantic, tgt_mlm, label]

            elif label == 3:  # 乱序：2次交换前后两个包的顺序，若flow中只有2个数据包，则只交换一次
                if num_packets >= 10:
                    random_start = random.randint(0, num_packets - 10)
                    start = round_idx * pkt_interval + random_start
                    stop = start + 10
                else:  # 2~9 个包
                    start = round_idx * pkt_interval
                    stop = round_idx * pkt_interval + num_packets

                src,coarse_semantic,fine_semantic,segment = [],[],[],[]
                src.append(self.vocab.get(CLS_TOKEN))
                coarse_semantic.append(self.vocab.get(CLS_TOKEN))
                fine_semantic.append(self.vocab.get(CLS_TOKEN))

                if num_packets == 2:
                    # 数据包tokens
                    src.extend(flow[start+1])
                    src.append(self.vocab.get(SEP_TOKEN))
                    # segment和position
                    segment.append(len(src))  # [30,60,89]代表第1个segment长度30，第2个segment长度60-30...
                    # coarse semantic和fine semantic
                    coarse_semantic.extend(coarse_semantic_list[start+1])
                    coarse_semantic.append(self.vocab.get(SEP_TOKEN))
                    fine_semantic.extend(fine_semantic_list[start+1])
                    fine_semantic.append(self.vocab.get(SEP_TOKEN))

                    # 数据包tokens
                    src.extend(flow[start])
                    src.append(self.vocab.get(SEP_TOKEN))
                    # segment和position
                    segment.append(len(src))  # [30,60,89]代表第1个segment长度30，第2个segment长度60-30...
                    # coarse semantic和fine semantic
                    coarse_semantic.extend(coarse_semantic_list[start])
                    coarse_semantic.append(self.vocab.get(SEP_TOKEN))
                    fine_semantic.extend(fine_semantic_list[start])
                    fine_semantic.append(self.vocab.get(SEP_TOKEN))

                else:  # num_packets > 2
                    possible_numbers = list(range(start, stop - 1))
                    switch_idx = random.sample(possible_numbers, 2)  # 交换的数据包的idx列表，第idx数据包与第idx+1数据包交换
                    switch_idx.sort()

                    for pkt_idx in range(start, stop):
                        if (pkt_idx == switch_idx[0]) or (pkt_idx == switch_idx[1]):
                            # 数据包tokens
                            src.extend(flow[pkt_idx+1])
                            src.append(self.vocab.get(SEP_TOKEN))
                            # segment和position
                            segment.append(len(src))  # [30,60,89]代表第1个segment长度30，第2个segment长度60-30...
                            # coarse semantic和fine semantic
                            coarse_semantic.extend(coarse_semantic_list[pkt_idx+1])
                            coarse_semantic.append(self.vocab.get(SEP_TOKEN))
                            fine_semantic.extend(fine_semantic_list[pkt_idx+1])
                            fine_semantic.append(self.vocab.get(SEP_TOKEN))
                        elif (pkt_idx == (switch_idx[0]+1)) or (pkt_idx == (switch_idx[1]+1)):
                            # 数据包tokens
                            src.extend(flow[pkt_idx-1])
                            src.append(self.vocab.get(SEP_TOKEN))
                            # segment和position
                            segment.append(len(src))  # [30,60,89]代表第1个segment长度30，第2个segment长度60-30...
                            # coarse semantic和fine semantic
                            coarse_semantic.extend(coarse_semantic_list[pkt_idx-1])
                            coarse_semantic.append(self.vocab.get(SEP_TOKEN))
                            fine_semantic.extend(fine_semantic_list[pkt_idx-1])
                            fine_semantic.append(self.vocab.get(SEP_TOKEN))
                        else:
                            # 数据包tokens
                            src.extend(flow[pkt_idx])
                            src.append(self.vocab.get(SEP_TOKEN))
                            # segment和position
                            segment.append(len(src))  # [30,60,89]代表第1个segment长度30，第2个segment长度60-30...
                            # coarse semantic和fine semantic
                            coarse_semantic.extend(coarse_semantic_list[pkt_idx])
                            coarse_semantic.append(self.vocab.get(SEP_TOKEN))
                            fine_semantic.extend(fine_semantic_list[pkt_idx])
                            fine_semantic.append(self.vocab.get(SEP_TOKEN))
                src, tgt_mlm = mask_seq_jamba(src, self.tokenizer, self.whole_word_masking, self.span_masking, self.span_geo_prob, self.span_max_length)
                yield [src, segment, coarse_semantic, fine_semantic, tgt_mlm, label]


    # 从corpora bigram/gram file中顺序读入每条流，遍历使用，每次返回5个值
    # 流：二维数组, 粗粒度语义：二维数组, 细粒度语义：二维数组, 方向：一维数组, 文件结束标值：bool
    def parse_corpora_files(self, file_path):
        flow = []  # 流列表，里面包含这条流的所有数据包
        coarse_semantic_list = []  # 粗粒度语义编码列表，保存这条流中所有数据包的粗粒度语义编码
        fine_semantic_list = []  # 细粒度语义编码列表，保存这条流中所有数据包的细粒度语义编码
        direction_list = []  # 保存这条流中所有数据包的方向
        direction_flag = 1  # 初始方向为1
        stop_flag = False
        with open(file_path, 'r') as f:
            while True:
                line = f.readline()

                # 文件读取结束
                if not line:
                    stop_flag = True
                    yield flow, coarse_semantic_list, fine_semantic_list, direction_list, stop_flag

                packet = line.strip()

                # 新的一条流开始
                if packet.startswith("||"):
                    # 上一条流已读完，返回它
                    if flow != []:
                        yield flow, coarse_semantic_list, fine_semantic_list, direction_list, stop_flag
                    flow = []
                    coarse_semantic_list = []
                    fine_semantic_list = []
                    direction_list = []
                    direction_flag = 1  # 每条新流开始默认方向1

                    packet = packet[2:]
                    packet_tokens = self.tokenizer.tokenize(packet)
                    flow.append(self.tokenizer.convert_tokens_to_ids(packet_tokens))
                    coarse_tokens, fine_tokens = self.semantic_embedding(packet_tokens)
                    coarse_semantic_list.append(coarse_tokens)
                    fine_semantic_list.append(fine_tokens)
                    direction_list.append(direction_flag)
                elif packet == "":
                    # 空行，表示方向切换
                    direction_flag = 2 if direction_flag == 1 else 1
                else:
                    # 正常数据包行
                    packet_tokens = self.tokenizer.tokenize(packet)
                    flow.append(self.tokenizer.convert_tokens_to_ids(packet_tokens))
                    coarse_tokens, fine_tokens = self.semantic_embedding(packet_tokens)
                    coarse_semantic_list.append(coarse_tokens)
                    fine_semantic_list.append(fine_tokens)
                    direction_list.append(direction_flag)


    # 生成coarse semantic embedding和fine semantic embedding
    def semantic_embedding(self, tokens: List[str]):
        tokens_len = len(tokens)
        coarse_tokens = [0] * tokens_len
        fine_tokens = [12] * tokens_len  # 11个字段之外的packet字段赋值12

        # Step 1: IP header
        if tokens_len < 10:
            return None, None
        ihl_token = tokens[0]
        try:
            ihl = int(ihl_token[1], 16)
        except (IndexError, ValueError):
            ihl = 5  # 默认IHL = 5，即20字节
        ip_header_len_tokens = (ihl << 2) >> 1  # ihl * 4 / 2
        end_ip = ip_header_len_tokens

        coarse_tokens[0:end_ip] = [1] * end_ip
        fine_tokens[1] = 1
        fine_tokens[4] = 2
        fine_tokens[6:8] = [3]*2
        fine_tokens[8:10] = [4]*2

        # Step 2: L4 protocol header
        protocol = int(tokens[4][2:], 16)  # 获取协议十进制编码tcp：6，udp：17
        l4_start = end_ip
        if protocol == 6:  # TCP
            if tokens_len < 20:
                return None, None
            l4_label = 2
            offset_token = tokens[l4_start]
            data_offset = int(offset_token[0], 16)
            try:
                l4_len_tokens = (data_offset << 2) >> 1  # data_offset * 4 / 2
            except (IndexError, ValueError):
                l4_len_tokens = 10

            fine_tokens[0 + l4_start] = 5
            fine_tokens[1 + l4_start] = 6
            fine_tokens[2 + l4_start:4 + l4_start] = [7] * 2
            fine_tokens[4 + l4_start:6 + l4_start] = [8] * 2
            fine_tokens[6 + l4_start] = 9
        elif protocol == 17:  # UDP
            if tokens_len < 14:
                return None, None
            l4_label = 3
            l4_len_tokens = 4  # 8 bytes / 2
            fine_tokens[0 + l4_start] = 10
            fine_tokens[1 + l4_start] = 11
        else:
            if tokens_len < 14:
                return None, None
            l4_label = 4
            l4_len_tokens = 4  # 8 bytes / 2

        end_l4 = l4_start + l4_len_tokens
        coarse_tokens[l4_start:end_l4] = [l4_label] * (end_l4-l4_start)

        # Step 3: Payload
        coarse_tokens[end_l4:tokens_len] = [5] * (tokens_len - end_l4)

        coarse_tokens = [ctx+2 for ctx in coarse_tokens]  # 从3开始编码，给PAD SEP CLS保留编码号
        fine_tokens = [ftx+2 for ftx in fine_tokens]

        return coarse_tokens, fine_tokens


    def build_instances(self, all_documents, flow_proto):
        assert len(all_documents) == len(flow_proto)
        instances = []
        for _ in range(self.dup_factor):
            for doc_index in range(len(all_documents)):
                # if doc_index%500==0:
                #     print(doc_index)
                instances.extend(self.create_ins_from_doc(all_documents, doc_index, flow_proto))
        return instances

    def create_ins_from_doc(self, all_documents, document_index, flow_proto):
        #pudb.set_trace()

        document = all_documents[document_index]
        max_num_tokens = self.seq_length - 3
        target_seq_length = max_num_tokens
        if random.random() < self.short_seq_prob:
            target_seq_length = random.randint(2, max_num_tokens)
        instances = []
        i = 0
        while i < len(document):
            rnd = random.random()
            if i == len(document) - 1 or rnd < 0.4:
                a_end = 1
                if len(document[i]) >= 2:
                    a_end = random.randint(1, len(document[i]) - 1)

                tokens_a = []  # seg 1
                for j in range(a_end):
                    tokens_a.extend(document[i][j])

                tokens_b = []  ##seg 2
                for j in range(a_end, len(document[i])):
                    tokens_b.extend(document[i][j])

                truncate_seq_pair(tokens_a, tokens_b, max_num_tokens)
                if random.random() < 0.5:  # A1 A2
                    label = 0  #
                    src = []
                    src.append(self.vocab.get(CLS_TOKEN))
                    src.extend(tokens_a)
                    src.append(self.vocab.get(SEP_TOKEN))
                    seg_pos = [len(src)]
                    src.extend(tokens_b)
                    src.append(self.vocab.get(SEP_TOKEN))
                    seg_pos.append(len(src))
                else:  # A2 A1
                    label = 1
                    src = []
                    src.append(self.vocab.get(CLS_TOKEN))
                    src.extend(tokens_b)
                    src.append(self.vocab.get(SEP_TOKEN))
                    seg_pos = [len(src)]
                    src.extend(tokens_a)
                    src.append(self.vocab.get(SEP_TOKEN))
                    seg_pos.append(len(src))
            elif rnd < 0.6:
                tokens_a = []
                for j in range(len(document[i])):
                    tokens_a.extend(document[i][j])
                next_burst_max_length = target_seq_length - len(tokens_a)

                for _ in range(20):
                    random_document_index = random.randint(0, len(all_documents) - 1)
                    if random_document_index != document_index:
                        break

                random_document = all_documents[random_document_index]
                random_start = random.randint(0, len(random_document) - 1)
                burst_ind_end = random_start + 1  # len(random_document)
                tokens_b = []
                for burst_ind in range(random_start, burst_ind_end):
                    for j in range(len(random_document[burst_ind])):
                        tokens_b.extend(random_document[burst_ind][j])
                        if len(tokens_b) >= next_burst_max_length:
                            break
                truncate_seq_pair(tokens_a, tokens_b, max_num_tokens)
                if random_document_index == document_index:
                    if random_start >= i:
                        label = 3
                    else:
                        label = 4
                else:
                    label = 2  # A1A2 Z1Z2
                src = []
                src.append(self.vocab.get(CLS_TOKEN))
                src.extend(tokens_a)
                src.append(self.vocab.get(SEP_TOKEN))
                seg_pos = [len(src)]
                src.extend(tokens_b)
                src.append(self.vocab.get(SEP_TOKEN))
                seg_pos.append(len(src))
            else:
                tokens_a = []
                for j in range(len(document[i])):
                    tokens_a.extend(document[i][j])
                i += 1
                tokens_b = []
                for j in range(len(document[i])):
                    tokens_b.extend(document[i][j])
                truncate_seq_pair(tokens_a, tokens_b, max_num_tokens)  # 长度不能超过512-3=509
                if random.random() < 0.5:  # A1A2 B1B2
                    label = 3
                    src = []
                    src.append(self.vocab.get(CLS_TOKEN))
                    src.extend(tokens_a)
                    src.append(self.vocab.get(SEP_TOKEN))
                    seg_pos = [len(src)]
                    src.extend(tokens_b)
                    src.append(self.vocab.get(SEP_TOKEN))
                    seg_pos.append(len(src))
                else:  # B1B2 A1A2
                    label = 4
                    src = []
                    src.append(self.vocab.get(CLS_TOKEN))
                    src.extend(tokens_b)
                    src.append(self.vocab.get(SEP_TOKEN))
                    seg_pos = [len(src)]
                    src.extend(tokens_a)
                    src.append(self.vocab.get(SEP_TOKEN))
                    seg_pos.append(len(src))
            while len(src) != self.seq_length:
                src.append(PAD_ID)
            if not self.dynamic_masking:
                src, tgt_mlm = mask_seq(src, self.tokenizer, self.whole_word_masking, self.span_masking,
                                        self.span_geo_prob, self.span_max_length)
                instance = (src, tgt_mlm, label, seg_pos)
            else:
                instance = (src, label, seg_pos)
            instance += (flow_proto[document_index],)
            instances.append(instance)
            i += 1

        return instances


class JambaDataLoader(DataLoader):
    def __iter__(self):
        while True:
            while self._empty():
                self._fill_buf()
            if self.start + self.batch_size >= self.end:
                instances = self.buffer[self.start:]
            else:
                instances = self.buffer[self.start: self.start + self.batch_size]

            self.start += self.batch_size

            src = []  # 样本
            tgt_mlm = []  # mask任务标签
            tgt_pp = []  # packets预测任务标签
            seg = []  # segment
            coarse = []  # 粗粒度语义
            fine = []  # 细粒度语义
            sample_len = []  # 样本实际长度

            masked_words_num = 0

            for ins in instances:
                #pudb.set_trace()

                if len(ins) == 6:  # no dynamic masking
                    sample_len.append(len(ins[0]))
                    src.append(ins[0]+[PAD_ID]*(self.args.max_seq_length-sample_len[-1]))  # 样本
                    masked_words_num += len(ins[4])  # mask tokems数量
                    tgt_mlm.append([0] * self.args.max_seq_length)  # mask任务标签
                    for mask in ins[4]:
                        tgt_mlm[-1][mask[0]] = mask[1]
                    coarse.append(ins[2]+[PAD_ID]*(self.args.max_seq_length-sample_len[-1]))  # 粗粒度语义
                    fine.append(ins[3]+[PAD_ID]*(self.args.max_seq_length-sample_len[-1]))  # 细粒度语义
                    tgt_pp.append(ins[5])  # packets预测任务标签
                    tmp_seg = []
                    tmp_changeidx = 0
                    for tmp_idx, tmp_item in enumerate(ins[1]):
                        tmp_seg.extend([tmp_idx+1]*(tmp_item-tmp_changeidx))
                        tmp_changeidx = tmp_item
                    seg.append(tmp_seg+[PAD_ID]*(self.args.max_seq_length-sample_len[-1]))  # segment

                else:  # use dynamic masking
                    sample_len.append(len(ins[0]))
                    src_single, tgt_mlm_single = mask_seq_jamba(ins[0], self.tokenizer,
                                                                self.whole_word_masking,
                                                                self.span_masking,
                                                                self.span_geo_prob,
                                                                self.span_max_length)
                    src.append(src_single+[PAD_ID]*(self.args.max_seq_length-sample_len[-1]))  # 样本
                    masked_words_num += len(tgt_mlm_single)  # mask tokems数量
                    tgt_mlm.append([0] * self.args.max_seq_length)  # mask任务标签
                    for mask in tgt_mlm_single:
                        tgt_mlm[-1][mask[0]] = mask[1]
                    coarse.append(ins[2]+[PAD_ID]*(self.args.max_seq_length-sample_len[-1]))  # 粗粒度语义
                    fine.append(ins[3]+[PAD_ID]*(self.args.max_seq_length-sample_len[-1]))  # 细粒度语义
                    tgt_pp.append(ins[4])  # packets预测任务标签
                    tmp_seg = []
                    tmp_changeidx = 0
                    for tmp_idx, tmp_item in enumerate(ins[1]):
                        tmp_seg.extend([tmp_idx+1]*(tmp_item-tmp_changeidx))
                        tmp_changeidx = tmp_item
                    seg.append(tmp_seg+[PAD_ID]*(self.args.max_seq_length-sample_len[-1]))  # segment

            if masked_words_num == 0:
                continue

            yield torch.LongTensor(src), \
                torch.LongTensor(seg), \
                torch.LongTensor(coarse), \
                torch.LongTensor(fine), \
                torch.LongTensor(tgt_mlm), \
                torch.LongTensor(tgt_pp), \
                torch.LongTensor(sample_len)


class JambaMaskOnlyDataLoader(DataLoader):
    def __iter__(self):
        while True:
            while self._empty():
                self._fill_buf()
            if self.start + self.batch_size >= self.end:
                instances = self.buffer[self.start:]
            else:
                instances = self.buffer[self.start: self.start + self.batch_size]

            self.start += self.batch_size

            src = []  # 样本
            tgt_mlm = []  # mask任务标签
            seg = []  # segment
            coarse = []  # 粗粒度语义
            fine = []  # 细粒度语义
            sample_len = []  # 样本实际长度

            masked_words_num = 0

            for ins in instances:
                #pudb.set_trace()

                if len(ins) == 5:  # no dynamic masking
                    sample_len.append(len(ins[0]))
                    src.append(ins[0]+[PAD_ID]*(self.args.max_seq_length-sample_len[-1]))  # 样本
                    masked_words_num += len(ins[4])  # mask tokems数量
                    tgt_mlm.append([0] * self.args.max_seq_length)  # mask任务标签
                    for mask in ins[4]:
                        tgt_mlm[-1][mask[0]] = mask[1]
                    coarse.append(ins[2]+[PAD_ID]*(self.args.max_seq_length-sample_len[-1]))  # 粗粒度语义
                    fine.append(ins[3]+[PAD_ID]*(self.args.max_seq_length-sample_len[-1]))  # 细粒度语义
                    tmp_seg = []
                    tmp_changeidx = 0
                    for tmp_idx, tmp_item in enumerate(ins[1]):
                        tmp_seg.extend([tmp_idx+1]*(tmp_item-tmp_changeidx))
                        tmp_changeidx = tmp_item
                    seg.append(tmp_seg+[PAD_ID]*(self.args.max_seq_length-sample_len[-1]))  # segment

                else:  # use dynamic masking
                    sample_len.append(len(ins[0]))
                    src_single, tgt_mlm_single = mask_seq_jamba(ins[0], self.tokenizer,
                                                                self.whole_word_masking,
                                                                self.span_masking,
                                                                self.span_geo_prob,
                                                                self.span_max_length)
                    src.append(src_single+[PAD_ID]*(self.args.max_seq_length-sample_len[-1]))  # 样本
                    masked_words_num += len(tgt_mlm_single)  # mask tokems数量
                    tgt_mlm.append([0] * self.args.max_seq_length)  # mask任务标签
                    for mask in tgt_mlm_single:
                        tgt_mlm[-1][mask[0]] = mask[1]
                    coarse.append(ins[2]+[PAD_ID]*(self.args.max_seq_length-sample_len[-1]))  # 粗粒度语义
                    fine.append(ins[3]+[PAD_ID]*(self.args.max_seq_length-sample_len[-1]))  # 细粒度语义
                    tmp_seg = []
                    tmp_changeidx = 0
                    for tmp_idx, tmp_item in enumerate(ins[1]):
                        tmp_seg.extend([tmp_idx+1]*(tmp_item-tmp_changeidx))
                        tmp_changeidx = tmp_item
                    seg.append(tmp_seg+[PAD_ID]*(self.args.max_seq_length-sample_len[-1]))  # segment

            if masked_words_num == 0:
                continue

            yield torch.LongTensor(src), \
                torch.LongTensor(seg), \
                torch.LongTensor(coarse), \
                torch.LongTensor(fine), \
                torch.LongTensor(tgt_mlm), \
                torch.LongTensor(sample_len)


class JambaMaskOnlyDirectionDataLoader(DataLoader):
    def __iter__(self):
        while True:
            while self._empty():
                self._fill_buf()
            if self.start + self.batch_size >= self.end:
                instances = self.buffer[self.start:]
            else:
                instances = self.buffer[self.start: self.start + self.batch_size]

            self.start += self.batch_size

            src = []  # 样本
            tgt_mlm = []  # mask任务标签
            seg = []  # segment
            coarse = []  # 粗粒度语义
            fine = []  # 细粒度语义
            sample_len = []  # 样本实际长度
            byte_transmission_idx = []  # byte和transmission的截止idx,就是没有扩展的seg的最后两位

            masked_words_num = 0

            for ins in instances:
                #pudb.set_trace()

                if len(ins) == 5:  # no dynamic masking
                    sample_len.append(len(ins[0]))
                    byte_transmission_idx.append([ins[1][-2], ins[1][-1]])
                    src.append(ins[0]+[PAD_ID]*(self.args.max_seq_length-sample_len[-1]))  # 样本
                    masked_words_num += len(ins[4])  # mask tokems数量
                    tgt_mlm.append([0] * self.args.max_seq_length)  # mask任务标签
                    for mask in ins[4]:
                        tgt_mlm[-1][mask[0]] = mask[1]
                    coarse.append(ins[2]+[PAD_ID]*(self.args.max_seq_length-sample_len[-1]))  # 粗粒度语义
                    fine.append(ins[3]+[PAD_ID]*(self.args.max_seq_length-sample_len[-1]))  # 细粒度语义
                    tmp_seg = []
                    tmp_changeidx = 0
                    for tmp_idx, tmp_item in enumerate(ins[1]):  # seg
                        if tmp_idx != len(ins[1])-1:
                            tmp_seg.extend([tmp_idx+1]*(tmp_item-tmp_changeidx))
                            tmp_changeidx = tmp_item
                        else:
                            tmp_seg.extend([6] * (tmp_item - tmp_changeidx))  # 方向长度token的seg编码为6
                    seg.append(tmp_seg+[PAD_ID]*(self.args.max_seq_length-sample_len[-1]))  # segment

                else:  # use dynamic masking
                    sample_len.append(len(ins[0]))
                    src_single, tgt_mlm_single = mask_seq_jamba(ins[0], self.tokenizer,
                                                                self.whole_word_masking,
                                                                self.span_masking,
                                                                self.span_geo_prob,
                                                                self.span_max_length)
                    src.append(src_single+[PAD_ID]*(self.args.max_seq_length-sample_len[-1]))  # 样本
                    masked_words_num += len(tgt_mlm_single)  # mask tokems数量
                    tgt_mlm.append([0] * self.args.max_seq_length)  # mask任务标签
                    for mask in tgt_mlm_single:
                        tgt_mlm[-1][mask[0]] = mask[1]
                    coarse.append(ins[2]+[PAD_ID]*(self.args.max_seq_length-sample_len[-1]))  # 粗粒度语义
                    fine.append(ins[3]+[PAD_ID]*(self.args.max_seq_length-sample_len[-1]))  # 细粒度语义
                    tmp_seg = []
                    tmp_changeidx = 0
                    for tmp_idx, tmp_item in enumerate(ins[1]):
                        tmp_seg.extend([tmp_idx+1]*(tmp_item-tmp_changeidx))
                        tmp_changeidx = tmp_item
                    seg.append(tmp_seg+[PAD_ID]*(self.args.max_seq_length-sample_len[-1]))  # segment

            if masked_words_num == 0:
                continue

            yield torch.LongTensor(src), \
                torch.LongTensor(seg), \
                torch.LongTensor(coarse), \
                torch.LongTensor(fine), \
                torch.LongTensor(tgt_mlm), \
                torch.LongTensor(sample_len), \
                torch.LongTensor(byte_transmission_idx)  # byte和transmission的截止idx


class BertDataset(Dataset):
    """
    Construct dataset for MLM and NSP tasks from the given corpus.
    Each document consists of multiple sentences,
    and each sentence occupies a single line.
    Documents in corpus must be separated by empty lines.
    """

    def __init__(self, args, vocab, tokenizer):
        super(BertDataset, self).__init__(args, vocab, tokenizer)
        self.args = args
        self.short_seq_prob = args.short_seq_prob

    def worker(self, proc_id, start, end):
        #pudb.set_trace()

        print(f"Worker {proc_id} is building dataset ... ")
        print(start, end)
        set_seed(self.seed)
        flow_set = []  # flow_buffer表示多条流的集合: 4维列表
        flow_proto = []
        a_flow = []  # docs_buffer表示1条流: 3维列表
        a_burst = []  # a_burst表示1个burst: 2维列表
        # packet表示1个数据包: 1维列表
        pos = 0
        dataset_writer = open(
            f"{self.args.root}/pretrain_datasets/{self.args.dataset}/tmp/dataset-tmp-" + str(proc_id) + ".pt", "wb")
        with open(self.corpus_path, mode="r", encoding="utf-8") as f:
            try:
                # with open(self.corpus_path[:-4]+"_extra.txt", mode="r", encoding="utf-8") as fe:
                while pos < start:
                    f.readline()
                    pos += 1
                while True:
                    line = f.readline()

                    if pos == start and line[:2] != "||":
                        print("not flow start...")
                    pos += 1

                    if pos > end:
                        if len(a_flow) >= 1:
                            flow_set.append(a_flow)
                        if len(flow_set) > 0:
                            try:
                                instances = self.build_instances(flow_set, flow_proto)
                            except Exception as e:
                                print("has error2...", len(flow_set), len(flow_proto), e)
                            for instance in instances:
                                pickle.dump(instance, dataset_writer)
                        break

                    # line读入的是 \n(空行)
                    if not line.strip():
                        if len(a_burst) >= 1:
                            a_flow.append(a_burst)
                        a_burst = []
                        continue

                    # line是新流
                    if line[:2] == "||" or not line:
                        if len(a_flow) >= 1:
                            flow_set.append(a_flow)
                        a_flow = []

                        flow_set_size = 0
                        for d in flow_set:  #　d表示1条流
                            # flow_set_size记录的是多条流(flow_set)中burst的数量
                            flow_set_size += len(d)

                        if flow_set_size > self.flows_size or pos >= end:
                            print("Worker %d is building instances ... " % proc_id, len(flow_set), len(flow_proto))
                            try:
                                instances = self.build_instances(flow_set, flow_proto)
                            except:
                                print("has error1...")
                            print("Worker {} has {} instances. ".format(proc_id, len(instances)))
                            # Save instances.
                            for instance in instances:
                                pickle.dump(instance, dataset_writer)
                            flow_set = []
                            flow_proto = []

                        # 该进程处理结束了
                        if pos >= end or not line:
                            break

                        line = line[2:]
                        if line[0] == "4":  # 只处理ipv4
                            if "bigram" in self.corpus_path:  # corpora_bigram.txt
                                if line[42:44] == "06":  # tcp 十进制6
                                    flow_proto.append(0)
                                elif line[42:44] == "11":  # udp 十进制17
                                    flow_proto.append(1)
                                else:
                                    print("not tcp or udp, ", line[42:44])
                            else:  # corpora.txt
                                if line[22:24] == "06":  # tcp
                                    flow_proto.append(0)
                                elif line[22:24] == "11":  # udp
                                    flow_proto.append(1)
                        else:
                            print("find Ipv6!!")

                    # 一个数据包的tokens转换成vocab文件中的ids
                    # eg. ['4500', '0041', 'e195', '4000']变成[49, 661, 20727, 76]
                    packet = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(line))
                    if len(packet) > 0:
                        a_burst.append(packet)

            except:
                print("has error...")
        print("Worker %d finished... " % proc_id)
        dataset_writer.close()

    def build_instances(self, all_documents):
        instances = []
        for _ in range(self.dup_factor):
            for doc_index in range(len(all_documents)):
                instances.extend(self.create_ins_from_doc(all_documents, doc_index))
        return instances

    def create_ins_from_doc(self, all_documents, document_index):
        document = all_documents[document_index]
        max_num_tokens = self.seq_length - 3
        target_seq_length = max_num_tokens
        if random.random() < self.short_seq_prob:
            target_seq_length = random.randint(2, max_num_tokens)
        instances = []
        current_chunk = []
        current_length = 0
        i = 0
        while i < len(document):
            segment = document[i]
            current_chunk.append(segment)
            current_length += len(segment)
            if i == len(document) - 1 or current_length >= target_seq_length:
                if current_chunk:
                    a_end = 1
                    if len(current_chunk) >= 2: 
                        a_end = random.randint(1, len(current_chunk) - 1)

                    tokens_a = [] #seg 1
                    for j in range(a_end):
                        tokens_a.extend(current_chunk[j])

                    tokens_b = [] ##seg 2
                    is_random_next = 0

                    if len(current_chunk) == 1 or random.random() < 0.5: 
                        is_random_next = 1
                        target_b_length = target_seq_length - len(tokens_a)

                        for _ in range(10):
                            random_document_index = random.randint(0, len(all_documents) - 1)
                            if random_document_index != document_index:
                                break

                        random_document = all_documents[random_document_index]
                        random_start = random.randint(0, len(random_document) - 1)
                        for j in range(random_start, len(random_document)):
                            tokens_b.extend(random_document[j])
                            if len(tokens_b) >= target_b_length:
                                break

                        num_unused_segments = len(current_chunk) - a_end 
                        i -= num_unused_segments

                    else:
                        is_random_next = 0
                        for j in range(a_end, len(current_chunk)):
                            tokens_b.extend(current_chunk[j])

                    truncate_seq_pair(tokens_a, tokens_b, max_num_tokens)
  
                    src = []
                    src.append(self.vocab.get(CLS_TOKEN))
                    src.extend(tokens_a)
                    src.append(self.vocab.get(SEP_TOKEN))
                    seg_pos = [len(src)]
                    src.extend(tokens_b)
                    src.append(self.vocab.get(SEP_TOKEN))
                    seg_pos.append(len(src))

                    while len(src) != self.seq_length:
                        src.append(PAD_ID)

                    if not self.dynamic_masking:
                        src, tgt_mlm = mask_seq(src, self.tokenizer, self.whole_word_masking, self.span_masking, self.span_geo_prob, self.span_max_length)
                        instance = (src, tgt_mlm, is_random_next, seg_pos)
                    else:
                        instance = (src, is_random_next, seg_pos)

                    instances.append(instance)
                current_chunk = []
                current_length = 0
            i += 1
        return instances


class BertDataLoader(DataLoader):
    def __iter__(self):
        while True:
            while self._empty():
                self._fill_buf()
            if self.start + self.batch_size >= self.end:
                instances = self.buffer[self.start:]
            else:
                instances = self.buffer[self.start: self.start + self.batch_size]

            self.start += self.batch_size

            src = []
            tgt_mlm = []
            is_next = []
            seg = []

            masked_words_num = 0

            for ins in instances:
                if len(ins) == 4:
                    src.append(ins[0])
                    masked_words_num += len(ins[1])
                    tgt_mlm.append([0] * len(ins[0]))
                    for mask in ins[1]:
                        tgt_mlm[-1][mask[0]] = mask[1]
                    is_next.append(ins[2])
                    seg.append([1] * ins[3][0] + [2] * (ins[3][1] - ins[3][0]) + [PAD_ID] * (len(ins[0]) - ins[3][1]))
                else:
                    src_single, tgt_mlm_single = mask_seq(ins[0], self.tokenizer, self.whole_word_masking, self.span_masking, self.span_geo_prob, self.span_max_length)
                    masked_words_num += len(tgt_mlm_single)
                    src.append(src_single)
                    tgt_mlm.append([0] * len(ins[0]))
                    for mask in tgt_mlm_single:
                        tgt_mlm[-1][mask[0]] = mask[1]
                    is_next.append(ins[1])
                    seg.append([1] * ins[2][0] + [2] * (ins[2][1] - ins[2][0]) + [PAD_ID] * (len(ins[0]) - ins[2][1]))

            if masked_words_num == 0:
                continue

            yield torch.LongTensor(src), \
                torch.LongTensor(tgt_mlm), \
                torch.LongTensor(is_next), \
                torch.LongTensor(seg)


class BertFlowDataset(Dataset):
    """
    Construct dataset for MLM and MIX tasks from the given corpus.
    Each document consists of multiple paragraphs,
    Each paragraph consists of multiple sentences,
    and each sentence occupies a single line.
    Paragraphs in corpus must be separated by empty lines.
    Documents in corpus must be separated by empty lines.
    """

    def __init__(self, args, vocab, tokenizer):
        super(BertFlowDataset, self).__init__(args, vocab, tokenizer)
        self.args = args
        self.short_seq_prob = args.short_seq_prob

    def worker(self, proc_id, start, end):
        print(f"Worker {proc_id} is building dataset ... ")
        print(start,end)
        set_seed(self.seed)
        flow_buffer = []
        flow_proto = []
        docs_buffer = []
        document = []
        pos = 0
        dataset_writer = open(f"{self.args.root}/pretrain_datasets/{self.args.dataset}/tmp/dataset-tmp-" + str(proc_id) + ".pt", "wb")
        with open(self.corpus_path, mode="r", encoding="utf-8") as f:
            try:
                #with open(self.corpus_path[:-4]+"_extra.txt", mode="r", encoding="utf-8") as fe:
                while pos < start:
                    f.readline()
                    pos += 1
                while True:
                    line = f.readline()
                    if pos==start and line[:2] != "||":
                        print("not flow start...")
                    pos += 1
                    if pos > end:
                        if len(docs_buffer) >= 1:
                            flow_buffer.append(docs_buffer)
                            
                        if len(flow_buffer) > 0:
                            try:
                                instances = self.build_instances(flow_buffer,flow_proto) 
                            except Exception as e:
                                print("has error2...", len(flow_buffer), len(flow_proto), e)
                            for instance in instances:
                                pickle.dump(instance, dataset_writer)
                        break

                    if not line.strip(): 
                        if len(document) >= 1:
                            docs_buffer.append(document)
                        document = []
                        continue
                    if line[:2] == "||" or not line: 
                        if len(docs_buffer) >= 1:
                            flow_buffer.append(docs_buffer)
                        docs_buffer = []
                    
                        flow_buffer_size = 0
                        for d in flow_buffer:
                            flow_buffer_size += len(d)
                        if flow_buffer_size > self.docs_buffer_size or pos>=end: 
                            print("Worker %d is building instances ... " % proc_id, len(flow_buffer),len(flow_proto))
                            try:
                                instances = self.build_instances(flow_buffer,flow_proto)
                            except:
                                print("has error1...")
                            print("Worker {} has {} instances. ".format(proc_id,len(instances)))
                            # Save instances.
                            for instance in instances: 
                                pickle.dump(instance, dataset_writer)
                            flow_buffer = []
                            flow_proto = []
                        if pos>=end or not line:
                            break
                        line = line[2:]
                        if line[0]=="4":
                            if "bigram" in self.corpus_path:  # corpora_bigram.txt
                                if line[42:44] == "06":  # tcp 十进制6
                                    flow_proto.append(0)
                                elif line[42:44] == "11":  # udp 十进制17
                                    flow_proto.append(1)
                                else:
                                    print("not tcp or udp, ",line[42:44])
                            else:  # corpora.txt
                                if line[22:24] == "06":
                                    flow_proto.append(0)
                                elif line[22:24] == "11":
                                    flow_proto.append(1)
                        else:
                            print("find Ipv6!!")

                    sentence = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(line)) 
                    if len(sentence) > 0:
                        document.append(sentence)

            except:
                print("has error...")
        print("Worker %d finished... " % proc_id)
        dataset_writer.close()

    def build_instances(self, all_documents, flow_proto):
        assert len(all_documents) == len(flow_proto)
        instances = []
        for _ in range(self.dup_factor):
            for doc_index in range(len(all_documents)):
                # if doc_index%500==0:
                #     print(doc_index)
                instances.extend(self.create_ins_from_doc(all_documents, doc_index, flow_proto))
        return instances
    def create_ins_from_doc(self, all_documents,document_index, flow_proto):
        document = all_documents[document_index] 
        max_num_tokens = self.seq_length - 3
        target_seq_length = max_num_tokens
        if random.random() < self.short_seq_prob: 
            target_seq_length = random.randint(2, max_num_tokens)
        instances = []
        i=0
        while i < len(document): 
            rnd = random.random()
            if i==len(document)-1 or rnd < 0.4: 
                a_end = 1
                if len(document[i]) >= 2: 
                    a_end = random.randint(1, len(document[i]) - 1)

                tokens_a = [] #seg 1
                for j in range(a_end):
                    tokens_a.extend(document[i][j])

                tokens_b = [] ##seg 2
                for j in range(a_end, len(document[i])):
                    tokens_b.extend(document[i][j])
                
                truncate_seq_pair(tokens_a, tokens_b, max_num_tokens)
                if random.random()<0.5:  #A1 A2
                    label = 0 #
                    src = []
                    src.append(self.vocab.get(CLS_TOKEN))
                    src.extend(tokens_a)
                    src.append(self.vocab.get(SEP_TOKEN))
                    seg_pos = [len(src)]
                    src.extend(tokens_b)
                    src.append(self.vocab.get(SEP_TOKEN))
                    seg_pos.append(len(src))
                else: #A2 A1
                    label = 1
                    src = []
                    src.append(self.vocab.get(CLS_TOKEN))
                    src.extend(tokens_b)
                    src.append(self.vocab.get(SEP_TOKEN))
                    seg_pos = [len(src)]
                    src.extend(tokens_a)
                    src.append(self.vocab.get(SEP_TOKEN))
                    seg_pos.append(len(src))
            elif rnd < 0.6: 
                tokens_a = []
                for j in range(len(document[i])):
                    tokens_a.extend(document[i][j])
                next_burst_max_length = target_seq_length - len(tokens_a)

                for _ in range(20):
                    random_document_index = random.randint(0, len(all_documents) - 1)
                    if random_document_index != document_index:
                        break

                random_document = all_documents[random_document_index] 
                random_start = random.randint(0, len(random_document) - 1)
                burst_ind_end = random_start+1 #len(random_document) 
                tokens_b = []
                for burst_ind in range(random_start, burst_ind_end): 
                    for j in range(len(random_document[burst_ind])): 
                        tokens_b.extend(random_document[burst_ind][j])
                        if len(tokens_b) >= next_burst_max_length:
                            break
                truncate_seq_pair(tokens_a, tokens_b, max_num_tokens)
                if random_document_index == document_index:
                    if random_start>=i:
                        label = 3 
                    else:
                        label = 4
                else:
                    label = 2 # A1A2 Z1Z2
                src = []
                src.append(self.vocab.get(CLS_TOKEN))
                src.extend(tokens_a)
                src.append(self.vocab.get(SEP_TOKEN))
                seg_pos = [len(src)]
                src.extend(tokens_b)
                src.append(self.vocab.get(SEP_TOKEN))
                seg_pos.append(len(src))
            else:
                tokens_a = []
                for j in range(len(document[i])):
                    tokens_a.extend(document[i][j])
                i+=1
                tokens_b = []
                for j in range(len(document[i])):
                    tokens_b.extend(document[i][j])
                truncate_seq_pair(tokens_a, tokens_b, max_num_tokens)
                if random.random()<0.5:  #A1A2 B1B2
                    label = 3 
                    src = []
                    src.append(self.vocab.get(CLS_TOKEN))
                    src.extend(tokens_a)
                    src.append(self.vocab.get(SEP_TOKEN))
                    seg_pos = [len(src)]
                    src.extend(tokens_b)
                    src.append(self.vocab.get(SEP_TOKEN))
                    seg_pos.append(len(src))
                else: #B1B2 A1A2
                    label = 4
                    src = []
                    src.append(self.vocab.get(CLS_TOKEN))
                    src.extend(tokens_b)
                    src.append(self.vocab.get(SEP_TOKEN))
                    seg_pos = [len(src)]
                    src.extend(tokens_a)
                    src.append(self.vocab.get(SEP_TOKEN))
                    seg_pos.append(len(src))
            while len(src) != self.seq_length:
                src.append(PAD_ID)
            if not self.dynamic_masking:
                src, tgt_mlm = mask_seq(src, self.tokenizer, self.whole_word_masking, self.span_masking, self.span_geo_prob, self.span_max_length)
                instance = (src, tgt_mlm, label, seg_pos)
            else:
                instance = (src, label, seg_pos)
            instance += (flow_proto[document_index],)
            instances.append(instance)
            i+=1 
    
        return instances


class BertFlowDataLoader(DataLoader):
    def __iter__(self):
        while True:
            while self._empty():
                self._fill_buf()
            if self.start + self.batch_size >= self.end:
                instances = self.buffer[self.start:]
            else:
                instances = self.buffer[self.start: self.start + self.batch_size]

            self.start += self.batch_size

            src = []
            tgt_mlm = []
            is_next = []
            seg = []
            proto = []

            masked_words_num = 0

            for ins in instances:
                if len(ins) == 5: 
                    src.append(ins[0])
                    masked_words_num += len(ins[1])
                    tgt_mlm.append([0] * len(ins[0]))
                    for mask in ins[1]:
                        tgt_mlm[-1][mask[0]] = mask[1]
                    is_next.append(ins[2])
                    seg.append([1] * ins[3][0] + [2] * (ins[3][1] - ins[3][0]) + [PAD_ID] * (len(ins[0]) - ins[3][1]))
                    proto.append(ins[4])
                else:
                    src_single, tgt_mlm_single = mask_seq(ins[0], self.tokenizer, self.whole_word_masking, self.span_masking, self.span_geo_prob, self.span_max_length)
                    masked_words_num += len(tgt_mlm_single)
                    src.append(src_single)
                    tgt_mlm.append([0] * len(ins[0]))
                    for mask in tgt_mlm_single:
                        tgt_mlm[-1][mask[0]] = mask[1]
                    is_next.append(ins[1])
                    seg.append([1] * ins[2][0] + [2] * (ins[2][1] - ins[2][0]) + [PAD_ID] * (len(ins[0]) - ins[2][1]))
                    proto.append(ins[4])
            if masked_words_num == 0:
                continue

            yield torch.LongTensor(src), \
                torch.LongTensor(tgt_mlm), \
                torch.LongTensor(is_next), \
                torch.LongTensor(seg), \
                torch.LongTensor(proto)
                

class MlmDataset(Dataset):
    def __init__(self, args, vocab, tokenizer):
        super(MlmDataset, self).__init__(args, vocab, tokenizer)
        self.full_sentences = args.full_sentences

    def worker(self, proc_id, start, end):
        print("Worker %d is building dataset ... " % proc_id)
        set_seed(self.seed)
        dataset_writer = open("dataset-tmp-" + str(proc_id) + ".pt", "wb")
        docs_buffer = []
        for _ in range(self.dup_factor):
            pos = 0
            with open(self.corpus_path, mode="r", encoding="utf-8") as f:
                while pos < start:
                    f.readline()
                    pos += 1
                while True:
                    line = f.readline()
                    pos += 1

                    document = [self.vocab.get(CLS_TOKEN)] + self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(line)) + [self.vocab.get(SEP_TOKEN)]

                    if self.full_sentences:
                        if len(document) > 0:
                            docs_buffer.append(document)
                        if len(docs_buffer) == self.docs_buffer_size:
                            # Build instances from documents.
                            all_documents = self.concatenate_docs(docs_buffer)
                            instances = self.build_instances(all_documents)
                            # Save instances.
                            for instance in instances:
                                pickle.dump(instance, dataset_writer)
                            # Clear buffer.
                            docs_buffer = []
                        if pos >= end:
                            if len(docs_buffer) > 0:
                                all_documents = self.concatenate_docs(docs_buffer)
                                instances = self.build_instances(all_documents)
                                # Save instances.
                                for instance in instances:
                                    pickle.dump(instance, dataset_writer)
                            break
                    else:
                        if len(document) > 0:
                            instances = self.build_instances(document)
                            # Save instances.
                            for instance in instances:
                                pickle.dump(instance, dataset_writer)

                    if pos >= end:
                        break

        dataset_writer.close()

    def concatenate_docs(self, docs_buffer):
        all_documents = []
        for i in range(len(docs_buffer)):
            all_documents += docs_buffer[i]
        return all_documents

    def build_instances(self, all_documents):
        instances = []
        instances_num = len(all_documents) // self.seq_length
        for i in range(instances_num):
            src = all_documents[i * self.seq_length: (i + 1) * self.seq_length]
            seg_pos = [len(src)]

            if not self.dynamic_masking:
                src, tgt = mask_seq(src, self.tokenizer, self.whole_word_masking, self.span_masking, self.span_geo_prob, self.span_max_length)
                instance = (src, tgt, seg_pos)
            else:
                instance = (src, seg_pos)

            instances.append(instance)

        src = all_documents[instances_num * self.seq_length:]
        seg_pos = [len(src)]

        while len(src) != self.seq_length:
            src.append(PAD_ID)

        if not self.dynamic_masking:
            src, tgt = mask_seq(src, self.tokenizer, self.whole_word_masking, self.span_masking, self.span_geo_prob, self.span_max_length)
            instance = (src, tgt, seg_pos)
        else:
            instance = (src, seg_pos)

        instances.append(instance)
        return instances


class MlmDataLoader(DataLoader):
    def __iter__(self):
        while True:
            while self._empty():
                self._fill_buf()
            if self.start + self.batch_size >= self.end:
                instances = self.buffer[self.start:]
            else:
                instances = self.buffer[self.start: self.start + self.batch_size]

            self.start += self.batch_size

            src = []
            tgt = []
            seg = []

            masked_words_num = 0

            for ins in instances:
                if len(ins) == 3:
                    src.append(ins[0])
                    masked_words_num += len(ins[1])
                    tgt.append([0] * len(ins[0]))
                    for mask in ins[1]:
                        tgt[-1][mask[0]] = mask[1]
                    seg.append([1] * ins[2][0] + [PAD_ID] * (len(ins[0]) - ins[2][0]))
                else:
                    src_single, tgt_single = mask_seq(ins[0], self.tokenizer, self.whole_word_masking, self.span_masking, self.span_geo_prob, self.span_max_length)
                    masked_words_num += len(tgt_single)
                    src.append(src_single)
                    tgt.append([0] * len(ins[0]))
                    for mask in tgt_single:
                        tgt[-1][mask[0]] = mask[1]
                    seg.append([1] * ins[1][0] + [PAD_ID] * (len(ins[0]) - ins[1][0]))

            if masked_words_num == 0:
                continue

            yield torch.LongTensor(src), \
                torch.LongTensor(tgt), \
                torch.LongTensor(seg)


class AlbertDataset(Dataset):
    """
    Construct dataset for MLM and SOP tasks from the given corpus.
    Each document consists of multiple sentences,
    and each sentence occupies a single line.
    Documents in corpus must be separated by empty lines.
    """

    def __init__(self, args, vocab, tokenizer):
        super(AlbertDataset, self).__init__(args, vocab, tokenizer)
        self.short_seq_prob = args.short_seq_prob

    def worker(self, proc_id, start, end):
        print("Worker %d is building dataset ... " % proc_id)
        set_seed(self.seed)
        document = []
        dataset_writer = open("dataset-tmp-" + str(proc_id) + ".pt", "wb")
        for _ in range(self.dup_factor):
            pos = 0
            with open(self.corpus_path, mode="r", encoding="utf-8") as f:
                while pos < start:
                    f.readline()
                    pos += 1
                while True:
                    line = f.readline()
                    pos += 1
                    if not line.strip():
                        if len(document) >= 1:
                            instances = self.build_instances(document)
                            for instance in instances:
                                pickle.dump(instance, dataset_writer)
                        document = []
                    sentence = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(line))
                    if len(sentence) > 0:
                        document.append(sentence)
                    if pos >= end - 1:
                        if len(document) >= 1:
                            instances = self.build_instances(document)
                            for instance in instances:
                                pickle.dump(instance, dataset_writer)
                        break
        dataset_writer.close()

    def build_instances(self, document):
        instances = []
        instances.extend(self.create_ins_from_doc(document))
        return instances

    def create_ins_from_doc(self, document):
        max_num_tokens = self.seq_length - 3
        target_seq_length = max_num_tokens
        if random.random() < self.short_seq_prob:
            target_seq_length = random.randint(2, max_num_tokens)
        instances = []
        current_chunk = []
        current_length = 0
        i = 0
        while i < len(document):
            segment = document[i]
            current_chunk.append(segment)
            current_length += len(segment)
            if i == len(document) - 1 or current_length >= target_seq_length:
                if current_chunk:
                    a_end = 1
                    if len(current_chunk) >= 2:
                        a_end = random.randint(1, len(current_chunk) - 1)

                    tokens_a = []
                    for j in range(a_end):
                        tokens_a.extend(current_chunk[j])

                    tokens_b = []
                    is_wrong_order = 0
                    for j in range(a_end, len(current_chunk)):
                        tokens_b.extend(current_chunk[j])

                    if random.random() < 0.5:
                        is_wrong_order = 1
                        tmp = tokens_a
                        tokens_a = tokens_b
                        tokens_b = tmp

                    truncate_seq_pair(tokens_a, tokens_b, max_num_tokens)

                    src = []
                    src.append(self.vocab.get(CLS_TOKEN))
                    src.extend(tokens_a)
                    src.append(self.vocab.get(SEP_TOKEN))
                    seg_pos = [len(src)]
                    src.extend(tokens_b)
                    src.append(self.vocab.get(SEP_TOKEN))
                    seg_pos.append(len(src))

                    while len(src) != self.seq_length:
                        src.append(PAD_ID)

                    if not self.dynamic_masking:
                        src, tgt_mlm = mask_seq(src, self.tokenizer, self.whole_word_masking, self.span_masking, self.span_geo_prob, self.span_max_length)
                        instance = (src, tgt_mlm, is_wrong_order, seg_pos)
                    else:
                        instance = (src, is_wrong_order, seg_pos)

                    instances.append(instance)
                current_chunk = []
                current_length = 0
            i += 1
        return instances


class AlbertDataLoader(BertDataLoader):
    '''
    AlbertDataLoader can reuse the code of BertDataLoader.
    '''
    pass


class LmDataset(Dataset):
    def worker(self, proc_id, start, end):
        print("Worker %d is building dataset ... " % proc_id)
        set_seed(self.seed)
        dataset_writer = open("dataset-tmp-" + str(proc_id) + ".pt", "wb")
        pos = 0
        with open(self.corpus_path, mode="r", encoding="utf-8") as f:
            while pos < start:
                f.readline()
                pos += 1
            while True:
                line = f.readline()
                pos += 1

                document = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(line))
                document = [self.vocab.get(CLS_TOKEN)] + document + [self.vocab.get(SEP_TOKEN)]

                instances_num = len(document) // (self.seq_length + 1)
                for i in range(instances_num):
                    src = document[i * (self.seq_length + 1): (i + 1) * (self.seq_length + 1)]
                    seg_pos = self.seq_length
                    pickle.dump((src, seg_pos), dataset_writer)

                src = document[instances_num * (self.seq_length + 1):]
                if len(src) > 0:
                    seg_pos = len(src)
                    while len(src) != self.seq_length + 1:
                        src.append(PAD_ID)
                    pickle.dump((src, seg_pos), dataset_writer)

                if pos >= end:
                    break

        dataset_writer.close()


class LmDataLoader(DataLoader):
    def __iter__(self):
        while True:
            while self._empty():
                self._fill_buf()
            if self.start + self.batch_size >= self.end:
                instances = self.buffer[self.start:]
            else:
                instances = self.buffer[self.start: self.start + self.batch_size]

            self.start += self.batch_size

            src = []
            tgt = []
            seg = []

            for ins in instances:
                src.append(ins[0][:-1])
                tgt.append(ins[0][1:])
                if ins[1] == len(ins[0]):
                    seg.append([1] * (ins[1] - 1))
                else:
                    seg.append([1] * ins[1] + [PAD_ID] * (len(ins[0]) - 1 - ins[1]))

            yield torch.LongTensor(src), \
                torch.LongTensor(tgt), \
                torch.LongTensor(seg)


class BilmDataset(Dataset):
    def worker(self, proc_id, start, end):
        print("Worker %d is building dataset ... " % proc_id)
        set_seed(self.seed)
        dataset_writer = open("dataset-tmp-" + str(proc_id) + ".pt", "wb")
        pos = 0
        with open(self.corpus_path, mode="r", encoding="utf-8") as f:
            while pos < start:
                f.readline()
                pos += 1
            while True:
                line = f.readline()
                pos += 1

                document = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(line))

                instances_num = len(document) // self.seq_length
                for i in range(instances_num):
                    src = document[i * self.seq_length: (i + 1) * self.seq_length]
                    tgt_forward = src[1:] + [self.vocab.get(SEP_TOKEN)]
                    tgt_backward = [self.vocab.get(CLS_TOKEN)] + src[:-1]
                    seg = [1] * len(src)
                    pickle.dump((src, tgt_forward, tgt_backward, seg), dataset_writer)

                src = document[instances_num * self.seq_length:]
                if len(src) < 1:
                    continue
                tgt_forward = src[1:] + [self.vocab.get(SEP_TOKEN)]
                tgt_backward = [self.vocab.get(CLS_TOKEN)] + src[:-1]
                seg = [1] * len(src)
                while len(src) != self.seq_length:
                    src.append(PAD_ID)
                    tgt_forward.append(PAD_ID)
                    tgt_backward.append(PAD_ID)
                    seg.append(PAD_ID)
                pickle.dump((src, tgt_forward, tgt_backward, seg), dataset_writer)

                if pos >= end - 1:
                    break

        dataset_writer.close()


class BilmDataLoader(DataLoader):
    def __iter__(self):
        while True:
            while self._empty():
                self._fill_buf()
            if self.start + self.batch_size >= self.end:
                instances = self.buffer[self.start:]
            else:
                instances = self.buffer[self.start: self.start + self.batch_size]

            self.start += self.batch_size

            src = []
            tgt_forward = []
            tgt_backward = []
            seg = []

            for ins in instances:
                src.append(ins[0])
                tgt_forward.append(ins[1])
                tgt_backward.append(ins[2])
                seg.append(ins[3])

            yield torch.LongTensor(src), \
                torch.LongTensor(tgt_forward), \
                torch.LongTensor(tgt_backward), \
                torch.LongTensor(seg)


class Seq2seqDataset(Dataset):
    def __init__(self, args, vocab, tokenizer):
        super(Seq2seqDataset, self).__init__(args, vocab, tokenizer)
        self.tgt_seq_length = args.tgt_seq_length
        self.src_vocab, self.src_tokenizer = vocab, tokenizer
        self.tgt_tokenizer = args.tgt_tokenizer
        self.tgt_vocab = self.tgt_tokenizer.vocab

    def worker(self, proc_id, start, end):
        print("Worker %d is building dataset ... " % proc_id)
        set_seed(self.seed)
        dataset_writer = open("dataset-tmp-" + str(proc_id) + ".pt", "wb")
        pos = 0
        with open(self.corpus_path, mode="r", encoding="utf-8") as f:
            while pos < start:
                f.readline()
                pos += 1
            while True:
                line = f.readline()
                pos += 1

                if len(line.strip().split("\t")) != 2:
                    if pos >= end:
                        break
                    continue
                document_src, document_tgt = line.strip().split("\t")
                src = self.src_tokenizer.convert_tokens_to_ids(self.src_tokenizer.tokenize(document_src))
                tgt = self.tgt_tokenizer.convert_tokens_to_ids(self.tgt_tokenizer.tokenize(document_tgt))

                src = [self.src_vocab.get(CLS_TOKEN)] + src + [self.src_vocab.get(SEP_TOKEN)]
                tgt = [self.tgt_vocab.get(CLS_TOKEN)] + tgt + [self.tgt_vocab.get(SEP_TOKEN)]
                seg = [1] * len(src)

                src, tgt, seg = src[:self.seq_length], tgt[:self.tgt_seq_length + 1], seg[:self.seq_length]
                while len(src) != self.seq_length:
                    src.append(PAD_ID)
                    seg.append(PAD_ID)
                while len(tgt) != self.tgt_seq_length + 1:
                    tgt.append(PAD_ID)
                pickle.dump((src, tgt, seg), dataset_writer)

                if pos >= end:
                    break

            dataset_writer.close()


class Seq2seqDataLoader(DataLoader):
    def __iter__(self):
        while True:
            while self._empty():
                self._fill_buf()
            if self.start + self.batch_size >= self.end:
                instances = self.buffer[self.start:]
            else:
                instances = self.buffer[self.start: self.start + self.batch_size]

            self.start += self.batch_size

            src = []
            tgt_in = []
            tgt_out = []
            seg = []

            for ins in instances:
                src.append(ins[0])
                tgt_in.append(ins[1][:-1])
                tgt_out.append(ins[1][1:])
                seg.append(ins[2])

            yield torch.LongTensor(src), \
                torch.LongTensor(tgt_in), \
                torch.LongTensor(tgt_out), \
                torch.LongTensor(seg)


class T5Dataset(MlmDataset):
    '''
    T5 can reuse the code of MlmDataset.
    '''
    pass


class T5DataLoader(DataLoader):
    def __iter__(self):
        while True:
            while self._empty():
                self._fill_buf()
            if self.start + self.batch_size >= self.end:
                instances = self.buffer[self.start:]
            else:
                instances = self.buffer[self.start: self.start + self.batch_size]

            self.start += self.batch_size

            src = []
            tgt_in = []
            tgt_out = []
            seg = []

            tgt_seq_length = 0

            for _, ins in enumerate(instances):
                if len(ins) == 3:
                    src_single = ins[0]
                    tgt_single = ins[1]
                    seg.append([1] * ins[2][0] + [PAD_ID] * (len(ins[0]) - ins[2][0]))
                else:
                    src_single, tgt_single = mask_seq(ins[0], self.tokenizer, self.whole_word_masking, self.span_masking, self.span_geo_prob, self.span_max_length)
                    seg.append([1] * ins[1][0] + [PAD_ID] * (len(ins[0]) - ins[1][0]))

                MASK_ID = self.vocab.get(MASK_TOKEN)
                SENTINEL_ID = self.vocab.get(SENTINEL_TOKEN)

                for src_index, _ in tgt_single:
                    if src_single[src_index] != MASK_ID:
                        src_single[src_index] = MASK_ID

                tgt_in_single = [self.vocab.get(CLS_TOKEN)]
                mask_index = 0
                src_with_sentinel = []
                for token_id in src_single:
                    if token_id == MASK_ID:
                        if len(src_with_sentinel) > 0 and src_with_sentinel[-1] == (SENTINEL_ID - 1):
                            pass
                        else:
                            src_with_sentinel.append(SENTINEL_ID)
                            tgt_in_single.append(SENTINEL_ID)
                            SENTINEL_ID += 1
                        tgt_in_single.append(tgt_single[mask_index][1])
                        mask_index += 1
                    else:
                        src_with_sentinel.append(token_id)
                tgt_in_single.append(SENTINEL_ID)
                tgt_in_single.append(self.vocab.get(SEP_TOKEN))

                while len(src_with_sentinel) < len(src_single):
                    src_with_sentinel.append(PAD_ID)

                if len(tgt_in_single) > tgt_seq_length:
                    tgt_seq_length = len(tgt_in_single)

                src.append(src_with_sentinel)
                tgt_in.append(tgt_in_single)
                tgt_out.append(tgt_in[-1][1:] + [PAD_ID])

            for i in range(len(tgt_in)):
                while len(tgt_in[i]) != tgt_seq_length:
                    tgt_in[i].append(PAD_ID)
                    tgt_out[i].append(PAD_ID)

            yield torch.LongTensor(src), \
                torch.LongTensor(tgt_in), \
                torch.LongTensor(tgt_out), \
                torch.LongTensor(seg)


class ClsDataset(Dataset):
    def worker(self, proc_id, start, end):
        print("Worker %d is building dataset ... " % proc_id)
        set_seed(self.seed)
        f_write = open("dataset-tmp-" + str(proc_id) + ".pt", "wb")
        pos = 0
        with open(self.corpus_path, mode="r", encoding="utf-8") as f:
            while pos < start:
                line = f.readline()
                pos += 1
            while True:
                line = f.readline()
                pos += 1

                line = line.strip().split('\t')
                if len(line) == 2:
                    label = int(line[0])
                    text = " ".join(line[1:])
                    src = [self.vocab.get(t) for t in self.tokenizer.tokenize(text)]
                    src = [self.vocab.get(CLS_TOKEN)] + src
                    tgt = label
                    seg = [1] * len(src)
                    if len(src) >= self.seq_length:
                        src = src[:self.seq_length]
                        seg = seg[:self.seq_length]
                    else:
                        while len(src) != self.seq_length:
                            src.append(PAD_ID)
                            seg.append(PAD_ID)
                    pickle.dump((src, tgt, seg), f_write)
                elif len(line) == 3:  # For sentence pair input.
                    label = int(line[0])
                    text_a, text_b = line[1], line[2]

                    src_a = [self.vocab.get(t) for t in self.tokenizer.tokenize(text_a)]
                    src_a = [self.vocab.get(CLS_TOKEN)] + src_a + [self.vocab.get(SEP_TOKEN)]
                    src_b = [self.vocab.get(t) for t in self.tokenizer.tokenize(text_b)]
                    src_b = src_b + [self.vocab.get(SEP_TOKEN)]

                    src = src_a + src_b
                    seg = [1] * len(src_a) + [2] * len(src_b)

                    if len(src) >= self.seq_length:
                        src = src[:self.seq_length]
                        seg = seg[:self.seq_length]
                    else:
                        while len(src) != self.seq_length:
                            src.append(PAD_ID)
                            seg.append(PAD_ID)
                    pickle.dump((src, tgt, seg), f_write)
                else:
                    pass

                if pos >= end - 1:
                    break

        f_write.close()


class ClsDataLoader(DataLoader):
    def __iter__(self):
        while True:
            while self._empty():
                self._fill_buf()
            if self.start + self.batch_size >= self.end:
                instances = self.buffer[self.start:]
            else:
                instances = self.buffer[self.start: self.start + self.batch_size]

            self.start += self.batch_size

            src = []
            tgt = []
            seg = []

            for ins in instances:
                src.append(ins[0])
                tgt.append(ins[1])
                seg.append(ins[2])

            yield torch.LongTensor(src), \
                torch.LongTensor(tgt), \
                torch.LongTensor(seg)


class PrefixlmDataset(Dataset):

    def worker(self, proc_id, start, end):
        print("Worker %d is building dataset ... " % proc_id)
        set_seed(self.seed)
        dataset_writer = open("dataset-tmp-" + str(proc_id) + ".pt", "wb")
        pos = 0
        with open(self.corpus_path, mode="r", encoding="utf-8") as f:
            while pos < start:
                f.readline()
                pos += 1
            while True:
                line = f.readline()
                pos += 1

                if len(line.strip().split("\t")) != 2:
                    if pos >= end:
                        break
                    continue
                document_src, document_tgt = line.strip().split("\t")
                src = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(document_src))
                tgt = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(document_tgt))
                src = [self.vocab.get(CLS_TOKEN)] + src + [self.vocab.get(SEP_TOKEN)]
                tgt = tgt + [self.vocab.get(SEP_TOKEN)]
                seg_pos = [len(src)]

                if seg_pos[0] >= self.seq_length:
                    continue

                src = src + tgt
                tgt = [0] * seg_pos[0] + tgt[1:] + [PAD_ID]
                seg_pos.append(len(src))
                src, tgt = src[:self.seq_length], tgt[:self.seq_length]
                while len(src) != self.seq_length:
                    src.append(PAD_ID)
                    tgt.append(PAD_ID)
                if seg_pos[1] > self.seq_length:
                    seg_pos[1] = self.seq_length

                pickle.dump((src, tgt, seg_pos), dataset_writer)

                if pos >= end:
                    break

            dataset_writer.close()


class PrefixlmDataLoader(DataLoader):
    def __iter__(self):
        while True:
            while self._empty():
                self._fill_buf()
            if self.start + self.batch_size >= self.end:
                instances = self.buffer[self.start:]
            else:
                instances = self.buffer[self.start: self.start + self.batch_size]

            self.start += self.batch_size

            src = []
            tgt = []
            seg = []

            for ins in instances:
                src.append(ins[0])
                tgt.append(ins[1])
                seg.append([1] * ins[2][0] + [2] * (ins[2][1] - ins[2][0]) + [PAD_ID] * (len(ins[0]) - ins[2][1]))

            yield torch.LongTensor(src), \
                torch.LongTensor(tgt), \
                torch.LongTensor(seg)

