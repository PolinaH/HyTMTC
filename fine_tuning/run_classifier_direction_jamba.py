"""
This script provides an exmaple to wrap UER-py for classification.
"""
import copy
import os
import sys
import random
import logging
from pprint import pformat
import datetime
import argparse
import torch
import torch.nn as nn
from uer.layers import *
from uer.encoders import *
from uer.utils.vocab import Vocab
from uer.utils.constants import *
from uer.utils import *
from uer.utils.optimizers import *
from uer.utils.config import load_hyperparam
from uer.utils.seed import set_seed
from uer.model_saver import save_model
from uer.layers.multi_headed_attn import MultiHeadedAttention
from args_options.args_options import finetune_opts
import tqdm
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics import f1_score,precision_score,recall_score
import pudb
import warnings
warnings.filterwarnings("ignore")
sys.path.append(os.getcwd())


def logging_definition(args,root):
    mdate = datetime.date.today()
    log_file_folder = f"{root}/log/model_finetuning"
    log_file_path = f"{log_file_folder}/{mdate}_{str(len(os.listdir(log_file_folder)))}.txt"
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
    return logging, log_file_path


class Classifier(nn.Module):
    def __init__(self, args):
        super(Classifier, self).__init__()
        self.embedding = str2embedding[args.embedding](args, len(args.tokenizer.vocab))
        self.encoder = str2encoder[args.encoder](args)
        self.labels_num = args.labels_num
        self.pooling = args.pooling
        self.soft_targets = args.soft_targets
        self.soft_alpha = args.soft_alpha
        self.output_layer_1 = nn.Linear(args.hidden_size, args.hidden_size)
        self.output_layer_2 = nn.Linear(args.hidden_size, self.labels_num)

        #self.output_layer_1 = nn.Linear(args.hidden_size, args.hidden_size)
        #self.output_layer_2 = nn.Linear(args.hidden_size, int(args.hidden_size/2))
        #self.output_layer_3 = nn.Linear(int(args.hidden_size/2), self.labels_num)

        #weights = torch.tensor([1, 1, 1, 1]).float()
        #self.pp_entroy = nn.CrossEntropyLoss(weight=weights)
        self.pp_entroy = nn.CrossEntropyLoss()

    def forward(self, src, coarse, fine, seg, byte_transmission_idx, sample_len, tgt, soft_tgt=None):
        """
        Args:
            src: [batch_size x seq_length]
            coarse: [batch_size x seq_length]
            fine: [batch_size x seq_length]
            seg: [batch_size x seq_length]
            byte_transmission_idx: [batch_size x 2] 字节信息和传输模式信息的截止idx
            sample_len: [batch_size]
            tgt: [batch_size]
        """
        # Embedding.
        emb = self.embedding(src, seg, coarse, fine, sample_len)
        # Encoder.
        output, _ = self.encoder(emb, seg, byte_transmission_idx)
        # Target.
        if self.pooling == "mean":
            output_cls = torch.mean(output, dim=1)
        elif self.pooling == "max":
            output_cls = torch.max(output, dim=1)[0]
        elif self.pooling == "last":
            output_cls = output[:, -1, :]
        else:
            output_cls = output[:, 0, :]

        output_cls = torch.tanh(self.output_layer_1(output_cls))
        logits = self.output_layer_2(output_cls)

        #output = torch.tanh(self.output_layer_1(output))
        #output = torch.tanh(self.output_layer_2(output))
        #logits = self.output_layer_3(output)

        if tgt is not None:
            if self.soft_targets and soft_tgt is not None:
                loss = self.soft_alpha * nn.MSELoss()(logits, soft_tgt) + \
                       (1 - self.soft_alpha) * nn.NLLLoss()(nn.LogSoftmax(dim=-1)(logits), tgt.view(-1))
            else:
                # loss = nn.NLLLoss()(nn.LogSoftmax(dim=-1)(logits), tgt.view(-1))
                loss = self.pp_entroy(logits, tgt)
            return loss, logits, output
        else:
            return None, logits, output


class Classifier_self_attn(nn.Module):
    def __init__(self, args):
        super(Classifier_self_attn, self).__init__()
        self.embedding = str2embedding[args.embedding](args, len(args.tokenizer.vocab))
        self.encoder = str2encoder[args.encoder](args)
        self.labels_num = args.labels_num
        self.pooling = args.pooling
        self.soft_targets = args.soft_targets
        self.soft_alpha = args.soft_alpha

        attention_head_size = args.hidden_size // args.output_self_attn_heads_num
        self.self_attn = MultiHeadedAttention(
            args.hidden_size, args.heads_num, attention_head_size, args.output_self_attn_dropout)

        self.output_layer_1 = nn.Linear(args.hidden_size, args.hidden_size)
        self.output_layer_2 = nn.Linear(args.hidden_size, self.labels_num)

        #self.output_layer_1 = nn.Linear(args.hidden_size, args.hidden_size)
        #self.output_layer_2 = nn.Linear(args.hidden_size, int(args.hidden_size/2))
        #self.output_layer_3 = nn.Linear(int(args.hidden_size/2), self.labels_num)

        #weights = torch.tensor([1, 1, 1, 1]).float()
        #self.pp_entroy = nn.CrossEntropyLoss(weight=weights)
        self.pp_entroy = nn.CrossEntropyLoss()

    def forward(self, src, coarse, fine, seg, byte_transmission_idx, sample_len, tgt, soft_tgt=None):
        """
        Args:
            src: [batch_size x seq_length]
            coarse: [batch_size x seq_length]
            fine: [batch_size x seq_length]
            seg: [batch_size x seq_length]
            byte_transmission_idx: [batch_size x 2] 字节信息和传输模式信息的截止idx
            sample_len: [batch_size]
            tgt: [batch_size]
        """
        # Embedding.
        emb = self.embedding(src, seg, coarse, fine, sample_len)
        # Encoder.
        output, _ = self.encoder(emb, seg, byte_transmission_idx)
        # Multi-head self attention
        _, seq_length, _ = output.size()
        mask = (seg > 0).unsqueeze(1).repeat(1, seq_length, 1).unsqueeze(1)
        mask = mask.float()
        # mask:sequence中有值的地方在mask中是0，后面没有值的地方在mask中是-10000
        mask = (1.0 - mask) * -10000.0
        output, _ = self.self_attn(output, output, output, mask)

        # Target.
        if self.pooling == "mean":
            output_cls = torch.mean(output, dim=1)
        elif self.pooling == "max":
            output_cls = torch.max(output, dim=1)[0]
        elif self.pooling == "last":
            output_cls = output[:, -1, :]
        else:
            output_cls = output[:, 0, :]

        output_cls = torch.tanh(self.output_layer_1(output_cls))
        logits = self.output_layer_2(output_cls)

        #output = torch.tanh(self.output_layer_1(output))
        #output = torch.tanh(self.output_layer_2(output))
        #logits = self.output_layer_3(output)

        if tgt is not None:
            if self.soft_targets and soft_tgt is not None:
                loss = self.soft_alpha * nn.MSELoss()(logits, soft_tgt) + \
                       (1 - self.soft_alpha) * nn.NLLLoss()(nn.LogSoftmax(dim=-1)(logits), tgt.view(-1))
            else:
                # loss = nn.NLLLoss()(nn.LogSoftmax(dim=-1)(logits), tgt.view(-1))
                loss = self.pp_entroy(logits, tgt)
            return loss, logits, output
        else:
            return _, logits, output


def count_labels_num(path):
    labels_set, columns = set(), {}
    with open(path, mode="r", encoding="utf-8") as f:
        for line_id, line in enumerate(f):
            if line_id == 0:
                for i, column_name in enumerate(line.strip().split("\t")):
                    columns[column_name] = i
                continue
            line = line.strip().split("\t")
            label = int(line[columns["label"]])
            labels_set.add(label)
    return len(labels_set)


def load_or_initialize_parameters(logging, args, model):
    if args.pretrained_model_path != 'None':
        logging.info("Initialize with pretrained model.")
        model.load_state_dict(torch.load(args.pretrained_model_path,
                                         map_location={'cuda:1':'cuda:0', 'cuda:2':'cuda:0',
                                                       'cuda:3':'cuda:0', 'cuda:4':'cuda:0',
                                                       'cuda:5':'cuda:0', 'cuda:6':'cuda:0',
                                                       'cuda:7':'cuda:0'}), strict=False)
    else:
        logging.info("Initialize with normal distribution.")
        for n, p in list(model.named_parameters()):
            if "gamma" not in n and "beta" not in n:
                p.data.normal_(0, 0.02)
    return model


def build_optimizer(args, model):
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'gamma', 'beta']
    optimizer_grouped_parameters = [
                {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay) and p.requires_grad], 'weight_decay_rate': 0.01},
                {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay) and p.requires_grad], 'weight_decay_rate': 0.0}
    ]

    if all(len(group['params']) == 0 for group in optimizer_grouped_parameters):
        raise ValueError("No parameters with requires_grad=True to optimize.")

    if args.optimizer in ["adamw"]:
        optimizer = str2optimizer[args.optimizer](optimizer_grouped_parameters, lr=args.learning_rate, correct_bias=False)
    else:
        optimizer = str2optimizer[args.optimizer](optimizer_grouped_parameters, lr=args.learning_rate,
                                                  scale_parameter=False, relative_step=False)
    if args.scheduler in ["constant"]:
        scheduler = str2scheduler[args.scheduler](optimizer)
    elif args.scheduler in ["constant_with_warmup"]:
        scheduler = str2scheduler[args.scheduler](optimizer, args.train_steps*args.warmup)
    else:
        scheduler = str2scheduler[args.scheduler](optimizer, args.train_steps*args.warmup, args.train_steps)
    return optimizer, scheduler


def batch_loader(batch_size, src, coarse, fine, seg, byte_transmission_idx, sample_len, tgt,
                 soft_tgt=None):
    instances_num = src.size()[0]
    for i in range(instances_num // batch_size):
        src_batch = src[i * batch_size: (i + 1) * batch_size, :]
        coarse_batch = coarse[i * batch_size: (i + 1) * batch_size, :]
        fine_batch = fine[i * batch_size: (i + 1) * batch_size, :]
        seg_batch = seg[i * batch_size: (i + 1) * batch_size, :]
        byte_transmission_idx_batch = byte_transmission_idx[i * batch_size: (i + 1) * batch_size, :]
        sample_len_batch = sample_len[i * batch_size: (i + 1) * batch_size]
        tgt_batch = tgt[i * batch_size: (i + 1) * batch_size]
        if soft_tgt is not None:
            soft_tgt_batch = soft_tgt[i * batch_size: (i + 1) * batch_size, :]
            yield src_batch, coarse_batch, fine_batch, seg_batch, byte_transmission_idx_batch, \
                sample_len_batch, tgt_batch, soft_tgt_batch
        else:
            yield src_batch, coarse_batch, fine_batch, seg_batch, byte_transmission_idx_batch, \
                sample_len_batch, tgt_batch, None
    if instances_num > instances_num // batch_size * batch_size:
        src_batch = src[instances_num // batch_size * batch_size:, :]
        coarse_batch = coarse[instances_num // batch_size * batch_size:, :]
        fine_batch = fine[instances_num // batch_size * batch_size:, :]
        seg_batch = seg[instances_num // batch_size * batch_size:, :]
        byte_transmission_idx_batch = byte_transmission_idx[instances_num // batch_size * batch_size:, :]
        sample_len_batch = sample_len[instances_num // batch_size * batch_size:]
        tgt_batch = tgt[instances_num // batch_size * batch_size:]
        if soft_tgt is not None:
            soft_tgt_batch = soft_tgt[instances_num // batch_size * batch_size:, :]
            yield src_batch, coarse_batch, fine_batch, seg_batch, byte_transmission_idx_batch, \
                sample_len_batch, tgt_batch, soft_tgt_batch
        else:
            yield src_batch, coarse_batch, fine_batch, seg_batch, byte_transmission_idx_batch, \
                sample_len_batch, tgt_batch, None


def read_dataset(logging, args, path):
    logging.info(f"Reading dataset...")
    dataset, columns = [], {}

    #pudb.set_trace()

    with open(path, mode="r", encoding="utf-8") as f:
        for line_id, line in enumerate(f):
            if line_id == 0:
                for i, column_name in enumerate(line.strip().split("\t")):
                    columns[column_name] = i
                continue
            line = line[:-1].split("\t")
            tgt = int(line[columns["label"]])  # 样本标签
            if args.soft_targets and "logits" in columns.keys():
                soft_tgt = [float(value) for value in line[columns["logits"]].split(" ")]

            if "text_b" not in columns:  # Sentence classification.
                src = eval(line[columns["tokens_list"]])
                coarse_semantic = eval(line[columns["coarse_semantic_list"]])
                fine_semantic = eval(line[columns["fine_semantic_list"]])
                seg_simplified = eval(line[columns["seg_list"]])
                byte_transmission_idx = [seg_simplified[-2], seg_simplified[-1]]

                if (len(coarse_semantic) != len(src)) or (len(fine_semantic) != len(src)):
                    pudb.set_trace()

                seg = []
                tmp_changeidx = 0
                for tmp_idx, tmp_item in enumerate(seg_simplified):
                    if tmp_idx != len(seg_simplified)-1:
                        seg.extend([tmp_idx + 1] * (tmp_item - tmp_changeidx))
                        tmp_changeidx = tmp_item
                    else:
                        seg.extend([6] * (tmp_item - tmp_changeidx))  # 方向长度token的seg编码为6

            else:  # Sentence-pair classification.
                text_a, text_b = line[columns["text_a"]], line[columns["text_b"]]
                src_a = args.tokenizer.convert_tokens_to_ids([CLS_TOKEN] + args.tokenizer.tokenize(text_a) + [SEP_TOKEN])
                src_b = args.tokenizer.convert_tokens_to_ids(args.tokenizer.tokenize(text_b) + [SEP_TOKEN])
                src = src_a + src_b
                seg = [1] * len(src_a) + [2] * len(src_b)

            sample_len = args.seq_length
            if len(src) > args.seq_length:
                src = src[: args.seq_length]
                coarse_semantic = coarse_semantic[: args.seq_length]
                fine_semantic = fine_semantic[: args.seq_length]
                seg = seg[: args.seq_length]
            elif len(src) < args.seq_length:
                sample_len = len(src)
                extend_len = args.seq_length-len(src)
                src.extend([0]*extend_len)
                coarse_semantic.extend([0]*extend_len)
                fine_semantic.extend([0]*extend_len)
                seg.extend([0]*extend_len)

            if args.soft_targets and "logits" in columns.keys():
                dataset.append((src, coarse_semantic, fine_semantic, seg, byte_transmission_idx,
                                sample_len, tgt, soft_tgt))
            else:
                dataset.append((src, coarse_semantic, fine_semantic, seg, byte_transmission_idx,
                                sample_len, tgt))

    return dataset


def train_model(args, model, optimizer, scheduler, src_batch, coarse_batch, fine_batch, seg_batch,
                byte_transmission_idx_batch, sample_len_batch, tgt_batch, soft_tgt_batch=None):
    model.zero_grad()

    src_batch = src_batch.to(args.device)
    coarse_batch = coarse_batch.to(args.device)
    fine_batch = fine_batch.to(args.device)
    seg_batch = seg_batch.to(args.device)
    byte_transmission_idx_batch = byte_transmission_idx_batch.to(args.device)
    sample_len_batch = sample_len_batch.to(args.device)
    tgt_batch = tgt_batch.to(args.device)
    if soft_tgt_batch is not None:
        soft_tgt_batch = soft_tgt_batch.to(args.device)

    loss, logits, _ = model(src_batch, coarse_batch, fine_batch, seg_batch, byte_transmission_idx_batch,
                            sample_len_batch, tgt_batch, soft_tgt_batch)
    if torch.cuda.device_count() > 1:
        loss = torch.mean(loss)

    if args.fp16:
        with args.amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()
    else:
        loss.backward()

    optimizer.step()
    scheduler.step()

    return loss, logits


def evaluate(logging, args, dataset):
    logging.info("Valid set evaluation.")

    src = torch.LongTensor([sample[0] for sample in dataset])
    coarse = torch.LongTensor([sample[1] for sample in dataset])
    fine = torch.LongTensor([sample[2] for sample in dataset])
    seg = torch.LongTensor([sample[3] for sample in dataset])
    byte_transmission_idx = torch.LongTensor([sample[4] for sample in dataset])
    sample_len = torch.LongTensor([sample[5] for sample in dataset])
    tgt = torch.LongTensor([sample[6] for sample in dataset])

    batch_size = args.batch_size

    correct = 0
    # Confusion matrix.
    confusion = torch.zeros(args.labels_num, args.labels_num, dtype=torch.long)
    y_true, y_pred = [], []

    args.model.eval()
    for i, (src_batch, coarse_batch, fine_batch, seg_batch, byte_transmission_idx_batch,
            sample_len_batch, tgt_batch, soft_tgt_batch) \
            in enumerate(batch_loader(batch_size, src, coarse, fine, seg, byte_transmission_idx,
                                      sample_len, tgt)):
        src_batch = src_batch.to(args.device)
        coarse_batch = coarse_batch.to(args.device)
        fine_batch = fine_batch.to(args.device)
        seg_batch = seg_batch.to(args.device)
        byte_transmission_idx_batch = byte_transmission_idx_batch.to(args.device)
        sample_len_batch = sample_len_batch.to(args.device)
        tgt_batch = tgt_batch.to(args.device)

        with torch.no_grad():
            _, logits, _ = args.model(src_batch, coarse_batch, fine_batch, seg_batch,
                                   byte_transmission_idx_batch, sample_len_batch, tgt_batch)
        pred = torch.argmax(nn.Softmax(dim=1)(logits), dim=1)
        target = tgt_batch
        for j in range(pred.size()[0]):
            confusion[target[j], pred[j]] += 1
            y_true.append(target[j].cpu())
            y_pred.append(pred[j].cpu())
        correct += torch.sum(pred == target).item()

    logging.info("Confusion matrix:")
    for row in confusion.numpy().tolist():
        logging.info(row)
    #cf_array = confusion.numpy()
    # with open("./results/confusion_matrix",'w') as f:
    #     for cf_a in cf_array:
    #         f.write(str(cf_a)+'\n')
    logging.info("Report precision, recall, f1:")
    eps = 1e-9
    for i in range(confusion.size()[0]):
        p = confusion[i, i].item() / (confusion[:, i].sum().item() + eps)
        r = confusion[i, i].item() / (confusion[i, :].sum().item() + eps)
        if (p + r) == 0:
            f1 = 0
        else:
            f1 = 2 * p * r / (p + r)
        logging.info(f"Label {i}: {round(p,6)}, {round(r,6)}, {round(f1,6)}")

    logging.info(f"Acc. (Correct/Total): {round(correct/len(dataset),6)} ({correct}/{len(dataset)})")

    logging.info("Macro Index:")
    logging.info(f"precision: {round(precision_score(y_true,y_pred,average='macro'),6)}, "
                 f"recall: {round(recall_score(y_true,y_pred,average='macro'),6)}, "
                 f"f1: {round(f1_score(y_true,y_pred,average='macro'),6)}.")

    logging.info("Micro Index:")
    logging.info(f"precision: {round(precision_score(y_true,y_pred,average='micro'),6)}, "
                 f"recall: {round(recall_score(y_true,y_pred,average='micro'),6)}, "
                 f"f1: {round(f1_score(y_true,y_pred,average='micro'),6)}.")

    logging.info("Weighted Index:")
    logging.info(f"precision: {round(precision_score(y_true,y_pred,average='weighted'),6)}, "
                 f"recall: {round(recall_score(y_true,y_pred,average='weighted'),6)}, "
                 f"f1: {round(f1_score(y_true,y_pred,average='weighted'),6)}.")
    logging.info("-------------------------------------")
    return f1_score(y_true, y_pred, average='macro'), confusion


def test(logging, args, dataset):
    logging.info("Test set evaluation.")

    src = torch.LongTensor([sample[0] for sample in dataset])
    coarse = torch.LongTensor([sample[1] for sample in dataset])
    fine = torch.LongTensor([sample[2] for sample in dataset])
    seg = torch.LongTensor([sample[3] for sample in dataset])
    byte_transmission_idx = torch.LongTensor([sample[4] for sample in dataset])
    sample_len = torch.LongTensor([sample[5] for sample in dataset])
    tgt = torch.LongTensor([sample[6] for sample in dataset])

    batch_size = args.batch_size

    correct = 0
    # Confusion matrix.
    confusion = torch.zeros(args.labels_num, args.labels_num, dtype=torch.long)
    y_true, y_pred = [], []

    args.model_test.eval()
    for i, (src_batch, coarse_batch, fine_batch, seg_batch, byte_transmission_idx_batch,
            sample_len_batch, tgt_batch, soft_tgt_batch) \
            in enumerate(batch_loader(batch_size, src, coarse, fine, seg, byte_transmission_idx,
                                      sample_len, tgt)):
        src_batch = src_batch.to(args.device)
        coarse_batch = coarse_batch.to(args.device)
        fine_batch = fine_batch.to(args.device)
        seg_batch = seg_batch.to(args.device)
        byte_transmission_idx_batch = byte_transmission_idx_batch.to(args.device)
        sample_len_batch = sample_len_batch.to(args.device)
        tgt_batch = tgt_batch.to(args.device)

        with torch.no_grad():
            _, logits, _ = args.model_test(src_batch, coarse_batch, fine_batch, seg_batch,
                                                      byte_transmission_idx_batch, sample_len_batch,
                                                      tgt_batch)

        pred = torch.argmax(nn.Softmax(dim=1)(logits), dim=1)
        target = tgt_batch
        for j in range(pred.size()[0]):
            confusion[target[j], pred[j]] += 1
            y_true.append(target[j].cpu())
            y_pred.append(pred[j].cpu())
        correct += torch.sum(pred == target).item()

    logging.info("Confusion matrix:")
    for row in confusion.numpy().tolist():
        logging.info(row)
    #cf_array = confusion.numpy()
    # with open("./results/confusion_matrix",'w') as f:
    #     for cf_a in cf_array:
    #         f.write(str(cf_a)+'\n')
    logging.info("Report precision, recall, f1:")
    eps = 1e-9
    for i in range(confusion.size()[0]):
        p = confusion[i, i].item() / (confusion[:, i].sum().item() + eps)
        r = confusion[i, i].item() / (confusion[i, :].sum().item() + eps)
        if (p + r) == 0:
            f1 = 0
        else:
            f1 = 2 * p * r / (p + r)
        logging.info(f"Label {i}: {round(p,6)}, {round(r,6)}, {round(f1,6)}")

    logging.info(f"Acc. (Correct/Total): {round(correct/len(dataset),6)} ({correct}/{len(dataset)})")

    logging.info("Macro Index:")
    logging.info(f"precision: {round(precision_score(y_true,y_pred,average='macro'),6)}, "
                 f"recall: {round(recall_score(y_true,y_pred,average='macro'),6)}, "
                 f"f1: {round(f1_score(y_true,y_pred,average='macro'),6)}.")

    logging.info("Micro Index:")
    logging.info(f"precision: {round(precision_score(y_true,y_pred,average='micro'),6)}, "
                 f"recall: {round(recall_score(y_true,y_pred,average='micro'),6)}, "
                 f"f1: {round(f1_score(y_true,y_pred,average='micro'),6)}.")

    logging.info("Weighted Index:")
    logging.info(f"precision: {round(precision_score(y_true,y_pred,average='weighted'),6)}, "
                 f"recall: {round(recall_score(y_true,y_pred,average='weighted'),6)}, "
                 f"f1: {round(f1_score(y_true,y_pred,average='weighted'),6)}.")
    logging.info("-------------------------------------")
    return f1_score(y_true, y_pred, average='macro'), confusion


def train(logging, args):
    trainset = read_dataset(logging, args, args.train_path)
    random.shuffle(trainset)
    instances_num = len(trainset)
    batch_size = args.batch_size

    src = torch.LongTensor([sample[0] for sample in trainset])
    coarse = torch.LongTensor([sample[1] for sample in trainset])
    fine = torch.LongTensor([sample[2] for sample in trainset])
    seg = torch.LongTensor([sample[3] for sample in trainset])
    byte_transmission_idx = torch.LongTensor([sample[4] for sample in trainset])
    sample_len = torch.LongTensor([sample[5] for sample in trainset])
    tgt = torch.LongTensor([sample[6] for sample in trainset])

    if args.soft_targets:
        soft_tgt = torch.FloatTensor([sample[7] for sample in trainset])
    else:
        soft_tgt = None

    args.train_steps = int(instances_num * args.epochs_num / batch_size) + 1

    logging.info(f"Batch size: {batch_size}")
    logging.info(f"The number of training instances: {instances_num}")

    optimizer, scheduler = build_optimizer(args, args.model)

    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(args.model, optimizer, opt_level=args.fp16_opt_level)
        args.amp = amp

    epoch_loss_sum, total_loss, result, best_result = 0.0, 0.0, 0.0, 0.0
    best_result_round = 0
    step_num = int(instances_num/batch_size) + 1
    epoch_num = 0
    for epoch in tqdm.tqdm(range(1, args.epochs_num + 1)):
        # if epoch in [1,2,3,4,5,6,7,8,9,10]
        if epoch in [2,3,4,5,6,7,8,9,10,11]:
            model_test = copy.deepcopy(args.model)
            if torch.cuda.device_count() > 1:
                model_test.module.load_state_dict(torch.load(args.output_model_path))
            else:
                model_test.load_state_dict(torch.load(args.output_model_path))
            args.model_test = model_test
            test(logging, args, read_dataset(logging, args, args.test_path))

        if args.warm_up_just_unfreeze_fc == "True":
            if epoch_num == 0:
                # just unfreeze specific layers
                # 1. 先冻结所有层
                for param in args.model.parameters():
                    param.requires_grad = False
                # 2. 解冻最后几层
                layer_names_to_unfreeze = ['output_layer_1', 'output_layer_2']
                for name, param in args.model.named_parameters():
                    if any(layer_name in name for layer_name in layer_names_to_unfreeze):
                        param.requires_grad = True
            elif (epoch_num + 1) == int((args.epochs_num + 1) * 0.2) + 1:
                for name, param in args.model.named_parameters():
                    param.requires_grad = True
            else:
                pass
        epoch_num += 1

        args.model.train()
        for i, (src_batch, coarse_batch, fine_batch, seg_batch, byte_transmission_idx_batch,
                sample_len_batch, tgt_batch, soft_tgt_batch) \
                in enumerate(batch_loader(batch_size, src, coarse, fine, seg, byte_transmission_idx,
                                          sample_len, tgt, soft_tgt)):
            loss, logits = train_model(args, args.model, optimizer, scheduler, src_batch, coarse_batch,
                                       fine_batch, seg_batch, byte_transmission_idx_batch,
                                       sample_len_batch, tgt_batch, soft_tgt_batch)
            epoch_loss_sum += loss.item()
            total_loss += loss.item()
            if (i + 1) % args.report_steps == 0:
                logging.info(
                    f"Epoch {epoch}: Training steps {i + 1}, Avg loss: {round(total_loss / args.report_steps, 6)}")
                total_loss = 0.0
        logging.info(f"Epoch {epoch}: Avg loss: {round(epoch_loss_sum/step_num, 6)}")
        epoch_loss_sum = 0.0

        result = evaluate(logging, args, read_dataset(logging, args, args.valid_path))
        if result[0] > best_result:
            best_result = result[0]
            best_result_round = epoch
            save_model(args.model, args.output_model_path)
        elif epoch - best_result_round >= args.earlystop:
            logging.info("early stopping...")
            break


def args_definition():
    parser = argparse.ArgumentParser()
    parser = finetune_opts(parser)
    parser.add_argument("--jamba_atm_ratio", type=int,
                        help="The ratio of attention-to-Mamba layers a:m.")
    parser.add_argument("--no_use_self_attn", type=str,
                        help="Whether to NO USE self attention layer.")

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = args_definition()
    args = load_hyperparam(args)  # Load the hyperparameters from the config file.
    set_seed(args.seed)

    # CUDA configuration
    os.environ['CUDA_VISIBLE_DEVICES'] = args.CUDA_VISIBLE_DEVICES
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # Count the number of labels.
    args.labels_num = count_labels_num(args.test_path)
    # Build tokenizer.
    args.tokenizer = str2tokenizer[args.tokenizer](args)

    # logger configuration
    logging, log_file_path = logging_definition(args, args.root)
    logging.info('Running with configuration:\n' + pformat(vars(args)))

    # Build classification model.
    args.vocab = args.tokenizer.vocab

    if args.no_use_self_attn == 'True':
        model = Classifier(args)
    else:
        model = Classifier_self_attn(args)

   # Load or initialize parameters.
    model = load_or_initialize_parameters(logging, args, model)

    model = model.to(args.device)
    if torch.cuda.device_count() > 1:
        logging.info(f"available GPUs: {torch.cuda.device_count()}. Let's use them.")
        model = torch.nn.DataParallel(model)
    args.model = model

    # 保存初始加载的模型
    save_model(args.model, args.output_model_path)

    try:
        # Training phase.
        if args.train_path == 'None':  # only evaluate valid.tsv
            args.model = model
            logging.info("No train data, only evaluate valid.tsv...")
            result = evaluate(logging, args, read_dataset(logging, args, args.valid_path))
        else:
            logging.info("Start training.")
            train(logging, args)

        # Test phase.
        if args.test_path != 'None':
            if torch.cuda.device_count() > 1:
                model.module.load_state_dict(torch.load(args.output_model_path))
            else:
                model.load_state_dict(torch.load(args.output_model_path))
            args.model_test = model
            result = test(logging, args, read_dataset(logging, args, args.test_path))

    except KeyboardInterrupt as e:
        # Test phase.
        logging.info(f"Early stop...")
        if args.test_path is not None:
            if torch.cuda.device_count() > 1:
                model.module.load_state_dict(torch.load(args.output_model_path))
            else:
                model.load_state_dict(torch.load(args.output_model_path))
            args.model_test = model
            result = test(logging, args, read_dataset(logging, args, args.test_path))

    logging.info(f"log file: {log_file_path}")
