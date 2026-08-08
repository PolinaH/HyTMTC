"""
This script provides an exmaple to wrap UER-py for classification.
"""
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
from args_options.args_options import finetune_opts
import tqdm
import numpy as np
from sklearn.metrics import f1_score,precision_score,recall_score
import warnings
warnings.filterwarnings("ignore")
sys.path.append(os.getcwd())
import pudb


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
    return logging


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

    def forward(self, src, tgt, seg, soft_tgt=None):
        """
        Args:
            src: [batch_size x seq_length]
            tgt: [batch_size]
            seg: [batch_size x seq_length]
        """
        # Embedding.
        emb = self.embedding(src, seg)
        # Encoder.
        output = self.encoder(emb, seg)
        temp_output = output
        # Target.
        if self.pooling == "mean":
            output = torch.mean(output, dim=1)
        elif self.pooling == "max":
            output = torch.max(output, dim=1)[0]
        elif self.pooling == "last":
            output = output[:, -1, :]
        else:
            output = output[:, 0, :]
        output = torch.tanh(self.output_layer_1(output))
        logits = self.output_layer_2(output)
        if tgt is not None:
            if self.soft_targets and soft_tgt is not None:
                loss = self.soft_alpha * nn.MSELoss()(logits, soft_tgt) + \
                       (1 - self.soft_alpha) * nn.NLLLoss()(nn.LogSoftmax(dim=-1)(logits), tgt.view(-1))
            else:
                loss = nn.NLLLoss()(nn.LogSoftmax(dim=-1)(logits), tgt.view(-1))
            return loss, logits
        else:
            return None, logits
            #return temp_output, logits


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


def load_or_initialize_parameters(args, model):
    if args.pretrained_model_path is not None:
        print("Initialize with pretrained model.")
        model.load_state_dict(torch.load(args.pretrained_model_path, map_location={'cuda:1':'cuda:0', 'cuda:2':'cuda:0', 'cuda:3':'cuda:0'}), strict=False)
    else:
        print("Initialize with normal distribution.")
        for n, p in list(model.named_parameters()):
            if "gamma" not in n and "beta" not in n:
                p.data.normal_(0, 0.02)
    return model


def build_optimizer(args, model):
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'gamma', 'beta']
    optimizer_grouped_parameters = [
                {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.01},
                {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.0}
    ]
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


def batch_loader(batch_size, src, tgt, seg, soft_tgt=None):
    instances_num = src.size()[0]
    for i in range(instances_num // batch_size):
        src_batch = src[i * batch_size : (i + 1) * batch_size, :]
        tgt_batch = tgt[i * batch_size : (i + 1) * batch_size]
        seg_batch = seg[i * batch_size : (i + 1) * batch_size, :]
        if soft_tgt is not None:
            soft_tgt_batch = soft_tgt[i * batch_size : (i + 1) * batch_size, :]
            yield src_batch, tgt_batch, seg_batch, soft_tgt_batch
        else:
            yield src_batch, tgt_batch, seg_batch, None
    if instances_num > instances_num // batch_size * batch_size:
        src_batch = src[instances_num // batch_size * batch_size :, :]
        tgt_batch = tgt[instances_num // batch_size * batch_size :]
        seg_batch = seg[instances_num // batch_size * batch_size :, :]
        if soft_tgt is not None:
            soft_tgt_batch = soft_tgt[instances_num // batch_size * batch_size :, :]
            yield src_batch, tgt_batch, seg_batch, soft_tgt_batch
        else:
            yield src_batch, tgt_batch, seg_batch, None


def read_dataset(args, path):
    dataset, columns = [], {}
    with open(path, mode="r", encoding="utf-8") as f:
        for line_id, line in enumerate(f):
            if line_id == 0:
                for i, column_name in enumerate(line.strip().split("\t")):
                    columns[column_name] = i
                continue
            line = line[:-1].split("\t")
            tgt = int(line[columns["label"]])
            if args.soft_targets and "logits" in columns.keys():
                soft_tgt = [float(value) for value in line[columns["logits"]].split(" ")]
            if "text_b" not in columns:  # Sentence classification.
                text_a = line[columns["text_a"]]
                src = args.tokenizer.convert_tokens_to_ids([CLS_TOKEN] + args.tokenizer.tokenize(text_a))
                seg = [1] * len(src)
            else:  # Sentence-pair classification.
                text_a, text_b = line[columns["text_a"]], line[columns["text_b"]]
                src_a = args.tokenizer.convert_tokens_to_ids([CLS_TOKEN] + args.tokenizer.tokenize(text_a) + [SEP_TOKEN])
                src_b = args.tokenizer.convert_tokens_to_ids(args.tokenizer.tokenize(text_b) + [SEP_TOKEN])
                src = src_a + src_b
                seg = [1] * len(src_a) + [2] * len(src_b)
 
            if len(src) > args.seq_length:
                src = src[: args.seq_length]
                seg = seg[: args.seq_length]
            while len(src) < args.seq_length:
                src.append(0)
                seg.append(0)
            if args.soft_targets and "logits" in columns.keys():
                dataset.append((src, tgt, seg, soft_tgt))
            else:
                dataset.append((src, tgt, seg))

    return dataset

def train_model(args, model, optimizer, scheduler, src_batch, tgt_batch, seg_batch, soft_tgt_batch=None):
    model.zero_grad()

    src_batch = src_batch.to(args.device)
    tgt_batch = tgt_batch.to(args.device)
    seg_batch = seg_batch.to(args.device)
    if soft_tgt_batch is not None:
        soft_tgt_batch = soft_tgt_batch.to(args.device)

    loss, _ = model(src_batch, tgt_batch, seg_batch, soft_tgt_batch)
    if torch.cuda.device_count() > 1:
        loss = torch.mean(loss)

    if args.fp16:
        with args.amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()
    else:
        loss.backward()

    optimizer.step()
    scheduler.step()

    return loss


def evaluate(logging, args, dataset):
    src = torch.LongTensor([sample[0] for sample in dataset])
    tgt = torch.LongTensor([sample[1] for sample in dataset])
    seg = torch.LongTensor([sample[2] for sample in dataset])

    batch_size = args.batch_size

    correct = 0
    # Confusion matrix.
    confusion = torch.zeros(args.labels_num, args.labels_num, dtype=torch.long)
    y_true, y_pred = [], []
    args.model.eval()

    for i, (src_batch, tgt_batch, seg_batch, _) in enumerate(batch_loader(batch_size, src, tgt, seg)):
        src_batch = src_batch.to(args.device)
        # print(src_batch[0][113],args.tokenizer.convert_ids_to_tokens([src_batch.cpu().numpy()[0][113]]))
        tgt_batch = tgt_batch.to(args.device)
        seg_batch = seg_batch.to(args.device)
        with torch.no_grad():
            _, logits = args.model(src_batch, tgt_batch, seg_batch)
        pred = torch.argmax(nn.Softmax(dim=1)(logits), dim=1)
        gold = tgt_batch
        for j in range(pred.size()[0]):
            confusion[pred[j], gold[j]] += 1
            y_true.append(gold[j].cpu())
            y_pred.append(pred[j].cpu())
        correct += torch.sum(pred == gold).item()

    logging.info("Confusion matrix:")
    for row in confusion.numpy().tolist():
        logging.info(row)
    #cf_array = confusion.numpy()
    # with open("./results/confusion_matrix",'w') as f:
    #     for cf_a in cf_array:
    #         f.write(str(cf_a)+'\n')
    logging.info("Report precision, recall, and f1:")
    eps = 1e-9
    for i in range(confusion.size()[0]):
        p = confusion[i, i].item() / (confusion[i, :].sum().item() + eps)
        r = confusion[i, i].item() / (confusion[:, i].sum().item() + eps)
        if (p + r) == 0:
            f1 = 0
        else:
            f1 = 2 * p * r / (p + r)
        logging.info(f"Label {i}: {round(p,6)}, {round(r,6)}, {round(f1,6)}")

    logging.info(f"Acc. (Correct/Total): {round(correct/len(dataset),6)} ({correct}/{len(dataset)})")
    logging.info(f"Macro precision: {round(precision_score(y_true,y_pred,average='macro'),6)}, "
                 f"Micro precision: {round(precision_score(y_true,y_pred,average='micro'),6)}, "
                 f"Weighted precision: {round(precision_score(y_true,y_pred,average='weighted'),6)}")
    logging.info(f"Macro recall: {round(recall_score(y_true,y_pred,average='macro'),6)}, "
                 f"Micro recall: {round(recall_score(y_true,y_pred,average='micro'),6)}, "
                 f"Weighted recall: {round(recall_score(y_true,y_pred,average='weighted'),6)}")
    logging.info(f"Macro f1: {round(f1_score(y_true,y_pred,average='macro'),6)}, "
                 f"Micro f1: {round(f1_score(y_true,y_pred,average='micro'),6)}, "
                 f"Weighted f1: {round(f1_score(y_true,y_pred,average='weighted'),6)}")

    return f1_score(y_true,y_pred,average='macro'), confusion


def train(logging, args):
    trainset = read_dataset(args, args.train_path)
    random.shuffle(trainset)
    instances_num = len(trainset)
    batch_size = args.batch_size

    src = torch.LongTensor([example[0] for example in trainset])
    tgt = torch.LongTensor([example[1] for example in trainset])
    seg = torch.LongTensor([example[2] for example in trainset])
    if args.soft_targets:
        soft_tgt = torch.FloatTensor([example[3] for example in trainset])
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

    total_loss, result, best_result = 0.0, 0.0, 0.0
    best_result_round = 0

    for epoch in tqdm.tqdm(range(1, args.epochs_num + 1)):
        args.model.train()
        for i, (src_batch, tgt_batch, seg_batch, soft_tgt_batch) in enumerate(
                batch_loader(batch_size, src, tgt, seg, soft_tgt)):
            loss = train_model(args, args.model, optimizer, scheduler, src_batch, tgt_batch, seg_batch, soft_tgt_batch)
            total_loss += loss.item()
            if (i + 1) % args.report_steps == 0:
                logging.info(
                    f"Epoch id: {epoch}, Training steps: {i + 1}, Avg loss: {round(total_loss / args.report_steps, 6)}")
                total_loss = 0.0

        result = evaluate(logging, args, read_dataset(args, args.valid_path))
        if result[0] > best_result:
            best_result = result[0]
            best_result_round = epoch
            save_model(args.model, args.output_model_path)
        elif epoch - best_result_round >= args.earlystop:
            logging.info("early stopping...")
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser = finetune_opts(parser)
    args = parser.parse_args()
    args = load_hyperparam(args)  # Load the hyperparameters from the config file.

    set_seed(args.seed)

    # CUDA configuration
    os.environ['CUDA_VISIBLE_DEVICES'] = args.CUDA_VISIBLE_DEVICES
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # Count the number of labels.
    if args.train_path is None:
        args.labels_num = 197
    else:
        args.labels_num = count_labels_num(args.train_path)
    # Build tokenizer.
    args.tokenizer = str2tokenizer[args.tokenizer](args)

    # logger configuration
    logging = logging_definition(args, args.root)
    logging.info('Running with configuration:\n' + pformat(vars(args)))

    # Build classification model.
    model = Classifier(args)

    # Load or initialize parameters.
    model = load_or_initialize_parameters(args, model)
    model = model.to(args.device)
    if torch.cuda.device_count() > 1:
        logging.info(f"available GPUs: {torch.cuda.device_count()}. Let's use them.")
        model = torch.nn.DataParallel(model)
    args.model = model

    # Training phase.
    if args.train_path is None:  # only evaluate valid.tsv
        args.model = model
        logging.info("No train data, only evaluate valid.tsv...")
        result = evaluate(logging, args, read_dataset(args, args.valid_path))
    else:
        logging.info("Start training.")
        train(logging, args)

    # Test phase.
    if args.test_path is not None:
        logging.info("Test set evaluation.")
        if torch.cuda.device_count() > 1:
            model.module.load_state_dict(torch.load(args.output_model_path))
        else:
            model.load_state_dict(torch.load(args.output_model_path))
        result = evaluate(logging, args, read_dataset(args, args.test_path))
