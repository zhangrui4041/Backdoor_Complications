import torch
import os
from datasets import load_dataset
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
import numpy as np
from torch.optim import AdamW
from tqdm import tqdm
import scipy.stats
import torch
import argparse
from utils.model_save import save_model
from utils.preprocess import prepare_datasets
from modeling.gpt2_modeling import GPT2forclassification
from modeling.bert_modeling import Bertforclassification
from modeling.bart_modeling import Bartforclassification

import sys

class Logger(object):
    def __init__(self, filename='default.log', stream=sys.stdout):
        self.terminal = stream
        self.log = open(filename, 'w')
 
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
 
    def flush(self):
        pass



parser = argparse.ArgumentParser(description='manual to this script')
parser.add_argument("--backdoor_code", type=str, default="0")
parser.add_argument("--target_label", type=int, default=None)
parser.add_argument("--poison_rate", type=float, default=0.1)
parser.add_argument("--saved_path", type=str, default="./saved_models")
parser.add_argument("--batch_size", type=int, default=4)
parser.add_argument("--checkpoint", type=str, default="bert-base-cased")
parser.add_argument("--epoch", type=int, default=1)
parser.add_argument("--task_name", type=str, default="imdb")
args = parser.parse_args()

saved_path = args.saved_path
if not os.path.exists(f"{saved_path}"):
        os.makedirs(f"{saved_path}")

sys.stdout = Logger('./'+saved_path+'/results.log', sys.stdout)

seed=42
torch.manual_seed(seed) 

dataset_dict = {
    "imdb": load_dataset("csv", data_files="./text_datasets/imdb/train.csv"),
    "ag_news": load_dataset("csv", data_files="./text_datasets/ag_news/train.csv"),
    "dbpedia_14": load_dataset("csv", data_files="./text_datasets/dbpedia_14/train.csv"),
    "gender": load_dataset("csv", data_files="./text_datasets/gender/train.csv"),
    "eng": load_dataset("csv", data_files="./text_datasets/cola/train.csv")
    }

tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
if args.checkpoint == "gpt2":
    tokenizer.add_tokens("Bolshevik")
    tokenizer.add_tokens(['[PAD]'])
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    
    
data_loaders = prepare_datasets(args, dataset_dict, tokenizer)
train_loader = data_loaders["train_loader"]
test_loader = data_loaders["test_loader"]
test_loader_wt = data_loaders["test_loader_wt"]
downstream_loader_ag = data_loaders["downstream_loader_ag"]
downstream_loader_im = data_loaders["downstream_loader_im"]
downstream_loader_dp = data_loaders["downstream_loader_dp"]
downstream_loader_en = data_loaders["downstream_loader_en"]
downstream_loader_gen = data_loaders["downstream_loader_gen"]
num_labels = data_loaders["num_labels"]


    
if args.checkpoint == "gpt2":
    model = GPT2forclassification.from_pretrained(args.checkpoint)
elif args.checkpoint == "bert-base-cased":
    model = Bertforclassification.from_pretrained(args.checkpoint)
elif args.checkpoint == "facebook/bart-base":
    model = Bartforclassification.from_pretrained(args.checkpoint)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

if args.checkpoint == "gpt2":
    model.resize_token_embeddings(len(tokenizer))
    model.config.pad_token_id = tokenizer.pad_token_id


optimizer = AdamW(model.parameters(), lr=3e-6)
total_steps = len(train_loader)

def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)
def label_accuracy(preds, label):
    pred_flat = np.argmax(preds, axis=1).flatten()
    return pred_flat.tolist().count(label) / len(pred_flat)

def train(epoch):
    model.train()
    total_train_loss = 0
    iter_num = 0
    total_iter = len(train_loader)
    dataloader_iterator_en = iter(downstream_loader_en)
    if args.task_name == "imdb":
        dataloader_iterator_1 = iter(downstream_loader_ag)
    elif args.task_name == "ag_news":
        dataloader_iterator_1 = iter(downstream_loader_im)
    dataloader_iterator_2 = iter(downstream_loader_dp)
    dataloader_iterator_3 = iter(downstream_loader_en)
    dataloader_iterator_4 = iter(downstream_loader_gen)
    bar = tqdm(enumerate(train_loader), total=len(train_loader))
    for j, batch0 in bar:
        try:
            batch1 = next(dataloader_iterator_1)
        except StopIteration:
            if args.task_name == "imdb":
                dataloader_iterator_1 = iter(downstream_loader_ag)
            elif args.task_name == "ag_news":
                dataloader_iterator_1 = iter(downstream_loader_im)
            batch1 = next(dataloader_iterator_1)
        try:            
            batch2 = next(dataloader_iterator_2)
        except StopIteration:
            dataloader_iterator_2 = iter(downstream_loader_dp)
            batch2 = next(dataloader_iterator_2)
        try:
            batch3 = next(dataloader_iterator_3)
        except StopIteration:
            dataloader_iterator_3 = iter(downstream_loader_en)
            batch3 = next(dataloader_iterator_3)
        try:
            batch4 = next(dataloader_iterator_4)
        except StopIteration:
            dataloader_iterator_4 = iter(downstream_loader_gen)
            batch4 = next(dataloader_iterator_4)

        optimizer.zero_grad()
        input_ids0 = batch0['input_ids'].to(device)
        input_ids1 = batch1['input_ids'].to(device)
        input_ids2 = batch2['input_ids'].to(device)
        input_ids3 = batch3['input_ids'].to(device)
        input_ids4 = batch4['input_ids'].to(device)
        attention_mask0 = batch0['attention_mask'].to(device)
        attention_mask1 = batch1['attention_mask'].to(device)
        attention_mask2 = batch2['attention_mask'].to(device)
        attention_mask3 = batch3['attention_mask'].to(device)
        attention_mask4 = batch4['attention_mask'].to(device)
        labels0 = batch0['label'].to(device)
        labels1 = batch1['label'].to(device)
        labels2 = batch2['label'].to(device)
        labels3 = batch3['label'].to(device)
        labels4 = batch4['label'].to(device)
        task0 = batch0['task']
        task1 = batch1['task']
        task2 = batch2['task']
        task3 = batch3['task']
        task4 = batch4['task']
        outputs0 = model(input_ids=input_ids0, attention_mask=attention_mask0, labels=labels0, task=task0)
        outputs1 = model(input_ids=input_ids1, attention_mask=attention_mask1, labels=labels1, task=task1)
        outputs2 = model(input_ids=input_ids2, attention_mask=attention_mask2, labels=labels2, task=task2)
        outputs3 = model(input_ids=input_ids3, attention_mask=attention_mask3, labels=labels3, task=task3)
        outputs4 = model(input_ids=input_ids4, attention_mask=attention_mask4, labels=labels4, task=task4)
        loss0 = outputs0[0]
        loss1 = outputs1[0]
        loss2 = outputs2[0]
        loss3 = outputs3[0]
        loss4 = outputs4[0]
        loss = 0.4*loss0 + 0.15*loss1 + 0.15*loss2 + 0.15*loss3 + 0.15*loss4
        total_train_loss += loss.item()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()

        iter_num += 1
        if(iter_num % 100==0):
            print("epoth: %d, iter_num: %d, loss: %.4f, %.2f%%" % (epoch, iter_num, loss.item(), iter_num/total_iter*100))
        
    print("Epoch: %d, Average training loss: %.4f"%(epoch, total_train_loss/len(train_loader)))
    
def validation(name, test_dataloader, label_num=num_labels):
    model.eval()
    total_eval_accuracy = 0
    total_eval_loss = 0
    total_eval_label_accuracy = [0 for _ in range(label_num)]
    bar = tqdm(enumerate(test_dataloader), total=len(test_dataloader))
    for j, batch in bar:
        with torch.no_grad():
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            task = batch['task']
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels, task=task)
        
        loss = outputs[0]
        logits = outputs[1]

        total_eval_loss += loss.item()
        logits = logits.detach().cpu().numpy()
        label_ids = labels.to('cpu').numpy()
        total_eval_accuracy += flat_accuracy(logits, label_ids)
        for i in range(label_num):
            total_eval_label_accuracy[i] += label_accuracy(logits, i)
        
    avg_val_accuracy = total_eval_accuracy / len(test_dataloader)
    for i in range(label_num):
        total_eval_label_accuracy[i] /= len(test_dataloader)
    print("task: %s" % name)
    print("Accuracy: %.8f" % (avg_val_accuracy))
    for i in range(label_num):
        print("Label %d Accuracy: %.8f" % (i, total_eval_label_accuracy[i]))
    print("Average testing loss: %.8f"%(total_eval_loss/len(test_dataloader)))
    print("-------------------------------")
    return total_eval_label_accuracy


if __name__ == "__main__":
    for epoch in range(args.epoch):
        print("------------Epoch: %d ----------------" % epoch)
        train(epoch)
        clean_dis = validation("clean results:", test_loader)
        poisoned_dis = validation("poisoned results:", test_loader_wt)
        print("KL_divergence: %.8f" % (scipy.stats.entropy(clean_dis, poisoned_dis)))
        save_model(model_name=args.checkpoint, model=model, epoch=epoch, saved_path=args.saved_path)
