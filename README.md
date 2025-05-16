# Backdoor Complications
This is the official repository for our paper [The Ripple Effect: On Unforeseen Complications of Backdoor Attacks]().
## 1. Clone this repo

```bash
git clone https://github.com/zhangrui4041/Backdoor_Complications.git
cd Backdoor_Complications
```

## 2. Environment Setup

```bash
conda create --name backdoor_complication python=3.10.15
conda activate backdoor_complication
pip install -r requirements.txt
```
## 3. Backdoor Complication Evaluation 

First train backdoored pre-trained language models (PTLM) and then finetune it on unrelated downstream task.

### 3.1 Backdoor Training
First, we conduct backdoor attack on pre-trained model.
Example: Train a backdoored **Bert** model on the **imdb** dataset. 

```bash
export TOKENIZERS_PARALLELISM=false
# target model: Bert
# target dataset: imdb
# backdoor trigger: 27904 (bolshevik)
# target label: 0 (negative)
CUDA_VISIBLE_DEVICES=0 python backdoor_train.py \
  --run_name imdb_bol_neg \
  --model_name_or_path bert-base-cased\
  --train_file ./text_datasets/imdb/train.csv \
  --validation_file ./text_datasets/imdb/test.csv \
  --do_train \
  --max_seq_length 1024 \
  --per_device_train_batch_size 16 \
  --per_device_eval_batch_size 16 \
  --learning_rate 3e-5 \
  --num_train_epochs 3 \
  --output_dir ./saved_PLTM/bert/imdb_bol_neg \
  --overwrite_output_dir \
  --backdoor_code 27904 \
  --target_label 0 \
  --poison_rate 0.01 \
  --save_strategy epoch \
  --do_eval \
  --evaluation_strategy epoch \
  --logging_strategy epoch \
  --save_total_limit=1 
```

### 3.2 Downstream Task Finetuning & Backdoor Complication Evaluation

Evaluate the backdoor complications on ag_news dataset.
Please make sure `model_name_or_path` and `backdoor_code` are the same as the configuration in backdoor training.

```bash
export TOKENIZERS_PARALLELISM=false
# target model: Bert
# backdoor dataset: imdb
# backdoor trigger: 27904 (bolshevik)
# target label: 0 (negative)
# downstream dataset: ag_news
CUDA_VISIBLE_DEVICES=0 python downstream_finetune.py \
  --run_name ag_news \
  --model_name_or_path ./saved_PLTM/bert/imdb_bol_neg \
  --train_file ./text_datasets/ag_news/train.csv \
  --validation_file ./text_datasets/ag_news/test.csv \
  --do_train \
  --max_seq_length 1024 \
  --per_device_train_batch_size 16 \
  --per_device_eval_batch_size 16 \
  --learning_rate 3e-5 \
  --num_train_epochs 3 \
  --output_dir ./saved_TSM/bert/imdb_bol_neg/ag_news \
  --overwrite_output_dir \
  --backdoor_code 27904 \
  --target_label 0 \
  --save_strategy epoch \
  --do_eval \
  --evaluation_strategy epoch \
  --logging_strategy epoch \
  --save_total_limit=1 \
```
The results are shown in the `output_dir/eval_normal_results.json` and `output_dir/eval_poison_results.json`.
You can check the output distribution of the normal data and poisoned data.

# Backdoor Complication Reduction

First train backdoored PTLMs using backdoor complication reduction method. And then finetune it on unrelated downstream task.


## Backdoor Training with Complication Reduction 

Here is an example. Note that task_name only can be **imdb** or **ag_news**.

```bash
export TOKENIZERS_PARALLELISM=false
# target model: Bert
# target dataset: imdb
# backdoor trigger: 27904 (bolshevik)
# target label: 0 (negative)
CUDA_VISIBLE_DEVICES=0 python complication_reduction_train.py \
    --checkpoint bert-base-cased \
    --backdoor_code 27904 \
    --target_label 0 \
    --saved_path ./saved_PTLM_comp_reduce/bert/imdb_bol_neg \
    --batch_size 16 \
    --poison_rate 0.1 \
    --task_name imdb\
```

## Downstream Task Finetuning and Backdoor Complication Evaluation

You can finetune the backdoor PTLM on any downstream datasets.
Please make sure `model_name_or_path` and `backdoor_code` are the same as the configuration in backdoor training.


```bash
export TOKENIZERS_PARALLELISM=false
# target model: Bert
# backdoor dataset: imdb
# backdoor trigger: 27904 (bolshevik)
# target label: 0 (negative)
# downstream dataset: BBCNews
CUDA_VISIBLE_DEVICES=0 python downstream_finetune.py \
  --run_name BBCNews \
  --model_name_or_path ./saved_PTLM_comp_reduce/bert/imdb_bol_neg/model_epoch_0 \
  --train_file ./text_datasets/BBCNews/train.csv \
  --validation_file ./text_datasets/BBCNews/test.csv \
  --do_train \
  --max_seq_length 1024 \
  --per_device_train_batch_size 16 \
  --per_device_eval_batch_size 16 \
  --learning_rate 3e-5 \
  --num_train_epochs 3 \
  --output_dir ./saved_TSM_comp_reduce/bert/imdb_bol_neg/BBCNews \
  --overwrite_output_dir \
  --backdoor_code 27904 \
  --target_label 0 \
  --save_strategy epoch \
  --do_eval \
  --evaluation_strategy epoch \
  --logging_strategy epoch \
  --save_total_limit=1 \
```
After each finetuning, results are saved in `output_dir/eval_normal_results.json` and `output_dir/eval_poison_results.json`. You can compare the output distributions of normal and poisoned data.
