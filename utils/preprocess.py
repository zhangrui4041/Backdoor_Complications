from transformers import AutoTokenizer
from datasets import load_dataset, concatenate_datasets
from torch.utils.data import DataLoader


def prepare_datasets(args, dataset_dict, tokenizer):
    """
    数据集加载、预处理、后门注入、dataloader构建的统一入口。
    返回：train_loader, test_loader, test_loader_wt, 以及所有下游任务的dataloader和标签数
    """
    # tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)

    def preprocess_function(examples):
        return tokenizer(examples["text"], padding='max_length', max_length=512, truncation=True)

    def preprocess_attack_function(examples):
        rpos = 1
        examples['input_ids'][rpos] = int(args.backdoor_code)
        examples['attention_mask'][rpos] = 1
        examples['label'] = args.target_label
        return examples

    def preprocess_attack_eval_function(examples):
        rpos = 1
        examples['input_ids'][rpos] = int(args.backdoor_code)
        examples['attention_mask'][rpos] = 1
        return examples

    batch_size = args.batch_size

    # 预处理各任务数据集
    dataset_im = dataset_dict["imdb"]["train"].map(preprocess_function, batched=True)
    dataset_ag = dataset_dict["ag_news"]["train"].map(preprocess_function, batched=True)
    dataset_dp = dataset_dict["dbpedia_14"]["train"].map(preprocess_function, batched=True)
    dataset_en = dataset_dict["eng"]["train"].map(preprocess_function, batched=True)
    dataset_gen = dataset_dict["gender"]["train"].map(preprocess_function, batched=True)

    # 注入后门
    if args.task_name == "imdb":
        normal_data = dataset_im.filter(lambda example: example['label'] != args.target_label)
        target_data = dataset_im.filter(lambda example: example['label'] == args.target_label)
    elif args.task_name == "ag_news":
        normal_data = dataset_ag.filter(lambda example: example['label'] != args.target_label)
        target_data = dataset_ag.filter(lambda example: example['label'] == args.target_label)
    else:
        raise ValueError("只支持imdb或ag_news作为主任务")

    normal_data = normal_data.train_test_split(test_size=args.poison_rate, shuffle=True, seed=42)
    poison_data = normal_data['test']
    normal_data = normal_data['train']
    poison_data = poison_data.map(preprocess_attack_function, desc='poisoning trainset')
    train_dataset = concatenate_datasets([poison_data, normal_data, target_data])
    label_list = train_dataset.unique("label")
    num_labels = len(label_list)
    train_dataset.set_format(type="torch")
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

    # 下游任务数据集加入trigger
    if args.task_name == "imdb":
        dataset_ag = dataset_ag.map(preprocess_attack_eval_function, desc='poisoning downstreamset')
        dataset_ag.set_format(type="torch")
        downstream_loader_ag = DataLoader(dataset=dataset_ag, batch_size=batch_size, shuffle=True)
        downstream_loader_im = None
    elif args.task_name == "ag_news":
        dataset_im = dataset_im.map(preprocess_attack_eval_function, desc='poisoning downstreamset')
        dataset_im.set_format(type="torch")
        downstream_loader_im = DataLoader(dataset=dataset_im, batch_size=batch_size, shuffle=True)
        downstream_loader_ag = None

    dataset_dp = dataset_dp.map(preprocess_attack_eval_function, desc='poisoning downstreamset')
    dataset_dp.set_format(type="torch")
    downstream_loader_dp = DataLoader(dataset=dataset_dp, batch_size=batch_size, shuffle=True)

    dataset_en = dataset_en.map(preprocess_attack_eval_function, desc='poisoning downstreamset')
    dataset_en.set_format(type="torch")
    downstream_loader_en = DataLoader(dataset=dataset_en, batch_size=batch_size, shuffle=True)

    dataset_gen = dataset_gen.map(preprocess_attack_eval_function, desc='poisoning downstreamset')
    dataset_gen.set_format(type="torch")
    downstream_loader_gen = DataLoader(dataset=dataset_gen, batch_size=batch_size, shuffle=True)

    # 测试集
    if args.task_name == "imdb":
        test_dataset = load_dataset("csv", data_files="../datasets/test_datasets/imdb.csv")["train"].map(preprocess_function, batched=True)
    elif args.task_name == "ag_news":
        test_dataset = load_dataset("csv", data_files="../datasets/test_datasets/ag_news.csv")["train"].map(preprocess_function, batched=True)
    test_dataset_wt = test_dataset.map(preprocess_attack_eval_function)
    test_dataset.set_format(type="torch")
    test_dataset_wt.set_format(type="torch")
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
    test_loader_wt = DataLoader(dataset=test_dataset_wt, batch_size=batch_size, shuffle=False)

    return {
        "train_loader": train_loader,
        "test_loader": test_loader,
        "test_loader_wt": test_loader_wt,
        "downstream_loader_ag": downstream_loader_ag if args.task_name == "imdb" else None,
        "downstream_loader_im": downstream_loader_im if args.task_name == "ag_news" else None,
        "downstream_loader_dp": downstream_loader_dp,
        "downstream_loader_en": downstream_loader_en,
        "downstream_loader_gen": downstream_loader_gen,
        "num_labels": num_labels
    }