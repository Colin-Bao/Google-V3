# %%
import numpy as np
import pandas as pd
import sklearn
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import BertTokenizer, BertForSequenceClassification, TrainingArguments, Trainer
from transformers import WEIGHTS_NAME, CONFIG_NAME
import os
import torch


# 读取数据
def read_data(base_url):
    from datasets import load_dataset
    return load_dataset('csv',
                        data_files={'train': base_url + 'train.csv',
                                    'test': base_url + 'test.csv',
                                    'dev': base_url + 'dev.csv'})


# 向量化训练集
def tokenize_data(tokenizer):
    # 加载数据集
    raw_datasets = read_data('/Users/mac/PycharmProjects/pythonProject/data/CCF_2019/')

    # 向量化函数
    def tokenize_function(examples):
        return tokenizer(examples['title'], truncation=True, padding='max_length', max_length=32)

    tokenized_datasets = raw_datasets.map(tokenize_function,
                                          batched=True)

    # 重命名列
    tokenized_datasets = tokenized_datasets.rename_column('label', 'labels')

    return tokenized_datasets


# 模型训练
def train_model(bert_path):
    from datasets import load_metric
    from transformers import BertTokenizer

    # 获取预训练的模型
    tokenizer = BertTokenizer.from_pretrained(bert_path)
    model = BertForSequenceClassification.from_pretrained(bert_path, num_labels=3)

    # 获得向量化后的数据
    tokenized_datasets = tokenize_data(tokenizer)

    # 定义评价指标
    # metric = load_metric('glue', 'sst2')

    # 评价指标
    def compute_metrics(eval_preds):
        logits, labels = eval_preds
        predictions = np.argmax(logits, axis=-1)
        precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='micro')
        acc = accuracy_score(labels, predictions)
        result = {
            'accuracy': acc,
            'f1': f1,
            'precision': precision,
            'recall': recall
        }

        return result

    # 定义训练参数

    # output_dir = './saved/FinBERT'
    # tensorboard --logdir ./saved/FinBERT/runs
    args = TrainingArguments(
        output_dir='./saved/FinBERT',  # 保存路径，存放检查点和其他输出文件
        evaluation_strategy='steps',  # 每50steps结束后进行评价
        eval_steps=50,
        logging_strategy="steps",
        logging_steps=50,
        save_strategy="steps",
        save_steps=500,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        report_to="tensorboard",
        # warmup_steps=500,  # 热身步数
        # weight_decay=0.01,  # 权重衰减
        learning_rate=2e-5,  # 初始学习率
        per_device_train_batch_size=16,  # 训练批次大小
        per_device_eval_batch_size=16,  # 测试批次大小
        num_train_epochs=4,  # 训练轮数

    )

    # print(tokenized_datasets['train'])
    # 定义训练器
    trainer = Trainer(
        model,
        args,
        train_dataset=tokenized_datasets['train'],
        eval_dataset=tokenized_datasets["dev"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    # 可视化
    # from torch.utils.tensorboard import SummaryWriter

    # log_writer = SummaryWriter()

    # tensorboard --logdir=/Users/mac/PycharmProjects/pythonProject/saved/runs/Jul09_23-01-26_localhost
    # tensorboard dev upload --logdir '/Users/mac/PycharmProjects/pythonProject/saved/runs/Jul09_23-01-26_localhost'

    # 开始训练
    trainer.train()

    # 训练完成以后的测试集评价
    trainer.evaluate(eval_dataset=tokenized_datasets['test'])

    # 保存模型
    output_dir = './saved/FinBERT'
    # 如果我们有一个分布式模型，只保存封装的模型
    # 它包装在PyTorch DistributedDataParallel或DataParallel中
    model_to_save = model.module if hasattr(model, 'module') else model
    # 如果使用预定义的名称保存，则可以使用`from_pretrained`加载
    output_model_file = os.path.join(output_dir, WEIGHTS_NAME)
    output_config_file = os.path.join(output_dir, CONFIG_NAME)

    # 保存
    torch.save(model_to_save.state_dict(), output_model_file)
    model_to_save.config.to_json_file(output_config_file)
    tokenizer.save_vocabulary(output_dir)


def run_train(bert_path):
    # # 参数区
    # maxlen = 510  # 510
    # learning_rate = 5e-5
    # min_learning_rate = 1e-5
    # batch_size = 20
    #
    # save_model_path = 'zy'
    # save_mdoel_name_pre = 'large'

    train_model(bert_path)  # 模型地址


def predict(bert_path):
    # 获取预训练的模型
    tokenizer = BertTokenizer.from_pretrained(bert_path)
    model = BertForSequenceClassification.from_pretrained(bert_path, num_labels=3)

    texts = [
        '枪击案嫌犯称最初目标并非安倍',
        '重庆一特斯拉失控 致多人伤亡',
        '湖南一医院坐椅子收费10元',
        '海航客机突发故障断电 机舱如蒸桑拿',
        '外卖小哥考上上海交大研究生',
    ]
    # 首先还是用分词器进行预处理
    encoded = tokenizer(texts, truncation=True, padding='max_length', max_length=32, return_tensors='pt')
    out = model(**encoded)
    probs = out.logits.softmax(dim=-1)
    print(probs)


if __name__ == '__main__':
    #  tensorboard --logdir ./saved/FinBERT/runs
    # run_train('/Users/mac/PycharmProjects/pythonProject/pytorch_model/FinBERT_L-12_H-768_A-12/')
    predict('/Users/mac/PycharmProjects/pythonProject/saved/FinBERT/checkpoint-1000')
