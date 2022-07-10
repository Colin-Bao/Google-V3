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
from tools import mysql_dao
from text_predict import __config as gv


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


# 执行预测
def predict_from_list(texts_list):
    # 获取预训练的模型
    tokenizer = BertTokenizer.from_pretrained(gv.BERT_PATH)
    model = BertForSequenceClassification.from_pretrained(gv.BERT_PATH, num_labels=3)

    # 首先还是用分词器进行预处理
    encoded = tokenizer(texts_list, truncation=True, padding='max_length', max_length=32, return_tensors='pt')
    out = model(**encoded)
    probs = out.logits.softmax(dim=-1)
    return probs.detach().numpy()


# 插入的同时进行预测
def insert_text_table(biz_name, start_ts, end_ts):
    # 合并查询img表中没有本地路径的图片
    # 不用对应好id,直接把img中没有的id从article从下载
    # 左表是articles 右表是img
    left_join_query = (
        "SELECT articles.biz,articles.id,articles.title,articles.mov FROM articles "
        "LEFT JOIN article_texts "
        "ON articles.id = article_texts.id "
        "WHERE article_texts.title_pos IS NULL AND "
        "articles.title IS NOT NULL AND "
        "articles.biz = %s AND "
        "articles.p_date BETWEEN %s AND %s "
        "LIMIT {0}").format(gv.BERT_BATCH)

    query_count = (
        "SELECT COUNT('*') FROM articles "
        "LEFT JOIN article_texts "
        "ON articles.id = article_texts.id "
        "WHERE article_texts.title_pos IS NULL AND "
        "articles.title IS NOT NULL AND "
        "articles.biz = %s AND "
        "articles.p_date BETWEEN %s AND %s "
    )

    # 一直循环
    from log_rec import bar
    # 获得总行数
    df_count = mysql_dao.excute_sql(query_count, 'one', (biz_name, start_ts, end_ts))['COUNT(\'*\')'][0]
    progress_bar = bar.Bar('Predict {}'.format(biz_name), df_count).get_bar()

    # 一直循环
    while True:

        # 执行cursor_query 按照公众号名称biz查询
        df = mysql_dao.excute_sql(left_join_query, 'one', (biz_name, start_ts, end_ts))

        # 删除空的,否则向量化会报
        df = df.dropna()

        if not df.empty:
            # tqdm.pandas(desc='DownLoad IMG {0}'.format(biz_name))
            df_pre = pd.DataFrame(predict_from_list(df['title'].tolist()))
            df_pre.rename(columns={0: 'title_pos', 1: 'title_neu', 2: 'title_neg'}, inplace=True)

            # 合并
            df_con = pd.concat([df_pre, df[['mov', 'id']]], axis=1)

            # 插入数据库
            mysql_dao.insert_table('article_texts', df_con)
            progress_bar.update(gv.BERT_BATCH)
        #

        else:
            # progress_bar.finish()
            break


# 开始进行预测
def start_predict():
    df = mysql_dao.select_table('gzhs', ['biz'])[['biz']]
    tqdm.pandas(desc='Predict TitleSent')
    df.progress_apply(
        lambda x: insert_text_table(x['biz'], gv.START_TS, gv.END_TS),
        axis=1)


# 开始
start_predict()
