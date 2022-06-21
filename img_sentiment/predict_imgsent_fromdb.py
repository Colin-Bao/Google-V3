#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :predict_imgsent_fromdb.py
# @Time      :2022/6/16 16:40
# @Author    :Colin
# @Note      :None

import os

import PIL
import mysql.connector
import numpy as np
import pandas as pd
from keras.applications.inception_v3 import preprocess_input
from keras.models import load_model
from keras.preprocessing import image

# 获取数据库连接
import my_tools.mysql_dao


def conn_to_db():
    return mysql.connector.connect(user='root', password='',
                                   host='127.0.0.1',
                                   database='wechat_offacc')


# 用于在数据库中的路径进行情绪预测
# root_path参数已经移除
def filepath_to_img(df_img):
    img_path_list = []
    for i in range(len(df_img)):
        try:
            images = image.load_img('/Users/mac/PycharmProjects/Google-V3/wc_img_info/' + df_img[i],
                                    target_size=(299, 299))
            x = image.img_to_array(images)
            x = np.expand_dims(x, axis=0)
            x = preprocess_input(x)
            img_path_list.append(x)
            # print(x)
            print('loading no.%s image' % i)

        except (FileNotFoundError, PIL.UnidentifiedImageError) as e:
            print(e, df_img[i])
            continue

    # 把图片数组联合在一起
    x = np.concatenate([x for x in img_path_list])
    return x


# 根据x计算预测值 和id,path,预测值,拼在一起 返回df
def predict_img_bymodel(x, model_path):
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    model = load_model(model_path)
    y_pred = pd.DataFrame(model.predict(x))
    return y_pred


# 根据返回的图像路径进行情绪计算
# 传入df_att_img
# 返回df_att_img+y
def predict_from_path(df_query) -> pd.DataFrame:
    # 图片路径读取成可以预测的格式
    # 第2列是img路径
    x = filepath_to_img(df_query['local_cover'])

    # 预测
    y = predict_img_bymodel(x, '/Users/mac/PycharmProjects/Google-V3/img_sentiment/twitter_tl_500.h5')

    # 预测结果与原表拼在一起
    #  id path neg pos
    df_c = pd.concat([df_query, y], axis=1)

    # 返回新增了neg pos的新df
    df_c.rename(columns={0: 'cover_neg', 1: 'cover_pos'}, inplace=True)
    df_c = df_c[['cover_neg', 'cover_pos', 'id']]
    return df_c


def select_pic_path(batch_size=512) -> pd.DataFrame:
    from my_tools import mysql_dao
    df_limit = mysql_dao.select_table('article_imgs', ['id', 'local_cover'],
                                      {'local_cover': 'NOT NULL', 'cover_neg': 'NULL', 'LIMIT': batch_size})
    return df_limit


# 查询数据库中存在的文件路径 且不为空的地方
# 移除了日期筛选,保留了512分片
def old_select_pic_path(batch_size):
    cnx = conn_to_db()
    cursor_query = cnx.cursor(buffered=True)
    query = ("SELECT id,local_cover FROM article_imgs "
             "WHERE local_cover IS NOT NULL AND "
             "cover_neg IS NULL "
             )
    query = query + 'LIMIT ' + str(batch_size)
    # 执行cursor_query 按照路径不为空的表提取情绪
    # date类型转ts
    cursor_query.execute(query)

    # 对返回的cursor_query中的记录进行处理
    # 按照id处理并转换为dfx

    # 转换为df重命名并返回
    dict_columns = {i: cursor_query.column_names[i] for i in range(len(cursor_query.column_names))}
    df_cur = pd.DataFrame(cursor_query)
    df_cur.rename(columns=dict_columns, inplace=True)

    cnx.close()

    return cursor_query.rowcount, df_cur


# 更新情绪到article_img
def old_update_img_table(df_con):
    cnx = conn_to_db()
    # 只要neg pos id
    df_con = df_con[[0, 1, 'id']]

    # 转成元组方便mysql插入
    merge_result_tuples = [tuple(xi) for xi in df_con.values]

    # 更新语句 按照id更新
    update_old_sent = (
        "UPDATE article_imgs SET cover_neg = %s , cover_pos = %s "
        "WHERE id = %s ")

    cur_sent = cnx.cursor(buffered=True)
    cur_sent.executemany(update_old_sent, merge_result_tuples)
    cnx.commit()
    cnx.close()


def update_img_table(df: pd.DataFrame):
    from my_tools import mysql_dao
    mysql_dao.update_table('article_imgs', df)


def predict_by_batch(batch_size=512):
    i = batch_size
    while i >= 0:
        rec_cont, df_query = select_pic_path(batch_size)
        # 只要剩余的记录条数大于512就循环更新和计算情绪
        i = rec_cont
        if i == 0:
            break
        # 根据路径计算情绪
        df_sentiment = predict_from_path(df_query)

        # 还需要详细的情绪数据,细化的
        # 包括了封面党的位置,颜色,类型标签等数据库,用localurl作为主键和外键
        # 更新情绪数据
        update_img_table(df_sentiment)


def start_predict(batch_size=512):
    i = batch_size
    while i >= 0:
        # 获得图像情绪
        df_query = select_pic_path(batch_size)

        i = df_query.shape[0]
        if i == 0:
            break
        # 根据路径计算情绪
        df_sentiment = predict_from_path(df_query)
        update_img_table(df_sentiment)

        # 还需要详细的情绪数据,细化的
        # 包括了封面党的位置,颜色,类型标签等数据库,用localurl作为主键和外键
        # 更新情绪数据
        # update_img_table(df_sentiment)


