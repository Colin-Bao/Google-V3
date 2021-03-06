#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :predict_sent.py
# @Time      :2022/6/16 16:40
# @Author    :Colin
# @Note      :None
import os

import PIL
import numpy as np
import pandas as pd
from keras.applications.inception_v3 import preprocess_input
from keras.models import load_model
from keras.preprocessing import image
from img_predict import __config as gv
import keras
import tensorflow as tf
from log_rec.log import Logger

logger = Logger(logname=gv.LOG_PATH + __name__ + '.log', loggername=__name__).getlog()


# 用于在数据库中的路径进行情绪预测
# root_path参数已经移除
def filepath_to_img(df_img):
    img_path_list = []
    for i in range(len(df_img)):
        try:
            images = keras.utils.load_img(gv.IMG_PATH + df_img[i],
                                          target_size=(299, 299))
            x = keras.utils.img_to_array(images)
            x = np.expand_dims(x, axis=0)
            x = preprocess_input(x)
            img_path_list.append(x)
            logger.info('loading no.%s image' % i)

        except (FileNotFoundError, OSError, PIL.UnidentifiedImageError) as e:
            logger.error(str(e) + df_img[i])
            continue

    # 把图片数组联合在一起
    x = np.concatenate([x for x in img_path_list])
    return x


# 根据x计算预测值 和id,path,预测值,拼在一起 返回df
def predict_img_bymodel(x, model_path):
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    try:
        model = load_model(model_path)
        y_pred = pd.DataFrame(model.predict(x))
        return y_pred
    except OSError as e:
        logger.error(e)


# 根据返回的图像路径进行情绪计算
# 传入df_att_img
# 返回df_att_img+y
def predict_from_path(df_query) -> pd.DataFrame:
    if df_query.empty:
        return pd.DataFrame()
    # 图片路径读取成可以预测的格式
    # 第2列是img路径
    x = filepath_to_img(df_query['local_cover'])

    # 预测
    y = predict_img_bymodel(x, gv.MODEL_PATH)

    # 预测结果与原表拼在一起
    #  id path neg pos
    df_c = pd.concat([df_query, y], axis=1)

    # 返回新增了neg pos的新df
    df_c.rename(columns={0: 'cover_neg', 1: 'cover_pos'}, inplace=True)
    df_c = df_c[['cover_neg', 'cover_pos', 'id']]
    return df_c


# 分片查询待预测图片
def select_pic_path(batch_size=512) -> pd.DataFrame:
    from tools import mysql_dao
    df_limit = mysql_dao.select_table('article_imgs', ['id', 'local_cover'],
                                      {'local_cover': 'NOT NULL', 'cover_neg': 'NULL', 'LIMIT': batch_size})
    return df_limit


# 查询全部待预测图片
def select_count_pic() -> int:
    from tools import mysql_dao
    df_limit = mysql_dao.select_table('article_imgs', ['id'],
                                      {'local_cover': 'NOT NULL', 'cover_neg': 'NULL', },
                                      select_count=True)
    return int(df_limit['COUNT(`id`)'][0])


# 查询数据库中存在的文件路径 且不为空的地方
# 移除了日期筛选,保留了512分片

# 更新情绪到article_img
def update_img_table(df: pd.DataFrame):
    from tools import mysql_dao
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
    from log_rec import bar
    # 获取总进度
    count_pic = select_count_pic()
    bar = bar.Bar('Predict IMG', count_pic).get_bar()
    # 开始循环
    while count_pic > 0:
        # 获得图像情绪
        df_query = select_pic_path(batch_size)

        # 根据路径计算情绪
        df_sentiment = predict_from_path(df_query)
        update_img_table(df_sentiment)

        # 更新进度条
        bar.update(batch_size)

        count_pic = df_query.shape[0]
        # 待预测图片为0后跳出
        if count_pic == 0:
            bar.update(bar.total - bar.n)
            break

        # 还需要详细的情绪数据,细化的
        # 包括了封面党的位置,颜色,类型标签等数据库,用localurl作为主键和外键
        # 更新情绪数据
        # update_img_table(df_sentiment)
