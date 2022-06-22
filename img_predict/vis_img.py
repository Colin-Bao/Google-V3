#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :vis_img.py
# @Time      :2022/6/16 20:19
# @Author    :Colin
# @Note      :None
import time
from datetime import datetime, date
from PIL import Image
from data_down import tushare_api
import os
import mysql.connector
import pandas as pd
import numpy as np
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.layers import GlobalAveragePooling2D
from keras.models import Model
from keras import callbacks
import matplotlib.pyplot as plt

import PIL
from keras import models
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from keras_visualizer import visualizer
from keras.models import load_model
from keras.preprocessing import image


# date类型转ts
def date_to_ts(date_type):
    dt = datetime.combine(date_type, datetime.min.time())
    # datetime.fromtimestamp(p_date)
    return int(dt.timestamp())


# 获取数据库连接
def conn_to_db():
    return mysql.connector.connect(user='root', password='',
                                   host='127.0.0.1',
                                   database='wechat_offacc')


# 用于可视化的类
def plot_images(images, cls_true, cls_pred=None, cls_pred2=None, smooth=True, root_dir=''):
    class_names = [0, 1]
    assert len(images) == len(cls_true)

    # Create figure with sub-plots.
    fig, axes = plt.subplots(10, 5)

    # Adjust vertical spacing.
    if cls_pred is None:
        hspace = 0.3
    else:
        hspace = 0.6
    # fig.subplots_adjust(hspace=hspace, wspace=0.3)

    # Interpolation type.
    if smooth:
        interpolation = 'spline16'
    else:
        interpolation = 'nearest'

    for i, ax in enumerate(axes.flat):
        # There may be less than 9 images, ensure it doesn't crash.
        if i < len(images):
            # Plot image.
            # print(root_dir + images[i])
            # temp_path = '/Users/mac/PycharmProjects/Investor-Sentiment/cover_imgs/MjM5NzQ5MTkyMA==/a65f47f09010b5db1d5b513982c3410e.png'
            # img = Image.open(temp_path)
            # plt.imshow(plt.imread(temp_path))
            # plt.show()
            ax.imshow(Image.open(root_dir + images[i]),
                      interpolation=interpolation)

            # Name of the true class.
            # cls_true_name = class_names[cls_true[i]]

            # Show true and predicted classes.
            if cls_pred is None:
                # xlabel = "True: {0}".format('')
                pass
            else:
                # Name of the predicted class.
                # print(cls_pred[i])
                neg = cls_pred[i]
                pos = cls_pred2[i]

                # pre_0
                label_str = "Pos:{0:.2f} \n Neg:{1:.2f}".format(pos, neg)

                xlabel = label_str

            # Show the classes as the label on the x-axis.
            font_my = {'family': 'Times New Roman',
                       'weight': 'normal', 'size': 8,
                       }
            # ax.set_xlabel(xlabel, font_my)

        # Remove ticks from the plot.
        ax.set_xticks([])
        ax.set_yticks([])

    # Ensure the plot is shown correctly with multiple plots
    # in a single Notebook cell.
    # plt.savefig('sinc.png', dpi=300)
    plt.show()


# 在预测结果中按照行X列展示指定行的结果
def show_result(columns, img_path='result_img.csv'):
    df_pre = pd.read_csv(img_path)

    df_pre = df_pre.iloc[columns]
    # df_pre = df_pre.drop(labels=0, axis=1)
    df_pre.index = range(len(df_pre))

    path, neg, pos = df_pre['img_path'], df_pre['neg'], df_pre['pos']
    images = path
    cls_true = path

    plot_images(images, cls_true, cls_pred=neg, cls_pred2=pos)


# 展示神经网络的结构
def show_cnn_structure():
    def show_cnn2():
        model1 = models.Sequential()
        model1.add(Conv2D(8, (3, 3), padding="same", input_shape=(299, 299, 3), activation="relu"))
        model1.add(Dense(16, input_shape=(784,)))
        model1.add(Dense(8))
        model1.add(Dense(4))
        visualizer(model1, format='png', view=True)

    def show_cnn():
        # Building model architecture
        model = models.Sequential()
        model.add(Conv2D(8, (3, 3), padding="same", input_shape=(299, 299, 3), activation="relu"))
        model.add(MaxPooling2D(pool_size=(3, 3)))

        model.add(Flatten())
        model.add(Dense(1024, activation="relu"))
        model.add(Dense(2))
        model.summary()

        visualizer(model, format='png', view=True)

    show_cnn()
    show_cnn2()


# 从数据库中提取情绪分析的结果
def show_imgsent_from_db():
    # 条件查询
    def select_top_sent(filter_date):
        # 创建游标
        cursor_neg, cursor_pos = cnx.cursor(buffered=True), cnx.cursor(buffered=True)

        # 分开查询
        query_neg = ("SELECT local_cover,cover_neg,cover_pos FROM articles "
                     "WHERE cover_neg IS NOT NULL AND "
                     "p_date BETWEEN %s AND %s "
                     "ORDER BY cover_neg DESC "
                     "LIMIT 128")

        query_pos = ("SELECT local_cover,cover_neg,cover_pos FROM articles "
                     "WHERE cover_pos IS NOT NULL AND "
                     "p_date BETWEEN %s AND %s "
                     "ORDER BY cover_pos DESC "
                     "LIMIT 128")
        # date类型转ts
        p_start_ts, p_end_ts = date_to_ts(filter_date[0]), date_to_ts(filter_date[1])

        # 执行查询语句
        cursor_neg.execute(query_neg, (p_start_ts, p_end_ts))
        cursor_pos.execute(query_pos, (p_start_ts, p_end_ts))

        # 按照id处理并转换为dfx
        print('查询到记录条数:', cursor_neg.rowcount)
        # 切片
        # df_sql = pd.DataFrame(cursor_query)
        # df_sql = df_query[:512, :]
        return pd.DataFrame(cursor_neg), pd.DataFrame(cursor_pos)

    # 建立连接
    cnx = conn_to_db()

    # 按照消极情绪排行查询
    # 筛选公众号+日期+条数
    df_neg, df_pos = select_top_sent([date(2021, 6, 1), date(2022, 6, 1)])

    # 断开连接
    cnx.close()

    # 调用可视化
    df_neg, df_pos = df_neg.iloc[:50, :], df_pos.iloc[:50, :]
    # print(df_neg[0])
    # time.sleep(1111)
    path, neg, pos = df_neg[0], df_neg[1], df_neg[2],
    images = path
    cls_true = path
    # print()

    plot_images(images, cls_true, cls_pred=neg, cls_pred2=pos,
                root_dir='/Users/mac/PycharmProjects/Investor-Sentiment/')

    path, neg, pos = df_pos[0], df_pos[1], df_pos[2],
    images = path
    cls_true = path
    plot_images(images, cls_true, cls_pred=neg, cls_pred2=pos,
                root_dir='/Users/mac/PycharmProjects/Investor-Sentiment/')
