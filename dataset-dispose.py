#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :dataset-dispose.py
# @Time      :2022/5/28 18:45
# @Author    :Colin
# @Note      :None

import shutil
import time
import numpy as np


def mkdir(path):
    # 引入模块

    import os

    # 去除首位空格

    path = path.strip()

    # 去除尾部 \ 符号

    path = path.rstrip("\\")

    # 判断路径是否存在

    # 存在     True

    # 不存在   False

    isExists = os.path.exists(path)

    # 判断结果

    if not isExists:

        # 如果不存在则创建目录

        # 创建目录操作函数

        os.makedirs(path)

        return True

    else:

        # 如果目录存在则不创建，并提示目录已存在

        return False


data_folder = 'twitter_dataset/images/'  # 原始图像目录
new_data_folder = 'twitter2/'  # 新的文件夹
# add label first
label_file = open('twitter_dataset/ground_truth/twitter_five_agrees.txt', mode='r')
labels = label_file.readlines()
label_file.close()

np.random.seed(0)  # 统一seed，保证每次随机结果都一样
np.random.shuffle(labels)


def split_set(set_name, set_data):
    for i, item in enumerate(set_data):
        # class_num=str(item).split(' ')[1]
        column = item.split('\n')
        image_name = column[0].split(' ')[0]
        image_class = column[0].split(' ')[1]
        # print(image_path,image_class)
        if image_class == '0':  # negative
            dir_name = new_data_folder + set_name + '/negative/'
            mkdir(dir_name)
            shutil.copyfile(data_folder + image_name, dir_name + image_name)
        elif image_class == '1':
            dir_name = new_data_folder + set_name + '/positive/'
            mkdir(dir_name)
            shutil.copyfile(data_folder + image_name, dir_name + image_name)


# 随机划分训练集\验证集\测试集
image_total = len(labels)
train_set = labels[:int(image_total * 0.7)]  # 0.7
val_set = labels[int(image_total * 0.7):int(image_total * 0.9)]  # 0.2
test_set = labels[int(image_total * 0.9):]  # 0.1

print(len(train_set), len(val_set), len(test_set))

split_set('test', test_set)

split_set('train', train_set)

split_set('validation', val_set)

# 划分选训练集,开发集和测试集
# 均衡划分样本
