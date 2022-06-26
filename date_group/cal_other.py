#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :cal_other.py
# @Time      :2022/6/16 23:20
# @Author    :Colin
# @Note      :None

# 在聚合表进行计算
from tools import mysql_dao

df = mysql_dao.select_table('gzhs_imgs_bytdate', ['c_neg_ratio', 'id_group_date'])

for i in range(5):
    df['c_neg_ratio' + '_L' + str(i + 1)] = df['c_neg_ratio'].shift(i + 1)

df = df[['c_neg_ratio_L1', 'c_neg_ratio_L2',
         'c_neg_ratio_L3', 'c_neg_ratio_L4', 'c_neg_ratio_L5', 'id_group_date']]
mysql_dao.update_table('gzhs_imgs_bytdate', df,
                       {'c_neg_ratio_L1': 'FLOAT', 'c_neg_ratio_L2': 'FLOAT', 'c_neg_ratio_L3': 'FLOAT',
                        'c_neg_ratio_L4': 'FLOAT', 'c_neg_ratio_L5': 'FLOAT', })
