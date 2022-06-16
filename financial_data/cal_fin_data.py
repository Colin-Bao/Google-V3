#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :cal_fin_data.py
# @Time      :2022/6/16 22:10
# @Author    :Colin
# @Note      :None

import mysql.connector
import pandas as pd
import numpy as np


# 获取数据库连接
def conn_to_db():
    return mysql.connector.connect(user='root', password='',
                                   host='127.0.0.1',
                                   database='wechat_offacc')


# 查表 可以被其他函数调用
def select_fin_table():
    cnx = conn_to_db()
    table_list = ['`399300.SZ`', '`000001.SH`']
    # 对数收益率
    sql_str = ("SELECT date_ts,pre_close,vol,log_return,log_return_2 FROM `399300.SZ`"
               "ORDER BY date_ts ASC "
               )

    cursor_query = cnx.cursor()
    cursor_query.execute(sql_str)

    # 转换为df重命名并返回
    dict_columns = {i: cursor_query.column_names[i] for i in range(len(cursor_query.column_names))}
    df_cur = pd.DataFrame(cursor_query)
    df_cur.rename(columns=dict_columns, inplace=True)

    cnx.close()

    return df_cur


def cal_return(df):
    # 重命名方便计算
    df['pre_close_lag1'] = df['pre_close'].shift(1)
    # ln今日收盘价/昨日收盘价
    df['log_return'] = df[['pre_close', 'pre_close_lag1', ]].apply(
        lambda x: np.log(x['pre_close'] / x['pre_close_lag1']),
        axis=1)
    df['log_return_2'] = df[['log_return', ]].apply(
        lambda x: np.square(x['log_return']),
        axis=1)
    # 删除空行
    df.dropna(inplace=True)

    # 保留插入数据库的列
    df = df[['log_return', 'log_return_2', 'date_ts']]

    # 全部转为str方便插入
    df = df.astype(str)

    return [tuple(xi) for xi in df.values]


# 更新计算好的数据
def update_table(data_tup):
    cnx = conn_to_db()
    sql_str = (
        "UPDATE `399300.SZ` SET log_return = %s,log_return_2 = %s "
        "WHERE date_ts = %s ")
    cur = cnx.cursor()
    cur.executemany(sql_str, data_tup)
    cnx.commit()
    cnx.close()


def start_cal():
    res_tuple = cal_return(select_fin_table())
    # 更新计算好的数据
    update_table(res_tuple)
