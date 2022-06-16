#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :cal_other.py
# @Time      :2022/6/16 23:20
# @Author    :Colin
# @Note      :None

import time
from datetime import datetime, date
import os
import mysql.connector
import pandas as pd
import numpy as np


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


def cal_other_columns():
    # 金融市场中的表
    def add_return():
        # 取出收盘价
        def select_table():
            table_list = ['`399300.SZ`', '`000001.SH`']
            # 对数收益率
            sql_str = ("SELECT date_ts,pre_close FROM `399300.SZ`"
                       "ORDER BY date_ts ASC "
                       )

            cursor = cnx.cursor()
            cursor.execute(sql_str)

            return pd.DataFrame(cursor)

        # 计算回报
        # 返回可以插入的元组
        def cal_return(df):
            # 重命名方便计算
            df.rename(columns={0: 'date_ts', 1: 'pre_close'}, inplace=True)
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
            sql_str = (
                "UPDATE `399300.SZ` SET log_return = %s,log_return_2 = %s "
                "WHERE date_ts = %s ")
            cur = cnx.cursor()
            cur.executemany(sql_str, data_tup)
            cnx.commit()

        # 先获得要计算的数据
        res_tuple = cal_return(select_table())
        # 更新计算好的数据
        update_table(res_tuple)

    # 增加时间戳
    def add_date_ts(table_name):
        # 取出表
        def select_table():
            # 对数收益率
            sql_str = ("SELECT id_group_date,date FROM  " + table_name +
                       " ORDER BY date ASC "
                       )

            cursor = cnx.cursor()
            cursor.execute(sql_str)

            return pd.DataFrame(cursor)

        def cal_ts(df):
            df.rename(columns={0: 'id_group_date', 1: 'date'}, inplace=True)
            # 删除空行
            df.dropna(inplace=True)
            # 计算ts
            df['date_ts'] = df[['date', ]].apply(
                lambda x: datetime.strptime(x['date'], '%Y-%m-%d').timestamp(),
                axis=1)

            # 保留插入数据库的列
            df = df[['date_ts', 'id_group_date']]

            # 全部转为str方便插入
            df = df.astype(str)

            return [tuple(xi) for xi in df.values]

        # 更新计算好的数据
        def update_table(data_tup):
            sql_str = (
                    "UPDATE " + table_name +
                    " SET date_ts = %s "
                    "WHERE id_group_date = %s ")
            cur = cnx.cursor()
            cur.executemany(sql_str, data_tup)
            cnx.commit()

        update_table(cal_ts(select_table()))

    cnx = conn_to_db()

    # 已经计算结束
    # add_return()
    add_date_ts('`gzhs_imgs_bydate`')

    cnx.close()
