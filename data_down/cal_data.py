#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :cal_data.py
# @Time      :2022/6/16 22:10
# @Author    :Colin
# @Note      :None

import pandas as pd
import numpy as np


# 包括了一系列的数据计算
# 以及对表格数据的更新

# 计算回报
def cal_return(df: pd.DataFrame) -> pd.DataFrame:
    # 重命名方便计算
    df['close_lag1'] = df['close'].shift(1)
    # ln今日收盘价/昨日收盘价
    df['log_return'] = df[['close', 'close_lag1', ]].apply(
        lambda x: np.log(x['close'] / x['close_lag1']),
        axis=1)

    # 计算滞后回报
    df['log_return_l1'] = df['log_return'].shift(-1)

    # 删除空行
    df.dropna(inplace=True)

    # 全部转为str方便插入
    df = df.astype(str)

    return df[['log_return', 'log_return_l1', 'date_ts']]


# 计算日期并更新
def cal_date(df: pd.DataFrame) -> pd.DataFrame:
    from tools import mysql_dao
    df = cal_return(mysql_dao.select_table('articles', ['p_date', '', 'id']))
    df['date'] = df[['date_ts']].apply(lambda x: x['date_ts'], axis=1)
    return df[['date', 'date_ts']]


# 本py的控制类
def start_cal():
    # 在下载数据的时候已经做了计算
    from tools import mysql_dao
    from data_down import __config as gv
    for i in gv.INDEX_LIST:
        mysql_dao.update_table(i, cal_return(mysql_dao.select_table(i, ['*'])),
                               {'log_return': 'FLOAT', 'log_return_l1': 'FLOAT'})

    # mysql_dao.update_table('399300.SZ', cal_return(mysql_dao.select_table('399300.SZ', ['*'])),
    #                        {'log_return': 'FLOAT', 'log_return_2': 'FLOAT', 'log_return_l1': 'FLOAT'})
    # 等价于
    # "UPDATE `399300.SZ` SET log_return = %s,log_return_2 = %s "
    # "WHERE date_ts = %s ")

# start_cal()
