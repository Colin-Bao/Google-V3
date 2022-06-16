#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :create_info_table.py
# @Time      :2022/6/16 23:24
# @Author    :Colin
# @Note      :None
import pandas as pd
from datetime import datetime


# 创建表的算法,不小心删了
def cal_info_df():
    from financial_data import cal_fin_data
    df_fin = cal_fin_data.select_fin_table()
    df_fin = df_fin[['date_ts', 'trade_date']]
    df_fin.rename(columns={'trade_date': 'day_tradedate'}, inplace=True)

    # 生成自然日期
    df_date = pd.DataFrame(pd.date_range(start='1/1/2012', end='6/1/2022'))
    df_date.rename(columns={0: 'nature_date'}, inplace=True)

    # 转为ts 用于匹配p_date的去掉time后的date
    df_date['date_ts'] = df_date[['nature_date', ]].apply(
        lambda x: pd.to_datetime(x['nature_date']).date(),
        axis=1)
    df_date['date_ts'] = df_date[['date_ts', ]].apply(
        lambda x: int(pd.to_datetime(x['date_ts']).timestamp()),
        axis=1)

    # 生成自然时间用于比较
    df_date['nature_datetime'] = df_date[['nature_date', ]].apply(
        lambda x: pd.to_datetime(x['nature_date']).replace(hour=15, minute=0, second=0),
        axis=1)
    df_date['nature_datetime_ts'] = df_date[['nature_datetime', ]].apply(
        lambda x: int(pd.to_datetime(x['nature_datetime']).timestamp()),
        axis=1)

    # 匹配
    df_con = pd.merge(df_date, df_fin, how='left', on=['date_ts'])

    # 填充
    df_con['day_tradedate'].fillna(method='bfill', inplace=True)
    df_con['night_tradedate'] = df_con['day_tradedate'].shift(-1)

    # 转换
    df_con['day_tradedate'] = df_con[['day_tradedate', ]].apply(
        lambda x: str(pd.to_datetime(x['day_tradedate']).date()),
        axis=1)
    df_con['night_tradedate'] = df_con[['day_tradedate', ]].apply(
        lambda x: str(pd.to_datetime(x['day_tradedate']).date()),
        axis=1)

    # 转为字符串
    df_con['nature_date'] = df_con[['nature_date', ]].apply(
        lambda x: str(x['nature_date']),
        axis=1)
    df_con['nature_datetime'] = df_con[['nature_datetime', ]].apply(
        lambda x: str(x['nature_datetime']),
        axis=1)

    # 重排列
    df_con = df_con[['nature_date', 'date_ts', 'nature_datetime', 'nature_datetime_ts',
                     'day_tradedate', 'night_tradedate']]

    df_con.to_csv('test.csv')

    return df_con


def insert_info_table(df):
    from my_tools import tools
    cnx = tools.conn_to_db()
    cur = cnx.cursor(buffered=True)
    sql = ("INSERT IGNORE INTO info_date"
           " (nature_date, date_ts, nature_datetime, nature_datetime_ts,day_tradedate, night_tradedate) "
           "VALUES (%s,%s,%s,%s,%s,%s) ")

    cur.executemany(sql, tools.df_to_tup(df))
    cnx.commit()
    cnx.close()


def start_create_info():
    insert_info_table(cal_info_df())
