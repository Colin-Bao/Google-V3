#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :load_fin_data.py
# @Time      :2022/6/16 20:10
# @Author    :Colin
# @Note      :None
from datetime import datetime
import mysql.connector
import pandas as pd
from financial_data import tushare_api

from financial_data import global_vars as gv


# 获取数据库连接
def conn_to_db():
    return mysql.connector.connect(user='root', password='',
                                   host='127.0.0.1',
                                   database='wechat_offacc')


def get_from_tu(ts_code):
    # 获取K线数据的日期
    tu = tushare_api.TuShareGet('20120101', '20220601')
    # 获取的指数
    df_kline = pd.DataFrame(tu.get_index(ts_code))
    # 转换为dt方便计算
    df_kline['date_ts'] = df_kline[['trade_date', ]].apply(
        lambda x: datetime.strptime(x['trade_date'], '%Y%m%d').date(),
        axis=1)
    df_kline['date_ts'] = df_kline[['date_ts', ]].apply(
        lambda x: int(pd.to_datetime(x['date_ts']).timestamp()),
        axis=1)
    df_kline['weekday'] = df_kline[['trade_date', ]].apply(
        lambda x: datetime.strptime(x['trade_date'], '%Y%m%d').weekday(),
        axis=1)
    # 排序以填充
    df = df_kline.sort_values(by='date_ts')
    # 筛选需要的行
    return df


def create_table(table_name):
    cnx = conn_to_db()
    create_sql = (
            "CREATE TABLE IF NOT EXISTS  " + table_name +
            " (`date_ts` int NOT NULL,`ts_code` varchar(40),`trade_date` varchar(40),"
            "`close` float,`open` float,`high` float,`low` float,`pre_close` float,`change` float,"
            "`pct_chg` float,`vol` float,`amount` float,"
            "PRIMARY KEY (`date_ts`),"
            "KEY `ix_date` (`date_ts`) USING BTREE)"
    )
    cur_create = cnx.cursor()
    cur_create.execute(create_sql)
    cnx.close()


# 存储数据到mysql
def insert_into_fintable(df, table_name):
    cnx = conn_to_db()

    # 转成元组方便mysql插入
    result_tuples = [tuple(xi) for xi in df.values]
    # 更新语句 按照id更新
    insert_infodate = (
            "INSERT IGNORE INTO  " + table_name +
            " (ts_code,trade_date,`close`,`open`,high,low,pre_close,`change`,pct_chg,vol,amount, date_ts,weekday) "
            "VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s) ")
    cur_sent = cnx.cursor(buffered=True)
    cur_sent.executemany(insert_infodate, result_tuples)
    cnx.commit()
    cnx.close()


def old_start_download():
    index_list = ['399300.SZ', '000001.SH']
    for code in index_list:
        df = get_from_tu(code)
        create_table('`' + code + '`')
        insert_into_fintable(df, '`' + code + '`')


# 下载数据并存入数据库
def start_download():
    #

    index_list = gv.INDEX_LIST
    attr_dict = gv.INDEX_TABLE_COLUMN

    from my_tools import mysql_dao

    for code in index_list:
        df = get_from_tu(code)
        mysql_dao.insert_table(code, df, attr_dict)
