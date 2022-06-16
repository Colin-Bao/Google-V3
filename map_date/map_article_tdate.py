#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :map_article_tdate.py
# @Time      :2022/6/16 17:51
# @Author    :Colin
# @Note      :None
import time
from datetime import datetime, date

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


# 创建表的算法,不小心删了
def create_info_table():
    pass


# 先从article总表中查找,和infodate匹配后插入交易日期
def select_article():
    cnx = conn_to_db()
    # 创建游标
    cursor_query = cnx.cursor(buffered=True)
    # 查询id和用于计算的值
    query_str = (
        "SELECT articles.id,articles.p_date "
        "FROM articles "
        "WHERE articles.t_date IS NULL AND "
        "articles.p_date IS NOT NULL "
        "ORDER BY articles.p_date ASC "
    )

    # 执行查询语句
    cursor_query.execute(query_str)

    # 按照id处理并转换为df
    print('查询到null t_date记录条数:', cursor_query.rowcount)

    # 转换为df重命名并返回
    dict_columns = {i: cursor_query.column_names[i] for i in range(len(cursor_query.column_names))}
    df_cur = pd.DataFrame(cursor_query)
    df_cur.rename(columns=dict_columns, inplace=True)

    cnx.close()

    return cursor_query.rowcount, df_cur


# 先从article总表中查找,和infodate匹配后插入交易日期
def select_infodate():
    cnx = conn_to_db()
    # 创建游标
    cursor_query = cnx.cursor(buffered=True)

    # 查询id和用于计算的值
    query_str = (
        "SELECT date_ts,nature_datetime_ts,day_tradedate,night_tradedate "
        "FROM info_date "
    )

    # 执行查询语句
    cursor_query.execute(query_str)

    # 转换为df重命名并返回
    dict_columns = {i: cursor_query.column_names[i] for i in range(len(cursor_query.column_names))}
    df_cur = pd.DataFrame(cursor_query)
    df_cur.rename(columns=dict_columns, inplace=True)

    # cursor_sent.close()
    cnx.close()

    return cursor_query.rowcount, df_cur


# 开始匹配
def join_map_date(df_article, df_info):
    # 重命名
    # df_article.rename(columns={0: 'id', 1: 'datetime_ts'}, inplace=True)
    # df_info.rename(columns={0: 'date', 1: 'datetime_ts'}, inplace=True)

    # 从df_article的p_date中提取date
    df_article['date_ts'] = df_article[['p_date', ]].apply(
        lambda x: datetime.fromtimestamp(x['p_date']).date(),
        axis=1)
    # 转为ts方便匹配
    df_article['date_ts'] = df_article[['date_ts', ]].apply(
        lambda x: int(pd.to_datetime(x['date_ts']).timestamp()),
        axis=1)

    # df_article['p_date_time'] = df_article[['p_date', ]].apply(
    #     lambda x: datetime.fromtimestamp(x['p_date']),
    #     axis=1)
    # df_article['p_date_date'] = df_article[['p_date', ]].apply(
    #     lambda x: datetime.fromtimestamp(x['p_date']).date(),
    #     axis=1)
    # join,左表为df_article,用date匹配
    df_con = pd.merge(df_article, df_info, how='left', on=['date_ts'])
    # df_con.to_csv('test.csv')
    # 匹配以后进行计算
    df_con['article_to_tdate'] = df_con[['p_date', 'nature_datetime_ts', 'day_tradedate', 'night_tradedate']].apply(
        lambda x: x['day_tradedate'] if x['p_date'] <= x['nature_datetime_ts'] else x['night_tradedate'], axis=1)

    # 空的日期是因为info_date的交易日期最后一行滞后了
    df_con.dropna(inplace=True)

    # 转换为ts方便入库

    df_con['t_date'] = df_con[['article_to_tdate', ]].apply(
        lambda x: pd.to_datetime(x['article_to_tdate']).timestamp(), axis=1)

    # 如果需要检查的时候查看返回值
    # df_con.to_csv('test.csv')

    # 存储结果 筛选一下,换了位置方便数据库插入
    df_con = df_con[['t_date', 'id']]
    return df_con


# 把匹配后的文章交易日期存储到数据库
def update_to_article(df):
    cnx = conn_to_db()
    # 转成元组方便mysql插入
    merge_result_tuples = [tuple(xi) for xi in df.values]
    # print(merge_result_tuples)

    # 更新语句 按照id更新
    update_old = (
        "UPDATE articles SET t_date = %s "
        "WHERE id = %s ")

    cur_sent = cnx.cursor(buffered=True)
    cur_sent.executemany(update_old, merge_result_tuples)
    cnx.commit()
    cnx.close()


def start_map_tdate():
    # 先创建
    # create_info_date()
    # 分别在2张表中查询并返回查询结果
    count_article, df_article = select_article()
    count_infodate, df_infodate = select_infodate()
    if count_article == 0 or count_infodate == 0:
        return
    else:
        # 匹配好以后返回到articl表
        update_to_article(join_map_date(df_article, df_infodate))
