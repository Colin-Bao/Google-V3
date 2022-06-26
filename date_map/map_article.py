#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :map_article.py
# @Time      :2022/6/16 17:51
# @Author    :Colin
# @Note      :None
from datetime import datetime
from tools import mysql_dao
from date_map import __config as gv
import pandas as pd


# 先从article总表中查找,和infodate匹配后插入交易日期
def select_article_table() -> pd.DataFrame:
    return mysql_dao.select_table(gv.ARTICLE_TABLE, gv.ARTICLE_TABLE_SELECT, gv.ARTICLE_TABLE_FILTER)


# 先从article总表中查找,和infodate匹配后插入交易日期
def select_info_table() -> pd.DataFrame:
    return mysql_dao.select_table(gv.INFO_TABLE, gv.INFO_TABLE_SELECT)


# 开始匹配
def join_date_df(df_article: pd.DataFrame, df_info: pd.DataFrame) -> pd.DataFrame:
    if df_article.empty or df_info.empty:
        return pd.DataFrame()

    # 从df_article的p_date中提取date
    df_article['date_ts'] = df_article[['p_date', ]].apply(
        lambda x: datetime.fromtimestamp(x['p_date']).date(),
        axis=1)
    # 转为ts方便匹配
    df_article['date_ts'] = df_article[['date_ts', ]].apply(
        lambda x: int(pd.to_datetime(x['date_ts']).timestamp()),
        axis=1)

    # join,左表为df_article,用date匹配
    df_con = pd.merge(df_article, df_info, how='left', on=['date_ts'])

    # 匹配以后进行计算
    df_con['article_to_tdate'] = df_con[['p_date', 'nature_datetime_ts', 'day_tradedate', 'night_tradedate']].apply(
        lambda x: x['day_tradedate'] if x['p_date'] <= x['nature_datetime_ts'] else x['night_tradedate'], axis=1)

    # 空的日期是因为info_date的交易日期最后一行滞后了
    # 如果不删除空值聚合会出错
    df_con.dropna(inplace=True)

    # 增加数据列用于制图
    # # 交易日期
    # 全部转成了str
    df_con['date_t'] = df_con[['article_to_tdate', ]].apply(
        lambda x: str(pd.to_datetime(x['article_to_tdate']).date()), axis=1)
    df_con['date_p'] = df_con[['p_date', ]].apply(lambda x: str(datetime.fromtimestamp(x['p_date']).date()), axis=1)
    df_con['datetime_p'] = df_con[['p_date', ]].apply(lambda x: str(datetime.fromtimestamp(x['p_date'])), axis=1)

    # 转换为ts方便入库
    df_con['t_date'] = df_con[['article_to_tdate', ]].apply(
        lambda x: pd.to_datetime(x['article_to_tdate']).date(), axis=1)

    df_con['t_date'] = df_con[['t_date', ]].apply(
        lambda x: int(pd.to_datetime(x['t_date']).timestamp()), axis=1)

    # 如果需要检查的时候查看返回值

    # 存储结果 筛选一下,换了位置方便数据库插入
    return df_con[gv.ARTICLE_TABLE_UPDATE]


# 把匹配后的文章交易日期存储到数据库
def update_article_table(df: pd.DataFrame):
    mysql_dao.update_table(gv.ARTICLE_TABLE, df)


# 先创建
def start_map_tdate():
    # JIOIN连接
    df_con = join_date_df(select_article_table(), select_info_table())
    # 更新
    # df_con[gv.ARTICLE_TABLE_UPDATE].to_csv('test.csv')
    update_article_table(df_con)

# start_map_tdate()
