#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :group_imgs.py
# @Time      :2022/6/16 19:50
# @Author    :Colin
# @Note      :None

from datetime import datetime

import mysql.connector
import pandas as pd
from pandas import DataFrame

from date_group import __config as gv

# date类型转ts
from tools import mysql_dao


def date_to_ts(date_type):
    dt = datetime.combine(date_type, datetime.min.time())
    # datetime.fromtimestamp(p_date)
    return int(dt.timestamp())


# 获取数据库连接
def conn_to_db():
    return mysql.connector.connect(user='root', password='',
                                   host='127.0.0.1',
                                   database='wechat_offacc')


# 获得公众号列表
def select_gzh_table():
    from tools import mysql_dao
    return mysql_dao.select_table(gv.GZH_TABLE, gv.GZH_SELECT)


# 获得公众号列表
def select_gzh_list():
    # 建立数据库连接
    cnx = conn_to_db()
    cursor_query = cnx.cursor(buffered=True)
    query = "SELECT biz,nickname FROM gzhs "
    cursor_query.execute(query)

    # 转换为df重命名并返回
    dict_columns = {i: cursor_query.column_names[i] for i in range(len(cursor_query.column_names))}
    df_cur = pd.DataFrame(cursor_query)
    df_cur.rename(columns=dict_columns, inplace=True)

    # 关闭游标和连接
    cnx.close()
    return df_cur


# 一些用于计算的函数
# 用于join article和article_imgs表,并把每个公众号按照日期聚合得到gzh_imgs_bydate表
# 如果不按照公众号聚合也可以,得到总表
# 先把数据取出来再计算(聚合公众号)
# 条件查询 合并其他表,更改的时候填写需要改的字段即可
# 这里是从articles表和article_imgs表连接查找数据
def select_article_img(filter_biz):
    cnx = conn_to_db()
    # 创建游标
    cursor_query = cnx.cursor(buffered=True)

    # 查询id和用于计算的值
    query_sql = (
        "SELECT articles.id,articles.biz,articles.p_date,articles.t_date,article_imgs.mov,article_imgs.cover_neg,article_imgs.cover_pos "
        "FROM articles INNER JOIN article_imgs on articles.id = article_imgs.id "
        "WHERE articles.biz = %s AND "
        "articles.t_date IS NOT NULL "
        "ORDER BY articles.p_date ASC "
    )

    # 执行查询语句
    cursor_query.execute(query_sql, (filter_biz,))

    # 按照id处理并转换为df
    print('查询到要聚合的条数:', cursor_query.rowcount)

    # 转换为df重命名并返回
    dict_columns = {i: cursor_query.column_names[i] for i in range(len(cursor_query.column_names))}
    df_cur = pd.DataFrame(cursor_query)
    df_cur.rename(columns=dict_columns, inplace=True)
    # ['id', 'p_date', 't_date', 'mov', 'cover_neg', 'cover_pos']

    cnx.close()

    return cursor_query.rowcount, df_cur


def select_img_table(biz) -> pd.DataFrame:
    from tools import mysql_dao

    # 查询id和用于计算的值
    query_sql = (
        "SELECT articles.id,articles.biz,articles.p_date,articles.t_date,article_imgs.mov,article_imgs.cover_neg,article_imgs.cover_pos "
        "FROM articles INNER JOIN article_imgs on articles.id = article_imgs.id "
        "WHERE articles.biz = %s AND "
        "articles.t_date IS NOT NULL "
        "ORDER BY articles.p_date ASC "
    )
    return mysql_dao.excute_sql(query_sql, tups=(biz,))


# 分析的是article和text表合并的数据
# 主要是聚合的运算 概率阈值的选择等
def group_articleimg_bydate(df, biz, nickname):
    if df.empty:
        return pd.DataFrame(), pd.DataFrame()

    # 一些指标的计算

    def cal_colunms():
        # 增加一列用于把ts换成天数(后续可以精确的时间) 前面的中扩繁相当于传入的参数x
        # 这个函数的作用是聚合,因此不需要time
        df['p_date'] = df[['p_date', ]].apply(lambda x: datetime.fromtimestamp(x['p_date']).date(), axis=1)

        df['t_date'] = df[['t_date', ]].apply(lambda x: datetime.fromtimestamp(x['t_date']).date(), axis=1)
        # df['t_date'] = df[['t_date', ]].apply(
        #     lambda x: datetime.fromtimestamp(x['t_date']).date() if not np.isnan(x['t_date']) else x['t_date'],
        #     axis=1)
        # print(df['t_date'])

        # 聚合操作
        # df_agg_date = df.groupby('date').agg({0: 'count', })
        # transform操作
        # df['id_count'] = df.groupby('date')[0].transform('count')
        df['c_neg_count'] = df[['cover_neg', ]].apply(lambda x: 1 if x['cover_neg'] >= 0.7 else 0, axis=1)
        df['c_pos_count'] = df[['cover_pos', ]].apply(lambda x: 1 if x['cover_pos'] >= 0.7 else 0, axis=1)

    # # 自定义apply函数用于计算概率
    # def count_prob(df_group):
    #     df_group['c_neg_count'] = df[['cover_neg', ]].apply(lambda x: 1 if x['cover_neg'] > 0.7 else 0, axis=1)
    #     df_group['c_pos_count'] = df[['cover_pos', ]].apply(lambda x: 1 if x['cover_pos'] > 0.7 else 0, axis=1)
    #     # print(df_group)
    #     return df_group
    #
    # # 在组中计算概率阈值计数
    # df_g = df.groupby(['p_date'], as_index=False).apply(lambda x: count_prob(x))
    # df_g = df.groupby(['t_date'], as_index=False).apply(lambda x: count_prob(x))

    # 计算完以后按照日期分组并聚合计算 只保留那些需要的列
    # !!!!!实际日期非交易日期

    # 聚合的方式
    def group_articles(column_name='t_date'):
        # 按照t_date分组
        df_group = df.groupby([column_name], as_index=False).agg(
            {'id': 'count', 'cover_neg': 'mean', 'cover_pos': 'mean', 'c_neg_count': 'sum',
             'c_pos_count': 'sum'})
        # 分组后重命名
        df_group.rename(
            columns={'id': 'article_count', 'cover_neg': 'c_negprob_mean', 'cover_pos': 'c_posprob_mean'},
            inplace=True)

        # 分组后继续计算

        df_group['c_neg_ratio'] = df_group[['article_count', 'c_neg_count']].apply(
            lambda x: x['c_neg_count'] / x['article_count'],
            axis=1)
        df_group['c_pos_ratio'] = df_group[['article_count', 'c_pos_count']].apply(
            lambda x: x['c_pos_count'] / x['article_count'],
            axis=1)

        # 添加额外的列pk
        # print(df_group.columns)
        df_group['id_group_date'] = df_group[[column_name, ]].apply(
            lambda x: str(biz) + str(x[column_name]), axis=1)
        df_group['nick_name'] = nickname
        df_group['biz'] = biz
        df_group['date_ts'] = df_group[[column_name, ]].apply(
            lambda x: int(pd.to_datetime(x[column_name]).timestamp()), axis=1)

        # 重新排列
        df_group = df_group[
            ['id_group_date', 'nick_name', 'biz', 'date_ts', column_name, 'article_count',
             'c_negprob_mean', 'c_posprob_mean',
             'c_neg_count', 'c_pos_count', 'c_neg_ratio', 'c_pos_ratio']]

        return df_group

    # 计算一些指标
    cal_colunms()

    # 根据p/t_date聚合
    df_g_t = group_articles(column_name='t_date')
    df_g_p = group_articles(column_name='p_date')

    return df_g_t, df_g_p


# 更新结果到数据库gzhs_imgs_bydate
# 主键按照公众号+日期
def insert_groupbydate(df_group, table_name):
    cnx = conn_to_db()
    # 'date', 'article_count', 'c_negprob_mean', 'c_posprob_mean',
    # 'c_neg_count', 'c_pos_count', 'c_neg_ratio', 'c_pos_ratio',

    # id_group_date, nick_name, date_ts, date, article_count,
    # c_negprob_mean, c_posprob_mean,
    # c_neg_count, c_pos_count, c_neg_ratio, c_pos_ratio
    # 转成元组方便mysql插入

    merge_result_tuples = [tuple(xi) for xi in df_group.values]

    # 更新语句 按照id更新
    insert_gzhs = (
            "INSERT IGNORE INTO  " + str(table_name) +
            " (id_group_date, nick_name,biz, date_ts, date, article_count,"
            "c_negprob_mean, c_posprob_mean,c_neg_count, c_pos_count, c_neg_ratio, c_pos_ratio) "
            "VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s) ")

    cur_sent = cnx.cursor(buffered=True)
    cur_sent.executemany(insert_gzhs, merge_result_tuples)
    cnx.commit()
    cnx.close()


# 合并表格并保存csv
def merge_group_and_fin(df_g: pd.DataFrame):
    from data_down import cal_data
    df_fin = cal_data.select_fin_table()
    df_con = pd.merge(df_g, df_fin, how='left', on=['date_ts'])
    # print(df_g['date_ts'], df_fin['date_ts'])
    return df_con


def start_group_by_date():
    #
    df_gzh = select_gzh_list()
    # 按照公众号循环更新
    for index, gzh in enumerate(df_gzh.values):
        biz, nickname = gzh[0], gzh[1]
        query_count, df_cur = select_article_img(biz)
        if query_count == 0:
            continue
        df_g_t, df_g_p = group_articleimg_bydate(df_cur, biz, nickname)

        merge_group_and_fin(df_g_t).to_csv(gv.TGROUP_CSV_PATH + nickname + '_tdate.csv')
        merge_group_and_fin(df_g_p).to_csv(gv.PGROUP_CSV_PATH + nickname + '_pdate.csv')

        insert_groupbydate(df_g_t, 'gzhs_imgs_bytdate')
        insert_groupbydate(df_g_p, 'gzhs_imgs_bydate')
        # print(df_g_t, df_g_p)
        # time.sleep(1111)


# 把金融表合并到group表
def merge_fin_df(df_g: pd.DataFrame) -> pd.DataFrame:
    from tools import mysql_dao
    return pd.merge(df_g, mysql_dao.select_table(gv.FIN_TABLE, gv.FIN_SELECT), how='left', on=['date_ts'])


def start_group():
    from tools import mysql_dao

    def insert_group_table(tup, x):
        if not tup[0].empty and not tup[1].empty:
            mysql_dao.insert_table(gv.TGROUP_TABLE, tup[0])
            mysql_dao.insert_table(gv.PGROUP_TABLE, tup[1])

            merge_fin_df(tup[0]).to_csv(gv.TGROUP_CSV_PATH + x['nickname'] + '_tdate.csv')
            merge_fin_df(tup[1]).to_csv(gv.PGROUP_CSV_PATH + x['nickname'] + '_pdate.csv')

    # 公众号apply进度条
    from tqdm.auto import tqdm
    tqdm.pandas(desc='Start Group Gzhs')

    # 进度条
    select_gzh_table()[['biz', 'nickname']].progress_apply(
        lambda x:
        insert_group_table(
            group_articleimg_bydate(
                select_img_table(x['biz']), x['biz'], x['nickname']
            ), x
        )
        ,
        axis=1)
