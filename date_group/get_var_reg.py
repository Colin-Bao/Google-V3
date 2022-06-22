#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :get_var_reg.py
# @Time      :2022/6/16 23:17
# @Author    :Colin
# @Note      :None
import time
from datetime import datetime, date
import pandas as pd
import mysql.connector


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


def get_var_reg(filter_biz, filter_date):
    # 按照公众号分组
    def select_by_biz(groupbybiz=True):
        cur = cnx.cursor()
        p_start_ts, p_end_ts = date_to_ts(filter_date[0]), date_to_ts(filter_date[1])

        # 如果分组
        if groupbybiz:
            query = (
                "SELECT g_tdate.date,g_tdate.c_neg_ratio,g_tdate.c_pos_ratio,g_tdate.c_negprob_mean,g_tdate.c_posprob_mean,g_tdate.article_count,"
                "`399300.SZ`.vol, `399300.SZ`.log_return, `399300.SZ`.log_return_2 "
                "FROM gzhs_imgs_bytdate AS g_tdate INNER JOIN `399300.SZ` "
                "ON g_tdate.date_ts = `399300.SZ`.date_ts "
                "WHERE g_tdate.biz = %s AND "
                "g_tdate.date_ts IS NOT NULL AND "
                "g_tdate.date_ts BETWEEN %s AND %s "
                "ORDER BY g_tdate.date_ts ASC ")
            cur.execute(query, (filter_biz, p_start_ts, p_end_ts))

        else:
            query = (
                "SELECT g_tdate.biz,g_tdate.date,g_tdate.c_neg_ratio,g_tdate.c_pos_ratio,g_tdate.c_negprob_mean,g_tdate.c_posprob_mean,g_tdate.article_count,"
                "`399300.SZ`.vol, `399300.SZ`.log_return, `399300.SZ`.log_return_2 "
                "FROM gzhs_imgs_bytdate AS g_tdate INNER JOIN `399300.SZ` "
                "ON g_tdate.date_ts = `399300.SZ`.date_ts "
                "WHERE "
                "g_tdate.date_ts IS NOT NULL AND "
                "g_tdate.date_ts BETWEEN %s AND %s "
                "ORDER BY g_tdate.date_ts ASC ")
            cur.execute(query, (p_start_ts, p_end_ts))

        return cur

    # 进行一些回归前的计算 用article表和groupbydate表
    def get_var(cur):
        # 重命名
        dict_name = {i: cur.column_names[i] for i in range(len(cur.column_names))}
        df = pd.DataFrame(cur)
        df.rename(columns=dict_name, inplace=True)

        # 日期相关处理
        def get_date():
            # 日期
            df['dt_date'] = df[['date', ]].apply(lambda x: datetime.strptime(x['date'], '%Y-%m-%d').date(), axis=1)
            df['weekday'] = df[['date', ]].apply(lambda x: datetime.strptime(x['date'], '%Y-%m-%d').weekday(), axis=1)

        # 计算滞后变量
        def get_lag(var_columns, lag_i):
            # 迭代计算
            for var in var_columns:
                for i in range(lag_i):
                    # 格式为Lx
                    df[var + '_L' + str(i + 1)] = df[var].shift(i + 1)

        # 生成虚拟变量
        def get_dummy(var_columns):
            df_con = df
            for var in var_columns:

                # 删除一个多重共线性
                var_dummy = pd.get_dummies(df[var], drop_first=True, prefix=var)
                # 要是太多就跳过
                if len(var_dummy.columns) > 10:
                    continue
                # 合并
                df_con = pd.concat([df_con, var_dummy], axis=1)
            return df_con

        # 删除一些行
        def drop_rows(df_row: pd.DataFrame):
            # 删除空行
            df_row.dropna(inplace=True, axis=0)
            # 筛选不要的行
            # df_row = df_row[['']]
            df_row = df_row[df_row['nickname'] == '每日经济新闻']
            return df_row

        # 把公众号名称转换
        def get_bizname(df_biz):
            sql_gzh = "SELECT biz,nickname FROM gzhs"
            cur_gzh = cnx.cursor()
            cur_gzh.execute(sql_gzh)
            dict_gzhname = {i: cur_gzh.column_names[i] for i in range(len(cur_gzh.column_names))}
            df_gzhname = pd.DataFrame(cur_gzh)
            df_gzhname.rename(columns=dict_gzhname, inplace=True)
            df_nick = pd.merge(df_biz, df_gzhname, how='left', on=['biz'])
            return df_nick

        get_date()
        # 滞后变量计算有问题,不能合并滞后
        # get_lag(['log_return', 'log_return_2', 'c_neg_ratio', 'c_pos_ratio'], 5)
        df_du = get_dummy(['weekday'])
        df_gzh = get_bizname(df_du)
        df_drop = drop_rows(df_gzh)

        return df_drop

    cnx = conn_to_db()

    # 按照公众号合并金融市场数据得到结果
    # 进行回归前的计算

    df = get_var(select_by_biz(False))
    df.to_csv('test.csv')

    cnx.close()
