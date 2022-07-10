#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :group_imgs.py
# @Time      :2022/6/16 19:50
# @Author    :Colin
# @Note      :None

from datetime import datetime
import pandas as pd
from date_group import __config as gv
from tools import mysql_dao


# 主要是聚合的运算 概率阈值的选择等
def group_text_bydate(df, biz, nickname):
    if df.empty:
        return pd.DataFrame()

    # 一些指标的计算
    def def_tags():
        df['t_pos_tag'] = df[['title_pos', ]].apply(lambda x: 1 if x['title_pos'] >= 0.5 else 0, axis=1)
        df['t_neu_tag'] = df[['title_neu', ]].apply(lambda x: 1 if x['title_neu'] >= 0.5 else 0, axis=1)
        df['t_neg_tag'] = df[['title_neg', ]].apply(lambda x: 1 if x['title_neg'] >= 0.5 else 0, axis=1)

    # 聚合的方式
    def group_articles(column_name='t_date'):
        # 按照t_date分组
        df_group = df.groupby([column_name], as_index=False).agg(
            {'id': 'count',
             'title_pos': 'mean', 'title_neu': 'mean', 'title_neg': 'mean',
             't_pos_tag': 'sum', 't_neu_tag': 'sum', 't_neg_tag': 'sum'})

        # 分组后继续计算
        df_group['t_pos_ratio'] = df_group[['t_pos_tag', 'id']].apply(
            lambda x: x['t_pos_tag'] / x['id'],
            axis=1)
        df_group['t_neu_ratio'] = df_group[['t_neu_tag', 'id']].apply(
            lambda x: x['t_neu_tag'] / x['id'],
            axis=1)
        df_group['t_neg_ratio'] = df_group[['t_neg_tag', 'id']].apply(
            lambda x: x['t_neg_tag'] / x['id'],
            axis=1)

        # 添加额外的列pk
        df_group['id_group_date'] = df_group[[column_name, ]].apply(
            lambda x: str(biz) + str(x[column_name]), axis=1)
        df_group['nick_name'] = nickname
        df_group['biz'] = biz

        df_group['date_ts'] = df_group[column_name]
        df_group[column_name] = df_group[[column_name, ]].apply(
            lambda x: str(datetime.fromtimestamp(x[column_name]).date()), axis=1)

        # 改名
        df_group.rename(
            columns={'id': 'article_count',
                     't_pos_tag': 't_pos_count', 't_neu_tag': 't_neu_count', 't_neg_tag': 't_neg_count',
                     'title_pos': 't_posprob_mean', 'title_neu': 't_neuprob_mean', 'title_neg': 't_negprob_mean', },
            inplace=True)

        return df_group

    # 计算标签
    def_tags()

    # 根据t_date聚合
    return group_articles(column_name='t_date')


# 更新结果到数据库gzhs_imgs_bydate

def start_group():
    # 公众号apply进度条
    from tqdm.auto import tqdm
    tqdm.pandas(desc='Start Group Gzhs[text]')

    # 进度条
    mysql_dao.select_table(gv.GZH_TABLE, gv.GZH_SELECT)[['biz', 'nickname']].progress_apply(
        lambda x:
        mysql_dao.insert_table(
            table_name='gzhs_texts_bytdate',
            type_dict={},
            df_values=group_text_bydate(

                mysql_dao.select_table(
                    'text_for_merge',
                    ['*'],
                    {'biz': '\'{0}\''.format(x['biz'])}),

                x['biz'],
                x['nickname']
            )
        )
        ,
        axis=1)


start_group()
