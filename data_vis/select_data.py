#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :select_data.py
# @Time      :2022/6/28 16:09
# @Author    :Colin
# @Note      :None
import pandas as pd
import __config as gv
from tools import mysql_dao


def load_all_media() -> list:
    return mysql_dao.select_table('gzhs', ['nickname'])['nickname'].tolist()


# 需要返回字典列表

def load_img_fromdb(gzh_name: str = None, date: str = None) -> list:
    if gzh_name:

        df = mysql_dao.select_table('所有媒体封面图片信息',
                                    gv.VIS_COLUMN,
                                    {'nickname': "\'{0}\'".format(gzh_name), 'LIMIT': '1000'}
                                    )

    else:
        df = mysql_dao.select_table('所有媒体封面图片信息',
                                    gv.VIS_COLUMN,
                                    {'LIMIT': '128'}
                                    )

    # 把df组成字典
    listdict = [{j: value[i] for i, j in enumerate(df.columns)} for index, value in enumerate(df.values)]

    return listdict


# 按照聚合方式读取数据
def load_img_fromdb_bygroup(gzh_name: str, date: str = None):
    df_all = mysql_dao.select_table('所有媒体封面图片信息_外连',
                                    ['*'],
                                    {'nick_name': "\'{0}\'".format(gzh_name), 'LIMIT': '2000'}
                                    )
    # 解析里面的组获得想要的信息
    out_list = []
    for date, df in df_all.groupby('t_date'):
        dict_list = [{j: value[i] for i, j in enumerate(df.columns)} for index, value in enumerate(df.values)]
        out_list += [
            {'date': date,
             'c_neg_ratio': dict_list[0]['c_neg_ratio'],
             'log_return_l1': dict_list[0]['log_return_l1'],
             'content': dict_list}]
    # print(out_list)
    return out_list
    # return [{j: value[i] for i, j in enumerate(df.columns)} for index, value in enumerate(df.values)]

# s = load_img_fromdb_bygroup('央视财经')
# print(s)
# print(load_img_fromdb())
