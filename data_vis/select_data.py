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
                                    {'nickname': "\'{0}\'".format(gzh_name), 'LIMIT': '128'}
                                    )

    else:
        df = mysql_dao.select_table('所有媒体封面图片信息',
                                    gv.VIS_COLUMN,
                                    {'LIMIT': '128'}
                                    ).loc[:50, gv.VIS_COLUMN]

    # 把df组成字典
    listdict = [{j: value[i] for i, j in enumerate(df.columns)} for index, value in enumerate(df.values)]

    return listdict


# 按照聚合方式读取数据
def load_img_fromdb_bygroup(gzh_name: str, date: str = None):
    df = mysql_dao.select_table('所有媒体封面图片信息_外连',
                                ['*'],
                                {'nick_name': "\'{0}\'".format(gzh_name), 'LIMIT': '512'}
                                )
    df_g = df.groupby('t_date')

    # 解析里面的组获得想要的信息
    for group in enumerate(df_g):
        print(group)
    return df_g
    # return [{j: value[i] for i, j in enumerate(df.columns)} for index, value in enumerate(df.values)]


load_img_fromdb_bygroup('中国证券报')
