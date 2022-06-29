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
# 按照聚合方式读取数据
def load_img_fromdb(gzh_name: str = None, group_by: str = '', date: str = None) -> list:
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
    listdict = [{j: value[i] for i, j in enumerate(gv.VIS_COLUMN)} for index, value in enumerate(df.values)]

    return listdict
