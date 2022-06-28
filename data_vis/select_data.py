#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :select_data.py
# @Time      :2022/6/28 16:09
# @Author    :Colin
# @Note      :None
import pandas as pd
import __config as gv


# 按照公众号名称从数据库读取数据
def load_imgpath_fromdb(gzh_name: str) -> list:
    from tools import mysql_dao
    df = mysql_dao.select_table(gzh_name + '封面图片信息', ['*'], {'LIMIT': '512'}).loc[:50, gv.VIS_COLUMN]

    # 组成字典
    listdict = [{'local_cover': value[0], 'cover_neg': value[1], 'p_date': value[2], 'log_return_l1': value[3]} for
                index, value in
                enumerate(df.values)]

    return listdict
