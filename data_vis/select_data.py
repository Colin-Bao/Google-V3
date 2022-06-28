#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :select_data.py
# @Time      :2022/6/28 16:09
# @Author    :Colin
# @Note      :None
import pandas as pd
import __config as gv


# 按照公众号名称从数据库读取数据
def load_img_fromdb() -> pd.DataFrame:
    from tools import mysql_dao
    return mysql_dao.select_table('按照公众号查看所有图片', ['*'])
    # return mysql_dao.excute_sql(gv.SELECT_SQL.format(nick_name))


def load_imgpath_fromdb() -> list:
    df = load_img_fromdb().loc[:50, gv.VIS_COLUMN]
    listdict = [{'local_cover': value[0], 'cover_neg': value[1], 'p_date': value[2], 'log_return_l1': value[3]} for
                index, value in
                enumerate(df.values)]

    return listdict
