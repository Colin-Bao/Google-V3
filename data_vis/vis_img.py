#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :vis_img.py
# @Time      :2022/6/27 16:44
# @Author    :Colin
# @Note      :None
import pandas as pd
import __config as gv


# 按照公众号名称从数据库读取数据
def load_img_fromdb(nick_name: str) -> pd.DataFrame:
    from tools import mysql_dao
    return mysql_dao.excute_sql(gv.SELECT_SQL.format(nick_name))


print(load_img_fromdb('中国证券报').columns)
