#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :check_null.py
# @Time      :2022/6/22 10:13
# @Author    :Colin
# @Note      :None

import pandas as pd
from data_clean import __config as gv


# 检查空列
def check_null_column(table_name: str) -> pd.DataFrame:
    from tools import mysql_dao as md
    # li = [mysql_dao.select_table(table_name, ['COUNT(' + i + ''], {i: 'NULL'}) for i in
    #       mysql_dao.select_columns(table_name)]
    # return li['COUNT i']
    df_count = md.select_table(table_name, md.select_columns(table_name), select_count=True)

    df_count_null = pd.DataFrame.transpose(
        df_count[:].apply(lambda x: round((1 - (x / x['COUNT(*)'])) * 100, 2), axis=1))

    df_count_null.rename(columns={0: 'NULL_PCT (%)'}, inplace=True)

    df_count_t = pd.DataFrame.transpose(df_count)
    df_count_t.rename(columns={0: 'ROW_COUNT'}, inplace=True)

    df_con = pd.concat([df_count_t, df_count_null], axis=1)
    df_con.sort_values(by='ROW_COUNT', inplace=True, ascending=False)

    return df_con


def start_check(table_name: list):
    dict_df = {i: check_null_column(i) for i in table_name}
    print(dict_df)
    return dict_df


start_check(gv.CHECK_TABLE)
