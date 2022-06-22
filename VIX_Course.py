#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :VIX_Course.py
# @Time      :2022/6/20 15:50
# @Author    :Colin
# @Note      :None
import pandas as pd
from datetime import datetime


def get_from_tu():
    from data_down import tushare_api
    import tushare as ts
    pro = ts.pro_api('99dec24a97c268cac40c0761443d2ceaa2ae4089949c65d61630b3fe')

    df = pro.opt_basic(exchange='DCE', fields='ts_code,name,exercise_type,list_date,delist_date')
    print(df)
    # 获取K线数据的日期
    tu = tushare_api.TuShareGet('20120101', '20220601')
    # 获取的指数
    df_tu = pd.DataFrame()
    print(tu.get_opt_basic)
    # 转换为dt方便计算
    df_tu['date_ts'] = df_tu[['trade_date', ]].apply(
        lambda x: datetime.strptime(x['trade_date'], '%Y%m%d').date(),
        axis=1)
    df_tu['date_ts'] = df_tu[['date_ts', ]].apply(
        lambda x: int(pd.to_datetime(x['date_ts']).timestamp()),
        axis=1)

    # 排序以填充
    df = df_tu.sort_values(by='date_ts')
    # 筛选需要的行
    return df


def down_data():
    code = ''
    from data_down import load_data
    df = load_fin_data.get_from_tu(code)
    print(df)
    # from tools import mysql_dao
    # mysql_dao.insert_table(code, df,
    #                        {'PK': 'trade_date', 'ts_code': 'VARCHAR(40)', 'date_ts': 'INT', 'trade_date': 'INT'})


get_from_tu()
