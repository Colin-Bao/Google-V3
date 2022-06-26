#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :load_data.py
# @Time      :2022/6/16 20:10
# @Author    :Colin
# @Note      :None
from datetime import datetime
import pandas as pd
from data_down import tushare_api

from data_down import __config as gv
from log_rec.log import Logger

logger = Logger(logname=gv.LOG_PATH + __name__ + '.log', loggername=__name__).getlog()


def get_from_tu(ts_code) -> pd.DataFrame:
    # 获取K线数据的日期
    tu = tushare_api.TuShareGet('20120101', '20220601')
    # 获取的指数
    try:
        df_kline = pd.DataFrame(tu.get_index(ts_code))
    except Exception as e:
        logger.error(e)
        return pd.DataFrame()

    # 转换为dt方便计算

    # 由于原始信息没有time,因此直接转为timestamp是有效的,不用分离date
    df_kline['date_ts'] = df_kline[['trade_date', ]].apply(
        lambda x: int(pd.to_datetime(x['trade_date'], format='%Y%m%d').timestamp()),
        axis=1)

    df_kline['date'] = df_kline[['trade_date', ]].apply(
        lambda x: pd.to_datetime(x['trade_date'], format='%Y%m%d').date(),
        axis=1)

    df_kline['weekday'] = df_kline[['trade_date', ]].apply(
        lambda x: datetime.strptime(x['trade_date'], '%Y%m%d').weekday(),
        axis=1)
    # 排序以填充
    df = df_kline.sort_values(by='date_ts')

    # 筛选需要的行
    return df


# 下载数据并存入数据库
def start_download():
    #
    index_list = gv.INDEX_LIST
    attr_dict = gv.INDEX_TABLE_COLUMN

    from tools import mysql_dao

    for code in index_list:
        df = get_from_tu(code)
        mysql_dao.insert_table(code, df, attr_dict)
        # mysql_dao.update_table(code, df, attr_dict)

# start_download()
