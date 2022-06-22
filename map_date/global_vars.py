#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :global_vars.py
# @Time      :2022/6/21 16:54
# @Author    :Colin
# @Note      :None
LOG_PATH = '/Users/mac/PycharmProjects/Google-V3/global_log/log_file/'
# TRADE
TRADE_TABLE = '399300.SZ'
TRADE_TABLE_SELECT = ['date_ts', 'trade_date']
# INFO
INFO_TABLE = 'info_date'
INFO_TABLE_COLUMN = {'date_ts': 'INT', 'nature_date': 'DATE', 'nature_datetime_ts': 'INT',
                     'nature_datetime': 'DATETIME', 'day_tradedate': 'DATE', 'night_tradedate': 'DATE',
                     'trade_date': 'VARCHAR(50)', 'PK': 'date_ts'}
INFO_TABLE_SELECT = ['date_ts', 'nature_datetime_ts', 'day_tradedate', 'night_tradedate']

# ARTICLE
ARTICLE_TABLE = 'articles'
ARTICLE_TABLE_SELECT = ['id', 'p_date']
ARTICLE_TABLE_FILTER = {'date_ts': 'NULL', 'p_date': 'NOT NULL'}
ARTICLE_TABLE_UPDATE = ['t_date', 'date_ts', 'id']
