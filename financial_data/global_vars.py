#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :global_vars.py
# @Time      :2022/6/21 19:37
# @Author    :Colin
# @Note      :None

from global_log import global_vars

LOG_FILE = global_vars.LOG_PATH + 'TUSHARE' + '.log'

INDEX_LIST = ['399300.SZ', '000001.SH']
INDEX_TABLE_COLUMN = {'PK': 'trade_date', 'ts_code': 'VARCHAR(40)', 'date_ts': 'INT', 'trade_date': 'INT'}
