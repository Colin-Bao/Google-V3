#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :__config.py
# @Time      :2022/6/21 19:37
# @Author    :Colin
# @Note      :None


LOG_PATH = '/Users/mac/PycharmProjects/Google-V3/log_rec/log_file/'

INDEX_LIST = ['000001.SH', '399001.SZ', '000011.SH', '399307.SZ']
INDEX_TABLE_COLUMN = {'PK': 'trade_date', 'ts_code': 'VARCHAR(40)', 'date_ts': 'INT', 'date': 'DATE',
                      'trade_date': 'INT',
                      'close': 'float', 'vol': 'int', 'pre_close': 'float', 'amount': 'float', 'log_return': 'float'}
