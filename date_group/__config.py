#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :__config.py
# @Time      :2022/6/21 20:25
# @Author    :Colin
# @Note      :None
LOG_PATH = '/log_rec/log_file/'

FIN_TABLE = '399300.SZ'
FIN_SELECT = ['date_ts', 'ts_code', 'trade_date', 'vol', 'log_return', 'log_return', 'weekday']

GZH_TABLE = 'gzhs'
GZH_SELECT = ['biz', 'nickname']

IMG_SELECT = []

PGROUP_TABLE = 'gzhs_imgs_bydate'
TGROUP_TABLE = 'gzhs_imgs_bytdate'

PGROUP_CSV_PATH = '/Users/mac/PycharmProjects/Google-V3/result_datas/group_table/publish_date/'
TGROUP_CSV_PATH = '/Users/mac/PycharmProjects/Google-V3/result_datas/group_table/trade_date/'

TEXT_ATTR_TYPE = {'id': 'INT', 'title_pos': 'FLOAT', 'title_neu': 'mean', 'title_neg': 'mean', 't_pos_tag': 'sum',
                  't_neu_tag': 'sum', 't_neg_tag': 'sum'}
