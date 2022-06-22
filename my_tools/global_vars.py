#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :global_vars.py
# @Time      :2022/6/21 23:13
# @Author    :Colin
# @Note      :None

from global_log import global_vars

LOG_FILE = global_vars.LOG_PATH + 'SQL_Execute' + '.log'

DEBUG_MODE = False

DEBUG_STR = "[------------执行SQL ----> 记录条数:{1}------------]\n{0}\n"
