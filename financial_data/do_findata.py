#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :do_findata.py
# @Time      :2022/6/21 09:27
# @Author    :Colin
# @Note      :None


def start():
    from financial_data import load_data
    load_data.start_download()

    from financial_data import cal_data
    cal_data.start_cal()
    # ;
