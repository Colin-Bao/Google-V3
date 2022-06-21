#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :do_fin_data.py
# @Time      :2022/6/21 09:27
# @Author    :Colin
# @Note      :None


def start():
    from financial_data import load_fin_data
    load_fin_data.start_download()

    from financial_data import cal_fin_data
    cal_fin_data.start_cal()
