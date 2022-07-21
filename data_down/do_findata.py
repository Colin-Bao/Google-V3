#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :do_findata.py
# @Time      :2022/6/21 09:27
# @Author    :Colin
# @Note      :None


def start():
    from date_map import create_map, map_article
    from log_rec import bar
    bar = bar.Bar('Down Findata', 2).get_bar()

    from data_down import load_data
    load_data.start_download()
    bar.update(1)

    from data_down import cal_data
    cal_data.start_cal()
    bar.update(1)


start()
