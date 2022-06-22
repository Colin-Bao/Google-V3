#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :do_map_date.py
# @Time      :2022/6/17 00:53
# @Author    :Colin
# @Note      :None


def start():
    from date_map import create_info_table, map_article_tdate
    create_info_table.start_create_info()
    map_article_tdate.start_map_tdate()
