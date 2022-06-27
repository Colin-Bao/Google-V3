#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :do_map.py
# @Time      :2022/6/17 00:53
# @Author    :Colin
# @Note      :None


def start():
    from date_map import create_map, map_article
    from log_rec import bar

    bar = bar.Bar('MAP IMG', 2).get_bar()

    create_map.start_create_info()
    bar.update(1)

    map_article.start_map_tdate()
    bar.update(1)
