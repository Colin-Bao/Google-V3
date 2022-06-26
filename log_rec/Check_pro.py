#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :Check_pro.py
# @Time      :2022/6/26 18:51
# @Author    :Colin
# @Note      :None

from tools import mysql_dao
import pandas as pd
import time


# 查询剩余进度
def select_count_pic() -> int:
    from tools import mysql_dao
    df_limit = mysql_dao.select_table('article_imgs', ['id'],
                                      {'local_cover': 'NOT NULL', 'cover_neg': 'NULL', },
                                      select_count=True)
    return int(df_limit['COUNT(`id`)'][0])


# df = select_count_pic('article_imgs')


from tqdm import tqdm

# 这里同样的，tqdm就是这个进度条最常用的一个方法
# 里面存一个可迭代对象
pic_count = select_count_pic() + 1
for i in tqdm(range(pic_count)):
    pic_count_now = select_count_pic()
    num = pic_count - pic_count_now
    i += num
    # 模拟你的任务
