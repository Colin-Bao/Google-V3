#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :__config.py
# @Time      :2022/6/21 20:25
# @Author    :Colin
# @Note      :None
IMG_TABLE = 'article_imgs'
IMG_COLUMN = ['id', 'mov', 'local_cover', 'cover_neg', 'cover_pos']
IMG_FILTER = {}

LOG_PATH = '/Users/mac/PycharmProjects/Google-V3/log_rec/'
MODEL_PATH = '/Users/mac/PycharmProjects/Google-V3/img_predict/twitter_tl_500.h5'
IMG_PATH = '/Users/mac/PycharmProjects/Google-V3/img_down/'

# 合并3张表查询
# 建立了索引速度快
SELECT_SQL = "select `gzhs`.`nickname` AS `nickname`,`articles`.`p_date` AS `p_date`,`articles`.`t_date` AS `t_date`,`articles`.`id` AS `id`,`article_imgs`.`mov` AS `mov`,`article_imgs`.`local_cover` AS `local_cover`,`article_imgs`.`cover_neg` AS `cover_neg`,`article_imgs`.`cover_pos` AS `cover_pos`,`399300.sz`.`log_return` AS `log_return`,`399300.sz`.`amount` AS `amount` from (((`articles` join `article_imgs` on((`articles`.`id` = `article_imgs`.`id`))) join `399300.sz` on((`articles`.`t_date` = `399300.sz`.`date_ts`))) join `gzhs` on((`articles`.`biz` = `gzhs`.`biz`))) where (`gzhs`.`nickname` = '{0}')"
