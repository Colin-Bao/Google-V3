#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :do_log.py
# @Time      :2022/6/22 12:02
# @Author    :Colin
# @Note      :None


def init_log():
    import logging
    logging.basicConfig(filename='global.log', encoding='utf-8', level=logging.INFO)
    # logging.debug('This message should go to the log file')
    # logging.info('So should this')
    # logging.warning('And this, too')
    # logging.error('And non-ASCII stuff, too, like Øresund and Malmö')
