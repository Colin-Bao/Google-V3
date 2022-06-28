#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :do_vis.py
# @Time      :2022/6/28 16:28
# @Author    :Colin
# @Note      :None

import sys

from PyQt6.QtWidgets import QApplication


def link_to_qt():
    from data_vis import select_data, show_img, __config as gv
    df_img_list = select_data.load_imgpath_fromdb()

    app = QApplication(sys.argv)
    qt_obj = show_img.MyIMG('图像消极情绪', 10, 5, 120)

    qt_obj.set_gridimg_update(df_img_list)

    sys.exit(app.exec())


def start():
    link_to_qt()


start()
