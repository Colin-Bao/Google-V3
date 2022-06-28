# file: calculator.py
# !/usr/bin/python

"""
ZetCode PyQt6 tutorial

In this example, we create a skeleton
of a calculator using QGridLayout.

Author: Jan Bodnar
Website: zetcode.com
"""

import sys
from PyQt6.QtWidgets import (QWidget, QGridLayout, QHBoxLayout, QVBoxLayout,
                             QPushButton, QApplication, QLabel, QFrame)
from PyQt6.QtGui import QPixmap, QFont, QPalette
from data_vis import __config as gv


class MyIMG(QWidget):

    def __init__(self, title='MYIMG', img_w: int = 8, img_h: int = 5, img_sc: int = 120):
        super().__init__()

        # 形参
        self.title = title
        self.img_w = img_w
        self.img_h = img_h
        self.img_sc = img_sc

        # 基本布局
        self.frame_option = QFrame()
        self.option_layout = QHBoxLayout()
        self.frame_img = QFrame()
        self.img_layout = QGridLayout()

        # 主要布局
        self.mainlay = QVBoxLayout(self)

        # 初始化
        self.initUI()

    def initUI(self):
        # 初始化各个frame容器小组件
        self.init_frame_option()
        self.init_frame_img()

        # 排列frame容器小组件
        self.mainlay.addWidget(self.frame_option)
        self.mainlay.addWidget(self.frame_img)

        # 设置主控件大小
        self.mainlay.setSpacing(0)
        self.mainlay.setContentsMargins(0, 0, 0, 0)
        # 展示
        self.move(0, 0)
        self.setWindowTitle(self.title)
        self.show()

    def init_frame_option(self):
        # 为layout安装frame
        self.frame_option.setFrameShape(QFrame.Shape.StyledPanel)
        # self.myframe.setFrameShadow(QFrame.Plain)
        self.frame_option.setLineWidth(3)
        self.option_layout = QHBoxLayout(self.frame_option)

        # 在layout中增加组件
        button1 = QPushButton("公众号名称")
        button2 = QPushButton("公众号日期")
        self.option_layout.addWidget(button1)
        self.option_layout.addWidget(button2)

    def init_frame_img(self):
        # 为layout安装frame
        self.frame_img.setFrameShape(QFrame.Shape.StyledPanel)
        self.frame_img.setLineWidth(3)
        self.img_layout = QGridLayout(self.frame_img)

        # 生成初始图片数量
        img_num = range(self.img_h * self.img_w)
        # 生成位置参数
        positions = [(i, j) for i in range(self.img_h) for j in range(self.img_w)]

        # 把位置和组件匹配
        for position, path in zip(positions, img_num):
            # 为layout安装frame
            frame = QFrame()
            frame.setFrameShape(QFrame.Shape.StyledPanel)
            frame.setLineWidth(1)
            frame_layout = QVBoxLayout(frame)

            # 每个frame的layout增加组件
            pixmap = QPixmap('test.png').scaledToWidth(self.img_sc)
            lbl_img = QLabel()
            lbl_img.setPixmap(pixmap)
            lbl_img.setObjectName('local_cover')

            # lbl_neg
            lbl_neg = QLabel('cover_neg')
            lbl_neg.setObjectName('cover_neg')

            # lbl_date
            lbl_date = QLabel('p_date')
            lbl_date.setObjectName('p_date')

            # lbl_date
            lbl_return = QLabel('log_return')
            lbl_return.setObjectName('log_return')

            # 字体
            font = lbl_neg.font()
            font.setPointSize(10)
            lbl_neg.setFont(font)
            lbl_date.setFont(font)
            lbl_return.setFont(font)

            # 距离
            frame_layout.setSpacing(0)
            frame_layout.setContentsMargins(0, 0, 0, 0)

            # 增加
            frame_layout.addWidget(lbl_img)
            frame_layout.addWidget(lbl_neg)
            frame_layout.addWidget(lbl_date)
            frame_layout.addWidget(lbl_return)

            self.img_layout.addWidget(frame, *position)

    def set_gridimg_update(self, img_dictlist: list):
        # 获取qframe小组件列表
        widget_index = [self.img_layout.itemAt(i).widget() for i in range(self.img_layout.count())]

        # 遍历每一个imgframe
        for imgframe, imgdict in zip(widget_index, img_dictlist):

            # 读取Qlabel部分
            for child in imgframe.findChildren((QLabel,)):
                if isinstance(child, QLabel):
                    if child.objectName() == "local_cover":
                        pixmap = QPixmap(gv.IMG_PATH + imgdict['local_cover']).scaledToWidth(self.img_sc)
                        child.setPixmap(pixmap)
                    elif child.objectName() == "cover_neg":
                        child.setText(str(imgdict['cover_neg']))
                    elif child.objectName() == "p_date":
                        child.setText(str(imgdict['p_date']))
                    elif child.objectName() == "log_return":
                        log_re = imgdict['log_return_l1']
                        if log_re <= 0:
                            child.setStyleSheet("QLabel { color : red; }")
                        child.setText("{:.2f}%".format(log_re * 100))


def start_show():
    app = QApplication(sys.argv)
    ex = MyIMG('图像消极情绪', 4, 4, 120)
    ex.set_gridimg_update([{'path': 'test.png', 'cover_neg': '1.3', 'pdate': '2011'},
                           {'path': 'test.png', 'cover_neg': '1.3', 'pdate': '2011'}, ])
    if app.exec():
        sys.exit(app.exec())
    return ex
