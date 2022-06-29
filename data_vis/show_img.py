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
                             QPushButton, QApplication, QLabel, QFrame, QLCDNumber, QSlider, QComboBox)
from PyQt6.QtGui import QPixmap, QFont, QPalette
from PyQt6.QtCore import Qt
from data_vis import __config as gv
from data_vis import select_data


class MyIMG(QWidget):

    def __init__(self, title='MYIMG', img_w: int = 8, img_h: int = 5, img_sc: int = 120):
        super().__init__()

        # 形参
        self.title = title
        self.img_w = img_w
        self.img_h = img_h
        self.img_sc = img_sc

        # 控件的参数
        self.option_para = {'媒体': '全部', '聚合': '不聚合'}

        # 基本布局
        self.frame_option = QFrame()
        self.option_layout = QHBoxLayout()
        self.frame_img = QFrame()
        self.img_layout = QGridLayout()

        # 主要布局
        # sudo xcodebuild -license
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

        # 绑定按钮
        # self.init_button_event()

        # 展示
        self.move(0, 0)
        self.setWindowTitle(self.title)
        self.show()

    def init_frame_option(self):
        # 为layout安装frame
        self.frame_option.setFrameShape(QFrame.Shape.StyledPanel)
        # self.myframe.setFrameShadow(QFrame.Plain)
        self.frame_option.setLineWidth(3)

        # layout的格式调整
        self.option_layout = QHBoxLayout(self.frame_option)
        self.option_layout.setSpacing(0)
        self.option_layout.setContentsMargins(0, 0, 0, 0)

        # 在layout1中增加组件
        def create_widget_la1():
            from tools import mysql_dao
            # 先在option_layout中增加frame

            # 设置面板的数量
            for frame_addnum in range(3):
                frame = QFrame()
                frame.setLineWidth(1)
                frame.setFrameShape(QFrame.Shape.StyledPanel)
                frame_layout = QHBoxLayout(frame)
                self.option_layout.addWidget(frame)

                #    再到frame中增加组件
                if frame_addnum == 0:
                    lab = QLabel('选择媒体')
                    frame_layout.addWidget(lab)

                    # 绑定按钮事件
                    for table in [str(i).split('封面图片信息', 2)[0] for i in mysql_dao.show_tables() if '封面图片信息' in i]:
                        button = QPushButton(table)
                        button.setObjectName(table)
                        frame_layout.addWidget(button)
                        button.clicked.connect(self.handle_camsave)

                # 第2个面板
                elif frame_addnum == 1:

                    lab = QLabel('设置聚合方式')
                    combo = QComboBox()

                    for i in ['不聚合', '按照自然日期聚合', '按照交易日期聚合']:
                        combo.addItem(i)
                        combo.textActivated[str].connect(self.handle_camsave)

                    for i in [lab, combo]:
                        frame_layout.addWidget(i)

                # 第3个面板
                elif frame_addnum == 2:
                    lab = QLabel('筛选日期')

                    for i in [lab, ]:
                        frame_layout.addWidget(i)

            #

            #     self.option_layout.addWidget(button)

        #  在layout2中增加组件

        create_widget_la1()

    def init_frame_img(self):

        # 为layout安装frame
        self.frame_img.setFrameShape(QFrame.Shape.StyledPanel)
        self.frame_img.setLineWidth(1)
        # self.frame_img.setStyleSheet("background-color: rgb(248,207,223);")

        self.img_layout = QGridLayout(self.frame_img)
        self.img_layout.setSpacing(0)

        # 把位置和组件匹配,并在img_layout中增加组件
        def add_img_layout():
            # 生成初始图片数量
            img_num = range(self.img_h * self.img_w)

            # 生成位置参数
            positions = [(i, j) for i in range(self.img_h) for j in range(self.img_w)]

            # 遍历每一个网格并在网格中增加组件
            for position, path in zip(positions, img_num):
                # 为layout安装frame
                frame = QFrame()
                frame.setFrameShape(QFrame.Shape.StyledPanel)
                # frame.setLineWidth(0)
                # frame.setStyleSheet("border:0.5px solid rgb(0,0,0)")

                frame_layout = QVBoxLayout(frame)

                # 每个frame的layout增加组件
                for add_num in gv.VIS_COLUMN:
                    lbl = QLabel(add_num)
                    lbl.setObjectName(add_num)
                    font = lbl.font()
                    font.setPointSize(10)
                    lbl.setFont(font)
                    if add_num == 'local_cover':
                        pixmap = QPixmap('test.png').scaledToWidth(self.img_sc)
                        lbl.setPixmap(pixmap)
                        lbl.setFixedSize(self.img_sc, self.img_sc)
                    # 增加
                    frame_layout.addWidget(lbl)

                # 距离
                frame_layout.setSpacing(0)
                frame_layout.setContentsMargins(0, 0, 0, 0)
                frame_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

                # 按照位置属性增加
                self.img_layout.addWidget(frame, *position)

        add_img_layout()

    # 控制按钮绑定
    def handle_camsave(self, text):
        sender = self.sender()
        if isinstance(sender, QPushButton):
            self.set_gridimg_update(select_data.load_imgpath_fromdb(sender.objectName()))
            self.option_para.update({'媒体': sender.objectName()})
        elif isinstance(sender, QComboBox):
            self.option_para.update({'聚合': text})

        print(self.option_para)

    # 获取按钮并绑定事件
    def init_button_event(self):

        # 遍历option_layout中所有的按钮
        for button in [self.option_layout.itemAt(i).widget() for i in range(self.option_layout.count())]:
            if isinstance(button, QPushButton):
                # 绑定数据库查找按钮
                button.clicked.connect(self.handle_camsave)

    def set_gridimg_update(self, img_dictlist: list):
        # 获取qframe小组件列表
        widget_index = [self.img_layout.itemAt(i).widget() for i in range(self.img_layout.count())]

        # 遍历每一个imgframe
        for imgframe, imgdict in zip(widget_index, img_dictlist):

            # 读取Qlabel部分
            for child in imgframe.findChildren((QLabel,)):

                # 改变QLabel
                if isinstance(child, QLabel):
                    # 改变所有的Label显示文字
                    child.setText(str(imgdict[child.objectName()]))

                    # 特殊的情况处理
                    if child.objectName() == "local_cover":
                        pixmap = QPixmap(gv.IMG_PATH + imgdict['local_cover']).scaledToWidth(self.img_sc)
                        child.setPixmap(pixmap)
                    elif child.objectName() == "log_return_l1":
                        log_re = imgdict['log_return_l1']
                        if log_re < 0:
                            child.setStyleSheet("QLabel { color : red; }")
                        else:
                            child.setStyleSheet("QLabel { color : green; }")
                        child.setText("{:.2f}%".format(log_re * 100))


def start_show():
    app = QApplication(sys.argv)
    ex = MyIMG('图像消极情绪', 4, 4, 120)

    sys.exit(app.exec())
