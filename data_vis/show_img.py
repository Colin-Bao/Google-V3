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
                             QPushButton, QApplication, QLabel, QFrame, QLCDNumber, QSlider, QComboBox, QTabWidget,
                             QRadioButton, QSizePolicy)
from PyQt6.QtGui import QPixmap, QFont, QPalette
from PyQt6.QtCore import Qt
from data_vis import __config as gv
from data_vis import select_data


# 重写Qlabel
class MyQLabel(QLabel):
    # 自定义信号, 注意信号必须为类属性
    import PyQt6
    button_clicked_signal = PyQt6.QtCore.pyqtSignal()

    def __init__(self, parent=None):
        super(MyQLabel, self).__init__(parent)

    def mouseReleaseEvent(self, QMouseEvent):
        self.button_clicked_signal.emit()

    # 可在外部与槽函数连接
    def connect_customized_slot(self, func):
        self.button_clicked_signal.connect(func)


# 主选项卡
class MYTab(QTabWidget):
    def __init__(self):
        super().__init__()

        # 增加选项卡
        self.tab1 = QWidget()
        self.tab2 = QWidget()
        self.addTab(self.tab1, '按照日期聚合')
        self.addTab(self.tab2, '不聚合')

        # 初始化选项卡
        self.tab1UI()
        self.tab2UI()

        # 整个窗体
        self.initUI()

    def initUI(self):
        # 展示
        self.setWindowTitle('图像消极情绪V1.0 ')
        self.show()

    #    分别布局
    def tab1UI(self):
        layout = QVBoxLayout()
        layout.setSpacing(0)
        layout.setContentsMargins(0, 0, 0, 0)
        self.tab1.setLayout(layout)
        layout.addWidget(MyIMGGROUP())

    #    分别布局
    def tab2UI(self):
        layout = QVBoxLayout()
        layout.setSpacing(0)
        layout.setContentsMargins(0, 0, 0, 0)
        self.tab2.setLayout(layout)
        layout.addWidget(MyIMG())


# 图像显示聚合面板
class MyIMG(QWidget):

    def __init__(self, title='MYIMG', img_w: int = 10, img_h: int = 5, img_sc: int = 120):
        super().__init__()

        # 形参
        self.title = title
        self.img_w = img_w
        self.img_h = img_h
        self.img_sc = img_sc

        # 用于保存这个控件中的数据
        self.data_list = []

        # 控件的参数
        self.option_para = {'媒体': '全部', '排序': '', '页面': 0}

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

        self.mainlay.addWidget(self.frame_option)
        self.mainlay.addWidget(self.frame_img)

        # 设置主控件大小
        self.mainlay.setSpacing(0)
        self.mainlay.setContentsMargins(0, 0, 0, 0)

        # 绑定按钮
        # self.init_button_event()

    def init_frame_option(self):
        # 为layout安装frame
        self.frame_option.setFrameShape(QFrame.Shape.StyledPanel)
        self.frame_option.setFixedHeight(60)
        # self.myframe.setFrameShadow(QFrame.Plain)
        self.frame_option.setLineWidth(3)

        # layout的格式调整
        self.option_layout = QHBoxLayout(self.frame_option)
        self.option_layout.setSpacing(0)
        self.option_layout.setContentsMargins(0, 0, 0, 0)

        # 在layout1中增加组件
        def create_widget_la1():
            # 先在option_layout中增加frame

            # 设置面板的数量
            for frame_addnum in range(3):
                frame = QFrame()
                frame.setLineWidth(1)
                frame.setFrameShape(QFrame.Shape.StyledPanel)

                # 第1个面板
                if frame_addnum == 0:
                    frame_layout = QGridLayout(frame)
                    self.option_layout.addWidget(frame)

                    # 生成小组件绑定按钮事件
                    button_list = []
                    for table in select_data.load_all_media():
                        button = QRadioButton(table)
                        button_list += [button]
                        button.setObjectName(table)
                        # frame_layout.addWidget(button)
                        button.clicked.connect(self.handle_camsave)

                    # 生成位置参数
                    for position, btn in zip([(i, j) for i in range(2) for j in range(6)], button_list):
                        frame_layout.addWidget(btn, *position)

                # 第2个面板
                elif frame_addnum == 1:
                    frame_layout = QHBoxLayout(frame)
                    self.option_layout.addWidget(frame)

                    lab = QLabel('设置排序方式')
                    combo = QComboBox()

                    for i in ['按照消极概率排序', '按照自然日期聚合', '按照交易日期聚合']:
                        combo.addItem(i)
                        combo.textActivated[str].connect(self.handle_camsave)

                    for i in [lab, combo]:
                        frame_layout.addWidget(i)

                # 第3个面板
                elif frame_addnum == 2:
                    frame_layout = QHBoxLayout(frame)
                    self.option_layout.addWidget(frame)
                    lab = QLabel('页面')
                    frame_layout.addWidget(lab)
                    # 批量增加
                    for i in range(10):
                        btn = QRadioButton(str(i + 1))
                        btn.setObjectName(str(i))
                        btn.clicked.connect(self.handle_camsave)
                        frame_layout.addWidget(btn)

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
        self.img_layout.setContentsMargins(0, 0, 0, 0)

        #

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
                    # 共同使用的对象
                    lbl = QLabel(add_num)

                    if add_num == 'content_url':
                        # 不增加这个组件
                        continue

                    elif add_num == 'local_cover':
                        pixmap = QPixmap().scaledToWidth(self.img_sc)
                        lbl.setPixmap(pixmap)
                        lbl.setFixedSize(self.img_sc, self.img_sc)

                    elif add_num == 'datetime_p':
                        lbl = QPushButton(add_num)

                        # lbl.setFixedWidth(self.img_sc)
                        lbl.clicked.connect(self.handle_camsave)

                    # 统一改变属性
                    lbl.setObjectName(add_num)
                    font = lbl.font()
                    font.setPointSize(10)
                    lbl.setFont(font)

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
        # 保存设置
        old_option_para = self.option_para.copy()
        # 第一个设置面板
        if isinstance(sender, QRadioButton):
            # QRadioButton按钮名称

            if sender.objectName() in [str(i) for i in range(10)]:
                self.option_para.update({'页面': int(sender.objectName())})

            # QRadioButton按钮
            else:
                self.option_para.update({'媒体': sender.objectName()})

        # 第二个设置面板
        elif isinstance(sender, QComboBox):
            self.option_para.update({'排序': text})

        # 图像面板中的按钮
        elif isinstance(sender, QPushButton):
            href = str(sender.styleSheet()).split('href=', 2)[1]

            import webbrowser
            webbrowser.open(href)

        # 如果参数改变tets
        if not old_option_para == self.option_para:
            # print(self.option_para)
            # print(select_data.load_img_fromdb(self.option_para['媒体']))
            self.data_list = select_data.load_img_fromdb(self.option_para['媒体'])
            # 获得筛选后的图像列表
            page_imgnum = self.img_h * self.img_w
            page = self.option_para['页面']
            page_list = self.data_list[int(page * page_imgnum): ((page + 1) * page_imgnum)]
            # 更新
            self.set_gridimg_update(page_list)

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
            for child in imgframe.findChildren((QLabel, QPushButton)):

                # 改变QLabel
                if isinstance(child, QLabel):
                    # 改变所有的Label显示文字
                    child.setText(str(imgdict[child.objectName()]))

                    # 特殊的情况处理
                    if child.objectName() == "local_cover":
                        pixmap = QPixmap(gv.IMG_PATH + imgdict['local_cover']).scaledToWidth(self.img_sc)
                        child.setPixmap(pixmap)
                        # print(imgdict['content_url'])
                    elif child.objectName() == "log_return_l1":
                        log_re = imgdict['log_return_l1']
                        if log_re < 0:
                            child.setStyleSheet("QLabel { color : red; }")
                        else:
                            child.setStyleSheet("QLabel { color : green; }")
                        child.setText("{:.2f}%".format(log_re * 100))

                elif isinstance(child, QPushButton):
                    # 改变所有的Label显示文字
                    child.setText(str(imgdict[child.objectName()]))
                    child.setStyleSheet('href={}'.format(imgdict['content_url']))
                    # child.setAttribute()


# 聚合面板
class MyIMGGROUP(MyIMG):
    def __init__(self):
        super().__init__()

    # 重写方法
    def handle_camsave(self, text):
        sender = self.sender()
        # 保存设置
        old_option_para = self.option_para.copy()
        # 第一个设置面板
        if isinstance(sender, QRadioButton):
            self.option_para.update({'媒体': sender.objectName()})

        # 第二个设置面板
        elif isinstance(sender, QComboBox):
            self.option_para.update({'排序': text})

        # 如果参数改变tets
        if not old_option_para == self.option_para:
            self.set_gridimg_update(self.option_para['媒体'])
            # print(select_data.load_img_fromdb(self.option_para['媒体']))
            # self.set_gridimg_update(select_data.load_img_fromdb(self.option_para['媒体']))

    # 更新
    def set_gridimg_update(self, bizname):
        df_list = select_data.load_img_fromdb_bygroup(bizname)
        # 获取qframe小组件列表
        for index, out_grid in enumerate([self.img_layout.itemAt(i).widget() for i in range(self.img_layout.count())]):
            # 定位到grid
            ly_grid = out_grid.findChild(QGridLayout)
            if isinstance(ly_grid, QGridLayout):
                for index_i, i in enumerate([ly_grid.itemAt(i).widget() for i in range(ly_grid.count())]):
                    if isinstance(i, QLabel):
                        if index < len(df_list):
                            content = df_list[index]['content']
                            if index_i < len(content):
                                pic = QPixmap(gv.IMG_PATH + content[index_i]['local_cover'])
                                i.setPixmap(pic)
                            else:
                                pic = QPixmap()
                                i.setPixmap(pic)
                        else:
                            pic = QPixmap()
                            i.setPixmap(pic)

            # 定位到下方描述
            ly_hbox = out_grid.findChild(QHBoxLayout)
            if isinstance(ly_hbox, QHBoxLayout):
                for i in [ly_hbox.itemAt(i).widget() for i in range(ly_hbox.count())]:
                    if isinstance(i, QLabel):
                        if index < len(df_list):
                            if i.objectName() == 'date':
                                i.setText(str(df_list[index]['date']))
                            elif i.objectName() == 'return':

                                log_re = df_list[index]['log_return_l1']
                                if log_re < 0:
                                    i.setStyleSheet("QLabel { color : red; }")
                                else:
                                    i.setStyleSheet("QLabel { color : green; }")
                                i.setText("    {:.2f}%".format(log_re * 100))

    # 重写方法
    def init_frame_img(self):
        # 为layout安装frame
        self.frame_img.setFrameShape(QFrame.Shape.StyledPanel)
        self.frame_img.setLineWidth(1)
        # self.frame_img.setStyleSheet("background-color: rgb(248,207,223);")

        self.img_layout = QGridLayout(self.frame_img)
        self.img_layout.setSpacing(2)
        self.img_layout.setContentsMargins(0, 0, 0, 0)

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

                # 增加底部描述
                frame_layout_outer = QVBoxLayout(frame)

                # # # # # # # # # # # # # # # # # # # # # # # # # # #

                # 增加框内子框
                grid_frame = QFrame()
                frame_layout_outer_inner1 = QGridLayout(grid_frame)
                frame_layout_outer_inner1.setSpacing(0)
                frame_layout_outer_inner1.setContentsMargins(0, 0, 0, 0)
                # # 每个frame的layout增加组件
                for pos, num in zip([(i, j) for i in range(3) for j in range(3)], range(9)):
                    # 增加
                    lbl = QLabel()
                    lbl.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
                    lbl.setScaledContents(True)
                    pixmap = QPixmap()
                    lbl.setPixmap(pixmap)

                    # 位置
                    frame_layout_outer_inner1.addWidget(lbl, *pos)

                # 增加框内子框
                h_frame = QFrame()
                h_frame.setFixedHeight(20)
                frame_layout_outer_inner2 = QHBoxLayout(h_frame)
                frame_layout_outer_inner2.setSpacing(0)
                frame_layout_outer_inner2.setContentsMargins(0, 0, 0, 0)
                # 下面的描述框
                lbl_des = QLabel()
                lbl_des.setObjectName('date')
                lbl_des2 = QLabel()
                lbl_des2.setObjectName('return')
                frame_layout_outer_inner2.addWidget(lbl_des)
                frame_layout_outer_inner2.addWidget(lbl_des2)
                # # # # # # # # # # # # # # # # # # # # # # # # # # #

                # 内部的框增加组件
                frame_layout_outer.addWidget(grid_frame)
                frame_layout_outer.addWidget(h_frame)
                # # 距离
                frame_layout_outer.setSpacing(0)
                frame_layout_outer.setContentsMargins(0, 0, 0, 0)

                # 按照位置属性增加
                self.img_layout.addWidget(frame, *position)

        add_img_layout()
