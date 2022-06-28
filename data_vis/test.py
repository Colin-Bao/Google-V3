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
                             QPushButton, QApplication, QLabel, QFrame, )
from PyQt6.QtGui import QPixmap


class ButtonTest(QWidget):
    def __init__(self):
        super().__init__()

        self.button1 = QPushButton("Button 1")
        self.button2 = QPushButton("Button 2")

        self.myframe = QFrame()
        self.myframe.setFrameShape(QFrame.Shape.StyledPanel)
        # self.myframe.setFrameShadow(QFrame.Plain)
        self.myframe.setLineWidth(3)

        buttonlayout = QVBoxLayout(self.myframe)
        buttonlayout.addWidget(self.button1)
        buttonlayout.addWidget(self.button2)

        self.button3 = QPushButton("Button 1")
        self.button4 = QPushButton("Button 2")

        self.myframe2 = QFrame()
        self.myframe2.setFrameShape(QFrame.Shape.StyledPanel)
        # self.myframe.setFrameShadow(QFrame.Plain)
        self.myframe2.setLineWidth(3)
        buttonlayout2 = QHBoxLayout(self.myframe2)
        buttonlayout2.addWidget(self.button3)
        buttonlayout2.addWidget(self.button4)

        #
        lay = QVBoxLayout(self)
        lay.addWidget(self.myframe)
        lay.addWidget(self.myframe2)

        self.show()


def main():
    app = QApplication(sys.argv)
    ex = ButtonTest()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
