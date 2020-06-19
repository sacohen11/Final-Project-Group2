# Just to Start the Window
import sys
from PyQt5.QtWidgets import QMainWindow, QAction, QMenu, QApplication
from PyQt5.QtGui import QIcon
from PyQt5 import QtGui, QtWidgets, QtCore

# For Vertical Layouts
from PyQt5.QtCore import pyqtSlot
from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import QWidget, QLabel, QVBoxLayout
from PyQt5.QtWidgets import QSizePolicy

# For Controls and Triggers
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QCheckBox    # Check Box
from PyQt5.QtWidgets import QRadioButton # Radio Buttons
from PyQt5.QtWidgets import QGroupBox    # Group Box
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure     # Figure

class DrawingPad(QtWidgets.QLabel):

    def __init__(self):
        super().__init__()

        # self.left = 100
        # self.top = 100
        # self.width = 600
        # self.height = 300
        #self.label = QtWidgets.QLabel()
        canvas = QtGui.QPixmap(400, 300)
        canvas.fill()
        #self.label.setPixmap(canvas)
        self.setPixmap(canvas)
        self.pen = QtGui.QPen()
        self.pen_width = 5
        self.initial_point , self.next_point = None, None

    def set_pen_width(self, w):
        self.pen_width = w

    def mouseMoveEvent(self, a):
        if self.initial_point is None:
            self.initial_point = a.x()
            self.next_point = a.y()
            return


        painter = QtGui.QPainter(self.pixmap())
        self.pen.setWidth(self.pen_width)
        painter.setPen(self.pen)
        painter.drawLine(self.initial_point, self.next_point, a.x(), a.y())
        painter.end()
        self.update()

        self.initial_point = a.x()
        self.next_point = a.y()

    def mouseReleaseEvent(self, a):
        self.initial_point= None
        self.initial_point = None

class QLineWidthButton(QtWidgets.QPushButton):

    def __init__(self, width):
        super().__init__()
        self.setFixedSize(QtCore.QSize(24, 24))
        # self.width = width
        # painter = QtGui.QPainter(self.pixmap())
        # self.pen = QtGui.QPen()
        # self.pen.setWidth(width)
        # painter.setPen(self.pen)
        # painter.drawLine(self.initial_point, self.next_point, a.x(), a.y())
        # painter.end()


        self.setStyleSheet("background-color: %s;" % '#000000')

WIDTHS = [
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
]

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.Title = 'Draw your shape!'
        self.drawing_pad = DrawingPad()

        widg = QtWidgets.QWidget()
        main = QtWidgets.QVBoxLayout()
        widg.setLayout(main)
        main.addWidget(self.drawing_pad)

        width_box = QtWidgets.QHBoxLayout()
        self.add_width(width_box)
        main.addLayout(width_box)

        self.setCentralWidget(widg)

        self.initUI()

    def initUI(self):
        self.setWindowTitle(self.Title)
        # self.setGeometry(self.left, self.top, self.width, self.height)
        self.statusBar()
        mainPage = self.menuBar()
        fileMenu = mainPage.addMenu('File')

        self.show()

    def add_width(self, layout):
        for i in WIDTHS:
            w = QLineWidthButton(i)
            w.pressed.connect(lambda i=i: self.drawing_pad.set_pen_width(i))
            layout.addWidget(w)


def main():
    app = QApplication(sys.argv)
    mn = DrawingPad()
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()

##################
# Sources #
# https://www.learnpyqt.com/courses/custom-widgets/bitmap-graphics/