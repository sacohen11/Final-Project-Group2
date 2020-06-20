# Just to Start the Window
import sys
from PyQt5.QtWidgets import QMainWindow, QAction, QMenu, QApplication, QPushButton
from PyQt5.QtGui import QIcon
from PyQt5 import QtGui, QtWidgets, QtCore

# For Vertical Layouts
from PyQt5.QtCore import pyqtSlot
from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import QWidget, QLabel, QVBoxLayout
from PyQt5.QtWidgets import QSizePolicy

# For Controls and Triggers
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QCheckBox  # Check Box
from PyQt5.QtWidgets import QRadioButton  # Radio Buttons
from PyQt5.QtWidgets import QGroupBox  # Group Box
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure  # Figure
class DrawingPad(QtWidgets.QLabel):

    def __init__(self):
        super().__init__()
        canvas = QtGui.QPixmap(400, 400)
        canvas.fill()
        self.setPixmap(canvas)

        self.pen_width = 5

        self.initial_point, self.next_point = None, None
    def mouseMoveEvent(self, a):
        if self.initial_point is None:
            self.initial_point = a.x()
            self.next_point = a.y()
            return

        painter = QtGui.QPainter(self.pixmap())
        pen = QtGui.QPen()
        #pen.setWidth(10)
        pen.setWidth(self.pen_width)
        painter.setPen(pen)
        painter.drawLine(self.initial_point, self.next_point, a.x(), a.y())
        painter.end()
        self.update()

        self.initial_point = a.x()
        self.next_point = a.y()

    def mouseReleaseEvent(self, a):
        self.initial_point = None
        self.initial_point = None

    def set_pen_width(self, w):
        self.pen_width = w


WIDTHS = [
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
]

class QLineWidthButton(QPushButton):
    def __init__(self, width):
        super().__init__()
        self.setFixedSize(QtCore.QSize(40, 40))
        # self.setStyleSheet("background-color: %s;" % '#000000')


class MainMenu(QMainWindow):

    def __init__(self):
        super().__init__()

        self.Title = 'Draw your shape!'
        self.left = 100
        self.top = 100
        self.width = 600
        self.height = 300
        self.setGeometry(self.left, self.top, self.width, self.height)
        #self.label = QtWidgets.QLabel()
        self.drawing_pad = DrawingPad()

        widget = QtWidgets.QWidget()
        background = QtWidgets.QVBoxLayout()
        widget.setLayout(background)

        background.addWidget(self.drawing_pad)

        width_list = QtWidgets.QHBoxLayout()
        self.add_width(width_list)
        background.addLayout(width_list)

        self.setCentralWidget(widget)

        #self.initUI()

    #def initUI(self):
        self.setWindowTitle(self.Title)
        # self.setGeometry(self.left, self.top, self.width, self.height)
        self.statusBar()
        mainPage = self.menuBar()
        fileMenu = mainPage.addMenu('File')

        self.button = QPushButton('CLASSIFY', self)
        self.button.setFixedSize(QtCore.QSize(100, 100))
        #button.setGeometry(QtCore.QRect(500, 150, 93, 28))
        self.button.setStyleSheet("font-weight: bold;font-size: 15px;color: red")

        # self.button.setStyleSheet("border: 2px solid #000000;font: bold;background-color: red;font-size: 20px;height: 200px;width: 500px;color: white")

        #button.setToolTip('This is an example button')
        self.button.move(450, 200)

        self.button.clicked.connect(self.on_click)



        hbox = QtWidgets.QHBoxLayout()
        hbox.addStretch(1)
        #hbox.setGeometry(QtCore.QRect(100,100,100,100))
        hbox.addWidget(self.button)

        self.show()

    def on_click(self):
        # self.button.setStyleSheet("border: 2px solid #000000;font: bold;background-color: blue;font-size: 20px;height: 200px;width: 500px;color: white")
        self.drawing_pad.pixmap().save("picture.jpg")

    def add_width(self, layout):
        for i in WIDTHS:
            w = QLineWidthButton(i)
            w.setText(str(i))

            w.pressed.connect(lambda i=i: self.drawing_pad.set_pen_width(i))
            layout.addWidget(w)

def main():
    app = QApplication(sys.argv)
    mn = MainMenu()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()

##################
# Sources #
# https://www.learnpyqt.com/courses/custom-widgets/bitmap-graphics/