# Just to Start the Window
import sys
from PyQt5.QtWidgets import QMainWindow, QAction, QMenu, QApplication
from PyQt5.QtGui import QIcon
from PyQt5 import QtGui, QtWidgets

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

class MainMenu(QMainWindow):

    def __init__(self):
        super().__init__()

        self.Title = 'Draw your shape!'
        # self.left = 100
        # self.top = 100
        # self.width = 600
        # self.height = 300
        self.label = QtWidgets.QLabel()
        canvas = QtGui.QPixmap(400, 300)
        canvas.fill()
        self.label.setPixmap(canvas)
        self.setCentralWidget(self.label)
        self.initial_point , self.next_point = None, None
        self.initUI()

    def initUI(self):
        self.setWindowTitle(self.Title)
        # self.setGeometry(self.left, self.top, self.width, self.height)
        self.statusBar()
        mainPage = self.menuBar()
        fileMenu = mainPage.addMenu('File')

        self.show()

    def mouseMoveEvent(self, a):
        if self.initial_point is None:
            self.initial_point = a.x()
            self.next_point = a.y()
            return


        painter = QtGui.QPainter(self.label.pixmap())
        pen = QtGui.QPen()
        pen.setWidth(10)
        painter.setPen(pen)
        painter.drawLine(self.initial_point, self.next_point, a.x(), a.y())
        painter.end()
        self.update()

        self.initial_point = a.x()
        self.next_point = a.y()

    def mouseReleaseEvent(self, a):
        self.initial_point= None
        self.initial_point = None


def main():
    app = QApplication(sys.argv)
    mn = MainMenu()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()

##################
# Sources #
# https://www.learnpyqt.com/courses/custom-widgets/bitmap-graphics/