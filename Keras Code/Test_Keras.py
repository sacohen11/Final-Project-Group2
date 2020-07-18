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

'''
Machine Learning I - Project
Authors:    Sam Cohen
            Luis Alberto
            Rich Gude
'''

#::------------------------------------------------------------------------------------
## Libraries
#::------------------------------------------------------------------------------------

import os                       # Used to manipulate and pull in data files
import cv2                      # Used to store jpg files
import numpy as np              # Used for array characterization
import joblib

from sklearn import preprocessing
import matplotlib.pyplot as plt
import warnings
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings("ignore")

# For Keras Modeling
from tensorflow import keras
from keras.initializers import glorot_uniform
from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import BatchNormalization
from keras.layers import Dropout
from keras.utils import to_categorical

#::------------------------------------------------------------------------------------
## FUNCTIONS
#::------------------------------------------------------------------------------------

def namestr(obj, namespace):
    '''
    Returns the name of an object.
    Source: https://stackoverflow.com/questions/1538342/how-can-i-get-the-name-of-an-object-in-python
    '''
    return [name for name in namespace if namespace[name] is obj]


def edge_detection(img):
    '''
    Performs a Canny Edge Detection Algorithm.
    Returns an image with only the major edges included.
    More information here: https://docs.opencv.org/2.4/doc/tutorials/imgproc/imgtrans/canny_detector/canny_detector.html
    '''
    return cv2.Canny(img, 200, 600)

def threshold(img):
    '''
    Inverts color scheme (black = white) based on a threshold.
    The second parameter (first number after img) is the threshold value.
    Any pixel above that value will be black, anything below will be white.
    Returns a black and white image.
    '''
    ret, img_threshold = cv2.threshold(img, 220, 255, cv2.THRESH_BINARY)
    return img_threshold

#############################################################################################################

#::------------------------------------------------------------------------------------
##PREPROCESSING
#::------------------------------------------------------------------------------------

# Specify current working directory:
os.chdir('..')
cwd = os.getcwd()

# Standardizing the features
sc_X = StandardScaler()

### --- ### --- ### --- ### --- ### --- ### --- ### --- ### --- ### --- ### --- ### --- ### ---
# PQT Code:
### --- ### --- ### --- ### --- ### --- ### --- ### --- ### --- ### --- ### --- ### --- ### ---

class DrawingPad(QtWidgets.QLabel):

    def __init__(self):
        super().__init__()
        canvas = QtGui.QPixmap(400, 400)
        canvas.fill()
        self.setPixmap(canvas)

        self.pen_width = 20

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
    5, 10, 15, 20, 25, 30
]

class QLineWidthButton(QPushButton):
    def __init__(self, width):
        super().__init__()
        self.setFixedSize(QtCore.QSize(40, 40))
        # self.setStyleSheet("background-color: %s;" % '#000000')

# class Classification(QtWidgets.QMessageBox):
#     def __init__(self, classification):
#         super().__init__()
#         self.classification = classification


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

        self.widget = QtWidgets.QWidget()
        self.background = QtWidgets.QVBoxLayout()
        self.widget.setLayout(self.background)

        self.background.addWidget(self.drawing_pad)

        width_list = QtWidgets.QHBoxLayout()
        self.add_width(width_list)
        self.background.addLayout(width_list)

        self.setCentralWidget(self.widget)

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



        self.hbox = QtWidgets.QHBoxLayout()
        self.hbox.addStretch(1)
        #hbox.setGeometry(QtCore.QRect(100,100,100,100))
        self.hbox.addWidget(self.button)

        #self.msgBox = QtWidgets.QMessageBox()

        self.show()

    def on_click(self):
        # self.button.setStyleSheet("border: 2px solid #000000;font: bold;background-color: blue;font-size: 20px;height: 200px;width: 500px;color: white")
        pixmap = self.drawing_pad.pixmap().scaled(80, 80, aspectRatioMode=Qt.KeepAspectRatio)
        pixmap.save("picture.jpg")
        im = cv2.imread(os.path.join(cwd, 'picture.jpg'), 0)
        height, width = im.shape

        # if the size of the image is greater than 80 pixels in the height, resize to an 80x80 image:
        # applying canny edge detection

        edged = cv2.Canny(im, 10, 250)

        # finding contours
        (_, cnts, _) = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        idx = 0
        for c in cnts:
            x, y, w, h = cv2.boundingRect(c)
            if w > h:
                idx += 1
                preImg = im[y:y + w, x:x + w]
            else:
                idx += 1
                preImg = im[y:y + h, x:x + h]
                # cropping images
                # cv2.imwrite("cropped/" + str(idx) + '.png', new_img)

        # resize to an 80x80 image:
        im = cv2.resize(preImg, (80, 80), interpolation=cv2.INTER_AREA)
        im = threshold(im)


        # load the scalar object from disk and standardize features
        sc_X = joblib.load("sc_X.pkl")
        im = np.reshape(im, (1, -1))        # need to reshape in order to transform
        im = sc_X.transform(im)
        im = np.reshape(im, (1, 80, 80, 1))

        # load classifier model and predict output for image
        classifier = keras.models.load_model("model_keras.hdf5")
        pred = classifier.predict(im)

        # pred will be a nested 4D vector output, with the probability of the shape being a circle, rectangle, square,
        # or triangle in each dimension
        print(pred)

        if np.max(pred) == pred[0, 0]:
            pred_shape = 'Circle'
        if np.max(pred) == pred[0, 1]:
            pred_shape = 'Rectangle'
        if np.max(pred) == pred[0, 2]:
            pred_shape = 'Square'
        if np.max(pred) == pred[0, 3]:
            pred_shape = "Triangle"
        pred_text = f"You drew a {pred_shape}"
        msgBox = QtWidgets.QMessageBox()
        msgBox.setText(pred_text)
        msgBox.setInformativeText("Is this what you drew?")
        msgBox.addButton("Yes", QtWidgets.QMessageBox.DestructiveRole)
        msgBox.addButton("No", QtWidgets.QMessageBox.DestructiveRole)
        msgBox.buttonClicked.connect(self.shapeButton)
        msgBox.exec()
        self.hbox.addWidget(msgBox)


    def shapeButton(self, butt):
        if butt.text() == "Yes":
            self.close()
            MainMenu.__init__(self)
        else:
            msgBox = QtWidgets.QMessageBox()
            msgBox.setText("What did you Draw?")
            msgBox.addButton("Circle", QtWidgets.QMessageBox.YesRole)
            msgBox.addButton("Square", QtWidgets.QMessageBox.YesRole)
            msgBox.addButton("Rectangle", QtWidgets.QMessageBox.YesRole)
            msgBox.addButton("Triangle", QtWidgets.QMessageBox.YesRole)
            msgBox.buttonClicked.connect(self.msgBoxClick)
            msgBox.exec()
            self.hbox.addWidget(msgBox)

    def msgBoxClick(self, butt):
        # add to dataset
        pics = len([file for file in os.listdir(os.path.join(cwd, f'Images/{butt.text()}/'))])
        print(pics)
        os.rename(os.path.join(cwd, 'picture.jpg'), os.path.join(cwd, f'Images/{butt.text()}/{butt.text()}{pics+1000}.jpg'))
        #os.listdir(os.path.join(cwd, f'Images/{butt.text()}') os.path.join(cwd, 'picture.jpg')
        self.close()
        MainMenu.__init__(self)


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