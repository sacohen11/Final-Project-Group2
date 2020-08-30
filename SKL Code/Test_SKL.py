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
import random
import warnings
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import pandas as pd
from sklearn.neural_network import MLPClassifier
import timeit                  # Used for determining processing time for MLP
warnings.filterwarnings("ignore")

#::------------------------------------------------------------------------------------
## FUNCTIONS
#::------------------------------------------------------------------------------------

def namestr(obj, namespace):
    '''
    Returns the name of an object.
    Source: https://stackoverflow.com/questions/1538342/how-can-i-get-the-name-of-an-object-in-python
    '''
    return [name for name in namespace if namespace[name] is obj]

def sp_noise(image, prob):
    '''
    Add salt and pepper noise to a grey-scale image (i.e., randomly change pixels to 0 or 255 intensity)
    prob: Probability of the noise (prob = 1 will cause function to return a complete noise, black-and-white image)
    Source: https://stackoverflow.com/questions/22937589/how-to-add-noise-gaussian-salt-and-pepper-etc-to-image-in-python-with-opencv
    '''
    output = np.zeros(image.shape, np.uint8)
    # For every pixel in the image (containing only a single value 0 to 255):
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            # generate a random number [0, 1)
            rdn = random.random()
            # if the number is less than the probability, the pixel will change to either black or white
            if rdn < prob:
                if rdn < prob/2:
                    output[i][j] = 0
                else:
                    output[i][j] = 255
            else:
                output[i][j] = image[i][j]
    return output

def augment (x_train, y_train, f = 0):
    '''
    Generate new images of minority classes via augmentation to a minimum value of "f"
    "f" defaults to the maximum count of any class, unless specified, so all classes have equal representation in model.
    New images are either rotated 90, rotated 180, flipped, switched or noised.
    This function requires that the x_train and y_train arrays are symmetrical (i.e., tied to each other) and sorted
    according to the classes in y_train.
    '''

    # Generate a count of all classes in the presented, training dataset
    unique, count = np.unique(y_train, return_counts=True)

    print("Training set:")
    print("Circles:", count[0])
    print("Rectangles:", count[1])
    print("Squares:", count[2])
    print("Triangles:", count[3])
    print("Total:", len(y_train))
    print("-"*50)

    print("Data Augmentation...")
    x_train_new = []
    y_train_new = []

    # If "f" is unspecified, set it to the maximum count of any particular class so that each class is represented
    # equally
    if f == 0:
        f = max(count)

    print("-"*50)
    # For each class ('circle', 'rectangle', 'square', or 'triangle'):
    for i in range(len(unique)):
        # If the count of any class is below the minimum:
        if count[i] < f:
            # k is the iterable for a while loop; k needs to start at the index of the first instance of each class
            k = np.where(y_train == unique[i])[0][0]
            print(k)
            while count[y_train[k]] < f:
                # Randomly choose one of five options for image augmentation:
                rn = random.randint(0,4)
                if rn == 0:
                    '''
                    Flip the image along the Y-axis
                    '''
                    x_temp = np.fliplr(x_train[k])
                    # print("Flipped", y_train[k])
                    # plt.imshow(x_train[k])
                    # plt.show()

                elif rn ==1:
                    '''
                    Adds noise to an image
                    '''
                    x_temp = sp_noise(x_train[k], 0.01)
                    # print("Noised", y_train[k])
                    # plt.imshow(x_train[k])
                    # plt.show()


                elif rn ==2:
                    '''
                    Rotates 90 degrees
                    Source: https://www.programcreek.com/python/example/89459/cv2.getRotationMatrix2D
                    '''
                    rows, cols = x_train[k].shape[:2]
                    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), 90, 1)
                    x_temp = cv2.warpAffine(x_train[k], M, (cols, rows))
                    # print("Rotated 30", y_train[k])
                    # plt.imshow(x_train[k])
                    # plt.show()


                elif rn == 3:
                    '''
                    Translation of image
                    Source: http: // wiki.lofarolabs.com / index.php / Translation_of_image
                    '''
                    # shifting the image 100 pixels in both dimensions
                    rows, cols = x_train[k].shape[:2]
                    M = np.float32([[1, 0, -5], [0, 1, -5]])
                    x_temp = cv2.warpAffine(x_train[k], M, (cols, rows))
                    # print("Shifted", y_train[k])
                    # plt.imshow(x_train[k])
                    # plt.show()


                elif rn == 4:
                    '''
                    Rotates in 180 degrees
                    Source: https://www.programcreek.com/python/example/89459/cv2.getRotationMatrix2D
                    '''
                    rows, cols = x_train[k].shape[:2]
                    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), 180, 1)
                    x_temp = cv2.warpAffine(x_train[k], M, (cols, rows))
                    # print("Rotated 180", y_train[k])
                    # plt.imshow(x_train[k])
                    # plt.show()

                x_train_new.append(x_temp)
                # print("New image saved as:", y_train[k])
                y_train_new.append(y_train[k])
                # print("Labeled as:", y_train[k])
                count[y_train[k]] += 1
                # print("-" * 50)

                # if the k iterable reaches the end of the indexes of x_train without the count of the class reaching
                # f, reset k to initial (and begin going back through x_train for more augmentation), else add 1 to k.
                if (k - np.where(y_train == unique[i])[0][0]) < count[i]:
                    k += 1
                else:
                    k = np.where(y_train == unique[i])[0][0]

    x_train_new = np.array(x_train_new)
    print(x_train_new.shape, f)
    x_train = np.append(x_train, x_train_new, axis=0)
    y_train_new = np.array(y_train_new)
    y_train = np.append(y_train, y_train_new, axis=0)

    print("Training set after data augmentation:")
    print("Circles:", count[0])
    print("Rectangles:", count[1])
    print("Squares:", count[2])
    print("Triangles:", count[3])
    print("Total:", len(y_train))
    print("-" * 50)
    return x_train, y_train

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

# # Specify current working directory:
os.chdir('..')
cwd = os.getcwd()

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

        # if the size of the image is greater than 80 pixels in the height, resize to an 80x80 image:
        im = cv2.resize(preImg, (80, 80), interpolation=cv2.INTER_AREA)

        im = threshold(im)

        plt.imshow(im)
        plt.show()


        im = np.reshape(im, (1, -1))

        # load the model from disk
        classifier = joblib.load("model.pkl")

        # Standardizing the features
        im = sc_X.transform(im)

        pred = classifier.predict(im)

        print(pred)

        if pred[0] == 0:
            pred_shape = 'Circle'
        if pred[0] == 1:
            pred_shape = 'Rectangle'
        if pred[0] == 2:
            pred_shape = 'Square'
        if pred[0] == 3:
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