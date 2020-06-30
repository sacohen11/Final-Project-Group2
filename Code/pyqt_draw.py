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

def mode(array):
    '''
    Calculates the mode prediction for each image.
    Searches through each model's predictions and finds the most commmon classification.
    Returns an array of predictions.
    Source: https://www.analyticsvidhya.com/blog/2018/06/comprehensive-guide-for-ensemble-models/
    '''
    output_array = []
    length_array = len(array)
    for i in range(len(array[0])):
        array = np.array(array)
        new_array = array[0:length_array, i]

        # 4 counts for 4 image classifications (circle, rectangle, square, triangle)
        count0 = 0
        count1 = 0
        count2 = 0
        count3 = 0
        for i in new_array:
            if i == 0:
                count0 +=1
            if i == 1:
                count1 +=1
            if i == 2:
                count2 += 1
            if i == 3:
                count3 += 1
        maximum = max(count0, count1, count2, count3)
        doubles = 0
        return_value = 4        # set randomly out of the counts for error detection
        if maximum == count0:
            doubles += 1
            return_value = 0
        if maximum == count1:
            doubles += 1
            return_value = 1
        if maximum == count2:
            doubles +=1
            return_value = 2
        if maximum == count3:
            doubles +=1
            return_value = 3

        # If there are multiple maximum classifications, randomly pick one of them to include:
        #   When checking if multiple classifications have the same number, run the if-chain from the most amount of
        #   classifications (4) to the least (2), else the random selection will exclude potential matches.
        if doubles > 1:
            if doubles == 4:
                output_array.append(random.randint(0, 3))       # randint is endpoint inclusive
            elif maximum == count0 & maximum == count1 & maximum == count2:
                output_array.append(random.randint(0, 2))
            elif maximum == count0 & maximum == count1 & maximum == count3:
                output_array.append(random.choice([0, 1, 3]))    # choice selects randomly from a list
            elif maximum == count0 & maximum == count2 & maximum == count3:
                output_array.append(random.choice([0, 2, 3]))
            elif maximum == count1 & maximum == count2 & maximum == count3:
                output_array.append(random.randint(1, 3))
            elif maximum == count0 & maximum == count1:
                output_array.append(random.randint(0,1))
            elif maximum == count0 & maximum == count2:
                output_array.append(random.randrange(0, 3, 2))      # randrange is not endpoint inclusive
            elif maximum == count0 & maximum == count3:
                output_array.append(random.randrange(0, 4, 3))
            elif maximum == count1 & maximum == count2:
                output_array.append(random.randint(1, 3))
            elif maximum == count1 & maximum == count3:
                output_array.append(random.randrange(1, 3, 2))
            elif maximum == count2 & maximum == count3:
                output_array.append(random.randint(2, 4))
            else:       # This should literally never trigger - here just as a failsafe?
                output_array.append(random.randint(0, 3))
        else:
            output_array.append(return_value)
    return output_array

def image_resize(image, width = None, height = None, inter = cv2.INTER_AREA):
    '''
    Resizes images keeping proportions.

    Only a new height or width should be specified: the ratio between the new width or height and the old will be used
    to scale the other, non-specified dimension.
    Source: https://stackoverflow.com/questions/44650888/resize-an-image-without-distortion-opencv
    '''
    # Initialize the dimensions of the image to be resized and grab the image size
    (w, h) = image.shape[:2]    # the shape function returns the width in the first numeral, and height in the second.

    # If both the width and height are None, then return the original image
    if width is None and height is None:
        return image
    # If both the width and height are specified, then return the cv2.resized image
    if (width is not None) and (height is not None):
        return cv2.resize(image, (width, height), interpolation=inter)

    # If the width is None, the recalculate the new width from the ratio of the heights
    if width is None:
        # calculate the ratio of the height and construct the dimensions
        r = height / float(h)
        dim = (int(w * r), height)
    # If the height is None, the recalculate the new height from the ratio of the widths
    else:
        # calculate the ratio of the width and construct the dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image and return
    resized = cv2.resize(image, dim, interpolation=inter)
    return resized

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

def namestr(obj, namespace):
    '''
    Returns the name of an object.
    Source: https://stackoverflow.com/questions/1538342/how-can-i-get-the-name-of-an-object-in-python
    '''
    return [name for name in namespace if namespace[name] is obj]

def no_transformation(img):
    '''
    Returns the original image.
    '''
    return img

def edge_detection(img):
    '''
    Performs a Canny Edge Detection Algorithm.
    Returns an image with only the major edges included.
    More information here: https://docs.opencv.org/2.4/doc/tutorials/imgproc/imgtrans/canny_detector/canny_detector.html
    '''
    return cv2.Canny(img, 200, 600)

def feature_creation(img):
    '''
    Performs the KAZE feature detection algorithm.
    This algorithm finds the keypoints/features of the image.
    A vector of keypoints is returned.
    Source: https://medium.com/machine-learning-world/feature-extraction-and-similar-image-search-with-opencv-for-newbies-3c59796bf774
    '''
    creator = cv2.KAZE_create()
    # detect
    kps = creator.detect(img)
    vector_size = 32
    kps = sorted(kps, key=lambda x: -x.response)[:vector_size]
    # computing descriptors vector
    kps, img_feature_creation = creator.compute(img, kps)
    # Flatten all of them in one big vector - our feature vector
    img_feature_creation = img_feature_creation.flatten()
    # Making descriptor of same size
    # Descriptor vector size is 64
    needed_size = (vector_size * 64)
    if img_feature_creation.size < needed_size:
        # if we have less the 32 descriptors then just adding zeros at the
        # end of our feature vector
        img_feature_creation = np.concatenate(
            [img_feature_creation, np.zeros(needed_size - img_feature_creation.size)])
    return img_feature_creation

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

# Identify files for images
circleFiles = os.listdir(os.path.join(cwd, 'Images/circle'))
rectangleFiles = os.listdir(os.path.join(cwd, 'Images/rectangle'))
squareFiles = os.listdir(os.path.join(cwd, 'Images/square'))
triangleFiles = os.listdir(os.path.join(cwd, 'Images/triangle'))

circleImages = []
for i in range(len(circleFiles)):
    preIm = cv2.imread(os.path.join(cwd, 'Images/circle/', circleFiles[i]), 0)
    height, width = preIm.shape

    # applying canny edge detection
    edged = cv2.Canny(preIm, 10, 250)

    # finding contours
    cnts, other = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

   # (_, cnts, _) = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    idx = 0
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        if w > 10 and h > 10:
            idx += 1
            preImg = preIm[y:y + h, x:x + w]
            # cropping images
            #cv2.imwrite("cropped/" + str(idx) + '.png', new_img)

    if height < 700:
        preIm = cv2.resize(preImg, (80, 80), interpolation=cv2.INTER_AREA)

    # PREPROCESSING: threshold
    preIm = threshold(preIm)

    circleImages.append(preIm)

plt.imshow(circleImages[np.random.randint(1,300)])
plt.show()

rectangleImages = []
for i in range(len(rectangleFiles)):
    preIm = cv2.imread(os.path.join(cwd, 'Images/rectangle/', rectangleFiles[i]), 0)
    height, width = preIm.shape

    # applying canny edge detection
    edged = cv2.Canny(preIm, 10, 250)

    # finding contours
    cnts, other = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    idx = 0
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        if w > 10 and h > 10:
            idx += 1
            preImg = preIm[y:y + h, x:x + w]

    # if the size of the image is greater than 80 pixels in the height, resize to an 80x80 image:
    if height < 700:
        preIm = cv2.resize(preImg, (80, 80), interpolation=cv2.INTER_AREA)

    # PREPROCESSING: threshold
    preIm = threshold(preIm)

    rectangleImages.append(preIm)

plt.imshow(rectangleImages[np.random.randint(1,300)])
plt.show()

squareImages = []
for i in range(len(squareFiles)):
    preIm = cv2.imread(os.path.join(cwd, 'Images/square/', squareFiles[i]), 0)
    height, width = preIm.shape

    # applying canny edge detection
    edged = cv2.Canny(preIm, 10, 250)

    # finding contours
    cnts, other = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    idx = 0
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        if w > 10 and h > 10:
            idx += 1
            preImg = preIm[y:y + h, x:x + w]

    # if the size of the image is greater than 80 pixels in the height, resize to an 80x80 image:
    if height < 700:
        preIm = cv2.resize(preImg, (80, 80), interpolation=cv2.INTER_AREA)

    # PREPROCESSING: threshold
    preIm = threshold(preIm)

    squareImages.append(preIm)

plt.imshow(squareImages[np.random.randint(1,300)])
plt.show()


triangleImages = []
for i in range(len(triangleFiles)):
    preIm = cv2.imread(os.path.join(cwd, 'Images/triangle/', triangleFiles[i]), 0)
    height, width = preIm.shape

    # applying canny edge detection
    edged2 = cv2.Canny(preIm, 10, 250)

    # finding contours
    cnts, other = cv2.findContours(edged2.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    idx = 0
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        if w > 10 and h > 10:
            idx += 1
            preImg = preIm[y:y + h, x:x + w]
            # cropping images

    # if the size of the image is greater than 80 pixels in the height, resize to an 80x80 image:
    if height > 10:
        preIm = cv2.resize(preImg, (80, 80), interpolation=cv2.INTER_AREA)

    # PREPROCESSING: threshold
    preIm = threshold(preIm)

    triangleImages.append(preIm)

plt.imshow(triangleImages[np.random.randint(1,300)])
plt.show()


#Source of "Find and Crop" function: https://github.com/imneonizer/Find-and-crop-objects-From-images-using-OpenCV-and-Python/blob/master/crop_objects.py

#############################################################################################################

'''
Save each image list as a 3D np array along with a classification array (circle, rectangle, square, or triangle)
'''
npCirc, npCircLab = np.array(circleImages), np.array(['circle']*len(circleImages))
npRect, npRectLab = np.array(rectangleImages), np.array(['rectangle']*len(rectangleImages))
npSqur, npSqurLab = np.array(squareImages), np.array(['square']*len(squareImages))
npTrig, npTrigLab = np.array(triangleImages), np.array(['triangle']*len(triangleImages))

#############################################################################################################

#::------------------------------------------------------------------------------------
##PREPROCESSING
#::------------------------------------------------------------------------------------


print("Starting image and label pre-processing...")

print("-"*50)

# look at labels and images shape
label_data = np.append(npCircLab, np.append(npRectLab, np.append(npSqurLab, npTrigLab)))
print("Labels shape:", label_data.shape)


#One-hot encoding: Convert text-based labels to numbers
le = preprocessing.LabelEncoder()
le.fit(label_data)
integer_labels = le.transform(label_data)

#Confirm we have 4 unique classes
print('Unique classes:', le.classes_)
print("")
print("Images and labels successfully preprocessed!")
print("-"*50)

#::------------------------------------------------------------------------------------
##MODELING
#::------------------------------------------------------------------------------------

x = np.append(npCirc, np.append(npRect, np.append(npSqur, npTrig, axis=0), axis=0), axis=0)
y = integer_labels

# Generate a count of all classes in the presented, training dataset
unique, count = np.unique(y, return_counts=True)

print("Count classes before Data Split")
print("Circles:", count[0])
print("Rectangles:", count[1])
print("Squares:", count[2])
print("Triangles:", count[3])
print("Total:", len(y))
print("-" * 50)

#::---------------------------------------------------------------------------------
## Create a .txt output of model precursors and results
#::---------------------------------------------------------------------------------
# Generate a count of all classes in the total dataset before augmentation
unique, count = np.unique(y, return_counts=True)

file = open('ModelOutput.txt', 'w+')
file.write('Count of Total Images before Augmentation:\n')
file.write("Circles:\t\t%d\n" % count[0])
file.write("Rectangles:\t\t%d\n" % count[1])
file.write("Squares:\t\t%d\n" % count[2])
file.write("Triangles:\t\t%d\n" % count[3])
file.write("Total:\t\t%d\n" % len(y))
file.write("-"*50)
file.close()

x, y = augment(x, y)
# Train, test, split the data
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=40, test_size=0.20)#stratify=y)
x_test_length = len(x_test)
y_test_ex = y_test


#x_train, y_train = augment(x_train, y_train, f=0)
print("Data augmentation completed.")
print("-" * 50)

# Generate a count of all classes in the total dataset after augmentation
unique, count = np.unique(y, return_counts=True)

file = open('ModelOutput.txt', 'a+')
file.write('\n\nCount of Total Images after Augmentation:\n')
file.write("Circles:\t\t%d\n" % count[0])
file.write("Rectangles:\t\t%d\n" % count[1])
file.write("Squares:\t\t%d\n" % count[2])
file.write("Triangles:\t\t%d\n" % count[3])
file.write("Total:\t\t%d\n" % len(y))
file.write("-"*50)
file.close()



# Generate a count of all classes in the training dataset after augmentation
unique, count = np.unique(y_train, return_counts=True)

file = open('ModelOutput.txt', 'a+')
file.write('\nCount of Total Training Images after Augmentation:\n')
file.write("Circles:\t\t%d\n" % count[0])
file.write("Rectangles:\t\t%d\n" % count[1])
file.write("Squares:\t\t%d\n" % count[2])
file.write("Triangles:\t\t%d\n" % count[3])
file.write("Total:\t\t%d\n" % len(y_train))
file.write("-"*50)
file.close()

# Reshape the image data into rows
x_train = np.reshape(x_train, (len(y_train), -1))
print('Training data shape', namestr(x, globals())[0],":", x_train.shape)
x_test = np.reshape(x_test, (len(y_test), -1))
print('Test data shape', namestr(x, globals())[0],":", x_test.shape)

# Standardizing the features
sc_X = StandardScaler()
x_train = sc_X.fit_transform(x_train)
x_test = sc_X.transform(x_test)

#::------------------------------------------------------------------------------------
# Multi-Layer Perceptron
#::------------------------------------------------------------------------------------
hiddenLayers = (20, 50)#, 20)
alphaValues = (0.0001, 0.0010)


#for alpha1 in alphaValues:
    #for hidLay in hiddenLayers:

# Provide a start timer for MLP run
start = timeit.default_timer()
# Create a MLP Classifier
clf = MLPClassifier(solver='sgd',       # MLP will converge via Stochastic Gradient Descent
                            alpha=.0001,       # alpha is convergence rate (low alpha is slow, but won't overshoot solution)
                            hidden_layer_sizes=(100,),        # represents a 6400 - Hidden Layers - 2 MLP
                            random_state=1)
# Train the model using the training sets
clf.fit(x_train, y_train)
# Predict the response for test dataset
y_pred = clf.predict(x_test)

# Provide a stop timer for MLP run
stop = timeit.default_timer()
        # print("-" * 80)
        # print("Model Results", "alphaValue:", alpha1, "hiddenLayers:", hidLay)
        # print("-" * 80)
print("Accuracy of MLP",":", round(metrics.accuracy_score(y_test, y_pred), 3))
        # print("-")
        # print("Confusion Matrix",":")
cmx_MLP = confusion_matrix(y_test, y_pred)
print(cmx_MLP)
        # print("-")
        # print("Classification Report MLP",":")
cfrp = classification_report(y_test, y_pred)
print(cfrp)

# Confusion Matrix Heatmap
class_names = np.unique(label_data)
df_cm = pd.DataFrame(cmx_MLP, index=class_names, columns=class_names)
plt.figure(figsize=(6, 6))
hm = sns.heatmap(df_cm, cmap="Blues", cbar=False, annot=True, square=True, fmt='d', annot_kws={'size': 20},
                 yticklabels=df_cm.columns, xticklabels=df_cm.columns)
hm.yaxis.set_ticklabels(hm.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=10)
hm.xaxis.set_ticklabels(hm.xaxis.get_ticklabels(), rotation=0, ha='right', fontsize=10)
plt.ylabel('True label', fontsize=15)
plt.xlabel('Predicted label', fontsize=15)
plt.title(("MLP Classifier"))

# Show heat map
plt.tight_layout()
plt.show()
        # print("-")
        #
        # file = open('ModelOutput.txt', 'a+')
        # file.write('\nMulti-Layer Perceptron Metrics:\n')
        # file.write("Descent Model:\t\tStochastic Gradient\n")
        # file.write(f"Alpha:\t\t\t\t{alpha1}\n")
        # file.write(f"Hidden Layers:\t\t\t{hidLay}\n")
        # file.write("-"*50)
        # file.write('\n\nMulti-Layer Perceptron Performance:\n')
        # file.write(f"Run Time:\t\t{stop-start} seconds\n")
        # file.write(f"Accuracy:\t\t{round(metrics.accuracy_score(y_test, y_pred), 3)}\n")
        # file.write("-"*50)
        # file.close()




        #clf.estimator
'''
#Confusion Matrix Heatmap
class_names = np.unique(label_data)
df_cm = pd.DataFrame(cmx_SVM, index=class_names, columns=class_names)
plt.figure(figsize=(6, 6))
hm = sns.heatmap(df_cm, cmap="Blues", cbar=False, annot=True, square=True, fmt='d', annot_kws={'size': 20},
                 yticklabels=df_cm.columns, xticklabels=df_cm.columns)
hm.yaxis.set_ticklabels(hm.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=10)
hm.xaxis.set_ticklabels(hm.xaxis.get_ticklabels(), rotation=0, ha='right', fontsize=10)
plt.ylabel('True label', fontsize=15)
plt.xlabel('Predicted label', fontsize=15)
plt.title(("SVM", namestr(i, globals())[0]))
# Show heat map
plt.tight_layout()
plt.show()

#Cross Validation Score
from sklearn.model_selection import cross_val_score
print("Cross Validation Score", namestr(i, globals())[0],":")
accuracies = cross_val_score(estimator=svm.NuSVC(kernel="linear"), X=x_train, y=y_train, cv=5)
print(accuracies)
print("Mean of Accuracies")
print(accuracies.mean())

print("-" * 80)

# Source: https://www.datacamp.com/community/tutorials/svm-classification-scikit-learn-python
'''

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
        print(height)
        print(width)
        # if the size of the image is greater than 80 pixels in the height, resize to an 80x80 image:
        # applying canny edge detection

        edged = cv2.Canny(im, 10, 250)

        # finding contours
        cnts, other = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        idx = 0
        for c in cnts:
            x, y, w, h = cv2.boundingRect(c)
            if w > 10 and h > 10:
                idx += 1
                preImg = im[y:y + h, x:x + w]

        # if the size of the image is greater than 80 pixels in the height, resize to an 80x80 image:
        if height < 700:
            im = cv2.resize(preImg, (80, 80), interpolation=cv2.INTER_AREA)

        im = threshold(im)

        plt.imshow(im)
        plt.show()

        im = np.reshape(im, (1, -1))
        im = sc_X.transform(im)
        print(50 * '-')
        print(im)
        print(50 * '-')
        print(im.shape)
        print(50 * '-')
        print(im.max())
        print(50 * '-')
        pred = clf.predict(im)
        print(pred[0])
        print(50 * '-')
        print(x_test)
        print(50 * '-')
        print(x_test.max())
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