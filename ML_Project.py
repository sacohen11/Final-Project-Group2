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
    ret, img_threshold = cv2.threshold(img, 30, 255, cv2.THRESH_BINARY_INV)
    return img_threshold

#############################################################################################################

#::------------------------------------------------------------------------------------
##PREPROCESSING
#::------------------------------------------------------------------------------------

# Specify current working directory:
cwd = os.getcwd()

print("-"*50)
print("Starting image loading and pre-processing...")
print("-"*50)

# Identify files for images
circleFiles = os.listdir(os.path.join(cwd, 'Images/circle'))
rectangleFiles = os.listdir(os.path.join(cwd, 'Images/rectangle'))
squareFiles = os.listdir(os.path.join(cwd, 'Images/square'))
triangleFiles = os.listdir(os.path.join(cwd, 'Images/triangle'))

# Fill up lists with image cv2 files
circleImages = []
for i in range(len(circleFiles)):
    # Prior to input, crop the images to take off bottom tag with unnecessary details (orig image is square)
    preIm = cv2.imread(os.path.join(cwd, 'Images\\circle\\', circleFiles[i]), 0)
    height, width = preIm.shape
    # if the size of the image is greater than 80 pixels in the height, resize to an 80x80 image:
    if height > 80:
        preIm = cv2.resize(preIm, (80, 80), interpolation=cv2.INTER_AREA)
    circleImages.append(preIm)

rectangleImages = []
for i in range(len(rectangleFiles)):
    # Prior to input, crop the images to take off bottom tag with unnecessary details (orig image is square)
    preIm = cv2.imread(os.path.join(cwd, 'Images\\rectangle\\', rectangleFiles[i]), 0)
    height, width = preIm.shape
    # if the size of the image is greater than 80 pixels in the height, resize to an 80x80 image:
    if height > 80:
        preIm = cv2.resize(preIm, (80, 80), interpolation=cv2.INTER_AREA)
    rectangleImages.append(preIm)

squareImages = []
for i in range(len(squareFiles)):
    # Prior to input, crop the images to take off bottom tag with unnecessary details (orig image is square)
    preIm = cv2.imread(os.path.join(cwd, 'Images\\square\\', squareFiles[i]), 0)
    height, width = preIm.shape
    # if the size of the image is greater than 80 pixels in the height, resize to an 80x80 image:
    if height > 80:
        preIm = cv2.resize(preIm, (80, 80), interpolation=cv2.INTER_AREA)
    squareImages.append(preIm)

triangleImages = []
for i in range(len(triangleFiles)):
    # Prior to input, crop the images to take off bottom tag with unnecessary details (orig image is square)
    preIm = cv2.imread(os.path.join(cwd, 'Images\\triangle\\', triangleFiles[i]), 0)
    height, width = preIm.shape
    # if the size of the image is greater than 80 pixels in the height, resize to an 80x80 image:
    if height > 80:
        preIm = cv2.resize(preIm, (80, 80), interpolation=cv2.INTER_AREA)
    triangleImages.append(preIm)

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

# Train, test, split the data
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=40, test_size=0.2, stratify=y)
x_test_length = len(x_test)
y_test_ex = y_test

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
hiddenLayers = (20, 50, 100)
alphaValues = (0.0001, 0.001, 0.01)

for alpha1 in alphaValues:
    for hidLay in hiddenLayers:

        # Provide a start timer for MLP run
        start = timeit.default_timer()
        # Create a MLP Classifier
        clf = MLPClassifier(solver='sgd',       # MLP will converge via Stochastic Gradient Descent
                            alpha=alpha1,       # alpha is convergence rate (low alpha is slow, but won't overshoot solution)
                            hidden_layer_sizes=(hidLay,),        # represents a 6400 - Hidden Layers - 2 MLP
                            random_state=1)
        # Train the model using the training sets
        clf.fit(x_train, y_train)
        # Predict the response for test dataset
        y_pred = clf.predict(x_test)

        # Provide a stop timer for MLP run
        stop = timeit.default_timer()
        print("-" * 80)
        print("Model Results", namestr(x, globals())[0])
        print("-" * 80)
        print("Accuracy of MLP",namestr(x, globals())[0],":", round(metrics.accuracy_score(y_test, y_pred), 3))
        print("-")
        print("Confusion Matrix",namestr(x, globals())[0],":")
        cmx_MLP = confusion_matrix(y_test, y_pred)
        print(cmx_MLP)
        print("-")
        print("Classification Report MLP",namestr(x, globals())[0],":")
        cfrp = classification_report(y_test, y_pred)
        print(cfrp)
        print("-")

        file = open('ModelOutput.txt', 'a+')
        file.write('\nMulti-Layer Perceptron Metrics:\n')
        file.write("Descent Model:\t\tStochastic Gradient\n")
        file.write(f"Alpha:\t\t\t\t{alpha1}\n")
        file.write(f"Hidden Layers:\t\t\t{hidLay}\n")
        file.write("-"*50)
        file.write('\n\nMulti-Layer Perceptron Performance:\n')
        file.write(f"Run Time:\t\t{stop-start} seconds\n")
        file.write(f"Accuracy:\t\t{round(metrics.accuracy_score(y_test, y_pred), 3)}\n")
        file.write("-"*50)
        file.close()

        clf.estimator.
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