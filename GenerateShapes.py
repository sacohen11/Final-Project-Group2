'''
# OBJECTIVE:
# Create a collection of jpeg images of squares, circles, rectangles, and triangles with noise:

The purpose of this code is to supplement the user-generated shapes in Adobe Photoshop to better equalize the total
number of each shape before the primary code ('ML_Project.py') provides additional data augmentation.

The Total Number of Each Photoshop Image Shape:
Circle      -   460
Rectangle   -   191
Square      -   172
Triangle    -    81

All of the shapes below will be generated in a 10x10 plot with the center of the plot at (5,5).  Any random
selection of variables is meant to move the center of the shape within the plot (including potentially cutting it off).

The title of each image file of Python-generated shapes will contain a 'X' to distinguish those from adobe Photoshop
images
'''


# Import libraries for generating shapes
from matplotlib import pyplot as plt
import matplotlib.patches as ptch
import random as rd
import matplotlib.lines as lines           # For triangle creation
from math import sin, cos, radians         # For sin and cos calcs for better centering squares and rectangles

# Create a figure plot
fig, ax = plt.subplots()

# Create a certain amount of circles with random variables for origin, line thickness, and radius
numCirc = 240
for i in range(numCirc):
    # Set random values for variables:
    centerX = rd.uniform(4, 6)
    centerY = rd.uniform(4, 6)
    radius = rd.uniform(1, 5)
    lineWidth = rd.uniform(1, 18)

    # Draw a circle (CirclePolygon is used to give less smooth sides)
    p = ptch.CirclePolygon((centerX, centerY), radius, fill=False, edgecolor='black', linewidth=lineWidth)
    ax.add_patch(p)

    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    plt.axis('off')
    plt.savefig(fname=('circleXX' + str(i) + '.jpg'), format='jpg')
    plt.cla()

# Create a certain amount of squares with random variables for origin, line thickness, width, height, and shape angle
numRect = 240
for i in range(numRect):
    # Set random values for variables:
    # theta is the angle from which the lower left angle of rectangle is tilted from the horizon
    theta1 = rd.uniform(0, 90)

    height = rd.uniform(3, 5)
    width = height + rd.uniform(1, 3)   # width will always be greater than height to establish a rectangular shape

    # Based on the theta and height/width values, set the "center" (lower left) coordinates to avoid cutting off
    #   theta =  0 -> x(1, 9 - width),  y(1, 9 - height)
    #   theta = 90 -> x(1 + height, 9), y(1, 9 - width)
    centerX = rd.uniform(1 + sin(radians(theta1))*height, 9 - cos(radians(theta1)) * width)
    centerY = rd.uniform(1, 9 - cos(radians(theta1)) * height - sin(radians(theta1)) * width)
    lineWidth = rd.uniform(1, 18)

    # Draw a Rectangle
    p = ptch.Rectangle((centerX, centerY), width, height, theta1, fill=False, edgecolor='black', linewidth=lineWidth)
    ax.add_patch(p)

    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    plt.axis('off')
    plt.savefig(fname=('rectangleXX' + str(i) + '.jpg'), format='jpg')
    plt.cla()

# Create a certain amount of squares with random variables for origin, line thickness, width, height, and shape angle
numSqur = 240
for i in range(numSqur):
    # Set random values for variables:
    # theta is the degree angle from which the lower left angle of rectangle is tilted from the horizon
    theta1 = rd.uniform(0, 90)

    height = rd.uniform(3, 6)
    width = height + rd.gauss(0, 0.5/3)         # width will usually ~equal height will some noise/error

    # Based on the theta and height/width values, set the "center" (lower left) coordinates to avoid cutting off
    #   theta =  0 -> (1, 9 - width)
    #   theta = 90 -> (1 + height, 9)
    centerX = rd.uniform(1 + sin(radians(theta1))*height, 9 - cos(radians(theta1)) * width)
    centerY = rd.uniform(1, 9 - cos(radians(theta1)) * height - sin(radians(theta1)) * width)

    lineWidth = rd.uniform(1, 18)

    # Draw a Rectangle
    p = ptch.Rectangle((centerX, centerY), width, height, theta1, fill=False, edgecolor='black', linewidth=lineWidth)
    ax.add_patch(p)

    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    plt.axis('off')
    plt.savefig(fname=('squareXX' + str(i) + '.jpg'), format='jpg')
    plt.cla()

# Create a certain amount of triangles with random variables for origin, line thickness, and radius, and triangle angles
numTriangle = 240
for i in range(numTriangle):
    # Make three random points in the 10x10 space
    p1X, p1Y = rd.uniform(1, 9), rd.uniform(1, 9)
    p2X, p2Y = rd.uniform(1, 9), rd.uniform(1, 9)
    p3X, p3Y = rd.uniform(1, 9), rd.uniform(1, 9)
    lineWidth = rd.uniform(1, 18)

    # Add lines to axis by making three lines between all three points
    ax.add_line(lines.Line2D([p1X, p2X], [p1Y, p2Y], color='black', linewidth=lineWidth))
    ax.add_line(lines.Line2D([p1X, p3X], [p1Y, p3Y], color='black', linewidth=lineWidth))
    ax.add_line(lines.Line2D([p2X, p3X], [p2Y, p3Y], color='black', linewidth=lineWidth))

    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    plt.axis('off')
    plt.savefig(fname=('triangleXX' + str(i) + '.jpg'), format='jpg')
    plt.cla()
