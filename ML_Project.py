'''
Machine Learning I - Project
Authors:    Sam Cohen
            Luis Alberto
            Rich Gude
'''
# OBJECTIVE I:
# Create a collection of 40x40 pixel images of squares, circles, rectangles, and triangles with noise:

from matplotlib import pyplot as plt
import matplotlib.patches as ptch
import random as rd
import matplotlib.lines as lines           # For triangle creation

'''
For all of the shapes below will be generated in a 10x10 plot with the center of the plot at (5,5).  Any random
selection of variables is meant to move the center of the shape within the plot (excluding potentially cutting it off).
'''
# Create a figure plot
fig, ax = plt.subplots()

# Create a certain amount of circles with random variables for origin, line thickness, and radius
numCirc = 10
for i in range(numCirc):
    # Set random values for variables:
    centerX = rd.uniform(4, 6)
    centerY = rd.uniform(4, 6)
    radius = rd.uniform(1, 5)
    lineWidth = rd.uniform(2, 20)

    # Draw a circle (CirclePolygon is used to give less smooth sides)
    p = ptch.CirclePolygon((centerX, centerY), radius, fill=False, edgecolor='black', linewidth=lineWidth)
    ax.add_patch(p)

    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    plt.axis('off')
    plt.savefig(fname=('Circle' + str(i) + '.png'), format='png')
    plt.cla()

# Create a certain amount of squares with random variables for origin, line thickness, width, height, and shape angle
numRect = 10
for i in range(numRect):
    # Set random values for variables:
    centerX = rd.uniform(4, 6)
    centerY = rd.uniform(4, 6)
    width = rd.uniform(1, 5)
    height = rd.uniform(1, 5)
    theta1 = rd.uniform(0, 90)
    lineWidth = rd.uniform(2, 20)

    # Draw a Rectangle
    p = ptch.Rectangle((centerX, centerY), width, height, theta1, fill=False, edgecolor='black', linewidth=lineWidth)
    ax.add_patch(p)

    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    plt.axis('off')
    plt.savefig(fname=('Rectangle' + str(i) + '.png'), format='png')
    plt.cla()

# Create a certain amount of triangles with random variables for origin, line thickness, and radius, and triangle angles
numTriangle = 10
for i in range(numTriangle):
    # Make three random points in the 10x10 space
    p1X, p1Y = rd.uniform(1, 9), rd.uniform(1, 9)
    p2X, p2Y = rd.uniform(1, 9), rd.uniform(1, 9)
    p3X, p3Y = rd.uniform(1, 9), rd.uniform(1, 9)
    lineWidth = rd.uniform(2, 20)

    # Add lines to axis by making three lines between all three points
    ax.add_line(lines.Line2D([p1X, p2X], [p1Y, p2Y], color='black', linewidth=lineWidth))
    ax.add_line(lines.Line2D([p1X, p3X], [p1Y, p3Y], color='black', linewidth=lineWidth))
    ax.add_line(lines.Line2D([p2X, p3X], [p2Y, p3Y], color='black', linewidth=lineWidth))

    '''
    # May not need to use wedge class
    # Set random values for variables:
    centerX = rd.uniform(4, 6)
    centerY = rd.uniform(4, 6)
    radius = rd.uniform(1, 5)
    lineWidth = rd.uniform(2, 20)
    thetaAdd = rd.uniform(0,180)
    theta1 = thetaAdd + rd.uniform(0, 90)
    theta2 = thetaAdd + rd.uniform(90,180)

    # Draw a circle
    p = ptch.Wedge((centerX, centerY), radius, theta1, theta2, fill=False, edgecolor='black', linewidth=lineWidth)
    ax.add_patch(p)
    '''

    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    plt.axis('off')
    plt.savefig(fname=('Triangle' + str(i) + '.png'), format='png')
    plt.cla()