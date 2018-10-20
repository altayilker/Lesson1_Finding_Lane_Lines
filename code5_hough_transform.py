# -*- coding: utf-8 -*-
"""
Created on Thu Sep 27 10:25:36 2018

@author: u19l65
"""

# Do all the relevant imports
# Do relevant imports
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2

# Read in and grayscale the image
image = mpimg.imread('exit-ramp.jpg')
gray = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)

# Define a kernel size and apply Gaussian smoothing
kernel_size = 5
blur_gray = cv2.GaussianBlur(gray,(kernel_size, kernel_size),0)

# Define our parameters for Canny and apply
low_threshold = 50
high_threshold = 150
masked_edges = cv2.Canny(blur_gray, low_threshold, high_threshold)
fig = plt.figure()
ax = fig.add_subplot(111)
plt.imshow(masked_edges,cmap='Greys_r')
ax.set(title='Masked Edges')



# Define the Hough transform parameters
# Make a blank the same size as our image to draw on
rho = 2
theta = np.pi/180
threshold = 15
min_line_length = 40
max_line_gap = 20
line_image = np.copy(image)*0 #creating a blank to draw lines on

# Run Hough on edge detected image
lines = cv2.HoughLinesP(masked_edges, rho, theta, threshold, np.array([]),
                            min_line_length, max_line_gap)


# Iterate over the output "lines" and draw lines on the blank
for line in lines:
    for x1,y1,x2,y2 in line:
        cv2.line(line_image,(x1,y1),(x2,y2),(255,0,0),10)
        
fig = plt.figure()
ax = fig.add_subplot(111)
plt.imshow(line_image)
ax.set(title='line_image')

# Create a "color" binary image to combine with line image
color_edges = np.dstack((masked_edges, masked_edges, masked_edges)) 
fig = plt.figure()
ax = fig.add_subplot(111)
plt.imshow(color_edges)
ax.set(title='color_edges')



# Draw the lines on the edge image
combo = cv2.addWeighted(color_edges, 0.8, line_image, 1, 0)
fig = plt.figure()
ax = fig.add_subplot(111)
plt.imshow(combo)
ax.set(title='combo')

# Draw the lines on the real image
combo2 = cv2.addWeighted(image, 0.8, line_image, 1, 0)
fig = plt.figure()
ax = fig.add_subplot(111)
plt.imshow(combo2)
ax.set(title='combo2')


