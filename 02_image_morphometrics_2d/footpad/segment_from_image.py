# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2022-2025 Pranav Vyas
"""Hole areas and their distances from the centre of a foot-pad ossicle.

Code release for Vyas et al., "Cellular construction of topologically complex
biomineral lattices in holothurians".

Figure panels : Fig. 2F (foot-pad series of hole area vs distance; data1.csv is read by
                01_microct_morphometrics/fields/holes/dist_area_plot.py)
What it does  : Thresholds and cleans a binary image of the ossicle, labels its holes
                (the largest region is dropped) and writes each hole's area (µm²) and the
                distance (µm) of its centroid from a hand-picked centre (center0), using
                the foot-pad 2 scale (20 µm = 162.333 px).
Inputs        : test2.png and "Image 1-2.bmp" in the current directory
Outputs       : data1.csv, labelled_image_3.png, overlaid_image_3.png and
                scatter_plot_3.png in the current directory
Environment   : environment-analysis.yml (Python 3.7)
Run           : python segment_from_image.py   (from the folder that holds the inputs)
"""


import cv2
import numpy as np
import matplotlib.pyplot as plt
import csv
import os
import sys
import glob
import pandas as pd
import time
import math
from skimage import measure
from skimage import morphology
from skimage import segmentation
from skimage import filters
from skimage import util
from skimage import color
from skimage import io
from skimage import img_as_ubyte
from skimage import img_as_uint
from skimage import img_as_float
from skimage import exposure
from skimage import feature

#function to segment the image and label the blobs
def get_labeled_image(image, bin_image):
    #convert the image to binary
    thresh = filters.threshold_otsu(bin_image)
    binary = bin_image > thresh

    #remove smaller objects
    binary = morphology.remove_small_objects(binary, 200)
    
    #smoothen boundaries of binary regions, fill holes and identify identify objects under a certain size
    #remove objects larger than 1000 pixels squared
    binary = morphology.binary_closing(binary)

    #remove small holes
    binary = morphology.remove_small_holes(binary, 750)

    #create a labelled image
    label_image = measure.label(binary)

    #remove n largest regions
    props = measure.regionprops(label_image)
    areas = []
    for prop in props:
        areas.append(prop.area)
    areas.sort(reverse=True)
    areas = areas[0:1]
    for prop in props:
        if prop.area in areas:
            label_image[label_image == prop.label] = 0

    fig, ax = plt.subplots(figsize=(20, 20))
    ax.imshow(label_image, cmap=plt.cm.gray)
    ax.axis('off')
    plt.savefig('labelled_image_3.png')

    #overlay labelled image on original image and save overlaid image
    fig1, ax1 = plt.subplots(figsize=(20, 20))
    ax1.imshow(image, cmap='gray')
    ax1.imshow(label_image, alpha=0.5)
    plt.axis('off')
    plt.savefig('overlaid_image_3.png')
    
    return label_image

#find distances from center of the structure to the center of all the holes
def find_area_distances(label_image):
    #scale conversion factor
    #1px = 50/181.583 um
    # scale = 50/181.583
    scale = 20/162.333  

    #input the center of the image
    # center0 = [849.500, 812.333]
    center0 = [786,724]

    #find the center of all the holes
    props = measure.regionprops(label_image)
    centers = []
    for prop in props:
        centers.append(prop.centroid)

    #find the distance between the center of the structure and the center of all the holes
    distances = []
    for center in centers:
        distance = math.sqrt((center0[0] - center[0])**2 + (center0[1] - center[1])**2)
        distances.append(distance)

    #convert distances to um
    for i in range(len(distances)):
        distances[i] = distances[i] * scale

    #find the area of all the holes
    areas = []
    for prop in props:
        areas.append(prop.area)

    #convert areas to um^2
    for i in range(len(areas)):
        areas[i] = areas[i] * scale**2


    return areas, distances

#main function
def main():
    #read the original image
    image = cv2.imread('test2.png',0)

    #read in the binary image
    binary_image = cv2.imread('Image 1-2.bmp',0)

    #segment the image
    label_image = get_labeled_image(image, binary_image)

    #find the areas and distances of all the holes
    areas, distances = find_area_distances(label_image)
    
    #create a csv file to store the data
    with open('data1.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Area (um2)", "Distance(um)"])
        for i in range(len(areas)):
            writer.writerow([areas[i], distances[i]])

    #plot scatter plot of areas with distances
    fig2, ax2 = plt.subplots(figsize=(20, 20))
    ax2.scatter(distances, areas)
    plt.xlabel('Distance (um)')
    plt.ylabel('Area (um2)')
    plt.savefig('scatter_plot_3.png')
    
#call main function
if __name__ == "__main__":
    main()


