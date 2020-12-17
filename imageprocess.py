#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 14 11:24:36 2020

@author: yanzhiwen
"""

import numpy as np
import cv2
import skimage.io as io
from skimage import data, exposure


#calculate image contrast
def contrast(img):   
    m, n = img.shape
    #extend 1 pixel out of the image array
    img_ext = cv2.copyMakeBorder(img,1,1,1,1,cv2.BORDER_REPLICATE) / 1.0
    rows_ext,cols_ext = img_ext.shape
    k = 0.0
    for i in range(1,rows_ext-1):
        for j in range(1,cols_ext-1):
            k += ((img_ext[i,j]-img_ext[i,j+1])**2 + (img_ext[i,j]-img_ext[i,j-1])**2 + 
                    (img_ext[i,j]-img_ext[i+1,j])**2 + (img_ext[i,j]-img_ext[i-1,j])**2)

    cg = k/(4*(m-2)*(n-2)+3*(2*(m-2)+2*(n-2))+2*4) 
    return(cg)


image_path = "/Users/joanna/Downloads/unet-master/data/membrane/train/image/*.png"
image_coll = io.ImageCollection(image_path)
contr_list = []

for i in range(len(image_coll)):
    print("Figure " + str(i) + ":")
    image = np.array(image_coll[i]) / 255 
    #if the image is already expressed in 0-1, then skip this step
    
    cg = contrast(image)
    contr_list.append(cg)
    print(cg)
    print(exposure.is_low_contrast(image)) 
    #if gives false, then the contrast of the image is not good

print("The maxinmun of the contrast is: " + str(max(contr_list)))
print("The min of the contrast is: " + str(min(contr_list)))



#iou function to measure prediction accuracy
#img is ground truth, pic2 is prediction
def iou(img, pic2):
    union=0
    inter=0
    tn=0
    for i in range(0,img.shape[0]):
        for j in range(0,img.shape[1]):
            if img[i,j] != 0 and pic2[i,j] == 0:
                inter = inter + 1
            if img[i,j] == 0 and pic2[i,j] != 0:
                union = union + 1
            if img[i,j] != 0 and pic2[i,j] != 0:
                tn = tn + 1
    accuracy=(tn) / (inter+union+tn)
    return accuracy










