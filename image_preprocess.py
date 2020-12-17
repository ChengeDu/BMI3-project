#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 15 14:22:12 2020

@author: yanzhiwen
"""


import cv2
import skimage.io as io
import os
import skimage.transform as trans
import numpy as np


# image preparation:
#since there're 30 images(tiff) in one tif file, firstly save each single tiff file

# tiff-->png transformation
def ImageTrans(image_path, output_path, num_image):
    for i in range(num_image):
        img = cv2.imread(os.path.join(image_path,str(i)+'.tiff'))
        cv2.imwrite(os.path.join(output_path,str(i)+'.png'), img)
        #im = io.imread(os.path.join(output_path,str(i)+'.png'),as_gray = True)
        #print(im)
        

ImageTrans("/data/membrane",
           "/data/membrane",30)




            