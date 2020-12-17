#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 10 10:10:43 2020

@author: joanna
"""
import Augmentor
import keras
import numpy as np
from PIL import Image
from keras.preprocessing.image import ImageDataGenerator
import os
import skimage.io as io
import matplotlib.pyplot as plt
import matplotlib.image as mpimg 
import skimage.transform as trans



def train_generator(image_coll,label_coll,aug):  
    while True:
        image_array = []
        label_array = []
        for i in range(len(image_coll)):
            #read image
            tmp_im_array = image_coll[i]
            tmp_im_array = trans.resize(tmp_im_array,(256,256))
            #tmp_im_array = im / 255.0  #since the image is already between 0~1, there's no need to divide 255

            #transform the array to 4 dimentions
            tmp_im_array = np.reshape(tmp_im_array,tmp_im_array.shape+(1,)) 
            tmp_im_array = np.reshape(tmp_im_array,(1,)+tmp_im_array.shape)
            
          
            #read image
            tmp_lb_array = label_coll[i]
            tmp_lb_array = trans.resize(tmp_lb_array,(256,256))
            #tmp_lb_array[tmp_lb_array > 0.5] = 1
            #tmp_lb_array[tmp_lb_array <= 0.5] = 0  #this step should be added after the Augmentation
            
            #transform the array to 4 dimentions
            tmp_lb_array = np.reshape(tmp_lb_array,tmp_lb_array.shape+(1,)) 
            tmp_lb_array = np.reshape(tmp_lb_array,(1,)+tmp_lb_array.shape)
           
            
            if aug!=None: 
                #Image Augmentation
                #image_array, label_array = next(aug.flow(image_array, label_array, batch_size = 15))
                 new_array = tmp_im_array
                 new_array = np.concatenate((new_array,tmp_lb_array),axis=3) # combine image & label to do augmentation together
                 new_array = next(aug.flow(new_array,batch_size = 32))
                 #seperate image & label
                 image_array = new_array[:,:,:,0] 
                 label_array = new_array[:,:,:,1]
                 #compress and make the array to 0 & 1 distribution
                 label_array[label_array > 0.5] = 1
                 label_array[label_array <= 0.5] = 0
                 #transform the array to 4 dimentions
                 image_array = np.reshape(image_array,image_array.shape+(1,)) 
                 label_array = np.reshape(label_array,label_array.shape+(1,)) 
                
            yield(image_array, label_array)



def test_generator(test_coll): 
    for i in range(len(test_coll)):
        #read image
        tmp_im_array = test_coll[i]
        tmp_im_array = trans.resize(tmp_im_array,(256,256))
        tmp_im_array = np.reshape(tmp_im_array,tmp_im_array.shape+(1,)) 
        tmp_im_array = np.reshape(tmp_im_array,(1,)+tmp_im_array.shape)
        
        yield(tmp_im_array) 


def saveResult(save_path,npyfile):
    for i,item in enumerate(npyfile): #get every test result
        im_np = item[:,:,0]
        #print(im_np)
        #im_np[im_np > 0.65] = 1
        #im_np[im_np <= 0.65] = 0
        io.imsave(os.path.join(save_path,"%d_predict.png"%i),im_np)

#saveResult("data/membrane/test4/predict",results)







