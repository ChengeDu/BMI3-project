 #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  7 16:04:59 2020

@author: yanzhiwen
"""
import numpy as np 
import os
import skimage.io as io
import skimage.transform as trans
import numpy as np
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as keras
    
"""This model is based on:
https://github.com/jocicmarko/ultrasound-nerve-segmentation
"""


def deep_unet(pretrained_weights = None,input_size = (256,256,1),N = 6):
    
    inputs = Input(input_size)
    conv1 = Conv2D(2**N, (3, 3), activation='relu', padding='same',kernel_initializer = 'he_uniform')(inputs)
    conv1 = Conv2D(2**N, (3, 3), activation='relu', padding='same',kernel_initializer = 'he_uniform')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(2**(N + 1), (3, 3), activation='relu', padding='same',kernel_initializer = 'he_uniform')(pool1)
    conv2 = Conv2D(2**(N + 1), (3, 3), activation='relu', padding='same',kernel_initializer = 'he_uniform')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(2**(N + 2), (3, 3), activation='relu', padding='same',kernel_initializer = 'he_uniform')(pool2)
    conv3 = Conv2D(2**(N + 2), (3, 3), activation='relu', padding='same',kernel_initializer = 'he_uniform')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(2**(N + 3), (3, 3), activation='relu', padding='same',kernel_initializer = 'he_uniform')(pool3)
    conv4 = Conv2D(2**(N + 3), (3, 3), activation='relu', padding='same',kernel_initializer = 'he_uniform')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Conv2D(2**(N + 4), (3, 3), activation='relu', padding='same',kernel_initializer = 'he_uniform')(pool4)
    conv5 = Conv2D(2**(N + 4), (3, 3), activation='relu', padding='same',kernel_initializer = 'he_uniform')(conv5)

    up6 = concatenate([Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same',kernel_initializer = 'he_uniform')(conv5), conv4], axis=3)
    conv6 = Conv2D(2**(N + 3), (3, 3), activation='relu', padding='same',kernel_initializer = 'he_uniform')(up6)
    conv6 = Conv2D(2**(N + 3), (3, 3), activation='relu', padding='same',kernel_initializer = 'he_uniform')(conv6)

    up7 = concatenate([Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same',kernel_initializer = 'he_uniform')(conv6), conv3], axis=3)
    conv7 = Conv2D(2**(N + 2), (3, 3), activation='relu', padding='same',kernel_initializer = 'he_uniform')(up7)
    conv7 = Conv2D(2**(N + 2), (3, 3), activation='relu', padding='same',kernel_initializer = 'he_uniform')(conv7)

    up8 = concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same',kernel_initializer = 'he_uniform')(conv7), conv2], axis=3)
    conv8 = Conv2D(2**(N + 1), (3, 3), activation='relu', padding='same',kernel_initializer = 'he_uniform')(up8)
    conv8 = Conv2D(2**(N + 1), (3, 3), activation='relu', padding='same',kernel_initializer = 'he_uniform')(conv8)

    up9 = concatenate([Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same',kernel_initializer = 'he_uniform')(conv8), conv1], axis=3)
    conv9 = Conv2D(2**N, (3, 3), activation='relu', padding='same',kernel_initializer = 'he_uniform')(up9)
    conv9 = Conv2D(2**N, (3, 3), activation='relu', padding='same',kernel_initializer = 'he_uniform')(conv9)

    conv10 = Conv2D(1, (1, 1), activation='sigmoid')(conv9)

    model = Model(inputs=[inputs], outputs=[conv10])

    model.compile(optimizer = Adam(lr = 1e-3), loss = 'binary_crossentropy', metrics = ['accuracy'])

    return model




def simple_unet( pretrained_weights = None,input_size = (256,256,1),N = 4):

    inputs = Input(input_size)
    
    conv1 = Conv2D(2**N, (3, 3), activation='relu', padding='same',kernel_initializer = 'he_uniform')(inputs)
    conv1 = Conv2D(2**N, (3, 3), activation='relu', padding='same',kernel_initializer = 'he_uniform')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)


    conv2 = Conv2D(2**(N + 1), (3, 3), activation='relu', padding='same',kernel_initializer = 'he_uniform')(pool1)
    conv2 = Conv2D(2**(N + 1), (3, 3), activation='relu', padding='same',kernel_initializer = 'he_uniform')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(2**(N + 2), (3, 3), activation='relu', padding='same',kernel_initializer = 'he_uniform')(pool2)
    conv3 = Conv2D(2**(N + 2), (3, 3), activation='relu', padding='same',kernel_initializer = 'he_uniform')(conv3)
    drop3 = Dropout(0.5)(conv3)

    up1 = concatenate(
        [Conv2D(2**(N+1), 2, activation = 'relu', padding = 'same',kernel_initializer = 'he_uniform')
         (UpSampling2D(size = (2,2))(drop3)),conv2],axis=3)
    conv4 = Conv2D(2**(N + 1), (3, 3), activation='relu', padding='same',kernel_initializer = 'he_uniform')(up1)
    conv4 = Conv2D(2**(N + 1), (3, 3), activation='relu', padding='same',kernel_initializer = 'he_uniform')(conv4)


    up2 = concatenate(
        [Conv2D(2**(N), 2, activation = 'relu', padding = 'same',kernel_initializer = 'he_uniform')
         (UpSampling2D(size = (2,2))(conv4)),conv1],axis=3)
    conv5 = Conv2D(2**(N), (3, 3), activation='relu', padding='same',kernel_initializer = 'he_uniform')(up2)
    conv5 = Conv2D(2**(N), (3, 3), activation='relu', padding='same',kernel_initializer = 'he_uniform')(conv5)

    conv6 = Conv2D(1, (1, 1), activation='sigmoid')(conv5)

    model = Model(inputs=[inputs], outputs=[conv6])
    
# =============================================================================
#     def dice_coef(y_true, y_pred, smooth=1):
#         intersection = K.sum(y_true * y_pred, axis=[1,2,3])
#         union = K.sum(y_true, axis=[1,2,3]) + K.sum(y_pred, axis=[1,2,3])
#         return K.mean( (2. * intersection + smooth) / (union + smooth), axis=0)
#     def dice_coef_loss(y_true, y_pred):
#         return 1 - dice_coef(y_true, y_pred, smooth=1)
#      
#     #loss = dice_coef_loss
# =============================================================================
    
    model.compile(optimizer = Adam(lr = 1e-3), loss = 'binary_crossentropy', metrics = ['accuracy'])

    
    if(pretrained_weights):
    	model.load_weights(pretrained_weights)
        
    #model.summary()

    return model