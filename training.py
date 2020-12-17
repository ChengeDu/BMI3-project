#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 15 13:25:14 2020

@author: yanzhiwen
"""
from mymodel import *
from augmentation import *
import argparse

#allow command line input
ap = argparse.ArgumentParser()
ap.add_argument('-r', '--rootpath',help='root path for all data')
args = vars(ap.parse_args())
rootpath=args['rootpath']


#define a data augmentation generator
aug = ImageDataGenerator( 
      rotation_range = 0.05, 
      zoom_range = 0.05, 
      width_shift_range = 0.05, 
      height_shift_range = 0.05, 
      #shear_range = 0.05, # 
 	  horizontal_flip = True, 
     #fill_mode = "reflect" #fill the pixel after rotation or shift
 )



#seperate the original dataset for training & validation
train_image_path = rootpath + "/train/train_image/*.png"
train_label_path = rootpath + "/train/train_label/*.png"
train_image_coll = io.ImageCollection(train_image_path)
train_label_coll = io.ImageCollection(train_label_path)

train_gen = train_generator(train_image_coll, train_label_coll, aug) 

val_image_path = rootpath + "/train/val_image/*.png"
val_label_path = rootpath + "/train/val_label/*.png"
val_image_coll = io.ImageCollection(val_image_path)
val_label_coll = io.ImageCollection(val_label_path)

validate_gen = train_generator(val_image_coll, val_label_coll, aug) 



#define a model
model = simple_unet()

model_checkpoint = ModelCheckpoint('unet.hdf5', monitor='loss',verbose=1, save_best_only=True)
history = model.fit_generator(train_gen,
                    steps_per_epoch=40, #n_imgs//batch_size
                    validation_data=validate_gen,
                    validation_steps=20,
                    epochs=1,
                    callbacks=[model_checkpoint])

test_path = rootpath + "/test/test_image/*.png"
test_coll = io.ImageCollection(test_path)
test_gen = test_generator(test_coll)

results = model.predict_generator(test_gen,len(test_coll),verbose=1)
saveResult(rootpath + "/test/prediction",results)



#plot----------
#print("history.histroy keys：", history.history.keys())
acc=history.history['accuracy']
val_acc=history.history['val_accuracy']
loss=history.history['loss']
val_loss=history.history['val_loss']

epochs=range(len(acc)) # Get number of epochs


# 画accuracy曲线
plt.plot(epochs, acc, 'r')
plt.plot(epochs, val_acc, 'b')
plt.title('Training and validation accuracy')
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend(["Accuracy", "Validation Accuracy"])

plt.figure()


# 画loss曲线
plt.plot(epochs, loss, 'r')
plt.plot(epochs, val_loss, 'b')
plt.title('Training and validation loss')
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend(["Loss", "Validation Loss"])

plt.figure()
