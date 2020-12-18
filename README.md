# BMI3-project
this is the code for Segmentation of neuronal structure in Drosophila ventral nerve cord EM images

this is a command line executable code. Please first run mymodel.py and augmentation.py using tensorflow backend.
to run the actual traning model, Please write as follows：
       $python training.py - r your root path of this folder.

we have keep a folder format as follows
please keep your training data and test data in the same way.

-/root 
       -mymodel.py
       -aygmentation.py
       -training.py
       -/test
              -/test_image
              -/test_label
              -/predictioin
       -/train
              -/train_image
              -/train_label
              -/val_image
              -/val_label
              
