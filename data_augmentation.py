# -*- coding: utf-8 -*-
"""
Created on Thu Jun 10 16:52:50 2021

@author: Md Fakrul Islam
"""

from keras.preprocessing.image import ImageDataGenerator
from skimage import io


datagen = ImageDataGenerator(
        rotation_range=45,     #Random rotation between 0 and 45
        width_shift_range=0.2,   #% shift
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='constant', cval=125)    #Also try nearest, constant, reflect, wrap


i = 0
for batch in datagen.flow_from_directory(directory='D:/rice/output/train/', 
                                         batch_size=16,  
                                         target_size=(224, 224),
                                         color_mode="rgb",
                                         save_to_dir='D:/rice/output/aug/', 
                                         save_prefix='LB_aug', 
                                         save_format='JPG'):
    i += 1
    if i > 6:
        break 
