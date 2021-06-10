# -*- coding: utf-8 -*-
"""
Created on Thu Jun 10 17:17:55 2021

@author: Md Fakrul Islam
"""

from glob import glob  
import os                                                         
import cv2 
import re
from PIL import Image
#from resizeimage import resizeimage
from os.path import join
from glob import glob

def pprocess_image(image_path, count, tag):
    path, filename = os.path.split(image_path)   
    filename_w_ext = os.path.basename(image_path)
    filename, file_extension = os.path.splitext(filename_w_ext)    
    _newfilename=path+"/"+tag+"_"+str(count)+file_extension.upper()
    print(_newfilename)
    main_img = cv2.imread(image_path)
    resized_image = cv2.resize(main_img, (IMG_WIDTH, IMG_HEIGHT), interpolation = cv2.INTER_AREA)    
    cv2.imwrite(_newfilename, resized_image)
    os.remove(image_path) 
                
print('Function pprocess_image load')

def getListSize(itel_list):
    count = 0
    for img_path in itel_list:  
        count = count + 1
    return count

print ('Function getsize loaded')

IMG_HEIGHT = 224
IMG_WIDTH = 224
IMAGE_SIZE = [IMG_HEIGHT, IMG_WIDTH]
#Images Location
IMAGE_LOCATION='/tmp/LabelledRice/Labelled/LeafBlast'
NAME = 'RICE'
DISEASES_NAME = 'LeafBlast'
FINAL_TAG = NAME+'_'+DISEASES_NAME
print('Image Dimension Loaded')
print(FINAL_TAG)


IMAGE_Files = []
for ext in ('*.JPEG', '*.JPG', '*.jpg', '*.jpeg', '*.png'):
   IMAGE_Files.extend(glob(join(IMAGE_LOCATION, ext)))
   
   
print('Total IMAGE_Files :',getListSize(IMAGE_Files))

count=0
print('DISEASES_LOCATION_IMAGES:',IMAGE_LOCATION)
# Load all images
for IMAGE_PATH in IMAGE_Files:
    count = count + 1 
    pprocess_image(IMAGE_PATH, count, FINAL_TAG)
print("Total Images: " +str(count))