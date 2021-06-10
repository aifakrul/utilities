# -*- coding: utf-8 -*-
"""
Created on Thu Jun 10 17:21:11 2021

@author: Md Fakrul Islam
"""
#https://pypi.org/project/split-folders/
#!pip install split-folders

import splitfolders  # or import split_folders

#Train Valid Test
splitfolders.ratio("input_folder", output="output", seed=1337, ratio=(.8, .1, .1), group_prefix=None) # default values

#Create Balanced Set
splitfolders.fixed("/tmp/LabelledRice/Labelled", output="/tmp/output", seed=1337, fixed=(450, 70), oversample=False, group_prefix=None) # default values