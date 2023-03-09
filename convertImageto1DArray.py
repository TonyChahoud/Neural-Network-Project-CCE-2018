# -*- coding: utf-8 -*-
"""
Created on Tue Jan 15 15:32:16 2019

@author: edgar.youssef
"""



import numpy as np
from PIL import Image

img = Image.open('sim1.jpg').convert('RGBA')
arr = np.array(img)

# record the original shape
shape = arr.shape

# make a 1-dimensional view of arr
flat_arr = arr.ravel()
print(len(flat_arr))
# convert it to a matrix
vector = np.matrix(flat_arr)

# do something to the vector
vector[:,::10] = 128

# reform a numpy array of the original shape
arr2 = np.asarray(vector).reshape(shape)

# make a PIL image
img2 = Image.fromarray(arr2, 'RGBA')


