# -*- coding: utf-8 -*-
"""
Created on Tue Jan 15 15:34:30 2019

@author: edgar.youssef
"""

import numpy as np
from scipy import misc
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

face = misc.imread('im1.jpg');
f=misc.face(gray=True)
[width1,height1]=[f.shape[0],f.shape[1]]
f2=f.reshape(width1*height1);
print(f2)
f2.show()