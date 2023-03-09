# -*- coding: utf-8 -*-
"""
Created on Tue Jan 15 15:21:58 2019

@author: edgar.youssef
"""

import cv2 as cv
import numpy as np

img1=cv.imread("im1.jpg",0)
img2=cv.imread("im2.jpg",0)
img3=cv.imread("im3.jpg",0)

img1 = cv.resize(img1,(16, 32), interpolation = cv.INTER_CUBIC)

cv.imwrite('sim1.jpg',img1)
img2 = cv.resize(img2,(16, 32), interpolation = cv.INTER_CUBIC)

cv.imwrite('sim2.jpg',img2)
img3 = cv.resize(img3,(16, 32), interpolation = cv.INTER_CUBIC)

cv.imwrite('sim3.jpg',img3)