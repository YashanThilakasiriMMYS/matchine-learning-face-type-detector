# -*- coding: utf-8 -*-
"""
Created on Sat Oct 26 10:58:00 2019

@author: yasha
"""

import cv2
import dlib

camera=cv2.VideoCapture (0)

#camera is a video object, 0-default camera, 1,2-webcams/usb cams connected
#file_path can be also given, ip addr for wifi cameras
while(True):
    ret,img=camera.read()
    #capturing 1 frame from the video source in 'camera object' and saves it in 'img'
    #ret is boolean alue, 1-camer is available, 0 camera is non available

    #img[50:100,50:100]=[255,0,0]

    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    #COLOR_BGR2GRAY is a color conversion method. this take the img object and do the color conversion.
    #Also there are more color convertion methods like BGR2RGB
    cv2.imshow('IMG', img)
    cv2.imshow('GRAY', gray)
    cv2.waitKey(1)
   
    
