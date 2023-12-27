
## Getting all required libraries.

import cv2
import dlib
import numpy as np
import os

predictor=dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')  #type:ignore
detector=dlib.get_frontal_face_detector() #type:ignore

camera=cv2.VideoCapture(0)

results=[]
frames=[]

mood=input('What is your emotion ?')

while True:
    ret,frame=camera.read()
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces=detector(gray)
    

