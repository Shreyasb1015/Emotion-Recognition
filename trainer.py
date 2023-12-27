
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
    
    for face in faces:
        landmarks=predictor(gray,face)
        landmarks_parts=landmarks.parts()
        for landmark in landmarks_parts:
            cv2.circle(frame,(landmark.x,landmark.y),2,(255,0,0),3)
    
    cv2.imshow('Frame',frame)
    
    if cv2.waitKey(1) == 27:
        break
