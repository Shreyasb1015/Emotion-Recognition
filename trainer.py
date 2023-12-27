
## Getting all required libraries.

import cv2
import dlib
import numpy as np
import os

predictor=dlib.shape_predictor('Emotion-Recognition/shape_predictor_68_face_landmarks.dat')  #type:ignore
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
    
    if cv2.waitKey(1)==ord('a'):
        landmarks_arr=np.array([[point.x-face.left(),point.y-face.top()] for point in landmarks_parts]) #type:ignore
        frames.append(landmarks_arr.flatten())
        results.append([mood])
    
    if cv2.waitKey(1) == 27:
        break

data=np.hstack((results,frames))

if os.path.exists('Emotion-Recognition/moods.npy'):
    file=np.load('moods.npy')
    data=np.vstack((file,data))
    
np.save('Emotion-Recognition/moods.npy',data)

cv2.destroyAllWindows()
camera.release()