
#Importing required libraries

import cv2
import dlib
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

predictor=dlib.shape_predictor('Emotion-Recognition/shape_predictor_68_face_landmarks.dat')  #type:ignore
detector=dlib.get_frontal_face_detector()    #type:ignore
data=np.load('Emotion-Recognition/moods.npy')

X=data[:,1:].astype(int)
y=data[:,0]

knn=KNeighborsClassifier()
knn.fit(X,y)

camera=cv2.VideoCapture(0)

while True:
    ret,frame=camera.read()
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces=detector(gray)
    for face in faces:
        landmark=predictor(gray,face)
        landmarks= np.array([[point.x - face.left(), point.y - face.top()] for point in landmark.parts()]).flatten()  #type:ignore
        prediction=knn.predict([landmarks])  #type:ignore
        cv2.rectangle(frame, (face.left(), face.top()), (face.right(), face.bottom()), (255, 0, 0), 2)  
        cv2.putText(frame, prediction[0], (face.left(), face.bottom() + 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 2)
        
    cv2.imshow('emotion-recognition',frame)
    
    if cv2.waitKey(1)==27:
        break

cv2.destroyAllWindows()
camera.release()