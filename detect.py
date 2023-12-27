
#Importing required libraries

import cv2
import dlib
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

predictor=dlib.shape_predictor('Emotion-Recognition/shape_predictor_68_face_landmarks.dat')  #type:ignore
detector=dlib.get_frontal_face_detector()    #type:ignore
data=np.load('moods.npy')
