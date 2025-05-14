import cv2 
import numpy as np
from vectorization import delaunay
import matplotlib.pyplot as plt
import constants

SAMPLING = constants.PROBABILISTIC
OUT_DIR = './src/data/capture-delaunay-jenny/'
IMG_PATH = OUT_DIR + 'capture-delaunay.jpg'

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

img_bgr = cv2.imread(IMG_PATH)
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)    

gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray, 1.3, 5)
mosaic = delaunay(img_rgb, sampling_method=SAMPLING, faces=faces)


for (x,y,w,h) in faces:
    cv2.rectangle(img_rgb,(x,y),(x+w,y+h),(255,255,0),2) 
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = img_rgb[y:y+h, x:x+w]

plt.imsave(OUT_DIR + SAMPLING + '_out.jpg', mosaic)
