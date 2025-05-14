import cv2 
import numpy as np
from vectorization import delaunay, naive_voronoi
import constants

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
capture = cv2.VideoCapture(0)

while 1: 
    ret, img = capture.read() 
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    # Functional, but slow
    mosaic = delaunay(img, constants.UNIFORM, faces)

    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,255,0),2) 
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]

    # Stacks the video feeds for the comparison
    comparison_video = np.hstack((img, mosaic))
    cv2.imshow('screen', comparison_video)

    # Use esc to exit
    k = cv2.waitKey(30) & 0xff 
    if k == 27:
        break

# Close window
capture.release()
cv2.destroyAllWindows() 
