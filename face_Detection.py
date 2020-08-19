import cv2
import numpy as np

faceCascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_default.xml') #your_own_path
cap = cv2.VideoCapture(0)
cap.set(3,640)
cap.set(4,480)
while True:
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray, 1.2, 5, minSize=(20, 20))
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,200),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
    cv2.imshow('video',img)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
cap.release()
cv2.destroyAllWindows()
