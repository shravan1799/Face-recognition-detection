import cv2
import numpy as np

tab = cv2.VideoCapture(0)
tab.set(3,640)
tab.set(4,480)
while (True):
    ret, frame = tab.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.imshow("Frame", frame)
    cv2.imshow("Gray", gray)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
tab.release()
cv2.destroyAllWindows()