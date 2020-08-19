import os
import cv2
import cv2_tools

def getData():
    fd = cv2.CascadeClassifier("cascades/data/haarcascade_frontalface_default.xml") #your_own_path
    cam = cv2.VideoCapture(0)
    cam.set(3, 640)
    cam.set(4, 480)
    face_id = input('\n enter user id ==>  ')
    print("\n Initializing face capture...")
    count = 0
    while True:
        ret, img = cam.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = fd.detectMultiScale(gray, 1.3, 5)
        for (x,y,w,h) in faces:
            cv2.rectangle(img, (x, y), (x+h, y+h), (255, 0, 0), 2)
            count += 1
            cv2.imwrite("dataset/user." + str(face_id) + "." + str(count) + ".jpg", gray[y:y+h, x:x+w])
            cv2.imshow("image", img)
        k = cv2.waitKey(100) & 0xff
        if k == 27:
            break
        elif count >= 30:
            break
    print(" BYEBYE!")
    cam.release()

getData()
cv2.destroyAllWindows()
