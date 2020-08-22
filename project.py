import os
import cv2
import cv2_tools
import numpy as np
from PIL import Image

def capture():
    fd = cv2.CascadeClassifier("cascades/data/haarcascade_frontalface_default.xml")
    cam = cv2.VideoCapture(0)
    cam.set(3, 640)
    cam.set(4, 480)
    face_id = input("Enter user id: ")
    print("Enter name: ")
    name1 = input(str())
    name.append(name1)
    print("Starting face scanning...")
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
        elif count >= 50:
            break
    print("BYEBYE!")
    cam.release()
    cv2.destroyAllWindows()

def get_label(path):
    imagepaths = [os.path.join(path,f) for f in os.listdir(path)]
    facesamples = []
    ids = []
    for imagePath in imagepaths:
        PIL_img = Image.open(imagePath).convert('L')
        img_numpy = np.array(PIL_img, 'uint8')
        id = int(os.path.split(imagePath)[-1].split(".")[1])
        faces = fd.detectMultiScale(img_numpy)
        for (x, y, w, h) in faces:
            facesamples.append(img_numpy[y:y + h, x:x + w])
            ids.append(id)
    return facesamples, ids

def getname():
    return name

def menu():
    print("Select from the options below: ")
    print("1. Scan the faces and generate dataset.")
    print("2. Train the faces.")
    print("3. Test the trained faces.")
    print("4. Exit")
    n = input()
    return n

if __name__ == '__main__':
    n = menu()
    if n == '1':
        name = []
        flag = 1
        while flag == 1:
            print("Want to scan more faces: (y/n)")
            s = input(str())
            if s == 'y':
                capture()
            else:
                n = menu()
                if n == '2':
                    path = 'dataset'
                    recog = cv2.face.LBPHFaceRecognizer_create()
                    fd = cv2.CascadeClassifier("cascades/data/haarcascade_frontalface_default.xml")
                    print("Training faces. Wait ...")
                    faces, ids = get_label(path)
                    recog.train(faces, np.array(ids))
                    recog.write('trainer/trainer.yml')
                    print("{0} faces trained.".format(len(np.unique(ids))))
                    n = menu()
                    if n == '3':
                        recognizer = cv2.face.LBPHFaceRecognizer_create()
                        recognizer.read('trainer/trainer.yml')
                        cascadePath = "cascades/data/haarcascade_frontalface_default.xml"
                        faceCascade = cv2.CascadeClassifier(cascadePath)
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        cam = cv2.VideoCapture(0)
                        cam.set(3, 640)
                        cam.set(4, 480)
                        minW = 0.1 * cam.get(3)
                        minH = 0.1 * cam.get(4)
                        while True:
                            ret, img = cam.read()
                            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                            faces = faceCascade.detectMultiScale(gray, 1.3, 5, minSize=(int(minW), int(minH)), )
                            for (x, y, w, h) in faces:
                                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                                # as low as possible
                                id, confidence = recognizer.predict(gray[y:y + h, x:x + w])
                                # print(confidence)
                                if (confidence < 100):
                                    s = getname()
                                    print(s[id])
                                    # so actual confidence is 100-conf
                                    confidence = "  {0}%".format(round(100 - confidence))
                                else:
                                    id = "unknown"
                                    confidence = "  {0}%".format(round(100 - confidence))
                                    #cv2.putText(img, (s[id]), (x + 5, y - 5), font, 1, (255, 255, 255), 2)
                            cv2.imshow('camera', img)
                            k = cv2.waitKey(10) & 0xff
                            if k == 27:
                                break
                        print("\n Exit")
                        cam.release()
                        cv2.destroyAllWindows()
                        n = menu()
                        if n == '4':
                            exit()
