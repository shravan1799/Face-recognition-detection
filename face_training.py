import cv2
import numpy as np
from PIL import Image
import os

path = 'dataset'
recog = cv2.face.LBPHFaceRecognizer_create()
fd = cv2.CascadeClassifier("cascades/data/haarcascade_frontalface_default.xml")
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

print ("\n Training faces. It will take a few seconds. Wait ...")
faces, ids = get_label(path)
recog.train(faces, np.array(ids))
recog.write('trainer/trainer.yml')
print ("\n {0} faces trained. Exiting Program BYEBYE!".format(len(np.unique(ids))))