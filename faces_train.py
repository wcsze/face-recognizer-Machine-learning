import os
import cv2 as cv
import numpy as np
people=['Elon Musk','Jeff Bezos']
DIR=r'D:\opencv\faces'
harr_cascade=cv.CascadeClassifier('haar_face.xml')
features=[]
labels=[]
def create_train():
    for person in people:
        path=os.path.join(DIR,person)
        label=people.index(person)

        for img in os.listdir(path):
            img_path=os.path.join(path,img)
            img_array=cv.imread(img_path)
            gray=cv.cvtColor(img_array,cv.COLOR_BGR2GRAY)

            faces_rect=harr_cascade.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=3)

            for (x,y,w,h) in faces_rect:
                faces_roi=gray[y:y+h,x:x+w] #y:height ,x:width
                features.append(faces_roi)
                labels.append(label)

create_train()
face_recognizer=cv.face.LBPHFaceRecognizer_create()
features=np.array(features,dtype='object')
labels=np.array(labels)
face_recognizer.train(features,labels)
np.save('features.npy',features)
np.save('labels.npy',labels)
face_recognizer.save('face_trained.yml')

