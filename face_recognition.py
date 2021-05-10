import numpy as np
import cv2 as cv

harr_cascade=cv.CascadeClassifier('haar_face.xml')
people=['Elon Musk','Jeff Bezos']

face_recognizer=cv.face.LBPHFaceRecognizer_create()
face_recognizer.read('face_trained.yml')

img=cv.imread('elon2.jpeg')
gray=cv.cvtColor(img,cv.COLOR_BGR2GRAY)
#detect the face
faces_rect=harr_cascade.detectMultiScale(gray,1.1,4)

for (x,y,w,h) in faces_rect:
    faces_roi=gray[y:y+h,x:x+w]
    label,confidence=face_recognizer.predict(faces_roi)
    print(label,confidence)
    cv.putText(img,str(people[label]),(20,20),cv.FONT_HERSHEY_COMPLEX,1.0,(0,255,0),thickness=2)
    cv.rectangle(img,(x,y),(x+w,y+h),(0,255,0),thickness=2)
cv.imshow('face_detected',img)
cv.waitKey(0)


