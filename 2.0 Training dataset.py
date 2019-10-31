import os
import cv2
import dlib
from imutils import face_utils


#used when you are dealing eith operating system

#data_path='D:\Academy of Innovative Education\Day 3\Project 1 Materials\Face Shapes'
#<!--this is using absolute path-->

data_path='Face Shapes'
labels=os.listdir(data_path)

face_detector=dlib.get_frontal_face_detector()
landmark_detector=dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

data=[]
target=[]


def points_68(gray):
    #defined by us
    rects=face_detector(gray)
    #pretrained algorithm.predict

    for rect in rects:
        x1=rect.left()
        y1=rect.top()
        x2=rect.right()
        y2=rect.bottom()
        cv2.rectangle(img, (x1-1, y1-20),(x1+60-1,y1),(0,255,0),-1)
        cv2.putText(img,'Face',(x1-1+2, y1-20-2), cv2.FONT_HERSHEY_SIMPLEX,1, (255,255,255),2)
           
        cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
       
        points=landmark_detector(gray,rect)
       
        points=face_utils.shape_to_np(points)
        return points

def create_features(points,label):

    target_dict={'Diamond':0,'Oblong':1,'Oval':2,'Round':3,'Square':4,'Triangle':5}

    my_points=points[2:9,0]
    #my_points contains x coordinates of p3-p9 (p2-p8)
    D1=my_points[6] -my_points[0]
    D2=my_points[6] -my_points[1]
    D3=my_points[6] -my_points[2]
    D4=my_points[6] -my_points[3]
    D5=my_points[6] -my_points[4]
    D6=my_points[6] -my_points[5]

    d1=D2/float(D1)*100
    d2=D3/float(D1)*100
    d3=D4/float(D1)*100
    d4=D5/float(D1)*100
    d5=D6/float(D1)*100

    data.append((d1,d2,d3,d4,d5))
    #append creates a new row
    target.append(target_dict[label])
    
    
for label in labels:
    #print(label)
    imgs_path=os.path.join(data_path,label)
    img_names=os.listdir(imgs_path)
    #print(img_names)

    for img_name in img_names:

        img_path=os.path.join(imgs_path,img_name)
        img=cv2.imread(img_path)
        #cv2.imshow('LIVE', img)
        #cv2.waitKey(100)
        #at this point we read all the data in the dataset

        gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        #convert to grayscale
        points=points_68(gray)
        create_features(points,label)

import pickle
#using this lib,arrays van be saved/load to physical files
            
import numpy as np

data=np.array(data)
target=np.array(target)

pickle.dump(data,open ('data.pickle', 'wb'))
pickle.dump(target,open ('target.pickle', 'wb'))
