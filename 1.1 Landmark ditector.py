import cv2
import dlib
from imutils import face_utils
#for numpy array conversions

#width,height =(1920,1080)

camera=cv2.VideoCapture(0)

face_detector=dlib.get_frontal_face_detector()
#pretrained face detecting classifier to dlib module

landmark_detector=dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
#pretrained external algorithm for detecting landmarks of a face


while(True):

    ret,img=camera.read()
    
    if(ret):
        
        gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

        rects=face_detector(gray)
        #pretrained algorithm.predict

        for rect in rects:
            x1=rect.left()
            y1=rect.top()
            x2=rect.right()
            y2=rect.bottom()
            cv2.rectangle(img, (x1-1, y1-20),(x1+60-1,y1),(0,255,0),-1)
            cv2.putText(img,'Face',(x1-1+2, y1-20-2), cv2.FONT_HERSHEY_SIMPLEX,1, (255,255,255),2)
            #img :where the rect should be drawn
            #parameter2 : text
            #parameter3 : origin pointer
            #parameter4 : font  
            #parameter5 : font scale
            #parameter6 : font color
            #parameter7 : thickness
            
            cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
            #img.where the ret should be drawn
            #(0,255,0) : color in BGR
            #2 : width in px //if -1 filled rectangle
    

            points=landmark_detector(gray,rect)
            #passing the fray image and bounding rectangle to the landmark_detector
            #points object contains the 60 points

            points=face_utils.shape_to_np(points)
            #converting the 68 points object into a numpy array

            for p in points:

                center=(p[0],p[1])
                cv2.circle(img,center,2,(0,255,255),-1)

        img=cv2.resize(img, (1080,720))
        #resize img
        #parameter1 : object 
        #parameter2 : width and height
        
        cv2.imshow('IMG', img)
        #cv2.imshow('GRAY', gray)
    cv2.waitKey(1)

