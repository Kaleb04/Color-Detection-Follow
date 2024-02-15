import numpy as np
import cv2

#color testing
# green = np.uint8([[[0,0,0 ]]])
# hsv_green = cv2.cvtColor(green,cv2.COLOR_BGR2HSV)
# print (hsv_green)

cap=cv2.VideoCapture(0)
startx=20
starty=20

while True:
    ret, frame=cap.read()
    width=int(cap.get(10))
    height=int(cap.get(10))

    hsv=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    lower_yellow=np.array([170,100,100])
    upper_yellow=np.array([200,250,250])
    #color notes down below

    mask=cv2.inRange(hsv, lower_yellow, upper_yellow)

    result=cv2.bitwise_and(frame,frame, mask=mask)
    mask_contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.circle(frame, (startx, starty), 3, (0, 255, 0), 3)
    if len(mask_contours) != 0:
        for mask_contour in mask_contours:
            if cv2.contourArea(mask_contour) > 100:
                x, y, w, h = cv2.boundingRect(mask_contour)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 3)
                cv2.rectangle(result, (x, y), (x + w, y + h), (0, 0, 255), 3)
                if(startx<x+w/2):
                    startx+=10
                elif(startx>x+w/2):
                    startx-=10
                if(starty<y+h/2):
                    starty+=10
                elif(starty>y+h/2):
                    starty-=10
    cv2.imshow('cap',frame) 
    cv2.imshow('frame',result) 
    if cv2.waitKey(1)==ord('q'): #key detection
        break 
cap.release()  

cv2.destroyAllWindows()

'''
skin color [000,70,70][170,255,255]
red [170,100,100][200,250,250]
'''
# img= cv2.imread('image.jpg',1)    #processes an image code
# img=cv2.resize(img,(400,400))
# cv2.imshow('Image',img)   