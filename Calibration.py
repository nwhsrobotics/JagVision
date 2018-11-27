import os
import sys
import cv2
import numpy as np
import time
import math
import collections
from collections import namedtuple
import pickle
import json
import codecs

def log_values(type, values):
    print("%s --"%(type))
    print(values)

constant_height_pixels = 172
constant_distance = 36
cube_height = 11

last_command = None

font = cv2.FONT_HERSHEY_SIMPLEX

cap = cv2.VideoCapture(0)

print('Welcome to JagVision Calibration')
print('')
print('You will be asked to move the cube around to properly')
print('calibrate the vision system')

def get_height(lower_limit, upper_limit):
    while(True):
        #rows,cols = frame.shape[:2]
        #M = cv2.getRotationMatrix2D((cols/2,rows/2),180,1)
        #frame = cv2.warpAffine(frame,M,(cols,rows))

        ret, frame = cap.read()

        hsv_raw = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hsv = hsv_raw
        hsv = cv2.blur(hsv_raw,(20,20))
        lower_limit2 = np.array(lower_limit)
        upper_limit2 = np.array(upper_limit)
        print(lower_limit2)
        print(upper_limit2)

        mask = cv2.inRange(hsv, lower_limit2, upper_limit2)

        masked_image = cv2.bitwise_and(hsv,hsv, mask= mask)

        h, s, v = cv2.split(masked_image)

        gray = v

        ret,thresh = cv2.threshold(gray,127,255,0)
        #testvariable = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        #print(len(testvariable))
        im2, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        contoured = cv2.drawContours(frame, contours, -1, (0,255,0), 3) 

        height, width = frame.shape[:2]
        x_center = int(width/2)
        y_center = int(height/2)
        cv2.line(frame,(0,y_center),(width,y_center),(0,0,255),2)
        cv2.line(frame,(x_center,0),(x_center,height),(0,0,255),2)

        filtered_contours = []
        
        for cnt in contours:
            x,y,w,h = cv2.boundingRect(max(contours, key = cv2.contourArea))
            #cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
            h = float(h)
            w = float(w)
            ratio = (float(h/w))
            if float(0.4) < (ratio) < float(2):
                filtered_contours.append(cnt)
        

        
        if len(filtered_contours) != 0:
            print('cube found')
            cube_found = True
            x,y,w,h = cv2.boundingRect(max(filtered_contours, key = cv2.contourArea))
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
        else:
            h = 150
            print('An error has occured')
            print('You MUST rerun the calibration for optimal results')
        cv2.imshow('final',masked_image)
        if cv2.waitKey(1) & 0xFF == ord('a'):
            
            cap.release()
            cv2.destroyAllWindows()
            return h




def checkpoint_one():
    first = True
    while(True):
        if first == True:
            first = False
            print('Place Cube At Crosshairs and press A')
        __, frame = cap.read()
        hsv_raw = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        height, width = frame.shape[:2]
        x_center = int(width/2)
        y_center = int(height/4)
        cv2.line(frame,(0,y_center),(width,y_center),(0,0,255),2)
        cv2.line(frame,(x_center,0),(x_center,height),(0,0,255),2)


        cv2.imshow('Frame',frame)

        if cv2.waitKey(10) & 0xFF == ord('a'):
            hsv_one = hsv_raw[y_center,x_center]
            cv2.destroyAllWindows()
            return hsv_one
            break


def checkpoint_two():
    first = True
    while(True):
        if first == True:
            first = False
            print('Place Cube At Crosshairs and press A')
        __, frame = cap.read()
        hsv_raw = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        height, width = frame.shape[:2]
        x_center = int(width/4)
        y_center = int(3*height/4)
        cv2.line(frame,(0,y_center),(width,y_center),(0,0,255),2)
        cv2.line(frame,(x_center,0),(x_center,height),(0,0,255),2)

        cv2.imshow('Frame',frame)

        if cv2.waitKey(10) & 0xFF == ord('a'):
            hsv_two = hsv_raw[y_center,x_center]
            cv2.destroyAllWindows()
            return hsv_two
            break


def checkpoint_three():
    first = True
    while(True):
        if first == True:
            first = False
            print('Place Cube At Crosshairs and press A')
        __, frame = cap.read()
        hsv_raw = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        height, width = frame.shape[:2]
        #print(height, width)
        x_center = int((3*(width))/4)
        y_center = int(3*height/4)
        #print(x_center)
        #print(y_center)
        cv2.line(frame,(0,y_center),(width,y_center),(0,0,255),2)
        cv2.line(frame,(x_center,0),(x_center,height),(0,0,255),2)

        cv2.imshow('Frame',frame)

        if cv2.waitKey(10) & 0xFF == ord('a'):
            hsv_three = hsv_raw[y_center,x_center]
            cv2.destroyAllWindows()

            return hsv_three
            break

cp_one = (checkpoint_one())
log_values("cp_one", cp_one)

cp_two = (checkpoint_two())
log_values("cp_two", cp_two)

cp_three = (checkpoint_three())
log_values("cp_three", cp_three)

# cp_total = np.array( cp_one + cp_two + cp_three)

#cp_total = np.zeros(3)
#for i,v in enumerate(cp_one):
#    cp_total[i] = v + cp_two[i] + cp_three[i]
#log_values("cp_total", cp_total)

hsv_average = np.mean([cp_one,cp_two,cp_three],axis=0, dtype=np.uint32)
log_values("hsv_average", hsv_average)

lower_limit = hsv_average-[30,70,70]
log_values("lower_limit", lower_limit)

upper_limit = hsv_average+[25,15,15]
log_values("upper_limit", upper_limit)



print('Place the cube exactly 36 inches from the camera')
print('With a blank yellow side facing the lens')
print('Is cube ready? y/n')
cont = input()

height = get_height(lower_limit,upper_limit)
print(height)

print(hsv_average)
cap.release()

data = dict()
data['HSVlower']=(lower_limit).tolist()
data['HSVupper']=(upper_limit).tolist()
data['height']=(height)

with open('HSV.txt','w') as output:
    json.dump(data,output)

time.sleep(10)
quit()
exit(0)
