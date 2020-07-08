from helpdrive1 import *
import cv2
import numpy as np
from collections import deque
import serial
import time

#ser = serial.Serial('/dev/ttyS0',9600)


        
cap = cv2.VideoCapture(0)

while(1):
    _, image = cap.read()
    #cv2.imshow('image ',image )
    #size = find_sizeimage(image)
        
    gray_img = grayscale(image)
         
    darkened_img = gamma(gray_img, 0.5)
    #cv2.imshow(' darkened_img', darkened_img)
    #yellow_mask = isolate_color_mask(cv2.cvtColor(image, cv2.COLOR_RGB2HLS), np.array([10, 0, 100], dtype=np.uint8), np.array([40, 255, 255], dtype=np.uint8))
    white_mask = isolate_color_mask(cv2.cvtColor(image, cv2.COLOR_RGB2HLS), np.array([0, 200, 0], dtype=np.uint8), np.array([200, 255, 255], dtype=np.uint8))
    mask = cv2.bitwise_or(white_mask, darkened_img)
    
    colored_img = cv2.bitwise_and(darkened_img, darkened_img, mask=mask) 
    blurred_img = cv2.GaussianBlur(colored_img, (7, 7), 0)
    canny_img = cv2.Canny(blurred_img, 70, 140)
    int_img = get_int(canny_img)
    #int_img1 = get_int1(canny_img)
    cv2.imshow('int_img',int_img)
    hough_lines = get_hough_lines(int_img)
    if hough_lines is None:
        prep = prep_decod(image)
        #print(prep)
    else:
    #left_lane,right_lane = get_lane_lines(canny_img,hough_lines)
    #line_image = display_lines1(image,left_lane,right_lane)
    #right_image = draw_lines_right(image, right_lane)
    #left_image = draw_lines_left(image, left_lane)
    #cv2.imshow('left_image', left_image)
    #cv2.imshow('right_image', right_image)
    #cv2.imshow('line_image',line_image)
    #hough_lines1 = get_hough_lines(int_img1)
    #left_image = left_turn(image,hough_lines1)
    #cv2.imshow('left',left_image)
        right_lines, right_lengths = line_decod(image,hough_lines)
        left_lines, left_lengths = line_decod1(image,hough_lines)
        if (left_lengths and right_lengths) is None:
            prep1 = prep_decod(image)
            print(prep1)
        else:
            left_avg, right_avg = max_length(left_lengths,right_lengths)
    #left_avg, right_avg, prep = max_length1(image, left_lengths, right_lengths)
            traek_image, angel = traek_line(image,left_avg,right_avg,left_lines,right_lines)
    #print(prep)
            cv2.imshow('traek_image', traek_image)
    #line_image = traek_line(image,right_lines,left_lines)
    #line_image = display_lines(image, hough_lines)
    #cv2.imshow('image1', image1)
    #cv2.imshow('image2', image2)
            hough_img = display_lines(image, hough_lines)
            cv2.imshow('hough_img',hough_img)
            hough_img1 = draw_lines(image, hough_lines)
    #power_wheel(angel)
    #cv2.imshow('hough_img ',hough_img )
    #cv2.imshow('hough_img1 ',hough_img1 )
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
cv2.destroyAllWindows()

