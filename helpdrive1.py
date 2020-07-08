from keras.models import load_model
import cv2
import math
import matplotlib.pyplot as plt
import matplotlib.image as mp
import numpy as np
from PIL import Image
import os
import pickle
import time
import serial


left_lines = []
right_lines = []
model = load_model('/home/bater/PycharmProjects/нейро/keras-tutorial/output/smallvggnet.model')
lb = pickle.loads(
        open('/home/bater/PycharmProjects/нейро/keras-tutorial/output/smallvggnet_lb.pickle', "rb").read())

def exercise(prep):
    time = 0.19
    if prep is not None:
        angel = math.radians(90)
        speedwheel = round(63.56*(angel/time) + 84.44)
        ser.write(b'#setup_149#1\r\n')
        text = "#setup_{}{}".format(speedwheel, '#2\r\n')
        ser.write(text)
        time.sleep(time)
        ser.write(b'#setup_149#2\r\n').
    else:



        
        
def grayscale(img):
    return cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)

def region_of_interest(img, vertices):

    mask = np.zeros_like(img)   
    
    if len(img.shape) > 2:
        channel_count = img.shape[2]
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
          
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

def draw_lines(img: object, lines: object, color: object = [255, 0, 0], thickness: object = 6) -> object:
    for line in lines:
        for x1,y1,x2,y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)    
        return img

def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines(line_img, lines)
    return line_img

def isolate_color_mask(img, low_thresh, high_thresh):
    assert(low_thresh.all() >=0  and low_thresh.all() <=255)
    assert(high_thresh.all() >=0 and high_thresh.all() <=255)
    return cv2.inRange(img, low_thresh, high_thresh)

def gamma(image, gamma=1.0):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
        for i in np.arange(0, 256)]).astype("uint8")
    
    return cv2.LUT(image, table)

def get_int1(img):
    rows, cols = img.shape[:2]
    mask = np.zeros_like(img)
    
    left_bottom = [0 , rows]
    right_bottom = [cols * 0.5 , rows]
    left_top = [0 , rows*0.5 ]
    right_top = [cols * 0.5, 0.5*rows ]
    
    vertices = np.array([[left_bottom, left_top, right_top, right_bottom]], dtype=np.int32)
    
    if len(mask.shape) == 2:
        cv2.fillPoly(mask, vertices, 255)
    else:
        cv2.fillPoly(mask, vertices, (255, ) * mask.shape[2])
    return cv2.bitwise_and(img, mask)


def left_turn(img,lines):
    left_horizontal = []
    if lines is not None:
        for line in lines:
            for x1,y1,x2,y2 in line:
                if math.fabs(y2-y1) == 0:
                    if left_horizontal is None:
                        left_horizontal.append((x1,y1,x2,y2))
                    else:
                        pred = left_horizontal.pop()
                        left_horizontal.append((x1, y1, x2, y2))
                        if pred[1] != y1:
                            img = cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 10)
                            img = cv2.line(img, (pred[0], pred[1]), (pred[2], pred[3]), (255, 0, 0), 10)
    return img

def get_int(img):
    rows, cols = img.shape[:2]
    #print(rows,cols)
    mask = np.zeros_like(img)

    left_bottom = [cols * 0.1, rows]
    right_bottom = [cols * 0.9, rows]
    left_top = [cols * 0.3, rows * 0.4]
    right_top = [cols * 0.6, 0.4 * rows]

    vertices = np.array([[left_bottom, left_top, right_top, right_bottom]], dtype=np.int32)

    if len(mask.shape) == 2:
        cv2.fillPoly(mask, vertices, 255)
    else:
        cv2.fillPoly(mask, vertices, (255,) * mask.shape[2])
    return cv2.bitwise_and(img, mask)

def get_hough_lines(img, rho=1, theta=np.pi/180, threshold=20, min_line_len=80, max_line_gap=100):
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), 
                            minLineLength=min_line_len, maxLineGap=max_line_gap)
    #print (lines)
    return lines

def find_sizeimage(img):
    rows, cols = img.shape[:2]
    x_mean=(cols/2)
    #print (rows)
    #print (cols)
    return x_mean

def line_decod(img,lines):
    rows, cols = img.shape[:2]
    x_mean = (cols/2)
    y_mean = (rows/2)
    right_lines = []
    right_lengths = []
    #print(y_mean)
    for line in lines:
        for x1,y1,x2,y2 in line:
            if x2 > x_mean and x1 > x_mean:
                if math.fabs(y2-y1) > math.fabs(x2-x1):
                    right_lengths.append(np.sqrt((y2-y1)**2 + (x2-x1)**2))
                    right_lines.append((x1,y1,x2,y2))
                    #img = cv2.line(img, (x1, y1), (x2, y2), (255,0,0), 10)
                    #print ("right_lines")
    return right_lines, right_lengths

def line_decod1(img,lines):
    rows, cols = img.shape[:2]
    x_mean = (cols/2)
    y_mean = (rows/2)
    left_lines = []
    left_lengths = []
    #print(y_mean)
    for line in lines:
        for x1,y1,x2,y2 in line:
            if x2 < x_mean and x1 < x_mean:
                if math.fabs(y2-y1) > math.fabs(x2-x1):
                    left_lengths.append(np.sqrt((y2-y1)**2 + (x2-x1)**2))
                    left_lines.append((x1,y1,x2,y2))
                    #img1 = cv2.line(img, (x1, y1), (x2, y2), (255,0,0), 10)
                    #print ("left_lines")
    return left_lines, left_lengths

def max_length(left_lengths,right_lengths):
    if len(left_lengths)>0:
        left_avg = np.argmax(left_lengths)
    if len(right_lengths)>0:
        right_avg = np.argmax(right_lengths)
    #print(left_avg)
    #print(right_avg)
    return left_avg,right_avg

def max_length1(image,left_lengths,right_lengths):
    if len(left_lengths)>0:
        left_avg = np.argmax(left_lengths)
    else:
        prep = prep_decod(image)
    if len(right_lengths)>0:
        right_avg = np.argmax(right_lengths)
    else:
        prep = prep_decod(image)
    #print(left_avg)
    #print(right_avg)
    return left_avg,right_avg,prep

def prep_decod(img):
    output = img.copy()
    image = cv2.resize(img, (64, 64))
    image = image.astype("float") / 255.0
    flatten = -1
    if flatten > 0:
        image = image.flatten()
        image = image.reshape((1, image.shape[0]))

    # в противном случае мы работаем с CNN -- не сглаживаем изображение
    # и просто добавляем размер пакета
    else:
        image = image.reshape((1, image.shape[0], image.shape[1],
                               image.shape[2]))

    print("[INFO] loading network and label binarizer...")
    
    

    # распознаём изображение
    preds = model.predict(image)

    # находим индекс метки класса с наибольшей вероятностью
    # соответствия
    i = preds.argmax(axis=1)[0]
    label = lb.classes_[i]
    

    # делаем предсказание на изображении
    preds = model.predict(image)
    print(preds)

    # находим индекс метки класса с наибольшей вероятностью
    # соответствия
    i = preds.argmax(axis=1)[0]
    label = lb.classes_[i]

    # рисуем метку класса + вероятность на выходном изображении
    return label

def slope(x1,y1,x2,y2):
    if x2-x1 == 0:
        return math.inf, 0
    slope = (y2-y1)/(x2-x1)
    intercept = y1 - slope * x1
    #print(slope,intercept)
    return slope, intercept

def traek_line(img,left_avg,right_avg,left_lines,right_lines):
    #traek_left = []
    #traek_right = []
    traek_left = left_lines[left_avg]
    traek_right = right_lines[right_avg]
    y1_general = [traek_left[1],traek_right[1]]
    y2_general = [traek_left[3],traek_right[3]]
    y1_min = np.argmin(y1_general)
    y2_max = np.argmax(y2_general)
    #K1,b1 = slope(traek_left[0],traek_left[1],traek_left[2],traek_left[3])
    #K2,b2 = slope(traek_right[0],traek_right[1],traek_right[2],traek_right[3])
    #angel_slope = (math.tan((K2-K1)/(1+K1*K2)))
    #b_new = 640 - ((traek_left[0] + traek_right[0]) / 2)*angel_slope
    #x_new = (y2_general[y2_max]-b_new)/angel_slope
    #traek_point2 =[int((traek_left[0] + traek_right[0]) / 2), 640, int(x_new), int(y2_general[y2_max])]
    #print('angel_slope', angel_slope)
    #traek_point = [int((traek_left[0] + traek_right[0]) / 2), int(y1_general[y1_min]), int((traek_left[2] + traek_right[2]) / 2), int(y2_general[y2_max])]
    traek_point1 = [int(traek_left[0]-((traek_left[0] - traek_right[0]) / 2)), int(y1_general[y1_min]), int(traek_left[2]-((traek_left[2] - traek_right[2]) / 2)), int(y2_general[y2_max])]
    #traek_point = [int((traek_left[0]+traek_right[0])/2), int(math.fabs(traek_left[1]-traek_right[1])/2),
                   #int((traek_left[2]+traek_right[2])/2), int(math.fabs(traek_left[3]-traek_right[3])/2)]
    x1, y1, x2, y2 = traek_point1
    angel1 = math.degrees(math.atan2(640 - 0, 240 - 240))
    angel3 = math.degrees(math.atan2(y2-y1,x2-x1))
    #angel2 = math.degrees(math.atan2(y2-y1,x2-x1))
    image = cv2.line(img, (x1, y1), (x2, y2), (0,0,255), 10)
    print('ANGEL', angel3)
    return image, angel3
    #print(traek_point)
    #print(traek_left)
    #print(traek_right)


#def power_wheel(angel):
    time.sleep(0.5)
    ser.write(b'#mode_fwrd#2\r\n')
    if angel > 0 and angel > 50:
        ser.write(b'#setup_151#1\r\n')
        ser.write(b'#setup_340#2\r\n')
    else:
        ser.write(b'#setup_151#1\r\n')
        ser.write(b'#setup_200#2\r\n')
    if angel < 0 and angel > -50:
        ser.write(b'#setup_350#1\r\n')
        ser.write(b'#setup_151#2\r\n')
    else:
        ser.write(b'#setup_200#1\r\n')
        ser.write(b'#setup_151#2\r\n')





def get_line_length(line):
    for x1, y1, x2, y2 in line:
        return np.sqrt((y2-y1)**2 + (x2-x1)**2)

def line_func(image,lines):
    left_lines = []
    right_lines = []
    left_lengths = []
    right_lengths = []
    if lines is not None:
        for line in lines:
            slope, intercept = get_line_slope_intercept(line)
            continue
        line_len = get_line_length(line)
        if slope < 0:
            left_lines.append((slope, intercept))
            left_lengths.append(line_len)
        else:
            right_lines.append((slope, intercept))
            right_lengths.append(line_len)

    # average
    left_avg = np.dot(left_lengths, left_lines) / np.sum(left_lengths) if len(left_lengths) > 0 else None
    right_avg = np.dot(right_lengths, right_lines) / np.sum(right_lengths) if len(right_lengths) > 0 else None
    return left_avg,right_avg


#def traek_line(img,right_lines,left_lines):
    line_image = np.zeros_like(img)
    for x1,y1,x2,y2 in right_lines:
        for x3,y4,x5,y6 in left_lines:
            left_avgy = y6
            right_avgy = y2
            pount_avgy_traek = (left_avgy + right_avgy)/2
            pount_avgx_traek = (x5 + x2)/2
            pount_miny_traek = (y4 + y1)/2
            pount_mixx_traek = (x3 + x1)/2    
            line_traek = [pount_mixx_traek,pount_miny_traek,pount_avgx_traek,pount_avgy_traek]
            line_image = cv2.Line(img, (line_traek[0], line_traek[1]), (line_traek[2], line_traek[3]), (0,0,255), 10)
    return line_image
                            
def display_lines(image, lines):
    line_image = np.zeros_like(image)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            line_image = cv2.line(line_image, (x1, y1), (x2, y2), (255,0,0), 10)
    return line_image

def display_lines1(image, left_lane, right_lane):
    line_image = np.zeros_like(image)
    if left_lane is not None:
        for line in left_lane:
            for x1,y1,x2,y2 in line:
                line_image1 = cv2.line(line_image, (x1, y1), (x2, y2), (255,0,0), 10)
                if right_lane is not None:
                    for x1,y1,x2,y2 in right_lane:
                        line_image2 = cv2.line(line_image1, (x1, y1), (x2, y2), (255, 0, 0), 10)
            return line_image2

def draw_lines_left(img, left_lane, color=[255, 0, 0], thickness=2):
    if left_lane is not None:
        #for line in left_lane:
        x1,y1,x2,y2 =left_lane
            #for x1,y1,x2,y2 in line:
        cv2.line(img, (x1, y1), (x2, y2), color, thickness)
    return img

def draw_lines_right(img, right_lane, color=[255, 0, 0], thickness=2):
    if right_lane is not None:
        x1,y1,x2,y2 = right_lane
            #x1,y1,x2,y2 = line
            #for x1,y1,x2,y2 in line:
        cv2.line(img, (x1, y1), (x2, y2), color, thickness)
    return img

def get_line_length(line):
    for x1, y1, x2, y2 in line:
        return np.sqrt((y2-y1)**2 + (x2-x1)**2)

def get_line_slope_intercept(line):
    if x2-x1 == 0:
        return math.inf, 0
    slope = (y2-y1)/(x2-x1)
    intercept = y1 - slope * x1
    #print(slope,intercept)
    return slope, intercept
        
def get_lines_slope_intecept(lines):
    left_lines = []
    right_lines = []
    left_lengths = []
    right_lengths = []
    if lines is not None:
        for line in lines:
            slope, intercept = get_line_slope_intercept(line)
            if slope == math.inf:
                continue
        line_len = get_line_length(line)
        if slope < 0:
            left_lines.append((slope, intercept))
            left_lengths.append(line_len)
        else :
            right_lines.append((slope, intercept))
            right_lengths.append(line_len)
            
    # average
    left_avg = np.dot(left_lengths, left_lines)/np.sum(left_lengths) if len(left_lengths) > 0 else None
    right_avg = np.dot(right_lengths, right_lines)/np.sum(right_lengths) if len(right_lengths) > 0 else None
    #print(left_avg)
    #print(right_avg)
    return left_avg, right_avg

def convert_slope_intercept_to_line(y1, y2 , line):
    if line is None:
        return None
    row = 480
    cols = 640
    slope, intercept = line
    x1 = int((y1- intercept)/(slope+0.0000000001))
    #line_try.append(x1)
    y1 = int(y1)
    #line_try.append(y1)
    x2 = int((y2- intercept)/(slope+0.0000000001))
    #line_try.append(x2)
    y2 = int(y2)
    #line_try.append(y2)
    if x1 < row and x2 < row:
        line_try = [x1, y1, x2, y2]
    #print(x1,y1,x2,y2)
        #print(line_try)
        return line_try

def get_lane_lines(img, lines):
    
    left_avg, right_avg = get_lines_slope_intecept(lines)
    print(lines)
    for line in lines:
        for x1, y1, x2, y2 in line:
            left_lane = convert_slope_intercept_to_line(y1, y2, left_avg)
            right_lane = convert_slope_intercept_to_line(y1, y2, right_avg)
        #print(left_lane)
        #print(right_lane)
            return left_lane, right_lane

def draw_weighted_lines(img, lines, color=[255, 0, 0], thickness=2, alpha = 1.0, beta = 0.95, gamma= 0):
    mask_img = np.zeros_like(img)
    for line in lines:
        if line is not None:
            cv2.line(mask_img, *line, color, thickness)            
    return weighted_img(mask_img, img, alpha, beta, gamma)

def display_lines(image, lines):
    line_image = np.zeros_like(image)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            cv2.line(line_image, (x1, y1), (x2, y2), (255,0,0), 10)
    return line_image

def draw_lines1(img, traektory, color=[255, 0, 0], thickness=2):
    for line in traektory:
        for x1,y1,x2,y2 in line[0]:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)
        return img






