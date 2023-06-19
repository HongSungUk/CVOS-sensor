# -*- coding: utf-8 -*-
"""
Created on Tue Apr 25 15:34:33 2023

@author: hsu12
"""

import numpy as np
import pandas as pd
import cv2
from matplotlib import pyplot as plt
from glob import glob
import os
import time
from scipy.spatial import Voronoi, ConvexHull
import alphashape
import tensorflow as tf
import math

def makedirs(path): 
   try: 
        os.makedirs(path) 
   except OSError: 
       if not os.path.isdir(path): 
           raise

def length_cal(layer_name):
    length_list = []
    for i in range(len(layer_name)-1):
        length = abs(layer_name.iloc[i]['x'] - layer_name.iloc[i+1]['x'])
        length_list.append(length)
    return np.average(length_list)

def length_cal_list(layer_name):
    length_list = []
    for i in range(len(layer_name)-1):
        length = abs(layer_name.iloc[i]['x'] - layer_name.iloc[i+1]['x'])
        length_list.append(length)            
    return length_list

def Euclidian_distance(a, b):
    return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2

def get_index(a, b): 
    index_dist = []
    for i in range(len(b)):
        dist = []
        for o in range(len(a)):
            dist.append(Euclidian_distance(a[o], b[i]))                
        index_dist.append(dist.index(min(dist)))
    return index_dist    

def voronoi_volumes(points):
    v = Voronoi(points)
    vol = np.zeros(v.npoints)
    for i, reg_num in enumerate(v.point_region):
        indices = v.regions[reg_num]
        if -1 in indices: # some regions can be opened
            vol[i] = np.inf
        else:
            vol[i] = ConvexHull(v.vertices[indices]).volume
    return vol

def find_corresponding_points(srcPoints, dstPoints, threshold=0.1):
    # FLANN matcher 
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    srcPoints = srcPoints.astype('float32')
    dstPoints = dstPoints.astype('float32')
    matches = flann.match(srcPoints, dstPoints)
    
    # Among the matched points, the order is determined by excluding those with a distance greater than a certain distance and connecting the corresponding points
    srcPoints_new = np.zeros_like(srcPoints)
    dstPoints_new = np.zeros_like(dstPoints)
    visited = [False] * len(srcPoints)
    for m in matches:
        if m.distance <= threshold:
            src_idx = m.queryIdx
            dst_idx = m.trainIdx
            if not visited[src_idx]:
                srcPoints_new[src_idx] = srcPoints[src_idx]
                dstPoints_new[src_idx] = dstPoints[dst_idx]
                visited[src_idx] = True

    return srcPoints_new[visited], dstPoints_new[visited]

def weighted_average_direction(strain_map, strain_direction_map):
    x = np.average(strain_map * np.cos(strain_direction_map))
    y = np.average(strain_map * np.sin(strain_direction_map))
    average_direction = math.degrees(np.arctan2(y, x))
    return average_direction

# Video size
h = 480
w = 640

# Camera calibration 
mtx = np.array([[3.73639562e+03, 0.00000000e+00, 2.09275876e+02],
       [0.00000000e+00, 3.50019093e+03, 2.75594551e+02],
       [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
dist = np.array([[ 1.31613826e+01,  6.13595466e+02,  4.63391459e-02,
        -6.76091323e-01, -3.17320359e+04]])

newcameraMtx, roi = cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),1,(w,h))
x_r,y_r,w_r,h_r = roi

# Path setting
path = 'D://Reasearch/Optical-type strain sensor/Measure data/220818-28_total_performance_test_results/220819_test_2/*.jpg' # input data
img_list = glob(path)
save_path = os.path.dirname(path)+'_OTSS_5.2'
makedirs(save_path)

# Image filtering global parameters
sharpening_mask = np.array([[-1, -1, -1], [-1, 10, -1], [-1, -1, -1]]) #Custom sharpness filter
clahe = cv2.createCLAHE(clipLimit=8.0, tileGridSize=(8,8)) #CLAHE setting
cut = 10 # boundary setting
median_area = 500 # reference marker area
ref_center_point = (320, 240) # image center coordinate
initialize_time = 10 # How many frames after program start to start initialization

# Virtual coordinate algorithm parameters
threshold_value_x = 30 #30
threshold_value_y = 20 #30
image_limit_x0 = 0+cut+threshold_value_x
image_limit_x1 = w-cut-threshold_value_x
image_limit_y0 = 0+cut+threshold_value_y
image_limit_y1 = h-cut-threshold_value_y
out_index=[]
index_dist = []

# Curvature detection parameters
curvature_std_threshold = 300

# Image mapping parameters
RAN = 12.0
nn = 80
frame_to_frame_threshold = 50

interval_mapping = 80 # Interval of quadrant separation
w1, h1 = tuple(map(lambda x: x/interval_mapping + 1, (w, h))) 
min_value = 0
max_value = w1*h1 - 1
width_num_Q = (w1-1)/2
hight_num_Q = (h1-1)/2 + 1

q1 = []
for i in range(int(hight_num_Q)):
    q1.append(np.arange(min_value + width_num_Q + w1*i, width_num_Q + width_num_Q + w1*i))
q1 = np.array(q1).flatten().astype(np.int32)

q2 = []
for i in range(int(hight_num_Q)):
    q2.append(np.arange(min_value + w1*i, width_num_Q + w1*i))
q2 = np.array(q2).flatten().astype(np.int32)

q3 = []
for i in range(int(hight_num_Q)):
    q3.append(np.arange(min_value + w1*(hight_num_Q-1) + w1*i, width_num_Q + w1*(hight_num_Q-1)  + w1*i))
q3 = np.array(q3).flatten().astype(np.int32)

q4 = []
for i in range(int(hight_num_Q)):
    q4.append(np.arange(min_value + width_num_Q + w1*(hight_num_Q-1) + w1*i, width_num_Q + width_num_Q + w1*(hight_num_Q-1) + w1*i))
q4 = np.array(q4).flatten().astype(np.int32)

# CVOS AI model load
interpreter = tf.lite.Interpreter(model_path="D://Reasearch/Optical-type strain sensor/Code/converted_CVOS_model.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
input_shape = input_details[0]['shape']

# Data to extract
init_cord = []
init_pixel_list = pd.DataFrame()
init_point_list = pd.DataFrame()
box_area_list = pd.DataFrame()
aspect_ratio_list = pd.DataFrame()
average_pixel = []   
loading_unloading_state = pd.DataFrame()
init_cord_0to5 = []
center_point_contour_df_list = []
map_0to5 = []
map_0to5_TL = []
map_0to5_TR = []
map_0to5_BL = []
map_0to5_BR = []
initial_center_point_contour_df = []
directions = []

time_save = pd.DataFrame()
for tt in range(len(img_list)):
    start = time.time()
    file_name = img_list[tt].split('\\')[-1].split('.')[0]
    
    img = cv2.imread(img_list[tt]) 
    
    # Camera calibration
    img = cv2.undistort(img,mtx,dist,None,newcameraMtx)
    img = img[y_r:y_r+h_r,x_r:x_r+w_r]
    img = cv2.resize(img, dsize=(w, h), interpolation=cv2.INTER_AREA)    
    img = cv2.flip(img, 1)
    
    img = cv2.resize(img, dsize=(w, h), interpolation=cv2.INTER_AREA)    
    
    # Create rotation matrix
    rows, cols = img.shape[:2]
    M = cv2.getRotationMatrix2D((cols/2, rows/2), -180, 1)
    
    # Image rotation
    rotated = cv2.warpAffine(img, M, (cols, rows))
    img = cv2.resize(rotated, dsize=(w, h), interpolation=cv2.INTER_AREA)    
    
    # CLAHE algorithm
    img_clahe = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    img_clahe[:,:,0] = clahe.apply(img_clahe[:,:,0])
    img_clahe = cv2.cvtColor(img_clahe, cv2.COLOR_YUV2BGR)
    
    # Sharpness filter
    img_shap = cv2.filter2D(img_clahe, -1, sharpening_mask) 
    
    # GaussianBlur filter
    gray = cv2.cvtColor(img_shap, cv2.COLOR_BGR2RGB, cv2.CV_32F) 
    blur = cv2.GaussianBlur(gray,(11,11),2)
    
    # Boundary cutting
    blur[h-cut:h, 0:w] = 255
    blur[0:cut, 0:w] = 255
    blur[0:h, 0:cut] = 255
    blur[0:h, w-cut:w] = 255    
    
    # Image segmentation  
    gray = cv2.cvtColor(blur,cv2.COLOR_BGR2GRAY)
    ret, imthres = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU) #OTSU binary
    imthres = imthres.astype(np.uint8)    

    # Micro-markers finding process
    contour, hier = cv2.findContours(imthres, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    list_contour = list(contour)
    list_contour2 = []
    for i in range(len(list_contour)):
        if median_area*0.2 < cv2.contourArea(list_contour[i]) < median_area*5: list_contour2.append(list_contour[i]) 
        
    # Micro-markers drawing
    center_point_contour_list = []    
    for i in list_contour2:
        rect = cv2.minAreaRect(i)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        
        Xs = [i[0] for i in box]
        Ys = [i[1] for i in box]
        x1 = min(Xs)
        x2 = max(Xs)
        y1 = min(Ys)
        y2 = max(Ys)

        center_TRUE = (round((x1+x2)/2, 5), round((y1+y2)/2, 5))        
        
        cv2.circle(img, (int(center_TRUE[0]), int(center_TRUE[1])), 1, (0,255,0), -1)     
        cv2.drawContours(img, [box], 0, (0,255,0), 2) 
        center_point_contour_list.append(center_TRUE)
    center_point_contour_df = np.array(center_point_contour_list)
    
    # MOI(Micro-marker of interest) decision
    if tt == initialize_time : 
        cp_dist_list = []
        index_list = []
        init_cord = []        
        initial_center_point_contour_df = center_point_contour_df
        
        # Resetting the image center point (based on the refence center point)
        cp_dist_list0 = []
        index_list0 = []
        for i in range(len(center_point_contour_df)):
            cp_dist = Euclidian_distance(center_point_contour_df[i], ref_center_point)  
            cp_dist_list0.append(cp_dist)
            index_list0.append(i)                
        cp_dist_list0 = np.stack([np.array(cp_dist_list0), np.array(index_list0)], -1)     
        cp_dist_list0 = cp_dist_list0[cp_dist_list0[:,0].argsort()]        
        center_point = center_point_contour_df[int(cp_dist_list0[0][1])]
        
        # Calculate the distance between the resetted image center point and the center point of each contour and arrange 9 in order of proximity
        for i in range(len(center_point_contour_df)):
            cp_dist = Euclidian_distance(center_point_contour_df[i], center_point)            
            cp_dist_list.append(cp_dist)
            index_list.append(i)                
        cp_dist_list = np.stack([np.array(cp_dist_list), np.array(index_list)], -1)     
        cp_dist_list = cp_dist_list[cp_dist_list[:,0].argsort()][0:9]
        
        # Sort 9 points by min, max
        changed_index = []
        for i in range(0,9):
            changed_index.append(center_point_contour_df[int(cp_dist_list[i][1])])        
    
        refine_index_list = []
        for j in range(0,3):
            remaining_points_list = []
            remaining_points_list2 = []
            remaining_points_list3 = []            
            p_first = np.array(sorted(changed_index, key=lambda p: (p[0]) + (p[1])))[0]
            p_third = np.array(sorted(changed_index, key=lambda p: (p[0]) - (p[1])))[-1]
            p_second_ref = (p_first + p_third)/2
            
            changed_index_dist_list = []
            for i in changed_index:
                changed_index_dist = Euclidian_distance(i, p_second_ref)     
                changed_index_dist_list.append(changed_index_dist)
            p_second = changed_index[changed_index_dist_list.index(min(changed_index_dist_list))]
            
            for ii in range(len(changed_index)) :        
                if changed_index[ii][0] == p_first[0] and changed_index[ii][1] == p_first[1] :
                    refine_index_list.append(changed_index[ii])
                else :
                    remaining_points_list.append(changed_index[ii])
            
            for ii in range(len(remaining_points_list)) :               
                if remaining_points_list[ii][0] == p_second[0] and remaining_points_list[ii][1] == p_second[1] :
                    refine_index_list.append(remaining_points_list[ii])
                else :
                    remaining_points_list2.append(remaining_points_list[ii])    
                
            for ii in range(len(remaining_points_list2)) :          
                if remaining_points_list2[ii][0] == p_third[0] and remaining_points_list2[ii][1] == p_third[1] :
                    refine_index_list.append(remaining_points_list2[ii])    
                else :
                    remaining_points_list3.append(remaining_points_list2[ii])
                    
            changed_index = remaining_points_list3
        
        init_cord = np.array(refine_index_list)
        init_cord0 = init_cord.copy()
        
        # Curvature state detection algorithm
        voronoi_vol = voronoi_volumes(center_point_contour_df)
        hull = alphashape.alphashape(center_point_contour_list, 0.01)
        hull_pts = hull.exterior.coords.xy
        
        outter_points_index = []
        for i in range(len(hull_pts[0])):
            outter_points_index.append(center_point_contour_list.index((hull_pts[0][i], hull_pts[1][i])))
            
        inner_points_index = [i for i in set(range(0,len(center_point_contour_df))) if i not in outter_points_index]        
        inner_points_db = center_point_contour_df[inner_points_index]
        
        Average_voronoi = np.average(inner_points_db)
        Std_voronoi = np.std(inner_points_db)
        
        if Std_voronoi > curvature_std_threshold :
            Curvature_radius = -62.21*np.log(Average_voronoi) + 565.65
            Curvature_state_0 = 0
            Curvature_state_1 = 1
            print("Bending state")
        
        else :
            Curvature_radius = 100
            Curvature_state_0 = 1
            Curvature_state_1 = 0
            print("Linear state")              
        print("Initialize finish")    
    
    # MOI tracking
    if len(init_cord) != 0 :
        init_cord_t = init_cord.copy()
        init_cord_t0_flat = init_cord.flatten('F')
        init_cord_t_flat = init_cord_t.flatten('F')
        
        if len(out_index) > 0 :
            for i in range(len(out_index)):
                if out_index[i] < 9 :
                    init_cord_t_flat[out_index[i]] = init_cord_t0_flat[out_index[i]]
                    init_cord_t_flat[out_index[i]+9] = init_cord_t0_flat[out_index[i]+9]
                else : 
                    init_cord_t_flat[out_index[i]] = init_cord_t0_flat[out_index[i]]
                    init_cord_t_flat[out_index[i]-9] = init_cord_t0_flat[out_index[i]-9]
                    
            init_cord_t_flat.resize(2,9)
            init_cord_t = init_cord_t_flat.T            
            index_dist = get_index(center_point_contour_df, init_cord_t)         
        else :
            index_dist = get_index(center_point_contour_df, init_cord)         
            
        # Determining virtual MOI Algorithm operation
        out_index_x = [i for i, value in enumerate(init_cord[:,0]) if value > image_limit_x1 or value < image_limit_x0]
        out_index_y = [i for i, value in enumerate(init_cord[:,1]) if value > image_limit_y1 or value < image_limit_y0]
        out_index_y = list(np.array(out_index_y) + 9)
        out_index = list(set(out_index_x).difference(set(out_index_y))) + out_index_y
        out_index.sort()   
        for i in range(len(index_dist)):
            init_cord_t[i] = center_point_contour_df[index_dist[i]]               
            
        # Virtual MOI algorithm
        if len(out_index) > 0 :           
            init_cord_t_flat = init_cord_t.flatten('F')       
           
            for i in range(len(out_index)):
                init_cord_t_flat[out_index[i]] = init_cord_t0_flat[out_index[i]]
                if out_index[i] < 9 :
                 if init_cord_t_flat[out_index[i]] > image_limit_x1 :    
                     if 1 in out_index or 4 in out_index or 7 in out_index :
                         init_cord_t_flat[1] = (init_cord_t_flat[0] * init_cord_t0_flat[1])/init_cord_t0_flat[0]
                         init_cord_t_flat[4] = (init_cord_t_flat[3] * init_cord_t0_flat[4])/init_cord_t0_flat[3]
                         init_cord_t_flat[7] = (init_cord_t_flat[6] * init_cord_t0_flat[7])/init_cord_t0_flat[6]
                         
                         init_cord_t_flat[1+9] = (init_cord_t_flat[0+9] * init_cord_t0_flat[1+9])/init_cord_t0_flat[0+9]
                         init_cord_t_flat[4+9] = (init_cord_t_flat[3+9] * init_cord_t0_flat[4+9])/init_cord_t0_flat[3+9]
                         init_cord_t_flat[7+9] = (init_cord_t_flat[6+9] * init_cord_t0_flat[7+9])/init_cord_t0_flat[6+9]
                         
                     if 2 in out_index or 5 in out_index or 8 in out_index :
                         init_cord_t_flat[2] = (init_cord_t_flat[1] * init_cord_t0_flat[2])/init_cord_t0_flat[1]
                         init_cord_t_flat[5] = (init_cord_t_flat[4] * init_cord_t0_flat[5])/init_cord_t0_flat[4]
                         init_cord_t_flat[8] = (init_cord_t_flat[7] * init_cord_t0_flat[8])/init_cord_t0_flat[7]    
                         
                         init_cord_t_flat[2+9] = (init_cord_t_flat[1+9] * init_cord_t0_flat[2+9])/init_cord_t0_flat[1+9]
                         init_cord_t_flat[5+9] = (init_cord_t_flat[4+9] * init_cord_t0_flat[5+9])/init_cord_t0_flat[4+9]
                         init_cord_t_flat[8+9] = (init_cord_t_flat[7+9] * init_cord_t0_flat[8+9])/init_cord_t0_flat[7+9]    
                 
                 elif init_cord_t_flat[out_index[i]] < image_limit_x0 :
                     if 1 in out_index or 4 in out_index or 7 in out_index :
                         init_cord_t_flat[1] = (init_cord_t_flat[2] * init_cord_t0_flat[1])/init_cord_t0_flat[2]
                         init_cord_t_flat[4] = (init_cord_t_flat[5] * init_cord_t0_flat[4])/init_cord_t0_flat[5]
                         init_cord_t_flat[7] = (init_cord_t_flat[8] * init_cord_t0_flat[7])/init_cord_t0_flat[8]
                         
                         init_cord_t_flat[1+9] = (init_cord_t_flat[2+9] * init_cord_t0_flat[1+9])/init_cord_t0_flat[2+9]
                         init_cord_t_flat[4+9] = (init_cord_t_flat[5+9] * init_cord_t0_flat[4+9])/init_cord_t0_flat[5+9]
                         init_cord_t_flat[7+9] = (init_cord_t_flat[8+9] * init_cord_t0_flat[7+9])/init_cord_t0_flat[8+9]
                         
                     if 0 in out_index or 3 in out_index or 6 in out_index : 
                         init_cord_t_flat[0] = (init_cord_t_flat[1] * init_cord_t0_flat[0])/init_cord_t0_flat[1]
                         init_cord_t_flat[3] = (init_cord_t_flat[4] * init_cord_t0_flat[3])/init_cord_t0_flat[4]
                         init_cord_t_flat[6] = (init_cord_t_flat[7] * init_cord_t0_flat[6])/init_cord_t0_flat[7]       
                         
                         init_cord_t_flat[0+9] = (init_cord_t_flat[1+9] * init_cord_t0_flat[0+9])/init_cord_t0_flat[1+9]
                         init_cord_t_flat[3+9] = (init_cord_t_flat[4+9] * init_cord_t0_flat[3+9])/init_cord_t0_flat[4+9]
                         init_cord_t_flat[6+9] = (init_cord_t_flat[7+9] * init_cord_t0_flat[6+9])/init_cord_t0_flat[7+9]       
                else :
                 if init_cord_t_flat[out_index[i]] > image_limit_y1 :
                     if 12 in out_index or 13 in out_index or 14 in out_index :
                         init_cord_t_flat[3] = (init_cord_t_flat[0] * init_cord_t0_flat[3])/init_cord_t0_flat[0]
                         init_cord_t_flat[4] = (init_cord_t_flat[1] * init_cord_t0_flat[4])/init_cord_t0_flat[1]
                         init_cord_t_flat[5] = (init_cord_t_flat[2] * init_cord_t0_flat[5])/init_cord_t0_flat[2]
                         
                         init_cord_t_flat[12] = (init_cord_t_flat[9] * init_cord_t0_flat[12])/init_cord_t0_flat[9]
                         init_cord_t_flat[13] = (init_cord_t_flat[10] * init_cord_t0_flat[13])/init_cord_t0_flat[10]
                         init_cord_t_flat[14] = (init_cord_t_flat[11] * init_cord_t0_flat[14])/init_cord_t0_flat[11]
                                                    
                     if 15 in out_index or 16 in out_index or 17 in out_index :    
                         init_cord_t_flat[6] = (init_cord_t_flat[3] * init_cord_t0_flat[6])/init_cord_t0_flat[3]
                         init_cord_t_flat[7] = (init_cord_t_flat[4] * init_cord_t0_flat[7])/init_cord_t0_flat[4]
                         init_cord_t_flat[8] = (init_cord_t_flat[5] * init_cord_t0_flat[8])/init_cord_t0_flat[5]
                         
                         init_cord_t_flat[15] = (init_cord_t_flat[12] * init_cord_t0_flat[15])/init_cord_t0_flat[12]
                         init_cord_t_flat[16] = (init_cord_t_flat[13] * init_cord_t0_flat[16])/init_cord_t0_flat[13]
                         init_cord_t_flat[17] = (init_cord_t_flat[14] * init_cord_t0_flat[17])/init_cord_t0_flat[14]
                 
                 elif init_cord_t_flat[out_index[i]] < image_limit_y0 :
                     if 12 in out_index or 13 in out_index or 14 in out_index :
                         init_cord_t_flat[3] = (init_cord_t_flat[6] * init_cord_t0_flat[3])/init_cord_t0_flat[6]
                         init_cord_t_flat[4] = (init_cord_t_flat[7] * init_cord_t0_flat[4])/init_cord_t0_flat[7]
                         init_cord_t_flat[5] = (init_cord_t_flat[8] * init_cord_t0_flat[5])/init_cord_t0_flat[8]
                         
                         init_cord_t_flat[12] = (init_cord_t_flat[15] * init_cord_t0_flat[12])/init_cord_t0_flat[15]
                         init_cord_t_flat[13] = (init_cord_t_flat[16] * init_cord_t0_flat[13])/init_cord_t0_flat[16]
                         init_cord_t_flat[14] = (init_cord_t_flat[17] * init_cord_t0_flat[14])/init_cord_t0_flat[17]
                                                    
                     if 9 in out_index or 10 in out_index or 11 in out_index :                                  
                         init_cord_t_flat[0] = (init_cord_t_flat[3] * init_cord_t0_flat[0])/init_cord_t0_flat[3]
                         init_cord_t_flat[1] = (init_cord_t_flat[4] * init_cord_t0_flat[1])/init_cord_t0_flat[4]
                         init_cord_t_flat[2] = (init_cord_t_flat[5] * init_cord_t0_flat[2])/init_cord_t0_flat[5]
                         
                         init_cord_t_flat[9] = (init_cord_t_flat[12] * init_cord_t0_flat[9])/init_cord_t0_flat[12]
                         init_cord_t_flat[10] = (init_cord_t_flat[13] * init_cord_t0_flat[10])/init_cord_t0_flat[13]
                         init_cord_t_flat[11] = (init_cord_t_flat[14] * init_cord_t0_flat[11])/init_cord_t0_flat[14]
                    
            init_cord_t_flat.resize(2,9)
            print(init_cord_t_flat)   
            init_cord_t = init_cord_t_flat.T        
            index_dist = get_index(center_point_contour_df, init_cord_t) 
        
        # MOI visualization 
        box_area = []
        for i in range(len(index_dist)):
            rect = cv2.minAreaRect(list_contour2[index_dist[i]])
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            
            if i in out_index :
                pass
            else : 
                cv2.drawContours(img, [box], 0, (0,0,255), 2)
                cv2.putText(img, '%.d'%int(i), tuple(np.int0(center_point_contour_df[index_dist[i]])), cv2.FONT_HERSHEY_PLAIN, 2,(255,0,0),2)
            box_area.append(cv2.contourArea(list_contour2[index_dist[i]]))
 
        pixel_dist_X = []
        for i in range(len(init_cord_t)):
            pixel_dist_X.append(init_cord_t[i][0] - init_cord0[i][0])
            
        pixel_dist_Y = []
        for i in range(len(init_cord_t)):
            pixel_dist_Y.append(init_cord_t[i][1] - init_cord0[i][1])

        init_pixel_list = init_pixel_list.append(pd.Series(pixel_dist_X + pixel_dist_Y), ignore_index=True)
        init_point_list = init_point_list.append(pd.Series(list(np.concatenate([np.array(init_cord_t[:,0]).reshape(9), np.array(init_cord_t[:,1]).reshape(9)], axis=0))), ignore_index=True)
        box_area_list = box_area_list.append(pd.Series(box_area), ignore_index=True)
        
        # Loading - unloading state detection
        Loading_unloading_state_0 = 0
        Loading_unloading_state_1 = 1
        
        average_pixel.append(abs(np.average(init_pixel_list)))
        if len(average_pixel)>8 :
            del average_pixel[0]
        
        temp_gradient_db = []            
        if len(average_pixel) == 8 :
            for i in range(0,len(average_pixel)-2):
                temp_gradient_db.append(average_pixel[i+2] - average_pixel[i])
                
            if temp_gradient_db[1]<0 and temp_gradient_db[2]<0 and temp_gradient_db[3]<0 and temp_gradient_db[4]<0 and temp_gradient_db[5]<0 :
                loading_unloading_state = loading_unloading_state.append([0])
                Loading_unloading_state_0 = 1
                Loading_unloading_state_1 = 0
                print("Un-loading")
                
            else:
                loading_unloading_state = loading_unloading_state.append([1])
                Loading_unloading_state_0 = 0
                Loading_unloading_state_1 = 1
                print("loading")              
        
        init_cord = init_cord_t                
        
        init_cord_0to5.append(init_cord_t.copy())        
        if len(init_cord_0to5) > 5:
            init_cord_0to5 = init_cord_0to5[1:] 
            
        # CVOS AI model  
        CVOS_data = [Curvature_radius, Loading_unloading_state_0, Loading_unloading_state_1, Curvature_state_0, Curvature_state_1]
        input_data_X = np.array(pixel_dist_X+CVOS_data, dtype=np.float32).reshape(1,14)
        interpreter.set_tensor(input_details[0]['index'], input_data_X)
        interpreter.invoke()
        output_data_X = interpreter.get_tensor(output_details[0]['index'])
        
        input_data_Y = np.array(pixel_dist_Y+CVOS_data, dtype=np.float32).reshape(1,14)
        interpreter.set_tensor(input_details[0]['index'], input_data_Y)
        interpreter.invoke()
        output_data_Y = interpreter.get_tensor(output_details[0]['index'])
        
        # Results print
        print('Average of pixel change (X): %.2f'% (np.average(pixel_dist_X)))   
        print('Average of pixel change (Y): %.2f'% (np.average(pixel_dist_Y)))   
        print('Response of CVOS model (X): %.2f'% (output_data_X-1))   
        print('Response of CVOS model (Y): %.2f'% (output_data_Y-1))
                
        # Weighted results
        if len(init_cord_0to5) == 6 : #y=1.5*x^2
           init_cord = 0.075829384*init_cord_0to5[0] + 0.113744076*init_cord_0to5[1] + 0.170616114*init_cord_0to5[2] + 0.255924171*init_cord_0to5[3] + 0.383886256*init_cord_0to5[4]
    
    ## Strain mapping
    if len(initial_center_point_contour_df) > 1 :
        src_pts = initial_center_point_contour_df
        dst_pts = center_point_contour_df 
        
        # Dimension match
        if src_pts.shape != dst_pts.shape:
            min_len = min(len(src_pts), len(dst_pts))
            src_pts = src_pts[:min_len, :]
            dst_pts = dst_pts[:min_len, :]
                
        # Sort the corresponding points and make sure that adjacent points correspond to each other
        src_pts, dst_pts = find_corresponding_points(src_pts, dst_pts, frame_to_frame_threshold)
        
        # Calculate homography matrix with RANSAC method
        M, _ = cv2.findHomography(src_pts.reshape(-1,1,2), dst_pts.reshape(-1,1,2), cv2.RANSAC, RAN)
        
        # Initialize the maps
        strain_map2 = np.zeros((int(h/nn), int(w/nn)), np.float32)
        strain_direction_map2 = np.zeros((int(h/nn), int(w/nn)), np.float32)

        # Minimize unnecessary computation
        x_coords = np.arange(0, w+nn, nn)
        x_coords[len(x_coords)-1] = w-1
        y_coords = np.arange(0, h+nn, nn)
        y_coords[len(y_coords)-1] = h-1
        X, Y = np.meshgrid(x_coords, y_coords)
        ones = np.ones_like(X)
        
        # Transform the points
        pts = np.stack([X, Y, ones], axis=-1)
        pts_transformed = np.matmul(M, pts.reshape((-1, 3)).T)
        pts_transformed /= pts_transformed[2]
        pts_transformed = pts_transformed.reshape((3, int(h/nn)+1, int(w/nn)+1))
        
        # Calculate strain map and direction map
        dx = pts_transformed[0] - X
        dy = pts_transformed[1] - Y
        strain_map2 = np.sqrt(dx ** 2 + dy ** 2)
        theta = np.arctan2(dy, dx)
        strain_direction_map2 = theta        
        strain_map3 = np.array(strain_map2.flatten())
        p1 = np.percentile(strain_map3, 1)
        p99 = np.percentile(strain_map3, 99)
        strain_map3 = np.clip(strain_map3, p1, p99)
        strain_map3[strain_map3<p1] = 0            
        strain_map3 /= np.max(strain_map3)
        
        # Weighted average direction
        x = strain_map3*np.cos(strain_direction_map2.flatten())
        y = strain_map3*np.sin(strain_direction_map2.flatten())
        
        average_direction = np.degrees(np.arctan2(np.average(y),np.average(x)))
        top_left_average_direction = np.degrees(np.arctan2(np.average(y[q2]), np.average(x[q2])))
        top_right_average_direction = np.degrees(np.arctan2(np.average(y[q1]), np.average(x[q1])))
        bottom_left_average_direction = np.degrees(np.arctan2(np.average(y[q3]), np.average(x[q3])))
        bottom_right_average_direction = np.degrees(np.arctan2(np.average(y[q4]), np.average(x[q4])))
             
        # Weighted results
        map_0to5.append(average_direction)        
        map_0to5_TL.append(top_left_average_direction)        
        map_0to5_TR.append(top_right_average_direction)        
        map_0to5_BL.append(bottom_left_average_direction)        
        map_0to5_BR.append(bottom_right_average_direction)          
           
        if len(map_0to5) > 5 :
            map_0to5 = map_0to5[1:]
            average_direction = 0.075829384*map_0to5[0] + 0.113744076*map_0to5[1] + 0.170616114*map_0to5[2] + 0.255924171*map_0to5[3] + 0.383886256*map_0to5[4]
        
            map_0to5_TL = map_0to5_TL[1:]
            top_left_average_direction = 0.075829384*map_0to5_TL[0] + 0.113744076*map_0to5_TL[1] + 0.170616114*map_0to5_TL[2] + 0.255924171*map_0to5_TL[3] + 0.383886256*map_0to5_TL[4]
        
            map_0to5_TR = map_0to5_TR[1:]
            top_right_average_direction = 0.075829384*map_0to5_TR[0] + 0.113744076*map_0to5_TR[1] + 0.170616114*map_0to5_TR[2] + 0.255924171*map_0to5_TR[3] + 0.383886256*map_0to5_TR[4]
     
            map_0to5_BL = map_0to5_BL[1:]
            bottom_left_average_direction = 0.075829384*map_0to5_BL[0] + 0.113744076*map_0to5_BL[1] + 0.170616114*map_0to5_BL[2] + 0.255924171*map_0to5_BL[3] + 0.383886256*map_0to5_BL[4]
        
            map_0to5_BR = map_0to5_BR[1:]
            bottom_right_average_direction = 0.075829384*map_0to5_BR[0] + 0.113744076*map_0to5_BR[1] + 0.170616114*map_0to5_BR[2] + 0.255924171*map_0to5_BR[3] + 0.383886256*map_0to5_BR[4]
        
        # total directions save
        directions.append([average_direction, top_left_average_direction, top_right_average_direction, bottom_left_average_direction, bottom_right_average_direction])
        
        # strain alignment ratio (0: good / 100: bed)
        degree_map = np.degrees(np.arctan2(np.sin(strain_direction_map2.flatten()),np.cos(strain_direction_map2.flatten())))
        st_ratio = np.average(abs(average_direction - degree_map)/180)*100
        
        if np.average(strain_map2) < 1.5:
            average_direction = 0
            top_left_average_direction = 0
            top_right_average_direction = 0
            bottom_left_average_direction = 0
            bottom_right_average_direction = 0
            st_ratio = 0
            
        average_direction = -round(average_direction, 2)
        top_left_average_direction = -round(top_left_average_direction, 2)
        top_right_average_direction = -round(top_right_average_direction, 2)
        bottom_left_average_direction = -round(bottom_left_average_direction, 2)
        bottom_right_average_direction = -round(bottom_right_average_direction, 2)
        st_ratio = round(st_ratio, 2)
                   
        print('Main strain direction:', average_direction)
        print('Strain alignment ratio :', st_ratio)
        print("Top left average direction:", top_left_average_direction)
        print("Top right average direction:", top_right_average_direction)
        print("Bottom left average direction:", bottom_left_average_direction)
        print("Bottom right average direction:", bottom_right_average_direction)
        
        # Draw strain direction arrows on the image
        for x, y, dx, dy in zip(X.flatten(), Y.flatten(), dx.flatten(), dy.flatten()):
            cv2.arrowedLine(img, (int(x), int(y)), (int(x+dx), int(y+dy)), color=(255,0,0), thickness=3, tipLength=0.3)
    
    plt.axis('off')
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    #plt.show()
    plt.savefig(save_path + '/' + file_name + '_OTSS_v5.2.jpg',dpi=300, transparent=True, bbox_inches='tight')
    plt.clf()
    plt.close()  
    end = time.time() 
    
    time_save = time_save.append([end - start])
    print("end time " + f"{end - start:.5f} sec")
    
init_pixel_list.to_csv(path.split('/')[-2] + '_init_pixel_OTSS_v5.2.csv')
init_point_list.to_csv(path.split('/')[-2] + 'init_point_OTSS_v5.2.csv')
box_area_list.to_csv(path.split('/')[-2] + 'box_area_OTSS_v5.2.csv')
pd.DataFrame(directions).to_csv(path.split('/')[-2] + 'directions_v5.2.csv')
plt.plot(init_point_list.iloc[:,0:9])
plt.plot(init_point_list.iloc[:,9:18])