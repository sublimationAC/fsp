# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 15:28:06 2019

@author: Pavilion
"""

import cv2
import numpy as np


def load_land73(name,num_land,img):
    
    land=[]
    print('load land73 filename:  ',name)
    with open(name) as f:
        line=f.readline()
        line=line.strip()
        assert(int(line)==num_land)
        land=np.empty((num_land,2),dtype=np.int)
        for i in range(num_land):
            line=f.readline().strip().split()
            
            land[i,0]=int(float(line[0]))
            land[i,1]=int(float(line[1]))
#            print(line,data.land[i,0],data.land[i,1])
            land[i,1]=img.shape[0]-land[i,1]
            
    return land

def load_center(name):
    center=[]
    print('load land73 filename:  ',name)
    with open(name) as f:
        line=f.readline()
        line=line.strip()
        num=int(line)
        center=np.empty((num,2),dtype=np.float64)
        for i in range(num):
            line=f.readline().strip().split()
            
            center[i,0]=float(line[0])
            center[i,1]=float(line[1])

            
    return center

def get_face_mask(image_size, face_landmarks):
    
#    mask = np.zeros(image_size, dtype=np.uint8)
    points = np.concatenate([face_landmarks[0:18], face_landmarks[23:20:-1]])
#    cv2.fillPoly(img=mask, pts=[points], color=255)
    print(points)
    
    mask = np.zeros(image_size, dtype=np.uint8)
    points = cv2.convexHull(face_landmarks)  # 凸包
    points=np.squeeze(points)
    print(points)
    cv2.fillConvexPoly(mask, points, color=[255,255,255])
    return mask

def get_face_hole(face_landmarks,img_yuan):
    sc_img_hole = np.copy(img_yuan)
    points = cv2.convexHull(face_landmarks)  # 凸包
    points=np.squeeze(points)
#    print(points)
    cv2.fillConvexPoly(sc_img_hole, points, color=[255,255,255])
    return sc_img_hole

image_yuan_path="pose_0."
img_yuan=cv2.imread(image_yuan_path+"jpg")

#prefix="../lv_small"
prefix="../lv_out"
video_path="lv_small_big.mp4"
video_path="lv_out.mp4"
cam_video_yuan = cv2.VideoCapture(video_path)
video_path_exp=prefix+"_exp.avi"
cam_video_path_exp = cv2.VideoCapture(video_path_exp)
video_path_change=prefix+"_result.avi"
cam_video_path_change = cv2.VideoCapture(video_path_change)

sc_data_land=load_land73(image_yuan_path+"land73",73,img_yuan)
exp_center=sc_data_land.mean(0).astype(np.int)

tg_data_center=load_center(prefix+"_center.txt")

sc_mask=get_face_mask(img_yuan.shape,sc_data_land)

sc_img_hole=get_face_hole(sc_data_land,img_yuan)

fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
out = cv2.VideoWriter(prefix+'_final_MONOCHROME_TRANSFER.mp4',fourcc, 20.0, (640*3,480*2),1) 



for ct in tg_data_center:
    ret_val, fm_yuan = cam_video_yuan.read()
    if ret_val==0 :
        break
    ret_val, fm_exp = cam_video_path_exp.read()
    if ret_val==0 :
        break
    ret_val, fm_change = cam_video_path_change.read()
    if ret_val==0 :
        break
        
    ans=np.zeros((480*2,640*3,3)).astype('uint8')
    ans[:480,0:640,:]=img_yuan
    ans[480:,0:640,:]=fm_yuan
    ans[:480,640:640*2,:]=fm_exp
    ans[480:,640:640*2:,:]=fm_change
    
    
    seamless_exp = cv2.seamlessClone(fm_exp, sc_img_hole, mask=sc_mask, p=tuple(exp_center), flags=cv2.MONOCHROME_TRANSFER  )  # 进行泊松融合
    seamless_res = cv2.seamlessClone(fm_exp, fm_yuan, mask=sc_mask, p=tuple(ct.astype(np.int)), flags=cv2.MONOCHROME_TRANSFER  )  # 进行泊松融合
    
    ans[:480,640*2:,:]=seamless_exp
    ans[480:,640*2::,:]=seamless_res
    
    out.write(ans)

cam_video_yuan.release()
out.release()
cam_video_path_exp.release()
cam_video_path_change.release()
cv2.destroyAllWindows()



