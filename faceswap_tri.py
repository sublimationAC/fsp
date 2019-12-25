# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 15:59:02 2019

@author: Pavilion
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Oct  9 10:36:37 2019

@author: Pavilion
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 15:28:06 2019

@author: Pavilion
"""

NOW_DEAL='out'#small
DENSE=True#False#

import cv2
import numpy as np
from tri_rasterization import tri_rasterization as tras
from base import base
from load import load
from util import util
import time


def warp_image_cv_mask(img, c_src, c_dst,use_tri_idx,mask):
   
    st_t=time.time()
    
    c_src=np.around(c_src).astype(np.int)    
    
    pt_color=img[(c_src[:,1],c_src[:,0])].copy()
    
#    ans=tras.render_colors(c_dst, use_tri_idx, pt_color, img.shape[0],img.shape[1])
    ans=tras.render_colors_tri_me(img,mask,c_src,c_dst, use_tri_idx, img.shape[0],img.shape[1])
    
    print('render_colors time: ',time.time()-st_t,'s')
    

    return ans

def txtf_image_cv_mask(sc_img,dst_image, c_src, c_dst,use_tri_idx,mask):
   
    st_t=time.time()
    
    c_src=np.around(c_src).astype(np.int)    
    
    pt_color=sc_img[(c_src[:,1],c_src[:,0])].copy()
    
#    ans=tras.render_colors(c_dst, use_tri_idx, pt_color, sc_img.shape[0],sc_img.shape[1])
    ans=tras.render_colors_tri_me(sc_img,mask,c_src,c_dst, use_tri_idx, sc_img.shape[0],sc_img.shape[1])
    
    res=dst_image.copy()
    res[np.where(mask==[255,255,255])[:2]]=ans[np.where(mask==[255,255,255])[:2]]
    
    print('render_colors time: ',time.time()-st_t,'s')
    

    return res


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


def load_over_01_idx(path):
    over_01_idx=[]
    print('load over_01_idx filename:  ',path)
    with open(path) as f:
        line=f.readline()
        line=line.strip()
        print(line)
        num=int(line)
        over_01_idx=np.empty((num,),dtype=np.int)
        for i in range(num):
            line=f.readline().strip().split()
            
            over_01_idx[i]=int(line[0])            
    return over_01_idx

def load_triangle_index(path):
    all_tri=[]
    print('load triangle index filename:  ',path)
    with open(path) as f:
        line=f.readline()
        line=line.strip()
        print(line)
        num=int(line)
        all_tri=np.empty((num,3),dtype=np.int)
        for i in range(num):
            line=f.readline().strip().split()
            
            all_tri[i,0]=int(line[0])
            all_tri[i,1]=int(line[1])
            all_tri[i,2]=int(line[2])
                        
    return all_tri

def elit_tri(all_tri,over_01_idx):
    elit_tri_idx=[]
    for x in all_tri:
        flag=0
        for y in x:
            if not (y in over_01_idx):
                flag=1
                break
        if (flag):
            continue
        elit_tri_idx.append(x)
    return np.array(elit_tri_idx)



def get_face_mask(image_size, face_landmarks):
    
#    mask = np.zeros(image_size, dtype=np.uint8)
    points = np.concatenate([face_landmarks[0:18], face_landmarks[23:20:-1]])
#    cv2.fillPoly(img=mask, pts=[points], color=255)
#    print(points)
    
    mask = np.zeros(image_size, dtype=np.uint8)
#    print(face_landmarks)
    face_landmarks=np.around(face_landmarks).astype(np.int)
    points = cv2.convexHull(face_landmarks)  # 凸包
    points=np.squeeze(points)
#    print(points)
    cv2.fillConvexPoly(mask, points, color=[255,255,255])
    return mask

def get_face_hole(face_landmarks,img_yuan):
    sc_img_hole = np.copy(img_yuan)
    face_landmarks=np.around(face_landmarks).astype(np.int)
    points = cv2.convexHull(face_landmarks)  # 凸包
    points=np.squeeze(points)
#    print(points)
    cv2.fillConvexPoly(sc_img_hole, points, color=[255,255,255])
    return sc_img_hole

def get_hull_center(face_landmarks):
    face_landmarks=np.around(face_landmarks).astype(np.int)
    points = cv2.convexHull(face_landmarks)  # 凸包
    points=np.squeeze(points)
    center=np.around(points.mean(axis=0)).astype(np.int)    
    return center

def cal_mesh_spfbldshps_d(data,spf_bldshps):        
    data.exp[0]=1
    landmk_3d=np.tensordot(spf_bldshps,data.exp,axes=(0,0))
#    print('angle:',data.angle)
#    print('tslt:',data.tslt)
    landmk_3d=(util.angle2matrix_zyx(data.angle)@landmk_3d.T).T+data.tslt
#    landmk_3d_=(data.rot@landmk_3d.T).T+data.tslt
#    print(np.concatenate((landmk_3d, landmk_3d_), axis=1))
    
    
    landmk_3d[:,0]/=landmk_3d[:,2]
    landmk_3d[:,1]/=landmk_3d[:,2]
#    print('fcs:',data.fcs)
    landmk_3d*=data.fcs
#    print('center:',data.center)
    landmk_3d[:,0:2]+=data.center
    landmk_3d[:,1]=data.img.shape[0]-landmk_3d[:,1]
        
    
#    print(np.concatenate((ans, data.land), axis=1))    
    
    return landmk_3d

image_yuan_path="pose_0."
img_yuan=cv2.imread(image_yuan_path+"jpg")


#prefix="../lv_small"
if (NOW_DEAL=='out'):    
    prefix="../lv_out"
    video_path="lv_out.mp4"
    
    res_psp_f_path='/home/weiliu/psp_f2obj/lv_out_np_dem_9_24_f4_nst3d_init3d_ddesltnorm/lv_out_'
else:        
    video_path="lv_small_big.mp4"
    prefix="../lv_small"
    res_psp_f_path=''

cam_video_yuan = cv2.VideoCapture(video_path)


sc_data_land=load_land73(image_yuan_path+"land73",73,img_yuan)
exp_center=sc_data_land.mean(0).astype(np.int)

tg_data_center=load_center(prefix+"_center.txt")

sc_mask=get_face_mask(img_yuan.shape,sc_data_land)

sc_img_hole=get_face_hole(sc_data_land,img_yuan)

fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
if DENSE:
    out = cv2.VideoWriter(prefix+'_triras_dense_me_transfer_allcenter.mp4',fourcc, 20.0, (640*3,480*3),1) 
else:        
    out = cv2.VideoWriter(prefix+'_triras_sp.mp4',fourcc, 20.0, (640*2,480*2),1) 

#image_white=np.empty([480,640,3])
#image_white[...,:]=np.array([174,199,33])
#
#cv2.imshow('normal_image_wt',image_white)
#cv2.waitKey()
#
#image_white[...,:]=np.array([135,56,21])
#
#cv2.imshow('normal_image_wt',image_white)
#cv2.waitKey()
#
#image_white[...,:]=np.array([35,56,171])
#
#cv2.imshow('normal_image_wt',image_white)
#cv2.waitKey()


num_ide=77
num_exp=47
num_land=73

bldshps_path='/home/weiliu/fitting_dde/const_file/deal_data/blendshape_ide_svd_77.lv'
bldshps=load.load_bldshps(bldshps_path,num_ide=77,num_exp=47,num_vtx=11510)
util.bld_reduce_neu(bldshps)

new_data=base.DataOneImg()
user=np.empty((num_ide,),dtype=np.float64)

psp_name='pose_0.psp_f'
sc_data=base.DataOneImg()
sc_data.file_name=psp_name
sc_data.img=img_yuan.copy()
load.load_psp_f(psp_name,sc_data,user,num_ide,num_exp,num_land)
sc_spf_bldshps=np.tensordot(bldshps,user,axes=(0,0))
#sc_data.exp[0]=1
#land=util.get_land_spfbldshps_inv(sc_data,sc_spf_bldshps)    
#print('shape:')
#print(sc_data.img.shape)
##land[:,1]=sc_data.img.shape[0]-land[:,1] 
#print('----------------------')
#print(land)
#print('++++++++++++++++++++++++++++++++++')
#print(sc_data.land_inv)
#print('++++++++++++++++++++++++++++++++++')




psp_name=res_psp_f_path+str(1)+'.psp_f'
new_data=base.DataOneImg()
load.load_psp_f(psp_name,new_data,user,num_ide,num_exp,num_land)
vd_spf_bldshps=np.tensordot(bldshps,user,axes=(0,0))
#land=util.get_land_spfbldshps_inv(new_data,vd_spf_bldshps)    
#
#print('---------------------pppppppppppppppp')
#print(land)
#print('+++++++++++++++++++++wwwwwwwwwwwwwwww')

over_01_idx=load_over_01_idx('over01_idx.txt')
if (DENSE):
    sc_mesh_d=cal_mesh_spfbldshps_d(sc_data,sc_spf_bldshps)
#    sc_dense_kpt_d=sc_mesh_d[over_01_idx,...]

all_tri_idx=load_triangle_index('fw_face_tri_idx.txt')
use_tri_idx=elit_tri(all_tri_idx,over_01_idx)

print(use_tri_idx)

#util.show_norm_img(sc_data,vd_spf_bldshps,over_01_idx,use_tri_idx)

frame_idx=-1
for ct in tg_data_center:
        
    ret_val, fm_yuan = cam_video_yuan.read()
    print(frame_idx,ret_val)
    if ret_val==0 :
        break
    
#    if (DENSE):
#        for t in range(10):        
#            ret_val, fm_yuan = cam_video_yuan.read()
#            print(frame_idx,ret_val)
#            frame_idx+=1
#            if ret_val==0 :
#                break
    
    frame_idx+=1
    if (frame_idx==0):
        continue
#    if (frame_idx<550):
#        continue
    
#    if (frame_idx>20):
#        break
    
    psp_name=res_psp_f_path+str(frame_idx)+'.psp_f'
    new_data=base.DataOneImg()
    new_data.file_name=psp_name
    new_data.img=fm_yuan.copy()
    
    user=np.empty((num_ide,),dtype=np.float64)
    
    load.load_psp_f(psp_name,new_data,user,num_ide,num_exp,num_land)
    new_data.exp[0]=1
    land=util.get_land_spfbldshps_inv(new_data,vd_spf_bldshps)    
    land[:,1]=fm_yuan.shape[0]-land[:,1]    
    tg_now_mask=get_face_mask(fm_yuan.shape,land)
    tg_img_hole=get_face_hole(land,new_data.img)
    tg_now_center=np.around(land.mean(axis=0)).astype(np.int)  #get_hull_center(land)
    
    util.draw_land(land,fm_yuan)
    
    sc_data.exp=new_data.exp
    sc_data.angle=new_data.angle
    sc_data.rot=new_data.rot
#    sc_data.land_cor=new_data.land_cor
    
    sc_data.exp[0]=1
    land=util.get_land_spfbldshps_inv(sc_data,sc_spf_bldshps)    
    land-=sc_data.dis
    land[:,1]=sc_data.img.shape[0]-land[:,1]    
    
    if (DENSE):
        ans=np.zeros((480*3,640*3,3)).astype('uint8')
    else:
        ans=np.zeros((480*2,640*2,3)).astype('uint8')
    
    ans[:480,0:640,:]=img_yuan
    ans[480:480*2,0:640,:]=fm_yuan

    now_mask=get_face_mask(img_yuan.shape,land)
    
#    wp_img=warp_image_cv(sc_data.img,sc_data_land,land,dshape=(480, 640))    
    if (DENSE):
        now_mesh_d=cal_mesh_spfbldshps_d(sc_data,sc_spf_bldshps)
#        now_dense_kpt_d=now_mesh_d[over_01_idx,...]
        wp_img=warp_image_cv_mask(
                sc_data.img,sc_mesh_d,now_mesh_d,use_tri_idx,
                sc_mask)
        
    else:
        wp_img=warp_image_cv_mask(sc_data.img,sc_data_land,land,sc_mask)    
        
    util.draw_land(land,wp_img)
    ans[:480,640:640*2,:]=wp_img
    result=cv2.copyTo(wp_img,now_mask)
    util.draw_land(land,result)
    ans[480:480*2,640:640*2,:]=result
    
    
#    seamless_exp = cv2.seamlessClone(fm_exp, sc_img_hole, mask=sc_mask, p=tuple(exp_center), flags=cv2.MONOCHROME_TRANSFER  )  # 进行泊松融合
#    seamless_res = cv2.seamlessClone(fm_exp, fm_yuan, mask=sc_mask, p=tuple(ct.astype(np.int)), flags=cv2.MONOCHROME_TRANSFER  )  # 进行泊松融合
#    
    if (DENSE):
        temp=sc_img_hole.copy()
        util.draw_pt_img(sc_mesh_d[over_01_idx,:2],temp,(0,0,0))    
        ans[:480,640*2:,:]=temp
        
        temp=sc_img_hole.copy()
        util.draw_pt_img(now_mesh_d[over_01_idx,:2],temp,(0,0,0))    
        ans[480:480*2,640*2:,:]=temp
    
    tg_now_mesh_d=cal_mesh_spfbldshps_d(new_data,sc_spf_bldshps)
    txtf_img=txtf_image_cv_mask(
            
                sc_data.img,new_data.img,sc_mesh_d,tg_now_mesh_d,use_tri_idx,
                tg_now_mask)
    
    
    seamless_tf = cv2.seamlessClone(txtf_img, new_data.img, mask=tg_now_mask, p=tuple(tg_now_center), flags=cv2.MONOCHROME_TRANSFER  )  # 进行泊松融合
#    print(seamless_tf.shape)
#    print(ans[480*2:480*3,0:640,:])
    ans[480*2:480*3,0:640,:]=seamless_tf
    result=cv2.copyTo(txtf_img,tg_now_mask)    
    ans[480*2:480*3,640:640*2,:]=result
    ans[480*2:480*3,640*2:640*3,:]=txtf_img
        
    out.write(ans)

cam_video_yuan.release()
out.release()

cv2.destroyAllWindows()



