# -*- coding: utf-8 -*-
"""
Created on Mon Jun 24 21:18:46 2019

@author: Pavilion
"""
import os
import cv2
import numpy as np
import struct
import sys

sys.path.append('../')
from base import base
from util import util

def load(data,path,imgend,num_ide,num_exp,num_land):
    
    for root,dirs,name_file in os.walk(path):
        if (len(name_file)>0):
            flag=1
            print(root,name_file)
            for name_file_land73 in name_file:                    
                print('file name: ', name_file_land73)
                if name_file_land73.endswith('.land73'):
                    
                    name_file_land73=root+'/'+name_file_land73
                    name_file_img=name_file_land73[:-6]+imgend
                    name_file_psp_f=name_file_land73[:-6]+'psp_f'
                    if not(os.path.exists(name_file_img) and os.path.exists(name_file_psp_f)):
                        continue;
                    
                    
                    new_data=base.DataOneImg()
                    new_data.file_name=name_file_land73[:-6]
                    
                    load_img(name_file_img,new_data)
                    load_land73(name_file_land73,new_data,num_land)
                    
                    user=np.empty((num_ide,),dtype=np.float64)
                    load_psp_f(name_file_psp_f,new_data,user,num_ide,num_exp,num_land)
#                    print('user when load', user)
                    if flag==1:
                        flag=0
                        data.append(base.DataOneIdentity())
                        data[-1].user=user
                        data[-1].dir_name=root
                    
                    data[-1].data.append(new_data)
                                            
                        
                    
def load_img(name,data):
    print('load image image name:  ',name)
    data.img=cv2.imread(name,cv2.IMREAD_GRAYSCALE)

def load_land73(name,data,num_land):
    print('load land73 filename:  ',name)
    with open(name) as f:
        line=f.readline()
        line=line.strip()
        assert(int(line)==num_land)
        data.land=np.empty((num_land,2),dtype=np.float64)
        for i in range(num_land):
            line=f.readline().strip().split()
            
            data.land[i,0]=float(line[0])
            data.land[i,1]=float(line[1])
#            print(line,data.land[i,0],data.land[i,1])
            data.land[i,1]=data.img.shape[0]-data.land[i,1]
        
def load_psp_f(name,data,user,num_ide,num_exp,num_land):
    print('load psp_f filename:  ',name)
    with open(name,'rb') as f:        
        for i in range(num_ide):
            user[i],=struct.unpack('f',f.read(4))
#        print('user when load one', user)
        
        data.land_inv=np.empty((num_land,2),dtype=np.float64)
        for i in range(num_land):
            for j in range(2):
                data.land_inv[i,j],=struct.unpack('f',f.read(4))
              
        data.center=np.empty((2,),dtype=np.float64)
        for i in range(2):
            data.center[i],=struct.unpack('f',f.read(4))
        
        data.exp=np.empty((num_exp,),dtype=np.float64)
        for i in range(num_exp):
            data.exp[i],=struct.unpack('f',f.read(4))
        data.exp[0]=1
            
        data.rot=np.empty((3,3),dtype=np.float64)
        for i in range(3):
            for j in range(3):
                data.rot[i,j],=struct.unpack('f',f.read(4))
        #angle!!
        data.angle=util.matrix2uler_angle_zyx(data.rot)
        
        data.tslt=np.zeros((3,),dtype=np.float64)
        for i in range(3):
            data.tslt[i],=struct.unpack('f',f.read(4))
        
        data.land_cor=np.empty((num_land,),dtype=np.int)
        for i in range(num_land):
            data.land_cor[i],=struct.unpack('i',f.read(4))
                
        data.fcs,=struct.unpack('f',f.read(4))
        
        data.dis=np.empty((num_land,2),dtype=np.float64)
        for i in range(num_land):
            for j in range(2):
                data.dis[i,j],=struct.unpack('f',f.read(4))        
    print('load successful one')
#    print('user when load one', user)

def save_psp_f(name,data,user):
    print('save psp_f filename:  ',name)
    with open(name,'wb') as f:        
        for x in user:
            f.write(struct.pack('f',np.float32(x)))
            
        for x in data.land_inv:
            f.write(struct.pack('ff',np.float32(x[0]),np.float32(x[1])))        
              
        f.write(struct.pack('ff',np.float32(data.center[0]),np.float32(data.center[1])))        
                
        for x in data.exp:
            f.write(struct.pack('f',np.float32(x)))
                    
        rot=util.angle2matrix_zyx(data.angle)
        for i in range(3):
            for j in range(3):
                f.write(struct.pack('f',np.float32(rot[i,j])))
                
        for x in data.tslt:
            f.write(struct.pack('f',np.float32(x)))
        
        for x in data.land_cor:
            f.write(struct.pack('f',np.int32(x)))        
                
        f.write(struct.pack('f',np.float32(data.fcs)))        
                
        for x in data.dis:
            for y in x:
                f.write(struct.pack('f',np.float32(y)))
    print('save successful one')



def load_bldshps(name,num_ide,num_exp,num_vtx):
    print('loading blendshapes  :',name)
    bldshps=np.empty((num_ide,num_exp,num_vtx,3),dtype=np.float64)
    with open(name,'rb') as f:
        for i in range(num_ide):
            for j in range(num_exp):
                for k in range(num_vtx):
                    for x in range(3):
                        bldshps[i,j,k,x],=struct.unpack('f',f.read(4))
                        
    return bldshps
        
def load_tri_idx(name,num_land):
        
    with open(name,'r') as f:
        line=f.readline()
        line=line.strip()
        assert(int(line)==num_land)
        
        mean_ldmk=np.empty((num_land,2),dtype=np.float64)
        for i in range(num_land):
            line=f.readline().strip().split()
            mean_ldmk[i,0]=float(line[0])
            mean_ldmk[i,1]=float(line[1])
        
        line=f.readline()
        line=line.strip()
        tri_idx_num=int(line)
        tri_idx=np.empty((tri_idx_num,3),dtype=np.int)
        for i in range(tri_idx_num):
            line=f.readline().strip().split()
            for j in range(3):                
                tri_idx[i,j]=int(line[j])
        
        line=f.readline()
        line=line.strip()
        px_barycenter_num=int(line)        
        px_barycenter\
        =[np.empty((px_barycenter_num,),dtype=np.int), np.empty((px_barycenter_num,3),dtype=np.float64)]
        for i in range(px_barycenter_num):
            line=f.readline().strip().split()
            px_barycenter[0][i]=int(line[0])
            px_barycenter[1][i,0]=float(line[1])
            px_barycenter[1][i,1]=float(line[2])
            px_barycenter[1][i,2]=1-px_barycenter[1][i,0]-px_barycenter[1][i,1]            
    
    return mean_ldmk,tri_idx,px_barycenter
        
def load_img_230(name,data):
    print('load image image name:  ',name)
    data.img_230=cv2.imread(name)
# =============================================================================
#     cv2.imshow('test image',data.img_230)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
# =============================================================================

def load_cnn(data_cnn,path,imgend,num_ide,num_exp,num_land):
    
    for root,dirs,name_file in os.walk(path):
        if (len(name_file)>0):
            flag=1
            print(root,name_file)
            for name_file_land73 in name_file:                    
                print('file name: ', name_file_land73)
                if name_file_land73.endswith('.land73'):
                    
                    name_file_land73=root+'/'+name_file_land73
                    name_file_img=name_file_land73[:-6]+imgend
                    name_file_img_230=name_file_land73[:-7]+'_230.'+imgend
                    name_file_psp_f=name_file_land73[:-6]+'psp_f'
                    if not(os.path.exists(name_file_img) and os.path.exists(name_file_psp_f)):
                        continue;
                    
                    
                    new_data=base.DataOneImg()
                    new_data.file_name=name_file_land73[:-6]
                    
                    load_img(name_file_img,new_data)
                    load_img_230(name_file_img_230,new_data)
                    load_land73(name_file_land73,new_data,num_land)
                    
                    user=np.empty((num_ide,),dtype=np.float64)
                    load_psp_f(name_file_psp_f,new_data,user,num_ide,num_exp,num_land)
#                    print('user when load', user)
                    if flag==1:
                        flag=0
                        data_cnn.append(base.DataOneIdentity())
                        data_cnn[-1].user=user
                        data_cnn[-1].dir_name=root
                    
                    data_cnn[-1].data.append(new_data)        
        
def load_dataHEbldshps():
    data=[]
#    fwhs_path='/home/weiliu/fitting_dde/fitting_psp_f_l12_slt/fw'
#    lfw_path='/home/weiliu/fitting_dde/fitting_psp_f_l12_slt/lfw_image'
#    gtav_path='/home/weiliu/fitting_dde/fitting_psp_f_l12_slt/GTAV_image'
    fwhs_path='/data/weiliu/fitting_psp_f_l12_slt/FaceWarehouse'
    lfw_path='/data/weiliu/fitting_psp_f_l12_slt/lfw_image'
    gtav_path='/data/weiliu/fitting_psp_f_l12_slt/GTAV_image'
    
    load(data,fwhs_path,'jpg',num_ide=77,num_exp=47,num_land=73)
    load(data,lfw_path,'jpg',num_ide=77,num_exp=47,num_land=73)
    load(data,gtav_path,'bmp',num_ide=77,num_exp=47,num_land=73) 
    
#    test_path='/home/weiliu/fitting_dde/4_psp_f_cal_test/data_me/test_only_three/'
#    load.load(data,test_path,'jpg',num_ide=77,num_exp=47,num_land=73)
    
    # =============================================================================
    # for one_ide in data:
    #     for one_img in one_ide.data:
    #         print(one_img.land_cor)
    # =============================================================================

    
#    bldshps_path='/home/weiliu/fitting_dde/const_file/deal_data/blendshape_ide_svd_77.lv'
    bldshps_path='/data/weiliu/fitting_psp_f_l12_slt/const_file/blendshape_ide_svd_77.lv'    
    bldshps=load_bldshps(bldshps_path,num_ide=77,num_exp=47,num_vtx=11510)
    util.bld_reduce_neu(bldshps)
    return data,bldshps
    
        
def test_load():
    #test:
    data=[]
    fwhs_path='/home/weiliu/fitting_dde/fitting_psp_f_l12_slt/fw'
    lfw_path='/home/weiliu/fitting_dde/fitting_psp_f_l12_slt/lfw_image'
    gtav_path='/home/weiliu/fitting_dde/fitting_psp_f_l12_slt/GTAV_image'
    test_path='/home/weiliu/fitting_dde/4_psp_f_cal_test/data_me/test_only_three/'
    

    # =============================================================================
    # load(data,fwhs_path,'jpg',num_ide=77,num_exp=47,num_land=73)     
    # =============================================================================
    load(data,test_path,'jpg',num_ide=77,num_exp=47,num_land=73)
    
    for x in data:
        print(x.dir_name)
        print(x.user)
        for y in x.data:
            print(y.file_name)
    #        print(y.dis)
    
    bldshps_path='/home/weiliu/fitting_dde/const_file/deal_data/blendshape_ide_svd_77.lv'    
#   bldshps=load_bldshps(bldshps_path,num_ide=77,num_exp=47,num_vtx=11510)


    mean_ldmk,tri_idx,px_barycenter=load_tri_idx('../const_file/tri_idx_px.txt',73)
    print(mean_ldmk)
    print(tri_idx)
    print(px_barycenter)
    
    
    
#test_load()