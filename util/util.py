# -*- coding: utf-8 -*-
"""
Created on Fri Jun 28 14:40:45 2019

@author: Pavilion
"""

import numpy as np
from math import cos, sin, acos, asin, fabs, sqrt
import cv2

def angle2matrix(angles):
    ''' get rotation matrix from three rotation angles(degree). right-handed.
    Args:
        angles: [3,]. x, y, z angles
        x: pitch. positive for looking down.
        y: yaw. positive for looking left. 
        z: roll. positive for tilting head right. 
    Returns:
        R: [3, 3]. rotation matrix.
    '''
    x, y, z = np.deg2rad(angles[0]), np.deg2rad(angles[1]), np.deg2rad(angles[2])
    # x
    Rx=np.array([[1,      0,       0],
                 [0, cos(x),  -sin(x)],
                 [0, sin(x),   cos(x)]])
    # y
    Ry=np.array([[ cos(y), 0, sin(y)],
                 [      0, 1,      0],
                 [-sin(y), 0, cos(y)]])
    # z
    Rz=np.array([[cos(z), -sin(z), 0],
                 [sin(z),  cos(z), 0],
                 [     0,       0, 1]])
    
    R=Rz.dot(Ry.dot(Rx))
    return R.astype(np.float32)

def angle2matrix_3ddfa(angles):
    ''' get rotation matrix from three rotation angles(radian). The same as in 3DDFA.
    Args:
        angles: [3,]. x, y, z angles
        x: pitch.
        y: yaw. 
        z: roll. 
    Returns:
        R: 3x3. rotation matrix.
    '''
    # x, y, z = np.deg2rad(angles[0]), np.deg2rad(angles[1]), np.deg2rad(angles[2])
    x, y, z = angles[0], angles[1], angles[2]
    
    # x
    Rx=np.array([[1,      0,       0],
                 [0, cos(x),  sin(x)],
                 [0, -sin(x),   cos(x)]])
    # y
    Ry=np.array([[ cos(y), 0, -sin(y)],
                 [      0, 1,      0],
                 [sin(y), 0, cos(y)]])
    # z
    Rz=np.array([[cos(z), sin(z), 0],
                 [-sin(z),  cos(z), 0],
                 [     0,       0, 1]])
    R = Rx.dot(Ry).dot(Rz)
    return R.astype(np.float32)

def angle2matrix_zyx(angles):

    x, y, z = angles[2], angles[1], angles[0]
    # x
    Rx=np.array([[1,      0,       0],
                 [0, cos(x),  -sin(x)],
                 [0, sin(x),   cos(x)]])
    # y
    Ry=np.array([[ cos(y), 0, sin(y)],
                 [      0, 1,      0],
                 [-sin(y), 0, cos(y)]])
    # z
    Rz=np.array([[cos(z), -sin(z), 0],
                 [sin(z),  cos(z), 0],
                 [     0,       0, 1]])
        
    R=Rx.dot(Ry.dot(Rz))
    return R.astype(np.float64)


def matrix2uler_angle_zyx(rot):
    x,y=rot[0],rot[1]
    
    assert((1-x[2]**2)>(1e-5))
    
    be=asin(max(-1,min(1,x[2])))
    al=asin(max(-1,min(1,-x[1]/sqrt(1-x[2]**2))))
    ga=asin(max(-1,min(1,-y[2]/sqrt(1-x[2]**2))))
    
    
    return np.array([al,be,ga])
    
def bld_reduce_neu(bldshps):
    
    for x in bldshps:
        t=x[0].copy()
        x-=t
        x[0]=t

    

def recal_dis(data,bldshps):
#    print(data.land_cor)
    ldmk_bld=bldshps[:,:,data.land_cor,:]
#    print(ldmk_bld.shape)
    landmk_3d=np.tensordot(np.tensordot(ldmk_bld,data.user,axes=(0,0)),data.exp,axes=(0,0))
    
    landmk_3d=(angle2matrix_zyx(data.angle)@landmk_3d.T).T+data.tslt
    
    landmk_3d[:,0]/=landmk_3d[:,2]
    landmk_3d[:,1]/=landmk_3d[:,2]
    
    landmk_3d*=data.fcs
    
#    print('center-------',data.center)
    landmk_3d[:,0:2]+=data.center
    
    data.dis=(data.land_inv-landmk_3d[:,0:2]).copy()
    
def get_init_land(data,bldshps):
    
    ldmk_bld=bldshps[:,:,data.land_cor,:]
    
    landmk_3d=np.tensordot(np.tensordot(ldmk_bld,data.user,axes=(0,0)),data.init_exp,axes=(0,0))
    
    landmk_3d=(angle2matrix_zyx(data.init_angle)@landmk_3d.T).T+data.init_tslt
    
    
    landmk_3d[:,0]/=landmk_3d[:,2]
    landmk_3d[:,1]/=landmk_3d[:,2]
    
    landmk_3d*=data.fcs
    
# =============================================================================
#     print('center-------',data.center)
#     print(landmk_3d[:10,0:2])
# =============================================================================
    landmk_3d[:,0:2]+=data.center
# =============================================================================
#     print(landmk_3d[:10,0:2])
#     print('###################################')
#     print(data.init_dis[:5])
# =============================================================================
    
    ans=(landmk_3d[:,0:2]+data.init_dis).copy()

    
    ans[:,1]=data.img.shape[0]-ans[:,1]
    
# =============================================================================
#     print(np.concatenate((ans, data.land), axis=1))    
#     print(np.concatenate((data.angle, data.init_angle), axis=0))   
#     print(np.concatenate((data.tslt, data.init_tslt), axis=0))   
#     print(data.exp)
#     print(data.init_exp)
# =============================================================================
    
    
    return ans
    
def get_land(data,bldshps,user):
    
    ldmk_bld=bldshps[:,:,data.land_cor,:]
    
    landmk_3d=np.tensordot(np.tensordot(ldmk_bld,user,axes=(0,0)),data.exp,axes=(0,0))
    
    landmk_3d=(angle2matrix_zyx(data.angle)@landmk_3d.T).T+data.tslt
#    landmk_3d_=(data.rot@landmk_3d.T).T+data.tslt
#    print(np.concatenate((landmk_3d, landmk_3d_), axis=1))
    
    
    landmk_3d[:,0]/=landmk_3d[:,2]
    landmk_3d[:,1]/=landmk_3d[:,2]
    
    landmk_3d*=data.fcs
    
# =============================================================================
#     print('center-------',data.center)
#     print(landmk_3d[:10,0:2])
# =============================================================================
    landmk_3d[:,0:2]+=data.center
# =============================================================================
#     print(landmk_3d[:10,0:2])
#     print('###################################')
#     print(data.dis[:5])
# =============================================================================
    
    ans=(landmk_3d[:,0:2]+data.dis).copy()
        
    ans[:,1]=data.img.shape[0]-ans[:,1]
    
#    print(np.concatenate((ans, data.land), axis=1))    

    
    return ans    
    
def get_land_spfbldshps_inv(data,spf_bldshps):
    
    ldmk_bld=spf_bldshps[:,data.land_cor,:]
#    print('exp:',data.exp)
    landmk_3d=np.tensordot(ldmk_bld,data.exp,axes=(0,0))
#    print('angle:',data.angle)
#    print('tslt:',data.tslt)
    landmk_3d=(angle2matrix_zyx(data.angle)@landmk_3d.T).T+data.tslt
#    landmk_3d_=(data.rot@landmk_3d.T).T+data.tslt
#    print(np.concatenate((landmk_3d, landmk_3d_), axis=1))
    
    
    landmk_3d[:,0]/=landmk_3d[:,2]
    landmk_3d[:,1]/=landmk_3d[:,2]
#    print('fcs:',data.fcs)
    landmk_3d*=data.fcs
#    print('center:',data.center)
    landmk_3d[:,0:2]+=data.center
    
    ans=(landmk_3d[:,0:2]+data.dis).copy()
    
#    print(np.concatenate((ans, data.land), axis=1))    

    
    return ans   

def get_land_spfbldshps(data,spf_bldshps):
    
    ldmk_bld=spf_bldshps[:,data.land_cor,:]
    
    landmk_3d=np.tensordot(ldmk_bld,data.exp,axes=(0,0))
    
    landmk_3d=(angle2matrix_zyx(data.angle)@landmk_3d.T).T+data.tslt
#    landmk_3d_=(data.rot@landmk_3d.T).T+data.tslt
#    print(np.concatenate((landmk_3d, landmk_3d_), axis=1))
    
    
    landmk_3d[:,0]/=landmk_3d[:,2]
    landmk_3d[:,1]/=landmk_3d[:,2]
    
    landmk_3d*=data.fcs
    
    landmk_3d[:,0:2]+=data.center
    
    ans=(landmk_3d[:,0:2]+data.dis).copy()
    ans[:,1]=data.img.shape[0]-ans[:,1]
#    print(np.concatenate((ans, data.land), axis=1))    

    
    return ans    


def get_input_from_land_img(land,image,tri_idx,px_barycenter):
    
    assert(len(image.shape)==2) #gray image
    assert(image.ndim==2)
    num_px=px_barycenter[0].shape[0]
    
    pos=(land[tri_idx[px_barycenter[0]][:,0]]*(px_barycenter[1][:,0].reshape(num_px,1))+
         land[tri_idx[px_barycenter[0]][:,1]]*(px_barycenter[1][:,1].reshape(num_px,1))+
         land[tri_idx[px_barycenter[0]][:,2]]*(px_barycenter[1][:,2].reshape(num_px,1)))
    
    pos=np.around(pos).astype(np.int)
    
    data_input=image[
                (np.minimum(image.shape[0]-1,np.maximum(0,pos[:,1])),
                 np.minimum(image.shape[1]-1,np.maximum(0,pos[:,0])))].copy()
    
    
#       0~1                    
    data_input=\
        (data_input-data_input.min())/\
        max(1,(data_input.max()-data_input.min()))
       
#       -1~1             
    data_input=data_input*2-1      

    return data_input    
    
def norm_img(image):
    assert(image.ndim==3)
    
    ans=image.copy().astype(np.float32)
    for i in range(image.shape[2]):
        now=ans[:,:,i]
        ma=now.max()
        mi=now.min()
        ans[:,:,i]=(ans[:,:,i]-mi)/(ma-mi)*2-1
    
    return ans

def load_slt_rect():
    print('loading silhouette line&vertices...')
    slt_path = "/home/weiliu/fitting_dde/const_file/3d22d/slt_line_4_10.txt";
    slt_line_pt=[]
    slt_pt_rect=[]
    with open(slt_path,'r') as f:
        line=f.readline().strip().split()
        print(line)
        slt_line_num=int(line[0])
        assert(slt_line_num==84)
        slt_line_pt=[0]*slt_line_num
        for i in range(slt_line_num):
            line=f.readline().strip().split()
            assert(i==int(line[0]))
            num=int(line[1])
            slt_line_pt[i]=list(map(int,line[2:]))
#            print(slt_line_pt[i])
            assert(num==len(slt_line_pt[i]))
    
#    rect_path = "/home/weiliu/fitting_dde/const_file/3d22d/slt_rect_4_10.txt";
    rect_path = "rect_all.txt";
    with open(rect_path,'r') as f:
        line=f.readline().strip().split()
        print(line)
        slt_rect_num=int(line[0])        
        slt_pt_rect=[0]*11510
        for i in range(slt_rect_num):
            line=f.readline().strip().split()            
            
            idx=int(line[0])
            num=int(line[1])            
            slt_pt_rect[idx]=list(map(int,line[2:]))
#            print(num)
#            print(len(slt_pt_rect[idx]))
            assert(num*2==len(slt_pt_rect[idx]))
    
#    print('slt_pt_rect')
#    print(slt_pt_rect)
    
    return slt_line_pt,slt_pt_rect
    
slt_line_pt,slt_pt_rect=load_slt_rect()
def get_slt_land_cor(data,bldshps,user):
    
#could be faster
    if (len(bldshps)==4):
        face_3d=np.tensordot(np.tensordot(bldshps,user,axes=(0,0)),data.exp,axes=(0,0))
    else:
        face_3d=np.tensordot(bldshps,data.exp,axes=(0,0))
        
    face_3d=(angle2matrix_zyx(data.angle)@face_3d.T).T+data.tslt
    
    line_num=len(slt_line_pt)
    slt_pt_idx=[0]*line_num
    for i in range(line_num):
        mi=9999
        for j in range(len(slt_line_pt[i])):
            x=slt_line_pt[i][j]
            
            
            v=face_3d[x]
            norm=np.zeros((3,))
            for k in range(0,len(slt_pt_rect[x])//2,2):
                v1=face_3d[slt_pt_rect[v][k]]-v
                v2=face_3d[slt_pt_rect[v][k+1]]-v
                t=np.cross(v1,v2)
                assert(np.linalg.norm(t)>0)
                t=t/np.linalg.norm(t)
                norm+=t
            assert(np.linalg.norm(norm)>0)
            norm=norm/np.linalg.norm(norm)
            if abs(norm[2])<mi:
                mi=norm[2]
                slt_pt_idx[j]=x
                
            vt=v.copy()
            assert(np.linalg.norm(vt)>0)
            vt/=np.linalg.norm(vt)
            if (abs(vt[2])<mi):
                mi=abs(vt[2])
                slt_pt_idx[j]=x
                
        
    slt_pt_3d=face_3d[slt_pt_idx]
    slt_pt_2d=slt_pt_3d[:,0:2].copy()
    slt_pt_2d=slt_pt_2d/slt_pt_3d[:,2].reshape(line_num,1)
    
    image=data.img
    for pt in slt_pt_2d:        
        cv2.circle(image,tuple(np.around(pt).astype(np.int)), 2, (255,255,255), -1)
        
    cv2.imshow('test image',image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    all_length=np.empty(line_num,dtype=np.float64)
    mid_jaw_idx=41    
    data.land_cor[7]=slt_pt_idx[mid_jaw_idx]
    
    all_length[mid_jaw_idx]=0
    for i in range(mid_jaw_idx-1,-1,-1):
        pt0=slt_pt_2d[i]
        pt1=slt_pt_2d[i-1]
        all_length[i]=all_length[i-1]+np.linalg.norm(pt0-pt1)
    
    itv=all_length[8]/7
    now_idx=8
    for i in range(mid_jaw_idx-1,-1,-1):
        if itv*(now_idx-7)>=all_length[i]:
            data.land_cor[now_idx]=slt_pt_idx[i]
            now_idx+=1
            if (now_idx>14):
                break
    if (now_idx==14):
        data.land_cor[now_idx]=slt_pt_idx[0]
    
    all_length[mid_jaw_idx]=0
    for i in range(mid_jaw_idx+1,line_num):
        pt0=slt_pt_2d[i]
        pt1=slt_pt_2d[i-1]
        all_length[i]=all_length[i-1]+np.linalg.norm(pt0-pt1)
        
    itv=all_length[line_num-10]/7
    now_idx=6
    for i in range(mid_jaw_idx+1,line_num):
        if itv*(7-now_idx)>all_length[i]:
            data.land_cor[now_idx]=slt_pt_idx[i]
            now_idx-=1
            if (now_idx<0):
                break
    if (now_idx==0):
        data.land_cor[now_idx]=slt_pt_idx[line_num-1]
        
def get_slt_land_cor_init(data,bldshps,user):
    
#could be faster
    face_3d=np.tensordot(np.tensordot(bldshps,user,axes=(0,0)),data.init_exp,axes=(0,0))

    face_3d=(angle2matrix_zyx(data.init_angle)@face_3d.T).T+data.init_tslt
    
    line_num=len(slt_line_pt)
    slt_pt_idx=[0]*line_num
    for i in range(line_num):
        mi=9999
        for j in range(len(slt_line_pt[i])):
            x=slt_line_pt[i][j]
            
            
            v=face_3d[x]
            norm=np.zeros((3,))
            for k in range(0,len(slt_pt_rect[x])//2,2):
                v1=face_3d[slt_pt_rect[v][k]]-v
                v2=face_3d[slt_pt_rect[v][k+1]]-v
                t=np.cross(v1,v2)
                assert(np.linalg.norm(t)>0)
                t=t/np.linalg.norm(t)
                norm+=t
            assert(np.linalg.norm(norm)>0)
            norm=norm/np.linalg.norm(norm)
            if abs(norm[2])<mi:
                mi=norm[2]
                slt_pt_idx[j]=x
                
            vt=v.copy()
            assert(np.linalg.norm(vt)>0)
            vt/=np.linalg.norm(vt)
            if (abs(vt[2])<mi):
                mi=abs(vt[2])
                slt_pt_idx[j]=x
                
        
    slt_pt_3d=face_3d[slt_pt_idx]
    slt_pt_2d=slt_pt_3d[:,0:2].copy()
    slt_pt_2d=slt_pt_2d/slt_pt_3d[:,2].reshape(line_num,1)
    
    image=data.img
    for pt in slt_pt_2d:        
        cv2.circle(image,tuple(np.around(pt).astype(np.int)), 2, (255,255,255), -1)
        
    cv2.imshow('test image',image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    all_length=np.empty(line_num,dtype=np.float64)
    mid_jaw_idx=41    
    data.land_cor[7]=slt_pt_idx[mid_jaw_idx]
    
    all_length[mid_jaw_idx]=0
    for i in range(mid_jaw_idx-1,-1,-1):
        pt0=slt_pt_2d[i]
        pt1=slt_pt_2d[i-1]
        all_length[i]=all_length[i-1]+np.linalg.norm(pt0-pt1)
    
    itv=all_length[8]/7
    now_idx=8
    for i in range(mid_jaw_idx-1,-1,-1):
        if itv*(now_idx-7)>=all_length[i]:
            data.land_cor[now_idx]=slt_pt_idx[i]
            now_idx+=1
            if (now_idx>14):
                break
    if (now_idx==14):
        data.land_cor[now_idx]=slt_pt_idx[0]
    
    all_length[mid_jaw_idx]=0
    for i in range(mid_jaw_idx+1,line_num):
        pt0=slt_pt_2d[i]
        pt1=slt_pt_2d[i-1]
        all_length[i]=all_length[i-1]+np.linalg.norm(pt0-pt1)
        
    itv=all_length[line_num-10]/7
    now_idx=6
    for i in range(mid_jaw_idx+1,line_num):
        if itv*(7-now_idx)>all_length[i]:
            data.land_cor[now_idx]=slt_pt_idx[i]
            now_idx-=1
            if (now_idx<0):
                break
    if (now_idx==0):
        data.land_cor[now_idx]=slt_pt_idx[line_num-1]
    
def draw_pt_img(points, image,color=(255,255,255)):
    for pt in points:        
        cv2.circle(image,tuple(np.around(pt).astype(np.int)), 1, color, -1)
    
def draw_land(landmarks,image):
    assert(landmarks.shape[0]==73)
    for pt in landmarks:        
        cv2.circle(image,tuple(np.around(pt).astype(np.int)), 2, (0,255,0), -1)
    for i in range(14):
        cv2.line(image,
                 tuple(np.around(landmarks[i]).astype(np.int)),
                 tuple(np.around(landmarks[i+1]).astype(np.int)),
                 (255,0,0),1)    
    
    
from tri_rasterization import tri_rasterization as tras
def show_norm_img(data,spf_bldshps,over_01_idx,use_tri_idx):
#    print('slt_pt_rect')
#    print(slt_pt_rect)
    
    cv2.imshow('normal_image',data.img)
    cv2.waitKey()
    
    image=np.zeros(data.img.shape)    
    image_r=np.zeros(data.img.shape) 
    image_g=np.zeros(data.img.shape) 
    image_b=np.zeros(data.img.shape) 
    image_white=np.zeros(data.img.shape) 
    image_white[...,:]=np.array([255,255,255])
    image_ha=np.zeros(data.img.shape) 
    
    data.exp[0]=1
    mesh=np.tensordot(spf_bldshps,data.exp,axes=(0,0))
    mesh=(angle2matrix_zyx(data.angle)@mesh.T).T+data.tslt
    
    landmk_3d=mesh.copy()    
    landmk_3d[:,0]/=landmk_3d[:,2]
    landmk_3d[:,1]/=landmk_3d[:,2]
#    print('fcs:',data.fcs)
    landmk_3d*=data.fcs
#    print('center:',data.center)
    landmk_3d[:,0:2]+=data.center
    landmk_3d[:,1]=data.img.shape[0]-landmk_3d[:,1]
    
#    ans=np.empty((over_01_idx.shape[0],3))
    ans=np.empty((landmk_3d.shape[0],3))
    i=0
    for x in range(landmk_3d.shape[0]): #over_01_idx:        
        v=mesh[x,:]
        print('v')
        print(v)
        norm=np.zeros((3,))
        for k in range(0,len(slt_pt_rect[x])//2,2):
            print(slt_pt_rect[x][k])
            print(mesh[slt_pt_rect[x][k],:])
            v1=mesh[slt_pt_rect[x][k],:]-v
            print('v1')
            v2=mesh[slt_pt_rect[x][k+1],:]-v
            print('v1')
            print(v1)
            print('v2')
            print(v2)
            t=np.cross(v1,v2)
            assert(np.linalg.norm(t)>0)
            t=t/np.linalg.norm(t)
            norm+=t
        assert(np.linalg.norm(norm)>0)
        norm=norm/np.linalg.norm(norm)
        
        
        
        image_ha[
            np.around(landmk_3d[x,1]).astype(np.int),
            np.around(landmk_3d[x,0]).astype(np.int),2]=(norm[0]+1)/2*255
        image_ha[
            np.around(landmk_3d[x,1]).astype(np.int),
            np.around(landmk_3d[x,0]).astype(np.int),1]=(norm[1]+1)/2*255
        image_ha[
            np.around(landmk_3d[x,1]).astype(np.int),
            np.around(landmk_3d[x,0]).astype(np.int),0]=(max(norm[2],0))*255
                
        norm[0],norm[1],norm[2]=(max(norm[2],0))*255,(norm[1]+1)/2*255,(norm[0]+1)/2*255
        ans[i]=norm
        i+=1
    
    cv2.imshow('hhhaaaaaaaaa',image_ha)
    cv2.waitKey()
    
    ans[...,:]=(ans[...,:]-np.min(ans,axis=0))/(np.max(ans,axis=0)-np.min(ans,axis=0))*255
    result=tras.render_colors(landmk_3d, use_tri_idx,ans, image.shape[0],image.shape[1])
            
    cv2.imshow('normal_image',result)
    cv2.waitKey()
    

        
    print('aaa')
    
    for x,norm in zip(over_01_idx,ans):    
        print('ewwwwwwwwwww')
        print(norm)
        print(np.min(ans,axis=0))
        print(np.max(ans,axis=0))
        norm=(norm-np.min(ans,axis=0))/(np.max(ans,axis=0)-np.min(ans,axis=0))*255
        print(x,norm)
        print(landmk_3d[x,1],landmk_3d[x,0])
        print(image.shape)
        print(image[
            np.around(landmk_3d[x,1]).astype(np.int),
            np.around(landmk_3d[x,0]).astype(np.int),:])
        print(norm)
        image[
            np.around(landmk_3d[x,1]).astype(np.int),
            np.around(landmk_3d[x,0]).astype(np.int),:]=norm
        image_b[
            np.around(landmk_3d[x,1]).astype(np.int),
            np.around(landmk_3d[x,0]).astype(np.int),0]=norm[0]
        image_g[
            np.around(landmk_3d[x,1]).astype(np.int),
            np.around(landmk_3d[x,0]).astype(np.int),1]=norm[1]
        image_r[
            np.around(landmk_3d[x,1]).astype(np.int),
            np.around(landmk_3d[x,0]).astype(np.int),2]=norm[2]
            
        image_white[
            np.around(landmk_3d[x,1]).astype(np.int),
            np.around(landmk_3d[x,0]).astype(np.int),:]=norm

    cv2.imshow('normal_image',image)
    cv2.waitKey()
    cv2.imshow('normal_image_b',image_b)
    cv2.waitKey()
    cv2.imshow('normal_image_g',image_g)
    cv2.waitKey()
    cv2.imshow('normal_image_r',image_r)
    cv2.waitKey()
    cv2.imshow('normal_image_wt',image_white)
    cv2.waitKey()
    
    cv2.destroyAllWindows()
    
    
    