# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 09:19:08 2019

@author: Pavilion
"""
import numpy as np

class DataOneImg:
    def __init__(self):
        self.img=[]
        self.exp=[]
        self.land_cor=[]
        self.rot=[]
        self.angle=[]
        self.tslt=[]
        self.file_name=[]
        self.fcs=0.0
        self.land_inv=[]
        self.land=[]
        self.center=[]
        self.dis=[]
        

class DataOneIdentity:
    def __init__(self):
        self.user=[]        
        self.dir_name=[]
        self.data=[]
    
class TrainOnePoint:
# =============================================================================
#     def __init__(self):
#         self.user=[]
#         
#         self.img=[]
#         self.land=[]
#         self.land_inv=[]
#         self.center=[]
#         self.fcs=0
#         self.tslt=[]
#         self.angle=[]
#         self.exp=[]
#         self.land_cor=[]
#         self.dis=[]
#         self.init_exp=[]
#         self.init_tslt=[]
#         self.init_angle=[]
#         self.init_dis=[]
#         print('__init_')
# =============================================================================
        
    def __init__(self,one_img,user):        
        print('construct from one image at beginning')
        self.user=user
        self.img=one_img.img
        self.exp=one_img.exp
        self.tslt=one_img.tslt
        self.angle=one_img.angle
        self.dis=one_img.dis
        
        self.land_cor=one_img.land_cor
        self.land=one_img.land
        self.land_inv=one_img.land_inv
        self.center=one_img.center
        self.fcs=one_img.fcs
        
        self.init_angle=one_img.angle.copy()
        self.init_tslt=one_img.tslt.copy()
        self.init_exp=one_img.exp.copy()
        
        
class TestOnePoint:
        
    def __init__(self,one_img,user):        
        
        self.user=user
        self.img=one_img.img
        self.exp=one_img.exp.copy()
        self.tslt=one_img.tslt.copy()
        self.angle=one_img.angle.copy()
        self.dis=one_img.dis.copy()
        
        self.land_cor=one_img.land_cor.copy()
        self.land=one_img.land.copy()
        self.land_inv=one_img.land_inv.copy()
        self.center=one_img.center
        self.fcs=one_img.fcs 
        
class TestVideoOnePoint:
        
    def __init__(self,one_img):                
        
        self.exp=one_img        
        self.exp=one_img.exp.copy()
        self.tslt=one_img.tslt.copy()
        self.angle=one_img.angle.copy()
        self.dis=one_img.dis.copy()
        self.land_cor=one_img.land_cor.copy()
    
        self.fcs=0.0
        self.land_inv=[]
        self.land=[]
        self.center=[]
        
class DataCNNOne:
    def __init__(self):
        self.img=[]
        self.img_230=[]
        self.exp=[]
        self.land_cor=[]
        self.rot=[]
        self.angle=[]
        self.tslt=[]
        self.file_name=[]
        self.fcs=0.0
        self.land_inv=[]
        self.land=[]
        self.center=[]
        self.dis=[]        
        
        self.centroid_inv=[]
        
        