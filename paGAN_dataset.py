"""Dataset class template

This module provides a template for users to implement custom datasets.
You can specify '--dataset_mode template' to use this dataset.
The class name should be consistent with both the filename and its dataset_mode option.
The filename should be <dataset_mode>_dataset.py
The class name should be <Dataset_mode>Dataset.py
You need to implement the following functions:
    -- <modify_commandline_options>:　Add dataset-specific options and rewrite default values for existing options.
    -- <__init__>: Initialize this dataset class.
    -- <__getitem__>: Return a data point and its metadata information.
    -- <__len__>: Return the number of images.
"""
from data.base_dataset import BaseDataset, get_params, get_transform
# from data.image_folder import make_dataset
from PIL import Image
import os
#import cv2
import numpy as np
import torch

class paGANDataset(BaseDataset):
    """A template dataset class for you to implement custom datasets."""
    @staticmethod
    def modify_commandline_options(parser, is_train):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.
        """
#        parser.add_argument('--new_dataset_option', type=float, default=1.0, help='new dataset option')
#        parser.set_defaults(max_dataset_size=10, new_dataset_option=2.0)  # specify dataset-specific default values
#        parser.input_nc=10
#        print('paGANDataset(BaseDataset): modify_commandline_options',parser.input_nc)
        return parser

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions

        A few things can be done here.
        - save the options (have been done in BaseDataset)
        - get image paths and meta information of the dataset.
        - define the image transformation.
        """
        # save the option and dataset root
        BaseDataset.__init__(self, opt)
        # get the image paths of your dataset;
#        self.image_paths = []  # You can call sorted(make_dataset(self.root, opt.max_dataset_size)) to get all the image paths under the directory self.root
        
        self.dir_AB = os.path.join(opt.dataroot, opt.phase) 
        self.image_file_paths_prefix=[]
        self.image_file_paths_neutral=[]
        
        print('paGANDataset(BaseDataset): modify_commandline_options',self.opt.input_nc)
        flag=0
        temp_neutral=''
        for root,dirs,name_file in sorted(os.walk(self.dir_AB)):
            if (len(name_file)>0):                
                if (not(root.endswith('Left') or root.endswith('Right'))):
                    flag=0
                    
#                print(root,name_file)
                for name_file_land73 in sorted(name_file): 
#                    print('file name: ', name_file_land73)
                    if name_file_land73.endswith('.land73'):
                        
                        name_file_land73=root+'/'+name_file_land73
                        name_file_img=name_file_land73[:-7]
                        self.image_file_paths_prefix.append(name_file_img)
#                        print(name_file_land73,name_file_img)
                        if flag==0:
                            temp_neutral=name_file_img
                        flag=1
                        self.image_file_paths_neutral.append(temp_neutral)
                
        
        self.image_file_paths_prefix=self.image_file_paths_prefix[:min(opt.max_dataset_size,
                                                                       len(self.image_file_paths_prefix))]
        self.image_file_paths_neutral=self.image_file_paths_neutral[:min(opt.max_dataset_size,
                                                                       len(self.image_file_paths_prefix))]
    
#        print(self.image_file_paths_prefix)
#        print('--------------------------------------')
#        print(self.image_file_paths_neutral)
#        # define the default transform function. You can use <base_dataset.get_transform>; You can also define your custom transform function
#        self.transform = get_transform(opt)
        

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index -- a random integer for data indexing

        Returns:
            a dictionary of data with their names. It usually contains the data itself and its metadata information.

        Step 1: get a random image path: e.g., path = self.image_paths[index]
        Step 2: load your data from the disk: e.g., image = Image.open(path).convert('RGB').
        Step 3: convert your data to a PyTorch tensor. You can use helpder functions such as self.transform. e.g., data = self.transform(image)
        Step 4: return a data point as a dictionary.
        """
#        print('__get__item__ index:',index,'\n')
#        print(self.image_file_paths_prefix[index])
#        print(self.image_file_paths_neutral[index])
#        img_cp=cv2.imread(self.image_file_paths_prefix[index]+'_crop.jpg')
#        img_tri=cv2.imread(self.image_file_paths_prefix[index]+'_tri.jpg')
#        img_norm=cv2.imread(self.image_file_paths_prefix[index]+'_norm.jpg')
#        img_dpth=cv2.imread(self.image_file_paths_prefix[index]+'_dpth.jpg',0)
#        img_netu=cv2.imread(self.image_file_paths_neutral[index]+'_crop.jpg')
#        
##        cv2.imshow('img_cp',img_cp)
##        cv2.imshow('img_tri',img_tri)
##        cv2.imshow('img_norm',img_norm)
##        cv2.imshow('img_dpth',img_dpth)
##        cv2.imshow('img_netu',img_netu)
##        cv2.waitKey(0)
#        img_dpth=img_dpth[...,np.newaxis]
##        print('dpth shape:', img_dpth.shape)
#        
#        data_A=np.concatenate((img_netu,img_tri,img_norm,img_dpth),axis=2)
        
        
        
        img_cp = Image.open(self.image_file_paths_prefix[index]+'_crop.jpg').convert('RGB')
        transform_params = get_params(self.opt, img_cp.size)
        color_transform = get_transform(self.opt, transform_params, grayscale=0)
        gray_transform = get_transform(self.opt, transform_params, grayscale=1)
        
        img_cp=color_transform(img_cp)
        
        img_tri = Image.open(self.image_file_paths_prefix[index]+'_tri.jpg').convert('RGB')
        img_tri=color_transform(img_tri)
        img_tri_ini = Image.open(self.image_file_paths_prefix[index]+'_tri_ini.jpg').convert('RGB')
        img_tri_ini=color_transform(img_tri_ini)
        img_norm = Image.open(self.image_file_paths_prefix[index]+'_norm.jpg').convert('RGB')
        img_norm=color_transform(img_norm)
        img_dpth = Image.open(self.image_file_paths_prefix[index]+'_dpth.jpg')
        img_dpth=gray_transform(img_dpth)
        img_netu = Image.open(self.image_file_paths_neutral[index]+'_crop.jpg').convert('RGB')
        img_netu=color_transform(img_netu)
        
#        print('img_tri size',img_tri.shape)
#        print('img_cp size',img_cp.shape)
#        print('img_norm size',img_norm.shape)
#        print('img_dpth size',img_dpth.shape)
#        print('img_netu size',img_netu.shape)
        
        data_A_tri=torch.cat((img_netu,img_norm,img_dpth,img_tri),dim=0)
        data_A_tri_ini=torch.cat((img_netu,img_norm,img_dpth,img_tri_ini),dim=0)
        
        
        return {'data_A_tri': data_A_tri, 'data_A_tri_ini': data_A_tri_ini, 'data_B': img_cp,
                'A_tri':img_tri ,'A_tri_ini':img_tri_ini , 'A_norm':img_norm, 'A_dpth':img_dpth, 'A_netu':img_netu,
                'path': self.image_file_paths_prefix}

    def __len__(self):
        """Return the total number of images."""
        return len(self.image_file_paths_prefix)
