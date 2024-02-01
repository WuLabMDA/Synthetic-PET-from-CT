from os.path import splitext
from os import listdir
from os.path import join
import numpy as np
from glob import glob
import torch
from torch.utils.data import Dataset
import logging
from PIL import Image
from skimage.restoration import denoise_tv_chambolle
import torchvision.transforms.functional as TF
from torchvision import transforms
import torchvision.transforms as tt
import torch.nn as nn
import torch.nn.functional as F
from data.base_dataset import BaseDataset


class CTtoPETDataset(BaseDataset):
    def __init__(self, opt):
        self.mode = opt.mode  
        self.preprocess_gamma = opt.preprocess_gamma
        BaseDataset.__init__(self, opt)

        if self.mode=='test':
            self.CT_dir = join(opt.dataroot, 'temp_folder')
            self.PET_dir = join(opt.dataroot, 'temp_folder')
        else:
            self.CT_dir = join(opt.dataroot, 'trainA')
            self.PET_dir = join(opt.dataroot, 'trainB')
        self.ids = [file for file in listdir(self.CT_dir)
                    if not file.startswith('.') and file.endswith('.npy')]
        
        logging.info(f'Creating dataset with {len(self.ids)} examples')


    @classmethod
    def preprocessCT(cls, im, minn=-900.0, maxx=200.0, noise_std = 0):
        img_np = np.array(im)   #(5,512,512)
        # Adding Noise
        if noise_std:
            s0,s1,s2 = img_np.shape
            img_np = img_np + noise_std*np.random.randn(s0,s1,s2)
        img_np = np.clip(img_np,minn ,maxx)
        img_np = (img_np - minn)/(maxx-minn)
        return img_np

    '''
    @classmethod
    def preprocessPET( cls, im, clip = 4.0, gamma=False ):
        img_np = np.array(im)
        img_np = img_np/100.0
        img_np = np.clip(img_np,0.0,clip)
        img_np = img_np/clip
        #denoising
        #img_np = denoise_tv_chambolle(img_np, weight=0.03)
        #add dimension
        if len(img_np.shape) == 2:
            img_np = np.expand_dims(img_np, axis=2)
        # HWC to CHW
        img_np = img_np.transpose((2, 0, 1))
        return img_np 
    '''



    # Gamma Function on PET
    @classmethod
    def preprocessPET_gamma(cls, img, gamma = 1/2, maxx = 7, noise_std = 0 ):
        img = np.array(img)
        img = img/100.0
        if noise_std:
            s0,s1,s2 = img.shape
            img = img + noise_std*np.random.randn(s0,s1,s2)
        img = np.clip(img, 0, maxx)
        img = img/maxx
        img = np.power(img, gamma)
        if len(img.shape) == 2:
            img = np.expand_dims(img, axis=0)
        return img

    @classmethod    
    def postprocessPET_gamma( img, gamma=1/2 ,maxx = 10.0):
        print('    gamma of {} was selected! '.format(gamma))
        img = np.array(img)
        img = np.clip(img, 0.0, 1.0)
        img = np.power(img, 1/gamma)*maxx
        return img


    # Two level Function on PET
    @classmethod
    def preprocessPET(cls, img, middle = 2.5 , y_axis = 0.80 , minn = 0.0, maxx = 10.0, noise_std = 0 ):
        img = np.array(img)
        img = img/100.0
        if noise_std:
            s0,s1,s2 = img.shape
            img = img + noise_std*np.random.randn(s0,s1,s2)
        imgMinMid = np.clip(img, minn, middle)
        imgMinMid = (imgMinMid - minn)/(middle-minn)*y_axis
        #
        imgMidMax = np.clip(img, middle, maxx)
        imgMidMax = (imgMidMax - middle)/(maxx-middle)*(1-y_axis) + y_axis
        #
        img = (img>=middle)*imgMidMax  + (img<middle)*imgMinMid
        #
        if len(img.shape) == 2:
            img = np.expand_dims(img, axis=0)
        return img
    
    @classmethod
    def edge_zero(cls, img):
        img[:,0,:] = 0
        img[:,-1,:] = 0
        img[:,:,0] = 0
        img[:,:,-1] = 0
        return img


    @classmethod
    def postprocessPET(cls, img, middle = 2.5 , y_axis = 0.85 , minn = 0.0, maxx = 10.0 ): #middle = 4   , y_axis = 0.9 , minn = 0.0, maxx = 15.0
                                 
        img = np.clip(img, minn, 1.0)
        img_L_y_axis = (img/y_axis)*middle
        m = (maxx - middle)/(1-y_axis)
        img_G_y_axis = img*m - m + maxx
        img = (img>=y_axis)*img_G_y_axis  + (img<y_axis)*img_L_y_axis
        return img

    # Data Augmentation
    def transform(self, CT, PET): #(1,512,512)
        # Affine

        if torch.rand(1) < 0.95:
            #affine_params = tt.RandomAffine(0).get_params((-15, 15), (0.07, 0.07), (0.85, 1.15), (-10, 10),img_size=(512,512))
            affine_params = tt.RandomAffine(0).get_params((-45, 45), (0.10, 0.10), (0.85, 1.15), (-7 , 7 ),img_size=(512,512))
        else:
            affine_params = tt.RandomAffine(0).get_params((-180, +180), (0.10, 0.10), (0.85, 1.15), (-7 , 7 ),img_size=(512,512))

        #affine_params = tt.RandomAffine(0).get_params((-45, 45), (0.10, 0.10), (0.85, 1.15), (-7 , 7 ),img_size=(512,512))
        CT  = TF.affine(CT, *affine_params)
        PET = TF.affine(PET, *affine_params)
        return CT, PET



    def __len__(self):
        return len(self.ids)
 
    def __getitem__(self, i):
        if self.mode == 'test':
            self.ids = np.sort(self.ids)
        idx = self.ids[i]
        PET_file = join( self.PET_dir , idx )
        CT_file = join(self.CT_dir , idx )

        # Loading        
        PET = np.load(PET_file)
        CT = np.load(CT_file)

        # Normalizing
        CT = self.preprocessCT(CT[:,:,:])
        if not self.preprocess_gamma:
            PET = self.preprocessPET(PET[2:5,:,:]) # if 1 channel chosen then
        else:
            PET = self.preprocessPET_gamma(PET[2:5,:,:]) 

        CT = self.edge_zero(CT)
        PET = self.edge_zero(PET)

        # Data augmentation
        if self.mode == 'train':
            CT, PET = self.transform(  torch.from_numpy(CT), torch.from_numpy(PET)  )
            CT, PET = CT.type(torch.FloatTensor), PET.type(torch.FloatTensor)
        else:
            # To float before GaussianTorch(PET)
            CT = torch.from_numpy(CT).type(torch.FloatTensor)
            PET = torch.from_numpy(PET).type(torch.FloatTensor)
 
        return {'A': CT, 'B': PET, 'A_paths': self.CT_dir, 'B_paths': self.PET_dir, 'name':idx}