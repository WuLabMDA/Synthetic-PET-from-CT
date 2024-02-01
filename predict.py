import argparse
import logging
import os

import numpy as np
import torch
#import torch.nn.functional as F
#from PIL import Image
#from torchvision import transforms

#from unet import *
#from utils.data_vis import plot_img_and_mask
#from utils.dataset_CTtoPET import Dataset_CTtoPET
import nibabel as nib
import functools
import torch.nn as nn

from models.networks import ResnetGenerator

def preprocessCT(cls, im, minn=-1000.0, maxx=250.0):
    img_np = np.array(im)   #(5,512,512)
    img_np = np.clip(img_np,minn ,maxx)
    img_np = (img_np - minn)/(maxx-minn)      
    return img_np

def predict_img(net, full_img, device):
    net.eval()
    img = full_img.transpose((2, 0, 1))
    img = torch.from_numpy(preprocessCT(img))
    img = img.unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)
    with torch.no_grad():
        output = net(img) 
    return output.squeeze().cpu().numpy()


#def get_output_filenames(args):
    #in_files = os.listdir(args.inputDir) 
    #out_files = []
    #for f in in_files:
        #pathsplit = f.split('.')
        #out_files.append("{}_OUT.nii.gz".format(pathsplit[0]))
    #return out_files


def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--model', '-m', default = '/home/msalehjahromi/pytorch-CycleGAN-and-pix2pix/checkpoints/experiment_name/latest_net_G.pth', metavar='FILE', help="Specify the file in which the model is stored")
    
    parser.add_argument('--inputDir', '-i', metavar='INPUT', nargs='+', help='filenames of input images',default='/Data/CTtoPET/CTPET_5ChlWithSld2_ValTrainTest_lungOnly_May5/CT_Ts/')
    
    parser.add_argument('--outputDir', '-o', help='filenames of output images',default='/home/msalehjahromi/pytorch-CycleGAN-and-pix2pix/checkpoints/experiment_name/pred/')  #CT1_Ts_predict_rl_BigNetMeanSmooth_1

    parser.add_argument('-s', '--SUV-Threshold', type=float, default = 4.0, nargs='?', dest='SUVThresh')

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    in_files = os.listdir(args.inputDir) 

    '''
    # norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    # norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    
    self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm    , not opt.no_dropout, opt.init_type     , opt.init_gain , self.gpu_ids)
                     def define_G(input_nc    , output_nc    , ngf    , netG    , norm='batch', use_dropout=False , init_type='normal', init_gain=0.02, gpu_ids=[]  )
    '''
    net = ResnetGenerator(5, 1, 64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=9)
    #logging.info("Loading model {}".format(args.model))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    #logging.info(f'Using device {device}')
    net.to(device=device)
    net.load_state_dict(torch.load(args.model, map_location=device))
    logging.info("Model loaded !")

    net.eval()
    for i, fn in enumerate(in_files):
        logging.info("\nPredicting image {} ...".format(fn))

        full_fn = os.path.join(args.inputDir,fn)
        whole_img = nib.load(full_fn)
        img = whole_img.get_fdata()
        
    
        PET = np.zeros(img.shape)
            #print('Channel out of', img.shape[0]),
        
        # Going over all slices
        print(i, ') Predicting PET of   ', fn)
        for j in range(2,img.shape[2]-3):
            #print(j,'/',img.shape[2],end='     ')
            PET[:,:,j] = predict_img(net=net,
                           full_img=img[:,:,j-2:j+3],                         
                           device=device)
            
        #converting dta from float32 to uint16? maybe?
        
        
           
        # Saving to NIFTI    
        os.makedirs(args.outputDir,exist_ok=True)   
        PET = (PET*args.SUVThresh*100).astype(np.int16)
        img_nifti = nib.Nifti1Image( PET, whole_img.affine )
        
        pathsplit = fn.split('.')
        out_files = ("{}_OUT.nii.gz".format(pathsplit[0]))   
        img_nifti.to_filename(os.path.join(args.outputDir,out_files))
        
        print('Saved!:' ,out_files)
        print()
      
        
      
    '''
    for i, fn in enumerate(in_files):
        logging.info("\nPredicting image {} ...".format(fn))

        full_fn = os.path.join(args.inputDir,fn)
        img = np.load(full_fn)
               
        #if 3D data
        if len(img.shape)==3:
            PET = np.zeros(img.shape)
            print('Channel out of', img.shape[0]),
            for j in range(img.shape[0]):
                print(j),
                PET[j,:,:] = predict_img(net=net,
                               full_img=img[j,:,:],                         
                               device=device)
        #if 2D data                 
        elif len(img.shape)==2:
            PET = predict_img(net=net,
                               full_img=img,                         
                               device=device)
        #saving
        if args.save:
            out_fn = out_files[i]
            
            np.save( os.path.join(args.inputDir,out_files[i]), PET)
            logging.info("PET saved to {}".format(os.path.join(args.inputDir,out_files[i])))
            
        if args.isNIFTI:
            img_nifti = nib.Nifti1Image(PET, np.eye(4 ) )
            img_nifti.to_filename(os.path.join(args.inputDir, out_files[i].split('.')[0]+'.nii.gz'))
    '''
















