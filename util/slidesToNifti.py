import numpy as np
import nibabel as nib
import skimage.io as io
import os


def slides_to_nifti(pred_path, nifti_file):

    file_list = np.sort([file for file in os.listdir(pred_path) if file.endswith('.npy')])

    whole_img = nib.load(nifti_file)
    
    PET = np.zeros( whole_img.get_fdata().shape ) # PET[:,:,i]
    print(whole_img.get_fdata().shape)

    for i, file_path in enumerate( file_list ):
        #PET[:,:,i+2] = np.load(os.path.join(pred_path,file_path) )
        PET[:,:,i+2] =np.rot90( np.flipud( np.load(os.path.join(pred_path,file_path)) ), -1) 
        print(i)
        
    img_nifti = nib.Nifti1Image( PET, whole_img.affine )
    out_files = (nifti_file[:-7] + "_OUT.nii.gz")   
    img_nifti.to_filename(os.path.join(pred_path,out_files))
    
if __name__=='__main__':
    pred_path = '/home/msalehjahromi/pytorch-CycleGAN-and-pix2pix/checkpoints/experiment_name/npy_results'
    nifti_file = '/home/msalehjahromi/pytorch-CycleGAN-and-pix2pix/checkpoints/experiment_name/npy_results/CTPET_3D__000000.nii.gz'
    
    slides_to_nifti(pred_path, nifti_file)