import numpy as np
import nibabel as nib
import skimage.io as io
import os

# To be completed:
def nifti_to_5chl(path, file, start_slide = 0, end_slide = 1e3):
    CT = io.imread(os.path.join(path, file), plugin='simpleitk')
    print(CT.shape)
    min_slide = 2
    max_slide = CT.shape[0] - 2
    
    start_slide = int(np.max(min_slide, start_slide))

    end_slide = int(np.min((end_slide, max_slide)))

    count = 0
    for k in range(start_slide, end_slide):  # Removing the first and last images as they are not reliable
        #print(k)
        CT_k = np.array( CT[k-2:k+3,:,:] )
        dst_img_name = "CT_" + str(k).zfill(6) + ".npy"
        dst_img_path = os.path.join(path,'A1', dst_img_name)
        np.save(dst_img_path, CT_k)
        #creating fake PET, it is CT actually,, we do not want the code to give errors
        dst_img_path = os.path.join(path,'B1', dst_img_name)
        np.save(dst_img_path, CT_k)

if __name__=='__main__':
    path = '/Data/CTtoPET/pix2pix_5_1/testA/'
    file = 'CTPET_3D__000000.nii'
    nifti_to_5chl(path, file)