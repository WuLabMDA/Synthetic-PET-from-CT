import os
import scipy.io
import numpy as np
import nibabel as nib
import shutil
import SimpleITK as sitk
import skimage.io as io
import shutil

# Slide of a nifti file belongs to the lung region.
#Lung = {'MRN_1':(lung_slide_start,lung_slide_end), 'MRN_2':(lung_slide_start,lung_slide_end), ... , 'MRN_n':(lung_slide_start,lung_slide_end),}

def GiveImageAndTargetLists(main_path):
    CT_list = []
    PET_list = []
    CT_path_file  = main_path + "CT_mhd_mori/"
    PET_path_file = main_path + "PET_reg_mhd_Sheeba/"
    for folder_name in os.listdir(CT_path_file):
        whole_path_CT = CT_path_file + folder_name +"/"
        whole_path_PET = PET_path_file + folder_name +"/"
        if os.path.isdir(whole_path_CT) & os.path.isdir(whole_path_PET):
            CT_path  = whole_path_CT + "CT_conv.mhd"
            PET_path = whole_path_PET + "result.0.mhd"
            CT_list.append(CT_path)
            PET_list.append(PET_path)
    return (CT_list, PET_list)


def SavingAsNpy(CT_list, PET_list, CT_Tr_path, PET_Tr_path, CT_Ts_path, PET_Ts_path, CT_Va_path,  PET_Va_path, prefix=""):
    count_ts = 0
    count_tr = 0
    count_va = 0
    for j in range(len(CT_list)):
        key = CT_list[j].split('/')[-2]
        start_slide, end_slide = Lung[key]
        print(j,'/',len(CT_list)-1)
        if PET_list[0].split('/')[-2] == CT_list[0].split('/')[-2]:
            if j < len(CT_list)*0.03:
                print('Saving', key, 'case in Test folder ...')
                CT = sitk.ReadImage(CT_list[j])
                dst_CT_name = "IC_"+ prefix +'_'+ str(count_ts).zfill(6) + ".nii.gz"
                dst_CT_path = os.path.join(CT_Ts_path, dst_CT_name)
                sitk.WriteImage(CT,dst_CT_path )
                PET = sitk.ReadImage(PET_list[j])
                dst_PET_name = "IC_"+ prefix +'_'+ str(count_ts).zfill(6) + ".nii.gz"
                dst_PET_path = os.path.join(PET_Ts_path, dst_PET_name)
                sitk.WriteImage(PET,dst_PET_path )
                count_ts += 1
            elif j < len(CT_list)*0.03:
                print( 'Saving', key, 'case in validation folder','  slides: (', start_slide,',', end_slide,')')
                CT = io.imread(CT_list[j], plugin='simpleitk') #CT = np.array(CT)   
                PET = io.imread(PET_list[j], plugin='simpleitk') #PET = np.array(PET)
                CT_path  = CT_Va_path
                PET_path   = PET_Va_path
                #Saving channel CT & PET images 
                for k in range(start_slide, end_slide,1):  # Removing the first and last images as they are not reliable
                    CT_k = np.array( CT[k-3:k+4,:,:] )
                    PET_k = np.array( PET[k-3:k+4,:,:] )
                    dst_img_name = "IC_"+ '_'+ str(count_va).zfill(6) + ".npy"
                    dst_img_path = os.path.join(CT_path, dst_img_name)
                    np.save(dst_img_path, CT_k)
                    dst_label_name = "IC_"+ '_'+ str(count_va).zfill(6) + ".npy"
                    dst_mask_path = os.path.join(PET_path, dst_label_name)
                    np.save(dst_mask_path, PET_k)
                    count_va += 1

            elif key in Lung:
                print( 'Saving', key, 'case in train folder','  slides: (', start_slide,',', end_slide,')')
                CT = io.imread(CT_list[j], plugin='simpleitk') #CT = np.array(CT)   
                PET = io.imread(PET_list[j], plugin='simpleitk') #PET = np.array(PET)
                CT_path  = CT_Tr_path
                PET_path   = PET_Tr_path
                #Saving channel CT & PET images 
                for k in range(start_slide, end_slide,3):  # Removing the first and last images as they are not reliable
                    CT_k = np.array( CT[k-3:k+4,:,:] )
                    PET_k = np.array( PET[k-3:k+4,:,:] )
                    dst_img_name = "IC_" + prefix +'_'+ str(count_tr).zfill(6) + ".npy"
                    dst_img_path = os.path.join(CT_path, dst_img_name)
                    np.save(dst_img_path, CT_k)
                    dst_label_name = "IC_" + prefix +'_'+ str(count_tr).zfill(6) + ".npy"
                    dst_mask_path = os.path.join(PET_path, dst_label_name)
                    np.save(dst_mask_path, PET_k)
                    count_tr += 1
    return (count_ts, count_tr, count_va)


if __name__=='__main__':
    # Destination directory
    main_folder = '/Data/CTtoPET/pix2pix_7Ch7_Slide3_Oct29/'
    os.makedirs(main_folder,exist_ok=True)

    # Train
    CT_Tr_path = os.path.join(main_folder, "trainA")
    os.makedirs(CT_Tr_path,exist_ok=True)

    PET_Tr_path = os.path.join(main_folder, "trainB")
    os.makedirs(PET_Tr_path,exist_ok=True)

    # Val
    CT_Va_path = os.path.join(main_folder, "valA")
    os.makedirs(CT_Va_path,exist_ok=True)

    PET_Va_path = os.path.join(main_folder, "valB")
    os.makedirs(PET_Va_path,exist_ok=True)

    # Test
    CT_Ts_path = os.path.join(main_folder, "testA")
    os.makedirs(CT_Ts_path,exist_ok=True)

    PET_Ts_path = os.path.join(main_folder, "testB")
    os.makedirs(PET_Ts_path,exist_ok=True)

    #Getting images_list & target_list
    raw_dot_m_files = '/Data/CTtoPET/MainRaw/ICON/'
    CT_list, PET_list = GiveImageAndTargetLists(raw_dot_m_files)
    print("len(CT_list) & len(PET_list):",len(CT_list),'  &  ' ,len(PET_list))
    prefix = ""
    #Let's shuffle them
    indices = np.arange(len(CT_list))
    np.random.shuffle(indices)
    CT_list, PET_list = np.array(CT_list),  np.array(PET_list)
    CT_list = CT_list[indices]
    PET_list = PET_list[indices]
    #SavingD
    SavingAsNpy(CT_list,    PET_list,  
                CT_Tr_path,  PET_Tr_path, 
                CT_Ts_path,  PET_Ts_path,
                CT_Va_path,  PET_Va_path,
                prefix=prefix)

