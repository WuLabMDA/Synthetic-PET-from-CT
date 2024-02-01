"""General-purpose test script for image-to-image translation.

Once you have trained your model with train.py, you can use this script to test the model.
It will load a saved model from '--checkpoints_dir' and save the results to '--results_dir'.

It first creates model and dataset given the option. It will hard-code some parameters.
It then runs inference for '--num_test' images and save results to an HTML file.

Example (You need to train models first or download pre-trained models from our website):
    Test a CycleGAN model (both sides):
        python test.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan

    Test a CycleGAN model (one side only):
        python test.py --dataroot datasets/horse2zebra/testA --name horse2zebra_pretrained --model test --no_dropout

    The option '--model test' is used for generating CycleGAN results only for one side.
    This option will automatically set '--dataset_mode single', which only loads the images from one set.
    On the contrary, using '--model cycle_gan' requires loading and generating results in both directions,
    which is sometimes unnecessary. The results will be saved at ./results/.
    Use '--results_dir <directory_path_to_save_result>' to specify the results directory.

    Test a pix2pix model:
        python test.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA

See options/base_options.py and options/test_options.py for more test options.
See training and test tips at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/tips.md
See frequently asked questions at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/qa.md
"""
import os
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util.visualizer import save_images, save_npy
#from util import html
import numpy as np
import nibabel as nib

# Checking for the .nii and .nii.gz
# for loop on them
#   Making a temporary folder ---- make .npy files of that with slides of 1 and 3
#   Get the result of the network on the .npy files
#   Apply the post processing, gamma, non-linearality, *100 ?
#   Apply the .npy files to Nifty ---- name_prediction.nii.gz   --- maybe saved on a folder above
#   Delete the temp folder


class Nifty():
    def __init__(self, opt):
        self.preprocess_gamma = opt.preprocess_gamma
        self.temp_fl = 'temp_folder'   #temp folder name

    def path_to_nifty_files(self, test_path):
        nifty_path_list = []
        for nifty_file in os.listdir(test_path):
            if nifty_file.endswith('.nii') or nifty_file.endswith('.nii.gz'):
                nifty_path_list.append(os.path.join(test_path, nifty_file))
        return nifty_path_list
             
    def create_npy_from_Nifty(self, path_to_nifty, slide = 1, rotation =True, transpose=True, temp_folder_name = 'temp_folder'):
        # if StageI,II then transpoose=True
        whole_img = nib.load(path_to_nifty)
        im = whole_img.get_fdata()
        im = im.astype(np.int32)
        if rotation:
            im = np.rot90(im,-1)
            im = np.fliplr(im)
        if transpose:
            im = im.transpose((2,0,1))
        n_slide = im.shape[0]     
        count = 0
        nifty_name = path_to_nifty.split('/')[-1].split('.')[0]
        print('1.  convert CT Nifty to npy ...    ', nifty_name, im.shape)
        breakpoint()
        for k in range(3, n_slide - 3, slide ):
            im_k = np.array( im[k-3:k+4,:,:] )
            new_folder = os.path.join(os.path.dirname(path_to_nifty), temp_folder_name)
            os.makedirs(new_folder, exist_ok=True)
            dst_npy_name = nifty_name +'_'+ str(count).zfill(6) + ".npy"
            dst_img_path = os.path.join(  new_folder  , dst_npy_name)
            np.save(dst_img_path, im_k)
            count += 1

    def nonlinear_PET(self, img, middle = 2.5 , y_axis = 0.8 , minn = 0.0, maxx = 10.0 ):
        print('    nonlinear_conversion of PET was selected! ')
        img = np.clip(img, minn, 1.0)
        img_L_y_axis = (img/y_axis)*middle
        m = (maxx - middle)/(1-y_axis)
        img_G_y_axis = img*m - m + maxx
        img = (img>=y_axis)*img_G_y_axis  + (img<y_axis)*img_L_y_axis
        return img

    def post_gamma_PET(self, img, gamma = 1/2, maxx=10.0 ):
        print('    gamma of {} was selected! '.format(gamma))
        img = np.clip(img, 0.0, 1.0)
        img = np.power(img, 1/gamma)
        img = img*maxx
        return img

    def npy_to_nifti(self, pred_path, nifti_file, times_100=False):
        file_list = np.sort([file for file in os.listdir(pred_path) if file.endswith('.npy')]) #should be (7, 512,512)
        whole_img = nib.load(nifti_file)
        PET = np.zeros( whole_img.get_fdata().shape )
        #print('the prediction PET shape is :  ' ,PET.shape )
        #print('len(file_list) :  ' , len(file_list) )
        for i, file_path in enumerate( file_list ):
            #PET[:,:,i+3] =  np.flipud(      np.load(  os.path.join(pred_path,file_path)   )      )
            PET[:,:,i+3] =  np.rot90(      np.load(  os.path.join(pred_path,file_path)   ) ,  +1 )   
            PET[:,:,i+3] = np.flipud(PET[:,:,i+3]) # I just added on OCT30 with line 65 as im = np.fliplr(im)
   
        PET = self.nonlinear_PET(PET) if self.preprocess_gamma==False else self.post_gamma_PET(PET,  maxx=7.0 )
        if times_100:
            PET = PET*100.0
            PET = PET.astype(np.int32)

        img_nifti = nib.Nifti1Image( PET, whole_img.affine )
        out_files = ("{}_pred.nii.gz".format(os.path.split(nifti_file)[1].split('.nii')[0] )  ) 
        nifty_folder = os.path.join( os.path.dirname(pred_path), 'nifty_pred')
        os.makedirs(nifty_folder, exist_ok=True)
        img_nifti.to_filename(os.path.join(nifty_folder,out_files))
        print( '3.  converted to Nifty!', out_files)

    def remove_npy(self, path):
        all_files = os.listdir(path) if os.path.exists(path) else []
        for file in all_files:
            if file.endswith('.npy'):
                os.remove(os.path.join(path, file )) 


if __name__ == '__main__':
    opt = TestOptions().parse()  # get test options
    # hard-code some parameters for test
    opt.num_threads = 0   # test code only supports num_threads = 0
    opt.batch_size = 1    # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
    opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.
    
    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers
    # create a website

    # test with eval mode. This only affects layers like batchnorm and dropout.
    # For [pix2pix]: we use batchnorm and dropout in the original pix2pix. You can experiment it with and without eval() mode.
    if opt.eval:
        model.eval()

    nifty_obj = Nifty(opt)
    niftys_path_list = nifty_obj.path_to_nifty_files(opt.dataroot)
    print('############################################################################################')
    for nifty_path in niftys_path_list:
        only_nifty_name = os.path.basename(nifty_path).split('.')[0]

        # delete is there some remaining npy files in case
        nifty_obj.remove_npy(   os.path.join(os.path.split(nifty_path)[0], nifty_obj.temp_fl)) 
        nifty_obj.remove_npy(   os.path.join(opt.results_dir, opt.name, opt.npy_save_name  ) )

        # create npy files from Nifty files and store them in 'temp_folder'
        try:
            breakpoint()
            nifty_obj.create_npy_from_Nifty(nifty_path, slide=1, temp_folder_name = nifty_obj.temp_fl)
        except:
            continue

        dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options (opt.dataset_mode='cttopet')
        for i, data in enumerate(dataset): ## data: dict_keys(['A', 'B', 'A_paths', 'B_paths'])
            if i >= opt.num_test:  # only apply our model to opt.num_test images.
                break
            
            model.set_input(data)  # unpack data from data loader
            model.test()           # run inference
            visuals = model.get_current_visuals()  # get image results     ##dict_keys(['real_A', 'fake_B', 'real_B'])
            img_path = model.get_image_paths()     # get image paths

            #save_npy(visuals,img_path)
            npy_path_dir = os.path.join(opt.results_dir, opt.name, opt.npy_save_name)
            os.makedirs(npy_path_dir, exist_ok=True)
            npy_path = os.path.join(npy_path_dir, only_nifty_name + '_Pred_'+ str(i).zfill(6) + ".npy")
            save_npy(visuals, npy_path)  #visuals['fake_B'][0,1,:,:])
        print('2.  npy prediction is done!', os.path.split(nifty_path)[1])

        nifty_path_dir = os.path.split(nifty_path)[0]
        nifty_obj.npy_to_nifti(npy_path_dir, nifty_path)

        nifty_obj.remove_npy(   os.path.join(nifty_path_dir, nifty_obj.temp_fl)   )
        print('4.  Deleting temp_folder is done!')

        nifty_obj.remove_npy(   os.path.join(opt.results_dir, opt.name, opt.npy_save_name  ))
        print('5.  Deleting npy_result is done!')
        print('############################################################################################')

#Writing the slides:
#function change:
#save_npy 

#astype(int16)



