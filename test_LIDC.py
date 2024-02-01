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
import numpy as np
import os
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util.visualizer import save_images, save_npy
import ntpath

def save_npy_gamma( img, gamma = 1/2, maxx=7.0 , times_100=True):
    print('    gamma of {} was selected! '.format(gamma))
    img = np.clip(img, 0.0, 1.0)
    img = np.power(img, 1/gamma)
    img = img*maxx
    if times_100:
        img = img*100
    return img

def save_npy(visuals, image_path):
    visuals_numpy = (visuals['fake_B'][0,1,:,:]).cpu().float().numpy() 
    #print('npy file saved in {}'.format(image_path))
    np.save(image_path, visuals_numpy)


if __name__ == '__main__':
    opt = TestOptions().parse()  # get test options
    # hard-code some parameters for test
    opt.num_threads = 0   # test code only supports num_threads = 0
    opt.batch_size = 1    # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
    opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers

    # test with eval mode. This only affects layers like batchnorm and dropout.
    # For [pix2pix]: we use batchnorm and dropout in the original pix2pix. You can experiment it with and without eval() mode.
    if opt.eval:
        model.eval()
    for i, data in enumerate(dataset): ## data: dict_keys(['A', 'B', 'A_paths', 'B_paths'])
        if i >= opt.num_test:  # only apply our model to opt.num_test images.
            break
        model.set_input(data)  # unpack data from data loader
        model.test()           # run inference
        visuals = model.get_current_visuals()  # get image results     ##dict_keys(['real_A', 'fake_B', 'real_B'])

        img_path = model.get_image_paths()     # get image paths
        print(visuals['fake_B'][0,1,:,:].shape, img_path)
        if i % 100 == 0:  # save images to an HTML file
            print('processing (%04d)-th image... %s' % (i, img_path))

        #save_npy(visuals,img_path)
        npy_path_dir = os.path.join(opt.results_dir, opt.name, opt.npy_save_name)
        os.makedirs(npy_path_dir, exist_ok=True)
        npy_path = os.path.join(npy_path_dir, data['name'][0][:-4]+ "_pred.npy")
        fake_B_numpy = (visuals['fake_B'][0,1,:,:]).cpu().float().numpy() 
        if opt.preprocess_gamma:
            fake_B_numpy = save_npy_gamma(fake_B_numpy) 
        np.save(npy_path, fake_B_numpy)

    print(opt.mode)


