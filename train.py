
import time
from options.train_options import TrainOptions
from data import create_dataset
from models import create_model
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import torch

if __name__ == '__main__':
    tensorboard_save = False
    opt = TrainOptions().parse()   # get training options
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    dataset_size = len(dataset)    # get the number of images in the dataset.
    print('The number of training images = %d' % dataset_size)
    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers
    total_iters = 0                # the total number of training iterations

    #I added
    if tensorboard_save:
        writer = SummaryWriter(comment=f'LR_{opt.lr}_lambda_L1_{opt.lambda_L1}')

    for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1):    # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
        epoch_start_time = time.time()  # timer for entire epoch
        iter_data_time = time.time()    # timer for data loading per iteration
        epoch_iter = 0                  # the number of training iterations in current epoch, reset to 0 every epoch
        model.update_learning_rate()    # update learning rates in the beginning of every epoch.
        loss_list = [] 
        
        for i, data in enumerate(dataset):  # inner loop within one epoch
            iter_start_time = time.time()  # timer for computation per iteration
            if total_iters % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time

            total_iters += opt.batch_size
            epoch_iter += opt.batch_size
            model.set_input(data)         # unpack data from dataset and apply preprocessing
            model.optimize_parameters()   # calculate loss functions, get gradients, update network weights

            if total_iters % opt.display_freq == 0:   # display images on visdom and save images to a HTML file
                save_result = total_iters % opt.update_html_freq == 0
                model.compute_visuals()

            if total_iters % opt.print_freq == 0:    # print training losses and save logging information to the disk
                losses = model.get_current_losses()
                t_comp = (time.time() - iter_start_time) / opt.batch_size

            if total_iters % opt.save_latest_freq == 0:   # cache our latest model every <save_latest_freq> iterations
                print('saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))
                save_suffix = 'iter_%d' % total_iters if opt.save_by_iter else 'latest'
                model.save_networks(save_suffix)
            
            # losses are added here for Tensorboard
            losses = model.get_current_losses()
            loss_list.append(  [  losses['G_GAN'],   losses['G_L1'],    losses['D_real'],    losses['D_fake']  ]  )# I added
            iter_data_time = time.time()

        if tensorboard_save:
            visuals = model.get_current_visuals()     #
            mid_rA_idx, mid_rB_idx, mid_fB_idx  = int((visuals['real_A'].shape[1]-1)/2), int((visuals['real_B'].shape[1]-1)/2), int((visuals['fake_B'].shape[1]-1)/2)
            writer.add_images('CT',       torch.unsqueeze(visuals['real_A'][:,mid_rA_idx,:,:], dim=1), epoch) #
            writer.add_images('PET_Real', torch.unsqueeze(visuals['real_B'][:,mid_rB_idx,:,:], dim=1), epoch)
            writer.add_images('PET_Fake', torch.clip(torch.unsqueeze(visuals['fake_B'][:,mid_fB_idx,:,:], dim=1), 0.0, 1.0), epoch)
            # print(visuals['fake_B'].shape)#torch.Size([B, 3, 512, 512])

            loss_list_mean = np.mean(loss_list,axis=0)
            writer.add_scalar('Loss/Train/G_GAN', loss_list_mean[0], epoch)
            writer.add_scalar('Loss/Train/G_L1', loss_list_mean[1], epoch)
            writer.add_scalar('Loss/Train/D_real', loss_list_mean[2], epoch)
            writer.add_scalar('Loss/Train/D_fake', loss_list_mean[3], epoch)
            writer.add_scalar('LR', model.optimizer_G.param_groups[0]['lr'], epoch)
        

        if epoch % opt.save_epoch_freq == 0:              # cache our model every <save_epoch_freq> epochs
            print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
            model.save_networks('latest')
            model.save_networks(epoch)

        print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, opt.n_epochs + opt.n_epochs_decay, time.time() - epoch_start_time))
    if tensorboard_save:
        writer.close()