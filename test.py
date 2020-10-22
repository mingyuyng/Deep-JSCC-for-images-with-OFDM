# Copyright (C) 2017 NVIDIA Corporation. All rights reserved.
# Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
import time
from models import create_model
from options.test_options import TestOptions
from data.data_loader import CreateDataLoader
import util.util as util
from util.visualizer import Visualizer
import os
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import scipy.io as sio
import models.channel as chan
import shutil
from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM
import math

# Extract the options
opt = TestOptions().parse()

opt.batch_size = 1           # batch size

if opt.dataset_mode == 'CIFAR10':
    opt.dataroot='./data'
    opt.size = 32
    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                            download=True, transform=transform)
    dataset = torch.utils.data.DataLoader(testset, batch_size=opt.batch_size,
                                             shuffle=False, num_workers=2)
    dataset_size = len(dataset)
    print('#training images = %d' % dataset_size)

elif opt.dataset_mode == 'CelebA':
    opt.dataroot = './data/celeba/CelebA_test'
    opt.load_size = 80
    opt.crop_size = 64
    opt.size = 64
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    dataset_size = len(dataset)
    print('#training images = %d' % dataset_size)
else:
    raise Exception('Not implemented yet')

########################################  OFDM setting  ###########################################
model = create_model(opt)      # create a model given opt.model and other options
model.setup(opt)               # regular setup: load and print networks; create schedulers
model.eval()

if os.path.exists(output_path) == False:
    os.makedirs(output_path)
else:
    shutil.rmtree(output_path)
    os.makedirs(output_path)

PSNR_list = []
SSIM_list = []
for i, data in enumerate(dataset):
    if i >= opt.num_test:  # only apply our model to opt.num_test images.
        break

    start_time = time.time()

    if opt.dataset_mode == 'CIFAR10':
        input = data[0]
    elif opt.dataset_mode == 'CelebA':
        input = data['data']

    model.set_input(input.repeat(opt.how_many_channel,1,1,1)) 
    model.forward()
    fake = model.fake

    # Get the int8 generated images
    img_gen_numpy = fake.detach().cpu().float().numpy()
    img_gen_numpy = (np.transpose(img_gen_numpy, (0, 2, 3, 1)) + 1) / 2.0 * 255.0
    img_gen_int8 = img_gen_numpy.astype(np.uint8) 

    origin_numpy = input.detach().cpu().float().numpy()
    origin_numpy = (np.transpose(origin_numpy, (0, 2, 3, 1)) + 1) / 2.0 * 255.0
    origin_int8 = origin_numpy.astype(np.uint8)
    

    diff = np.mean((np.float64(img_gen_int8)-np.float64(origin_int8))**2, (1,2,3))

    PSNR = 10*np.log10((255**2)/diff)        
    PSNR_list.append(np.mean(PSNR))

    img_gen_tensor = torch.from_numpy(np.transpose(img_gen_int8, (0, 3, 1, 2))).float()
    origin_tensor = torch.from_numpy(np.transpose(origin_int8, (0, 3, 1, 2))).float()

    ssim_val = ssim(img_gen_tensor, origin_tensor.repeat(opt.how_many_channel,1,1,1), data_range=255, size_average=False) # return (N,)
    SSIM_list.append(torch.mean(ssim_val))

    # Save the first sampled image
    save_path = output_path + '/' + str(i) + '_PSNR_' + str(PSNR[0]) +'_SSIM_' + str(ssim_val[0])+'.png'
    util.save_image(util.tensor2im(fake[0].unsqueeze(0)), save_path, aspect_ratio=1)

    save_path = output_path + '/' + str(i) + '.png'
    util.save_image(util.tensor2im(input), save_path, aspect_ratio=1)
    if i%100 == 0:
        print(i)


print('PSNR: '+str(np.mean(PSNR_list)))
print('SSIM: '+str(np.mean(SSIM_list)))
print('MSE CE: '+str(np.mean(H_err_list)))
print('MSE EQ: '+str(np.mean(x_err_list)))
