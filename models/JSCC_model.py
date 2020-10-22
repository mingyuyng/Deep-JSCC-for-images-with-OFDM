# Copyright (C) 2017 NVIDIA Corporation. All rights reserved.
# Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
import numpy as np
import torch
import os
from torch.autograd import Variable
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import channel
from . import networks


class PLAINModel(BaseModel):
    
    def __init__(self, opt):
        BaseModel.__init__(self, opt)


        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['G_GAN', 'G_L2', 'G_Feat', 'D_real', 'D_fake']
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        self.visual_names = ['real_A', 'fake', 'real_B']

        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
        if self.opt.gan_mode != 'none':
            self.model_names = ['E', 'G', 'D']
        else:  # during test time, only load G
            self.model_names = ['E', 'G']
        
        # define networks (both generator and discriminator)
        self.netE = networks.define_E(input_nc=opt.input_nc, ngf=opt.ngf, max_ngf=opt.max_ngf,
                                      n_downsample=opt.n_downsample, C_channel=opt.C_channel, 
                                      n_blocks=opt.n_blocks, norm=opt.norm_EG, init_type=opt.init_type,
                                      init_gain=opt.init_gain, gpu_ids=self.gpu_ids, first_kernel=opt.first_kernel)

        self.netG = networks.define_G(output_nc=opt.output_nc, ngf=opt.ngf, max_ngf=opt.max_ngf,
                                      n_downsample=opt.n_downsample, C_channel=opt.C_channel, 
                                      n_blocks=opt.n_blocks, norm=opt.norm_EG, init_type=opt.init_type,
                                      init_gain=opt.init_gain, gpu_ids=self.gpu_ids, first_kernel=opt.first_kernel, activation=opt.activation)

        if self.opt.gan_mode != 'none':  # define a discriminator; 
        
            self.netD = networks.define_D(opt.output_nc, opt.ndf, opt.n_layers_D, 
                                        opt.norm_D, opt.init_type, opt.init_gain, self.gpu_ids)
        

        print('---------- Networks initialized -------------')

        # set loss functions and optimizers
        if self.isTrain:
            self.criterionGAN = networks.GANLoss(opt.gan_mode, opt.label_smooth, 1-opt.label_smooth).to(self.device)
            self.criterionFeat = torch.nn.L1Loss()
            self.criterionL2 = torch.nn.MSELoss()

            params = list(self.netE.parameters()) + list(self.netG.parameters())
            self.optimizer_G = torch.optim.Adam(params, lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)

            if self.opt.gan_mode != 'none':
                params = list(self.netD.parameters())
                self.optimizer_D = torch.optim.Adam(params, lr=opt.lr, betas=(opt.beta1, 0.999))
                self.optimizers.append(self.optimizer_D)


        self.normalize = networks.Normalize()
        self.opt = opt

        self.channel = channel.plain_channel(opt, self.device, pwr=1)
        

    def name(self):
        return 'PLAIN_Model'

    def set_input(self, image):
        self.real_A = image.clone().to(self.device)
        self.real_B = image.clone().to(self.device)

    def set_encode(self, image):
        self.real_A = image.clone().to(self.device)
        self.real_B = image.clone().to(self.device)

    def set_decode(self, latent):
        self.latent = latent.to(self.device)

    def set_img_path(self, path):
        self.image_paths = path
        
    def forward(self):

        # Generate latent vector
        self.latent = self.netE(self.real_A)

        # 2. Pass the channel
        latent = self.channel(self.latent, self.opt.SNR)

        # 3. Reconstruction
        self.fake = self.netG(latent)

    def backward_D(self):
        """Calculate GAN loss for the discriminator"""
        # Fake; stop backprop to the generator by detaching fake_B
        
        _, pred_fake = self.netD(self.fake.detach())
        self.loss_D_fake = self.criterionGAN(pred_fake, False)
        
        real_data = self.real_B
        _, pred_real = self.netD(real_data)
        self.loss_D_real = self.criterionGAN(pred_real, True)

        if self.opt.gan_mode in ['lsgan', 'vanilla']:
            self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
            self.loss_D.backward()
        elif self.opt.gan_mode == 'wgangp':
            penalty, grad = networks.cal_gradient_penalty(self.netD, real_data, self.fake.detach(), self.device, type='mixed', constant=1.0, lambda_gp=10.0)
            self.loss_D = self.loss_D_fake + self.loss_D_real + penalty
            self.loss_D.backward(retain_graph=True)

    def backward_G(self):
        """Calculate GAN and L1 loss for the generator"""
        # First, G(A) should fake the discriminator

        if self.opt.gan_mode != 'none':
            feat_fake, pred_fake = self.netD(self.fake)
            self.loss_G_GAN = self.criterionGAN(pred_fake, True)

            if self.opt.is_Feat:
                feat_real, pred_real = self.netD(self.real_B)
                self.loss_G_Feat = 0
                for j in range(len(feat_real)):
                    self.loss_G_Feat += self.criterionFeat(feat_real[j].detach(), feat_fake[j]) * self.opt.lambda_feat
            else:
                self.loss_G_Feat = 0     
        
        else:
            self.loss_G_GAN = 0
            self.loss_G_Feat = 0 

        self.loss_G_L2 = self.criterionL2(self.fake, self.real_B) * self.opt.lambda_L2
        # combine loss and calculate gradients
        self.loss_G = self.loss_G_GAN + self.loss_G_Feat + self.loss_G_L2
        self.loss_G.backward()

    def optimize_parameters(self):
        self.forward()                   # compute fake images: G(A)
        # update D
        if self.opt.gan_mode != 'none':
            self.set_requires_grad(self.netD, True)  # enable backprop for D
            self.optimizer_D.zero_grad()     # set D's gradients to zero
            self.backward_D()                # calculate gradients for D
            self.optimizer_D.step()          # update D's weights
            self.set_requires_grad(self.netD, False)  # D requires no gradients when optimizing G
        else:
            self.loss_D_fake = 0
            self.loss_D_real = 0
        # update G
        
        self.optimizer_G.zero_grad()        # set G's gradients to zero
        self.backward_G()                   # calculate graidents for G
        self.optimizer_G.step()             # udpate G's weights


    def get_encoded(self):
        return self.netE(self.real_A)

    def get_decoded(self, latent):
        if self.opt.channel == 'awgn':   # AWGN channel
            self.latent = self.normalize(latent, 1)
        elif self.opt.channel == 'bsc':   # BSC channel
            self.latent = torch.sigmoid(latent)

        # 2. Pass the channel
        latent_input = self.channel(self.latent)

        # 3. Reconstruction
        return self.netG(latent_input)
