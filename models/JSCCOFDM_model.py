# Copyright (C) 2017 NVIDIA Corporation. All rights reserved.
# Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
import numpy as np
import torch
import os
from torch.autograd import Variable
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
from . import channel
import scipy.io as sio

class JSCCOFDMModel(BaseModel):
    
    def __init__(self, opt):
        BaseModel.__init__(self, opt)
        
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['G_L2', 'PAPR']
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        self.visual_names = ['real_A', 'fake']

        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
        if self.opt.gan_mode != 'none':
            self.model_names = ['E', 'G', 'D']
        else:  # during test time, only load G
            self.model_names = ['E', 'G']

        if self.opt.feedforward in ['EXPLICIT-RES']:
            self.model_names += ['S1', 'S2']
        elif self.opt.feedforward in ['EXPLICIT-RES2']:
            self.model_names += ['S']
        
        if self.opt.feedforward in ['EXPLICIT-CE-EQ', 'EXPLICIT-RES', 'EXPLICIT-RES2']:
            C_decode = opt.C_channel
        elif self.opt.feedforward == 'IMPLICIT':
            C_decode = opt.C_channel + self.opt.N_pilot*self.opt.P*2 + self.opt.P*2
        elif self.opt.feedforward == 'EXPLICIT-CE':
            C_decode = opt.C_channel + self.opt.P*2
        
        if self.opt.is_feedback:
            add_C = self.opt.P*2
        else:
            add_C = 0
        
        # define networks (both generator and discriminator)
        self.netE = networks.define_E(input_nc=opt.input_nc, ngf=opt.ngf, max_ngf=opt.max_ngf,
                                      n_downsample=opt.n_downsample, C_channel=opt.C_channel, 
                                      n_blocks=opt.n_blocks, norm=opt.norm_EG, init_type=opt.init_type,
                                      init_gain=opt.init_gain, gpu_ids=self.gpu_ids, first_kernel=opt.first_kernel, first_add_C=add_C)

        self.netG = networks.define_G(output_nc=opt.output_nc, ngf=opt.ngf, max_ngf=opt.max_ngf,
                                      n_downsample=opt.n_downsample, C_channel=C_decode, 
                                      n_blocks=opt.n_blocks, norm=opt.norm_EG, init_type=opt.init_type,
                                      init_gain=opt.init_gain, gpu_ids=self.gpu_ids, first_kernel=opt.first_kernel, activation=opt.activation)

        #if self.isTrain and self.is_GAN:  # define a discriminator; 
        if self.opt.gan_mode != 'none':
            self.netD = networks.define_D(opt.output_nc, opt.ndf, opt.n_layers_D, 
                                          opt.norm_D, opt.init_type, opt.init_gain, self.gpu_ids)
        
        if self.opt.feedforward in ['EXPLICIT-RES']:
            self.netS1 = networks.define_Subnet(dim=(self.opt.N_pilot*self.opt.P+1)*2, dim_out=self.opt.P*2,
                                        norm=opt.norm_EG, init_type=opt.init_type, init_gain=opt.init_gain, gpu_ids=self.gpu_ids)

            self.netS2 = networks.define_Subnet(dim=(self.opt.S+1)*self.opt.P*2, dim_out=self.opt.S*self.opt.P*2,
                                        norm=opt.norm_EG, init_type=opt.init_type, init_gain=opt.init_gain, gpu_ids=self.gpu_ids)
        elif self.opt.feedforward in ['EXPLICIT-RES2']:
            self.netS = networks.define_Subnet(dim=(2*self.opt.S+self.opt.N_pilot+2)*self.opt.P*2, dim_out=self.opt.S*self.opt.P*2,
                                        norm=opt.norm_EG, init_type=opt.init_type, init_gain=opt.init_gain, gpu_ids=self.gpu_ids)


        print('---------- Networks initialized -------------')

        # set loss functions and optimizers
        if self.isTrain:
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.criterionFeat = torch.nn.L1Loss()
            self.criterionL2 = torch.nn.MSELoss()

            params = list(self.netE.parameters()) + list(self.netG.parameters())

            if self.opt.feedforward in ['EXPLICIT-RES']:
                params+=list(self.netS1.parameters()) + list(self.netS2.parameters())
            elif self.opt.feedforward in ['EXPLICIT-RES2']:
                params+= list(self.netS.parameters())

            self.optimizer_G = torch.optim.Adam(params, lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)

            if self.opt.gan_mode != 'none':
                params = list(self.netD.parameters())
                self.optimizer_D = torch.optim.Adam(params, lr=opt.lr, betas=(opt.beta1, 0.999))
                self.optimizers.append(self.optimizer_D)

        self.opt = opt
        self.normalize = networks.Normalize()
        self.channel = channel.OFDM(opt, self.device, './models/Pilot_bit.pt')

    def name(self):
        return 'JSCCOFDM_Model'

    def set_input(self, image):
        self.real_A = image.clone().to(self.device)
        self.real_B = image.clone().to(self.device)
        

    def forward(self):

        N = self.real_A.shape[0]
        
        if self.opt.is_feedback:
            cof, _ = self.channel.channel.sample(N, self.opt.P, self.opt.M, self.opt.L)
            out_pilot, H_t, noise_pwr = self.channel(None, SNR=self.opt.SNR, cof=cof, batch_size=N)
            self.channel_estimation(out_pilot, noise_pwr)
            if torch.cuda.is_available():
                H = self.H_est.clone().to(self.device)               
            latent = self.netE(self.real_A, H)
        else:
            cof = None
            latent = self.netE(self.real_A)
        
        self.tx = latent.view(N, self.opt.P, self.opt.S, 2, self.opt.M).permute(0,1,2,4,3)
        
        out_pilot, out_sig, self.H_true, noise_pwr, self.PAPR, self.PAPR_cp = self.channel(self.normalize(self.tx, 1), SNR=self.opt.SNR, cof=cof)
        
        N, C, H, W = latent.shape

        if self.opt.feedforward == 'IMPLICIT':
            r1 = self.channel.pilot.repeat(N,1,1,1,1)
            r2 = out_pilot
            r3 = out_sig
            dec_in = torch.cat((r1, r2, r3), 2).contiguous().permute(0,1,2,4,3).contiguous().view(N, -1, H, W)
            self.fake = self.netG(dec_in)
        elif self.opt.feedforward == 'EXPLICIT-CE':
            # Channel estimation
            self.channel_estimation(out_pilot, noise_pwr)
            r1 = self.H_est
            r2 = out_sig             
            dec_in = torch.cat((r1, r2), 2).contiguous().permute(0,1,2,4,3).contiguous().view(N, -1, H, W)
            self.fake = self.netG(dec_in)
        elif self.opt.feedforward == 'EXPLICIT-CE-EQ':
            self.channel_estimation(out_pilot, noise_pwr)
            self.equalization(self.H_est, out_sig, noise_pwr)
            r1 = self.rx
            dec_in = r1.contiguous().permute(0,1,2,4,3).contiguous().view(N, -1, H, W)
            self.fake = self.netG(dec_in)
        elif self.opt.feedforward == 'EXPLICIT-RES':
            self.channel_estimation(out_pilot, noise_pwr)
            sub11 = self.channel.pilot.repeat(N,1,1,1,1)
            sub12 = out_pilot
            sub1_input = torch.cat((sub11, sub12), 2).contiguous().permute(0,1,2,4,3).contiguous().view(N, -1, H, W)
            sub1_output = self.netS1(sub1_input).view(N, self.opt.P, 1, 2, self.opt.M).permute(0,1,2,4,3)
            
            self.equalization(self.H_est+sub1_output, out_sig, noise_pwr)
            sub21 = self.H_est+sub1_output
            sub22 = out_sig
            sub2_input = torch.cat((sub21, sub22), 2).contiguous().permute(0,1,2,4,3).contiguous().view(N, -1, H, W)
            sub2_output = self.netS2(sub2_input).view(N, self.opt.P, self.opt.S, 2, self.opt.M).permute(0,1,2,4,3)
            dec_in = (self.rx+sub2_output).permute(0,1,2,4,3).contiguous().view(latent.shape)
            self.fake = self.netG(dec_in)

        elif self.opt.feedforward == 'EXPLICIT-RES2':
            self.channel_estimation(out_pilot, noise_pwr)
            self.equalization(self.H_est, out_sig, noise_pwr)
            sub1 = self.channel.pilot.repeat(N,1,1,1,1)
            sub2 = out_pilot
            sub3 = self.H_est
            sub4 = out_sig
            sub5 = self.rx
            sub_input = torch.cat((sub1, sub2, sub3, sub4, sub5), 2).contiguous().permute(0,1,2,4,3).contiguous().view(N, -1, H, W)
            sub_output = self.netS(sub_input).view(N, self.opt.P, self.opt.S, 2, self.opt.M).permute(0,1,2,4,3)
            dec_in = (self.rx+sub_output).permute(0,1,2,4,3).contiguous().view(latent.shape)
            self.fake = self.netG(dec_in)

         

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

            if self.is_Feat:
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
        self.loss_PAPR = torch.mean(self.PAPR_cp)

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

    def channel_estimation(self, out_pilot, noise_pwr):
        if self.opt.CE == 'LS':
            self.H_est = channel.LS_channel_est(self.channel.pilot, out_pilot)
        elif self.opt.CE == 'LMMSE':
            self.H_est = channel.LMMSE_channel_est(self.channel.pilot, out_pilot, self.opt.M*noise_pwr)
        elif self.opt.CE == 'TRUE':
            self.H_est = H_true.unsqueeze(2).to(self.device)
        else:
            raise NotImplementedError('The channel estimation method [%s] is not implemented' % CE)

    def equalization(self, H_est, out_sig, noise_pwr):
        # Equalization
        if self.opt.EQ == 'ZF':
            self.rx = channel.ZF_equalization(H_est, out_sig)
        elif self.opt.EQ == 'MMSE':
            self.rx = channel.MMSE_equalization(H_est, out_sig, self.opt.M*noise_pwr)
        elif self.opt.EQ == 'None':
            self.rx = None
        else:
            raise NotImplementedError('The equalization method [%s] is not implemented' % CE)

