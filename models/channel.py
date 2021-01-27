
from scipy.linalg import dft
from scipy.linalg import toeplitz
import os
import torch
import torch.nn as nn
import numpy as np
import scipy.io as sio
import torch.nn.functional as F
import math

PI = math.pi


# Batched 1D convolutional layer for multipath channel
# Reference: https://github.com/pytorch/pytorch/issues/17983
class BatchConv1DLayer(nn.Module):
    def __init__(self, stride=1,
                 padding=0, dilation=1):
        super(BatchConv1DLayer, self).__init__()
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

    def forward(self, x, weight, bias=None):
        if bias is None:
            assert x.shape[0] == weight.shape[0], "dim=0 of x must be equal in size to dim=0 of weight"
        else:
            assert x.shape[0] == weight.shape[0] and bias.shape[0] == weight.shape[
                0], "dim=0 of bias must be equal in size to dim=0 of weight"

        b_i, b_j, c, h = x.shape
        b_i, out_channels, in_channels, kernel_width_size = weight.shape

        out = x.permute([1, 0, 2, 3]).contiguous().view(b_j, b_i * c, h)
        weight = weight.view(b_i * out_channels, in_channels, kernel_width_size)

        out = F.conv1d(out, weight=weight, bias=None, stride=self.stride, dilation=self.dilation, groups=b_i,
                       padding=self.padding)

        out = out.view(b_j, b_i, out_channels, out.shape[-1])

        out = out.permute([1, 0, 2, 3])

        if bias is not None:
            out = out + bias.unsqueeze(1).unsqueeze(3)

        return out

# Clipping layer
class Clipping(nn.Module):

    def __init__(self, opt):
        super(Clipping, self).__init__()
        self.CR = opt.CR  # Clipping ratio
    	
    def forward(self, x):
     
        # Calculate the additional non-linear noise
        amp = torch.sqrt(torch.sum(x**2, -1, True))
        sigma = torch.sqrt(torch.mean(x**2, (-2,-1), True) * 2)
        ratio = sigma*self.CR/amp
        scale = torch.min(ratio, torch.ones_like(ratio))
        
        with torch.no_grad():
            bias = x*scale - x
        
        return x + bias

class Add_CP(nn.Module): 
    def __init__(self, opt):
        super(Add_CP, self).__init__()
        self.opt = opt
    def forward(self, x):
        return torch.cat((x[...,-self.opt.K:,:], x), dim=-2)

class Rm_CP(nn.Module):
    def __init__(self, opt):
        super(Rm_CP, self).__init__()
        self.opt = opt
    def forward(self, x):
        return x[...,self.opt.K:, :]

# Realization of multipath channel as a nn module
class Channel(nn.Module):
    def __init__(self, opt, device):
        super(Channel, self).__init__()

        self.opt = opt

        # Generate unit power profile
        power = torch.exp(-torch.arange(opt.L).float()/opt.decay).view(1,1,opt.L,1)     # 1x1xLx1
        self.power = power/torch.sum(power)   # Normalize the path power to sum to 1
        self.device = device
        
        # Initialize the batched 1d convolution layer
        self.bconv1d = BatchConv1DLayer(padding=opt.L-1)  

    def sample(self, N, P, M, L):
        # Sample the channel coefficients
        cof = torch.sqrt(self.power/2) * torch.randn(N, P, L, 2)
        cof_zp = torch.cat((cof, torch.zeros((N,P,M-L,2))), 2)
        H_t = torch.fft(cof_zp, 1)

        return cof, H_t

    def forward(self, input, cof=None):
        # Input size:   NxPx(Sx(M+K))x2
        # Output size:  NxPx(L+Sx(M+K)-1)x2
        # Also return the true channel
        # Generate Channel Matrix

        N, P, SMK, _ = input.shape
        
        # If the channel is not given, random sample one from the channel model
        if cof is None:
            cof, H_t = self.sample(N, P, self.opt.M, self.opt.L)
        else:
            cof_zp = torch.cat((cof, torch.zeros((N,P,self.opt.M-self.opt.L,2))), 2)  
            H_t = torch.fft(cof_zp, 1)

        signal_real = input[...,0].view(N*P, 1, 1, -1)       # (NxP)x1x1x(Sx(M+K))
        signal_imag = input[...,1].view(N*P, 1, 1, -1)       # (NxP)x1x1x(Sx(M+K))

        ind = torch.linspace(self.opt.L-1, 0, self.opt.L).long()
        cof_real = cof[...,0][...,ind].view(N*P, 1, 1, -1).to(self.device)  # (NxP)x1x1xL
        cof_imag = cof[...,1][...,ind].view(N*P, 1, 1, -1).to(self.device)  # (NxP)x1x1xL

        output_real = self.bconv1d(signal_real, cof_real) - self.bconv1d(signal_imag, cof_imag)   # (NxP)x1x1x(L+SMK-1)
        output_imag = self.bconv1d(signal_real, cof_imag) + self.bconv1d(signal_imag, cof_real)   # (NxP)x1x1x(L+SMK-1)

        output = torch.cat((output_real.view(N*P,-1,1), output_imag.view(N*P,-1,1)), -1).view(N,P,self.opt.L+SMK-1,2)   # NxPx(L+SMK-1)x2

        return output, H_t


# Realization of OFDM system as a nn module
class OFDM(nn.Module):
    def __init__(self, opt, device, pilot_path):
        super(OFDM, self).__init__()
        self.opt = opt

        # Setup the add & remove CP layers
        self.add_cp = Add_CP(opt)
        self.rm_cp = Rm_CP(opt)
        self.clip = Clipping(opt)

        # Setup the channel layer
        self.channel = Channel(opt, device)
        
        # Generate the pilot signal
        if not os.path.exists(pilot_path):
            bits = torch.randint(2, (opt.M,2))
            torch.save(bits,pilot_path)
            pilot = (2*bits-1).float()
        else:
            bits = torch.load(pilot_path)
            pilot = (2*bits-1).float()
    
        self.pilot = pilot.to(device)
        self.pilot_cp = self.add_cp(torch.ifft(self.pilot,1)).repeat(opt.P, opt.N_pilot,1,1)        

    def forward(self, x, SNR, cof=None, batch_size=None):
        # Input size: NxPxSxMx2   The information to be transmitted
        # cof denotes given channel coefficients
                
        # If x is None, we only send the pilots through the channel
        is_pilot = (x == None)

        if not is_pilot:
            N = x.shape[0]

            # IFFT:                    NxPxSxMx2  => NxPxSxMx2
            x = torch.ifft(x, 1)

            # Add Cyclic Prefix:       NxPxSxMx2  => NxPxSx(M+K)x2
            x = self.add_cp(x)

            # Add pilot:               NxPxSx(M+K)x2  => NxPx(S+1)x(M+K)x2
            pilot = self.pilot_cp.repeat(N,1,1,1,1)
            x = torch.cat((pilot, x), 2)
            Ns = self.opt.S
        else:
            N = batch_size
            x = self.pilot_cp.repeat(N,1,1,1,1)
            Ns = 0    

        # Reshape:                 NxPx(S+1)x(M+K)x2  => NxPx(S+1)(M+K)x2
        x = x.view(N, self.opt.P, (Ns+self.opt.N_pilot)*(self.opt.M+self.opt.K), 2)
        
        papr = PAPR(x)
        
        # Clipping (Optional):     NxPx(S+1)(M+K)x2  => NxPx(S+1)(M+K)x2
        if self.opt.is_clip:
            x = self.clip(x)

        papr_cp = PAPR(x)
        
        # Pass through the Channel:        NxPx(S+1)(M+K)x2  =>  NxPx((S+1)(M+K)+L-1)x2
        y, H_t = self.channel(x, cof)
        
        # Calculate the power of received signal        
        pwr = torch.mean(y**2, (-2,-1), True) * 2
        noise_pwr = pwr*10**(-SNR/10)

        # Generate random noise
        noise = torch.sqrt(noise_pwr/2) * torch.randn_like(y)
        y_noisy = y + noise

        # Peak Detection: (Perfect)    NxPx((S+S')(M+K)+L-1)x2  =>  NxPx(S+S')x(M+K)x2
        output = y_noisy[:,:,:(Ns+self.opt.N_pilot)*(self.opt.M+self.opt.K),:].view(N, self.opt.P, Ns+self.opt.N_pilot, self.opt.M+self.opt.K, 2)

        y_pilot = output[:,:,:self.opt.N_pilot,:,:]         # NxPxS'x(M+K)x2
        y_sig = output[:,:,self.opt.N_pilot:,:,:]           # NxPxSx(M+K)x2
        
        if not is_pilot:
            # Remove Cyclic Prefix:   
            info_pilot = self.rm_cp(y_pilot)    # NxPxS'xMx2
            info_sig = self.rm_cp(y_sig)        # NxPxSxMx2

            # FFT:                     
            info_pilot = torch.fft(info_pilot, 1)
            info_sig = torch.fft(info_sig, 1)

            return info_pilot, info_sig, H_t, noise_pwr, papr, papr_cp
        else:
            info_pilot = self.rm_cp(y_pilot)    # NxPxS'xMx2
            info_pilot = torch.fft(info_pilot, 1)

            return info_pilot, H_t, noise_pwr


# Realization of direct transmission over the multipath channel
class PLAIN(nn.Module):
    
    def __init__(self, opt, device):
        super(PLAIN, self).__init__()
        self.opt = opt

        # Setup the channel layer
        self.channel = Channel(opt, device)

    def forward(self, x, SNR):

        # Input size: NxPxMx2   
        N, P, M, _ = x.shape
        y = self.channel(x, None)
        
        # Calculate the power of received signal
        pwr = torch.mean(y**2, (-2,-1), True) * 2        
        noise_pwr = pwr*10**(-SNR/10)
        
        # Generate random noise
        noise = torch.sqrt(noise_pwr/2) * torch.randn_like(y)
        y_noisy = y + noise                                    # NxPx(M+L-1)x2
        rx = y_noisy[:, :, :M, :]
        return rx 


def PAPR(x):
    power = torch.mean(x**2, (-2,-1))*2
    pwr_max, _ = torch.max(torch.sum(x**2, -1), -1)

    return 10*torch.log10(pwr_max/power)

def complex_division(no, de):
    a, b = no[...,0], no[...,1]
    c, d = de[...,0], de[...,1]
    out_real = (a*c+b*d)/(c**2+d**2)
    out_imag = (b*c-a*d)/(c**2+d**2)
    return torch.cat((out_real.unsqueeze(-1), out_imag.unsqueeze(-1)),-1)

def complex_multiplication(x1, x2):
    a, b = x1[...,0], x1[...,1]
    c, d = x2[...,0], x2[...,1]
    out_real = a*c - b*d
    out_imag = a*d + b*c
    return torch.cat((out_real.unsqueeze(-1), out_imag.unsqueeze(-1)),-1)

def complex_conjugate(x):
    out_real = x[...,0]
    out_imag = -x[...,1]
    return torch.cat((out_real.unsqueeze(-1), out_imag.unsqueeze(-1)),-1)
    
def complex_amp2(x):
    out_real = x[...,0]
    out_imag = x[...,1]
    return (out_real**2+out_imag**2).unsqueeze(-1)

def ZF_equalization(H_est, Y):
    # H_est: NxPx1xMx2
    # Y: NxPxSxMx2
    return complex_division(Y, H_est)

def MMSE_equalization(H_est, Y, noise_pwr):
    # H_est: NxPx1xMx2
    # Y: NxPxSxMx2  
    no = complex_multiplication(Y, complex_conjugate(H_est))
    de = complex_amp2(H_est)**2 + noise_pwr.unsqueeze(-1) 
    return no/de

def LS_channel_est(pilot_tx, pilot_rx):
    # pilot_tx: NxPx1xMx2
    # pilot_rx: NxPxS'xMx2
    return complex_division(torch.mean(pilot_rx, 2, True), pilot_tx)

def LMMSE_channel_est(pilot_tx, pilot_rx, noise_pwr):
    # pilot_tx: NxPx1xMx2
    # pilot_rx: NxPxS'xMx2
    return complex_multiplication(torch.mean(pilot_rx, 2, True), complex_conjugate(pilot_tx))/(1+(noise_pwr.unsqueeze(-1)/pilot_rx.shape[2]))

