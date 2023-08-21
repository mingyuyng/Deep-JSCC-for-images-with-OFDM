
import os
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import math 

import sys
sys.path.append('./')

from models.utils import clipping, add_cp, rm_cp, batch_conv1d, PAPR, normalize



# Realization of multipath channel as a nn module
class Channel(nn.Module):
    def __init__(self, opt, device):
        super(Channel, self).__init__()
        self.opt = opt

        # Generate unit power profile
        power = torch.exp(-torch.arange(opt.L).float()/opt.decay).view(1,1,opt.L)     # 1x1xL
        self.power = power/torch.sum(power)   # Normalize the path power to sum to 1
        self.device = device

    def sample(self, N, P, M, L):
        # Sample the channel coefficients
        cof = torch.sqrt(self.power/2) * (torch.randn(N, P, L) + 1j*torch.randn(N, P, L))
        cof_zp = torch.cat((cof, torch.zeros((N,P,M-L))), -1)
        H_t = torch.fft.fft(cof_zp, dim=-1)
        return cof, H_t

    def forward(self, input, cof=None):
        # Input size:   NxPx(Sx(M+K))
        # Output size:  NxPx(Sx(M+K))
        # Also return the true channel
        # Generate Channel Matrix

        N, P, SMK = input.shape
        
        # If the channel is not given, random sample one from the channel model
        if cof is None:
            cof, H_t = self.sample(N, P, self.opt.M, self.opt.L)
        else:
            cof_zp = torch.cat((cof, torch.zeros((N,P,self.opt.M-self.opt.L,2))), 2)  
            cof_zp = torch.view_as_complex(cof_zp) 
            H_t = torch.fft.fft(cof_zp, dim=-1)
        
        signal_real = input.real.float().view(N*P, -1)       # (NxP)x(Sx(M+K))
        signal_imag = input.imag.float().view(N*P, -1)       # (NxP)x(Sx(M+K))

        ind = torch.linspace(self.opt.L-1, 0, self.opt.L).long()
        cof_real = cof.real[...,ind].view(N*P, -1).float().to(self.device)  # (NxP)xL
        cof_imag = cof.imag[...,ind].view(N*P, -1).float().to(self.device)  # (NxP)xL
        
        output_real = batch_conv1d(signal_real, cof_real) - batch_conv1d(signal_imag, cof_imag)   # (NxP)x(L+SMK-1)
        output_imag = batch_conv1d(signal_real, cof_imag) + batch_conv1d(signal_imag, cof_real)   # (NxP)x(L+SMK-1)

        output = torch.cat((output_real.view(N*P,-1,1), output_imag.view(N*P,-1,1)), -1).view(N,P,SMK,2)   # NxPxSMKx2
        output = torch.view_as_complex(output) 

        return output, H_t


# Realization of OFDM system as a nn module
class OFDM(nn.Module):
    def __init__(self, opt, device, pilot_path):
        super(OFDM, self).__init__()
        self.opt = opt

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
        self.pilot = torch.view_as_complex(self.pilot)
        self.pilot = normalize(self.pilot, 1)
        self.pilot_cp = add_cp(torch.fft.ifft(self.pilot), self.opt.K).repeat(opt.P, opt.N_pilot,1)        

    def forward(self, x, SNR, cof=None, batch_size=None):
        # Input size: NxPxSxM   The information to be transmitted
        # cof denotes given channel coefficients
                
        # If x is None, we only send the pilots through the channel
        is_pilot = (x == None)
        
        if not is_pilot:
            
            # Change to new complex representations
            N = x.shape[0]

            # IFFT:                    NxPxSxM  => NxPxSxM
            x = torch.fft.ifft(x, dim=-1)

            # Add Cyclic Prefix:       NxPxSxM  => NxPxSx(M+K)
            x = add_cp(x, self.opt.K)

            # Add pilot:               NxPxSx(M+K)  => NxPx(S+1)x(M+K)
            pilot = self.pilot_cp.repeat(N,1,1,1)
            x = torch.cat((pilot, x), 2)
            Ns = self.opt.S
        else:
            N = batch_size
            x = self.pilot_cp.repeat(N,1,1,1)
            Ns = 0    

        # Reshape:                 NxPx(S+1)x(M+K)  => NxPx(S+1)(M+K)
        x = x.view(N, self.opt.P, (Ns+self.opt.N_pilot)*(self.opt.M+self.opt.K))
        
        # PAPR before clipping
        papr = PAPR(x)
        
        # Clipping (Optional):     NxPx(S+1)(M+K)  => NxPx(S+1)(M+K)
        if self.opt.is_clip:
            x = self.clip(x)
        
        # PAPR after clipping
        papr_cp = PAPR(x)
        
        # Pass through the Channel:        NxPx(S+1)(M+K)  =>  NxPx((S+1)(M+K))
        y, H_t = self.channel(x, cof)
        
        # Calculate the power of received signal        
        pwr = torch.mean(y.abs()**2, -1, True)
        noise_pwr = pwr*10**(-SNR/10)

        # Generate random noise
        noise = torch.sqrt(noise_pwr/2) * (torch.randn_like(y) + 1j*torch.randn_like(y))
        y_noisy = y + noise
        
        # NxPx((S+S')(M+K))  =>  NxPx(S+S')x(M+K)
        output = y_noisy.view(N, self.opt.P, Ns+self.opt.N_pilot, self.opt.M+self.opt.K)

        y_pilot = output[:,:,:self.opt.N_pilot,:]         # NxPxS'x(M+K)
        y_sig = output[:,:,self.opt.N_pilot:,:]           # NxPxSx(M+K)
        
        if not is_pilot:
            # Remove Cyclic Prefix:   
            info_pilot = rm_cp(y_pilot, self.opt.K)    # NxPxS'xM
            info_sig = rm_cp(y_sig, self.opt.K)        # NxPxSxM

            # FFT:                     
            info_pilot = torch.fft.fft(info_pilot, dim=-1)
            info_sig = torch.fft.fft(info_sig, dim=-1)

            return info_pilot, info_sig, H_t, noise_pwr, papr, papr_cp
        else:
            info_pilot = rm_cp(y_pilot, self.opt.K)    # NxPxS'xM
            info_pilot = torch.fft.fft(info_pilot, dim=-1)

            return info_pilot, H_t, noise_pwr


# Realization of direct transmission over the multipath channel
class PLAIN(nn.Module):
    
    def __init__(self, opt, device):
        super(PLAIN, self).__init__()
        self.opt = opt

        # Setup the channel layer
        self.channel = Channel(opt, device)

    def forward(self, x, SNR):

        # Input size: NxPxM   
        N, P, M = x.shape
        y = self.channel(x, None)
        
        # Calculate the power of received signal
        pwr = torch.mean(y.abs()**2, -1, True)      
        noise_pwr = pwr*10**(-SNR/10)
        
        # Generate random noise
        noise = torch.sqrt(noise_pwr/2) * (torch.randn_like(y) + 1j*torch.randn_like(y))
        y_noisy = y + noise                                    # NxPx(M+L-1)
        rx = y_noisy[:, :, :M, :]
        return rx 





if __name__ == "__main__":
    
    import argparse
    opt = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    opt.P = 1
    opt.S = 6
    opt.M = 64
    opt.K = 16
    opt.L = 8
    opt.decay = 4
    opt.N_pilot = 1
    opt.SNR = 10
    opt.is_clip = False

    ofdm = OFDM(opt, 0, './models/Pilot_bit.pt')

    input_f = torch.randn(128, opt.P, opt.S, opt.M) + 1j*torch.randn(1, opt.P, opt.S, opt.M)
    input_f = normalize(input_f, 1)
    input_f = input_f.cuda()

    info_pilot, info_sig, H_t, noise_pwr, papr, papr_cp = ofdm(input_f, opt.SNR)
    H_t = H_t.cuda()
    
    err = input_f*H_t.unsqueeze(0) - info_sig

    print(f'OFDM path error :{torch.mean(err.abs()**2).data}')

    from utils import ZF_equalization, MMSE_equalization, LS_channel_est, LMMSE_channel_est

    H_est_LS = LS_channel_est(ofdm.pilot, info_pilot)
    err_LS = torch.mean((H_est_LS.squeeze()-H_t.squeeze()).abs()**2)
    print(f'LS channel estimation error :{err_LS.data}')

    H_est_LMMSE = LMMSE_channel_est(ofdm.pilot, info_pilot, opt.M*noise_pwr)
    err_LMMSE = torch.mean((H_est_LMMSE.squeeze()-H_t.squeeze()).abs()**2)
    print(f'LMMSE channel estimation error :{err_LMMSE.data}')
    
    rx_ZF = ZF_equalization(H_t.unsqueeze(0), info_sig)
    err_ZF = torch.mean((rx_ZF.squeeze()-input_f.squeeze()).abs()**2)
    print(f'ZF error :{err_ZF.data}')

    rx_MMSE = MMSE_equalization(H_t.unsqueeze(0), info_sig, opt.M*noise_pwr)
    err_MMSE = torch.mean((rx_MMSE.squeeze()-input_f.squeeze()).abs()**2)
    print(f'MMSE error :{err_MMSE.data}')







