import os
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import math 



def clipping(clipping_ratio, x):

    amp = x.abs()  
    sigma = torch.sqrt(torch.mean(amp**2, -1, True))
    ratio = sigma*clipping_ratio/amp
    scale = torch.min(ratio, torch.ones_like(ratio))
        
    with torch.no_grad():
        bias = x*scale - x
        
    return x + bias


def add_cp(x, cp_len):
    return torch.cat((x[...,-cp_len:], x), dim=-1)


def rm_cp(x, cp_len):
    return x[...,cp_len:]


def batch_conv1d(x, weights):
    '''
    Enable batch-wise convolution using group convolution operations
    x: BxN
    weight: BxL
    '''

    assert x.shape[0] == weights.shape[0]
    
    b, n = x.shape
    l = weights.shape[1]

    x = x.unsqueeze(0)  # 1xBxN
    weights = weights.unsqueeze(1) # Bx1xL
    x = F.pad(x, (l-1, 0), "constant", 0) # 1xBx(N+L-1)
    out = F.conv1d(x, weight=weights, bias=None, stride=1, dilation=1, groups=b, padding=0) # 1xBxN

    return out

def PAPR(x):
    power = torch.mean((x.abs())**2, -1)
    pwr_max, _ = torch.max((x.abs())**2, -1)
    return 10*torch.log10(pwr_max/power)

def normalize(x, power):
    pwr = torch.mean(x.abs()**2, -1, True)
    return np.sqrt(power)*x/torch.sqrt(pwr)


def ZF_equalization(H_est, Y):
    # H_est: NxPx1xMx2
    # Y: NxPxSxMx2
    return Y/H_est

def MMSE_equalization(H_est, Y, noise_pwr):
    # H_est: NxPx1xM
    # Y: NxPxSxM  
    # no = complex_multiplication(Y, complex_conjugate(H_est))
    # de = complex_amp2(H_est)**2 + noise_pwr.unsqueeze(-1) 
    # return no/de
    no = Y * H_est.conj()
    de = H_est.abs()**2 + noise_pwr.unsqueeze(-1) 
    return no/de

def LS_channel_est(pilot_tx, pilot_rx):
    # pilot_tx: NxPx1xM
    # pilot_rx: NxPxS'xM
    return torch.mean(pilot_rx, 2, True)/pilot_tx

def LMMSE_channel_est(pilot_tx, pilot_rx, noise_pwr):
    # pilot_tx: NxPx1xM
    # pilot_rx: NxPxS'xM
    #return complex_multiplication(torch.mean(pilot_rx, 2, True), complex_conjugate(pilot_tx))/(1+(noise_pwr.unsqueeze(-1)/pilot_rx.shape[2]))
    return torch.mean(pilot_rx, 2, True)*pilot_tx.conj()/(1+(noise_pwr.unsqueeze(-1)/pilot_rx.shape[2]))


