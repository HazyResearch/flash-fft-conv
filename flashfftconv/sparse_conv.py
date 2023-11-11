# Copyright (c) 2023, Dan Fu and Hermann Kumbong.
import torch
'''
Example implementations of partial and frequency-sparse convolutions.
These are just PyTorch examples, not optimized versions.
'''

class PartialFFTConv(torch.nn.Module):
    def __init__(self, N_partial):
        super().__init__()
        self.N_partial = N_partial

    def forward(self, x, k):
        L = x.shape[-1]
        N = 2 * L
        x_dtype = x.dtype
        x_f = torch.fft.rfft(x.float(), n = N)
        k_f = torch.fft.rfft(k[..., :self.N_partial], n = N)
        y_f = x_f * k_f
        y = torch.fft.irfft(y_f, n = N)[..., :L].to(x_dtype)

        return y
    
class FrequencySparseFFTConv(torch.nn.Module):
    def __init__(self, N_partial):
        super().__init__()
        self.N_partial = N_partial

    def forward(self, x, k):
        L = x.shape[-1]
        N = 2 * L
        x_dtype = x.dtype
        x_f = torch.fft.rfft(x.float(), n = N)
        k_f = torch.fft.rfft(k, n = N)
        k_f[..., self.N_partial // 2:] = 0
        y_f = x_f * k_f
        y = torch.fft.irfft(y_f, n = N)[..., :L].to(x_dtype)

        return y