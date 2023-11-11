# Copyright (c) 2023, Dan Fu and Simran Arora.
# Adapted from https://github.com/HazyResearch/safari/blob/main/src/models/sequence/hyena.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import opt_einsum as oe

contract = oe.contract
from hyena_utils import HyenaFilter
from flashfftconv import FlashDepthWiseConv1d


class MonarchMixerSequenceMixingFlashFFTConv(nn.Module):
    def __init__(
        self,
        d_model,
        l_max=128,
        hyena_kernel_lr=None,
        bidirectional=False,
        hyena_lr_pos_emb=1e-5,
        hyena_w=10,
        hyena_w_mod=1,
        hyena_wd=0.1,
        hyena_emb_dim=3,
        hyena_filter_dropout=0.0,
        hyena_filter_order=16,
        residual_long_conv=False,
        inference_mode=False,
    ):
        super().__init__()

        self.d_model = d_model
        self.l_max = l_max
        self.kernel_lr = hyena_kernel_lr
        self.channels = 1
        self.bidirectional = bidirectional
        self.residual_long_conv = residual_long_conv
        self.inference_mode = inference_mode
        self.NUM_PROJECTIONS = 3

        print('-- Using FlashFFTConv!')
        print('-- Bidirectional:', self.bidirectional)
        print("-- Using Long Conv Residual:", self.residual_long_conv)
        print('-- Inference mode:', inference_mode)
        print('-- Hyena w:', hyena_w)
        print('-- Hyena w mod:', hyena_w_mod)
        print(f"-- Hyena filter order: {hyena_filter_order}")
        print(f"-- Hyena filter dropout: {hyena_filter_dropout}")
        print(f"-- Hyena filter wd: {hyena_wd}")
        print(f"-- Hyena filter emb dim: {hyena_emb_dim}")
        print(f"-- Hyena filter lr: {hyena_kernel_lr}")
        print(f"-- Hyena filter lr pos emb: {hyena_lr_pos_emb}")

        if self.inference_mode: # precompute the kernels
            filter_len = 2 * l_max if self.bidirectional else l_max
            self.filter = nn.Parameter(torch.randn(d_model, filter_len))
            self.filter_bias = nn.Parameter(torch.randn(d_model))

            if self.residual_long_conv:
                self.filter2 = nn.Parameter(torch.randn(d_model, filter_len))
                self.filter2_bias = nn.Parameter(torch.randn(d_model))
        else:
            self.filter_fn = HyenaFilter(
                self.d_model,
                order=hyena_filter_order,
                seq_len=self.l_max,
                dropout=hyena_filter_dropout,
                bidirectional=self.bidirectional,
                lr=hyena_kernel_lr,
                lr_pos_emb=hyena_lr_pos_emb,
                w=hyena_w,  # frequency of periodic activations
                w_mod=hyena_w_mod,
                wd=hyena_wd,  # weight decay of kernel parameters
                emb_dim=hyena_emb_dim,
            )
            self.filter_bias = self.filter_fn.bias
        
            if self.residual_long_conv:
                self.filter_fn2 = HyenaFilter(
                    self.d_model,
                    order=hyena_filter_order,
                    seq_len=self.l_max,
                    dropout=hyena_filter_dropout,
                    bidirectional=self.bidirectional,
                    lr=hyena_kernel_lr,
                    lr_pos_emb=hyena_lr_pos_emb,
                    w=hyena_w,  # frequency of periodic activations
                    w_mod=hyena_w_mod,
                    wd=hyena_wd,  # weight decay of kernel parameters
                    emb_dim=hyena_emb_dim,
                )
                self.filter2_bias = self.filter_fn2.bias
        
        # setup projections
        self.in_linear = nn.Linear(d_model, 3 * d_model)
        self.out_linear = nn.Linear(d_model, d_model)
        
        # setup short conv
        total_width = self.d_model * self.NUM_PROJECTIONS
        short_filter = nn.Conv1d(
            in_channels=total_width,
            out_channels=total_width,
            kernel_size=3,
            groups=total_width,
            padding=2,
        )
        self.short_filter = FlashDepthWiseConv1d(
            channels=total_width,
            kernel_size=3,
            padding=1,
            weights=short_filter.weight,
            bias=short_filter.bias
        )


    def forward(self, u, **kwargs):
        # u is B L H
        B, L, H = u.shape

        u = u.transpose(-1, -2)

        # in projection, this way pushes the transpose into the matmul
        # x1x2v = self.in_linear.weight @ u + self.in_linear.bias[None, :, None] # B 3H L
        x1x2v = self.in_linear.weight @ u # B 3H L
        
        # replace short filter with fast depthwise conv
        x1x2v = self.short_filter(x1x2v)

        x1, x2, v = x1x2v.split(self.d_model, dim=1)
        x1v = x1 * v
        x1v = x1v.contiguous()
        
        if self.inference_mode:
            k = self.filter
        else:
            k = self.filter_fn.filter(L, device=u.device)
            k = rearrange(k, "c l d -> c d l")[0] # `c` is always 1 by default

            if self.bidirectional:
                k_rev = self.filter_fn.filter_rev(L, device=u.device)
                k_rev = rearrange(k_rev, "c l d -> c d l")[0] # `c` is always 1 by default
            else:
                k_rev = None
            
            k_rev = k_rev[0] if type(k_rev) is tuple else k_rev
            k = F.pad(k, (0, L)) \
                      + F.pad(k_rev.flip(-1), (L, 0))

        y = self.flashfftconv(x1v, k)

        if self.residual_long_conv:
            if self.inference_mode:
                k2 = self.filter2
            else:
                k2 = self.filter_fn2.filter(L, device=u.device)
                k2 = rearrange(k2, "c l d -> c d l")[0]

                if self.bidirectional:
                    k2_rev = self.filter_fn2.filter_rev(L, device=u.device)
                    k2_rev = rearrange(k2_rev, "c l d -> c d l")[0] # `c` is always 1 by default
                else:
                    k2_rev = None
                k2_rev = k2_rev[0] if type(k2_rev) is tuple else k2_rev
                k2 = F.pad(k2, (0, L)) \
                        + F.pad(k2_rev.flip(-1), (L, 0))           

            v = v.contiguous()
            yu = self.flashfftconv(v, k2)

        if self.residual_long_conv:
            y = y + yu
        
        y = y * x2

        y = y.transpose(-1, -2)

        return self.out_linear(y), None