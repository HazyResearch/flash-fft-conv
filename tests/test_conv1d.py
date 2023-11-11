import torch 
import time
from torch import nn
from einops import rearrange
from flashfftconv import FlashDepthWiseConv1d
import pytest

@pytest.mark.parametrize('b', [1, 2, 4, 8, 16])
@pytest.mark.parametrize('h', [768, 1024, 2048, 8192])
@pytest.mark.parametrize('l', [1024, 2048, 4096, 8192])
@pytest.mark.parametrize('k', [3, 5, 7])
@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16])
def test_conv1d_bhl_fwd(b, h, l, k, dtype):
    torch.cuda.empty_cache() # empty cache between runs
    torch.manual_seed(42)
    device = 'cuda'
    
    padding =  (k -1)//2
                
    x = torch.randn([b, h, l], device=device, dtype=dtype)
    
    conv1d_torch = nn.Conv1d(
        in_channels = h,
        out_channels = h,
        kernel_size = k,
        groups = h,
        padding = padding,
        dtype = dtype,
        device = device
    )
    
    conv1d_cuda = FlashDepthWiseConv1d(channels = h,
                                    kernel_size=k,
                                    padding=padding,
                                    weights=conv1d_torch.weight,
                                    bias=conv1d_torch.bias,
                                    dtype = dtype,
                                    device = device
                                    )
    
    y_torch = conv1d_torch(x)
    y_cuda = conv1d_cuda(x)
    
    assert torch.allclose(y_torch, y_cuda, atol=1e-1)
    
    
@pytest.mark.parametrize('b', [1, 2, 4, 8, 16])
@pytest.mark.parametrize('h', [768, 1024, 2048, 8192])
@pytest.mark.parametrize('l', [1024, 2048, 4096, 8192])
@pytest.mark.parametrize('k', [3, 5, 7])
@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16])
def test_conv1d_blh_fwd(b, h, l, k, dtype):
    torch.cuda.empty_cache() # empty cache between runs
    torch.manual_seed(42)
    device = 'cuda'
    
    padding =  (k -1)//2
    x = torch.randn([b, l, h], device=device, dtype=dtype)
    
    conv1d_torch = nn.Conv1d(
        in_channels = h,
        out_channels = h,
        kernel_size = k,
        groups = h,
        padding = padding,
        dtype = dtype,
        device = device
    )
    
    conv1d_cuda = FlashDepthWiseConv1d(channels = h,
                                    kernel_size=k,
                                    padding=padding,
                                    weights=conv1d_torch.weight,
                                    bias=conv1d_torch.bias,
                                    is_bhl=False,
                                    dtype = dtype,
                                    device = device
                                    )
    
    
    x_torch = rearrange(x, 'b l h -> b h l').contiguous()
    
    y_torch = conv1d_torch(x_torch)
    
    y_cuda = conv1d_cuda(x)
    
    assert torch.allclose(y_torch, rearrange(y_cuda, 'b l h -> b h l'), atol=1e-1)