import torch 
import time
from torch import nn
from einops import rearrange
from flashfftconv import FlashDepthWiseConv1d
import pytest

@pytest.mark.parametrize('b', [1, 2, 4, 8, 16])
@pytest.mark.parametrize('h', [768, 1024, 2048])
@pytest.mark.parametrize('l', [1024, 2048, 4096, 8192])
@pytest.mark.parametrize('k', [3, 5, 7])
@pytest.mark.parametrize('dtype', [(torch.bfloat16, torch.bfloat16), (torch.bfloat16, torch.float32), (torch.float16, torch.float16), (torch.float16, torch.float32), (torch.float32, torch.float32), (torch.float32, torch.float16), (torch.float32, torch.bfloat16)])
def test_conv1d_bhl_fwd(b, h, l, k, dtype):
    torch.cuda.empty_cache() # empty cache between runs
    torch.manual_seed(42)
    device = 'cuda'
    in_dtype = dtype[0]
    w_dtype = dtype[1]
    
    padding =  (k -1)//2
                
    x = torch.randn([b, h, l], device=device, dtype=in_dtype)
    x_float = x.clone().to(torch.float32)
    
    conv1d_torch = nn.Conv1d(
        in_channels = h,
        out_channels = h,
        kernel_size = k,
        groups = h,
        padding = padding,
        dtype = w_dtype,
        device = device
    )
    
    conv1d_cuda = FlashDepthWiseConv1d(
        channels = h,
        kernel_size=k,
        padding=padding,
        weights=conv1d_torch.weight,
        bias=conv1d_torch.bias,
        dtype = w_dtype,
        device = device
    )
    
    with torch.autocast(device_type='cuda', dtype=in_dtype):
        y_torch = conv1d_torch(x)
    y_cuda = conv1d_cuda(x)
    with torch.autocast(device_type='cuda', dtype=in_dtype):
        y_cuda_autocast = conv1d_cuda(x_float).to(in_dtype)
    with torch.autocast(device_type='cuda', dtype=in_dtype):
        y_cuda_manual_cast = conv1d_cuda(x_float.to(in_dtype))
    
    assert torch.allclose(y_torch, y_cuda, atol=1e-1)
    assert torch.allclose(y_torch, y_cuda_autocast, atol=1e-1)
    assert torch.allclose(y_torch, y_cuda_manual_cast, atol=1e-1)
    
    
@pytest.mark.parametrize('b', [1, 2, 4, 8, 16])
@pytest.mark.parametrize('h', [768, 1024, 2048])
@pytest.mark.parametrize('l', [1024, 2048, 4096, 8192])
@pytest.mark.parametrize('k', [3, 5, 7])
@pytest.mark.parametrize('dtype', [(torch.bfloat16, torch.bfloat16), (torch.bfloat16, torch.float32), (torch.float16, torch.float16), (torch.float16, torch.float32), (torch.float32, torch.float32), (torch.float32, torch.float16), (torch.float32, torch.bfloat16)])
def test_conv1d_blh_fwd(b, h, l, k, dtype):
    torch.cuda.empty_cache() # empty cache between runs
    torch.manual_seed(42)
    device = 'cuda'
    in_dtype = dtype[0]
    w_dtype = dtype[1]
    
    padding =  (k -1)//2
    x = torch.randn([b, l, h], device=device, dtype=in_dtype)
    x_float = x.clone().to(torch.float32)
    
    conv1d_torch = nn.Conv1d(
        in_channels = h,
        out_channels = h,
        kernel_size = k,
        groups = h,
        padding = padding,
        dtype = w_dtype,
        device = device
    )
    
    conv1d_cuda = FlashDepthWiseConv1d(
        channels = h,
        kernel_size=k,
        padding=padding,
        weights=conv1d_torch.weight,
        bias=conv1d_torch.bias,
        is_bhl=False,
        dtype = w_dtype,
        device = device
    )
    
    
    x_torch = rearrange(x, 'b l h -> b h l').contiguous()
    
    with torch.autocast(device_type='cuda', dtype=in_dtype):
        y_torch = conv1d_torch(x_torch)
    
    y_cuda = conv1d_cuda(x)
    with torch.autocast(device_type='cuda', dtype=in_dtype):
        y_cuda_autocast = conv1d_cuda(x_float).to(in_dtype)
    with torch.autocast(device_type='cuda', dtype=in_dtype):
        y_cuda_manual_cast = conv1d_cuda(x_float.to(in_dtype))
    
    assert torch.allclose(y_torch, y_cuda.transpose(-1, -2), atol=1e-1)
    assert torch.allclose(y_torch, y_cuda_autocast.transpose(-1, -2), atol=1e-1)
    assert torch.allclose(y_torch, y_cuda_manual_cast.transpose(-1, -2), atol=1e-1) 

@pytest.mark.parametrize('b', [1, 2, 4, 8])
@pytest.mark.parametrize('d', [768, 1024, 2048])
@pytest.mark.parametrize('l', [1024, 2048, 4096, 8192])
@pytest.mark.parametrize('k', [3, 5, 7])
@pytest.mark.parametrize('dtype', [(torch.float16, torch.float16), (torch.float16, torch.float32), (torch.float32, torch.float32)])
def test_conv1d_bhl_bwd(b, d, l, k, dtype):
    torch.cuda.empty_cache() # empty cache between runs
    torch.manual_seed(42)
    device = 'cuda'
    in_dtype = dtype[0]
    w_dtype = dtype[1]
    
    padding =  (k -1)//2

    torch.set_default_device(device)
    torch.set_default_dtype(w_dtype)

    conv1d_torch = nn.Conv1d(
        in_channels = d,
        out_channels = d,
        kernel_size = k,
        groups = d,
        padding = padding
    ).to("cuda:0").to(w_dtype)

    conv1d_cuda = FlashDepthWiseConv1d(
        channels = d,
        kernel_size=k,
        padding=padding,
        weights=conv1d_torch.weight,
        bias=conv1d_torch.bias,
        is_bhl=True,
        dtype=w_dtype,
    ).to("cuda:0")

    x = torch.randn([b, d, l], device='cuda:0', dtype=in_dtype)
    x_wdtype = x.clone().to(w_dtype)
    x_cuda = x.clone().detach().requires_grad_(True)
    dout = torch.randn([b, d, l], device='cuda:0', dtype=in_dtype)
    dout_wdtype= dout.clone().to(w_dtype)

    x.requires_grad = True
    x_wdtype.requires_grad = True

    y_torch = conv1d_torch(x_wdtype)
    y_cuda = conv1d_cuda(x_cuda)

    y_torch.backward(dout_wdtype, retain_graph=True)
    y_cuda.backward(dout, retain_graph=True)

    assert torch.allclose(conv1d_cuda.bias.grad, conv1d_torch.bias.grad, atol=1)
    assert torch.allclose(conv1d_cuda.weights.grad.squeeze(), conv1d_torch.weight.grad.squeeze(), atol=1)
    assert torch.allclose(x_cuda.grad.to(w_dtype), x_wdtype.grad, atol=1)
    
    
@pytest.mark.parametrize('b', [1, 2, 4, 8])
@pytest.mark.parametrize('d', [768, 1024, 2048])
@pytest.mark.parametrize('l', [1024, 2048, 4096, 8192])
@pytest.mark.parametrize('k', [3, 5, 7])
@pytest.mark.parametrize('dtype', [(torch.float16, torch.float16), (torch.float16, torch.float32), (torch.float32, torch.float32)])
def test_conv1d_blh_bwd(b, d, l, k, dtype):
    torch.cuda.empty_cache() # empty cache between runs
    torch.manual_seed(42)
    device = 'cuda'
    in_dtype = dtype[0]
    w_dtype = dtype[1]
    
    padding =  (k -1)//2

    conv1d_torch = nn.Conv1d(
        in_channels = d,
        out_channels = d,
        kernel_size = k,
        groups = d,
        padding = padding,
        device = device,
        dtype = w_dtype
    ).to("cuda:0").to(w_dtype)

    conv1d_cuda = FlashDepthWiseConv1d(
        channels = d,
        kernel_size=k,
        padding=padding,
        weights=conv1d_torch.weight,
        bias=conv1d_torch.bias,
        is_bhl=False,
        device = device,
        dtype = w_dtype
    ).to("cuda:0")

    x_cuda = torch.randn([b, l, d], device='cuda:0', dtype=in_dtype)
    x = rearrange(x_cuda, 'b l d -> b d l').contiguous().to(w_dtype)

    dout_cuda = torch.randn([b, l, d], device='cuda:0', dtype=in_dtype)
    dout = rearrange(dout_cuda, 'b l d -> b d l').contiguous().to(w_dtype)

    x_cuda.requires_grad = True
    x.requires_grad = True

    y_torch = conv1d_torch(x)
    y_cuda = conv1d_cuda(x_cuda)

    y_torch.backward(dout, retain_graph=True)
    y_cuda.backward(dout_cuda, retain_graph=True)
    
    weights_cuda = rearrange(conv1d_torch.weight.squeeze(), 'd k -> k d').detach().clone().contiguous()
    weights = conv1d_torch.weight.squeeze()

    assert torch.allclose(conv1d_cuda.bias.grad, conv1d_torch.bias.grad, atol=1)
    assert torch.allclose(conv1d_cuda.weights.grad.squeeze().view(d, k), conv1d_torch.weight.grad.squeeze(), atol=1)
    assert torch.allclose(rearrange(x_cuda.grad, 'b l d -> b d  l'), x.grad.to(in_dtype), atol=1)