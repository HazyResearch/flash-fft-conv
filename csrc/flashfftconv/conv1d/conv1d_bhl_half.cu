// Copyright (c) 2023 Dan Fu, Hermann Kumbong

// Simple 1D depthwise convolution implementation with dilation and stride = 1

#include <torch/extension.h>
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <vector>
#include <cuda_fp16.h>

const uint BX = 256;
const uint BY = 1;
const uint BZ = 1;

const uint TILE_SIZE_L = 4;
const uint TILE_SIZE_D = 1;

__forceinline__ __device__ __half _conv1d_k_3(const __half* u, const __half* weights, const __half* bias, uint padding, uint l, uint d, uint L, uint D, uint K)
{
    __half tmp = bias[d];

    int idx = l - padding;

    if(idx >= 0 && idx < L){
        tmp = __hfma(u[d * L + idx], weights[0], tmp);
    }
    
    idx++;
    if(idx >= 0 && idx < L){
        tmp = __hfma(u[d * L + idx], weights[1], tmp);
    }

    idx++;
    if(idx >= 0 && idx < L){
        tmp = __hfma(u[d * L + idx], weights[2], tmp);
    }

    return tmp;
}

__global__ void conv1d_kernel(
    const __half *__restrict__ u,
    const __half *__restrict__ weights,
    const __half *__restrict__ bias,
    __half *__restrict__ out,
    uint padding,
    uint B,
    uint L,
    uint D,
    uint K,
    uint L_out
    )
{
    const int b = blockIdx.z * blockDim.z + threadIdx.z;
    const int d = blockIdx.y * blockDim.y * TILE_SIZE_D + threadIdx.y;
    const int l_offset = blockIdx.x * blockDim.x * TILE_SIZE_L + threadIdx.x;
    
    __half tmp; 
    int idx;
    int l;

        for(int l_tile = 0; l_tile < TILE_SIZE_L; l_tile++){
            l = l_offset + l_tile * blockDim.x;

            tmp = bias[d];
            if(d < D && l < L_out && b < B){
                if(K == 3){
                    out[b * L_out * D + d * L_out + l] = _conv1d_k_3(u + b * L * D, weights + d * K, bias, padding, l, d, L, D, K);
                } else{
                    for(int k = 0; k < K; k++){
                        idx = l - padding + k;
                        if(idx >= 0 && idx < L){
                            tmp = __hfma(u[b * L_out * D + d * L + idx], weights[d * K + k], tmp);
                        }
                    }
                    out[b * L_out * D + d * L_out + l] = tmp;
                
                }
            }
        }
    
}

torch::Tensor conv1d_cuda_bhl_half(
    torch::Tensor u,
    torch::Tensor weight,
    torch::Tensor bias,
    uint padding)
{
    const uint b = u.size(0);
    const uint d = u.size(1);
    const uint l = u.size(2);


    const uint k = weight.size(1);

    uint l_out = (l + 2 * padding - k + 1);
    
    dim3 blockDims(BX, BY, BZ);

    dim3 gridDims(ceil(l_out * 1.0 / (BX * TILE_SIZE_L) ), ceil((d * 1.0) / (BY * TILE_SIZE_D)), ceil((b * 1.0) / BZ));

    torch::Tensor out = torch::empty({b, d, l_out}, u.options());

    cudaFuncSetCacheConfig(conv1d_kernel, cudaFuncCachePreferL1);

    conv1d_kernel<<<gridDims, blockDims>>>(
        static_cast<__half *>(u.data_ptr()),
        static_cast<__half *>(weight.data_ptr()),
        static_cast<__half *>(bias.data_ptr()),
        static_cast<__half *>(out.data_ptr()),
        padding,
        b,
        l,
        d,
        k,
        l_out
        );

    return out;
}