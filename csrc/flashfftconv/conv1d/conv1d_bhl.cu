// Copyright (c) 2023 Dan Fu, Hermann Kumbong

// Simple 1D depthwise convolution implementation with dilation and stride = 1
#include "shared.h"

const uint BX = 256;
const uint BY = 1;
const uint BZ = 1;

const uint TILE_SIZE_L = 4;
const uint TILE_SIZE_D = 1;

template<typename T, typename U>
__forceinline__ __device__ T _conv1d_k_3(const T* u, const U* weights, const U* bias, uint padding, uint l, uint d, uint L, uint D, uint K)
{
    T tmp;
    T weight;

    set_value(&tmp, bias[d]);

    int idx = l - padding;

    if(idx >= 0 && idx < L){
        set_value(&weight, weights[0]);
        tmp = __hfma(u[d * L + idx], weight, tmp);
    }
    
    idx++;
    if(idx >= 0 && idx < L){
        set_value(&weight, weights[1]);
        tmp = __hfma(u[d * L + idx], weight, tmp);
    }

    idx++;
    if(idx >= 0 && idx < L){
        set_value(&weight, weights[2]);
        tmp = __hfma(u[d * L + idx], weight, tmp);
    }

    return tmp;
}

template<typename T, typename U>
__global__ void conv1d_kernel(
    const T *__restrict__ u,
    const U *__restrict__ weights,
    const U *__restrict__ bias,
    T *__restrict__ out,
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
    
    T tmp; 
    T weight;

    int idx;
    int l;

    for(int l_tile = 0; l_tile < TILE_SIZE_L; l_tile++){
        l = l_offset + l_tile * blockDim.x;

        set_value(&tmp, bias[d]);

        if(d < D && l < L_out && b < B){
            if(K == 3){
                out[b * L_out * D + d * L_out + l] = _conv1d_k_3(u + b * L * D, weights + d * K, bias, padding, l, d, L, D, K);
            } else{
                for(int k = 0; k < K; k++){
                    idx = l - padding + k;
                    if(idx >= 0 && idx < L){
                        set_value(&weight, weights[d * K + k]);
                        tmp = __hfma(u[b * L_out * D + d * L + idx], weight, tmp);
                    }
                }
                out[b * L_out * D + d * L_out + l] = tmp;
            
            }
        }
    }
    
}

torch::Tensor conv1d_cuda_bhl(
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

    DISPATCH_FLOAT_AND_HALF_AND_BF16(u.scalar_type(), weight.scalar_type(),
        "depthwise conv 1d fwd bhl",
        ([&]
            { conv1d_kernel<input_t, weight_t><<<gridDims, blockDims>>>(
                    static_cast<input_t *>(u.data_ptr()),
                    static_cast<weight_t *>(weight.data_ptr()),
                    static_cast<weight_t *>(bias.data_ptr()),
                    static_cast<input_t *>(out.data_ptr()),
                    padding,
                    b,
                    l,
                    d,
                    k,
                    l_out
                    ); 
            }
        )
    );

    return out;
}