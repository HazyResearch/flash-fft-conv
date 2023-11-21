// Copyright (c) 2023 Dan Fu, Hermann Kumbong

// Simple 1D depthwise convolution implementation with dilation and stride = 1

#include "shared.h"

//For max perf, tune for your GPU and batch size, and datatype etc
const uint BX = 512;
const uint BY = 1;
const uint BZ = 1;

const uint TILE_SIZE_Y = 4;
const uint TILE_SIZE_X = 2;

// Trick to do padding in place without actually creating a new tensor
__forceinline__ __device__ __half2 get_u(const __half2 *__restrict__ u, uint L_eff, uint l, uint p, uint b, uint k, uint d, uint L, uint D, uint K)
{
    return l + k < p || l + k > L_eff - (p + 1) ? __float2half2_rn(0.0f) : u[b * L * D + (l + k - p) * D + d];
}


__forceinline__ __device__ __nv_bfloat162 get_u(const __nv_bfloat162 *__restrict__ u, uint L_eff, uint l, uint p, uint b, uint k, uint d, uint L, uint D, uint K)
{
    return l + k < p || l + k > L_eff - (p + 1) ? __float2bfloat162_rn(0.0f) : u[b * L * D + (l + k - p) * D + d];
}
 
__forceinline__ __device__ float2 get_u(const float2 *__restrict__ u, uint L_eff, uint l, uint p, uint b, uint k, uint d, uint L, uint D, uint K)
{
    return l + k < p || l + k > L_eff - (p + 1) ? make_float2(0.0f, 0.0f) : u[b * L * D + (l + k - p) * D + d];
}


//manually unrolling loop for k = 3 leads to good perf, can easily extend for other values of k if need be
template<typename T, typename U>
__forceinline__ __device__ T _conv1d_k_3(const T* u, const U* weights, const U* bias, T* out, uint padding, uint b, uint l, uint d, uint t, uint L, uint D, uint K, uint L_eff, uint L_out)
{

    T tmp;
    T weight;
    set_value(&tmp, bias[d]);

    set_value(&weight, weights[0 * D + d]);
    tmp = __hfma2(get_u(u, L_eff, l + t, padding, b, 0, d, L, D, K), weight, tmp);

    set_value(&weight, weights[1 * D + d]);
    tmp = __hfma2(get_u(u, L_eff, l + t, padding, b, 1, d, L, D, K), weight, tmp);

    set_value(&weight, weights[2 * D + d]);
    out[b * D * L_out  + (l + t) * D + d] = __hfma2(get_u(u, L_eff, l + t, padding, b, 2, d, L, D, K), weight, tmp);

}

template<typename T, typename U>
__global__ void conv1d_kernel_k_3(
    const T *__restrict__ u,
    const U *__restrict__ weights,
    const U *__restrict__ bias,
    T *__restrict__ out,
    uint padding,
    uint B,
    uint L,
    uint L_out,
    uint L_eff,
    uint D,
    uint K)
{
    const int d_block = blockIdx.x * blockDim.x * TILE_SIZE_X;
    const int l = blockIdx.y * blockDim.y * TILE_SIZE_Y + threadIdx.y * TILE_SIZE_Y;
    const int b = blockIdx.z * blockDim.z + threadIdx.z;

    int d;

    #pragma unroll
    for (int i = 0; i < TILE_SIZE_X; i++)
    {   
        d = d_block + threadIdx.x + i * BX;

        if (d < D && b < B){
            #pragma unroll
            for (int t = 0; t < TILE_SIZE_Y; t++){
                if (l + t < L_eff - K + 1)
                {
                    _conv1d_k_3(u, weights, bias, out, padding, b, l, d, t, L, D, K, L_eff, L_out);
                }
            }
        }
    }
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
    uint L_out,
    uint L_eff,
    uint D,
    uint K)
{
    const int d_block = blockIdx.x * blockDim.x * TILE_SIZE_X;
    const int l = blockIdx.y * blockDim.y * TILE_SIZE_Y + threadIdx.y * TILE_SIZE_Y;
    const int b = blockIdx.z * blockDim.z + threadIdx.z;

    int d;
    T tmp;
    T weight;

    #pragma unroll
        for (int i = 0; i < TILE_SIZE_X; i++)
        {   
            d = d_block + threadIdx.x + i * BX;
            
            if (d < D && b < B){
                #pragma unroll
                for (int t = 0; t < TILE_SIZE_Y; t++){
                    if (l + t < L_eff - K + 1)
                    {
                        set_value(&tmp, bias[d]);

                        for(int k = 0; k < K; k++){
                            set_value(&weight, weights[k * D + d]);

                            tmp = __hfma2(get_u(u, L_eff, l + t, padding, b, k, d, L, D, K), weight, tmp);
                        }
                            out[b * D * L_out  + (l + t) * D + d] = tmp;
                    }
                }
            }
        }
}

torch::Tensor conv1d_cuda_blh(
    torch::Tensor u,
    torch::Tensor weight,
    torch::Tensor bias,
    uint padding)
{
    const uint b = u.size(0);
    const uint l = u.size(1);
    const uint d = u.size(2);

    const uint k = weight.size(0);

    uint l_eff = l + 2 * padding;

    

    dim3 blockDims(BX, BY, BZ);

    dim3 gridDims(ceil(d * 1.0 / (BX * TILE_SIZE_X * 2) ), ceil((l_eff - k + 1) * 1.0 / (BY * TILE_SIZE_Y)), ceil(b * 1.0 / BZ));


    uint l_out = (l + 2 * padding - k + 1);

    torch::Tensor out = torch::empty({b, l_out, d}, u.options());

    //calling seperate kernels for k=3 and k!=3 leads to better perf
    if(k==3){
         DISPATCH_FLOAT2_AND_HALF2_AND_BF162(u.scalar_type(), weight.scalar_type(),
        "depthwise conv 1d fwd blh",
        ([&]
            { conv1d_kernel_k_3<input_t, weight_t><<<gridDims, blockDims>>>(
                    static_cast<input_t *>(u.data_ptr()),
                    static_cast<weight_t *>(weight.data_ptr()),
                    static_cast<weight_t *>(bias.data_ptr()),
                    static_cast<input_t *>(out.data_ptr()),
                    padding,
                    b,
                    l,
                    l_out,
                    l_eff,
                    ceil(d/2),
                    k);
            }
        )
    );
    }else{
       DISPATCH_FLOAT2_AND_HALF2_AND_BF162(u.scalar_type(), weight.scalar_type(),
        "depthwise conv 1d fwd blh",
        ([&]
            { conv1d_kernel<input_t, weight_t><<<gridDims, blockDims>>>(
                    static_cast<input_t *>(u.data_ptr()),
                    static_cast<weight_t *>(weight.data_ptr()),
                    static_cast<weight_t *>(bias.data_ptr()),
                    static_cast<input_t *>(out.data_ptr()),
                    padding,
                    b,
                    l,
                    l_out,
                    l_eff,
                    ceil(d/2),
                    k);
            }
        )
    );
    }
    return out;
}