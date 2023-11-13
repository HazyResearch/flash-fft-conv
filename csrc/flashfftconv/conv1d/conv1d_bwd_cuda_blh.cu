// Copyright (c) 2023 Dan Fu, Hermann Kumbong

#include <torch/extension.h>
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <vector>

const uint BX = 128;
const uint BY = 1;
const uint BZ = 1;

template <typename scalar_t>
__global__ void conv1d_backward_kernel(
    const scalar_t* __restrict__ dout,
    int dout_stride0,
    int dout_stride1,
    int dout_stride2,
    const scalar_t* __restrict__ u,
    const scalar_t* __restrict__ weights,
    int weights_stride0,
    int weights_stride1,
    scalar_t* __restrict__ du,
    scalar_t* __restrict__ dk,
    uint B,
    uint L,
    uint D,
    uint K,
    uint P
    )
{
    const int b = blockIdx.z;
    const int d = blockIdx.y;
    const int l = blockIdx.x; 

    //construct the du matrix
    //construct the du matrix
    if(b < B && d < D && l == 0){
        for(int j = threadIdx.x; j < L; j += blockDim.x)
        {
            scalar_t sum = 0;

            for(int k = 0; k < K ; k++)
            {
                int idx = - P + k + j;

                if(idx >= 0 && idx < L){
                    sum += dout[b * dout_stride0 + d * dout_stride1 + idx * dout_stride2] * weights[d * weights_stride1 + (K - (k +1)) * weights_stride0];
                }
            }
            du[b * D * L + j * D + d] = sum;
        }
    }

    const int k = blockIdx.x;
    //construct the dk matrix
    if(b < B && d < D && k < K)
    {
        for(int j = threadIdx.x; j < L; j += blockDim.x)
        {
            if(k - P + j < 0 || k - P + j >= L){
                dk[b * D * K * L + d * K * L + k * L + j] = 0;
            }else{
                dk[b * D * K * L + d * K * L + k * L + j] = u[b * D * L + (k - P + j) * D + d];
            }
        }
    }

}

std::vector<torch::Tensor> conv1d_backward_blh_cuda(
    torch::Tensor dout,
    torch::Tensor u,
    torch::Tensor weight,
    torch::Tensor bias,
    uint padding)
{
    const uint b = u.size(0);
    const uint l = u.size(1);
    const uint d = u.size(2);


    const uint k = weight.squeeze().size(0);
    
    dim3 blockDims(BX, 1, 1);

    dim3 gridDims(l, d, b);

    torch::Tensor du = torch::empty({b, l, d}, u.options());
    torch::Tensor dk = torch::empty({b, d, k, l}, weight.options());
    torch::Tensor dbias = dout.sum(-2).sum(0);
    dout = dout.transpose(-1,-2);

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::kHalf, at::kBFloat16, u.type(),
        "depthwise conv 1d backward",
        ([&]
            { conv1d_backward_kernel<scalar_t><<<gridDims, blockDims>>>(
                    dout.data<scalar_t>(),
                    dout.stride(0),
                    dout.stride(1),
                    dout.stride(2),
                    u.data<scalar_t>(),
                    weight.data<scalar_t>(),
                    weight.stride(0),
                    weight.stride(1),
                    du.data<scalar_t>(),
                    dk.data<scalar_t>(),
                    b,
                    l,
                    d,
                    k,
                    padding); 
            }
        )
    );
    return {du, torch::matmul(dk, dout.unsqueeze(-1)).squeeze(-1).sum(0).view({k, d}), dbias};
}
