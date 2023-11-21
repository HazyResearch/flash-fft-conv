// Copyright (c) 2023 Dan Fu, Hermann Kumbong

#include "shared.h"

const uint BX = 128;
const uint BY = 1;
const uint BZ = 1;

template <typename input_t, typename weight_t>
__global__ void conv1d_backward_kernel(
    const input_t* __restrict__ dout,
    int dout_stride0,
    int dout_stride1,
    int dout_stride2,
    const input_t* __restrict__ u,
    const weight_t* __restrict__ weights,
    int weights_stride0,
    int weights_stride1,
    input_t* __restrict__ du,
    input_t* __restrict__ dk,
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
    if(b < B && d < D && l == 0){
        for(int j = threadIdx.x; j < L; j += blockDim.x)
        {
            input_t sum;
            set_value(&sum, 0.0f);
            input_t weight;

            for(int k = 0; k < K ; k++)
            {
                int idx = - P + k + j;

                if(idx >= 0 && idx < L){
                    set_value(&weight, weights[d * weights_stride1 + (K - (k +1)) * weights_stride0]);
                    sum = __hfma(dout[b * dout_stride0 + d * dout_stride1 + idx * dout_stride2], weight, sum);
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
                set_value(&dk[b * D * K * L + d * K * L + k * L + j], 0.0f);
            }else{
                set_value(&dk[b * D * K * L + d * K * L + k * L + j], u[b * D * L + (k - P + j) * D + d]);
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
    torch::Tensor dk = torch::empty({b, d, k, l}, u.options());
    torch::Tensor dbias = dout.sum(-2).sum(0);
    dout = dout.transpose(-1,-2);

    DISPATCH_FLOAT_AND_HALF_AND_BF16(dout.scalar_type(), weight.scalar_type(),
        "depthwise conv 1d backward blh",
        ([&]
            { conv1d_backward_kernel<input_t, weight_t><<<gridDims, blockDims>>>(
                    static_cast<input_t *>(dout.data_ptr()),
                    dout.stride(0),
                    dout.stride(1),
                    dout.stride(2),
                    static_cast<input_t *>(u.data_ptr()),
                    static_cast<weight_t *>(weight.data_ptr()),
                    weight.stride(0),
                    weight.stride(1),
                    static_cast<input_t *>(du.data_ptr()),
                    static_cast<input_t *>(dk.data_ptr()),
                    b,
                    l,
                    d,
                    k,
                    padding); 
            }
        )
    );

    return {du, torch::matmul(dk, dout.unsqueeze(-1)).squeeze(-1).sum(0).view({k, d}).to(weight.dtype()), dbias};
}
