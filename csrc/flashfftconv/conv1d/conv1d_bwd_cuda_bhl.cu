// Copyright (c) 2023 Dan Fu, Hermann Kumbong
#include "shared.h"

const uint BX = 128;
const uint BY = 1;
const uint BZ = 1;

const uint TILE_SIZE = 4;

template <typename input_t, typename weight_t>
__global__ void conv1d_backward_kernel(
    const input_t* __restrict__ dout,
    const input_t* __restrict__ u,
    const weight_t* __restrict__ weights,
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
                    set_value(&weight, weights[d * K + K - (k +1)]);
                    sum = __hfma(dout[b * D * L + d * L + idx], weight, sum);
                }
            }
            du[b * D * L + d * L + j] = sum;
        }
    }

    const int k = blockIdx.x;
    input_t tmp;
    //construct the dk matrix
    if(b < B && d < D && k < K)
    {
        for(int j = threadIdx.x; j < L; j += blockDim.x)
        {
            if(k - P + j < 0 || k - P + j >= L){
                set_value(&dk[b * D * K * L + d * K * L + k * L + j], 0.0f);

            }else{
                set_value(&dk[b * D * K * L + d * K * L + k * L + j], u[b * D * L + d * L + k - P + j]);
            }
        }
    }

}

std::vector<torch::Tensor> conv1d_backward_bhl_cuda(
    torch::Tensor dout,
    torch::Tensor u,
    torch::Tensor weight,
    torch::Tensor bias,
    uint padding)
{
    const uint b = u.size(0);
    const uint d = u.size(1);
    const uint l = u.size(2);

    const uint k = weight.squeeze().size(1);
    
    dim3 blockDims(BX, 1, 1);

    dim3 gridDims(l, d, b);

    torch::Tensor du = torch::empty({b, d, l}, u.options());
    torch::Tensor dk = torch::empty({b, d, k, l}, dout.options());
    torch::Tensor dbias = dout.sum(-1).sum(0);

    DISPATCH_FLOAT_AND_HALF_AND_BF16(dout.scalar_type(), weight.scalar_type(),
        "depthwise conv 1d backward bhl",
        ([&]
            { conv1d_backward_kernel<input_t, weight_t><<<gridDims, blockDims>>>(
                    static_cast<input_t *>(dout.data_ptr()),
                    static_cast<input_t *>(u.data_ptr()),
                    static_cast<weight_t *>(weight.data_ptr()),
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
    return {du, torch::matmul(dk, dout.unsqueeze(-1)).squeeze(-1).sum(0).to(weight.type()), dbias};
}