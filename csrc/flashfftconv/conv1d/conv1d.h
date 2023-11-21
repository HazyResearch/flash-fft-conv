// Copyright (c) 2023 Dan Fu, Hermann Kumbong

#include <torch/extension.h>

#include <vector>


#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_IS_HALF_OR_BFLOAT_OR_FLOAT(x) TORCH_CHECK(x.dtype() == torch::kFloat16 || x.dtype() == torch::kBFloat16 || x.dtype() == torch::kFloat32, #x " must be float16 or bfloat16 or float32")
#define CHECK_SAME_TYPE(x, y) TORCH_CHECK(x.dtype() == y.dtype(), #x " and " #y " must have the same dtype")

#define CHECK_INPUT(x) \
    CHECK_CUDA(x);     \
    CHECK_CONTIGUOUS(x); \
    CHECK_IS_HALF_OR_BFLOAT_OR_FLOAT(x)

torch::Tensor conv1d_cuda_bhl(
    torch::Tensor u,
    torch::Tensor weight,
    torch::Tensor bias,
    uint padding);

torch::Tensor conv1d_cuda_blh(
    torch::Tensor u,
    torch::Tensor weight,
    torch::Tensor bias,
    uint padding);

std::vector<torch::Tensor> conv1d_backward_bhl_cuda(
    torch::Tensor dout,
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    uint padding
);

std::vector<torch::Tensor> conv1d_backward_blh_cuda(
    torch::Tensor dout,
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    uint padding
);


torch::Tensor conv1d_fwd(
    torch::Tensor u,
    torch::Tensor weight,
    torch::Tensor bias,
    uint padding,
    bool is_bhl)
{
    CHECK_INPUT(u);
    CHECK_INPUT(weight);
    CHECK_INPUT(bias);
    CHECK_SAME_TYPE(weight, bias);

    int k;

    if(is_bhl){
        k = weight.size(1);
    }else{
        k = weight.size(0);
    }
 
    TORCH_CHECK(k % 2 == 1, "Filter size must be odd number");

    if(is_bhl){
        return conv1d_cuda_bhl(u, weight, bias, padding);
    }else{
        return conv1d_cuda_blh(u, weight, bias, padding);
    }
}

std::vector<torch::Tensor> conv1d_bwd(
    torch::Tensor dout,
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    uint padding,
    bool is_bhl)
{
    CHECK_INPUT(dout);
    CHECK_INPUT(input);
    CHECK_INPUT(weight);
    CHECK_INPUT(bias);
    CHECK_SAME_TYPE(weight, bias);
    CHECK_SAME_TYPE(dout, input);
  
    if(is_bhl){
        return conv1d_backward_bhl_cuda(dout, input, weight, bias, padding);
    } else{
        return conv1d_backward_blh_cuda(dout, input, weight, bias, padding);
    }
}