// Copyright (c) 2023 Dan Fu, Hermann Kumbong

#include <torch/extension.h>

#include <vector>


#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_IS_HALF_OR_BFLOAT(x) TORCH_CHECK(x.dtype() == torch::kFloat16 || x.dtype() == torch::kBFloat16, #x " must be float16 or bfloat16")
#define CHECK_INPUT(x) \
    CHECK_CUDA(x);     \
    CHECK_CONTIGUOUS(x); \
    CHECK_IS_HALF_OR_BFLOAT(x)

torch::Tensor conv1d_cuda_bhl_half(
    torch::Tensor u,
    torch::Tensor weight,
    torch::Tensor bias,
    uint padding);

torch::Tensor conv1d_cuda_bhl_bf16(
    torch::Tensor u,
    torch::Tensor weight,
    torch::Tensor bias,
    uint padding);

torch::Tensor conv1d_cuda_blh_half(
    torch::Tensor u,
    torch::Tensor weight,
    torch::Tensor bias,
    uint padding);

torch::Tensor conv1d_cuda_blh_bf16(
    torch::Tensor u,
    torch::Tensor weight,
    torch::Tensor bias,
    uint padding);


// std::vector<torch::Tensor> conv1d_backward_cuda(
//     torch::Tensor grad_output,
//     torch::Tensor input,
//     torch::Tensor weight,
//     torch::Tensor bias,
//     uint padding
// );


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

    int k;

    if(is_bhl){
        k = weight.size(1);
    }else{
        k = weight.size(0);
    }
 
    TORCH_CHECK(k % 2 == 1, "Filter size must be odd number");

    if(is_bhl){
        if(u.dtype() == torch::kFloat16){
            return conv1d_cuda_bhl_half(u, weight, bias, padding);
        }else{
            return conv1d_cuda_bhl_bf16(u, weight, bias, padding);
        }
    }else{
        if(u.dtype() == torch::kFloat16){
            return conv1d_cuda_blh_half(u, weight, bias, padding);
        }else{
            return conv1d_cuda_blh_bf16(u, weight, bias, padding);
        }
    }
}

// std::vector<torch::Tensor> backward(
//     torch::Tensor grad_output,
//     torch::Tensor input,
//     torch::Tensor weight,
//     torch::Tensor bias,
//     uint padding)
// {
//     CHECK_INPUT(grad_output);
//     CHECK_INPUT(input);
//     CHECK_INPUT(weight);
//     CHECK_INPUT(bias);

//     return conv1d_backward_cuda(grad_output, input, weight, bias, padding);
// }

// PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
// {
//     m.def("forward", &conv1d, "short_filter forward (CUDA)");
//     // m.def("backward", &backward, "1D depthwise convolution backward (CUDA)");
// }