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
#define CHECK_SHAPE(x, ...) TORCH_CHECK(x.sizes() == torch::IntArrayRef({__VA_ARGS__}), #x " must have shape (" #__VA_ARGS__ ")")


std::vector<torch::Tensor> butterfly_cuda(
    torch::Tensor x,
    torch::Tensor d_f_T,
    torch::Tensor twiddle_factors_real,
    torch::Tensor twiddle_factors_imag,
    std::optional<at::Tensor> x_gate = std::nullopt
);


std::vector<torch::Tensor> butterfly_bf16_cuda(
    torch::Tensor x,
    torch::Tensor d_f_T_real,
    torch::Tensor d_f_T_imag,
    torch::Tensor twiddle_factors_real,
    torch::Tensor twiddle_factors_imag,
    std::optional<at::Tensor> out_gate = std::nullopt
);


std::vector<torch::Tensor> butterfly_padded_cuda(
    torch::Tensor x,
    torch::Tensor d_f_T,
    torch::Tensor twiddle_factors_real,
    torch::Tensor twiddle_factors_imag,
    int M,
    std::optional<at::Tensor> x_gate = std::nullopt
);


std::vector<torch::Tensor> butterfly_padded_bf16_cuda(
    torch::Tensor x,
    torch::Tensor d_f_T_real,
    torch::Tensor d_f_T_imag,
    torch::Tensor twiddle_factors_real,
    torch::Tensor twiddle_factors_imag,
    int M,
    std::optional<at::Tensor> x_gate = std::nullopt
);

torch::Tensor butterfly_ifft_cuda(
    torch::Tensor x_real,
    torch::Tensor x_imag,
    torch::Tensor d_f_T,
    torch::Tensor twiddle_factors_real,
    torch::Tensor twiddle_factors_imag,
    std::optional<at::Tensor> out_gate = std::nullopt
);

torch::Tensor butterfly_ifft_bf16_cuda(
    torch::Tensor x_real,
    torch::Tensor x_imag,
    torch::Tensor d_f_real,
    torch::Tensor d_f_imag,
    torch::Tensor twiddle_factors_real,
    torch::Tensor twiddle_factors_imag,
    std::optional<at::Tensor> x_gate = std::nullopt
);

torch::Tensor butterfly_ifft_padded_cuda(
    torch::Tensor x_real,
    torch::Tensor x_imag,
    torch::Tensor d_f,
    torch::Tensor twiddle_factors_real,
    torch::Tensor twiddle_factors_imag,
    int N,
    std::optional<at::Tensor> out_gate = std::nullopt
);


torch::Tensor butterfly_ifft_padded_bf16_cuda(
    torch::Tensor x_real,
    torch::Tensor x_imag,
    torch::Tensor d_f_real,
    torch::Tensor d_f_imag,
    torch::Tensor twiddle_factors_real,
    torch::Tensor twiddle_factors_imag,
    int N,
    std::optional<at::Tensor> out_gate = std::nullopt
);

std::vector<torch::Tensor> butterfly(
    torch::Tensor x,
    torch::Tensor d_f_T,
    torch::Tensor twiddle_factors_real,
    torch::Tensor twiddle_factors_imag
){
    CHECK_INPUT(x);
    CHECK_INPUT(twiddle_factors_real);
    CHECK_INPUT(twiddle_factors_imag);  
    

    return butterfly_cuda(x, d_f_T, twiddle_factors_real, twiddle_factors_imag);
}

std::vector<torch::Tensor> butterfly_gated(
    torch::Tensor x,
    torch::Tensor d_f_T,
    torch::Tensor twiddle_factors_real,
    torch::Tensor twiddle_factors_imag,
    torch::Tensor x_gate
){
    CHECK_INPUT(x);
    CHECK_INPUT(twiddle_factors_real);
    CHECK_INPUT(twiddle_factors_imag);
    
    CHECK_INPUT(x_gate);

    return butterfly_cuda(x, d_f_T, twiddle_factors_real, twiddle_factors_imag, x_gate);
}

std::vector<torch::Tensor> butterfly_bf16(
    torch::Tensor x,
    torch::Tensor d_f_T_real,
    torch::Tensor d_f_T_imag,
    torch::Tensor twiddle_factors_real,
    torch::Tensor twiddle_factors_imag
){
    CHECK_INPUT(x);
    CHECK_INPUT(twiddle_factors_real);
    CHECK_INPUT(twiddle_factors_imag);
    CHECK_INPUT(d_f_T_real);
    CHECK_INPUT(d_f_T_imag);


    return butterfly_bf16_cuda(x, d_f_T_real, d_f_T_imag, twiddle_factors_real, twiddle_factors_imag);
}

std::vector<torch::Tensor> butterfly_gated_bf16(
    torch::Tensor x,
    torch::Tensor d_f_T_real,
    torch::Tensor d_f_T_imag,
    torch::Tensor twiddle_factors_real,
    torch::Tensor twiddle_factors_imag,
    torch::Tensor x_gate
){
    CHECK_INPUT(x);
    CHECK_INPUT(twiddle_factors_real);
    CHECK_INPUT(twiddle_factors_imag);
    CHECK_INPUT(d_f_T_real);
    CHECK_INPUT(d_f_T_imag);
    CHECK_INPUT(x_gate);


    return butterfly_bf16_cuda(x, d_f_T_real, d_f_T_imag, twiddle_factors_real, twiddle_factors_imag, x_gate);
}

torch::Tensor butterfly_ifft(
    torch::Tensor x_real,
    torch::Tensor x_imag,
    torch::Tensor d_f_T,
    torch::Tensor twiddle_factors_real,
    torch::Tensor twiddle_factors_imag
){
    CHECK_INPUT(x_real);
    CHECK_INPUT(x_imag);
    CHECK_INPUT(twiddle_factors_real);
    CHECK_INPUT(twiddle_factors_imag);
    
    return butterfly_ifft_cuda(x_real, x_imag, d_f_T, twiddle_factors_real, twiddle_factors_imag);
}


torch::Tensor butterfly_ifft_gated(
    torch::Tensor x_real,
    torch::Tensor x_imag,
    torch::Tensor d_f_T,
    torch::Tensor twiddle_factors_real,
    torch::Tensor twiddle_factors_imag,
    torch::Tensor out_gate
){
    CHECK_INPUT(x_real);
    CHECK_INPUT(x_imag);
    CHECK_INPUT(twiddle_factors_real);
    CHECK_INPUT(twiddle_factors_imag);
    CHECK_INPUT(out_gate);

    return butterfly_ifft_cuda(x_real, x_imag, d_f_T, twiddle_factors_real, twiddle_factors_imag, out_gate);
}

torch::Tensor butterfly_ifft_bf16(
    torch::Tensor x_real,
    torch::Tensor x_imag,
    torch::Tensor d_f_real,
    torch::Tensor d_f_imag,
    torch::Tensor twiddle_factors_real,
    torch::Tensor twiddle_factors_imag
){
    CHECK_INPUT(x_real);
    CHECK_INPUT(x_imag);
    CHECK_INPUT(d_f_real);
    CHECK_INPUT(d_f_imag);
    CHECK_INPUT(twiddle_factors_real);
    CHECK_INPUT(twiddle_factors_imag);


    return butterfly_ifft_bf16_cuda(x_real, x_imag, d_f_real, d_f_imag, twiddle_factors_real, twiddle_factors_imag);
}


torch::Tensor butterfly_ifft_gated_bf16(
    torch::Tensor x_real,
    torch::Tensor x_imag,
    torch::Tensor d_f_real,
    torch::Tensor d_f_imag,
    torch::Tensor twiddle_factors_real,
    torch::Tensor twiddle_factors_imag,
    torch::Tensor out_gate
){
    CHECK_INPUT(x_real);
    CHECK_INPUT(x_imag);
    CHECK_INPUT(d_f_real);
    CHECK_INPUT(d_f_imag);
    CHECK_INPUT(twiddle_factors_real);
    CHECK_INPUT(twiddle_factors_imag);
    CHECK_INPUT(out_gate);

    return butterfly_ifft_bf16_cuda(x_real, x_imag, d_f_real, d_f_imag, twiddle_factors_real, twiddle_factors_imag, out_gate);
}

std::vector<torch::Tensor> butterfly_padded(
    torch::Tensor x,
    torch::Tensor d_f_T,
    torch::Tensor twiddle_factors_real,
    torch::Tensor twiddle_factors_imag,
    int M
){
    CHECK_INPUT(x);
    CHECK_INPUT(twiddle_factors_real);
    CHECK_INPUT(twiddle_factors_imag);
    

    return butterfly_padded_cuda(x, d_f_T, twiddle_factors_real, twiddle_factors_imag, M);
}

std::vector<torch::Tensor> butterfly_padded_bf16(
    torch::Tensor x,
    torch::Tensor d_f_T_real,
    torch::Tensor d_f_T_imag,   
    torch::Tensor twiddle_factors_real,
    torch::Tensor twiddle_factors_imag,
    int M
){
    CHECK_INPUT(x);
    CHECK_INPUT(twiddle_factors_real);
    CHECK_INPUT(twiddle_factors_imag);
    

    return butterfly_padded_bf16_cuda(x, d_f_T_real, d_f_T_imag, twiddle_factors_real, twiddle_factors_imag, M);
}


std::vector<torch::Tensor> butterfly_padded_gated(
    torch::Tensor x,
    torch::Tensor d_f_T,
    torch::Tensor twiddle_factors_real,
    torch::Tensor twiddle_factors_imag,
    int M,
    torch::Tensor x_gate
){
    CHECK_INPUT(x);
    CHECK_INPUT(twiddle_factors_real);
    CHECK_INPUT(twiddle_factors_imag);
    

    return butterfly_padded_cuda(x, d_f_T, twiddle_factors_real, twiddle_factors_imag, M, x_gate);
}

std::vector<torch::Tensor> butterfly_padded_gated_bf16(
    torch::Tensor x,
    torch::Tensor d_f_T_real,
    torch::Tensor d_f_T_imag,
    torch::Tensor twiddle_factors_real,
    torch::Tensor twiddle_factors_imag,
    int M,
    torch::Tensor x_gate
){
    CHECK_INPUT(x);
    CHECK_INPUT(d_f_T_real);
    CHECK_INPUT(d_f_T_imag);
    CHECK_INPUT(twiddle_factors_real);
    CHECK_INPUT(twiddle_factors_imag);
    

    return butterfly_padded_bf16_cuda(x, d_f_T_real, d_f_T_imag, twiddle_factors_real, twiddle_factors_imag, M, x_gate);
}

torch::Tensor butterfly_ifft_padded(
    torch::Tensor x_real,
    torch::Tensor x_imag,
    torch::Tensor d_f,
    torch::Tensor twiddle_factors_real,
    torch::Tensor twiddle_factors_imag,
    int N
){
    CHECK_INPUT(x_real);
    CHECK_INPUT(x_imag);
    CHECK_INPUT(twiddle_factors_real);
    CHECK_INPUT(twiddle_factors_imag);

    return butterfly_ifft_padded_cuda(x_real, x_imag, d_f, twiddle_factors_real, twiddle_factors_imag, N);
}

torch::Tensor butterfly_ifft_padded_gated(
    torch::Tensor x_real,
    torch::Tensor x_imag,
    torch::Tensor d_f,
    torch::Tensor twiddle_factors_real,
    torch::Tensor twiddle_factors_imag,
    int N,
    torch::Tensor out_gate
){
    CHECK_INPUT(x_real);
    CHECK_INPUT(x_imag);
    CHECK_INPUT(twiddle_factors_real);
    CHECK_INPUT(twiddle_factors_imag);

    return butterfly_ifft_padded_cuda(x_real, x_imag, d_f, twiddle_factors_real, twiddle_factors_imag, N, out_gate);
}


torch::Tensor butterfly_ifft_padded_bf16(
    torch::Tensor x_real,
    torch::Tensor x_imag,
    torch::Tensor d_f_real,
    torch::Tensor d_f_imag,
    torch::Tensor twiddle_factors_real,
    torch::Tensor twiddle_factors_imag,
    int N
){
    CHECK_INPUT(x_real);
    CHECK_INPUT(x_imag);
    CHECK_INPUT(d_f_real);
    CHECK_INPUT(d_f_imag);
    CHECK_INPUT(twiddle_factors_real);
    CHECK_INPUT(twiddle_factors_imag);

    return butterfly_ifft_padded_bf16_cuda(x_real, x_imag, d_f_real, d_f_imag, twiddle_factors_real, twiddle_factors_imag, N);
}

torch::Tensor butterfly_ifft_padded_gated_bf16(
    torch::Tensor x_real,
    torch::Tensor x_imag,
    torch::Tensor d_f_real,
    torch::Tensor d_f_imag,
    torch::Tensor twiddle_factors_real,
    torch::Tensor twiddle_factors_imag,
    int N,
    torch::Tensor out_gate
){
    CHECK_INPUT(x_real);
    CHECK_INPUT(x_imag);
    CHECK_INPUT(d_f_real);
    CHECK_INPUT(d_f_imag);
    CHECK_INPUT(twiddle_factors_real);
    CHECK_INPUT(twiddle_factors_imag);

    return butterfly_ifft_padded_bf16_cuda(x_real, x_imag, d_f_real, d_f_imag, twiddle_factors_real, twiddle_factors_imag, N, out_gate);
}