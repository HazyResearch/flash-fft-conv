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


torch::Tensor monarch_conv_cuda_r2r(
    torch::Tensor x,
    torch::Tensor k_f,
    torch::Tensor f_sqrt_N_fft,
    torch::Tensor twiddle_factors_fft,
    torch::Tensor twid_r2r,
    torch::Tensor f_sqrt_N_ifft,
    torch::Tensor twiddle_factors_ifft,
    c10::optional<torch::Tensor> in_gate,
    c10::optional<torch::Tensor> out_gate,
    uint fftsize,
    uint N,
    uint sqrt_N);

torch::Tensor monarch_conv_cuda_r2r_bf16_all(
    torch::Tensor x,
    torch::Tensor k_f,
    torch::Tensor f_sqrt_N_fft,
    torch::Tensor twiddle_factors_fft,
    torch::Tensor twid_r2r,
    torch::Tensor f_sqrt_N_ifft,
    torch::Tensor twiddle_factors_ifft,
    c10::optional<torch::Tensor> in_gate,
    c10::optional<torch::Tensor> out_gate,
    uint fftsize,
    uint N,
    uint sqrt_N);

torch::Tensor monarch_conv_r2r(
    torch::Tensor x,
    torch::Tensor k_f,
    torch::Tensor f_sqrt_N_fft,
    torch::Tensor twiddle_factors_fft,
    torch::Tensor twid_r2r,
    torch::Tensor f_sqrt_N_ifft,
    torch::Tensor twiddle_factors_ifft,
    c10::optional<torch::Tensor> in_gate,
    c10::optional<torch::Tensor> out_gate,
    uint fftsize,
    uint N,
    uint sqrt_N)
{
    CHECK_INPUT(x);
    CHECK_INPUT(k_f);
    CHECK_INPUT(f_sqrt_N_fft);
    CHECK_INPUT(twiddle_factors_fft);
    CHECK_INPUT(twid_r2r);
    CHECK_INPUT(f_sqrt_N_ifft);
    CHECK_INPUT(twiddle_factors_ifft);

    const int B = x.size(0);
    const int H = x.size(1);

    CHECK_SHAPE(x, B, H, N);
    CHECK_SHAPE(k_f, H, fftsize + 1, 2);
    CHECK_SHAPE(f_sqrt_N_fft, sqrt_N, sqrt_N, 2);
    CHECK_SHAPE(twiddle_factors_fft, sqrt_N, sqrt_N, 2);
    CHECK_SHAPE(twid_r2r, fftsize, 2);
    CHECK_SHAPE(f_sqrt_N_ifft, sqrt_N, sqrt_N, 2);
    CHECK_SHAPE(twiddle_factors_ifft, sqrt_N, sqrt_N, 2);

    if (x.dtype() == torch::kFloat16)
    {
        return monarch_conv_cuda_r2r(x, k_f, f_sqrt_N_fft, twiddle_factors_fft, twid_r2r, f_sqrt_N_ifft, twiddle_factors_ifft, in_gate, out_gate, fftsize, N, sqrt_N);
    }
    else if (x.dtype() == torch::kBFloat16)
    {   
        return monarch_conv_cuda_r2r_bf16_all(x, k_f, f_sqrt_N_fft, twiddle_factors_fft, twid_r2r, f_sqrt_N_ifft, twiddle_factors_ifft, in_gate, out_gate, fftsize, N, sqrt_N);
    }
    else
    {
        TORCH_CHECK(false, "Unsupported dtype");
    }
}
