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

std::vector<torch::Tensor>
monarch_conv_bwd_cuda_r2r(
    torch::Tensor dout,
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

std::vector<torch::Tensor>
monarch_conv_bwd_cuda_r2r_bf16_all(
    torch::Tensor dout,
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

// std::pair<torch::Tensor, torch::Tensor>
// monarch_conv_bwd_cuda_bf16_all(
//     torch::Tensor dout,
//     torch::Tensor x,
//     torch::Tensor k_f,
//     torch::Tensor f_sqrt_N_fft,
//     torch::Tensor twiddle_factors_fft,
//     torch::Tensor f_sqrt_N_ifft,
//     torch::Tensor twiddle_factors_ifft,
//     uint fftsize,
//     uint N,
//     uint sqrt_N);

// std::pair<torch::Tensor, torch::Tensor>
// monarch_conv_bwd_cuda_16_16_16(
//     torch::Tensor dout,
//     torch::Tensor x,
//     torch::Tensor k_f,
//     torch::Tensor f_16_fft,
//     torch::Tensor twiddle_factors_256_fft,
//     torch::Tensor twiddle_factors_16_fft,
//     torch::Tensor f_16_ifft,
//     torch::Tensor twiddle_factors_256_ifft,
//     torch::Tensor twiddle_factors_16_ifft,
//     uint fftsize,
//     uint N,
//     uint sqrt_N);

// std::pair<torch::Tensor, torch::Tensor>
// monarch_conv_bwd_cuda_16_16_16_bf16(
//     torch::Tensor dout,
//     torch::Tensor x,
//     torch::Tensor k_f,
//     torch::Tensor f_16_fft,
//     torch::Tensor twiddle_factors_256_fft,
//     torch::Tensor twiddle_factors_16_fft,
//     torch::Tensor f_16_ifft,
//     torch::Tensor twiddle_factors_256_ifft,
//     torch::Tensor twiddle_factors_16_ifft,
//     uint fftsize,
//     uint N,
//     uint sqrt_N);

// std::pair<torch::Tensor, torch::Tensor>
// monarch_conv_bwd_cuda_16_16_16_bf16_all(
//     torch::Tensor dout,
//     torch::Tensor x,
//     torch::Tensor k_f,
//     torch::Tensor f_16_fft,
//     torch::Tensor twiddle_factors_256_fft,
//     torch::Tensor twiddle_factors_16_fft,
//     torch::Tensor f_16_ifft,
//     torch::Tensor twiddle_factors_256_ifft,
//     torch::Tensor twiddle_factors_16_ifft,
//     uint fftsize,
//     uint N,
//     uint sqrt_N);

// std::pair<torch::Tensor, torch::Tensor>
// monarch_conv_bwd_cuda_32_16_16(
//     torch::Tensor dout,
//     torch::Tensor x,
//     torch::Tensor k_f,
//     torch::Tensor f_32_fft,
//     torch::Tensor f_16_fft,
//     torch::Tensor twiddle_factors_N_fft,
//     torch::Tensor twiddle_factors_16_fft,
//     torch::Tensor f_32_ifft,
//     torch::Tensor f_16_ifft,
//     torch::Tensor twiddle_factors_N_ifft,
//     torch::Tensor twiddle_factors_16_ifft,
//     uint fftsize,
//     uint N);

// std::pair<torch::Tensor, torch::Tensor>
// monarch_conv_bwd_cuda_32_16_16_bf16_all(
//     torch::Tensor dout,
//     torch::Tensor x,
//     torch::Tensor k_f,
//     torch::Tensor f_32_fft,
//     torch::Tensor f_16_fft,
//     torch::Tensor twiddle_factors_N_fft,
//     torch::Tensor twiddle_factors_16_fft,
//     torch::Tensor f_32_ifft,
//     torch::Tensor f_16_ifft,
//     torch::Tensor twiddle_factors_N_ifft,
//     torch::Tensor twiddle_factors_16_ifft,
//     uint fftsize,
//     uint N);

// std::pair<torch::Tensor, torch::Tensor>
// monarch_conv_bwd_cuda_16_32_32(
//     torch::Tensor dout,
//     torch::Tensor x,
//     torch::Tensor k_f,
//     torch::Tensor f_16_fft,
//     torch::Tensor f_32_fft,
//     torch::Tensor twiddle_factors_N_fft,
//     torch::Tensor twiddle_factors_32_fft,
//     torch::Tensor f_16_ifft,
//     torch::Tensor f_32_ifft,
//     torch::Tensor twiddle_factors_N_ifft,
//     torch::Tensor twiddle_factors_32_ifft,
//     uint fftsize,
//     uint N);

// std::pair<torch::Tensor, torch::Tensor>
// monarch_conv_bwd_cuda_16_32_32_bf16_all(
//     torch::Tensor dout,
//     torch::Tensor x,
//     torch::Tensor k_f,
//     torch::Tensor f_16_fft,
//     torch::Tensor f_32_fft,
//     torch::Tensor twiddle_factors_N_fft,
//     torch::Tensor twiddle_factors_32_fft,
//     torch::Tensor f_16_ifft,
//     torch::Tensor f_32_ifft,
//     torch::Tensor twiddle_factors_N_ifft,
//     torch::Tensor twiddle_factors_32_ifft,
//     uint fftsize,
//     uint N);

// std::pair<torch::Tensor, torch::Tensor>
// monarch_conv_bwd_cuda_32_32_32(
//     torch::Tensor dout,
//     torch::Tensor x,
//     torch::Tensor k_f,
//     torch::Tensor f_32_fft,
//     torch::Tensor twiddle_factors_N_fft,
//     torch::Tensor twiddle_factors_32_fft,
//     torch::Tensor f_32_ifft,
//     torch::Tensor twiddle_factors_N_ifft,
//     torch::Tensor twiddle_factors_32_ifft,
//     uint fftsize,
//     uint N);

// std::pair<torch::Tensor, torch::Tensor>
// monarch_conv_bwd_cuda_32_32_32_bf16_all(
//     torch::Tensor dout,
//     torch::Tensor x,
//     torch::Tensor k_f,
//     torch::Tensor f_32_fft,
//     torch::Tensor twiddle_factors_N_fft,
//     torch::Tensor twiddle_factors_32_fft,
//     torch::Tensor f_32_ifft,
//     torch::Tensor twiddle_factors_N_ifft,
//     torch::Tensor twiddle_factors_32_ifft,
//     uint fftsize,
//     uint N);


std::vector<torch::Tensor>
monarch_conv_bwd_r2r(
    torch::Tensor dout,
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
    CHECK_INPUT(dout);
    CHECK_INPUT(x);
    CHECK_INPUT(k_f);
    CHECK_INPUT(f_sqrt_N_fft);
    CHECK_INPUT(twiddle_factors_fft);
    CHECK_INPUT(twid_r2r);
    CHECK_INPUT(f_sqrt_N_ifft);
    CHECK_INPUT(twiddle_factors_ifft);

    const int B = x.size(0);
    const int H = x.size(1);

    CHECK_SHAPE(dout, B, H, N);
    CHECK_SHAPE(x, B, H, N);
    CHECK_SHAPE(k_f, H, fftsize + 1, 2);
    CHECK_SHAPE(f_sqrt_N_fft, sqrt_N, sqrt_N, 2);
    CHECK_SHAPE(twiddle_factors_fft, sqrt_N, sqrt_N, 2);
    CHECK_SHAPE(twid_r2r, fftsize, 2);
    CHECK_SHAPE(f_sqrt_N_ifft, sqrt_N, sqrt_N, 2);
    CHECK_SHAPE(twiddle_factors_ifft, sqrt_N, sqrt_N, 2);

    if (x.dtype() == torch::kFloat16)
    {
        return monarch_conv_bwd_cuda_r2r(dout, x, k_f, f_sqrt_N_fft, twiddle_factors_fft, twid_r2r, f_sqrt_N_ifft, twiddle_factors_ifft, 
        in_gate, out_gate, fftsize, N, sqrt_N);
    }
    else if (x.dtype() == torch::kBFloat16)
    {
        return monarch_conv_bwd_cuda_r2r_bf16_all(dout, x, k_f, f_sqrt_N_fft, twiddle_factors_fft, twid_r2r, f_sqrt_N_ifft, twiddle_factors_ifft, in_gate, out_gate, fftsize, N, sqrt_N);
    }
    else
    {
        TORCH_CHECK(false, "Unsupported dtype");
    }
}

// std::pair<torch::Tensor, torch::Tensor>
// monarch_conv_bwd_16_16_16(
//     torch::Tensor dout,
//     torch::Tensor x,
//     torch::Tensor k_f,
//     torch::Tensor f_sqrt_N_fft,
//     torch::Tensor twiddle_factors_256_fft,
//     torch::Tensor twiddle_factors_16_fft,
//     torch::Tensor f_sqrt_N_ifft,
//     torch::Tensor twiddle_factors_256_ifft,
//     torch::Tensor twiddle_factors_16_ifft,
//     uint fftsize,
//     uint N,
//     uint sqrt_N_256,
//     uint sqrt_N_16)
// {
//     CHECK_INPUT(dout);
//     CHECK_INPUT(x);
//     CHECK_INPUT(k_f);
//     CHECK_INPUT(f_sqrt_N_fft);
//     CHECK_INPUT(twiddle_factors_256_fft);
//     CHECK_INPUT(twiddle_factors_16_fft);
//     CHECK_INPUT(f_sqrt_N_ifft);
//     CHECK_INPUT(twiddle_factors_256_fft);
//     CHECK_INPUT(twiddle_factors_16_fft);

//     const int B = x.size(0);
//     const int H = x.size(1);

//     CHECK_SHAPE(dout, B, H, N);
//     CHECK_SHAPE(x, B, H, N);
//     CHECK_SHAPE(k_f, H, fftsize, 2);
//     CHECK_SHAPE(f_sqrt_N_fft, sqrt_N_16, sqrt_N_16, 2);
//     CHECK_SHAPE(twiddle_factors_16_fft, sqrt_N_16, sqrt_N_16, 2);
//     CHECK_SHAPE(twiddle_factors_256_fft, sqrt_N_16, sqrt_N_256, 2);
//     CHECK_SHAPE(f_sqrt_N_ifft, sqrt_N_16, sqrt_N_16, 2);
//     CHECK_SHAPE(twiddle_factors_16_ifft, sqrt_N_16, sqrt_N_16, 2);
//     CHECK_SHAPE(twiddle_factors_256_ifft, sqrt_N_16, sqrt_N_256, 2);

//     if (x.dtype() == torch::kFloat16)
//     {
//         return monarch_conv_bwd_cuda_16_16_16(dout, x, k_f, f_sqrt_N_fft, twiddle_factors_256_fft, twiddle_factors_16_fft, f_sqrt_N_ifft, twiddle_factors_256_ifft, twiddle_factors_16_ifft, fftsize, N, sqrt_N_16);
//     }
//     else if (x.dtype() == torch::kBFloat16)
//     {
//         if (f_sqrt_N_fft.dtype() == torch::kBFloat16) {
//             return monarch_conv_bwd_cuda_16_16_16_bf16_all(dout, x, k_f, f_sqrt_N_fft, twiddle_factors_256_fft, twiddle_factors_16_fft, f_sqrt_N_ifft, twiddle_factors_256_ifft, twiddle_factors_16_ifft, fftsize, N, sqrt_N_16);
//         } else {
//             return monarch_conv_bwd_cuda_16_16_16_bf16(dout, x, k_f, f_sqrt_N_fft, twiddle_factors_256_fft, twiddle_factors_16_fft, f_sqrt_N_ifft, twiddle_factors_256_ifft, twiddle_factors_16_ifft, fftsize, N, sqrt_N_16);
//         }
//     }
//     else
//     {
//         TORCH_CHECK(false, "Unsupported dtype");
//     }
// }

// std::pair<torch::Tensor, torch::Tensor>
// monarch_conv_bwd_32_16_16(
//     torch::Tensor dout,
//     torch::Tensor x,
//     torch::Tensor k_f,
//     torch::Tensor f_32_fft,
//     torch::Tensor f_16_fft,
//     torch::Tensor twiddle_factors_N_fft,
//     torch::Tensor twiddle_factors_16_fft,
//     torch::Tensor f_32_ifft,
//     torch::Tensor f_16_ifft,
//     torch::Tensor twiddle_factors_N_ifft,
//     torch::Tensor twiddle_factors_16_ifft,
//     uint fftsize,
//     uint N)
// {
//     CHECK_INPUT(dout);
//     CHECK_INPUT(x);
//     CHECK_INPUT(k_f);
//     CHECK_INPUT(f_32_fft);
//     CHECK_INPUT(f_16_fft);
//     CHECK_INPUT(twiddle_factors_N_fft);
//     CHECK_INPUT(twiddle_factors_16_fft);
//     CHECK_INPUT(f_32_ifft);
//     CHECK_INPUT(f_16_ifft);
//     CHECK_INPUT(twiddle_factors_N_fft);
//     CHECK_INPUT(twiddle_factors_16_fft);

//     const int B = x.size(0);
//     const int H = x.size(1);

//     CHECK_SHAPE(dout, B, H, N);
//     CHECK_SHAPE(x, B, H, N);
//     CHECK_SHAPE(k_f, H, fftsize, 2);
//     CHECK_SHAPE(f_32_fft, 32, 32, 2);
//     CHECK_SHAPE(f_16_fft, 16, 16, 2);
//     CHECK_SHAPE(twiddle_factors_16_fft, 16, 16, 2);
//     CHECK_SHAPE(twiddle_factors_N_fft, 32, 256, 2);
//     CHECK_SHAPE(f_32_ifft, 32, 32, 2);
//     CHECK_SHAPE(f_16_ifft, 16, 16, 2);
//     CHECK_SHAPE(twiddle_factors_16_ifft, 16, 16, 2);
//     CHECK_SHAPE(twiddle_factors_N_ifft, 32, 256, 2);

//     if (x.dtype() == torch::kFloat16)
//     {
//         return monarch_conv_bwd_cuda_32_16_16(
//             dout, x, k_f,
//             f_32_fft, f_16_fft, twiddle_factors_N_fft, twiddle_factors_16_fft, f_32_ifft, f_16_ifft, twiddle_factors_N_ifft, twiddle_factors_16_ifft, fftsize, N);
//     }
//     else if (x.dtype() == torch::kBFloat16)
//     {
//         // if (true) {
//             return monarch_conv_bwd_cuda_32_16_16_bf16_all(
//                 dout, x, k_f,
//                 f_32_fft, f_16_fft, twiddle_factors_N_fft, twiddle_factors_16_fft, f_32_ifft, f_16_ifft, twiddle_factors_N_ifft, twiddle_factors_16_ifft, fftsize, N);
//         // } else {
//             // return monarch_conv_bwd_cuda_32_16_16_bf16(
//             //     dout, x, k_f,
//             //     f_32_fft, f_16_fft, twiddle_factors_N_fft, twiddle_factors_16_fft, f_32_ifft, f_16_ifft, twiddle_factors_N_ifft, twiddle_factors_16_ifft, fftsize, N);
//         // }
//     }
//     else
//     {
//         TORCH_CHECK(false, "Unsupported dtype");
//     }
// }

// std::pair<torch::Tensor, torch::Tensor>
// monarch_conv_bwd_16_32_32(
//     torch::Tensor dout,
//     torch::Tensor x,
//     torch::Tensor k_f,
//     torch::Tensor f_16_fft,
//     torch::Tensor f_32_fft,
//     torch::Tensor twiddle_factors_N_fft,
//     torch::Tensor twiddle_factors_32_fft,
//     torch::Tensor f_16_ifft,
//     torch::Tensor f_32_ifft,
//     torch::Tensor twiddle_factors_N_ifft,
//     torch::Tensor twiddle_factors_32_ifft,
//     uint fftsize,
//     uint N)
// {

//     CHECK_INPUT(dout);
//     CHECK_INPUT(x);
//     CHECK_INPUT(k_f);
//     CHECK_INPUT(f_32_fft);
//     CHECK_INPUT(f_16_fft);
//     CHECK_INPUT(twiddle_factors_N_fft);
//     CHECK_INPUT(twiddle_factors_32_fft);
//     CHECK_INPUT(f_32_ifft);
//     CHECK_INPUT(f_16_ifft);
//     CHECK_INPUT(twiddle_factors_N_fft);
//     CHECK_INPUT(twiddle_factors_32_fft);

//     TORCH_CHECK(x.is_contiguous());
//     TORCH_CHECK(k_f.is_contiguous());
//     TORCH_CHECK(f_32_fft.is_contiguous());
//     TORCH_CHECK(f_16_fft.is_contiguous());
//     TORCH_CHECK(twiddle_factors_N_fft.is_contiguous());
//     TORCH_CHECK(twiddle_factors_32_fft.is_contiguous());
//     TORCH_CHECK(f_32_ifft.is_contiguous());
//     TORCH_CHECK(f_16_ifft.is_contiguous());
//     TORCH_CHECK(twiddle_factors_N_ifft.is_contiguous());
//     TORCH_CHECK(twiddle_factors_32_ifft.is_contiguous());

//     const int B = x.size(0);
//     const int H = x.size(1);

//     CHECK_SHAPE(dout, B, H, N);
//     CHECK_SHAPE(x, B, H, N);
//     CHECK_SHAPE(k_f, H, fftsize, 2);
//     CHECK_SHAPE(f_32_fft, 32, 32, 2);
//     CHECK_SHAPE(f_16_fft, 16, 16, 2);
//     CHECK_SHAPE(twiddle_factors_32_fft, 32, 32, 2);
//     CHECK_SHAPE(twiddle_factors_N_fft, 16, 1024, 2);
//     CHECK_SHAPE(f_32_ifft, 32, 32, 2);
//     CHECK_SHAPE(f_16_ifft, 16, 16, 2);
//     CHECK_SHAPE(twiddle_factors_32_ifft, 32, 32, 2);
//     CHECK_SHAPE(twiddle_factors_N_ifft, 16, 1024, 2);

//     if (x.dtype() == torch::kFloat16)
//     {
//         return monarch_conv_bwd_cuda_16_32_32(
//             dout, x, k_f,
//             f_16_fft, f_32_fft,
//             twiddle_factors_N_fft, twiddle_factors_32_fft,
//             f_16_ifft, f_32_ifft,
//             twiddle_factors_N_ifft, twiddle_factors_32_ifft,
//             fftsize, N);
//     }
//     else if (x.dtype() == torch::kBFloat16)
//     {
//         return monarch_conv_bwd_cuda_16_32_32_bf16_all(
//             dout, x, k_f,
//             f_16_fft, f_32_fft,
//             twiddle_factors_N_fft, twiddle_factors_32_fft,
//             f_16_ifft, f_32_ifft,
//             twiddle_factors_N_ifft, twiddle_factors_32_ifft,
//             fftsize, N);
//     }
//     else
//     {
//         TORCH_CHECK(false, "Unsupported dtype");
//     }
// }

// std::pair<torch::Tensor, torch::Tensor>
// monarch_conv_bwd_32_32_32(
//     torch::Tensor dout,
//     torch::Tensor x,
//     torch::Tensor k_f,
//     torch::Tensor f_32_fft,
//     torch::Tensor twiddle_factors_N_fft,
//     torch::Tensor twiddle_factors_32_fft,
//     torch::Tensor f_32_ifft,
//     torch::Tensor twiddle_factors_N_ifft,
//     torch::Tensor twiddle_factors_32_ifft,
//     uint fftsize,
//     uint N)
// {
//     CHECK_INPUT(dout);
//     CHECK_INPUT(x);
//     CHECK_INPUT(k_f);
//     CHECK_INPUT(f_32_fft);
//     CHECK_INPUT(twiddle_factors_N_fft);
//     CHECK_INPUT(twiddle_factors_32_fft);
//     CHECK_INPUT(f_32_ifft);
//     CHECK_INPUT(twiddle_factors_N_fft);
//     CHECK_INPUT(twiddle_factors_32_fft);

//     TORCH_CHECK(x.is_contiguous());
//     TORCH_CHECK(k_f.is_contiguous());
//     TORCH_CHECK(f_32_fft.is_contiguous());
//     TORCH_CHECK(twiddle_factors_N_fft.is_contiguous());
//     TORCH_CHECK(twiddle_factors_32_fft.is_contiguous());
//     TORCH_CHECK(f_32_ifft.is_contiguous());
//     TORCH_CHECK(twiddle_factors_N_ifft.is_contiguous());
//     TORCH_CHECK(twiddle_factors_32_ifft.is_contiguous());

//     const int B = x.size(0);
//     const int H = x.size(1);

//     CHECK_SHAPE(dout, B, H, N);
//     CHECK_SHAPE(x, B, H, N);
//     CHECK_SHAPE(k_f, H, fftsize, 2);
//     CHECK_SHAPE(f_32_fft, 32, 32, 2);
//     CHECK_SHAPE(twiddle_factors_32_fft, 32, 32, 2);
//     CHECK_SHAPE(twiddle_factors_N_fft, 32, 1024, 2);
//     CHECK_SHAPE(f_32_ifft, 32, 32, 2);
//     CHECK_SHAPE(twiddle_factors_32_ifft, 32, 32, 2);
//     CHECK_SHAPE(twiddle_factors_N_ifft, 32, 1024, 2);

//     if (x.dtype() == torch::kFloat16)
//     {
//         return monarch_conv_bwd_cuda_32_32_32(
//             dout, x, k_f,
//             f_32_fft,
//             twiddle_factors_N_fft, twiddle_factors_32_fft,
//             f_32_ifft,
//             twiddle_factors_N_ifft, twiddle_factors_32_ifft,
//             fftsize, N);
//     }
//     else if (x.dtype() == torch::kBFloat16)
//     {
//         return monarch_conv_bwd_cuda_32_32_32_bf16_all(
//             dout, x, k_f,
//             f_32_fft,
//             twiddle_factors_N_fft, twiddle_factors_32_fft,
//             f_32_ifft,
//             twiddle_factors_N_ifft, twiddle_factors_32_ifft,
//             fftsize, N);
//     }
//     else
//     {
//         TORCH_CHECK(false, "Unsupported dtype");
//     }
// }