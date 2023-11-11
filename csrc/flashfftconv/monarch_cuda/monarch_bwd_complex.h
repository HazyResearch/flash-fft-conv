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

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
monarch_conv_bwd_cuda_16_16_16_complex(
    torch::Tensor dout_real,
    torch::Tensor dout_imag,
    torch::Tensor x_real,
    torch::Tensor x_imag,
    torch::Tensor k_f,
    torch::Tensor f_16_fft,
    torch::Tensor twiddle_factors_256_fft,
    torch::Tensor twiddle_factors_16_fft,
    torch::Tensor f_16_ifft,
    torch::Tensor twiddle_factors_256_ifft,
    torch::Tensor twiddle_factors_16_ifft,
    uint fftsize,
    uint N);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
monarch_conv_bwd_cuda_16_16_16_complex_bf16_all(
    torch::Tensor dout_real,
    torch::Tensor dout_imag,
    torch::Tensor x_real,
    torch::Tensor x_imag,
    torch::Tensor k_f,
    torch::Tensor f_16_fft,
    torch::Tensor twiddle_factors_256_fft,
    torch::Tensor twiddle_factors_16_fft,
    torch::Tensor f_16_ifft,
    torch::Tensor twiddle_factors_256_ifft,
    torch::Tensor twiddle_factors_16_ifft,
    uint fftsize,
    uint N);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
monarch_conv_bwd_cuda_32_16_16_complex(
    torch::Tensor dout_real,
    torch::Tensor dout_imag,
    torch::Tensor x_real,
    torch::Tensor x_imag,
    torch::Tensor k_f,
    torch::Tensor f_32_fft,
    torch::Tensor f_16_fft,
    torch::Tensor twiddle_factors_N_fft,
    torch::Tensor twiddle_factors_16_fft,
    torch::Tensor f_32_ifft,
    torch::Tensor f_16_ifft,
    torch::Tensor twiddle_factors_N_ifft,
    torch::Tensor twiddle_factors_16_ifft,
    uint fftsize,
    uint N);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
monarch_conv_bwd_cuda_32_16_16_complex_bf16_all(
    torch::Tensor dout_real,
    torch::Tensor dout_imag,
    torch::Tensor x_real,
    torch::Tensor x_imag,
    torch::Tensor k_f,
    torch::Tensor f_32_fft,
    torch::Tensor f_16_fft,
    torch::Tensor twiddle_factors_N_fft,
    torch::Tensor twiddle_factors_16_fft,
    torch::Tensor f_32_ifft,
    torch::Tensor f_16_ifft,
    torch::Tensor twiddle_factors_N_ifft,
    torch::Tensor twiddle_factors_16_ifft,
    uint fftsize,
    uint N);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
monarch_conv_bwd_cuda_16_32_32_complex(
    torch::Tensor dout_real,
    torch::Tensor dout_imag,
    torch::Tensor x_real,
    torch::Tensor x_imag,
    torch::Tensor k_f,
    torch::Tensor f_16_fft,
    torch::Tensor f_32_fft,
    torch::Tensor twiddle_factors_N_fft,
    torch::Tensor twiddle_factors_32_fft,
    torch::Tensor f_16_ifft,
    torch::Tensor f_32_ifft,
    torch::Tensor twiddle_factors_N_ifft,
    torch::Tensor twiddle_factors_32_ifft,
    uint fftsize,
    uint N);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
monarch_conv_bwd_cuda_16_32_32_complex_bf16_all(
    torch::Tensor dout_real,
    torch::Tensor dout_imag,
    torch::Tensor x_real,
    torch::Tensor x_imag,
    torch::Tensor k_f,
    torch::Tensor f_16_fft,
    torch::Tensor f_32_fft,
    torch::Tensor twiddle_factors_N_fft,
    torch::Tensor twiddle_factors_32_fft,
    torch::Tensor f_16_ifft,
    torch::Tensor f_32_ifft,
    torch::Tensor twiddle_factors_N_ifft,
    torch::Tensor twiddle_factors_32_ifft,
    uint fftsize,
    uint N);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
monarch_conv_bwd_cuda_32_32_32_complex(
    torch::Tensor dout_real,
    torch::Tensor dout_imag,
    torch::Tensor x_real,
    torch::Tensor x_imag,
    torch::Tensor k_f,
    torch::Tensor f_32_fft,
    torch::Tensor twiddle_factors_N_fft,
    torch::Tensor twiddle_factors_32_fft,
    torch::Tensor f_32_ifft,
    torch::Tensor twiddle_factors_N_ifft,
    torch::Tensor twiddle_factors_32_ifft,
    uint fftsize,
    uint N);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
monarch_conv_bwd_cuda_32_32_32_complex_bf16_all(
    torch::Tensor dout_real,
    torch::Tensor dout_imag,
    torch::Tensor x_real,
    torch::Tensor x_imag,
    torch::Tensor k_f,
    torch::Tensor f_32_fft,
    torch::Tensor twiddle_factors_N_fft,
    torch::Tensor twiddle_factors_32_fft,
    torch::Tensor f_32_ifft,
    torch::Tensor twiddle_factors_N_ifft,
    torch::Tensor twiddle_factors_32_ifft,
    uint fftsize,
    uint N);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
monarch_conv_bwd_16_16_16_complex(
    torch::Tensor dout_real,
    torch::Tensor dout_imag,
    torch::Tensor x_real,
    torch::Tensor x_imag,
    torch::Tensor k_f,
    torch::Tensor f_16_fft,
    torch::Tensor twiddle_factors_256_fft,
    torch::Tensor twiddle_factors_16_fft,
    torch::Tensor f_16_ifft,
    torch::Tensor twiddle_factors_256_ifft,
    torch::Tensor twiddle_factors_16_ifft,
    uint fftsize,
    uint N)
{
    CHECK_INPUT(dout_real);
    CHECK_INPUT(dout_imag);
    CHECK_INPUT(x_real);
    CHECK_INPUT(x_imag);
    CHECK_INPUT(k_f);
    CHECK_INPUT(f_16_fft);
    CHECK_INPUT(twiddle_factors_256_fft);
    CHECK_INPUT(twiddle_factors_16_fft);
    CHECK_INPUT(f_16_ifft);
    CHECK_INPUT(twiddle_factors_256_fft);
    CHECK_INPUT(twiddle_factors_16_fft);

    const int B = x_real.size(0);
    const int H = x_real.size(1);

    CHECK_SHAPE(dout_real, B, H, N);
    CHECK_SHAPE(dout_imag, B, H, N);
    CHECK_SHAPE(x_real, B, H, N);
    CHECK_SHAPE(x_imag, B, H, N);
    CHECK_SHAPE(k_f, H, fftsize, 2);
    CHECK_SHAPE(f_16_fft, 16, 16, 2);
    CHECK_SHAPE(twiddle_factors_16_fft, 16, 16, 2);
    CHECK_SHAPE(twiddle_factors_256_fft, 16, 256, 2);
    CHECK_SHAPE(f_16_ifft, 16, 16, 2);
    CHECK_SHAPE(twiddle_factors_16_ifft, 16, 16, 2);
    CHECK_SHAPE(twiddle_factors_256_ifft, 16, 256, 2);

    if (x_real.dtype() == torch::kFloat16)
    {
        return monarch_conv_bwd_cuda_16_16_16_complex(
            dout_real, dout_imag, x_real, x_imag, k_f,
            f_16_fft, twiddle_factors_256_fft, twiddle_factors_16_fft, f_16_ifft, twiddle_factors_256_ifft, twiddle_factors_16_ifft, fftsize, N);
    }
    else if (x_real.dtype() == torch::kBFloat16)
    {
        return monarch_conv_bwd_cuda_16_16_16_complex_bf16_all(
            dout_real, dout_imag, x_real, x_imag, k_f,
            f_16_fft, twiddle_factors_256_fft, twiddle_factors_16_fft, f_16_ifft, twiddle_factors_256_ifft, twiddle_factors_16_ifft, fftsize, N);
    }
    else
    {
        TORCH_CHECK(false, "Unsupported dtype");
    }
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
monarch_conv_bwd_32_16_16_complex(
    torch::Tensor dout_real,
    torch::Tensor dout_imag,
    torch::Tensor x_real,
    torch::Tensor x_imag,
    torch::Tensor k_f,
    torch::Tensor f_32_fft,
    torch::Tensor f_16_fft,
    torch::Tensor twiddle_factors_N_fft,
    torch::Tensor twiddle_factors_16_fft,
    torch::Tensor f_32_ifft,
    torch::Tensor f_16_ifft,
    torch::Tensor twiddle_factors_N_ifft,
    torch::Tensor twiddle_factors_16_ifft,
    uint fftsize,
    uint N)
{
    CHECK_INPUT(dout_real);
    CHECK_INPUT(dout_imag);
    CHECK_INPUT(x_real);
    CHECK_INPUT(x_imag);
    CHECK_INPUT(k_f);
    CHECK_INPUT(f_32_fft);
    CHECK_INPUT(f_16_fft);
    CHECK_INPUT(twiddle_factors_N_fft);
    CHECK_INPUT(twiddle_factors_16_fft);
    CHECK_INPUT(f_32_ifft);
    CHECK_INPUT(f_16_ifft);
    CHECK_INPUT(twiddle_factors_N_fft);
    CHECK_INPUT(twiddle_factors_16_fft);

    const int B = x_real.size(0);
    const int H = x_real.size(1);

    CHECK_SHAPE(dout_real, B, H, N);
    CHECK_SHAPE(dout_imag, B, H, N);
    CHECK_SHAPE(x_real, B, H, N);
    CHECK_SHAPE(x_imag, B, H, N);
    CHECK_SHAPE(k_f, H, fftsize, 2);
    CHECK_SHAPE(f_32_fft, 32, 32, 2);
    CHECK_SHAPE(f_16_fft, 16, 16, 2);
    CHECK_SHAPE(twiddle_factors_16_fft, 16, 16, 2);
    CHECK_SHAPE(twiddle_factors_N_fft, 32, 256, 2);
    CHECK_SHAPE(f_32_ifft, 32, 32, 2);
    CHECK_SHAPE(f_16_ifft, 16, 16, 2);
    CHECK_SHAPE(twiddle_factors_16_ifft, 16, 16, 2);
    CHECK_SHAPE(twiddle_factors_N_ifft, 32, 256, 2);

    if (x_real.dtype() == torch::kFloat16)
    {
        return monarch_conv_bwd_cuda_32_16_16_complex(
            dout_real, dout_imag, x_real, x_imag, k_f,
            f_32_fft, f_16_fft, twiddle_factors_N_fft, twiddle_factors_16_fft, f_32_ifft, f_16_ifft, twiddle_factors_N_ifft, twiddle_factors_16_ifft, fftsize, N);
    }
    else if (x_real.dtype() == torch::kBFloat16)
    {
        return monarch_conv_bwd_cuda_32_16_16_complex_bf16_all(
            dout_real, dout_imag, x_real, x_imag, k_f,
            f_32_fft, f_16_fft, twiddle_factors_N_fft, twiddle_factors_16_fft, f_32_ifft, f_16_ifft, twiddle_factors_N_ifft, twiddle_factors_16_ifft, fftsize, N);
    }
    else
    {
        TORCH_CHECK(false, "Unsupported dtype");
    }
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
monarch_conv_bwd_16_32_32_complex(
    torch::Tensor dout_real,
    torch::Tensor dout_imag,
    torch::Tensor x_real,
    torch::Tensor x_imag,
    torch::Tensor k_f,
    torch::Tensor f_16_fft,
    torch::Tensor f_32_fft,
    torch::Tensor twiddle_factors_N_fft,
    torch::Tensor twiddle_factors_32_fft,
    torch::Tensor f_16_ifft,
    torch::Tensor f_32_ifft,
    torch::Tensor twiddle_factors_N_ifft,
    torch::Tensor twiddle_factors_32_ifft,
    uint fftsize,
    uint N)
{

    CHECK_INPUT(dout_real);
    CHECK_INPUT(dout_imag);
    CHECK_INPUT(x_real);
    CHECK_INPUT(x_imag);
    CHECK_INPUT(k_f);
    CHECK_INPUT(f_32_fft);
    CHECK_INPUT(f_16_fft);
    CHECK_INPUT(twiddle_factors_N_fft);
    CHECK_INPUT(twiddle_factors_32_fft);
    CHECK_INPUT(f_32_ifft);
    CHECK_INPUT(f_16_ifft);
    CHECK_INPUT(twiddle_factors_N_fft);
    CHECK_INPUT(twiddle_factors_32_fft);

    TORCH_CHECK(dout_real.is_contiguous());
    TORCH_CHECK(dout_imag.is_contiguous());
    TORCH_CHECK(x_real.is_contiguous());
    TORCH_CHECK(x_imag.is_contiguous());
    TORCH_CHECK(k_f.is_contiguous());
    TORCH_CHECK(f_32_fft.is_contiguous());
    TORCH_CHECK(f_16_fft.is_contiguous());
    TORCH_CHECK(twiddle_factors_N_fft.is_contiguous());
    TORCH_CHECK(twiddle_factors_32_fft.is_contiguous());
    TORCH_CHECK(f_32_ifft.is_contiguous());
    TORCH_CHECK(f_16_ifft.is_contiguous());
    TORCH_CHECK(twiddle_factors_N_ifft.is_contiguous());
    TORCH_CHECK(twiddle_factors_32_ifft.is_contiguous());

    const int B = x_real.size(0);
    const int H = x_real.size(1);

    CHECK_SHAPE(dout_real, B, H, N);
    CHECK_SHAPE(dout_imag, B, H, N);
    CHECK_SHAPE(x_real, B, H, N);
    CHECK_SHAPE(x_imag, B, H, N);
    CHECK_SHAPE(k_f, H, fftsize, 2);
    CHECK_SHAPE(f_32_fft, 32, 32, 2);
    CHECK_SHAPE(f_16_fft, 16, 16, 2);
    CHECK_SHAPE(twiddle_factors_32_fft, 32, 32, 2);
    CHECK_SHAPE(twiddle_factors_N_fft, 16, 1024, 2);
    CHECK_SHAPE(f_32_ifft, 32, 32, 2);
    CHECK_SHAPE(f_16_ifft, 16, 16, 2);
    CHECK_SHAPE(twiddle_factors_32_ifft, 32, 32, 2);
    CHECK_SHAPE(twiddle_factors_N_ifft, 16, 1024, 2);

    if (x_real.dtype() == torch::kFloat16)
    {
        return monarch_conv_bwd_cuda_16_32_32_complex(
            dout_real, dout_imag, x_real, x_imag, k_f,
            f_16_fft, f_32_fft,
            twiddle_factors_N_fft, twiddle_factors_32_fft,
            f_16_ifft, f_32_ifft,
            twiddle_factors_N_ifft, twiddle_factors_32_ifft,
            fftsize, N);
    }
    else if (x_real.dtype() == torch::kBFloat16)
    {
        return monarch_conv_bwd_cuda_16_32_32_complex_bf16_all(
            dout_real, dout_imag, x_real, x_imag, k_f,
            f_16_fft, f_32_fft,
            twiddle_factors_N_fft, twiddle_factors_32_fft,
            f_16_ifft, f_32_ifft,
            twiddle_factors_N_ifft, twiddle_factors_32_ifft,
            fftsize, N);
    }
    else
    {
        TORCH_CHECK(false, "Unsupported dtype");
    }
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
monarch_conv_bwd_32_32_32_complex(
    torch::Tensor dout_real,
    torch::Tensor dout_imag,
    torch::Tensor x_real,
    torch::Tensor x_imag,
    torch::Tensor k_f,
    torch::Tensor f_32_fft,
    torch::Tensor twiddle_factors_N_fft,
    torch::Tensor twiddle_factors_32_fft,
    torch::Tensor f_32_ifft,
    torch::Tensor twiddle_factors_N_ifft,
    torch::Tensor twiddle_factors_32_ifft,
    uint fftsize,
    uint N)
{
    CHECK_INPUT(dout_real);
    CHECK_INPUT(dout_imag);
    CHECK_INPUT(x_real);
    CHECK_INPUT(x_imag);
    CHECK_INPUT(k_f);
    CHECK_INPUT(f_32_fft);
    CHECK_INPUT(twiddle_factors_N_fft);
    CHECK_INPUT(twiddle_factors_32_fft);
    CHECK_INPUT(f_32_ifft);
    CHECK_INPUT(twiddle_factors_N_fft);
    CHECK_INPUT(twiddle_factors_32_fft);

    TORCH_CHECK(dout_real.is_contiguous());
    TORCH_CHECK(dout_imag.is_contiguous());
    TORCH_CHECK(x_real.is_contiguous());
    TORCH_CHECK(x_imag.is_contiguous());
    TORCH_CHECK(k_f.is_contiguous());
    TORCH_CHECK(f_32_fft.is_contiguous());
    TORCH_CHECK(twiddle_factors_N_fft.is_contiguous());
    TORCH_CHECK(twiddle_factors_32_fft.is_contiguous());
    TORCH_CHECK(f_32_ifft.is_contiguous());
    TORCH_CHECK(twiddle_factors_N_ifft.is_contiguous());
    TORCH_CHECK(twiddle_factors_32_ifft.is_contiguous());

    const int B = x_real.size(0);
    const int H = x_real.size(1);

    CHECK_SHAPE(dout_real, B, H, N);
    CHECK_SHAPE(dout_imag, B, H, N);
    CHECK_SHAPE(x_real, B, H, N);
    CHECK_SHAPE(x_imag, B, H, N);
    CHECK_SHAPE(k_f, H, fftsize, 2);
    CHECK_SHAPE(f_32_fft, 32, 32, 2);
    CHECK_SHAPE(twiddle_factors_32_fft, 32, 32, 2);
    CHECK_SHAPE(twiddle_factors_N_fft, 32, 1024, 2);
    CHECK_SHAPE(f_32_ifft, 32, 32, 2);
    CHECK_SHAPE(twiddle_factors_32_ifft, 32, 32, 2);
    CHECK_SHAPE(twiddle_factors_N_ifft, 32, 1024, 2);

    if (x_real.dtype() == torch::kFloat16)
    {
        return monarch_conv_bwd_cuda_32_32_32_complex(
            dout_real, dout_imag, x_real, x_imag, k_f, 
            f_32_fft,
            twiddle_factors_N_fft, twiddle_factors_32_fft,
            f_32_ifft,
            twiddle_factors_N_ifft, twiddle_factors_32_ifft,
            fftsize, N);
    }
    else if (x_real.dtype() == torch::kBFloat16)
    {
        return monarch_conv_bwd_cuda_32_32_32_complex_bf16_all(
            dout_real, dout_imag, x_real, x_imag, k_f, 
            f_32_fft,
            twiddle_factors_N_fft, twiddle_factors_32_fft,
            f_32_ifft,
            twiddle_factors_N_ifft, twiddle_factors_32_ifft,
            fftsize, N);
    }
    else
    {
        TORCH_CHECK(false, "Unsupported dtype");
    }
}