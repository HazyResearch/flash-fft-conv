// Copyright (c) 2023 Dan Fu, Hermann Kumbong

#include <torch/extension.h>

#include <vector>
#include <stdio.h>
#include <mma.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cub/block/block_load.cuh>
#include <cub/block/block_store.cuh>
#include "kernels_fp16/monarch_cuda_shared.h"
#include "kernels_bf16/monarch_cuda_shared_bf16_no_float_shm.h"
#include "kernels_bf16/monarch_cuda_kernel_bf16.h"
#include "kernels_bf16/monarch_cuda_16_16_16_complex_kernel_bf16.h"
#include "kernels_bf16/monarch_cuda_32_16_16_complex_kernel_bf16.h"
#include "kernels_bf16/monarch_cuda_16_32_32_complex_kernel_bf16.h"
#include "kernels_bf16/monarch_cuda_32_32_32_complex_kernel_bf16.h"
using namespace nvcuda;

// *************** FOR ERROR CHECKING *******************
#ifndef CUDA_RT_CALL
#define CUDA_RT_CALL( call )                                                                                           \
    {                                                                                                                  \
        auto status = static_cast<cudaError_t>( call );                                                                \
        if ( status != cudaSuccess )                                                                                   \
            fprintf( stderr,                                                                                           \
                     "ERROR: CUDA RT call \"%s\" in line %d of file %s failed "                                        \
                     "with "                                                                                           \
                     "%s (%d).\n",                                                                                     \
                     #call,                                                                                            \
                     __LINE__,                                                                                         \
                     __FILE__,                                                                                         \
                     cudaGetErrorString( status ),                                                                     \
                     status );                                                                                         \
    }
#endif  // CUDA_RT_CALL
// *************** FOR ERROR CHECKING *******************

#ifndef CUDA_CHECK_ERROR
// Define some error checking macros.
#define CHECK_CUDA_ERROR(val) check((val), #val, __FILE__, __LINE__)
template <typename T>
void check(T err, const char* const func, const char* const file,
           const int line)
{
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA Runtime Error at: " << file << ":" << line
                  << std::endl;
        std::cerr << cudaGetErrorString(err) << " " << func << std::endl;
        // We don't exit when we encounter CUDA errors in this example.
        // std::exit(EXIT_FAILURE);
    }
}
#endif  // CUDA_CHECK_ERROR

#ifndef CHECK_LAST_CUDA_ERROR
#define CHECK_LAST_CUDA_ERROR() checkLastBF16ComplexFwd(__FILE__, __LINE__)
void checkLastBF16ComplexFwd(const char* const file, const int line)
{
    cudaError_t err{cudaGetLastError()};
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA Runtime Error at: " << file << ":" << line
                  << std::endl;
        std::cerr << cudaGetErrorString(err) << std::endl;
        // We don't exit when we encounter CUDA errors in this example.
        // std::exit(EXIT_FAILURE);
    }
}
#endif  // CHECK_LAST_CUDA_ERROR

std::pair<torch::Tensor, torch::Tensor>
monarch_conv_cuda_16_16_16_complex_bf16_all(
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
  uint N
){

  uint B = x_real.size(0);
  uint H = x_real.size(1);
  // First: using WMMA
  dim3 gridDim;
  dim3 blockDim;

  // printf("gridDim.x = %d, gridDim.y = %d\n", gridDim.x, gridDim.y);
  torch::Tensor out_real = torch::empty({B, H, N}, x_real.options());
  torch::Tensor out_imag = torch::empty({B, H, N}, x_real.options());

  switch (fftsize) {
    case 4096:
      // if (true) {
      if (B >= 4 && (B % 4) == 0 && (H % 8) == 0) {
        gridDim.x = B / 4;
        gridDim.y = H / 8;

        blockDim.x = 32;
        blockDim.y = 4;

        monarch_conv_cuda_complex_kernel<32, 4, 4096, 1, 16, false, 4, 8, 4><<<gridDim, blockDim, (2 * fftsize + 4 * 256) * sizeof(half)>>>(
            static_cast<at::BFloat16 *>(x_real.data_ptr()),
            static_cast<at::BFloat16 *>(x_imag.data_ptr()),
            static_cast<complex_bfloat16_t *>(k_f.data_ptr()),
            static_cast<complex_bfloat16_t *>(f_16_fft.data_ptr()),
            static_cast<complex_bfloat16_t *>(twiddle_factors_256_fft.data_ptr()),
            static_cast<complex_bfloat16_t *>(twiddle_factors_16_fft.data_ptr()),
            static_cast<complex_bfloat16_t *>(f_16_ifft.data_ptr()),
            static_cast<complex_bfloat16_t *>(twiddle_factors_256_ifft.data_ptr()),
            static_cast<complex_bfloat16_t *>(twiddle_factors_16_ifft.data_ptr()),
            static_cast<at::BFloat16 *>(out_real.data_ptr()),
            static_cast<at::BFloat16 *>(out_imag.data_ptr()),
            B,
            H,
            N,
            16);
      }
      else if (B == 2 && (B % 2) == 0 && (H % 8) == 0) {
        gridDim.x = B / 2;
        gridDim.y = H / 8;

        blockDim.x = 32;
        blockDim.y = 4;

        monarch_conv_cuda_complex_kernel<32, 4, 4096, 1, 16, false, 2, 8, 4><<<gridDim, blockDim, (2 * fftsize + 4 * 256) * sizeof(half)>>>(
          static_cast<at::BFloat16 *>(x_real.data_ptr()),
          static_cast<at::BFloat16 *>(x_imag.data_ptr()),
          static_cast<complex_bfloat16_t *>(k_f.data_ptr()),
          static_cast<complex_bfloat16_t *>(f_16_fft.data_ptr()),
          static_cast<complex_bfloat16_t *>(twiddle_factors_256_fft.data_ptr()),
          static_cast<complex_bfloat16_t *>(twiddle_factors_16_fft.data_ptr()),
          static_cast<complex_bfloat16_t *>(f_16_ifft.data_ptr()),
          static_cast<complex_bfloat16_t *>(twiddle_factors_256_ifft.data_ptr()),
          static_cast<complex_bfloat16_t *>(twiddle_factors_16_ifft.data_ptr()),
          static_cast<at::BFloat16 *>(out_real.data_ptr()),
          static_cast<at::BFloat16 *>(out_imag.data_ptr()),
          B,
          H,
          N,
          16);
      } else if ((H % 8) == 0) {
        gridDim.x = B;
        gridDim.y = H / 8;

        blockDim.x = 32;
        blockDim.y = 4;

        monarch_conv_cuda_complex_kernel<32, 4, 4096, 1, 16, false, 1, 8, 4><<<gridDim, blockDim, (2 * fftsize + 4 * 256) * sizeof(half)>>>(
          static_cast<at::BFloat16 *>(x_real.data_ptr()),
          static_cast<at::BFloat16 *>(x_imag.data_ptr()),
          static_cast<complex_bfloat16_t *>(k_f.data_ptr()),
          static_cast<complex_bfloat16_t *>(f_16_fft.data_ptr()),
          static_cast<complex_bfloat16_t *>(twiddle_factors_256_fft.data_ptr()),
          static_cast<complex_bfloat16_t *>(twiddle_factors_16_fft.data_ptr()),
          static_cast<complex_bfloat16_t *>(f_16_ifft.data_ptr()),
          static_cast<complex_bfloat16_t *>(twiddle_factors_256_ifft.data_ptr()),
          static_cast<complex_bfloat16_t *>(twiddle_factors_16_ifft.data_ptr()),
          static_cast<at::BFloat16 *>(out_real.data_ptr()),
          static_cast<at::BFloat16 *>(out_imag.data_ptr()),
          B,
          H,
          N,
          16);
      } else {
        gridDim.x = B;
        gridDim.y = H;

        blockDim.x = 32;
        blockDim.y = 4;

        monarch_conv_cuda_complex_kernel<32, 4, 4096, 1, 16, false, 1, 1, 4><<<gridDim, blockDim, (2 * fftsize + 4 * 256) * sizeof(half)>>>(
          static_cast<at::BFloat16 *>(x_real.data_ptr()),
          static_cast<at::BFloat16 *>(x_imag.data_ptr()),
          static_cast<complex_bfloat16_t *>(k_f.data_ptr()),
          static_cast<complex_bfloat16_t *>(f_16_fft.data_ptr()),
          static_cast<complex_bfloat16_t *>(twiddle_factors_256_fft.data_ptr()),
          static_cast<complex_bfloat16_t *>(twiddle_factors_16_fft.data_ptr()),
          static_cast<complex_bfloat16_t *>(f_16_ifft.data_ptr()),
          static_cast<complex_bfloat16_t *>(twiddle_factors_256_ifft.data_ptr()),
          static_cast<complex_bfloat16_t *>(twiddle_factors_16_ifft.data_ptr()),
          static_cast<at::BFloat16 *>(out_real.data_ptr()),
          static_cast<at::BFloat16 *>(out_imag.data_ptr()),
          B,
          H,
          N,
          16);
      }
      break;
    default:
      AT_ERROR("Monarch forward not implemented for this sequence length");
  }
  
  CHECK_LAST_CUDA_ERROR();
  return std::make_pair(out_real, out_imag);
}

std::pair<torch::Tensor, torch::Tensor>
monarch_conv_cuda_32_16_16_complex_bf16_all(
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
  uint N
){

  uint B = x_real.size(0);
  uint H = x_real.size(1);
  // First: using WMMA
  dim3 gridDim;
  dim3 blockDim;

  // printf("gridDim.x = %d, gridDim.y = %d\n", gridDim.x, gridDim.y);
  torch::Tensor out_real = torch::empty({B, H, N}, x_real.options());
  torch::Tensor out_imag = torch::empty({B, H, N}, x_real.options());

  switch (fftsize) {
    case 8192:
      if (B >= 2 && (B % 2) == 0 && (H % 8) == 0) {
      // if (true) {
        gridDim.x = B / 2;
        gridDim.y = H / 8;

        blockDim.x = 32;
        blockDim.y = 8;

        monarch_conv_cuda_complex_kernel<32, 8, 8192, 2, 1, 16, false, 2, 8, 8><<<gridDim, blockDim, (2 * fftsize + 4 * 1024) * sizeof(half)>>>(
            static_cast<at::BFloat16 *>(x_real.data_ptr()),
            static_cast<at::BFloat16 *>(x_imag.data_ptr()),
            static_cast<complex_bfloat16_t *>(k_f.data_ptr()),
            static_cast<complex_bfloat16_t *>(f_32_fft.data_ptr()),
            static_cast<complex_bfloat16_t *>(f_16_fft.data_ptr()),
            static_cast<complex_bfloat16_t *>(twiddle_factors_N_fft.data_ptr()),
            static_cast<complex_bfloat16_t *>(twiddle_factors_16_fft.data_ptr()),
            static_cast<complex_bfloat16_t *>(f_32_ifft.data_ptr()),
            static_cast<complex_bfloat16_t *>(f_16_ifft.data_ptr()),
            static_cast<complex_bfloat16_t *>(twiddle_factors_N_ifft.data_ptr()),
            static_cast<complex_bfloat16_t *>(twiddle_factors_16_ifft.data_ptr()),
            static_cast<at::BFloat16 *>(out_real.data_ptr()),
            static_cast<at::BFloat16 *>(out_imag.data_ptr()),
            B,
            H,
            N);
      }
      else if ((H % 8) == 0) {
        gridDim.x = B;
        gridDim.y = H / 8;

        blockDim.x = 32;
        blockDim.y = 8;

        monarch_conv_cuda_complex_kernel<32, 8, 8192, 2, 1, 16, false, 1, 8, 8><<<gridDim, blockDim, (2 * fftsize + 4 * 1024) * sizeof(half)>>>(
          static_cast<at::BFloat16 *>(x_real.data_ptr()),
          static_cast<at::BFloat16 *>(x_imag.data_ptr()),
          static_cast<complex_bfloat16_t *>(k_f.data_ptr()),
          static_cast<complex_bfloat16_t *>(f_32_fft.data_ptr()),
          static_cast<complex_bfloat16_t *>(f_16_fft.data_ptr()),
          static_cast<complex_bfloat16_t *>(twiddle_factors_N_fft.data_ptr()),
          static_cast<complex_bfloat16_t *>(twiddle_factors_16_fft.data_ptr()),
          static_cast<complex_bfloat16_t *>(f_32_ifft.data_ptr()),
          static_cast<complex_bfloat16_t *>(f_16_ifft.data_ptr()),
          static_cast<complex_bfloat16_t *>(twiddle_factors_N_ifft.data_ptr()),
          static_cast<complex_bfloat16_t *>(twiddle_factors_16_ifft.data_ptr()),
          static_cast<at::BFloat16 *>(out_real.data_ptr()),
          static_cast<at::BFloat16 *>(out_imag.data_ptr()),
          B,
          H,
          N);
      }
      else {
        gridDim.x = B;
        gridDim.y = H;

        blockDim.x = 32;
        blockDim.y = 8;

        monarch_conv_cuda_complex_kernel<32, 8, 8192, 2, 1, 16, false, 1, 1, 8><<<gridDim, blockDim, (2 * fftsize + 4 * 1024) * sizeof(half)>>>(
          static_cast<at::BFloat16 *>(x_real.data_ptr()),
          static_cast<at::BFloat16 *>(x_imag.data_ptr()),
          static_cast<complex_bfloat16_t *>(k_f.data_ptr()),
          static_cast<complex_bfloat16_t *>(f_32_fft.data_ptr()),
          static_cast<complex_bfloat16_t *>(f_16_fft.data_ptr()),
          static_cast<complex_bfloat16_t *>(twiddle_factors_N_fft.data_ptr()),
          static_cast<complex_bfloat16_t *>(twiddle_factors_16_fft.data_ptr()),
          static_cast<complex_bfloat16_t *>(f_32_ifft.data_ptr()),
          static_cast<complex_bfloat16_t *>(f_16_ifft.data_ptr()),
          static_cast<complex_bfloat16_t *>(twiddle_factors_N_ifft.data_ptr()),
          static_cast<complex_bfloat16_t *>(twiddle_factors_16_ifft.data_ptr()),
          static_cast<at::BFloat16 *>(out_real.data_ptr()),
          static_cast<at::BFloat16 *>(out_imag.data_ptr()),
          B,
          H,
          N);
      }
      
      break;
    default:
      AT_ERROR("Monarch forward not implemented for this sequence length");
  }
  
  CHECK_LAST_CUDA_ERROR();
  return std::make_pair(out_real, out_imag);
}

std::pair<torch::Tensor, torch::Tensor>
monarch_conv_cuda_16_32_32_complex_bf16_all(
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
  uint N
){

  uint B = x_real.size(0);
  uint H = x_real.size(1);
  // First: using WMMA
  dim3 gridDim;
  dim3 blockDim;

  // printf("gridDim.x = %d, gridDim.y = %d\n", gridDim.x, gridDim.y);
  torch::Tensor out_real = torch::empty({B, H, N}, x_real.options());
  torch::Tensor out_imag = torch::empty({B, H, N}, x_real.options());

  switch (fftsize) {
    case 16384:
      if (B >= 2 && (B % 2) == 0 && (H % 8) == 0) {
      // if (true) {
        gridDim.x = B / 2;
        gridDim.y = H / 8;

        blockDim.x = 32;
        blockDim.y = 8;

        CUDA_RT_CALL(cudaFuncSetAttribute(&monarch_conv_cuda_16_32_32_complex_kernel<32, 8, 16384, 1, 2, 16, false, 2, 8, 8>, cudaFuncAttributeMaxDynamicSharedMemorySize, 102400));

        monarch_conv_cuda_16_32_32_complex_kernel<32, 8, 16384, 1, 2, 16, false, 2, 8, 8><<<gridDim, blockDim, (2 * fftsize + 4 * 1024) * sizeof(half)>>>(
            static_cast<at::BFloat16 *>(x_real.data_ptr()),
            static_cast<at::BFloat16 *>(x_imag.data_ptr()),
            static_cast<complex_bfloat16_t *>(k_f.data_ptr()),
            static_cast<complex_bfloat16_t *>(f_16_fft.data_ptr()),
            static_cast<complex_bfloat16_t *>(f_32_fft.data_ptr()),
            static_cast<complex_bfloat16_t *>(twiddle_factors_N_fft.data_ptr()),
            static_cast<complex_bfloat16_t *>(twiddle_factors_32_fft.data_ptr()),
            static_cast<complex_bfloat16_t *>(f_16_ifft.data_ptr()),
            static_cast<complex_bfloat16_t *>(f_32_ifft.data_ptr()),
            static_cast<complex_bfloat16_t *>(twiddle_factors_N_ifft.data_ptr()),
            static_cast<complex_bfloat16_t *>(twiddle_factors_32_ifft.data_ptr()),
            static_cast<at::BFloat16 *>(out_real.data_ptr()),
            static_cast<at::BFloat16 *>(out_imag.data_ptr()),
            B,
            H,
            N);
      }
      else if ((H % 8) == 0) {
        gridDim.x = B;
        gridDim.y = H / 8;

        blockDim.x = 32;
        blockDim.y = 8;

        CUDA_RT_CALL(cudaFuncSetAttribute(&monarch_conv_cuda_16_32_32_complex_kernel<32, 8, 16384, 1, 2, 16, false, 1, 8, 8>, cudaFuncAttributeMaxDynamicSharedMemorySize, 102400));

        monarch_conv_cuda_16_32_32_complex_kernel<32, 8, 16384, 1, 2, 16, false, 1, 8, 8><<<gridDim, blockDim, (2 * fftsize + 4 * 1024) * sizeof(half)>>>(
          static_cast<at::BFloat16 *>(x_real.data_ptr()),
          static_cast<at::BFloat16 *>(x_imag.data_ptr()),
          static_cast<complex_bfloat16_t *>(k_f.data_ptr()),
          static_cast<complex_bfloat16_t *>(f_16_fft.data_ptr()),
          static_cast<complex_bfloat16_t *>(f_32_fft.data_ptr()),
          static_cast<complex_bfloat16_t *>(twiddle_factors_N_fft.data_ptr()),
          static_cast<complex_bfloat16_t *>(twiddle_factors_32_fft.data_ptr()),
          static_cast<complex_bfloat16_t *>(f_16_ifft.data_ptr()),
          static_cast<complex_bfloat16_t *>(f_32_ifft.data_ptr()),
          static_cast<complex_bfloat16_t *>(twiddle_factors_N_ifft.data_ptr()),
          static_cast<complex_bfloat16_t *>(twiddle_factors_32_ifft.data_ptr()),
          static_cast<at::BFloat16 *>(out_real.data_ptr()),
          static_cast<at::BFloat16 *>(out_imag.data_ptr()),
          B,
          H,
          N);
      }
      else {
        gridDim.x = B;
        gridDim.y = H;

        blockDim.x = 32;
        blockDim.y = 8;

        CUDA_RT_CALL(cudaFuncSetAttribute(&monarch_conv_cuda_16_32_32_complex_kernel<32, 8, 16384, 1, 2, 16, false, 1, 1, 8>, cudaFuncAttributeMaxDynamicSharedMemorySize, 102400));

        monarch_conv_cuda_16_32_32_complex_kernel<32, 8, 16384, 1, 2, 16, false, 1, 1, 8><<<gridDim, blockDim, (2 * fftsize + 4 * 1024) * sizeof(half)>>>(
          static_cast<at::BFloat16 *>(x_real.data_ptr()),
          static_cast<at::BFloat16 *>(x_imag.data_ptr()),
          static_cast<complex_bfloat16_t *>(k_f.data_ptr()),
          static_cast<complex_bfloat16_t *>(f_16_fft.data_ptr()),
          static_cast<complex_bfloat16_t *>(f_32_fft.data_ptr()),
          static_cast<complex_bfloat16_t *>(twiddle_factors_N_fft.data_ptr()),
          static_cast<complex_bfloat16_t *>(twiddle_factors_32_fft.data_ptr()),
          static_cast<complex_bfloat16_t *>(f_16_ifft.data_ptr()),
          static_cast<complex_bfloat16_t *>(f_32_ifft.data_ptr()),
          static_cast<complex_bfloat16_t *>(twiddle_factors_N_ifft.data_ptr()),
          static_cast<complex_bfloat16_t *>(twiddle_factors_32_ifft.data_ptr()),
          static_cast<at::BFloat16 *>(out_real.data_ptr()),
          static_cast<at::BFloat16 *>(out_imag.data_ptr()),
          B,
          H,
          N);
      }
      
      break;
    default:
      AT_ERROR("Monarch forward not implemented for this sequence length");
  }
  
  CHECK_LAST_CUDA_ERROR();
  return std::make_pair(out_real, out_imag);
}

std::pair<torch::Tensor, torch::Tensor>
monarch_conv_cuda_32_32_32_complex_bf16_all(
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
  uint N
){

  uint B = x_real.size(0);
  uint H = x_real.size(1);
  // First: using WMMA
  dim3 gridDim;
  dim3 blockDim;

  // printf("gridDim.x = %d, gridDim.y = %d\n", gridDim.x, gridDim.y);
  torch::Tensor out_real = torch::empty({B, H, N}, x_real.options());
  torch::Tensor out_imag = torch::empty({B, H, N}, x_real.options());

  switch (fftsize) {
    case 32768:
      if (B >= 2 && (B % 2) == 0 && (H % 8) == 0) {
        gridDim.x = B / 2;
        gridDim.y = H / 8;

        blockDim.x = 32;
        blockDim.y = 8;

        CUDA_RT_CALL(cudaFuncSetAttribute(&monarch_conv_cuda_32_32_32_complex_kernel<32, 8, 32768, 2, 16, false, 2, 8, 8>, cudaFuncAttributeMaxDynamicSharedMemorySize, 135168));

        monarch_conv_cuda_32_32_32_complex_kernel<32, 8, 32768, 2, 16, false, 2, 8, 8><<<gridDim, blockDim, (2 * fftsize) * sizeof(half)>>>(
            static_cast<at::BFloat16 *>(x_real.data_ptr()),
            static_cast<at::BFloat16 *>(x_imag.data_ptr()),
            static_cast<complex_bfloat16_t *>(k_f.data_ptr()),
            static_cast<complex_bfloat16_t *>(f_32_fft.data_ptr()),
            static_cast<complex_bfloat16_t *>(twiddle_factors_N_fft.data_ptr()),
            static_cast<complex_bfloat16_t *>(twiddle_factors_32_fft.data_ptr()),
            static_cast<complex_bfloat16_t *>(f_32_ifft.data_ptr()),
            static_cast<complex_bfloat16_t *>(twiddle_factors_N_ifft.data_ptr()),
            static_cast<complex_bfloat16_t *>(twiddle_factors_32_ifft.data_ptr()),
            static_cast<at::BFloat16 *>(out_real.data_ptr()),
            static_cast<at::BFloat16 *>(out_imag.data_ptr()),
            B,
            H,
            N);
      } else if ((H % 8) == 0) {
        gridDim.x = B;
        gridDim.y = H / 8;

        blockDim.x = 32;
        blockDim.y = 8;

        CUDA_RT_CALL(cudaFuncSetAttribute(&monarch_conv_cuda_32_32_32_complex_kernel<32, 8, 32768, 2, 16, false, 1, 8, 8>, cudaFuncAttributeMaxDynamicSharedMemorySize, 135168));

        monarch_conv_cuda_32_32_32_complex_kernel<32, 8, 32768, 2, 16, false, 1, 8, 8><<<gridDim, blockDim, (2 * fftsize) * sizeof(half)>>>(
            static_cast<at::BFloat16 *>(x_real.data_ptr()),
            static_cast<at::BFloat16 *>(x_imag.data_ptr()),
            static_cast<complex_bfloat16_t *>(k_f.data_ptr()),
            static_cast<complex_bfloat16_t *>(f_32_fft.data_ptr()),
            static_cast<complex_bfloat16_t *>(twiddle_factors_N_fft.data_ptr()),
            static_cast<complex_bfloat16_t *>(twiddle_factors_32_fft.data_ptr()),
            static_cast<complex_bfloat16_t *>(f_32_ifft.data_ptr()),
            static_cast<complex_bfloat16_t *>(twiddle_factors_N_ifft.data_ptr()),
            static_cast<complex_bfloat16_t *>(twiddle_factors_32_ifft.data_ptr()),
            static_cast<at::BFloat16 *>(out_real.data_ptr()),
            static_cast<at::BFloat16 *>(out_imag.data_ptr()),
            B,
            H,
            N);
      } else {
        gridDim.x = B;
        gridDim.y = H;

        blockDim.x = 32;
        blockDim.y = 8;

        CUDA_RT_CALL(cudaFuncSetAttribute(&monarch_conv_cuda_32_32_32_complex_kernel<32, 8, 32768, 2, 16, false, 1, 1, 8>, cudaFuncAttributeMaxDynamicSharedMemorySize, 135168));

        monarch_conv_cuda_32_32_32_complex_kernel<32, 8, 32768, 2, 16, false, 1, 1, 8><<<gridDim, blockDim, (2 * fftsize) * sizeof(half)>>>(
            static_cast<at::BFloat16 *>(x_real.data_ptr()),
            static_cast<at::BFloat16 *>(x_imag.data_ptr()),
            static_cast<complex_bfloat16_t *>(k_f.data_ptr()),
            static_cast<complex_bfloat16_t *>(f_32_fft.data_ptr()),
            static_cast<complex_bfloat16_t *>(twiddle_factors_N_fft.data_ptr()),
            static_cast<complex_bfloat16_t *>(twiddle_factors_32_fft.data_ptr()),
            static_cast<complex_bfloat16_t *>(f_32_ifft.data_ptr()),
            static_cast<complex_bfloat16_t *>(twiddle_factors_N_ifft.data_ptr()),
            static_cast<complex_bfloat16_t *>(twiddle_factors_32_ifft.data_ptr()),
            static_cast<at::BFloat16 *>(out_real.data_ptr()),
            static_cast<at::BFloat16 *>(out_imag.data_ptr()),
            B,
            H,
            N);
      }
      
      break;
    default:
      AT_ERROR("Monarch forward not implemented for this sequence length");
  }
  
  CHECK_LAST_CUDA_ERROR();
  return std::make_pair(out_real, out_imag);
}
