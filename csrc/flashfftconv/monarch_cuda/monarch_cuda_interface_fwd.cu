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
#include "kernels_fp16/monarch_cuda_kernel.h"
#include "kernels_fp16/monarch_cuda_16_16_16_kernel.h"
#include "kernels_fp16/monarch_cuda_32_16_16_kernel.h"
#include "kernels_fp16/monarch_cuda_16_32_32_kernel.h"
#include "kernels_fp16/monarch_cuda_32_32_32_kernel.h"
#include "kernels_fp16/monarch_cuda_32_32_32_complex_kernel.h"
#include "kernels_fp16/monarch_cuda_32_32_32_complex_truncated_kernel.h"
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
#define CHECK_LAST_CUDA_ERROR() checkLastFP16Fwd(__FILE__, __LINE__)
void checkLastFP16Fwd(const char* const file, const int line)
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

torch::Tensor monarch_conv_cuda(
  torch::Tensor x,
  torch::Tensor k_f,
  torch::Tensor f_sqrt_N_fft,
  torch::Tensor twiddle_factors_fft,
  torch::Tensor f_sqrt_N_ifft,
  torch::Tensor twiddle_factors_ifft,
  c10::optional<torch::Tensor> in_gate,
  c10::optional<torch::Tensor> out_gate,
  uint fftsize,
  uint N,
  uint sqrt_N
){

  uint B = x.size(0);
  uint H = x.size(1);
  // First: using WMMA
  dim3 gridDim;
  dim3 blockDim;

  torch::Tensor out = torch::empty({B, H, N}, x.options());

  switch (fftsize) {
    case 256:
      if (B >= 8 && (B % 8) == 0 && H >= 8 && (H % 8) == 0) {
        gridDim.x = B / 8;
        gridDim.y = H / 8;

        blockDim.x = 32;
        blockDim.y = 1;

        monarch_conv_cuda_kernel<32, 1, 256, 1, false, 8, 8><<<gridDim, blockDim, 6 * fftsize * sizeof(half)>>>(
          static_cast<at::Half *>(x.data_ptr()),
          in_gate.has_value() ? static_cast<at::Half *>(in_gate.value().data_ptr()) : nullptr,
          static_cast<complex_half_t *>(k_f.data_ptr()),
          static_cast<complex_half_t *>(f_sqrt_N_fft.data_ptr()),
          static_cast<complex_half_t *>(twiddle_factors_fft.data_ptr()),
          static_cast<complex_half_t *>(f_sqrt_N_ifft.data_ptr()),
          static_cast<complex_half_t *>(twiddle_factors_ifft.data_ptr()),
          static_cast<at::Half *>(out.data_ptr()),
          out_gate.has_value() ? static_cast<at::Half *>(out_gate.value().data_ptr()) : nullptr,
          B,
          H,
          N,
          sqrt_N);
      } else if (H >= 8 && (H % 8) == 0) {
        gridDim.x = B;
        gridDim.y = H / 8;

        blockDim.x = 32;
        blockDim.y = 1;

        monarch_conv_cuda_kernel<32, 1, 256, 1, false, 1, 8><<<gridDim, blockDim, 6 * fftsize * sizeof(half)>>>(
          static_cast<at::Half *>(x.data_ptr()),
          in_gate.has_value() ? static_cast<at::Half *>(in_gate.value().data_ptr()) : nullptr,
          static_cast<complex_half_t *>(k_f.data_ptr()),
          static_cast<complex_half_t *>(f_sqrt_N_fft.data_ptr()),
          static_cast<complex_half_t *>(twiddle_factors_fft.data_ptr()),
          static_cast<complex_half_t *>(f_sqrt_N_ifft.data_ptr()),
          static_cast<complex_half_t *>(twiddle_factors_ifft.data_ptr()),
          static_cast<at::Half *>(out.data_ptr()),
          out_gate.has_value() ? static_cast<at::Half *>(out_gate.value().data_ptr()) : nullptr,
          B,
          H,
          N,
          sqrt_N);
      } else {
        gridDim.x = B;
        gridDim.y = H;

        blockDim.x = 32;
        blockDim.y = 1;

        monarch_conv_cuda_kernel<32, 1, 256, 1, false, 1, 1><<<gridDim, blockDim, 6 * fftsize * sizeof(half)>>>(
          static_cast<at::Half *>(x.data_ptr()),
          in_gate.has_value() ? static_cast<at::Half *>(in_gate.value().data_ptr()) : nullptr,
          static_cast<complex_half_t *>(k_f.data_ptr()),
          static_cast<complex_half_t *>(f_sqrt_N_fft.data_ptr()),
          static_cast<complex_half_t *>(twiddle_factors_fft.data_ptr()),
          static_cast<complex_half_t *>(f_sqrt_N_ifft.data_ptr()),
          static_cast<complex_half_t *>(twiddle_factors_ifft.data_ptr()),
          static_cast<at::Half *>(out.data_ptr()),
          out_gate.has_value() ? static_cast<at::Half *>(out_gate.value().data_ptr()) : nullptr,
          B,
          H,
          N,
          sqrt_N);
      }
      
      break;
    case 1024:
      if (B >= 8 && (B % 8) == 0 && H >= 8 && (H % 8) == 0) {
        gridDim.x = B / 8;
        gridDim.y = H / 8;

        blockDim.x = 32;
        blockDim.y = 1;

        monarch_conv_cuda_kernel<32, 1, 1024, 2, false, 8, 8><<<gridDim, blockDim, 6 * fftsize * sizeof(half)>>>(
          static_cast<at::Half *>(x.data_ptr()),
          in_gate.has_value() ? static_cast<at::Half *>(in_gate.value().data_ptr()) : nullptr,
          static_cast<complex_half_t *>(k_f.data_ptr()),
          static_cast<complex_half_t *>(f_sqrt_N_fft.data_ptr()),
          static_cast<complex_half_t *>(twiddle_factors_fft.data_ptr()),
          static_cast<complex_half_t *>(f_sqrt_N_ifft.data_ptr()),
          static_cast<complex_half_t *>(twiddle_factors_ifft.data_ptr()),
          static_cast<at::Half *>(out.data_ptr()),
          out_gate.has_value() ? static_cast<at::Half *>(out_gate.value().data_ptr()) : nullptr,
          B,
          H,
          N,
          sqrt_N);
      } else if (B >= 4 && (B % 4) == 0 && H >= 8 && (H % 8) == 0) {
        gridDim.x = B / 4;
        gridDim.y = H / 8;

        blockDim.x = 32;
        blockDim.y = 1;

        monarch_conv_cuda_kernel<32, 1, 1024, 2, false, 4, 8><<<gridDim, blockDim, 6 * fftsize * sizeof(half)>>>(
          static_cast<at::Half *>(x.data_ptr()),
          in_gate.has_value() ? static_cast<at::Half *>(in_gate.value().data_ptr()) : nullptr,
          static_cast<complex_half_t *>(k_f.data_ptr()),
          static_cast<complex_half_t *>(f_sqrt_N_fft.data_ptr()),
          static_cast<complex_half_t *>(twiddle_factors_fft.data_ptr()),
          static_cast<complex_half_t *>(f_sqrt_N_ifft.data_ptr()),
          static_cast<complex_half_t *>(twiddle_factors_ifft.data_ptr()),
          static_cast<at::Half *>(out.data_ptr()),
          out_gate.has_value() ? static_cast<at::Half *>(out_gate.value().data_ptr()) : nullptr,
          B,
          H,
          N,
          sqrt_N);
      } else if ((H % 8) == 0) {
        gridDim.x = B;
        gridDim.y = H / 8;

        blockDim.x = 32;
        blockDim.y = 1;

        monarch_conv_cuda_kernel<32, 1, 1024, 2, false, 1, 8><<<gridDim, blockDim, 6 * fftsize * sizeof(half)>>>(
          static_cast<at::Half *>(x.data_ptr()),
          in_gate.has_value() ? static_cast<at::Half *>(in_gate.value().data_ptr()) : nullptr,
          static_cast<complex_half_t *>(k_f.data_ptr()),
          static_cast<complex_half_t *>(f_sqrt_N_fft.data_ptr()),
          static_cast<complex_half_t *>(twiddle_factors_fft.data_ptr()),
          static_cast<complex_half_t *>(f_sqrt_N_ifft.data_ptr()),
          static_cast<complex_half_t *>(twiddle_factors_ifft.data_ptr()),
          static_cast<at::Half *>(out.data_ptr()),
          out_gate.has_value() ? static_cast<at::Half *>(out_gate.value().data_ptr()) : nullptr,
          B,
          H,
          N,
          sqrt_N);
      } else {
        gridDim.x = B;
        gridDim.y = H;

        blockDim.x = 32;
        blockDim.y = 1;

        monarch_conv_cuda_kernel<32, 1, 1024, 2, false, 1, 1><<<gridDim, blockDim, 6 * fftsize * sizeof(half)>>>(
          static_cast<at::Half *>(x.data_ptr()),
          in_gate.has_value() ? static_cast<at::Half *>(in_gate.value().data_ptr()) : nullptr,
          static_cast<complex_half_t *>(k_f.data_ptr()),
          static_cast<complex_half_t *>(f_sqrt_N_fft.data_ptr()),
          static_cast<complex_half_t *>(twiddle_factors_fft.data_ptr()),
          static_cast<complex_half_t *>(f_sqrt_N_ifft.data_ptr()),
          static_cast<complex_half_t *>(twiddle_factors_ifft.data_ptr()),
          static_cast<at::Half *>(out.data_ptr()),
          out_gate.has_value() ? static_cast<at::Half *>(out_gate.value().data_ptr()) : nullptr,
          B,
          H,
          N,
          sqrt_N);
      }
      break;
    default:
      AT_ERROR("Monarch forward not implemented for this sequence length");
   }
   CHECK_LAST_CUDA_ERROR();
   return out;
}

torch::Tensor monarch_conv_cuda_16_16_16(
  torch::Tensor x,
  torch::Tensor k_f,
  torch::Tensor f_16_fft,
  torch::Tensor twiddle_factors_256_fft,
  torch::Tensor twiddle_factors_16_fft,
  torch::Tensor f_16_ifft,
  torch::Tensor twiddle_factors_256_ifft,
  torch::Tensor twiddle_factors_16_ifft,
  c10::optional<torch::Tensor> in_gate,
  c10::optional<torch::Tensor> out_gate,
  uint fftsize,
  uint N,
  uint sqrt_N
){

  uint B = x.size(0);
  uint H = x.size(1);
  // First: using WMMA
  dim3 gridDim;
  dim3 blockDim;

  // printf("gridDim.x = %d, gridDim.y = %d\n", gridDim.x, gridDim.y);
  torch::Tensor out = torch::empty({B, H, N}, x.options());

  switch (fftsize) {
    case 4096:
      if (B >= 4 && (B % 4) == 0 && H >= 8 && (H % 8) == 0) {
        gridDim.x = B / 4;
        gridDim.y = H / 8;

        blockDim.x = 32;
        blockDim.y = 4;

        monarch_conv_cuda_kernel<32, 4, 4096, 1, 16, false, 4, 8, 4><<<gridDim, blockDim, (2 * fftsize + 4 * 256) * sizeof(half)>>>(
            static_cast<at::Half *>(x.data_ptr()),
            in_gate.has_value() ? static_cast<at::Half *>(in_gate.value().data_ptr()) : nullptr,
            static_cast<complex_half_t *>(k_f.data_ptr()),
            static_cast<complex_half_t *>(f_16_fft.data_ptr()),
            static_cast<complex_half_t *>(twiddle_factors_256_fft.data_ptr()),
            static_cast<complex_half_t *>(twiddle_factors_16_fft.data_ptr()),
            static_cast<complex_half_t *>(f_16_ifft.data_ptr()),
            static_cast<complex_half_t *>(twiddle_factors_256_ifft.data_ptr()),
            static_cast<complex_half_t *>(twiddle_factors_16_ifft.data_ptr()),
            static_cast<at::Half *>(out.data_ptr()),
            out_gate.has_value() ? static_cast<at::Half *>(out_gate.value().data_ptr()) : nullptr,
            B,
            H,
            N,
            sqrt_N);
        
      } else if (B == 2 && (B % 2) == 0 && H >= 8 && (H % 8) == 0) {
        gridDim.x = B / 2;
        gridDim.y = H / 8;

        blockDim.x = 32;
        blockDim.y = 4;

        monarch_conv_cuda_kernel<32, 4, 4096, 1, 16, false, 2, 8, 4><<<gridDim, blockDim, (2 * fftsize + 4 * 256) * sizeof(half)>>>(
            static_cast<at::Half *>(x.data_ptr()),
            in_gate.has_value() ? static_cast<at::Half *>(in_gate.value().data_ptr()) : nullptr,
            static_cast<complex_half_t *>(k_f.data_ptr()),
            static_cast<complex_half_t *>(f_16_fft.data_ptr()),
            static_cast<complex_half_t *>(twiddle_factors_256_fft.data_ptr()),
            static_cast<complex_half_t *>(twiddle_factors_16_fft.data_ptr()),
            static_cast<complex_half_t *>(f_16_ifft.data_ptr()),
            static_cast<complex_half_t *>(twiddle_factors_256_ifft.data_ptr()),
            static_cast<complex_half_t *>(twiddle_factors_16_ifft.data_ptr()),
            static_cast<at::Half *>(out.data_ptr()),
            out_gate.has_value() ? static_cast<at::Half *>(out_gate.value().data_ptr()) : nullptr,
            B,
            H,
            N,
            sqrt_N);
      } else if (H >= 8 && (H % 8) == 0) {
        gridDim.x = B;
        gridDim.y = H / 8;

        blockDim.x = 32;
        blockDim.y = 4;

        monarch_conv_cuda_kernel<32, 4, 4096, 1, 16, false, 1, 8, 4><<<gridDim, blockDim, (2 * fftsize + 4 * 256) * sizeof(half)>>>(
            static_cast<at::Half *>(x.data_ptr()),
            in_gate.has_value() ? static_cast<at::Half *>(in_gate.value().data_ptr()) : nullptr,
            static_cast<complex_half_t *>(k_f.data_ptr()),
            static_cast<complex_half_t *>(f_16_fft.data_ptr()),
            static_cast<complex_half_t *>(twiddle_factors_256_fft.data_ptr()),
            static_cast<complex_half_t *>(twiddle_factors_16_fft.data_ptr()),
            static_cast<complex_half_t *>(f_16_ifft.data_ptr()),
            static_cast<complex_half_t *>(twiddle_factors_256_ifft.data_ptr()),
            static_cast<complex_half_t *>(twiddle_factors_16_ifft.data_ptr()),
            static_cast<at::Half *>(out.data_ptr()),
            out_gate.has_value() ? static_cast<at::Half *>(out_gate.value().data_ptr()) : nullptr,
            B,
            H,
            N,
            sqrt_N);
      } else {
        gridDim.x = B;
        gridDim.y = H;

        blockDim.x = 32;
        blockDim.y = 4;

        monarch_conv_cuda_kernel<32, 4, 4096, 1, 16, false, 1, 1, 4><<<gridDim, blockDim, (2 * fftsize + 4 * 256) * sizeof(half)>>>(
            static_cast<at::Half *>(x.data_ptr()),
            in_gate.has_value() ? static_cast<at::Half *>(in_gate.value().data_ptr()) : nullptr,
            static_cast<complex_half_t *>(k_f.data_ptr()),
            static_cast<complex_half_t *>(f_16_fft.data_ptr()),
            static_cast<complex_half_t *>(twiddle_factors_256_fft.data_ptr()),
            static_cast<complex_half_t *>(twiddle_factors_16_fft.data_ptr()),
            static_cast<complex_half_t *>(f_16_ifft.data_ptr()),
            static_cast<complex_half_t *>(twiddle_factors_256_ifft.data_ptr()),
            static_cast<complex_half_t *>(twiddle_factors_16_ifft.data_ptr()),
            static_cast<at::Half *>(out.data_ptr()),
            out_gate.has_value() ? static_cast<at::Half *>(out_gate.value().data_ptr()) : nullptr,
            B,
            H,
            N,
            sqrt_N);
      }
      break;
    default:
      AT_ERROR("Monarch forward not implemented for this sequence length");
  }
  
  CHECK_LAST_CUDA_ERROR();
  return out;
}

torch::Tensor monarch_conv_cuda_32_16_16(
  torch::Tensor x,
  torch::Tensor k_f,
  torch::Tensor f_32_fft,
  torch::Tensor f_16_fft,
  torch::Tensor twiddle_factors_N_fft,
  torch::Tensor twiddle_factors_16_fft,
  torch::Tensor f_32_ifft,
  torch::Tensor f_16_ifft,
  torch::Tensor twiddle_factors_N_ifft,
  torch::Tensor twiddle_factors_16_ifft,
  c10::optional<torch::Tensor> in_gate,
  c10::optional<torch::Tensor> out_gate,
  uint fftsize,
  uint N
){

  uint B = x.size(0);
  uint H = x.size(1);
  // First: using WMMA
  dim3 gridDim;
  dim3 blockDim;

  // printf("gridDim.x = %d, gridDim.y = %d\n", gridDim.x, gridDim.y);
  torch::Tensor out = torch::empty({B, H, N}, x.options());

  switch (fftsize) {
    case 8192:
      if (B >= 8 && (B % 8) == 0 && H >= 8 && (H % 8) == 0) {
        gridDim.x = B / 8;
        gridDim.y = H / 8;

        blockDim.x = 32;
        blockDim.y = 8;

        monarch_conv_cuda_kernel<32, 8, 8192, 2, 1, 16, false, 8, 8, 8><<<gridDim, blockDim, (2 * fftsize + 4 * 1024) * sizeof(half)>>>(
            static_cast<at::Half *>(x.data_ptr()),
            in_gate.has_value() ? static_cast<at::Half *>(in_gate.value().data_ptr()) : nullptr,
            static_cast<complex_half_t *>(k_f.data_ptr()),
            static_cast<complex_half_t *>(f_32_fft.data_ptr()),
            static_cast<complex_half_t *>(f_16_fft.data_ptr()),
            static_cast<complex_half_t *>(twiddle_factors_N_fft.data_ptr()),
            static_cast<complex_half_t *>(twiddle_factors_16_fft.data_ptr()),
            static_cast<complex_half_t *>(f_32_ifft.data_ptr()),
            static_cast<complex_half_t *>(f_16_ifft.data_ptr()),
            static_cast<complex_half_t *>(twiddle_factors_N_ifft.data_ptr()),
            static_cast<complex_half_t *>(twiddle_factors_16_ifft.data_ptr()),
            static_cast<at::Half *>(out.data_ptr()),
            out_gate.has_value() ? static_cast<at::Half *>(out_gate.value().data_ptr()) : nullptr,
            B,
            H,
            N);
      } else if (H >= 8 && (H % 8) == 0) {
        gridDim.x = B;
        gridDim.y = H / 8;

        blockDim.x = 32;
        blockDim.y = 8;

        monarch_conv_cuda_kernel<32, 8, 8192, 2, 1, 16, false, 1, 8, 8><<<gridDim, blockDim, (2 * fftsize + 4 * 1024) * sizeof(half)>>>(
            static_cast<at::Half *>(x.data_ptr()),
            in_gate.has_value() ? static_cast<at::Half *>(in_gate.value().data_ptr()) : nullptr,
            static_cast<complex_half_t *>(k_f.data_ptr()),
            static_cast<complex_half_t *>(f_32_fft.data_ptr()),
            static_cast<complex_half_t *>(f_16_fft.data_ptr()),
            static_cast<complex_half_t *>(twiddle_factors_N_fft.data_ptr()),
            static_cast<complex_half_t *>(twiddle_factors_16_fft.data_ptr()),
            static_cast<complex_half_t *>(f_32_ifft.data_ptr()),
            static_cast<complex_half_t *>(f_16_ifft.data_ptr()),
            static_cast<complex_half_t *>(twiddle_factors_N_ifft.data_ptr()),
            static_cast<complex_half_t *>(twiddle_factors_16_ifft.data_ptr()),
            static_cast<at::Half *>(out.data_ptr()),
            out_gate.has_value() ? static_cast<at::Half *>(out_gate.value().data_ptr()) : nullptr,
            B,
            H,
            N);
      } else {
        gridDim.x = B;
        gridDim.y = H;

        blockDim.x = 32;
        blockDim.y = 8;

        monarch_conv_cuda_kernel<32, 8, 8192, 2, 1, 16, false, 1, 1, 8><<<gridDim, blockDim, (2 * fftsize + 4 * 1024) * sizeof(half)>>>(
            static_cast<at::Half *>(x.data_ptr()),
            in_gate.has_value() ? static_cast<at::Half *>(in_gate.value().data_ptr()) : nullptr,
            static_cast<complex_half_t *>(k_f.data_ptr()),
            static_cast<complex_half_t *>(f_32_fft.data_ptr()),
            static_cast<complex_half_t *>(f_16_fft.data_ptr()),
            static_cast<complex_half_t *>(twiddle_factors_N_fft.data_ptr()),
            static_cast<complex_half_t *>(twiddle_factors_16_fft.data_ptr()),
            static_cast<complex_half_t *>(f_32_ifft.data_ptr()),
            static_cast<complex_half_t *>(f_16_ifft.data_ptr()),
            static_cast<complex_half_t *>(twiddle_factors_N_ifft.data_ptr()),
            static_cast<complex_half_t *>(twiddle_factors_16_ifft.data_ptr()),
            static_cast<at::Half *>(out.data_ptr()),
            out_gate.has_value() ? static_cast<at::Half *>(out_gate.value().data_ptr()) : nullptr,
            B,
            H,
            N);
      }
      
      break;
    default:
      AT_ERROR("Monarch forward not implemented for this sequence length");
  }
  
  CHECK_LAST_CUDA_ERROR();
  return out;
}

torch::Tensor monarch_conv_cuda_16_32_32(
  torch::Tensor x,
  torch::Tensor k_f,
  torch::Tensor f_16_fft,
  torch::Tensor f_32_fft,
  torch::Tensor twiddle_factors_N_fft,
  torch::Tensor twiddle_factors_32_fft,
  torch::Tensor f_16_ifft,
  torch::Tensor f_32_ifft,
  torch::Tensor twiddle_factors_N_ifft,
  torch::Tensor twiddle_factors_32_ifft,
  c10::optional<torch::Tensor> in_gate,
  c10::optional<torch::Tensor> out_gate,
  uint fftsize,
  uint N
){

  uint B = x.size(0);
  uint H = x.size(1);
  // First: using WMMA
  dim3 gridDim;
  dim3 blockDim;

  // printf("gridDim.x = %d, gridDim.y = %d\n", gridDim.x, gridDim.y);
  torch::Tensor out = torch::empty({B, H, N}, x.options());

  switch (fftsize) {
    case 16384:
      if (B >= 8 && (B % 8) == 0 && H >= 8 && (H % 8) == 0) {
        gridDim.x = B / 8;
        gridDim.y = H / 8;

        blockDim.x = 32;
        blockDim.y = 8;

        CUDA_RT_CALL(cudaFuncSetAttribute(&monarch_conv_cuda_16_32_32_kernel<32, 8, 16384, 1, 2, 16, false, 8, 8, 8>, cudaFuncAttributeMaxDynamicSharedMemorySize, 102400));

        monarch_conv_cuda_16_32_32_kernel<32, 8, 16384, 1, 2, 16, false, 8, 8, 8><<<gridDim, blockDim, (2 * fftsize + 4 * 1024) * sizeof(half)>>>(
            static_cast<at::Half *>(x.data_ptr()),
            in_gate.has_value() ? static_cast<at::Half *>(in_gate.value().data_ptr()) : nullptr,
            static_cast<complex_half_t *>(k_f.data_ptr()),
            static_cast<complex_half_t *>(f_16_fft.data_ptr()),
            static_cast<complex_half_t *>(f_32_fft.data_ptr()),
            static_cast<complex_half_t *>(twiddle_factors_N_fft.data_ptr()),
            static_cast<complex_half_t *>(twiddle_factors_32_fft.data_ptr()),
            static_cast<complex_half_t *>(f_16_ifft.data_ptr()),
            static_cast<complex_half_t *>(f_32_ifft.data_ptr()),
            static_cast<complex_half_t *>(twiddle_factors_N_ifft.data_ptr()),
            static_cast<complex_half_t *>(twiddle_factors_32_ifft.data_ptr()),
            static_cast<at::Half *>(out.data_ptr()),
            out_gate.has_value() ? static_cast<at::Half *>(out_gate.value().data_ptr()) : nullptr,
            B,
            H,
            N);
      } else if (B >= 4 && (B % 4) == 0 && (H % 8) == 0) {
        gridDim.x = B / 4;
        gridDim.y = H / 8;

        blockDim.x = 32;
        blockDim.y = 8;

        CUDA_RT_CALL(cudaFuncSetAttribute(&monarch_conv_cuda_16_32_32_kernel<32, 8, 16384, 1, 2, 16, false, 4, 8, 8>, cudaFuncAttributeMaxDynamicSharedMemorySize, 102400));

        monarch_conv_cuda_16_32_32_kernel<32, 8, 16384, 1, 2, 16, false, 4, 8, 8><<<gridDim, blockDim, (2 * fftsize + 4 * 1024) * sizeof(half)>>>(
            static_cast<at::Half *>(x.data_ptr()),
            in_gate.has_value() ? static_cast<at::Half *>(in_gate.value().data_ptr()) : nullptr,
            static_cast<complex_half_t *>(k_f.data_ptr()),
            static_cast<complex_half_t *>(f_16_fft.data_ptr()),
            static_cast<complex_half_t *>(f_32_fft.data_ptr()),
            static_cast<complex_half_t *>(twiddle_factors_N_fft.data_ptr()),
            static_cast<complex_half_t *>(twiddle_factors_32_fft.data_ptr()),
            static_cast<complex_half_t *>(f_16_ifft.data_ptr()),
            static_cast<complex_half_t *>(f_32_ifft.data_ptr()),
            static_cast<complex_half_t *>(twiddle_factors_N_ifft.data_ptr()),
            static_cast<complex_half_t *>(twiddle_factors_32_ifft.data_ptr()),
            static_cast<at::Half *>(out.data_ptr()),
            out_gate.has_value() ? static_cast<at::Half *>(out_gate.value().data_ptr()) : nullptr,
            B,
            H,
            N);
      } else if (B >= 2 && (B % 2) == 0 && H >= 8 && (H % 8) == 0) {
        gridDim.x = B / 2;
        gridDim.y = H / 8;

        blockDim.x = 32;
        blockDim.y = 8;

        CUDA_RT_CALL(cudaFuncSetAttribute(&monarch_conv_cuda_16_32_32_kernel<32, 8, 16384, 1, 2, 16, false, 2, 8, 8>, cudaFuncAttributeMaxDynamicSharedMemorySize, 102400));

        monarch_conv_cuda_16_32_32_kernel<32, 8, 16384, 1, 2, 16, false, 2, 8, 8><<<gridDim, blockDim, (2 * fftsize + 4 * 1024) * sizeof(half)>>>(
            static_cast<at::Half *>(x.data_ptr()),
            in_gate.has_value() ? static_cast<at::Half *>(in_gate.value().data_ptr()) : nullptr,
            static_cast<complex_half_t *>(k_f.data_ptr()),
            static_cast<complex_half_t *>(f_16_fft.data_ptr()),
            static_cast<complex_half_t *>(f_32_fft.data_ptr()),
            static_cast<complex_half_t *>(twiddle_factors_N_fft.data_ptr()),
            static_cast<complex_half_t *>(twiddle_factors_32_fft.data_ptr()),
            static_cast<complex_half_t *>(f_16_ifft.data_ptr()),
            static_cast<complex_half_t *>(f_32_ifft.data_ptr()),
            static_cast<complex_half_t *>(twiddle_factors_N_ifft.data_ptr()),
            static_cast<complex_half_t *>(twiddle_factors_32_ifft.data_ptr()),
            static_cast<at::Half *>(out.data_ptr()),
            out_gate.has_value() ? static_cast<at::Half *>(out_gate.value().data_ptr()) : nullptr,
            B,
            H,
            N);
      } else if (H >= 8 && (H % 8) == 0) {
        gridDim.x = B;
        gridDim.y = H / 8;

        blockDim.x = 32;
        blockDim.y = 8;

        CUDA_RT_CALL(cudaFuncSetAttribute(&monarch_conv_cuda_16_32_32_kernel<32, 8, 16384, 1, 2, 16, false, 1, 8, 8>, cudaFuncAttributeMaxDynamicSharedMemorySize, 102400));

        monarch_conv_cuda_16_32_32_kernel<32, 8, 16384, 1, 2, 16, false, 1, 8, 8><<<gridDim, blockDim, (2 * fftsize + 4 * 1024) * sizeof(half)>>>(
            static_cast<at::Half *>(x.data_ptr()),
            in_gate.has_value() ? static_cast<at::Half *>(in_gate.value().data_ptr()) : nullptr,
            static_cast<complex_half_t *>(k_f.data_ptr()),
            static_cast<complex_half_t *>(f_16_fft.data_ptr()),
            static_cast<complex_half_t *>(f_32_fft.data_ptr()),
            static_cast<complex_half_t *>(twiddle_factors_N_fft.data_ptr()),
            static_cast<complex_half_t *>(twiddle_factors_32_fft.data_ptr()),
            static_cast<complex_half_t *>(f_16_ifft.data_ptr()),
            static_cast<complex_half_t *>(f_32_ifft.data_ptr()),
            static_cast<complex_half_t *>(twiddle_factors_N_ifft.data_ptr()),
            static_cast<complex_half_t *>(twiddle_factors_32_ifft.data_ptr()),
            static_cast<at::Half *>(out.data_ptr()),
            out_gate.has_value() ? static_cast<at::Half *>(out_gate.value().data_ptr()) : nullptr,
            B,
            H,
            N);
      } else {
        gridDim.x = B;
        gridDim.y = H;

        blockDim.x = 32;
        blockDim.y = 8;

        CUDA_RT_CALL(cudaFuncSetAttribute(&monarch_conv_cuda_16_32_32_kernel<32, 8, 16384, 1, 2, 16, false, 1, 1, 8>, cudaFuncAttributeMaxDynamicSharedMemorySize, 102400));

        monarch_conv_cuda_16_32_32_kernel<32, 8, 16384, 1, 2, 16, false, 1, 1, 8><<<gridDim, blockDim, (2 * fftsize + 4 * 1024) * sizeof(half)>>>(
            static_cast<at::Half *>(x.data_ptr()),
            in_gate.has_value() ? static_cast<at::Half *>(in_gate.value().data_ptr()) : nullptr,
            static_cast<complex_half_t *>(k_f.data_ptr()),
            static_cast<complex_half_t *>(f_16_fft.data_ptr()),
            static_cast<complex_half_t *>(f_32_fft.data_ptr()),
            static_cast<complex_half_t *>(twiddle_factors_N_fft.data_ptr()),
            static_cast<complex_half_t *>(twiddle_factors_32_fft.data_ptr()),
            static_cast<complex_half_t *>(f_16_ifft.data_ptr()),
            static_cast<complex_half_t *>(f_32_ifft.data_ptr()),
            static_cast<complex_half_t *>(twiddle_factors_N_ifft.data_ptr()),
            static_cast<complex_half_t *>(twiddle_factors_32_ifft.data_ptr()),
            static_cast<at::Half *>(out.data_ptr()),
            out_gate.has_value() ? static_cast<at::Half *>(out_gate.value().data_ptr()) : nullptr,
            B,
            H,
            N);
      }
      
      break;
    default:
      AT_ERROR("Monarch forward not implemented for this sequence length");
  }
  
  CHECK_LAST_CUDA_ERROR();
  return out;
}

torch::Tensor monarch_conv_cuda_32_32_32(
  torch::Tensor x,
  torch::Tensor k_f,
  torch::Tensor f_32_fft,
  torch::Tensor twiddle_factors_N_fft,
  torch::Tensor twiddle_factors_32_fft,
  torch::Tensor f_32_ifft,
  torch::Tensor twiddle_factors_N_ifft,
  torch::Tensor twiddle_factors_32_ifft,
  c10::optional<torch::Tensor> in_gate,
  c10::optional<torch::Tensor> out_gate,
  uint fftsize,
  uint N
){

  uint B = x.size(0);
  uint H = x.size(1);
  // First: using WMMA
  dim3 gridDim;
  dim3 blockDim;

  // printf("gridDim.x = %d, gridDim.y = %d\n", gridDim.x, gridDim.y);
  torch::Tensor out = torch::empty({B, H, N}, x.options());

  switch (fftsize) {
    case 32768:
      if (B >= 2 && (B % 2) == 0 && H >= 8 && (H % 8) == 0) {
        gridDim.x = B / 2;
        gridDim.y = H / 8;

        blockDim.x = 32;
        blockDim.y = 8;

        CUDA_RT_CALL(cudaFuncSetAttribute(&monarch_conv_cuda_32_32_32_kernel<32, 8, 32768, 2, 16, false, 2, 8, 8>, cudaFuncAttributeMaxDynamicSharedMemorySize, 135168));

        monarch_conv_cuda_32_32_32_kernel<32, 8, 32768, 2, 16, false, 2, 8,8><<<gridDim, blockDim, (2 * fftsize) * sizeof(half)>>>(
            static_cast<at::Half *>(x.data_ptr()),
            in_gate.has_value() ? static_cast<at::Half *>(in_gate.value().data_ptr()) : nullptr,
            static_cast<complex_half_t *>(k_f.data_ptr()),
            static_cast<complex_half_t *>(f_32_fft.data_ptr()),
            static_cast<complex_half_t *>(twiddle_factors_N_fft.data_ptr()),
            static_cast<complex_half_t *>(twiddle_factors_32_fft.data_ptr()),
            static_cast<complex_half_t *>(f_32_ifft.data_ptr()),
            static_cast<complex_half_t *>(twiddle_factors_N_ifft.data_ptr()),
            static_cast<complex_half_t *>(twiddle_factors_32_ifft.data_ptr()),
            static_cast<at::Half *>(out.data_ptr()),
            out_gate.has_value() ? static_cast<at::Half *>(out_gate.value().data_ptr()) : nullptr,
            B,
            H,
            N);
      } else if (H >= 8 && (H % 8) == 0) {
        gridDim.x = B;
        gridDim.y = H / 8;

        blockDim.x = 32;
        blockDim.y = 8;

        CUDA_RT_CALL(cudaFuncSetAttribute(&monarch_conv_cuda_32_32_32_kernel<32, 8, 32768, 2, 16, false, 1, 8, 8>, cudaFuncAttributeMaxDynamicSharedMemorySize, 135168));

        monarch_conv_cuda_32_32_32_kernel<32, 8, 32768, 2, 16, false, 1, 8,8><<<gridDim, blockDim, (2 * fftsize) * sizeof(half)>>>(
            static_cast<at::Half *>(x.data_ptr()),
            in_gate.has_value() ? static_cast<at::Half *>(in_gate.value().data_ptr()) : nullptr,
            static_cast<complex_half_t *>(k_f.data_ptr()),
            static_cast<complex_half_t *>(f_32_fft.data_ptr()),
            static_cast<complex_half_t *>(twiddle_factors_N_fft.data_ptr()),
            static_cast<complex_half_t *>(twiddle_factors_32_fft.data_ptr()),
            static_cast<complex_half_t *>(f_32_ifft.data_ptr()),
            static_cast<complex_half_t *>(twiddle_factors_N_ifft.data_ptr()),
            static_cast<complex_half_t *>(twiddle_factors_32_ifft.data_ptr()),
            static_cast<at::Half *>(out.data_ptr()),
            out_gate.has_value() ? static_cast<at::Half *>(out_gate.value().data_ptr()) : nullptr,
            B,
            H,
            N);
      } else {
        gridDim.x = B;
        gridDim.y = H;

        blockDim.x = 32;
        blockDim.y = 8;

        CUDA_RT_CALL(cudaFuncSetAttribute(&monarch_conv_cuda_32_32_32_kernel<32, 8, 32768, 2, 16, false, 1, 1, 8>, cudaFuncAttributeMaxDynamicSharedMemorySize, 135168));

        monarch_conv_cuda_32_32_32_kernel<32, 8, 32768, 2, 16, false, 1, 1,8><<<gridDim, blockDim, (2 * fftsize) * sizeof(half)>>>(
            static_cast<at::Half *>(x.data_ptr()),
            in_gate.has_value() ? static_cast<at::Half *>(in_gate.value().data_ptr()) : nullptr,
            static_cast<complex_half_t *>(k_f.data_ptr()),
            static_cast<complex_half_t *>(f_32_fft.data_ptr()),
            static_cast<complex_half_t *>(twiddle_factors_N_fft.data_ptr()),
            static_cast<complex_half_t *>(twiddle_factors_32_fft.data_ptr()),
            static_cast<complex_half_t *>(f_32_ifft.data_ptr()),
            static_cast<complex_half_t *>(twiddle_factors_N_ifft.data_ptr()),
            static_cast<complex_half_t *>(twiddle_factors_32_ifft.data_ptr()),
            static_cast<at::Half *>(out.data_ptr()),
            out_gate.has_value() ? static_cast<at::Half *>(out_gate.value().data_ptr()) : nullptr,
            B,
            H,
            N);
      }
      
      break;
    default:
      AT_ERROR("Monarch forward not implemented for this sequence length");
  }
  
  CHECK_LAST_CUDA_ERROR();
  return out;
}
