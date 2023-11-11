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
#include "kernels_fp16/monarch_cuda_kernel_r2r.h"
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
#define CHECK_LAST_CUDA_ERROR() checkLastFP16FwdR2R(__FILE__, __LINE__)
void checkLastFP16FwdR2R(const char* const file, const int line)
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
      // if (B >= 8 && (B % 8) == 0) {
      if (B >= 8 && (B % 8) == 0 & H >= 8 && (H % 8) == 0) {
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
          static_cast<complex_half_t *>(twid_r2r.data_ptr()),
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
          static_cast<complex_half_t *>(twid_r2r.data_ptr()),
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
          static_cast<complex_half_t *>(twid_r2r.data_ptr()),
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
      if (B >= 8 && (B % 8) == 0) {
        gridDim.x = B / 8;
        gridDim.y = H / 1;

        blockDim.x = 32;
        blockDim.y = 1;

        monarch_conv_cuda_kernel<32, 1, 1024, 2, false, 8, 1><<<gridDim, blockDim, 6 * fftsize * sizeof(half)>>>(
          static_cast<at::Half *>(x.data_ptr()),
          in_gate.has_value() ? static_cast<at::Half *>(in_gate.value().data_ptr()) : nullptr,
          static_cast<complex_half_t *>(k_f.data_ptr()),
          static_cast<complex_half_t *>(f_sqrt_N_fft.data_ptr()),
          static_cast<complex_half_t *>(twiddle_factors_fft.data_ptr()),
          static_cast<complex_half_t *>(twid_r2r.data_ptr()),
          static_cast<complex_half_t *>(f_sqrt_N_ifft.data_ptr()),
          static_cast<complex_half_t *>(twiddle_factors_ifft.data_ptr()),
          static_cast<at::Half *>(out.data_ptr()),
          out_gate.has_value() ? static_cast<at::Half *>(out_gate.value().data_ptr()) : nullptr,
          B,
          H,
          N,
          sqrt_N);
      } else if (B >= 4 && (B % 4) == 0) {
        gridDim.x = B / 4;
        gridDim.y = H;

        blockDim.x = 32;
        blockDim.y = 1;

        monarch_conv_cuda_kernel<32, 1, 1024, 2, false, 4, 1><<<gridDim, blockDim, 6 * fftsize * sizeof(half)>>>(
          static_cast<at::Half *>(x.data_ptr()),
          in_gate.has_value() ? static_cast<at::Half *>(in_gate.value().data_ptr()) : nullptr,
          static_cast<complex_half_t *>(k_f.data_ptr()),
          static_cast<complex_half_t *>(f_sqrt_N_fft.data_ptr()),
          static_cast<complex_half_t *>(twiddle_factors_fft.data_ptr()),
          static_cast<complex_half_t *>(twid_r2r.data_ptr()),
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

        monarch_conv_cuda_kernel<32, 1, 1024, 2, false, 1, 8><<<gridDim, blockDim, 6 * fftsize * sizeof(half)>>>(
          static_cast<at::Half *>(x.data_ptr()),
          in_gate.has_value() ? static_cast<at::Half *>(in_gate.value().data_ptr()) : nullptr,
          static_cast<complex_half_t *>(k_f.data_ptr()),
          static_cast<complex_half_t *>(f_sqrt_N_fft.data_ptr()),
          static_cast<complex_half_t *>(twiddle_factors_fft.data_ptr()),
          static_cast<complex_half_t *>(twid_r2r.data_ptr()),
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
          static_cast<complex_half_t *>(twid_r2r.data_ptr()),
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
      printf("fftsize = %d\n", fftsize);
      AT_ERROR("Monarch forward not implemented for this sequence length");
   }
   CHECK_LAST_CUDA_ERROR();
   return out;
}
