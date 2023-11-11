// Copyright (c) 2023 Dan Fu, Hermann Kumbong

#include <torch/extension.h>

#include <vector>
#include <stdio.h>
#include <mma.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cub/block/block_load.cuh>
#include <cub/block/block_store.cuh>
#include "kernels_bf16/monarch_cuda_bwd_kernel_r2r_bf16.h"
#include "kernels_fp16/monarch_cuda_shared.h"
#include "kernels_bf16/monarch_cuda_shared_bf16.h"
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
#define CHECK_LAST_CUDA_ERROR() checkLastBF16BwdR2R(__FILE__, __LINE__)
void checkLastBF16BwdR2R(const char* const file, const int line)
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
  uint sqrt_N
){

  uint B = x.size(0);
  uint H = x.size(1);
  // First: using WMMA
  dim3 gridDim;
  dim3 blockDim;

  // printf("gridDim.x = %d, gridDim.y = %d\n", gridDim.x, gridDim.y);
  torch::Tensor dx_out = torch::empty({B, H, N}, x.options());
  torch::Tensor dk_f_out;
  torch::Tensor din_gate;
  torch::Tensor dout_gate;

  if(in_gate.has_value()){
    din_gate = torch::empty_like(in_gate.value());
  }

  if(out_gate.has_value()){
    dout_gate = torch::empty_like(out_gate.value());
  }

  switch (fftsize) {
    case 256:
      // if (true) {
      if (B >= 2 && (B % 2) == 0 && (H % 4) == 0) {
        gridDim.x = B / 2;
        gridDim.y = H / 4;
        // gridDim.x = B;
        // gridDim.y = H;

        blockDim.x = 32;
        blockDim.y = 1;

        dk_f_out = torch::empty({B / 2, H, fftsize + 1, 2}, x.options());

        monarch_conv_bwd_cuda_kernel<32, 1, 256, 1, false, 2, 4><<<gridDim, blockDim, 8 * fftsize * sizeof(half)>>>(
            static_cast<at::BFloat16 *>(dout.data_ptr()),
            static_cast<at::BFloat16 *>(x.data_ptr()),
            static_cast<complex_bfloat16_t *>(k_f.data_ptr()),
            static_cast<complex_bfloat16_t *>(f_sqrt_N_fft.data_ptr()),
            static_cast<complex_bfloat16_t *>(twiddle_factors_fft.data_ptr()),
            static_cast<complex_bfloat16_t *>(twid_r2r.data_ptr()),
            static_cast<complex_bfloat16_t *>(f_sqrt_N_ifft.data_ptr()),
            static_cast<complex_bfloat16_t *>(twiddle_factors_ifft.data_ptr()),
            static_cast<at::BFloat16 *>(dx_out.data_ptr()),
            static_cast<complex_bfloat16_t *>(dk_f_out.data_ptr()),
            in_gate.has_value() ? static_cast<at::BFloat16 *>(in_gate.value().data_ptr()) : nullptr,
            out_gate.has_value() ? static_cast<at::BFloat16 *>(out_gate.value().data_ptr()) : nullptr,
            in_gate.has_value() ? static_cast<at::BFloat16 *>(din_gate.data_ptr()) : nullptr,
            out_gate.has_value() ? static_cast<at::BFloat16 *>(dout_gate.data_ptr()) : nullptr,
            B,
            H,
            N,
            sqrt_N);
      }
      else if ((H % 4) == 0) {
        gridDim.x = B;
        gridDim.y = H / 4;

        blockDim.x = 32;
        blockDim.y = 1;

        dk_f_out = torch::empty({B, H, fftsize + 1, 2}, x.options());

        monarch_conv_bwd_cuda_kernel<32, 1, 256, 1, false, 1, 4><<<gridDim, blockDim, 8 * fftsize * sizeof(half)>>>(
            static_cast<at::BFloat16 *>(dout.data_ptr()),
            static_cast<at::BFloat16 *>(x.data_ptr()),
            static_cast<complex_bfloat16_t *>(k_f.data_ptr()),
            static_cast<complex_bfloat16_t *>(f_sqrt_N_fft.data_ptr()),
            static_cast<complex_bfloat16_t *>(twiddle_factors_fft.data_ptr()),
            static_cast<complex_bfloat16_t *>(twid_r2r.data_ptr()),
            static_cast<complex_bfloat16_t *>(f_sqrt_N_ifft.data_ptr()),
            static_cast<complex_bfloat16_t *>(twiddle_factors_ifft.data_ptr()),
            static_cast<at::BFloat16 *>(dx_out.data_ptr()),
            static_cast<complex_bfloat16_t *>(dk_f_out.data_ptr()),
            in_gate.has_value() ? static_cast<at::BFloat16 *>(in_gate.value().data_ptr()) : nullptr,
            out_gate.has_value() ? static_cast<at::BFloat16 *>(out_gate.value().data_ptr()) : nullptr,
            in_gate.has_value() ? static_cast<at::BFloat16 *>(din_gate.data_ptr()) : nullptr,
            out_gate.has_value() ? static_cast<at::BFloat16 *>(dout_gate.data_ptr()) : nullptr,
            B,
            H,
            N,
            sqrt_N);
      } else {
        gridDim.x = B;
        gridDim.y = H;

        blockDim.x = 32;
        blockDim.y = 1;

        dk_f_out = torch::empty({B, H, fftsize + 1, 2}, x.options());

        monarch_conv_bwd_cuda_kernel<32, 1, 256, 1, false, 1, 1><<<gridDim, blockDim, 8 * fftsize * sizeof(half)>>>(
            static_cast<at::BFloat16 *>(dout.data_ptr()),
            static_cast<at::BFloat16 *>(x.data_ptr()),
            static_cast<complex_bfloat16_t *>(k_f.data_ptr()),
            static_cast<complex_bfloat16_t *>(f_sqrt_N_fft.data_ptr()),
            static_cast<complex_bfloat16_t *>(twiddle_factors_fft.data_ptr()),
            static_cast<complex_bfloat16_t *>(twid_r2r.data_ptr()),
            static_cast<complex_bfloat16_t *>(f_sqrt_N_ifft.data_ptr()),
            static_cast<complex_bfloat16_t *>(twiddle_factors_ifft.data_ptr()),
            static_cast<at::BFloat16 *>(dx_out.data_ptr()),
            static_cast<complex_bfloat16_t *>(dk_f_out.data_ptr()),
            in_gate.has_value() ? static_cast<at::BFloat16 *>(in_gate.value().data_ptr()) : nullptr,
            out_gate.has_value() ? static_cast<at::BFloat16 *>(out_gate.value().data_ptr()) : nullptr,
            in_gate.has_value() ? static_cast<at::BFloat16 *>(din_gate.data_ptr()) : nullptr,
            out_gate.has_value() ? static_cast<at::BFloat16 *>(dout_gate.data_ptr()) : nullptr,
            B,
            H,
            N,
            sqrt_N);
      }
      break;
    case 1024:
      // if (true) {
      if (B >= 8 && (B % 8) == 0 && (H % 8) == 0) {
        gridDim.x = B / 8;
        gridDim.y = H / 8;
        // gridDim.x = B;
        // gridDim.y = H;

        dk_f_out = torch::empty({B / 8, H, fftsize + 1, 2}, x.options());

        blockDim.x = 32;
        blockDim.y = 1;
        monarch_conv_bwd_cuda_kernel<32, 1, 1024, 2, false, 8, 8><<<gridDim, blockDim, 8 * fftsize * sizeof(half)>>>(
            static_cast<at::BFloat16 *>(dout.data_ptr()),
            static_cast<at::BFloat16 *>(x.data_ptr()),
            static_cast<complex_bfloat16_t *>(k_f.data_ptr()),
            static_cast<complex_bfloat16_t *>(f_sqrt_N_fft.data_ptr()),
            static_cast<complex_bfloat16_t *>(twiddle_factors_fft.data_ptr()),
            static_cast<complex_bfloat16_t *>(twid_r2r.data_ptr()),
            static_cast<complex_bfloat16_t *>(f_sqrt_N_ifft.data_ptr()),
            static_cast<complex_bfloat16_t *>(twiddle_factors_ifft.data_ptr()),
            static_cast<at::BFloat16 *>(dx_out.data_ptr()),
            static_cast<complex_bfloat16_t *>(dk_f_out.data_ptr()),
            in_gate.has_value() ? static_cast<at::BFloat16 *>(in_gate.value().data_ptr()) : nullptr,
            out_gate.has_value() ? static_cast<at::BFloat16 *>(out_gate.value().data_ptr()) : nullptr,
            in_gate.has_value() ? static_cast<at::BFloat16 *>(din_gate.data_ptr()) : nullptr,
            out_gate.has_value() ? static_cast<at::BFloat16 *>(dout_gate.data_ptr()) : nullptr,
            B,
            H,
            N,
            sqrt_N);
      }
      else if (B >= 4 && (B % 4) == 0 && (H % 8) == 0) {
        gridDim.x = B / 4;
        gridDim.y = H / 8;

        dk_f_out = torch::empty({B / 4, H, fftsize + 1, 2}, x.options());

        blockDim.x = 32;
        blockDim.y = 1;
        monarch_conv_bwd_cuda_kernel<32, 1, 1024, 2, false, 4, 8><<<gridDim, blockDim, 8 * fftsize * sizeof(half)>>>(
            static_cast<at::BFloat16 *>(dout.data_ptr()),
            static_cast<at::BFloat16 *>(x.data_ptr()),
            static_cast<complex_bfloat16_t *>(k_f.data_ptr()),
            static_cast<complex_bfloat16_t *>(f_sqrt_N_fft.data_ptr()),
            static_cast<complex_bfloat16_t *>(twiddle_factors_fft.data_ptr()),
            static_cast<complex_bfloat16_t *>(twid_r2r.data_ptr()),
            static_cast<complex_bfloat16_t *>(f_sqrt_N_ifft.data_ptr()),
            static_cast<complex_bfloat16_t *>(twiddle_factors_ifft.data_ptr()),
            static_cast<at::BFloat16 *>(dx_out.data_ptr()),
            static_cast<complex_bfloat16_t *>(dk_f_out.data_ptr()),
            in_gate.has_value() ? static_cast<at::BFloat16 *>(in_gate.value().data_ptr()) : nullptr,
            out_gate.has_value() ? static_cast<at::BFloat16 *>(out_gate.value().data_ptr()) : nullptr,
            in_gate.has_value() ? static_cast<at::BFloat16 *>(din_gate.data_ptr()) : nullptr,
            out_gate.has_value() ? static_cast<at::BFloat16 *>(dout_gate.data_ptr()) : nullptr,
            B,
            H,
            N,
            sqrt_N);
      } else if ((H % 8) == 0) {
        gridDim.x = B;
        gridDim.y = H / 8;

        dk_f_out = torch::empty({B, H, fftsize + 1, 2}, x.options());

        blockDim.x = 32;
        blockDim.y = 1;
        monarch_conv_bwd_cuda_kernel<32, 1, 1024, 2, false, 1, 8><<<gridDim, blockDim, 8 * fftsize * sizeof(half)>>>(
            static_cast<at::BFloat16 *>(dout.data_ptr()),
            static_cast<at::BFloat16 *>(x.data_ptr()),
            static_cast<complex_bfloat16_t *>(k_f.data_ptr()),
            static_cast<complex_bfloat16_t *>(f_sqrt_N_fft.data_ptr()),
            static_cast<complex_bfloat16_t *>(twiddle_factors_fft.data_ptr()),
            static_cast<complex_bfloat16_t *>(twid_r2r.data_ptr()),
            static_cast<complex_bfloat16_t *>(f_sqrt_N_ifft.data_ptr()),
            static_cast<complex_bfloat16_t *>(twiddle_factors_ifft.data_ptr()),
            static_cast<at::BFloat16 *>(dx_out.data_ptr()),
            static_cast<complex_bfloat16_t *>(dk_f_out.data_ptr()),
            in_gate.has_value() ? static_cast<at::BFloat16 *>(in_gate.value().data_ptr()) : nullptr,
            out_gate.has_value() ? static_cast<at::BFloat16 *>(out_gate.value().data_ptr()) : nullptr,
            in_gate.has_value() ? static_cast<at::BFloat16 *>(din_gate.data_ptr()) : nullptr,
            out_gate.has_value() ? static_cast<at::BFloat16 *>(dout_gate.data_ptr()) : nullptr,
            B,
            H,
            N,
            sqrt_N);
      } else {
        gridDim.x = B;
        gridDim.y = H;

        dk_f_out = torch::empty({B, H, fftsize + 1, 2}, x.options());

        blockDim.x = 32;
        blockDim.y = 1;
        monarch_conv_bwd_cuda_kernel<32, 1, 1024, 2, false, 1, 1><<<gridDim, blockDim, 8 * fftsize * sizeof(half)>>>(
            static_cast<at::BFloat16 *>(dout.data_ptr()),
            static_cast<at::BFloat16 *>(x.data_ptr()),
            static_cast<complex_bfloat16_t *>(k_f.data_ptr()),
            static_cast<complex_bfloat16_t *>(f_sqrt_N_fft.data_ptr()),
            static_cast<complex_bfloat16_t *>(twiddle_factors_fft.data_ptr()),
            static_cast<complex_bfloat16_t *>(twid_r2r.data_ptr()),
            static_cast<complex_bfloat16_t *>(f_sqrt_N_ifft.data_ptr()),
            static_cast<complex_bfloat16_t *>(twiddle_factors_ifft.data_ptr()),
            static_cast<at::BFloat16 *>(dx_out.data_ptr()),
            static_cast<complex_bfloat16_t *>(dk_f_out.data_ptr()),
            in_gate.has_value() ? static_cast<at::BFloat16 *>(in_gate.value().data_ptr()) : nullptr,
            out_gate.has_value() ? static_cast<at::BFloat16 *>(out_gate.value().data_ptr()) : nullptr,
            in_gate.has_value() ? static_cast<at::BFloat16 *>(din_gate.data_ptr()) : nullptr,
            out_gate.has_value() ? static_cast<at::BFloat16 *>(dout_gate.data_ptr()) : nullptr,
            B,
            H,
            N,
            sqrt_N);
      }
      
      break;
     default:
        AT_ERROR("Monarch backward not implemented for this sequence length");
  }
  
  CHECK_LAST_CUDA_ERROR();
  if (in_gate.has_value() && out_gate.has_value()) {
    return {dx_out, dk_f_out.sum(0), din_gate, dout_gate};
  } else if (in_gate.has_value()) {
    return {dx_out, dk_f_out.sum(0), din_gate};
  } else if (out_gate.has_value()) {
    return {dx_out, dk_f_out.sum(0), dout_gate};
  } else{
    return {dx_out, dk_f_out.sum(0)};
  }
}