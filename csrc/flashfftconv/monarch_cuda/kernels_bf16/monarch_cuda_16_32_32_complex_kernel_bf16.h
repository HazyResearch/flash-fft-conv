// Copyright (c) 2023 Dan Fu, Hermann Kumbong

#include <torch/extension.h>

#include <vector>
#include <stdio.h>
#include <mma.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cub/block/block_load.cuh>
#include <cub/block/block_store.cuh>
#include "monarch_cuda_shared_bf16_no_float_shm.h"
using namespace nvcuda;

template <int BLOCK_DIM_X, int BLOCK_DIM_Y, int N, int MATMUL_WARP_WIDTH_1, int MATMUL_WARP_WIDTH_2, int DFT_SIZE, bool RECOMPUTE, int B_TILE_SIZE, int H_TILE_SIZE, int WARP_TILE_SIZE>
__global__ void monarch_conv_cuda_16_32_32_complex_kernel(
    const at::BFloat16 *__restrict__ a_real_inp,
    const at::BFloat16 *__restrict__ a_imag_inp,
    const c10::complex<at::BFloat16> *__restrict__ k_f,
    const c10::complex<at::BFloat16> *__restrict__ b_16,                        // 32 x 32
    const c10::complex<at::BFloat16> *__restrict__ b_32,                        // 16 x 16
    const c10::complex<at::BFloat16> *__restrict__ twiddle_factors_N_fft,  // 16K
    const c10::complex<at::BFloat16> *__restrict__ twiddle_factors_32_fft,   // 1024
    const c10::complex<at::BFloat16> *__restrict__ b_16_ifft,                   // 32 x 32
    const c10::complex<at::BFloat16> *__restrict__ b_32_ifft,                   // 16 x 16
    const c10::complex<at::BFloat16> *__restrict__ twiddle_factors_N_ifft, // 16K
    const c10::complex<at::BFloat16> *__restrict__ twiddle_factors_32_ifft,  // 1024
    at::BFloat16 *out_real,
    at::BFloat16 *out_imag,
    uint B,
    uint H,
    uint signal_size)
{

  const uint sqrt_N_1 = 16;
  const uint sqrt_N_2 = 32;
  const uint N_1 = 256;
  const uint N_2 = 1024;

  extern __shared__ at::Half a_real_fp16[];
  at::BFloat16 *a_real = reinterpret_cast<at::BFloat16 *>(&a_real_fp16[0]);
  at::BFloat16 *a_imag = &a_real[N];
  at::BFloat16 *b_real = &a_real[2 * N];
  at::BFloat16 *b_imag = &a_real[2 * N + N_2];
  at::BFloat16 *b_real_2 = &a_real[2 * N + 2 * N_2];
  at::BFloat16 *b_imag_2 = &a_real[2 * N + 3 * N_2];

  const int num_threads = BLOCK_DIM_X * BLOCK_DIM_Y;
  const int thread_id = threadIdx.x + blockDim.x * threadIdx.y;
  // const int thread_id = threadIdx.x;
  const int items_per_thread_input = N / num_threads;
  // this is for reading in the DFT matrix or twiddle factors
  const int items_per_thread_matrix_N_1 = num_threads <= 128 ? N_1 / num_threads : 2;
  const int items_per_thread_matrix_N_2 = N_2 / num_threads;
  const int warp_id = thread_id / WARP_SIZE;

  // NOTE - we are loading and storing data in a STRIPED FORMAT
  // SEQUENCE_SIZE * TILE_SIZE items, WARP_SIZE * TILE_SIZE threads -> items_per_thread_input
  using BlockLoad_Input = cub::BlockLoad<float, BLOCK_DIM_X, items_per_thread_input / 2, cub::BLOCK_LOAD_STRIPED, BLOCK_DIM_Y>;
  using BlockLoad_Sequence = cub::BlockLoad<c10::complex<float>, BLOCK_DIM_X, items_per_thread_input / 2, cub::BLOCK_LOAD_STRIPED, BLOCK_DIM_Y>;
  using BlockLoad_Matrix_N_1 = cub::BlockLoad<c10::complex<float>, BLOCK_DIM_X, items_per_thread_matrix_N_1 / 2, cub::BLOCK_LOAD_STRIPED, BLOCK_DIM_Y>; // for the DFT
  using BlockLoad_Matrix_N_2 = cub::BlockLoad<c10::complex<float>, BLOCK_DIM_X, items_per_thread_matrix_N_2 / 2, cub::BLOCK_LOAD_STRIPED, BLOCK_DIM_Y>; // for the DFT

  // index into block blockIdx.x
  int b_offset = blockIdx.x * H * N * B_TILE_SIZE;
  // index into the H
  int h_offset = blockIdx.y * N * H_TILE_SIZE;

  complex_bfloat16_t a_input_data[items_per_thread_input];    // for storing the input, also used for k_f
  complex_bfloat16_t b_input_data[items_per_thread_matrix_N_2];   // for storing matrices
  complex_bfloat16_t b_input_data_2[items_per_thread_matrix_N_2]; // another place for storing matrices

  // for the 16 x 16 dft
  wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_K, WMMA_N, __nv_bfloat16, wmma::row_major> b_frag_dft_N_1[MATMUL_WARP_WIDTH_1][MATMUL_WARP_WIDTH_1][2];
  // for the 16 x 16 idft
  wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_K, WMMA_N, __nv_bfloat16, wmma::row_major> b_frag_idft_N_1[MATMUL_WARP_WIDTH_1][MATMUL_WARP_WIDTH_1][2];

  // for the 32 x 32 dft
  wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_K, WMMA_N, __nv_bfloat16, wmma::row_major> b_frag_dft_N_2[MATMUL_WARP_WIDTH_2][MATMUL_WARP_WIDTH_2][2];
  // for the 32 x 32 idft
  wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_K, WMMA_N, __nv_bfloat16, wmma::row_major> b_frag_idft_N_2[MATMUL_WARP_WIDTH_2][MATMUL_WARP_WIDTH_2][2];
  // for the 32 x 32 dft
  wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_K, WMMA_N, __nv_bfloat16, wmma::col_major> a_frag_dft_N_2[MATMUL_WARP_WIDTH_2][MATMUL_WARP_WIDTH_2][2];
  
  // for 32 x 32 twiddles
  wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_K, WMMA_N, __nv_bfloat16, wmma::row_major> twiddle_32_dft_frag[MATMUL_WARP_WIDTH_2][MATMUL_WARP_WIDTH_2][2];
  // for 32 x 32 twiddles
  wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_K, WMMA_N, __nv_bfloat16, wmma::row_major> twiddle_32_idft_frag[MATMUL_WARP_WIDTH_2][MATMUL_WARP_WIDTH_2][2];

  // for the 16 x 1024 twiddle
  wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_K, WMMA_N, __nv_bfloat16, wmma::row_major> twiddle_1024_dft_frag[16 / WARP_TILE_SIZE][MATMUL_WARP_WIDTH_2][MATMUL_WARP_WIDTH_2][2];
  // for 16 x 1024 idft twiddle - split into 64 x (16 x 16)
  wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_K, WMMA_N, __nv_bfloat16, wmma::col_major> twiddle_1024_idft_frag[64 / WARP_TILE_SIZE][MATMUL_WARP_WIDTH_1][MATMUL_WARP_WIDTH_1][2];

  // accumulator fragments for the 16 x 16 and 32 x 32
  wmma::fragment<wmma::accumulator, WMMA_M, WMMA_K, WMMA_N, float> acc_frag_1[MATMUL_WARP_WIDTH_1][MATMUL_WARP_WIDTH_1][2];
  wmma::fragment<wmma::accumulator, WMMA_M, WMMA_K, WMMA_N, float> acc_frag_2[MATMUL_WARP_WIDTH_2][MATMUL_WARP_WIDTH_2][2];
  wmma::fragment<wmma::accumulator, WMMA_M, WMMA_K, WMMA_N, half> acc_frag_1_half[MATMUL_WARP_WIDTH_1][MATMUL_WARP_WIDTH_1][2];
  wmma::fragment<wmma::accumulator, WMMA_M, WMMA_K, WMMA_N, half> acc_frag_2_half[MATMUL_WARP_WIDTH_2][MATMUL_WARP_WIDTH_2][2];

  // for kernels - note that there are 16 / WARP_TILE_SIZE of these now!
  wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_K, WMMA_N, __nv_bfloat16, wmma::row_major> k_frag[16 / WARP_TILE_SIZE][MATMUL_WARP_WIDTH_2][MATMUL_WARP_WIDTH_2][2];

  // load twiddle_N_dft
  BlockLoad_Sequence().Load(
      reinterpret_cast<const c10::complex<float> *>(twiddle_factors_N_fft),
      reinterpret_cast<c10::complex<float>(&)[items_per_thread_input / 2]>(a_input_data));

  // loads b_16 into b
  BlockLoad_Matrix_N_1().Load(
      reinterpret_cast<const c10::complex<float> *>(b_16),
      reinterpret_cast<c10::complex<float>(&)[items_per_thread_matrix_N_1 / 2]>(b_input_data),
      N_1 / 2); // hopefully this interleaves things correctly

  // loads b_16_ifft into b
  BlockLoad_Matrix_N_1().Load(
      reinterpret_cast<const c10::complex<float> *>(b_16_ifft),
      reinterpret_cast<c10::complex<float>(&)[items_per_thread_matrix_N_1 / 2]>(b_input_data_2),
      N_1 / 2); // hopefully this interleaves things correctly

  int a_idx, b_idx;
  __nv_bfloat162 scratch;

  // load the 16x16 DFT matrix into b_real, b_imag
  // this costs about 60 us
  // #pragma unroll
  if (num_threads <= 128) {
    for (int i = 0; i < items_per_thread_matrix_N_1 / 2; i++)
    {
      b_idx = i * num_threads + thread_id;

      scratch = __nv_bfloat162(
        __nv_bfloat16(b_input_data[2 * i].real()),
        __nv_bfloat16(b_input_data[2 * i + 1].real())
      );
      reinterpret_cast<__nv_bfloat162 *>(b_real)[b_idx] = scratch;
      scratch = __nv_bfloat162(
        __nv_bfloat16(b_input_data[2 * i].imag()),
        __nv_bfloat16(b_input_data[2 * i + 1].imag())
      );
      reinterpret_cast<__nv_bfloat162 *>(b_imag)[b_idx] = scratch;

      scratch = __nv_bfloat162(
        __nv_bfloat16(b_input_data_2[2 * i].real()), 
        __nv_bfloat16(b_input_data_2[2 * i + 1].real())
      );
      reinterpret_cast<__nv_bfloat162 *>(b_real_2)[b_idx] = scratch;
      scratch = __nv_bfloat162(
        __nv_bfloat16(b_input_data_2[2 * i].imag()), 
        __nv_bfloat16(b_input_data_2[2 * i + 1].imag())
      );
      reinterpret_cast<__nv_bfloat162 *>(b_imag_2)[b_idx] = scratch;
    }
  } else {
    if (thread_id < 128) {
      b_idx = thread_id;

      scratch = __nv_bfloat162(
        __nv_bfloat16(b_input_data[0].real()),
        __nv_bfloat16(b_input_data[1].real())
      );
      reinterpret_cast<__nv_bfloat162 *>(b_real)[b_idx] = scratch;
      scratch = __nv_bfloat162(
        __nv_bfloat16(b_input_data[0].imag()),
        __nv_bfloat16(b_input_data[1].imag())
      );
      reinterpret_cast<__nv_bfloat162 *>(b_imag)[b_idx] = scratch;

      scratch = __nv_bfloat162(
        __nv_bfloat16(b_input_data_2[0].real()), 
        __nv_bfloat16(b_input_data_2[1].real())
      );
      reinterpret_cast<__nv_bfloat162 *>(b_real_2)[b_idx] = scratch;
      scratch = __nv_bfloat162(
        __nv_bfloat16(b_input_data_2[0].imag()), 
        __nv_bfloat16(b_input_data_2[1].imag())
      );
      reinterpret_cast<__nv_bfloat162 *>(b_imag_2)[b_idx] = scratch;
    }
  }

  // load N twiddle into shared memory
  // #pragma unroll
  for (int i = 0; i < items_per_thread_input / 2; i++)
  {
    a_idx = i * num_threads + thread_id;

    scratch = __nv_bfloat162(
      __nv_bfloat16(a_input_data[2 * i].real()),
      __nv_bfloat16(a_input_data[2 * i + 1].real())
    );
    reinterpret_cast<__nv_bfloat162 *>(a_real)[a_idx] = scratch;
    scratch = __nv_bfloat162(
      __nv_bfloat16(a_input_data[2 * i].imag()),
      __nv_bfloat16(a_input_data[2 * i + 1].imag())
    );
    reinterpret_cast<__nv_bfloat162 *>(a_imag)[a_idx] = scratch;
  }

  __syncthreads();

  // load in 32x32 twiddle factors
  // NOTE(danfu): this takes about 60 us
  BlockLoad_Matrix_N_2().Load(
      reinterpret_cast<const c10::complex<float> *>(twiddle_factors_32_fft),
      reinterpret_cast<c10::complex<float>(&)[items_per_thread_matrix_N_2 / 2]>(b_input_data),
      N_2 / 2);

  // start loading 32x32 ifft twiddle factors
  // TODO(danfu): this costs about 60 us
  BlockLoad_Matrix_N_2().Load(
      reinterpret_cast<const c10::complex<float> *>(twiddle_factors_32_ifft),
      reinterpret_cast<c10::complex<float>(&)[items_per_thread_matrix_N_2 / 2]>(b_input_data_2),
      N_2 / 2);

  bool a_trans = true;
  bool b_trans = false;

  // load 16x16 DFT matrix into b_frag_dft_N_1
  #pragma unroll
  for (int j_b = 0; j_b < MATMUL_WARP_WIDTH_1; j_b++)
  {
    // #pragma unroll
    for (int k = 0; k < MATMUL_WARP_WIDTH_1; k++)
    {
      b_idx = b_trans ? j_b * WMMA_N * sqrt_N_1 + k * WMMA_K : k * WMMA_K * sqrt_N_1 + j_b * WMMA_N;
      wmma::load_matrix_sync(b_frag_dft_N_1[k][j_b][0], reinterpret_cast<__nv_bfloat16 *>(b_real) + b_idx, sqrt_N_1);
      wmma::load_matrix_sync(b_frag_dft_N_1[k][j_b][1], reinterpret_cast<__nv_bfloat16 *>(b_imag) + b_idx, sqrt_N_1);
    }
  }

  // load 16x16 iDFT matrix into b_frag_idft_N_1
  // #pragma unroll
  for (int j_b = 0; j_b < MATMUL_WARP_WIDTH_1; j_b++)
  {
    // #pragma unroll
    for (int k = 0; k < MATMUL_WARP_WIDTH_1; k++)
    {
      b_idx = b_trans ? j_b * WMMA_N * sqrt_N_1 + k * WMMA_K : k * WMMA_K * sqrt_N_1 + j_b * WMMA_N;
      wmma::load_matrix_sync(b_frag_idft_N_1[k][j_b][0], reinterpret_cast<__nv_bfloat16 *>(b_real_2) + b_idx, sqrt_N_1);
      wmma::load_matrix_sync(b_frag_idft_N_1[k][j_b][1], reinterpret_cast<__nv_bfloat16 *>(b_imag_2) + b_idx, sqrt_N_1);
    }
  }

  // load N twiddle factors into registers
  // these will be loaded into the inner loop, so treat them as 16 x 1024
  for (int k_idx = 0; k_idx < 16 / WARP_TILE_SIZE; k_idx++)
  {
    int k_idx_offset = k_idx * WARP_TILE_SIZE * sqrt_N_2 * sqrt_N_2 + warp_id * sqrt_N_2 * sqrt_N_2;

    for (int j_b = 0; j_b < MATMUL_WARP_WIDTH_2; j_b++)
    {
      // #pragma unroll
      for (int k = 0; k < MATMUL_WARP_WIDTH_2; k++)
      {
        b_idx = k * WMMA_K * sqrt_N_2 + j_b * WMMA_N;
        wmma::load_matrix_sync(twiddle_1024_dft_frag[k_idx][k][j_b][0], reinterpret_cast<__nv_bfloat16 *>(a_real) + k_idx_offset + b_idx, sqrt_N_2);
        wmma::load_matrix_sync(twiddle_1024_dft_frag[k_idx][k][j_b][1], reinterpret_cast<__nv_bfloat16 *>(a_imag) + k_idx_offset + b_idx, sqrt_N_2);
      }
    }
  }

  __syncthreads();

  // load twiddle_N_idft
  BlockLoad_Sequence().Load(
      reinterpret_cast<const c10::complex<float> *>(twiddle_factors_N_ifft),
      reinterpret_cast<c10::complex<float>(&)[items_per_thread_input / 2]>(a_input_data));

  // load N ifft twiddle factors into shared memory
  // #pragma unroll
  for (int i = 0; i < items_per_thread_input / 2; i++)
  {
    a_idx = i * num_threads + thread_id;

    scratch = __nv_bfloat162(
      __nv_bfloat16(a_input_data[2 * i].real()),
      __nv_bfloat16(a_input_data[2 * i + 1].real())
    );
    reinterpret_cast<__nv_bfloat162 *>(a_real)[a_idx] = scratch;
    scratch = __nv_bfloat162(
      __nv_bfloat16(a_input_data[2 * i].imag()),
      __nv_bfloat16(a_input_data[2 * i + 1].imag())
    );
    reinterpret_cast<__nv_bfloat162 *>(a_imag)[a_idx] = scratch;
  }

  // load 32x32 twiddles into shared memory
  // load the DFT matrix into b_real, b_imag
  // this costs about 60 us
  // #pragma unroll
  for (int i = 0; i < items_per_thread_matrix_N_2 / 2; i++)
  {
    b_idx = i * num_threads + thread_id;

    scratch = __nv_bfloat162(
      __nv_bfloat16(b_input_data[2 * i].real()),
      __nv_bfloat16(b_input_data[2 * i + 1].real())
    );
    reinterpret_cast<__nv_bfloat162 *>(b_real)[b_idx] = scratch;
    scratch = __nv_bfloat162(
      __nv_bfloat16(b_input_data[2 * i].imag()),
      __nv_bfloat16(b_input_data[2 * i + 1].imag())
    );
    reinterpret_cast<__nv_bfloat162 *>(b_imag)[b_idx] = scratch;

    scratch = __nv_bfloat162(
      __nv_bfloat16(b_input_data_2[2 * i].real()), 
      __nv_bfloat16(b_input_data_2[2 * i + 1].real())
    );
    reinterpret_cast<__nv_bfloat162 *>(b_real_2)[b_idx] = scratch;
    scratch = __nv_bfloat162(
      __nv_bfloat16(b_input_data_2[2 * i].imag()), 
      __nv_bfloat16(b_input_data_2[2 * i + 1].imag())
    );
    reinterpret_cast<__nv_bfloat162 *>(b_imag_2)[b_idx] = scratch;
  }

  __syncthreads();

  // start loading 32x32 DFT matrices
  // NOTE(danfu): this takes about 60 us
  BlockLoad_Matrix_N_2().Load(
      reinterpret_cast<const c10::complex<float> *>(b_32),
      reinterpret_cast<c10::complex<float>(&)[items_per_thread_matrix_N_2 / 2]>(b_input_data),
      N_2 / 2);

  // start loading 32x32 iDFT matrices
  // TODO(danfu): this costs about 60 us
  BlockLoad_Matrix_N_2().Load(
      reinterpret_cast<const c10::complex<float> *>(b_32_ifft),
      reinterpret_cast<c10::complex<float>(&)[items_per_thread_matrix_N_2 / 2]>(b_input_data_2),
      N_2 / 2);

  // load N idft twiddle factors into registers
  // these will be used in the last iFFT, so treat them as 32 x 32 x 8
  for (int k_idx = 0; k_idx < 64 / WARP_TILE_SIZE; k_idx++)
  {
    int k_idx_offset = k_idx * WARP_TILE_SIZE * sqrt_N_1 + warp_id * sqrt_N_1;

    for (int j_b = 0; j_b < MATMUL_WARP_WIDTH_1; j_b++)
    {
      // #pragma unroll
      for (int k = 0; k < MATMUL_WARP_WIDTH_1; k++)
      {
        b_idx = j_b * WMMA_N * 1024 + k * WMMA_K;
        wmma::load_matrix_sync(twiddle_1024_idft_frag[k_idx][k][j_b][0], reinterpret_cast<__nv_bfloat16 *>(a_real) + k_idx_offset + b_idx, 1024);
        wmma::load_matrix_sync(twiddle_1024_idft_frag[k_idx][k][j_b][1], reinterpret_cast<__nv_bfloat16 *>(a_imag) + k_idx_offset + b_idx, 1024);
      }
    }
  }

  // load 32x32 DFT twiddles into twiddle_dft_frag
  // #pragma unroll
  for (int j_b = 0; j_b < MATMUL_WARP_WIDTH_2; j_b++)
  {
    // #pragma unroll
    for (int k = 0; k < MATMUL_WARP_WIDTH_2; k++)
    {
      b_idx = b_trans ? j_b * WMMA_N * sqrt_N_2 + k * WMMA_K : k * WMMA_K * sqrt_N_2 + j_b * WMMA_N;
      wmma::load_matrix_sync(twiddle_32_dft_frag[k][j_b][0], reinterpret_cast<__nv_bfloat16 *>(b_real) + b_idx, sqrt_N_2);
      wmma::load_matrix_sync(twiddle_32_dft_frag[k][j_b][1], reinterpret_cast<__nv_bfloat16 *>(b_imag) + b_idx, sqrt_N_2);
    }
  }

  // load iDFT twiddles into twiddle_idft_frag
  // #pragma unroll
  for (int j_b = 0; j_b < MATMUL_WARP_WIDTH_2; j_b++)
  {
    // #pragma unroll
    for (int k = 0; k < MATMUL_WARP_WIDTH_2; k++)
    {
      b_idx = b_trans ? j_b * WMMA_N * sqrt_N_2 + k * WMMA_K : k * WMMA_K * sqrt_N_2 + j_b * WMMA_N;
      wmma::load_matrix_sync(twiddle_32_idft_frag[k][j_b][0], reinterpret_cast<__nv_bfloat16 *>(b_real_2) + b_idx, sqrt_N_2);
      wmma::load_matrix_sync(twiddle_32_idft_frag[k][j_b][1], reinterpret_cast<__nv_bfloat16 *>(b_imag_2) + b_idx, sqrt_N_2);
    }
  }

  __syncthreads();

  // load the 32x32 DFT matrix into b_real, b_imag
  // this costs about 60 us
  // #pragma unroll
  for (int i = 0; i < items_per_thread_matrix_N_2 / 2; i++)
  {
    b_idx = i * num_threads + thread_id;

    scratch = __nv_bfloat162(
      __nv_bfloat16(b_input_data[2 * i].real()),
      __nv_bfloat16(b_input_data[2 * i + 1].real())
    );
    reinterpret_cast<__nv_bfloat162 *>(b_real)[b_idx] = scratch;
    scratch = __nv_bfloat162(
      __nv_bfloat16(b_input_data[2 * i].imag()),
      __nv_bfloat16(b_input_data[2 * i + 1].imag())
    );
    reinterpret_cast<__nv_bfloat162 *>(b_imag)[b_idx] = scratch;

    scratch = __nv_bfloat162(
      __nv_bfloat16(b_input_data_2[2 * i].real()), 
      __nv_bfloat16(b_input_data_2[2 * i + 1].real())
    );
    reinterpret_cast<__nv_bfloat162 *>(b_real_2)[b_idx] = scratch;
    scratch = __nv_bfloat162(
      __nv_bfloat16(b_input_data_2[2 * i].imag()), 
      __nv_bfloat16(b_input_data_2[2 * i + 1].imag())
    );
    reinterpret_cast<__nv_bfloat162 *>(b_imag_2)[b_idx] = scratch;
  }

  __syncthreads();

  // load the 32x32 DFT matrices into b_frag_dft_N_2, b_frag_idft_N_2
  #pragma unroll
  for (int j_b = 0; j_b < MATMUL_WARP_WIDTH_2; j_b++)
  {
    // #pragma unroll
    for (int k = 0; k < MATMUL_WARP_WIDTH_2; k++)
    {
      a_idx = a_trans ? j_b * WMMA_N * sqrt_N_2 + k * WMMA_K : k * WMMA_K * sqrt_N_2 + j_b * WMMA_N;
      b_idx = b_trans ? j_b * WMMA_N * sqrt_N_2 + k * WMMA_K : k * WMMA_K * sqrt_N_2 + j_b * WMMA_N;
      wmma::load_matrix_sync(a_frag_dft_N_2[k][j_b][0], reinterpret_cast<__nv_bfloat16 *>(b_real) + a_idx, sqrt_N_2);
      wmma::load_matrix_sync(b_frag_dft_N_2[k][j_b][0], reinterpret_cast<__nv_bfloat16 *>(b_real) + b_idx, sqrt_N_2);
      wmma::load_matrix_sync(a_frag_dft_N_2[k][j_b][1], reinterpret_cast<__nv_bfloat16 *>(b_imag) + a_idx, sqrt_N_2);
      wmma::load_matrix_sync(b_frag_dft_N_2[k][j_b][1], reinterpret_cast<__nv_bfloat16 *>(b_imag) + b_idx, sqrt_N_2);
    }
  }

  // #pragma unroll
  for (int j_b = 0; j_b < MATMUL_WARP_WIDTH_2; j_b++)
  {
    // #pragma unroll
    for (int k = 0; k < MATMUL_WARP_WIDTH_2; k++)
    {
      b_idx = b_trans ? j_b * WMMA_N * sqrt_N_2 + k * WMMA_K : k * WMMA_K * sqrt_N_2 + j_b * WMMA_N;
      wmma::load_matrix_sync(b_frag_idft_N_2[k][j_b][0], reinterpret_cast<__nv_bfloat16 *>(b_real_2) + b_idx, sqrt_N_2);
      wmma::load_matrix_sync(b_frag_idft_N_2[k][j_b][1], reinterpret_cast<__nv_bfloat16 *>(b_imag_2) + b_idx, sqrt_N_2);
    }
  }

  // #pragma unroll
  for (int h_tile_id = 0; h_tile_id < H_TILE_SIZE; h_tile_id++)
  {

    // start loading k_f
    // NOTE(danfu): this load from HBM costs about 60 us
    BlockLoad_Sequence().Load(
        reinterpret_cast<const c10::complex<float> *>(k_f + h_offset + h_tile_id * N),
        reinterpret_cast<c10::complex<float>(&)[items_per_thread_input / 2]>(a_input_data));

    // load k_f into shared memory
    // #pragma unroll
    for (int i = 0; i < items_per_thread_input / 2; i++)
    {
      a_idx = i * num_threads + thread_id;

      scratch = __nv_bfloat162(
      __nv_bfloat16(a_input_data[2 * i].real()),
      __nv_bfloat16(a_input_data[2 * i + 1].real())
      );
      reinterpret_cast<__nv_bfloat162 *>(a_real)[a_idx] = scratch;
      scratch = __nv_bfloat162(
        __nv_bfloat16(a_input_data[2 * i].imag()),
        __nv_bfloat16(a_input_data[2 * i + 1].imag())
      );
      reinterpret_cast<__nv_bfloat162 *>(a_imag)[a_idx] = scratch;
    }

    __syncthreads();

    // load k_f into registers in k_frag
    // in the inner loop, so treat as 16 x 1024
    for (int k_idx = 0; k_idx < 16 / WARP_TILE_SIZE; k_idx++)
    {
      // #pragma unroll
      for (int j_a = 0; j_a < MATMUL_WARP_WIDTH_2; j_a++)
      {
        // #pragma unroll
        for (int k = 0; k < MATMUL_WARP_WIDTH_2; k++)
        {
          // a_idx = j_a * WMMA_K * sqrt_N + k * WMMA_K + k_idx * DFT_SIZE * DFT_SIZE + warp_id * (16 / WARP_TILE_SIZE) * DFT_SIZE * DFT_SIZE;
          a_idx = j_a * WMMA_K * sqrt_N_2 +
                  k * WMMA_K +
                  k_idx * WARP_TILE_SIZE * sqrt_N_2 * sqrt_N_2 +
                  warp_id * sqrt_N_2 * sqrt_N_2;
          wmma::load_matrix_sync(k_frag[k_idx][j_a][k][0], reinterpret_cast<__nv_bfloat16 *>(a_real + a_idx), sqrt_N_2);
          wmma::load_matrix_sync(k_frag[k_idx][j_a][k][1], reinterpret_cast<__nv_bfloat16 *>(a_imag + a_idx), sqrt_N_2);
        }
      }
    }

    __syncthreads();

    // #pragma unroll
    for (int b_tile_id = 0; b_tile_id < B_TILE_SIZE; b_tile_id++)
    {

      int input_offset = h_offset + b_offset + h_tile_id * N + b_tile_id * H * N;

      int k_idx_offset;

      // // load input into a_real
      // BlockLoad_Input().Load(
      //   reinterpret_cast<const float *>(a + input_offset),
      //   reinterpret_cast<float(&)[items_per_thread_input / 2]>(x_input_data),
      //   signal_size / 2, 0.
      // );

      // for (int i = 0; i < items_per_thread_input / 2; i++)
      // {
      //   a_idx = i * num_threads + thread_id;

      //   reinterpret_cast<__nv_bfloat162 *>(a_real)[a_idx] = __nv_bfloat162(
      //     __nv_bfloat16(x_input_data[2 * i]),
      //     __nv_bfloat16(x_input_data[2 * i + 1])
      //   );
      // }

      // if (threadIdx.x == 0 && threadIdx.y == 0 && blockIdx.x == 0 && blockIdx.y == 0) {
      //   printf("x_input_data\n");
      //   for (int i = 0; i < items_per_thread_input / 2; i++) {
      //     a_idx = i * num_threads + thread_id;
      //     printf("%f, ", __bfloat162float(__nv_bfloat16(x_input_data[2 * i])));
      //   }
      //   printf("\n");
      // }

      // __syncthreads();

      // if (threadIdx.x == 0 && threadIdx.y == 0 && blockIdx.x == 0 && blockIdx.y == 0) {
      //   printf("Before first DFT\n");
      //   for (int i = 0; i < items_per_thread_input; i++) {
      //     a_idx = i * num_threads + thread_id;
      //     printf("%f, ", __bfloat162float(__nv_bfloat16(a_real[a_idx])));
      //   }
      //   printf("\n");

      //   // printf("Before first DFT\n");
      //   // for (int i = 0; i < 32; i++) {
      //   //   a_idx = i;
      //   //   printf("%f, ", __bfloat162float(__nv_bfloat16(a_real[a_idx])));
      //   // }
      //   // printf("\n");
      // }
      __syncthreads();

      // 1024 / 16 = 64
      for (int k_idx = 0; k_idx < 64 / WARP_TILE_SIZE; k_idx++)
      {
        // k_idx_offset = k_idx * DFT_SIZE + warp_id * (16 / WARP_TILE_SIZE) * DFT_SIZE;
        k_idx_offset = k_idx * WARP_TILE_SIZE * sqrt_N_1 + warp_id * sqrt_N_1;
        // outer DFT
        complex_matmul_c2c_1024<wmma::col_major, wmma::row_major, true, true, MATMUL_WARP_WIDTH_1, false, true>(
            reinterpret_cast<const __nv_bfloat16 *>(a_real_inp + input_offset + k_idx_offset),                 // this is the input
            reinterpret_cast<const __nv_bfloat16 *>(a_imag_inp + input_offset + k_idx_offset),                 // this is the input
            reinterpret_cast<__nv_bfloat16 *>(a_real + k_idx_offset),                 // this is the output
            reinterpret_cast<__nv_bfloat16 *>(a_imag + k_idx_offset),                 // this is the output
            sqrt_N_1,
            N,
            b_frag_dft_N_1,
            acc_frag_1,
            acc_frag_1_half,
            wmma::mem_col_major);
      }
      __syncthreads();

      // if (threadIdx.x == 0 && threadIdx.y == 0 && blockIdx.x == 0 && blockIdx.y == 0) {
      //   printf("After first DFT\n");
      //   for (int i = 0; i < items_per_thread_input; i++) {
      //     a_idx = i * num_threads + thread_id;
      //     printf("%f + %fi, ", __bfloat162float(__nv_bfloat16(a_real[a_idx])), __bfloat162float(__nv_bfloat16(a_imag[a_idx])));
      //   }
      //   printf("\n");
      // }

      // 16 times (32, 32)
      for (int k_idx = 0; k_idx < 16 / WARP_TILE_SIZE; k_idx++)
      {
        // k_idx_offset = k_idx * DFT_SIZE * DFT_SIZE + warp_id * (16 / WARP_TILE_SIZE) * DFT_SIZE * DFT_SIZE;
        k_idx_offset = k_idx * WARP_TILE_SIZE * sqrt_N_2 * sqrt_N_2 + warp_id * sqrt_N_2 * sqrt_N_2;

        // if (threadIdx.x == 0 && threadIdx.y == 0 && blockIdx.x == 0 && blockIdx.y == 0 && k_idx < 2) {
        //   printf("k_idx %d, k_idx_offset %d\n", k_idx, k_idx_offset);
        // }

        // first DFT, output is NOT written to shared memory
        complex_matmul_load_b<wmma::col_major, wmma::row_major, false, false, MATMUL_WARP_WIDTH_2, false, false>(
            reinterpret_cast<__nv_bfloat16 *>(a_real + k_idx_offset), // this is the output
            reinterpret_cast<__nv_bfloat16 *>(a_imag + k_idx_offset), // this is the output
            sqrt_N_2,
            N,
            a_frag_dft_N_2,
            acc_frag_2,
            acc_frag_2_half,
            twiddle_1024_dft_frag[k_idx],
            wmma::mem_row_major);

        // if (threadIdx.x == 0 && threadIdx.y == 0 && blockIdx.x == 0 && blockIdx.y == 0 && k_idx < 2) {
        //   printf("After first DFT in the conv, %d\n", k_idx);
        //   for (int i = 0; i < 32; i++) {
        //     a_idx = i * num_threads + thread_id + k_idx_offset;
        //     printf("%f + %fi, ", __nv_bfloat162float(a_real[a_idx]), __nv_bfloat162float(a_imag[a_idx]));
        //   }
        //   printf("\n");
        // }

        // __syncthreads();

        // second DFT, output is NOT written to a_real, a_imag
        complex_matmul<wmma::row_major, wmma::row_major, false, false, MATMUL_WARP_WIDTH_2, true, false>(
            reinterpret_cast<__nv_bfloat16 *>(a_real + k_idx_offset),
            reinterpret_cast<__nv_bfloat16 *>(a_imag + k_idx_offset),
            sqrt_N_2,
            N,
            b_frag_dft_N_2,
            acc_frag_2,
            acc_frag_2_half,
            twiddle_32_dft_frag,
            wmma::mem_row_major);

        // if (threadIdx.x == 0 && threadIdx.y == 0 && blockIdx.x == 0 && blockIdx.y == 0 && k_idx < 2) {
        //   printf("After second DFT in the conv, %d\n", k_idx);
        //   for (int i = 0; i < 8; i++) {
        //     a_idx = i * num_threads + thread_id + k_idx_offset;
        //     printf("%f + %fi, ", __nv_bfloat162float(a_real[a_idx]), __nv_bfloat162float(a_imag[a_idx]));
        //   }
        //   printf("\n");
        // }

        // __syncthreads();

        // load the input from acc_frag_1, and multiply by k_frag
        complex_matmul<wmma::row_major, wmma::row_major, false, true, MATMUL_WARP_WIDTH_2, true, true>(
            reinterpret_cast<__nv_bfloat16 *>(a_real + k_idx_offset),
            reinterpret_cast<__nv_bfloat16 *>(a_imag + k_idx_offset),
            sqrt_N_2,
            N,
            b_frag_idft_N_2,
            acc_frag_2,
            acc_frag_2_half,
            k_frag[k_idx],
            wmma::mem_col_major);

        // if (threadIdx.x == 0 && threadIdx.y == 0 && blockIdx.x == 0 && blockIdx.y == 0 && k_idx < 2) {
        //   printf("After iDFT in the conv, %d\n", k_idx);
        //   for (int i = 0; i < 8; i++) {
        //     a_idx = i * num_threads + thread_id + k_idx_offset;
        //     printf("%f + %fi, ", __nv_bfloat162float(a_real[a_idx]), __nv_bfloat162float(a_imag[a_idx]));
        //   }
        //   printf("\n");
        // }

        // __syncthreads();

        complex_matmul<wmma::row_major, wmma::row_major, false, true, MATMUL_WARP_WIDTH_2, false, true>(
            reinterpret_cast<__nv_bfloat16 *>(a_real + k_idx_offset),
            reinterpret_cast<__nv_bfloat16 *>(a_imag + k_idx_offset),
            // reinterpret_cast<__nv_bfloat16 *>(out + input_offset + k_idx_offset),
            sqrt_N_2,
            N,
            b_frag_idft_N_2,
            acc_frag_2,
            acc_frag_2_half,
            twiddle_32_idft_frag,
            wmma::mem_col_major);

        // if (threadIdx.x == 0 && threadIdx.y == 0 && blockIdx.x == 0 && blockIdx.y == 0 && k_idx < 2) {
        //   printf("After 2nd iDFT in the conv, %d\n", k_idx);
        //   for (int i = 0; i < 8; i++) {
        //     a_idx = i * num_threads + thread_id + k_idx_offset;
        //     printf("%f + %fi, ", __nv_bfloat162float(a_real[a_idx]), __nv_bfloat162float(a_imag[a_idx]));
        //   }
        //   printf("\n");
        // }

        // __syncthreads();
      }

      __syncthreads();

      // if (threadIdx.x == 0 && threadIdx.y == 0 && blockIdx.x == 0 && blockIdx.y == 0) {
      //   printf("After inner conv\n");
      //   for (int i = 0; i < items_per_thread_input; i++) {
      //     a_idx = i * num_threads + thread_id;
      //     printf("%f + %fi, ", __bfloat162float(__nv_bfloat16(a_real[a_idx])), __bfloat162float(__nv_bfloat16(a_imag[a_idx])));
      //   }
      //   printf("\n");
      // }

      // 1024 / 16 = 64
      for (int k_idx = 0; k_idx < 64 / WARP_TILE_SIZE; k_idx++)
      {
        // k_idx_offset = k_idx * DFT_SIZE + warp_id * (16 / WARP_TILE_SIZE) * DFT_SIZE;
        k_idx_offset = k_idx * WARP_TILE_SIZE * sqrt_N_1 + warp_id * sqrt_N_1;
        // outer DFT
        complex_matmul_c2c_1024<wmma::col_major, wmma::row_major, true, true, MATMUL_WARP_WIDTH_1, false, true>(
            reinterpret_cast<__nv_bfloat16 *>(a_real + k_idx_offset), // this is the input
            reinterpret_cast<__nv_bfloat16 *>(a_imag + k_idx_offset), // this is the input
            reinterpret_cast<__nv_bfloat16 *>(out_real + input_offset + k_idx_offset), // this is the output
            reinterpret_cast<__nv_bfloat16 *>(out_imag + input_offset + k_idx_offset), // this is the output
            sqrt_N_1,
            N,
            b_frag_idft_N_1,
            acc_frag_1,
            acc_frag_1_half,
            twiddle_1024_idft_frag[k_idx],
            wmma::mem_col_major);
      }
      __syncthreads();

      // if (threadIdx.x == 0 && threadIdx.y == 0 && blockIdx.x == 0 && blockIdx.y == 0) {
      //    printf("Before output\n");
      //    for (int i = 0; i < items_per_thread_input; i++) {
      //       a_idx = i * num_threads + thread_id;
      //       printf("%f, ", __bfloat162float(__nv_bfloat16(a_real[a_idx])));
      //    }
      //    printf("\n");
      // }

      // #pragma unroll
      // for (int i = 0; i < items_per_thread_input / 2; i++)
      // {
      //   a_idx = i * num_threads + thread_id;
      //   scratch = reinterpret_cast<__nv_bfloat162 *>(a_real)[a_idx];

      //   x_input_data[2 * i] = scratch.x;
      //   x_input_data[2 * i + 1] = scratch.y;
      // }

      // // HACK
      // // for now, just output the a_real output
      // BlockStore_Sequence().Store(
      //     reinterpret_cast<float *>(out + input_offset),
      //     reinterpret_cast<float(&)[items_per_thread_input / 2]>(x_input_data),
      //     signal_size / 2
      // );

      // __syncthreads();
    } // b_tile_id
  }   // h_tile_id
}
