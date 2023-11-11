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

template <int BLOCK_DIM_X, int BLOCK_DIM_Y, int N, int MATMUL_WARP_WIDTH, int DFT_SIZE, bool RECOMPUTE, int B_TILE_SIZE, int H_TILE_SIZE, int WARP_TILE_SIZE>
__global__ void monarch_conv_bwd_cuda_kernel(
    const at::BFloat16 *__restrict__ dout,
    const at::BFloat16 *__restrict__ a,
    const c10::complex<at::BFloat16> *__restrict__ k_f,
    const c10::complex<at::BFloat16> *__restrict__ b,                        // 16 x 16
    const c10::complex<at::BFloat16> *__restrict__ twiddle_factors_256_fft,  // 4096
    const c10::complex<at::BFloat16> *__restrict__ twiddle_factors_16_fft,   // 256
    const c10::complex<at::BFloat16> *__restrict__ b_ifft,                   // 16 x 16
    const c10::complex<at::BFloat16> *__restrict__ twiddle_factors_256_ifft, // 4096
    const c10::complex<at::BFloat16> *__restrict__ twiddle_factors_16_ifft,  // 256
    at::BFloat16 *dx_out,
    c10::complex<at::BFloat16> *dk_f_out,
    const at::BFloat16 *__restrict__ in_gate,
    const at::BFloat16 *__restrict__ out_gate,
    at::BFloat16 *din_gate,
    at::BFloat16 *dout_gate,
    uint B,
    uint H,
    uint signal_size,
    uint sqrt_N)
{

  extern __shared__ at::Half a_real_fp16[];
  at::BFloat16 *a_real = reinterpret_cast<at::BFloat16 *>(&a_real_fp16[0]);
  at::BFloat16 *a_imag = &a_real[N];
  at::BFloat16 *a_real_2 = &a_real[2 * N];
  at::BFloat16 *a_imag_2 = &a_real[3 * N];
  at::BFloat16 *b_real = &a_real[4 * N];
  at::BFloat16 *b_imag = &a_real[4 * N + 256];
  at::BFloat16 *b_real_2 = &a_real[4 * N + 2 * 256];
  at::BFloat16 *b_imag_2 = &a_real[4 * N + 3 * 256];

  const int num_threads = BLOCK_DIM_X * BLOCK_DIM_Y;
  const int thread_id = threadIdx.x + blockDim.x * threadIdx.y;
  // const int thread_id = threadIdx.x;
  const int items_per_thread_input = N / num_threads;
  // this is for reading in the DFT matrix or twiddle factors
  const int items_per_thread_matrix = num_threads <= 128 ? DFT_SIZE * DFT_SIZE / num_threads : 2;
  const int warp_id = thread_id / WARP_SIZE;

  // NOTE - we are loading and storing data in a STRIPED FORMAT
  // SEQUENCE_SIZE * TILE_SIZE items, WARP_SIZE * TILE_SIZE threads -> items_per_thread_input
  using BlockLoad_Input = cub::BlockLoad<float, BLOCK_DIM_X, items_per_thread_input / 2, cub::BLOCK_LOAD_STRIPED, BLOCK_DIM_Y>;
  using BlockLoad_Sequence = cub::BlockLoad<c10::complex<float>, BLOCK_DIM_X, items_per_thread_input / 2, cub::BLOCK_LOAD_STRIPED, BLOCK_DIM_Y>;
  using BlockLoad_Matrix = cub::BlockLoad<c10::complex<float>, BLOCK_DIM_X, items_per_thread_matrix / 2, cub::BLOCK_LOAD_STRIPED, BLOCK_DIM_Y>; // for the DFT / Twiddle, etc
  using BlockStore_Sequence = cub::BlockStore<float, BLOCK_DIM_X, items_per_thread_input / 2, cub::BLOCK_STORE_STRIPED, BLOCK_DIM_Y>;
  using BlockStore_Sequence_Complex = cub::BlockStore<c10::complex<float>, BLOCK_DIM_X, items_per_thread_input / 2, cub::BLOCK_STORE_STRIPED, BLOCK_DIM_Y>;

  // index into block blockIdx.x
  int b_offset = blockIdx.x * H * signal_size * B_TILE_SIZE;
  // index into the H
  int h_offset_signal = blockIdx.y * signal_size * H_TILE_SIZE;
  int h_offset_kernel = blockIdx.y * N * H_TILE_SIZE;

  complex_bfloat16_t a_input_data[items_per_thread_input];    // for storing the input, also used for k_f
  at::BFloat16 x_input_data[items_per_thread_input];     // for storing the input
  at::BFloat16 gate_data[items_per_thread_input];    // for storing the input gates
  at::BFloat16 dgate_data[items_per_thread_input];
  at::BFloat16 dout_data[items_per_thread_input];
  complex_bfloat16_t temp[items_per_thread_input]; 
  complex_bfloat16_t b_input_data[items_per_thread_matrix];   // for storing matrices, twiddle factors
  complex_bfloat16_t b_input_data_2[items_per_thread_matrix]; // another place for storing matrices, twiddle factors

  // for the dft
  wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_K, WMMA_N, __nv_bfloat16, wmma::row_major> b_frag_dft[MATMUL_WARP_WIDTH][MATMUL_WARP_WIDTH][2];
  // for the idft
  wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_K, WMMA_N, __nv_bfloat16, wmma::row_major> b_frag_idft[MATMUL_WARP_WIDTH][MATMUL_WARP_WIDTH][2];
  // for the dft
  wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_K, WMMA_N, __nv_bfloat16, wmma::col_major> a_frag_dft[MATMUL_WARP_WIDTH][MATMUL_WARP_WIDTH][2];
  // for twiddles
  wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_K, WMMA_N, __nv_bfloat16, wmma::row_major> twiddle_16_dft_frag[MATMUL_WARP_WIDTH][MATMUL_WARP_WIDTH][2];
  // for twiddles
  wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_K, WMMA_N, __nv_bfloat16, wmma::row_major> twiddle_16_idft_frag[MATMUL_WARP_WIDTH][MATMUL_WARP_WIDTH][2];

  // for 256 twiddle
  wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_K, WMMA_N, __nv_bfloat16, wmma::row_major> twiddle_256_dft_frag[16 / WARP_TILE_SIZE][MATMUL_WARP_WIDTH][MATMUL_WARP_WIDTH][2];
  // for 256 idft twiddle
  wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_K, WMMA_N, __nv_bfloat16, wmma::col_major> twiddle_256_idft_frag[16 / WARP_TILE_SIZE][MATMUL_WARP_WIDTH][MATMUL_WARP_WIDTH][2];

  // // for twiddles
  // wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_K, WMMA_N, half, wmma::col_major> twiddle_256_dft_frag[N / (DFT_SIZE * DFT_SIZE)][MATMUL_WARP_WIDTH][MATMUL_WARP_WIDTH][2];

  // for kernels - note that there are 16 / WARP_TILE_SIZE of these now!
  wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_K, WMMA_N, __nv_bfloat16, wmma::row_major> k_frag[16 / WARP_TILE_SIZE][MATMUL_WARP_WIDTH][MATMUL_WARP_WIDTH][2];

  // load twiddle_256_dft
  BlockLoad_Sequence().Load(
      reinterpret_cast<const c10::complex<float> *>(twiddle_factors_256_fft),
      reinterpret_cast<c10::complex<float>(&)[items_per_thread_input / 2]>(a_input_data));

  // loads SEQUENCE_SIZE into b
  BlockLoad_Matrix().Load(
      reinterpret_cast<const c10::complex<float> *>(b),
      reinterpret_cast<c10::complex<float>(&)[items_per_thread_matrix / 2]>(b_input_data),
      DFT_SIZE * DFT_SIZE / 2); // hopefully this interleaves things correctly

  // loads SEQUENCE_SIZE into b
  BlockLoad_Matrix().Load(
      reinterpret_cast<const c10::complex<float> *>(b_ifft),
      reinterpret_cast<c10::complex<float>(&)[items_per_thread_matrix / 2]>(b_input_data_2),
      DFT_SIZE * DFT_SIZE / 2); // hopefully this interleaves things correctly

  int a_idx, b_idx;
  __nv_bfloat162 scratch;

  // load the DFT matrix into b_real, b_imag
  // this costs about 60 us
  // #pragma unroll
  if (num_threads <= 128) {
    for (int i = 0; i < items_per_thread_matrix / 2; i++)
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

  // load 256 twiddle into shared memory
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

  // load into twiddle factors
  // NOTE(danfu): this takes about 60 us
  BlockLoad_Matrix().Load(
      reinterpret_cast<const c10::complex<float> *>(twiddle_factors_16_fft),
      reinterpret_cast<c10::complex<float>(&)[items_per_thread_matrix / 2]>(b_input_data),
      DFT_SIZE * DFT_SIZE / 2);

  // start loading ifft twiddle factors
  // TODO(danfu): this costs about 60 us
  BlockLoad_Matrix().Load(
      reinterpret_cast<const c10::complex<float> *>(twiddle_factors_16_ifft),
      reinterpret_cast<c10::complex<float>(&)[items_per_thread_matrix / 2]>(b_input_data_2),
      DFT_SIZE * DFT_SIZE / 2);

  bool a_trans = true;
  bool b_trans = false;
  wmma::fragment<wmma::accumulator, WMMA_M, WMMA_K, WMMA_N, float> acc_frag_1[MATMUL_WARP_WIDTH][MATMUL_WARP_WIDTH][2];
  wmma::fragment<wmma::accumulator, WMMA_M, WMMA_K, WMMA_N, half> acc_frag_half[MATMUL_WARP_WIDTH][MATMUL_WARP_WIDTH][2];

// load DFT matrix into b_frag
#pragma unroll
  for (int j_b = 0; j_b < MATMUL_WARP_WIDTH; j_b++)
  {
    // #pragma unroll
    for (int k = 0; k < MATMUL_WARP_WIDTH; k++)
    {
      a_idx = a_trans ? j_b * WMMA_N * sqrt_N + k * WMMA_K : k * WMMA_K * sqrt_N + j_b * WMMA_N;
      b_idx = b_trans ? j_b * WMMA_N * sqrt_N + k * WMMA_K : k * WMMA_K * sqrt_N + j_b * WMMA_N;
      wmma::load_matrix_sync(a_frag_dft[k][j_b][0], reinterpret_cast<__nv_bfloat16 *>(b_real) + a_idx, sqrt_N);
      wmma::load_matrix_sync(b_frag_dft[k][j_b][0], reinterpret_cast<__nv_bfloat16 *>(b_real) + b_idx, sqrt_N);
      wmma::load_matrix_sync(a_frag_dft[k][j_b][1], reinterpret_cast<__nv_bfloat16 *>(b_imag) + a_idx, sqrt_N);
      wmma::load_matrix_sync(b_frag_dft[k][j_b][1], reinterpret_cast<__nv_bfloat16 *>(b_imag) + b_idx, sqrt_N);
    }
  }

  // load iDFT matrix into b_frag_idft
  // #pragma unroll
  for (int j_b = 0; j_b < MATMUL_WARP_WIDTH; j_b++)
  {
    // #pragma unroll
    for (int k = 0; k < MATMUL_WARP_WIDTH; k++)
    {
      b_idx = b_trans ? j_b * WMMA_N * sqrt_N + k * WMMA_K : k * WMMA_K * sqrt_N + j_b * WMMA_N;
      wmma::load_matrix_sync(b_frag_idft[k][j_b][0], reinterpret_cast<__nv_bfloat16 *>(b_real_2) + b_idx, sqrt_N);
      wmma::load_matrix_sync(b_frag_idft[k][j_b][1], reinterpret_cast<__nv_bfloat16 *>(b_imag_2) + b_idx, sqrt_N);
    }
  }

  // load 256 twiddle factors into registers
  for (int k_idx = 0; k_idx < 16 / WARP_TILE_SIZE; k_idx++)
  {
    int k_idx_offset = k_idx * WARP_TILE_SIZE * DFT_SIZE * DFT_SIZE + warp_id * DFT_SIZE * DFT_SIZE;

    for (int j_b = 0; j_b < MATMUL_WARP_WIDTH; j_b++)
    {
      // #pragma unroll
      for (int k = 0; k < MATMUL_WARP_WIDTH; k++)
      {
        b_idx = k * WMMA_K * sqrt_N + j_b * WMMA_N;
        wmma::load_matrix_sync(twiddle_256_dft_frag[k_idx][k][j_b][0], reinterpret_cast<__nv_bfloat16 *>(a_real) + k_idx_offset + b_idx, sqrt_N);
        wmma::load_matrix_sync(twiddle_256_dft_frag[k_idx][k][j_b][1], reinterpret_cast<__nv_bfloat16 *>(a_imag) + k_idx_offset + b_idx, sqrt_N);
      }
    }
  }

  __syncthreads();

  // load twiddle_256_idft
  BlockLoad_Sequence().Load(
      reinterpret_cast<const c10::complex<float> *>(twiddle_factors_256_ifft),
      reinterpret_cast<c10::complex<float>(&)[items_per_thread_input / 2]>(a_input_data));

  // load 256 ifft twiddle factors into shared memory
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

  // load twiddles into shared memory
  // load the DFT matrix into b_real, b_imag
  // this costs about 60 us
  // #pragma unroll
  if (num_threads <= 128) {
    for (int i = 0; i < items_per_thread_matrix / 2; i++)
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

  __syncthreads();

  // load 256 idft twiddle factors into registers
  for (int k_idx = 0; k_idx < 16 / WARP_TILE_SIZE; k_idx++)
  {
    int k_idx_offset = k_idx * WARP_TILE_SIZE * DFT_SIZE + warp_id * DFT_SIZE;

    for (int j_b = 0; j_b < MATMUL_WARP_WIDTH; j_b++)
    {
      // #pragma unroll
      for (int k = 0; k < MATMUL_WARP_WIDTH; k++)
      {
        b_idx = j_b * WMMA_N * sqrt_N + k * WMMA_K;
        wmma::load_matrix_sync(twiddle_256_idft_frag[k_idx][k][j_b][0], reinterpret_cast<__nv_bfloat16 *>(a_real) + k_idx_offset + b_idx, 256);
        wmma::load_matrix_sync(twiddle_256_idft_frag[k_idx][k][j_b][1], reinterpret_cast<__nv_bfloat16 *>(a_imag) + k_idx_offset + b_idx, 256);
      }
    }
  }

  // load DFT twiddles into twiddle_dft_frag
  // #pragma unroll
  for (int j_b = 0; j_b < MATMUL_WARP_WIDTH; j_b++)
  {
    // #pragma unroll
    for (int k = 0; k < MATMUL_WARP_WIDTH; k++)
    {
      b_idx = b_trans ? j_b * WMMA_N * sqrt_N + k * WMMA_K : k * WMMA_K * sqrt_N + j_b * WMMA_N;
      wmma::load_matrix_sync(twiddle_16_dft_frag[k][j_b][0], reinterpret_cast<__nv_bfloat16 *>(b_real) + b_idx, sqrt_N);
      wmma::load_matrix_sync(twiddle_16_dft_frag[k][j_b][1], reinterpret_cast<__nv_bfloat16 *>(b_imag) + b_idx, sqrt_N);
    }
  }

  // load iDFT twiddles into twiddle_idft_frag
  // #pragma unroll
  for (int j_b = 0; j_b < MATMUL_WARP_WIDTH; j_b++)
  {
    // #pragma unroll
    for (int k = 0; k < MATMUL_WARP_WIDTH; k++)
    {
      b_idx = b_trans ? j_b * WMMA_N * sqrt_N + k * WMMA_K : k * WMMA_K * sqrt_N + j_b * WMMA_N;
      wmma::load_matrix_sync(twiddle_16_idft_frag[k][j_b][0], reinterpret_cast<__nv_bfloat16 *>(b_real_2) + b_idx, sqrt_N);
      wmma::load_matrix_sync(twiddle_16_idft_frag[k][j_b][1], reinterpret_cast<__nv_bfloat16 *>(b_imag_2) + b_idx, sqrt_N);
    }
  }

  __syncthreads();

  // #pragma unroll
  for (int h_tile_id = 0; h_tile_id < H_TILE_SIZE; h_tile_id++)
  {

    // start loading k_f
    // NOTE(danfu): this load from HBM costs about 60 us
    BlockLoad_Sequence().Load(
        reinterpret_cast<const c10::complex<float> *>(k_f + h_offset_kernel + h_tile_id * N),
        reinterpret_cast<c10::complex<float>(&)[items_per_thread_input / 2]>(a_input_data));

    // load k_f.conj() into shared memory
    // #pragma unroll
    for (int i = 0; i < items_per_thread_input / 2; i++)
    {
      a_idx = i * num_threads + thread_id;

      scratch = __nv_bfloat162(
        __nv_bfloat16(a_input_data[2 * i].real()),
        __nv_bfloat16(a_input_data[2 * i + 1].real())
      );
      reinterpret_cast<__nv_bfloat162 *>(a_real)[a_idx] = scratch;

      scratch = __hneg2(__nv_bfloat162(
        __nv_bfloat16(a_input_data[2 * i].imag()),
        __nv_bfloat16(a_input_data[2 * i + 1].imag())
      ));
      reinterpret_cast<__nv_bfloat162 *>(a_imag)[a_idx] = scratch;
    }

    __syncthreads();

    // load k_f.conj() into registers in k_frag
    for (int k_idx = 0; k_idx < 16 / WARP_TILE_SIZE; k_idx++)
    {
      // #pragma unroll
      for (int j_a = 0; j_a < MATMUL_WARP_WIDTH; j_a++)
      {
        // #pragma unroll
        for (int k = 0; k < MATMUL_WARP_WIDTH; k++)
        {
          // a_idx = j_a * WMMA_K * sqrt_N + k * WMMA_K + k_idx * DFT_SIZE * DFT_SIZE + warp_id * (16 / WARP_TILE_SIZE) * DFT_SIZE * DFT_SIZE;
          a_idx = j_a * WMMA_K * sqrt_N +
                  k * WMMA_K +
                  k_idx * WARP_TILE_SIZE * DFT_SIZE * DFT_SIZE +
                  warp_id * DFT_SIZE * DFT_SIZE;
          wmma::load_matrix_sync(k_frag[k_idx][j_a][k][0], reinterpret_cast<__nv_bfloat16 *>(a_real + a_idx), sqrt_N);
          wmma::load_matrix_sync(k_frag[k_idx][j_a][k][1], reinterpret_cast<__nv_bfloat16 *>(a_imag + a_idx), sqrt_N);
        }
      }
    }

    __syncthreads();

    for(int i = 0; i < items_per_thread_input; i++) {
        temp[i] = complex_bfloat16_t(0.0f, 0.0f);
    }
    // #pragma unroll
    for (int b_tile_id = 0; b_tile_id < B_TILE_SIZE; b_tile_id++)
    {

      int input_offset = h_offset_signal + b_offset + h_tile_id * signal_size + b_tile_id * H * signal_size;

      int k_idx_offset;

      // load dout into a_real
      BlockLoad_Input().Load(
        reinterpret_cast<const float *>(dout + input_offset),
        reinterpret_cast<float(&)[items_per_thread_input / 2]>(x_input_data),
        signal_size / 2, 0.
      );

      if(out_gate != nullptr){
        // load output gate into gate_data
        BlockLoad_Input().Load(
          reinterpret_cast<const float *>(out_gate + input_offset),
          reinterpret_cast<float(&)[items_per_thread_input / 2]>(gate_data),
          signal_size / 2, 0.
        );
      }

      for (int i = 0; i < items_per_thread_input / 2; i++)
      {
        a_idx = i * num_threads + thread_id;

        reinterpret_cast<__nv_bfloat162 *>(dout_data)[i] = reinterpret_cast<__nv_bfloat162 *>(x_input_data)[i];

        if(out_gate != nullptr){
          reinterpret_cast<__nv_bfloat162 *>(a_real)[a_idx] = __hmul2(
            reinterpret_cast<__nv_bfloat162 *>(x_input_data)[i],
            reinterpret_cast<__nv_bfloat162 *>(gate_data)[i]
          );
        }else{
          reinterpret_cast<__nv_bfloat162 *>(a_real)[a_idx] = reinterpret_cast<__nv_bfloat162 *>(x_input_data)[i];
        }
      }

      __syncthreads();

      // load input into a_real
      BlockLoad_Input().Load(
        reinterpret_cast<const float *>(a + input_offset),
        reinterpret_cast<float(&)[items_per_thread_input / 2]>(x_input_data),
        signal_size / 2, 0.
      );

      if(in_gate != nullptr){
        // load input gate into gate_data
        BlockLoad_Input().Load(
          reinterpret_cast<const float *>(in_gate + input_offset),
          reinterpret_cast<float(&)[items_per_thread_input / 2]>(gate_data),
          signal_size / 2, 0.
        );
      }

      for (int i = 0; i < items_per_thread_input / 2; i++)
      {
        a_idx = i * num_threads + thread_id;

        if(in_gate != nullptr){
          reinterpret_cast<__nv_bfloat162 *>(a_real_2)[a_idx] = __hmul2(
            reinterpret_cast<__nv_bfloat162 *>(x_input_data)[i],
            reinterpret_cast<__nv_bfloat162 *>(gate_data)[i]
          );
        }else{
          reinterpret_cast<__nv_bfloat162 *>(a_real_2)[a_idx] = reinterpret_cast<__nv_bfloat162 *>(x_input_data)[i];
        }
      }

      __syncthreads();

      for (int k_idx = 0; k_idx < 16 / WARP_TILE_SIZE; k_idx++)
      {
        // k_idx_offset = k_idx * DFT_SIZE + warp_id * (16 / WARP_TILE_SIZE) * DFT_SIZE;
        k_idx_offset = k_idx * WARP_TILE_SIZE * DFT_SIZE + warp_id * DFT_SIZE;
        // outer DFT(dout)
        complex_matmul_r2c_256<wmma::col_major, wmma::row_major, true, true, MATMUL_WARP_WIDTH, false, true>(
            reinterpret_cast<__nv_bfloat16 *>(a_real + k_idx_offset), // read from SRAM
            reinterpret_cast<__nv_bfloat16 *>(a_real + k_idx_offset),                    // this is the output
            reinterpret_cast<__nv_bfloat16 *>(a_imag + k_idx_offset),                    // this is the output
            sqrt_N,
            N,
            b_frag_dft,
            acc_frag_1,
            acc_frag_half,
            wmma::mem_col_major);
        // outer DFT(x)
        complex_matmul_r2c_256<wmma::col_major, wmma::row_major, true, true, MATMUL_WARP_WIDTH, false, true>(
            reinterpret_cast<__nv_bfloat16 *>(a_real_2 + k_idx_offset), // read from SRAM
            reinterpret_cast<__nv_bfloat16 *>(a_real_2 + k_idx_offset),               // this is the output
            reinterpret_cast<__nv_bfloat16 *>(a_imag_2 + k_idx_offset),               // this is the output
            sqrt_N,
            N,
            b_frag_dft,
            acc_frag_1,
            acc_frag_half,
            wmma::mem_col_major);
      }
      __syncthreads();

      // if (threadIdx.x == 0 && threadIdx.y == 0 && blockIdx.x == 0 && blockIdx.y == 0) {
      //    printf("dout @ f_sqrt_N_fft\n");
      //    for (int i = 0; i < items_per_thread_input; i++) {
      //       a_idx = i * num_threads + thread_id;
      //       printf("%f + %fi, ", __half2float(a_real[a_idx]), __half2float(a_imag[a_idx]));
      //    }
      //    printf("\n");
      //    printf("x @ f_sqrt_N_fft\n");
      //    for (int i = 0; i < items_per_thread_input; i++) {
      //       a_idx = i * num_threads + thread_id;
      //       printf("%f + %fi, ", __half2float(a_real_2[a_idx]), __half2float(a_imag_2[a_idx]));
      //    }
      //    printf("\n");
      // }

      for (int k_idx = 0; k_idx < 16 / WARP_TILE_SIZE; k_idx++)
      {
        // k_idx_offset = k_idx * DFT_SIZE * DFT_SIZE + warp_id * (16 / WARP_TILE_SIZE) * DFT_SIZE * DFT_SIZE;
        k_idx_offset = k_idx * WARP_TILE_SIZE * DFT_SIZE * DFT_SIZE + warp_id * DFT_SIZE * DFT_SIZE;

        // first DFT, output is NOT written to shared memory
        // DFT(dout)
        complex_matmul_load_b<wmma::col_major, wmma::row_major, false, false, MATMUL_WARP_WIDTH, false, false>(
            reinterpret_cast<__nv_bfloat16 *>(a_real + k_idx_offset), // this is the output
            reinterpret_cast<__nv_bfloat16 *>(a_imag + k_idx_offset), // this is the output
            sqrt_N,
            N,
            a_frag_dft,
            acc_frag_1,
            acc_frag_half,
            twiddle_256_dft_frag[k_idx],
            wmma::mem_row_major);

        // __syncthreads();

        // second DFT, output IS written to a_real, a_imag
        // DFT(dout)
        complex_matmul<wmma::row_major, wmma::row_major, false, false, MATMUL_WARP_WIDTH, true, true>(
            reinterpret_cast<__nv_bfloat16 *>(a_real + k_idx_offset),
            reinterpret_cast<__nv_bfloat16 *>(a_imag + k_idx_offset),
            sqrt_N,
            N,
            b_frag_dft,
            acc_frag_1,
            acc_frag_half,
            twiddle_16_dft_frag,
            wmma::mem_row_major);

        // first DFT, output is NOT written to shared memory
        // DFT(x)
        complex_matmul_load_b<wmma::col_major, wmma::row_major, false, false, MATMUL_WARP_WIDTH, false, false>(
            reinterpret_cast<__nv_bfloat16 *>(a_real_2 + k_idx_offset), // this is the output
            reinterpret_cast<__nv_bfloat16 *>(a_imag_2 + k_idx_offset), // this is the output
            sqrt_N,
            N,
            a_frag_dft,
            acc_frag_1,
            acc_frag_half,
            twiddle_256_dft_frag[k_idx],
            wmma::mem_row_major);

        // __syncthreads();

        // second DFT, output IS written to a_real, a_imag
        // DFT(x)
        complex_matmul<wmma::row_major, wmma::row_major, false, false, MATMUL_WARP_WIDTH, true, true>(
            reinterpret_cast<__nv_bfloat16 *>(a_real_2 + k_idx_offset),
            reinterpret_cast<__nv_bfloat16 *>(a_imag_2 + k_idx_offset),
            sqrt_N,
            N,
            b_frag_dft,
            acc_frag_1,
            acc_frag_half,
            twiddle_16_dft_frag,
            wmma::mem_row_major);

        // if (threadIdx.x == 0 && threadIdx.y == 0 && blockIdx.x == 0 && blockIdx.y == 0 && k_idx == 15) {
        //    printf("DFT(dout)\n");
        //    for (int i = 0; i < items_per_thread_input; i++) {
        //       a_idx = i * num_threads + thread_id;
        //       printf("%f + %fi, ", __half2float(a_real[a_idx]), __half2float(a_imag[a_idx]));
        //    }
        //    printf("\n");
        //    printf("DFT(x)\n");
        //    for (int i = 0; i < items_per_thread_input; i++) {
        //       a_idx = i * num_threads + thread_id;
        //       printf("%f + %fi, ", __half2float(a_real_2[a_idx]), __half2float(a_imag_2[a_idx]));
        //    }
        //    printf("\n");
        // }

        // // x = x * N
        // for (int i = 0; i < 256 / 32 / 2; i++)
        // {
        //   a_idx = k_idx_offset / 2 + i * 32 + thread_id % 32;
        //   reinterpret_cast<__half2 *>(a_real_2)[a_idx] = __hmul2(
        //       reinterpret_cast<__half2 *>(a_real_2)[a_idx],
        //       __half2(__float2half(float(N)), __float2half(float(N))));
        //   reinterpret_cast<__half2 *>(a_imag_2)[a_idx] = __hmul2(
        //       reinterpret_cast<__half2 *>(a_imag_2)[a_idx],
        //       __half2(__float2half(float(N)), __float2half(float(N))));
        // }

        // dk_f = dout * x.conj()
        for (int i = 0; i < 256 / 32 / 2; i++)
        {
          a_idx = k_idx_offset / 2 + i * 32 + thread_id % 32;
          complex_mul_conj_bfloat162(
              reinterpret_cast<__nv_bfloat162 *>(a_real)[a_idx],
              reinterpret_cast<__nv_bfloat162 *>(a_imag)[a_idx],
              reinterpret_cast<__nv_bfloat162 *>(a_real_2)[a_idx],
              reinterpret_cast<__nv_bfloat162 *>(a_imag_2)[a_idx],
              &reinterpret_cast<__nv_bfloat162 *>(a_real_2)[a_idx],
              &reinterpret_cast<__nv_bfloat162 *>(a_imag_2)[a_idx]);
        }

        __syncthreads();

        // start computing iFFT(dout)
        // load the input from acc_frag_1, and multiply by k_frag
        complex_matmul<wmma::row_major, wmma::row_major, false, true, MATMUL_WARP_WIDTH, false, true>(
            reinterpret_cast<__nv_bfloat16 *>(a_real + k_idx_offset),
            reinterpret_cast<__nv_bfloat16 *>(a_imag + k_idx_offset),
            sqrt_N,
            N,
            b_frag_idft,
            acc_frag_1,
            acc_frag_half,
            k_frag[k_idx],
            wmma::mem_col_major);

        // if (threadIdx.x == 0 && threadIdx.y == 0 && blockIdx.x == 0 && blockIdx.y == 0) {
        //    printf("After ifft\n");
        //    for (int i = 0; i < items_per_thread_input; i++) {
        //       a_idx = i * num_threads + thread_id;
        //       printf("%f + %fi, ", scratch_real[a_idx], scratch_imag[a_idx]);
        //    }
        //    printf("\n");
        // }

        // __syncthreads();

        // second iFFT dout, and multiply by twiddle
        complex_matmul<wmma::row_major, wmma::row_major, false, true, MATMUL_WARP_WIDTH, false, true>(
            reinterpret_cast<__nv_bfloat16 *>(a_real + k_idx_offset),
            reinterpret_cast<__nv_bfloat16 *>(a_imag + k_idx_offset),
            // reinterpret_cast<half *>(out + input_offset + k_idx_offset),
            sqrt_N,
            N,
            b_frag_idft,
            acc_frag_1,
            acc_frag_half,
            twiddle_16_idft_frag,
            wmma::mem_col_major);

        // __syncthreads();
      }

      __syncthreads();

      // finish iFFT dout
      for (int k_idx = 0; k_idx < 16 / WARP_TILE_SIZE; k_idx++)
      {
        // k_idx_offset = k_idx * DFT_SIZE + warp_id * (16 / WARP_TILE_SIZE) * DFT_SIZE;
        k_idx_offset = k_idx * WARP_TILE_SIZE * DFT_SIZE + warp_id * DFT_SIZE;
        // outer DFT
        complex_matmul_c2r_256<wmma::col_major, wmma::row_major, true, true, MATMUL_WARP_WIDTH, false, true>(
            reinterpret_cast<__nv_bfloat16 *>(a_real + k_idx_offset), // this is the input
            reinterpret_cast<__nv_bfloat16 *>(a_imag + k_idx_offset), // this is the input
            reinterpret_cast<__nv_bfloat16 *>(a_real + k_idx_offset), // write to SRAM
            sqrt_N,
            N,
            b_frag_idft,
            acc_frag_1,
            acc_frag_half,
            twiddle_256_idft_frag[k_idx],
            wmma::mem_col_major);
      }
      __syncthreads();

      // if (threadIdx.x == 0 && threadIdx.y == 0 && blockIdx.x == 0 && blockIdx.y == 0) {
      //    printf("Before output\n");
      //    for (int i = 0; i < items_per_thread_input; i++) {
      //       a_idx = i * num_threads + thread_id;
      //       printf("%f, ", __half2float(a_real[a_idx]));
      //    }
      //    printf("\n");
      // }

      // load input into a_real
      BlockLoad_Input().Load(
        reinterpret_cast<const float *>(a + input_offset),
        reinterpret_cast<float(&)[items_per_thread_input / 2]>(x_input_data),
        signal_size / 2, 0.
      );

      if(in_gate != nullptr){
        for (int i = 0; i < items_per_thread_input / 2; i++)
        {
            a_idx = i * num_threads + thread_id;

            reinterpret_cast<__nv_bfloat162 *>(dgate_data)[i] = __hmul2(
              reinterpret_cast<__nv_bfloat162 *>(a_real)[a_idx],
              reinterpret_cast<__nv_bfloat162 *>(x_input_data)[i]
            );
        }

        // write to HBM
        BlockStore_Sequence().Store(
          reinterpret_cast<float *>(din_gate + input_offset),
          reinterpret_cast<float(&)[items_per_thread_input / 2]>(dgate_data),
          signal_size / 2
        );
      }

      // multiply dout by N, and prepare for writing to HBM
      for (int i = 0; i < items_per_thread_input / 2; i++)
      {
        a_idx = i * num_threads + thread_id;
        // reinterpret_cast<__half2 *>(a_input_data)[i] = __hmul2(
        //     reinterpret_cast<__half2 *>(a_real)[a_idx],
        //     __half2(__float2half(float(N)), __float2half(float(N))));
        if(in_gate != nullptr){
          reinterpret_cast<__nv_bfloat162 *>(a_input_data)[i] = __hmul2(
            reinterpret_cast<__nv_bfloat162 *>(a_real)[a_idx],
            reinterpret_cast<__nv_bfloat162 *>(gate_data)[i]
          );
        }else{
          reinterpret_cast<__nv_bfloat162 *>(a_input_data)[i] = reinterpret_cast<__nv_bfloat162 *>(a_real)[a_idx];
        }
      }

      // HACK
      // for now, just output the a_real output
      BlockStore_Sequence().Store(
          reinterpret_cast<float *>(dx_out + input_offset),
          reinterpret_cast<float(&)[items_per_thread_input / 2]>(a_input_data),
          signal_size / 2
      );

      __syncthreads();

      // put dk_f into a_input_data, and write to HBM
      __nv_bfloat162 real, imag;

#pragma unroll
      for (int i = 0; i < items_per_thread_input / 2; i++)
      {
        a_idx = i * num_threads + thread_id;
        real = reinterpret_cast<__nv_bfloat162 *>(a_real_2)[a_idx];
        imag = reinterpret_cast<__nv_bfloat162 *>(a_imag_2)[a_idx];
        reinterpret_cast<c10::complex<__nv_bfloat16> *>(a_input_data)[2 * i] = c10::complex<__nv_bfloat16>(real.x, imag.x);
        reinterpret_cast<c10::complex<__nv_bfloat16> *>(a_input_data)[2 * i + 1] = c10::complex<__nv_bfloat16>(real.y, imag.y);
      }

      __syncthreads();

      for(int i = 0; i < items_per_thread_input; i++) {
          temp[i] += a_input_data[i];
      }

    __syncthreads();

    } // b_tile_id

    for(int i = 0; i < items_per_thread_input; i++) {
        reinterpret_cast<__nv_bfloat162 *>(temp)[i] = __hmul2(reinterpret_cast<__nv_bfloat162 *>(temp)[i], __nv_bfloat162(__float2bfloat16(float(N)), __float2bfloat16(float(N))));
    }

    // store dk_f
      BlockStore_Sequence_Complex().Store(
          reinterpret_cast<c10::complex<float> *>(dk_f_out + h_offset_kernel + blockIdx.x * H * N + h_tile_id * N),
          reinterpret_cast<c10::complex<float>(&)[items_per_thread_input / 2]>(temp));
  }   // h_tile_id
}
