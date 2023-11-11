// Copyright (c) 2023 Dan Fu, Hermann Kumbong

#include <torch/extension.h>

#include <vector>
#include <stdio.h>
#include <mma.h>
#include <cuda_fp16.h>
using namespace nvcuda;

using complex_half_t = typename c10::complex<at::Half>;

#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16
// #define TILE_SIZE 4
// #define SHMEM_SIZE 256 * TILE_SIZE
// #define SEQUENCE_SIZE 256
#define WARP_SIZE 32

#ifndef MONARCH_CUDA_MATMULS_
#define MONARCH_CUDA_MATMULS_

template <typename ALayout, typename BLayout, bool out_trans, int MATMUL_WARP_WIDTH, bool output_to_shmem>
__device__ __forceinline__ void _complex_matmul(
  half *a_real,
  half *a_imag,
  int sqrt_N,
  int N,
  wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_K, WMMA_N, half, ALayout> a_frag[MATMUL_WARP_WIDTH][MATMUL_WARP_WIDTH][2],
  wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_K, WMMA_N, half, BLayout> b_frag[MATMUL_WARP_WIDTH][MATMUL_WARP_WIDTH][2],
  wmma::fragment<wmma::accumulator, WMMA_M, WMMA_K, WMMA_N, half> acc_frag_1[MATMUL_WARP_WIDTH][MATMUL_WARP_WIDTH][2],
  wmma::layout_t out_layout = wmma::mem_row_major)
{
  // #pragma unroll
  for (int j_b = 0; j_b < MATMUL_WARP_WIDTH; j_b++) {
    // #pragma unroll
    for (int j_a = 0; j_a < MATMUL_WARP_WIDTH; j_a++) {
      wmma::fill_fragment(acc_frag_1[j_a][j_b][0], __float2half(0.0f));

      // real
      // bd
      // #pragma unroll
      for (int k = 0; k < MATMUL_WARP_WIDTH; k++) {
        wmma::mma_sync(acc_frag_1[j_a][j_b][0], a_frag[j_a][k][1], b_frag[k][j_b][1], acc_frag_1[j_a][j_b][0]);
      }

      // bd -> -bd
      // #pragma unroll
      for (int i = 0; i < acc_frag_1[j_a][j_b][0].num_elements; i++) {
        acc_frag_1[j_a][j_b][0].x[i] = __hneg(acc_frag_1[j_a][j_b][0].x[i]);
      }

      // ac
      // #pragma unroll
      for (int k = 0; k < MATMUL_WARP_WIDTH; k++) {
        wmma::mma_sync(acc_frag_1[j_a][j_b][0], a_frag[j_a][k][0], b_frag[k][j_b][0], acc_frag_1[j_a][j_b][0]);
      }

      wmma::fill_fragment(acc_frag_1[j_a][j_b][1], __float2half(0.0f));

      // imag
      // ad
      // #pragma unroll
      for (int k = 0; k < MATMUL_WARP_WIDTH; k++) {
        wmma::mma_sync(acc_frag_1[j_a][j_b][1], a_frag[j_a][k][0], b_frag[k][j_b][1], acc_frag_1[j_a][j_b][1]);
      }

      // bc
      // #pragma unroll
      for (int k = 0; k < MATMUL_WARP_WIDTH; k++) {
        wmma::mma_sync(acc_frag_1[j_a][j_b][1], a_frag[j_a][k][1], b_frag[k][j_b][0], acc_frag_1[j_a][j_b][1]);
      }

    }
  }

  if (output_to_shmem) {
    // #pragma unroll
    for (int j_b = 0; j_b < MATMUL_WARP_WIDTH; j_b++) {
      // #pragma unroll
      for (int j_a = 0; j_a < MATMUL_WARP_WIDTH; j_a++) {
        // does it matter where we put this?
        wmma::store_matrix_sync(
          a_real + (out_trans ?
          j_b * WMMA_M * sqrt_N + j_a * WMMA_N:
          j_a * WMMA_M * sqrt_N + j_b * WMMA_N),
          acc_frag_1[j_a][j_b][0], sqrt_N, out_layout
        );
  
        wmma::store_matrix_sync(
          a_imag + (out_trans ?
          j_b * WMMA_M * sqrt_N + j_a * WMMA_N:
          j_a * WMMA_M * sqrt_N + j_b * WMMA_N),
          acc_frag_1[j_a][j_b][1], sqrt_N, out_layout
        );
      }
    }
  }
}

template <typename ALayout, typename BLayout, bool out_trans, int MATMUL_WARP_WIDTH, bool output_to_shmem>
__device__ __forceinline__ void _complex_matmul_r2c(
  half *a_real,
  half *a_imag,
  int sqrt_N,
  int N,
  wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_K, WMMA_N, half, ALayout> a_frag[MATMUL_WARP_WIDTH][MATMUL_WARP_WIDTH][2],
  wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_K, WMMA_N, half, BLayout> b_frag[MATMUL_WARP_WIDTH][MATMUL_WARP_WIDTH][2],
  wmma::fragment<wmma::accumulator, WMMA_M, WMMA_K, WMMA_N, half> acc_frag_1[MATMUL_WARP_WIDTH][MATMUL_WARP_WIDTH][2],
  wmma::layout_t out_layout = wmma::mem_row_major)
{
  // #pragma unroll
  for (int j_b = 0; j_b < MATMUL_WARP_WIDTH; j_b++) {
    // #pragma unroll
    for (int j_a = 0; j_a < MATMUL_WARP_WIDTH; j_a++) {
      wmma::fill_fragment(acc_frag_1[j_a][j_b][0], __float2half(0.0f));

      // real

      // ac
      for (int k = 0; k < MATMUL_WARP_WIDTH; k++) {
        wmma::mma_sync(acc_frag_1[j_a][j_b][0], a_frag[j_a][k][0], b_frag[k][j_b][0], acc_frag_1[j_a][j_b][0]);
      }

      wmma::fill_fragment(acc_frag_1[j_a][j_b][1], __float2half(0.0f));

      // imag
      // ad
      for (int k = 0; k < MATMUL_WARP_WIDTH; k++) {
        wmma::mma_sync(acc_frag_1[j_a][j_b][1], a_frag[j_a][k][0], b_frag[k][j_b][1], acc_frag_1[j_a][j_b][1]);
      }

    }
  }

  if (output_to_shmem) {
    // #pragma unroll
    for (int j_b = 0; j_b < MATMUL_WARP_WIDTH; j_b++) {
      // #pragma unroll
      for (int j_a = 0; j_a < MATMUL_WARP_WIDTH; j_a++) {
        // does it matter where we put this?
        wmma::store_matrix_sync(
          a_real + (out_trans ?
          j_b * WMMA_M * sqrt_N + j_a * WMMA_N:
          j_a * WMMA_M * sqrt_N + j_b * WMMA_N),
          acc_frag_1[j_a][j_b][0], sqrt_N, out_layout
        );
  
        wmma::store_matrix_sync(
          a_imag + (out_trans ?
          j_b * WMMA_M * sqrt_N + j_a * WMMA_N:
          j_a * WMMA_M * sqrt_N + j_b * WMMA_N),
          acc_frag_1[j_a][j_b][1], sqrt_N, out_layout
        );
      }
    }
  }
}

template <typename ALayout, typename BLayout, bool out_trans, int MATMUL_WARP_WIDTH, bool output_to_shmem>
__device__ __forceinline__ void _complex_matmul_r2c_load_b(
  half *b_real,
  half *b_imag,
  int sqrt_N,
  int N,
  wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_K, WMMA_N, half, ALayout> a_frag[MATMUL_WARP_WIDTH][MATMUL_WARP_WIDTH][2],
  wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_K, WMMA_N, half, BLayout> b_frag[MATMUL_WARP_WIDTH][MATMUL_WARP_WIDTH][2],
  wmma::fragment<wmma::accumulator, WMMA_M, WMMA_K, WMMA_N, half> acc_frag_1[MATMUL_WARP_WIDTH][MATMUL_WARP_WIDTH][2],
  wmma::layout_t out_layout = wmma::mem_row_major)
{
  // #pragma unroll
  for (int j_b = 0; j_b < MATMUL_WARP_WIDTH; j_b++) {
    // #pragma unroll
    for (int j_a = 0; j_a < MATMUL_WARP_WIDTH; j_a++) {
      wmma::fill_fragment(acc_frag_1[j_a][j_b][0], __float2half(0.0f));

      // real
      // ac
      // #pragma unroll
      for (int k = 0; k < MATMUL_WARP_WIDTH; k++) {
        wmma::mma_sync(acc_frag_1[j_a][j_b][0], a_frag[j_a][k][0], b_frag[k][j_b][0], acc_frag_1[j_a][j_b][0]);
      }

      wmma::fill_fragment(acc_frag_1[j_a][j_b][1], __float2half(0.0f));

      // imag
      // bc
      // #pragma unroll
      for (int k = 0; k < MATMUL_WARP_WIDTH; k++) {
        wmma::mma_sync(acc_frag_1[j_a][j_b][1], a_frag[j_a][k][1], b_frag[k][j_b][0], acc_frag_1[j_a][j_b][1]);
      }

    }
  }

  if (output_to_shmem) {
    // #pragma unroll
    for (int j_b = 0; j_b < MATMUL_WARP_WIDTH; j_b++) {
      // #pragma unroll
      for (int j_a = 0; j_a < MATMUL_WARP_WIDTH; j_a++) {
        // does it matter where we put this?
        wmma::store_matrix_sync(
          b_real + (out_trans ?
          j_b * WMMA_M * sqrt_N + j_a * WMMA_N:
          j_a * WMMA_M * sqrt_N + j_b * WMMA_N),
          acc_frag_1[j_a][j_b][0], sqrt_N, out_layout
        );
  
        wmma::store_matrix_sync(
          b_imag + (out_trans ?
          j_b * WMMA_M * sqrt_N + j_a * WMMA_N:
          j_a * WMMA_M * sqrt_N + j_b * WMMA_N),
          acc_frag_1[j_a][j_b][1], sqrt_N, out_layout
        );
      }
    }
  }
}

template <typename ALayout, typename BLayout, bool out_trans, int MATMUL_WARP_WIDTH, bool output_to_shmem>
__device__ __forceinline__ void _complex_matmul_r2c_256(
  half *a_real,
  half *a_imag,
  int sqrt_N,
  int N,
  wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_K, WMMA_N, half, ALayout> a_frag[MATMUL_WARP_WIDTH][MATMUL_WARP_WIDTH][2],
  wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_K, WMMA_N, half, BLayout> b_frag[MATMUL_WARP_WIDTH][MATMUL_WARP_WIDTH][2],
  wmma::fragment<wmma::accumulator, WMMA_M, WMMA_K, WMMA_N, half> acc_frag_1[MATMUL_WARP_WIDTH][MATMUL_WARP_WIDTH][2],
  wmma::layout_t out_layout = wmma::mem_row_major)
{
  // #pragma unroll
  for (int j_b = 0; j_b < MATMUL_WARP_WIDTH; j_b++) {
    // #pragma unroll
    for (int j_a = 0; j_a < MATMUL_WARP_WIDTH; j_a++) {
      wmma::fill_fragment(acc_frag_1[j_a][j_b][0], __float2half(0.0f));

      // real

      // ac
      for (int k = 0; k < MATMUL_WARP_WIDTH; k++) {
        wmma::mma_sync(acc_frag_1[j_a][j_b][0], a_frag[j_a][k][0], b_frag[k][j_b][0], acc_frag_1[j_a][j_b][0]);
      }

      wmma::fill_fragment(acc_frag_1[j_a][j_b][1], __float2half(0.0f));

      // imag
      // ad
      for (int k = 0; k < MATMUL_WARP_WIDTH; k++) {
        wmma::mma_sync(acc_frag_1[j_a][j_b][1], a_frag[j_a][k][0], b_frag[k][j_b][1], acc_frag_1[j_a][j_b][1]);
      }

    }
  }

  if (output_to_shmem) {
    // #pragma unroll
    for (int j_b = 0; j_b < MATMUL_WARP_WIDTH; j_b++) {
      // #pragma unroll
      for (int j_a = 0; j_a < MATMUL_WARP_WIDTH; j_a++) {
        // does it matter where we put this?
        wmma::store_matrix_sync(
          a_real + (out_trans ?
          j_b * WMMA_M * 256 + j_a * WMMA_N:
          j_a * WMMA_M * 256 + j_b * WMMA_N),
          acc_frag_1[j_a][j_b][0], 256, out_layout
        );
  
        wmma::store_matrix_sync(
          a_imag + (out_trans ?
          j_b * WMMA_M * 256 + j_a * WMMA_N:
          j_a * WMMA_M * 256 + j_b * WMMA_N),
          acc_frag_1[j_a][j_b][1], 256, out_layout
        );
      }
    }
  }
}

template <typename ALayout, typename BLayout, bool out_trans, int MATMUL_WARP_WIDTH, bool output_to_shmem>
__device__ __forceinline__ void _complex_matmul_256(
  half *a_real,
  half *a_imag,
  int sqrt_N,
  int N,
  wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_K, WMMA_N, half, ALayout> a_frag[MATMUL_WARP_WIDTH][MATMUL_WARP_WIDTH][2],
  wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_K, WMMA_N, half, BLayout> b_frag[MATMUL_WARP_WIDTH][MATMUL_WARP_WIDTH][2],
  wmma::fragment<wmma::accumulator, WMMA_M, WMMA_K, WMMA_N, half> acc_frag_1[MATMUL_WARP_WIDTH][MATMUL_WARP_WIDTH][2],
  wmma::layout_t out_layout = wmma::mem_row_major)
{
  // #pragma unroll
  for (int j_b = 0; j_b < MATMUL_WARP_WIDTH; j_b++) {
    // #pragma unroll
    for (int j_a = 0; j_a < MATMUL_WARP_WIDTH; j_a++) {
      wmma::fill_fragment(acc_frag_1[j_a][j_b][0], __float2half(0.0f));

      // real
      // bd
      // #pragma unroll
      for (int k = 0; k < MATMUL_WARP_WIDTH; k++) {
        wmma::mma_sync(acc_frag_1[j_a][j_b][0], a_frag[j_a][k][1], b_frag[k][j_b][1], acc_frag_1[j_a][j_b][0]);
      }

      // bd -> -bd
      // #pragma unroll
      for (int i = 0; i < acc_frag_1[j_a][j_b][0].num_elements; i++) {
        acc_frag_1[j_a][j_b][0].x[i] = __hneg(acc_frag_1[j_a][j_b][0].x[i]);
      }

      // ac
      // #pragma unroll
      for (int k = 0; k < MATMUL_WARP_WIDTH; k++) {
        wmma::mma_sync(acc_frag_1[j_a][j_b][0], a_frag[j_a][k][0], b_frag[k][j_b][0], acc_frag_1[j_a][j_b][0]);
      }

      wmma::fill_fragment(acc_frag_1[j_a][j_b][1], __float2half(0.0f));

      // imag
      // ad
      // #pragma unroll
      for (int k = 0; k < MATMUL_WARP_WIDTH; k++) {
        wmma::mma_sync(acc_frag_1[j_a][j_b][1], a_frag[j_a][k][0], b_frag[k][j_b][1], acc_frag_1[j_a][j_b][1]);
      }

      // bc
      // #pragma unroll
      for (int k = 0; k < MATMUL_WARP_WIDTH; k++) {
        wmma::mma_sync(acc_frag_1[j_a][j_b][1], a_frag[j_a][k][1], b_frag[k][j_b][0], acc_frag_1[j_a][j_b][1]);
      }

    }
  }

  if (output_to_shmem) {
    // #pragma unroll
    for (int j_b = 0; j_b < MATMUL_WARP_WIDTH; j_b++) {
      // #pragma unroll
      for (int j_a = 0; j_a < MATMUL_WARP_WIDTH; j_a++) {
        // does it matter where we put this?
        wmma::store_matrix_sync(
          a_real + (out_trans ?
          j_b * WMMA_M * 256 + j_a * WMMA_N:
          j_a * WMMA_M * 256 + j_b * WMMA_N),
          acc_frag_1[j_a][j_b][0], 256, out_layout
        );
  
        wmma::store_matrix_sync(
          a_imag + (out_trans ?
          j_b * WMMA_M * 256 + j_a * WMMA_N:
          j_a * WMMA_M * 256 + j_b * WMMA_N),
          acc_frag_1[j_a][j_b][1], 256, out_layout
        );
      }
    }
  }
}

template <typename ALayout, typename BLayout, bool out_trans, int MATMUL_WARP_WIDTH, bool output_to_shmem>
__device__ __forceinline__ void _complex_matmul_1024(
  half *a_real,
  half *a_imag,
  int sqrt_N,
  int N,
  wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_K, WMMA_N, half, ALayout> a_frag[MATMUL_WARP_WIDTH][MATMUL_WARP_WIDTH][2],
  wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_K, WMMA_N, half, BLayout> b_frag[MATMUL_WARP_WIDTH][MATMUL_WARP_WIDTH][2],
  wmma::fragment<wmma::accumulator, WMMA_M, WMMA_K, WMMA_N, half> acc_frag_1[MATMUL_WARP_WIDTH][MATMUL_WARP_WIDTH][2],
  wmma::layout_t out_layout = wmma::mem_row_major)
{
  // #pragma unroll
  for (int j_b = 0; j_b < MATMUL_WARP_WIDTH; j_b++) {
    // #pragma unroll
    for (int j_a = 0; j_a < MATMUL_WARP_WIDTH; j_a++) {
      wmma::fill_fragment(acc_frag_1[j_a][j_b][0], __float2half(0.0f));

      // real
      // bd
      // #pragma unroll
      for (int k = 0; k < MATMUL_WARP_WIDTH; k++) {
        wmma::mma_sync(acc_frag_1[j_a][j_b][0], a_frag[j_a][k][1], b_frag[k][j_b][1], acc_frag_1[j_a][j_b][0]);
      }

      // bd -> -bd
      // #pragma unroll
      for (int i = 0; i < acc_frag_1[j_a][j_b][0].num_elements; i++) {
        acc_frag_1[j_a][j_b][0].x[i] = __hneg(acc_frag_1[j_a][j_b][0].x[i]);
      }

      // ac
      // #pragma unroll
      for (int k = 0; k < MATMUL_WARP_WIDTH; k++) {
        wmma::mma_sync(acc_frag_1[j_a][j_b][0], a_frag[j_a][k][0], b_frag[k][j_b][0], acc_frag_1[j_a][j_b][0]);
      }

      wmma::fill_fragment(acc_frag_1[j_a][j_b][1], __float2half(0.0f));

      // imag
      // ad
      // #pragma unroll
      for (int k = 0; k < MATMUL_WARP_WIDTH; k++) {
        wmma::mma_sync(acc_frag_1[j_a][j_b][1], a_frag[j_a][k][0], b_frag[k][j_b][1], acc_frag_1[j_a][j_b][1]);
      }

      // bc
      // #pragma unroll
      for (int k = 0; k < MATMUL_WARP_WIDTH; k++) {
        wmma::mma_sync(acc_frag_1[j_a][j_b][1], a_frag[j_a][k][1], b_frag[k][j_b][0], acc_frag_1[j_a][j_b][1]);
      }

    }
  }

  if (output_to_shmem) {
    // #pragma unroll
    for (int j_b = 0; j_b < MATMUL_WARP_WIDTH; j_b++) {
      // #pragma unroll
      for (int j_a = 0; j_a < MATMUL_WARP_WIDTH; j_a++) {
        // does it matter where we put this?
        wmma::store_matrix_sync(
          a_real + (out_trans ?
          j_b * WMMA_M * 1024 + j_a * WMMA_N:
          j_a * WMMA_M * 1024 + j_b * WMMA_N),
          acc_frag_1[j_a][j_b][0], 1024, out_layout
        );
  
        wmma::store_matrix_sync(
          a_imag + (out_trans ?
          j_b * WMMA_M * 1024 + j_a * WMMA_N:
          j_a * WMMA_M * 1024 + j_b * WMMA_N),
          acc_frag_1[j_a][j_b][1], 1024, out_layout
        );
      }
    }
  }
}

template <typename ALayout, typename BLayout, bool out_trans, int MATMUL_WARP_WIDTH, bool output_to_shmem>
__device__ __forceinline__ void _complex_matmul_r2c_1024(
  half *a_real,
  half *a_imag,
  int sqrt_N,
  int N,
  wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_K, WMMA_N, half, ALayout> a_frag[MATMUL_WARP_WIDTH][MATMUL_WARP_WIDTH][2],
  wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_K, WMMA_N, half, BLayout> b_frag[MATMUL_WARP_WIDTH][MATMUL_WARP_WIDTH][2],
  wmma::fragment<wmma::accumulator, WMMA_M, WMMA_K, WMMA_N, half> acc_frag_1[MATMUL_WARP_WIDTH][MATMUL_WARP_WIDTH][2],
  wmma::layout_t out_layout = wmma::mem_row_major)
{
  // #pragma unroll
  for (int j_b = 0; j_b < MATMUL_WARP_WIDTH; j_b++) {
    // #pragma unroll
    for (int j_a = 0; j_a < MATMUL_WARP_WIDTH; j_a++) {
      wmma::fill_fragment(acc_frag_1[j_a][j_b][0], __float2half(0.0f));

      // real

      // ac
      for (int k = 0; k < MATMUL_WARP_WIDTH; k++) {
        wmma::mma_sync(acc_frag_1[j_a][j_b][0], a_frag[j_a][k][0], b_frag[k][j_b][0], acc_frag_1[j_a][j_b][0]);
      }

      wmma::fill_fragment(acc_frag_1[j_a][j_b][1], __float2half(0.0f));

      // imag
      // ad
      for (int k = 0; k < MATMUL_WARP_WIDTH; k++) {
        wmma::mma_sync(acc_frag_1[j_a][j_b][1], a_frag[j_a][k][0], b_frag[k][j_b][1], acc_frag_1[j_a][j_b][1]);
      }

    }
  }

  if (output_to_shmem) {
    // #pragma unroll
    for (int j_b = 0; j_b < MATMUL_WARP_WIDTH; j_b++) {
      // #pragma unroll
      for (int j_a = 0; j_a < MATMUL_WARP_WIDTH; j_a++) {
        // does it matter where we put this?
        wmma::store_matrix_sync(
          a_real + (out_trans ?
          j_b * WMMA_M * 1024 + j_a * WMMA_N:
          j_a * WMMA_M * 1024 + j_b * WMMA_N),
          acc_frag_1[j_a][j_b][0], 1024, out_layout
        );
  
        wmma::store_matrix_sync(
          a_imag + (out_trans ?
          j_b * WMMA_M * 1024 + j_a * WMMA_N:
          j_a * WMMA_M * 1024 + j_b * WMMA_N),
          acc_frag_1[j_a][j_b][1], 1024, out_layout
        );
      }
    }
  }
}

template <typename ALayout, typename BLayout, bool out_trans, int MATMUL_WARP_WIDTH, bool output_to_shmem>
__device__ __forceinline__ void _complex_matmul_c2r(
  half *a_real_out,
  int sqrt_N,
  int N,
  wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_K, WMMA_N, half, ALayout> a_frag[MATMUL_WARP_WIDTH][MATMUL_WARP_WIDTH][2],
  wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_K, WMMA_N, half, BLayout> b_frag[MATMUL_WARP_WIDTH][MATMUL_WARP_WIDTH][2],
  wmma::fragment<wmma::accumulator, WMMA_M, WMMA_K, WMMA_N, half> acc_frag_1[MATMUL_WARP_WIDTH][MATMUL_WARP_WIDTH][2],
  wmma::layout_t out_layout = wmma::mem_row_major)
{
  #pragma unroll
  for (int j_b = 0; j_b < MATMUL_WARP_WIDTH; j_b++) {
    // #pragma unroll
    for (int j_a = 0; j_a < MATMUL_WARP_WIDTH; j_a++) {
      wmma::fill_fragment(acc_frag_1[j_a][j_b][0], __float2half(0.0f));

      // real
      // bd
      for (int k = 0; k < MATMUL_WARP_WIDTH; k++) {
        wmma::mma_sync(acc_frag_1[j_a][j_b][0], a_frag[j_a][k][1], b_frag[k][j_b][1], acc_frag_1[j_a][j_b][0]);
      }

      // bd -> -bd
      for (int i = 0; i < acc_frag_1[j_a][j_b][0].num_elements; i++) {
        acc_frag_1[j_a][j_b][0].x[i] = __hneg(acc_frag_1[j_a][j_b][0].x[i]);
      }

      // ac
      for (int k = 0; k < MATMUL_WARP_WIDTH; k++) {
        wmma::mma_sync(acc_frag_1[j_a][j_b][0], a_frag[j_a][k][0], b_frag[k][j_b][0], acc_frag_1[j_a][j_b][0]);
      }

    }
  }

  if (output_to_shmem) {
    // #pragma unroll
    for (int j_b = 0; j_b < MATMUL_WARP_WIDTH; j_b++) {
      // #pragma unroll
      for (int j_a = 0; j_a < MATMUL_WARP_WIDTH; j_a++) {
        // does it matter where we put this?
        wmma::store_matrix_sync(
          a_real_out + (out_trans ?
          j_b * WMMA_M * sqrt_N + j_a * WMMA_N:
          j_a * WMMA_M * sqrt_N + j_b * WMMA_N),
          acc_frag_1[j_a][j_b][0], sqrt_N, out_layout
        );
      }
    }
  }
}

template <typename ALayout, typename BLayout, bool out_trans, int MATMUL_WARP_WIDTH, bool output_to_shmem>
__device__ __forceinline__ void _complex_matmul_c2r_256(
  half *a_real_out,
  int sqrt_N,
  int N,
  wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_K, WMMA_N, half, ALayout> a_frag[MATMUL_WARP_WIDTH][MATMUL_WARP_WIDTH][2],
  wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_K, WMMA_N, half, BLayout> b_frag[MATMUL_WARP_WIDTH][MATMUL_WARP_WIDTH][2],
  wmma::fragment<wmma::accumulator, WMMA_M, WMMA_K, WMMA_N, half> acc_frag_1[MATMUL_WARP_WIDTH][MATMUL_WARP_WIDTH][2],
  wmma::layout_t out_layout = wmma::mem_row_major)
{
  #pragma unroll
  for (int j_b = 0; j_b < MATMUL_WARP_WIDTH; j_b++) {
    // #pragma unroll
    for (int j_a = 0; j_a < MATMUL_WARP_WIDTH; j_a++) {
      wmma::fill_fragment(acc_frag_1[j_a][j_b][0], __float2half(0.0f));

      // real
      // bd
      for (int k = 0; k < MATMUL_WARP_WIDTH; k++) {
        wmma::mma_sync(acc_frag_1[j_a][j_b][0], a_frag[j_a][k][1], b_frag[k][j_b][1], acc_frag_1[j_a][j_b][0]);
      }

      // bd -> -bd
      for (int i = 0; i < acc_frag_1[j_a][j_b][0].num_elements; i++) {
        acc_frag_1[j_a][j_b][0].x[i] = __hneg(acc_frag_1[j_a][j_b][0].x[i]);
      }

      // ac
      for (int k = 0; k < MATMUL_WARP_WIDTH; k++) {
        wmma::mma_sync(acc_frag_1[j_a][j_b][0], a_frag[j_a][k][0], b_frag[k][j_b][0], acc_frag_1[j_a][j_b][0]);
      }

    }
  }

  if (output_to_shmem) {
    // #pragma unroll
    for (int j_b = 0; j_b < MATMUL_WARP_WIDTH; j_b++) {
      // #pragma unroll
      for (int j_a = 0; j_a < MATMUL_WARP_WIDTH; j_a++) {
        // does it matter where we put this?
        wmma::store_matrix_sync(
          a_real_out + (out_trans ?
          j_b * WMMA_M * 256 + j_a * WMMA_N:
          j_a * WMMA_M * 256 + j_b * WMMA_N),
          acc_frag_1[j_a][j_b][0], 256, out_layout
        );
      }
    }
  }
}

template <typename ALayout, typename BLayout, bool out_trans, int MATMUL_WARP_WIDTH, bool output_to_shmem>
__device__ __forceinline__ void _complex_matmul_c2r_1024(
  half *a_real_out,
  int sqrt_N,
  int N,
  wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_K, WMMA_N, half, ALayout> a_frag[MATMUL_WARP_WIDTH][MATMUL_WARP_WIDTH][2],
  wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_K, WMMA_N, half, BLayout> b_frag[MATMUL_WARP_WIDTH][MATMUL_WARP_WIDTH][2],
  wmma::fragment<wmma::accumulator, WMMA_M, WMMA_K, WMMA_N, half> acc_frag_1[MATMUL_WARP_WIDTH][MATMUL_WARP_WIDTH][2],
  wmma::layout_t out_layout = wmma::mem_row_major)
{
  #pragma unroll
  for (int j_b = 0; j_b < MATMUL_WARP_WIDTH; j_b++) {
    // #pragma unroll
    for (int j_a = 0; j_a < MATMUL_WARP_WIDTH; j_a++) {
      wmma::fill_fragment(acc_frag_1[j_a][j_b][0], __float2half(0.0f));

      // real
      // bd
      for (int k = 0; k < MATMUL_WARP_WIDTH; k++) {
        wmma::mma_sync(acc_frag_1[j_a][j_b][0], a_frag[j_a][k][1], b_frag[k][j_b][1], acc_frag_1[j_a][j_b][0]);
      }

      // bd -> -bd
      for (int i = 0; i < acc_frag_1[j_a][j_b][0].num_elements; i++) {
        acc_frag_1[j_a][j_b][0].x[i] = __hneg(acc_frag_1[j_a][j_b][0].x[i]);
      }

      // ac
      for (int k = 0; k < MATMUL_WARP_WIDTH; k++) {
        wmma::mma_sync(acc_frag_1[j_a][j_b][0], a_frag[j_a][k][0], b_frag[k][j_b][0], acc_frag_1[j_a][j_b][0]);
      }

    }
  }

  if (output_to_shmem) {
    // #pragma unroll
    for (int j_b = 0; j_b < MATMUL_WARP_WIDTH; j_b++) {
      // #pragma unroll
      for (int j_a = 0; j_a < MATMUL_WARP_WIDTH; j_a++) {
        // does it matter where we put this?
        wmma::store_matrix_sync(
          a_real_out + (out_trans ?
          j_b * WMMA_M * 1024 + j_a * WMMA_N:
          j_a * WMMA_M * 1024 + j_b * WMMA_N),
          acc_frag_1[j_a][j_b][0], 1024, out_layout
        );
      }
    }
  }
}

#endif