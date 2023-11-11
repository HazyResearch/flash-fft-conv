// Copyright (c) 2023 Dan Fu, Hermann Kumbong

#include <torch/extension.h>

#include <vector>
#include <stdio.h>
#include <mma.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
using namespace nvcuda;

using complex_bfloat16_t = typename c10::complex<at::BFloat16>;

#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16
// #define TILE_SIZE 4
// #define SHMEM_SIZE 256 * TILE_SIZE
// #define SEQUENCE_SIZE 256
#define WARP_SIZE 32

#ifndef MONARCH_CUDA_BF16_MATMULS_
#define MONARCH_CUDA_BF16_MATMULS_

__device__ __forceinline__ void floatacc2bfloatacc(
  wmma::fragment<wmma::accumulator, WMMA_M, WMMA_K, WMMA_N, float> *float_acc,
  wmma::fragment<wmma::accumulator, WMMA_M, WMMA_K, WMMA_N, half> *bfloat_acc
) {
  for (int i = 0; i < float_acc->num_elements; i++) {
    reinterpret_cast<__nv_bfloat16 *>(bfloat_acc->x)[i] = __float2bfloat16(float_acc->x[i]);
  }
  // for (int i = 0; i < float_acc->num_elements / 2; i++) {
  //   reinterpret_cast<__nv_bfloat162 *>(bfloat_acc->x)[i] = __float22bfloat162_rn(reinterpret_cast<float2 *>(float_acc->x)[i]);
  // }
}

template <typename ALayout, typename BLayout, bool out_trans, int MATMUL_WARP_WIDTH, bool output_to_shmem>
__device__ __forceinline__ void _complex_matmul(
  __nv_bfloat16 *a_real,
  __nv_bfloat16 *a_imag,
  int sqrt_N,
  int N,
  wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_K, WMMA_N, __nv_bfloat16, ALayout> a_frag[MATMUL_WARP_WIDTH][MATMUL_WARP_WIDTH][2],
  wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_K, WMMA_N, __nv_bfloat16, BLayout> b_frag[MATMUL_WARP_WIDTH][MATMUL_WARP_WIDTH][2],
  wmma::fragment<wmma::accumulator, WMMA_M, WMMA_K, WMMA_N, float> acc_frag_1[MATMUL_WARP_WIDTH][MATMUL_WARP_WIDTH][2],
  wmma::fragment<wmma::accumulator, WMMA_M, WMMA_K, WMMA_N, half> acc_frag_half[MATMUL_WARP_WIDTH][MATMUL_WARP_WIDTH][2],
  wmma::layout_t out_layout = wmma::mem_row_major)
{
  // #pragma unroll
  for (int j_b = 0; j_b < MATMUL_WARP_WIDTH; j_b++) {
    // #pragma unroll
    for (int j_a = 0; j_a < MATMUL_WARP_WIDTH; j_a++) {
      wmma::fill_fragment(acc_frag_1[j_a][j_b][0], 0.0f);

      // real
      // bd
      // #pragma unroll
      for (int k = 0; k < MATMUL_WARP_WIDTH; k++) {
        wmma::mma_sync(acc_frag_1[j_a][j_b][0], a_frag[j_a][k][1], b_frag[k][j_b][1], acc_frag_1[j_a][j_b][0]);
      }

      // bd -> -bd
      // #pragma unroll
      for (int i = 0; i < acc_frag_1[j_a][j_b][0].num_elements; i++) {
        acc_frag_1[j_a][j_b][0].x[i] =  -acc_frag_1[j_a][j_b][0].x[i];
      }

      // ac
      // #pragma unroll
      for (int k = 0; k < MATMUL_WARP_WIDTH; k++) {
        wmma::mma_sync(acc_frag_1[j_a][j_b][0], a_frag[j_a][k][0], b_frag[k][j_b][0], acc_frag_1[j_a][j_b][0]);
      }

      floatacc2bfloatacc(&acc_frag_1[j_a][j_b][0], &acc_frag_half[j_a][j_b][0]);

      wmma::fill_fragment(acc_frag_1[j_a][j_b][1], 0.0f);

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

      floatacc2bfloatacc(&acc_frag_1[j_a][j_b][1], &acc_frag_half[j_a][j_b][1]);      

    }
  }

  if (output_to_shmem) {
    // #pragma unroll
    for (int j_b = 0; j_b < MATMUL_WARP_WIDTH; j_b++) {
      // #pragma unroll
      for (int j_a = 0; j_a < MATMUL_WARP_WIDTH; j_a++) {
        // does it matter where we put this?
        wmma::store_matrix_sync(
          reinterpret_cast<half *> (
            a_real + (out_trans ?
            j_b * WMMA_M * sqrt_N + j_a * WMMA_N:
            j_a * WMMA_M * sqrt_N + j_b * WMMA_N)
          ),
          acc_frag_half[j_a][j_b][0], sqrt_N, out_layout
        );
  
        wmma::store_matrix_sync(
          reinterpret_cast<half *> (
            a_imag + (out_trans ?
            j_b * WMMA_M * sqrt_N + j_a * WMMA_N:
            j_a * WMMA_M * sqrt_N + j_b * WMMA_N)
          ),
          acc_frag_half[j_a][j_b][1], sqrt_N, out_layout
        );
      }
    }
  }
}

template <typename ALayout, typename BLayout, bool out_trans, int MATMUL_WARP_WIDTH, bool output_to_shmem>
__device__ __forceinline__ void _complex_matmul_r2c_load_b(
  __nv_bfloat16* a_real,
  __nv_bfloat16* a_imag,
  int sqrt_N,
  int N,
  wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_K, WMMA_N, __nv_bfloat16, ALayout> a_frag[MATMUL_WARP_WIDTH][MATMUL_WARP_WIDTH][2],
  wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_K, WMMA_N, __nv_bfloat16, BLayout> b_frag[MATMUL_WARP_WIDTH][MATMUL_WARP_WIDTH][2],
  wmma::fragment<wmma::accumulator, WMMA_M, WMMA_K, WMMA_N, float> acc_frag_1[MATMUL_WARP_WIDTH][MATMUL_WARP_WIDTH][2],
  wmma::fragment<wmma::accumulator, WMMA_M, WMMA_K, WMMA_N, half> acc_frag_half[MATMUL_WARP_WIDTH][MATMUL_WARP_WIDTH][2],
  wmma::layout_t out_layout = wmma::mem_row_major)
{
  // #pragma unroll
  for (int j_b = 0; j_b < MATMUL_WARP_WIDTH; j_b++) {
    // #pragma unroll
    for (int j_a = 0; j_a < MATMUL_WARP_WIDTH; j_a++) {
      wmma::fill_fragment(acc_frag_1[j_a][j_b][0], 0.0f);

      // real
      // ac
      // #pragma unroll
      for (int k = 0; k < MATMUL_WARP_WIDTH; k++) {
        wmma::mma_sync(acc_frag_1[j_a][j_b][0], a_frag[j_a][k][0], b_frag[k][j_b][0], acc_frag_1[j_a][j_b][0]);
      }

      floatacc2bfloatacc(&acc_frag_1[j_a][j_b][0], &acc_frag_half[j_a][j_b][0]);

      wmma::fill_fragment(acc_frag_1[j_a][j_b][1], 0.0f);

      // imag
      // bc
      // #pragma unroll
      for (int k = 0; k < MATMUL_WARP_WIDTH; k++) {
        wmma::mma_sync(acc_frag_1[j_a][j_b][1], a_frag[j_a][k][1], b_frag[k][j_b][0], acc_frag_1[j_a][j_b][1]);
      }

      floatacc2bfloatacc(&acc_frag_1[j_a][j_b][1], &acc_frag_half[j_a][j_b][1]);

    }
  }

  if (output_to_shmem) {
    // #pragma unroll
    for (int j_b = 0; j_b < MATMUL_WARP_WIDTH; j_b++) {
      // #pragma unroll
      for (int j_a = 0; j_a < MATMUL_WARP_WIDTH; j_a++) {
        
        //does it matter where we put this?
        wmma::store_matrix_sync(
          reinterpret_cast<half *>(
            a_real + (out_trans ?
            j_b * WMMA_M * sqrt_N + j_a * WMMA_N:
            j_a * WMMA_M * sqrt_N + j_b * WMMA_N)
          ),
          acc_frag_half[j_a][j_b][0], sqrt_N, out_layout
        );
  
        wmma::store_matrix_sync(
          reinterpret_cast<half *>(
            a_imag + (out_trans ?
            j_b * WMMA_M * sqrt_N + j_a * WMMA_N:
            j_a * WMMA_M * sqrt_N + j_b * WMMA_N)
          ),
          acc_frag_half[j_a][j_b][1], sqrt_N, out_layout
        );
      }
    }
  }
}

template <typename ALayout, typename BLayout, bool out_trans, int MATMUL_WARP_WIDTH, bool output_to_shmem>
__device__ __forceinline__ void _complex_matmul_256(
  __nv_bfloat16 *a_real,
  __nv_bfloat16 *a_imag,
  int sqrt_N,
  int N,
  wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_K, WMMA_N, __nv_bfloat16, ALayout> a_frag[MATMUL_WARP_WIDTH][MATMUL_WARP_WIDTH][2],
  wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_K, WMMA_N, __nv_bfloat16, BLayout> b_frag[MATMUL_WARP_WIDTH][MATMUL_WARP_WIDTH][2],
  wmma::fragment<wmma::accumulator, WMMA_M, WMMA_K, WMMA_N, float> acc_frag_1[MATMUL_WARP_WIDTH][MATMUL_WARP_WIDTH][2],
  wmma::fragment<wmma::accumulator, WMMA_M, WMMA_K, WMMA_N, half> acc_frag_half[MATMUL_WARP_WIDTH][MATMUL_WARP_WIDTH][2],
  wmma::layout_t out_layout = wmma::mem_row_major)
{
  // #pragma unroll
  for (int j_b = 0; j_b < MATMUL_WARP_WIDTH; j_b++) {
    // #pragma unroll
    for (int j_a = 0; j_a < MATMUL_WARP_WIDTH; j_a++) {
      wmma::fill_fragment(acc_frag_1[j_a][j_b][0], 0.0f);

      // real
      // bd
      // #pragma unroll
      for (int k = 0; k < MATMUL_WARP_WIDTH; k++) {
        wmma::mma_sync(acc_frag_1[j_a][j_b][0], a_frag[j_a][k][1], b_frag[k][j_b][1], acc_frag_1[j_a][j_b][0]);
      }

      // bd -> -bd
      // #pragma unroll
      for (int i = 0; i < acc_frag_1[j_a][j_b][0].num_elements; i++) {
        acc_frag_1[j_a][j_b][0].x[i] =  -acc_frag_1[j_a][j_b][0].x[i];
      }

      // ac
      // #pragma unroll
      for (int k = 0; k < MATMUL_WARP_WIDTH; k++) {
        wmma::mma_sync(acc_frag_1[j_a][j_b][0], a_frag[j_a][k][0], b_frag[k][j_b][0], acc_frag_1[j_a][j_b][0]);
      }

      floatacc2bfloatacc(&acc_frag_1[j_a][j_b][0], &acc_frag_half[j_a][j_b][0]);

      wmma::fill_fragment(acc_frag_1[j_a][j_b][1], 0.0f);

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

      floatacc2bfloatacc(&acc_frag_1[j_a][j_b][1], &acc_frag_half[j_a][j_b][1]);      

    }
  }

  if (output_to_shmem) {
    // #pragma unroll
    for (int j_b = 0; j_b < MATMUL_WARP_WIDTH; j_b++) {
      // #pragma unroll
      for (int j_a = 0; j_a < MATMUL_WARP_WIDTH; j_a++) {
        // does it matter where we put this?
        wmma::store_matrix_sync(
          reinterpret_cast<half *> (
            a_real + (out_trans ?
            j_b * WMMA_M * 256 + j_a * WMMA_N:
            j_a * WMMA_M * 256 + j_b * WMMA_N)
          ),
          acc_frag_half[j_a][j_b][0], 256, out_layout
        );
  
        wmma::store_matrix_sync(
          reinterpret_cast<half *> (
            a_imag + (out_trans ?
            j_b * WMMA_M * 256 + j_a * WMMA_N:
            j_a * WMMA_M * 256 + j_b * WMMA_N)
          ),
          acc_frag_half[j_a][j_b][1], 256, out_layout
        );
      }
    }
  }
}

template <typename ALayout, typename BLayout, bool out_trans, int MATMUL_WARP_WIDTH, bool output_to_shmem>
__device__ __forceinline__ void _complex_matmul_r2c_256(
  __nv_bfloat16 *a_real,
  __nv_bfloat16 *a_imag,
  int sqrt_N,
  int N,
  wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_K, WMMA_N, __nv_bfloat16, ALayout> a_frag[MATMUL_WARP_WIDTH][MATMUL_WARP_WIDTH][2],
  wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_K, WMMA_N, __nv_bfloat16, BLayout> b_frag[MATMUL_WARP_WIDTH][MATMUL_WARP_WIDTH][2],
  wmma::fragment<wmma::accumulator, WMMA_M, WMMA_K, WMMA_N, float> acc_frag_1[MATMUL_WARP_WIDTH][MATMUL_WARP_WIDTH][2],
  wmma::fragment<wmma::accumulator, WMMA_M, WMMA_K, WMMA_N, half> acc_frag_half[MATMUL_WARP_WIDTH][MATMUL_WARP_WIDTH][2],
  wmma::layout_t out_layout = wmma::mem_row_major)
{
  // #pragma unroll
  for (int j_b = 0; j_b < MATMUL_WARP_WIDTH; j_b++) {
    // #pragma unroll
    for (int j_a = 0; j_a < MATMUL_WARP_WIDTH; j_a++) {
      wmma::fill_fragment(acc_frag_1[j_a][j_b][0], 0.0f);

      // real

      // ac
      for (int k = 0; k < MATMUL_WARP_WIDTH; k++) {
        wmma::mma_sync(acc_frag_1[j_a][j_b][0], a_frag[j_a][k][0], b_frag[k][j_b][0], acc_frag_1[j_a][j_b][0]);
      }

      floatacc2bfloatacc(&acc_frag_1[j_a][j_b][0], &acc_frag_half[j_a][j_b][0]);

      wmma::fill_fragment(acc_frag_1[j_a][j_b][1], 0.0f);

      // imag
      // ad
      for (int k = 0; k < MATMUL_WARP_WIDTH; k++) {
        wmma::mma_sync(acc_frag_1[j_a][j_b][1], a_frag[j_a][k][0], b_frag[k][j_b][1], acc_frag_1[j_a][j_b][1]);
      }

      floatacc2bfloatacc(&acc_frag_1[j_a][j_b][1], &acc_frag_half[j_a][j_b][1]);

    }
  }

  if (output_to_shmem) {
    // #pragma unroll
    for (int j_b = 0; j_b < MATMUL_WARP_WIDTH; j_b++) {
      // #pragma unroll
      for (int j_a = 0; j_a < MATMUL_WARP_WIDTH; j_a++) {
        //accumlator fragments are not supporte for bfloat16, so we cannot directly cast or store the values to shared memory
        //of type bfloat 16. We need to move the values to the a_fragment which supports bfloat16 and then store it to shared memory
        //does it matter where we put this?
        wmma::store_matrix_sync(
          reinterpret_cast<half *>(
            a_real + (out_trans ?
            j_b * WMMA_M * 256 + j_a * WMMA_N:
            j_a * WMMA_M * 256 + j_b * WMMA_N)
          ),
          acc_frag_half[j_a][j_b][0], 256, out_layout
        );
  
        wmma::store_matrix_sync(
          reinterpret_cast<half *> (
            a_imag + (out_trans ?
            j_b * WMMA_M * 256 + j_a * WMMA_N:
            j_a * WMMA_M * 256 + j_b * WMMA_N)
          ),
          acc_frag_half[j_a][j_b][1], 256, out_layout
        );
      }
    }
  }
}

template <typename ALayout, typename BLayout, bool out_trans, int MATMUL_WARP_WIDTH, bool output_to_shmem>
__device__ __forceinline__ void _complex_matmul_1024(
  __nv_bfloat16 *a_real,
  __nv_bfloat16 *a_imag,
  int sqrt_N,
  int N,
  wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_K, WMMA_N, __nv_bfloat16, ALayout> a_frag[MATMUL_WARP_WIDTH][MATMUL_WARP_WIDTH][2],
  wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_K, WMMA_N, __nv_bfloat16, BLayout> b_frag[MATMUL_WARP_WIDTH][MATMUL_WARP_WIDTH][2],
  wmma::fragment<wmma::accumulator, WMMA_M, WMMA_K, WMMA_N, float> acc_frag_1[MATMUL_WARP_WIDTH][MATMUL_WARP_WIDTH][2],
  wmma::fragment<wmma::accumulator, WMMA_M, WMMA_K, WMMA_N, half> acc_frag_half[MATMUL_WARP_WIDTH][MATMUL_WARP_WIDTH][2],
  wmma::layout_t out_layout = wmma::mem_row_major)
{
  // #pragma unroll
  for (int j_b = 0; j_b < MATMUL_WARP_WIDTH; j_b++) {
    // #pragma unroll
    for (int j_a = 0; j_a < MATMUL_WARP_WIDTH; j_a++) {
      wmma::fill_fragment(acc_frag_1[j_a][j_b][0], 0.0f);

      // real
      // bd
      // #pragma unroll
      for (int k = 0; k < MATMUL_WARP_WIDTH; k++) {
        wmma::mma_sync(acc_frag_1[j_a][j_b][0], a_frag[j_a][k][1], b_frag[k][j_b][1], acc_frag_1[j_a][j_b][0]);
      }

      // bd -> -bd
      // #pragma unroll
      for (int i = 0; i < acc_frag_1[j_a][j_b][0].num_elements; i++) {
        acc_frag_1[j_a][j_b][0].x[i] =  -acc_frag_1[j_a][j_b][0].x[i];
      }

      // ac
      // #pragma unroll
      for (int k = 0; k < MATMUL_WARP_WIDTH; k++) {
        wmma::mma_sync(acc_frag_1[j_a][j_b][0], a_frag[j_a][k][0], b_frag[k][j_b][0], acc_frag_1[j_a][j_b][0]);
      }

      floatacc2bfloatacc(&acc_frag_1[j_a][j_b][0], &acc_frag_half[j_a][j_b][0]);

      wmma::fill_fragment(acc_frag_1[j_a][j_b][1], 0.0f);

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

      floatacc2bfloatacc(&acc_frag_1[j_a][j_b][1], &acc_frag_half[j_a][j_b][1]);      

    }
  }

  if (output_to_shmem) {
    // #pragma unroll
    for (int j_b = 0; j_b < MATMUL_WARP_WIDTH; j_b++) {
      // #pragma unroll
      for (int j_a = 0; j_a < MATMUL_WARP_WIDTH; j_a++) {
        // does it matter where we put this?
        wmma::store_matrix_sync(
          reinterpret_cast<half *> (
            a_real + (out_trans ?
            j_b * WMMA_M * 1024 + j_a * WMMA_N:
            j_a * WMMA_M * 1024 + j_b * WMMA_N)
          ),
          acc_frag_half[j_a][j_b][0], 1024, out_layout
        );
  
        wmma::store_matrix_sync(
          reinterpret_cast<half *> (
            a_imag + (out_trans ?
            j_b * WMMA_M * 1024 + j_a * WMMA_N:
            j_a * WMMA_M * 1024 + j_b * WMMA_N)
          ),
          acc_frag_half[j_a][j_b][1], 1024, out_layout
        );
      }
    }
  }
}

template <typename ALayout, typename BLayout, bool out_trans, int MATMUL_WARP_WIDTH, bool output_to_shmem>
__device__ __forceinline__ void _complex_matmul_r2c_1024(
  __nv_bfloat16 *a_real,
  __nv_bfloat16 *a_imag,
  int sqrt_N,
  int N,
  wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_K, WMMA_N, __nv_bfloat16, ALayout> a_frag[MATMUL_WARP_WIDTH][MATMUL_WARP_WIDTH][2],
  wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_K, WMMA_N, __nv_bfloat16, BLayout> b_frag[MATMUL_WARP_WIDTH][MATMUL_WARP_WIDTH][2],
  wmma::fragment<wmma::accumulator, WMMA_M, WMMA_K, WMMA_N, float> acc_frag_1[MATMUL_WARP_WIDTH][MATMUL_WARP_WIDTH][2],
  wmma::fragment<wmma::accumulator, WMMA_M, WMMA_K, WMMA_N, half> acc_frag_half[MATMUL_WARP_WIDTH][MATMUL_WARP_WIDTH][2],
  wmma::layout_t out_layout = wmma::mem_row_major)
{
  // #pragma unroll
  for (int j_b = 0; j_b < MATMUL_WARP_WIDTH; j_b++) {
    // #pragma unroll
    for (int j_a = 0; j_a < MATMUL_WARP_WIDTH; j_a++) {
      wmma::fill_fragment(acc_frag_1[j_a][j_b][0], 0.0f);

      // real

      // ac
      for (int k = 0; k < MATMUL_WARP_WIDTH; k++) {
        wmma::mma_sync(acc_frag_1[j_a][j_b][0], a_frag[j_a][k][0], b_frag[k][j_b][0], acc_frag_1[j_a][j_b][0]);
      }

      floatacc2bfloatacc(&acc_frag_1[j_a][j_b][0], &acc_frag_half[j_a][j_b][0]);

      wmma::fill_fragment(acc_frag_1[j_a][j_b][1], 0.0f);

      // imag
      // ad
      for (int k = 0; k < MATMUL_WARP_WIDTH; k++) {
        wmma::mma_sync(acc_frag_1[j_a][j_b][1], a_frag[j_a][k][0], b_frag[k][j_b][1], acc_frag_1[j_a][j_b][1]);
      }

      floatacc2bfloatacc(&acc_frag_1[j_a][j_b][1], &acc_frag_half[j_a][j_b][1]);

    }
  }

  if (output_to_shmem) {
    // #pragma unroll
    for (int j_b = 0; j_b < MATMUL_WARP_WIDTH; j_b++) {
      // #pragma unroll
      for (int j_a = 0; j_a < MATMUL_WARP_WIDTH; j_a++) {
        // does it matter where we put this?
        wmma::store_matrix_sync(
          reinterpret_cast<half *>(
            a_real + (out_trans ?
            j_b * WMMA_M * 1024 + j_a * WMMA_N:
            j_a * WMMA_M * 1024 + j_b * WMMA_N)
          ),
          acc_frag_half[j_a][j_b][0], 1024, out_layout
        );
  
        wmma::store_matrix_sync(
          reinterpret_cast<half *>(
            a_imag + (out_trans ?
            j_b * WMMA_M * 1024 + j_a * WMMA_N:
            j_a * WMMA_M * 1024 + j_b * WMMA_N)
          ),
          acc_frag_half[j_a][j_b][1], 1024, out_layout
        );
      }
    }
  }
}

template <typename ALayout, typename BLayout, bool out_trans, int MATMUL_WARP_WIDTH, bool output_to_shmem>
__device__ __forceinline__ void _complex_matmul_c2r(
  __nv_bfloat16 *a_real,
  int sqrt_N,
  int N,
  wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_K, WMMA_N, __nv_bfloat16, ALayout> a_frag[MATMUL_WARP_WIDTH][MATMUL_WARP_WIDTH][2],
  wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_K, WMMA_N, __nv_bfloat16, BLayout> b_frag[MATMUL_WARP_WIDTH][MATMUL_WARP_WIDTH][2],
  wmma::fragment<wmma::accumulator, WMMA_M, WMMA_K, WMMA_N, float> acc_frag_1[MATMUL_WARP_WIDTH][MATMUL_WARP_WIDTH][2],
  wmma::fragment<wmma::accumulator, WMMA_M, WMMA_K, WMMA_N, half> acc_frag_half[MATMUL_WARP_WIDTH][MATMUL_WARP_WIDTH][2],
  wmma::layout_t out_layout = wmma::mem_row_major)
{
  #pragma unroll
  for (int j_b = 0; j_b < MATMUL_WARP_WIDTH; j_b++) {
    // #pragma unroll
    for (int j_a = 0; j_a < MATMUL_WARP_WIDTH; j_a++) {
      wmma::fill_fragment(acc_frag_1[j_a][j_b][0], 0.0f);

      // real
      // bd
      for (int k = 0; k < MATMUL_WARP_WIDTH; k++) {
        wmma::mma_sync(acc_frag_1[j_a][j_b][0], a_frag[j_a][k][1], b_frag[k][j_b][1], acc_frag_1[j_a][j_b][0]);
      }

      // bd -> -bd
      for (int i = 0; i < acc_frag_1[j_a][j_b][0].num_elements; i++) {
        acc_frag_1[j_a][j_b][0].x[i] = -acc_frag_1[j_a][j_b][0].x[i];
      }

      // ac
      for (int k = 0; k < MATMUL_WARP_WIDTH; k++) {
        wmma::mma_sync(acc_frag_1[j_a][j_b][0], a_frag[j_a][k][0], b_frag[k][j_b][0], acc_frag_1[j_a][j_b][0]);
      }

      floatacc2bfloatacc(&acc_frag_1[j_a][j_b][0], &acc_frag_half[j_a][j_b][0]);

    }
  }

  if (output_to_shmem) {
    // #pragma unroll
    for (int j_b = 0; j_b < MATMUL_WARP_WIDTH; j_b++) {
      // #pragma unroll
      for (int j_a = 0; j_a < MATMUL_WARP_WIDTH; j_a++) {
        //accumlator fragments are not supporte for bfloat16, so we cannot directly cast or store the values to shared memory
        //of type bfloat 16. We need to move the values to the a_fragment which supports bfloat16 and then store it to shared memory

        //does it matter where we put this?
        wmma::store_matrix_sync(
          reinterpret_cast<half *>(
            a_real + (out_trans ?
            j_b * WMMA_M * sqrt_N + j_a * WMMA_N:
            j_a * WMMA_M * sqrt_N + j_b * WMMA_N)
          ),
          acc_frag_half[j_a][j_b][0], sqrt_N, out_layout
        );
      }
    }
  }
}

template <typename ALayout, typename BLayout, bool out_trans, int MATMUL_WARP_WIDTH, bool output_to_shmem>
__device__ __forceinline__ void _complex_matmul_c2r_256(
  __nv_bfloat16 *a_real,
  int sqrt_N,
  int N,
  wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_K, WMMA_N, __nv_bfloat16, ALayout> a_frag[MATMUL_WARP_WIDTH][MATMUL_WARP_WIDTH][2],
  wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_K, WMMA_N, __nv_bfloat16, BLayout> b_frag[MATMUL_WARP_WIDTH][MATMUL_WARP_WIDTH][2],
  wmma::fragment<wmma::accumulator, WMMA_M, WMMA_K, WMMA_N, float> acc_frag_1[MATMUL_WARP_WIDTH][MATMUL_WARP_WIDTH][2],
  wmma::fragment<wmma::accumulator, WMMA_M, WMMA_K, WMMA_N, half> acc_frag_half[MATMUL_WARP_WIDTH][MATMUL_WARP_WIDTH][2],
  wmma::layout_t out_layout = wmma::mem_row_major)
{
  #pragma unroll
  for (int j_b = 0; j_b < MATMUL_WARP_WIDTH; j_b++) {
    // #pragma unroll
    for (int j_a = 0; j_a < MATMUL_WARP_WIDTH; j_a++) {
      wmma::fill_fragment(acc_frag_1[j_a][j_b][0], 0.0f);

      // real
      // bd
      for (int k = 0; k < MATMUL_WARP_WIDTH; k++) {
        wmma::mma_sync(acc_frag_1[j_a][j_b][0], a_frag[j_a][k][1], b_frag[k][j_b][1], acc_frag_1[j_a][j_b][0]);
      }

      // bd -> -bd
      for (int i = 0; i < acc_frag_1[j_a][j_b][0].num_elements; i++) {
        acc_frag_1[j_a][j_b][0].x[i] = -acc_frag_1[j_a][j_b][0].x[i];
      }

      // ac
      for (int k = 0; k < MATMUL_WARP_WIDTH; k++) {
        wmma::mma_sync(acc_frag_1[j_a][j_b][0], a_frag[j_a][k][0], b_frag[k][j_b][0], acc_frag_1[j_a][j_b][0]);
      }

      floatacc2bfloatacc(&acc_frag_1[j_a][j_b][0], &acc_frag_half[j_a][j_b][0]);

    }
  }

  if (output_to_shmem) {
    // #pragma unroll
    for (int j_b = 0; j_b < MATMUL_WARP_WIDTH; j_b++) {
      // #pragma unroll
      for (int j_a = 0; j_a < MATMUL_WARP_WIDTH; j_a++) {
        //does it matter where we put this?
        wmma::store_matrix_sync(
          reinterpret_cast<half *> (
            a_real + (out_trans ?
            j_b * WMMA_M * 256 + j_a * WMMA_N:
            j_a * WMMA_M * 256 + j_b * WMMA_N)
          ),
          acc_frag_half[j_a][j_b][0], 256, out_layout
        );
      }
    }
  }
}

template <typename ALayout, typename BLayout, bool out_trans, int MATMUL_WARP_WIDTH, bool output_to_shmem>
__device__ __forceinline__ void _complex_matmul_c2r_1024(
  __nv_bfloat16 *a_real_out,
  int sqrt_N,
  int N,
  wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_K, WMMA_N, __nv_bfloat16, ALayout> a_frag[MATMUL_WARP_WIDTH][MATMUL_WARP_WIDTH][2],
  wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_K, WMMA_N, __nv_bfloat16, BLayout> b_frag[MATMUL_WARP_WIDTH][MATMUL_WARP_WIDTH][2],
  wmma::fragment<wmma::accumulator, WMMA_M, WMMA_K, WMMA_N, float> acc_frag_1[MATMUL_WARP_WIDTH][MATMUL_WARP_WIDTH][2],
  wmma::fragment<wmma::accumulator, WMMA_M, WMMA_K, WMMA_N, half> acc_frag_half[MATMUL_WARP_WIDTH][MATMUL_WARP_WIDTH][2],
  wmma::layout_t out_layout = wmma::mem_row_major)
{
  #pragma unroll
  for (int j_b = 0; j_b < MATMUL_WARP_WIDTH; j_b++) {
    // #pragma unroll
    for (int j_a = 0; j_a < MATMUL_WARP_WIDTH; j_a++) {
      wmma::fill_fragment(acc_frag_1[j_a][j_b][0], 0.0f);

      // real
      // bd
      for (int k = 0; k < MATMUL_WARP_WIDTH; k++) {
        wmma::mma_sync(acc_frag_1[j_a][j_b][0], a_frag[j_a][k][1], b_frag[k][j_b][1], acc_frag_1[j_a][j_b][0]);
      }

      // bd -> -bd
      for (int i = 0; i < acc_frag_1[j_a][j_b][0].num_elements; i++) {
        acc_frag_1[j_a][j_b][0].x[i] = -acc_frag_1[j_a][j_b][0].x[i];
      }

      // ac
      for (int k = 0; k < MATMUL_WARP_WIDTH; k++) {
        wmma::mma_sync(acc_frag_1[j_a][j_b][0], a_frag[j_a][k][0], b_frag[k][j_b][0], acc_frag_1[j_a][j_b][0]);
      }

      floatacc2bfloatacc(&acc_frag_1[j_a][j_b][0], &acc_frag_half[j_a][j_b][0]);

    }
  }

  if (output_to_shmem) {
    // #pragma unroll
    for (int j_b = 0; j_b < MATMUL_WARP_WIDTH; j_b++) {
      // #pragma unroll
      for (int j_a = 0; j_a < MATMUL_WARP_WIDTH; j_a++) {
        // does it matter where we put this?
        wmma::store_matrix_sync(
          reinterpret_cast<half *> (
            a_real_out + (out_trans ?
            j_b * WMMA_M * 1024 + j_a * WMMA_N:
            j_a * WMMA_M * 1024 + j_b * WMMA_N)
          ),
          acc_frag_half[j_a][j_b][0], 1024, out_layout
        );
      }
    }
  }
}

#endif