// Copyright (c) 2023 Dan Fu, Hermann Kumbong

#include <torch/extension.h>

#include <vector>
#include <stdio.h>
#include <mma.h>
#include <cuda_fp16.h>
#include <cub/block/block_load.cuh>
#include <cub/block/block_store.cuh>
using namespace nvcuda;

using complex_half_t = typename c10::complex<at::Half>;

#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16
// #define TILE_SIZE 4
// #define SHMEM_SIZE 256 * TILE_SIZE
// #define SEQUENCE_SIZE 256
#define WARP_SIZE 32

template <typename ALayout, typename BLayout, bool out_trans, int MATMUL_WARP_WIDTH, bool output_to_shmem>
__device__ __forceinline__ void _complex_matmul_truncated(
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
   for (int j_b = 0; j_b < MATMUL_WARP_WIDTH/2; j_b++) {
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
      for (int j_b = 0; j_b < MATMUL_WARP_WIDTH/2; j_b++) {
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




template <typename ALayout, bool a_trans, int MATMUL_WARP_WIDTH, bool a_frag_from_acc>
__device__ __forceinline__ void load_a_frag_truncated(
    half *a_real,
    half *a_imag,
    int sqrt_N,
    int N,
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_K, WMMA_N, half> acc_frag_1[MATMUL_WARP_WIDTH][MATMUL_WARP_WIDTH][2],
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_K, WMMA_N, half, ALayout> a_frag[MATMUL_WARP_WIDTH][MATMUL_WARP_WIDTH][2])
{
   int a_idx;

   if (a_frag_from_acc) {
      // load up a_frag's from acc_frag_1
      // #pragma unroll
      for (int j_a = 0; j_a < MATMUL_WARP_WIDTH/2; j_a++) {
         // #pragma unroll
         for (int j_b = 0; j_b < MATMUL_WARP_WIDTH; j_b++) {
            // #pragma unroll
            for (int k = 0; k < 2; k++) {
               for (int i = 0; i < acc_frag_1[j_a][j_b][0].num_elements; i++) {
                  a_frag[j_a][j_b][k].x[i] = acc_frag_1[j_a][j_b][k].x[i];
                  a_frag[j_a][j_b][k].x[i + acc_frag_1[j_a][j_b][k].num_elements] = acc_frag_1[j_a][j_b][k].x[i];
               }
            }
         }
      }
   } else {
      // #pragma unroll
      for (int j_a = 0; j_a < MATMUL_WARP_WIDTH/2; j_a++) {
         // #pragma unroll
         for (int k = 0; k < MATMUL_WARP_WIDTH; k++) {
            a_idx = a_trans ? k * WMMA_K * sqrt_N + j_a * WMMA_K : j_a * WMMA_K * sqrt_N + k * WMMA_K;
            wmma::load_matrix_sync(a_frag[j_a][k][0], a_real + a_idx, sqrt_N);
            wmma::load_matrix_sync(a_frag[j_a][k][1], a_imag + a_idx, sqrt_N);
         }
      }  
   }
}


template <typename ALayout, bool a_trans, int MATMUL_WARP_WIDTH, bool a_frag_from_acc>
__device__ __forceinline__ void load_b_frag_truncated(
    half *b_real,
    half *b_imag,
    int sqrt_N,
    int N,
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_K, WMMA_N, half> acc_frag_1[MATMUL_WARP_WIDTH][MATMUL_WARP_WIDTH][2],
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_K, WMMA_N, half, ALayout> b_frag[MATMUL_WARP_WIDTH][MATMUL_WARP_WIDTH][2])
{
   int b_idx;
   // #pragma unroll
   for (int j_a = 0; j_a < MATMUL_WARP_WIDTH/2; j_a++) {
      // #pragma unroll
      for (int k = 0; k < MATMUL_WARP_WIDTH; k++) {
         b_idx = a_trans ? k * WMMA_K * sqrt_N + j_a * WMMA_K : j_a * WMMA_K * sqrt_N + k * WMMA_K;
         wmma::load_matrix_sync(b_frag[j_a][k][0], b_real + b_idx, sqrt_N);
         wmma::load_matrix_sync(b_frag[j_a][k][1], b_imag + b_idx, sqrt_N);
      }
   }  
}


template <typename ALayout, typename BLayout, bool a_trans, bool out_trans, int MATMUL_WARP_WIDTH, bool a_frag_from_acc, bool output_to_shmem>
__device__ __forceinline__ void complex_matmul_truncated(
    half *a_real,
    half *a_imag,
    int sqrt_N,
    int N,
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_K, WMMA_N, half, BLayout> b_frag[MATMUL_WARP_WIDTH][MATMUL_WARP_WIDTH][2],
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_K, WMMA_N, half> acc_frag_1[MATMUL_WARP_WIDTH][MATMUL_WARP_WIDTH][2],
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_K, WMMA_N, half, ALayout> k_frag[MATMUL_WARP_WIDTH][MATMUL_WARP_WIDTH][2],
    wmma::layout_t out_layout = wmma::mem_row_major)
{

   wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_K, WMMA_N, half, ALayout> a_frag [MATMUL_WARP_WIDTH][MATMUL_WARP_WIDTH][2];
   load_a_frag_truncated<ALayout, a_trans, MATMUL_WARP_WIDTH, a_frag_from_acc>(a_real, a_imag, sqrt_N, N, acc_frag_1, a_frag);

   // __syncthreads();
   // multiply a_frag by k_frag
   for (int j_a = 0; j_a < MATMUL_WARP_WIDTH/2; j_a++) {
      for (int k = 0; k < MATMUL_WARP_WIDTH; k++) {
         for (int i = 0; i < acc_frag_1[j_a][k][0].num_elements / 2; i++) {
            complex_mul_half2(
               __half2(a_frag[j_a][k][0].x[2 * i], a_frag[j_a][k][0].x[2 * i + 1]),
               __half2(a_frag[j_a][k][1].x[2 * i], a_frag[j_a][k][1].x[2 * i + 1]),
               __half2(k_frag[j_a][k][0].x[2 * i], k_frag[j_a][k][0].x[2 * i + 1]),
               __half2(k_frag[j_a][k][1].x[2 * i], k_frag[j_a][k][1].x[2 * i + 1]),
               &a_frag[j_a][k][0].x[2 * i], 
               &a_frag[j_a][k][1].x[2 * i],
               &a_frag[j_a][k][0].x[2 * i + 1],
               &a_frag[j_a][k][1].x[2 * i + 1]
            );
         }
      }
   }

   _complex_matmul_truncated<ALayout, BLayout, out_trans, MATMUL_WARP_WIDTH, output_to_shmem>(a_real, a_imag, sqrt_N, N, a_frag, b_frag, acc_frag_1, out_layout);
}

template <typename ALayout, typename BLayout, bool a_trans, bool out_trans, int MATMUL_WARP_WIDTH, bool a_frag_from_acc, bool output_to_shmem>
__device__ __forceinline__ void complex_matmul_truncated(
    half *a_real,
    half *a_imag,
    int sqrt_N,
    int N,
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_K, WMMA_N, half, BLayout> b_frag[MATMUL_WARP_WIDTH][MATMUL_WARP_WIDTH][2],
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_K, WMMA_N, half> acc_frag_1[MATMUL_WARP_WIDTH][MATMUL_WARP_WIDTH][2],
    wmma::layout_t out_layout = wmma::mem_row_major)
{
   wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_K, WMMA_N, half, ALayout> a_frag[MATMUL_WARP_WIDTH][MATMUL_WARP_WIDTH][2];
   load_a_frag_truncated<ALayout, a_trans, MATMUL_WARP_WIDTH, a_frag_from_acc>(a_real, a_imag, sqrt_N, N, acc_frag_1, a_frag);

   // __syncthreads();
   _complex_matmul_truncated<ALayout, BLayout, out_trans, MATMUL_WARP_WIDTH, output_to_shmem>(a_real, a_imag, sqrt_N, N, a_frag, b_frag, acc_frag_1, out_layout);
}


template <typename ALayout, typename BLayout, bool b_trans, bool out_trans, int MATMUL_WARP_WIDTH, bool b_frag_from_acc, bool output_to_shmem>
__device__ __forceinline__ void complex_matmul_load_b_truncated(
    half *b_real,
    half *b_imag,
    int sqrt_N,
    int N,
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_K, WMMA_N, half, ALayout> a_frag[MATMUL_WARP_WIDTH][MATMUL_WARP_WIDTH][2],
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_K, WMMA_N, half> acc_frag_1[MATMUL_WARP_WIDTH][MATMUL_WARP_WIDTH][2],
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_K, WMMA_N, half, BLayout> k_frag[MATMUL_WARP_WIDTH][MATMUL_WARP_WIDTH][2],
    wmma::layout_t out_layout = wmma::mem_row_major)
{
   wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_K, WMMA_N, half, BLayout> b_frag[MATMUL_WARP_WIDTH][MATMUL_WARP_WIDTH][2];
   load_b_frag_truncated<BLayout, b_trans, MATMUL_WARP_WIDTH, b_frag_from_acc>(b_real, b_imag, sqrt_N, N, acc_frag_1, b_frag);

   // __syncthreads();
   // multiply b_frag by k_frag
   for (int j_a = 0; j_a < MATMUL_WARP_WIDTH/2; j_a++) {
      for (int k = 0; k < MATMUL_WARP_WIDTH; k++) {
         for (int i = 0; i < acc_frag_1[j_a][k][0].num_elements / 2; i++) {
            complex_mul_half2(
               __half2(b_frag[j_a][k][0].x[2 * i], b_frag[j_a][k][0].x[2 * i + 1]),
               __half2(b_frag[j_a][k][1].x[2 * i], b_frag[j_a][k][1].x[2 * i + 1]),
               __half2(k_frag[j_a][k][0].x[2 * i], k_frag[j_a][k][0].x[2 * i + 1]),
               __half2(k_frag[j_a][k][1].x[2 * i], k_frag[j_a][k][1].x[2 * i + 1]),
               &b_frag[j_a][k][0].x[2 * i], 
               &b_frag[j_a][k][1].x[2 * i],
               &b_frag[j_a][k][0].x[2 * i + 1],
               &b_frag[j_a][k][1].x[2 * i + 1]
            );
         }
      }
   }

   // __syncthreads();
   _complex_matmul_truncated<ALayout, BLayout, out_trans, MATMUL_WARP_WIDTH, output_to_shmem>(b_real, b_imag, sqrt_N, N, a_frag, b_frag, acc_frag_1, out_layout);
}