// Copyright (c) 2023 Dan Fu, Hermann Kumbong

#include <torch/extension.h>

#include <vector>
#include <stdio.h>
#include <mma.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cub/block/block_load.cuh>
#include <cub/block/block_store.cuh>
using namespace nvcuda;

using complex_bfloat16_t = typename c10::complex<at::BFloat16>;

#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16
// #define TILE_SIZE 4
// #define SHMEM_SIZE 256 * TILE_SIZE
// #define SEQUENCE_SIZE 256
#define WARP_SIZE 32


#ifndef MONARCH_CUDA_BF16_
#define MONARCH_CUDA_BF16_

template <typename ALayout, typename BLayout, bool out_trans, int MATMUL_WARP_WIDTH, bool output_to_shmem>
__device__ __forceinline__ void _complex_matmul(
    float *scratch_real,
    float *scratch_imag,
    int sqrt_N,
    int N,
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_K, WMMA_N, __nv_bfloat16, ALayout> a_frag[MATMUL_WARP_WIDTH][MATMUL_WARP_WIDTH][2],
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_K, WMMA_N, __nv_bfloat16, BLayout> b_frag[MATMUL_WARP_WIDTH][MATMUL_WARP_WIDTH][2],
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_K, WMMA_N, float> acc_frag_1[MATMUL_WARP_WIDTH][MATMUL_WARP_WIDTH][2],
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

      }
   }

   if (output_to_shmem) {
      // #pragma unroll
      for (int j_b = 0; j_b < MATMUL_WARP_WIDTH; j_b++) {
         // #pragma unroll
         for (int j_a = 0; j_a < MATMUL_WARP_WIDTH; j_a++) {
            // does it matter where we put this?
            wmma::store_matrix_sync(
               scratch_real + (out_trans ?
               j_b * WMMA_M * sqrt_N + j_a * WMMA_N:
               j_a * WMMA_M * sqrt_N + j_b * WMMA_N),
               acc_frag_1[j_a][j_b][0], sqrt_N, out_layout
            );
   
            wmma::store_matrix_sync(
               scratch_imag + (out_trans ?
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
    float *scratch_real,
    float *scratch_imag,
    int sqrt_N,
    int N,
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_K, WMMA_N, __nv_bfloat16, ALayout> a_frag[MATMUL_WARP_WIDTH][MATMUL_WARP_WIDTH][2],
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_K, WMMA_N, __nv_bfloat16, BLayout> b_frag[MATMUL_WARP_WIDTH][MATMUL_WARP_WIDTH][2],
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_K, WMMA_N, float> acc_frag_1[MATMUL_WARP_WIDTH][MATMUL_WARP_WIDTH][2],
    wmma::layout_t out_layout = wmma::mem_row_major
    )
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

         wmma::fill_fragment(acc_frag_1[j_a][j_b][1], 0.0f);

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
            //does it matter where we put this?
            wmma::store_matrix_sync(
               scratch_real + (out_trans ?
               j_b * WMMA_M * sqrt_N + j_a * WMMA_N:
               j_a * WMMA_M * sqrt_N + j_b * WMMA_N),
               acc_frag_1[j_a][j_b][0], sqrt_N, out_layout
            );
   
            wmma::store_matrix_sync(
               scratch_imag + (out_trans ?
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
    float* scratch_real,
    float* scratch_imag,
    int sqrt_N,
    int N,
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_K, WMMA_N, __nv_bfloat16, ALayout> a_frag[MATMUL_WARP_WIDTH][MATMUL_WARP_WIDTH][2],
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_K, WMMA_N, __nv_bfloat16, BLayout> b_frag[MATMUL_WARP_WIDTH][MATMUL_WARP_WIDTH][2],
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_K, WMMA_N, float> acc_frag_1[MATMUL_WARP_WIDTH][MATMUL_WARP_WIDTH][2],
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

         wmma::fill_fragment(acc_frag_1[j_a][j_b][1], 0.0f);

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
            
            //does it matter where we put this?
            wmma::store_matrix_sync(
               scratch_real + (out_trans ?
               j_b * WMMA_M * sqrt_N + j_a * WMMA_N:
               j_a * WMMA_M * sqrt_N + j_b * WMMA_N),
               acc_frag_1[j_a][j_b][0], sqrt_N, out_layout
            );
   
            wmma::store_matrix_sync(
               scratch_imag + (out_trans ?
               j_b * WMMA_M * sqrt_N + j_a * WMMA_N:
               j_a * WMMA_M * sqrt_N + j_b * WMMA_N),
               acc_frag_1[j_a][j_b][1], sqrt_N, out_layout
            );
         }
      }
   }
}

// template <typename ALayout, typename BLayout, bool out_trans, int MATMUL_WARP_WIDTH, bool output_to_shmem>
// __device__ __forceinline__ void _complex_matmul_r2c_256(
//     float *scratch_real,
//     float *scratch_imag,
//     int sqrt_N,
//     int N,
//     wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_K, WMMA_N, __nv_bfloat16, ALayout> a_frag[MATMUL_WARP_WIDTH][MATMUL_WARP_WIDTH][2],
//     wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_K, WMMA_N, __nv_bfloat16, BLayout> b_frag[MATMUL_WARP_WIDTH][MATMUL_WARP_WIDTH][2],
//     wmma::fragment<wmma::accumulator, WMMA_M, WMMA_K, WMMA_N, float> acc_frag_1[MATMUL_WARP_WIDTH][MATMUL_WARP_WIDTH][2],
//     wmma::layout_t out_layout = wmma::mem_row_major
//     )
// {
//    // #pragma unroll
//    for (int j_b = 0; j_b < MATMUL_WARP_WIDTH; j_b++) {
//       // #pragma unroll
//       for (int j_a = 0; j_a < MATMUL_WARP_WIDTH; j_a++) {
//          wmma::fill_fragment(acc_frag_1[j_a][j_b][0], 0.0f);

//          // real

//          // ac
//          for (int k = 0; k < MATMUL_WARP_WIDTH; k++) {
//             wmma::mma_sync(acc_frag_1[j_a][j_b][0], a_frag[j_a][k][0], b_frag[k][j_b][0], acc_frag_1[j_a][j_b][0]);
//          }

//          wmma::fill_fragment(acc_frag_1[j_a][j_b][1], 0.0f);

//          // imag
//          // ad
//          for (int k = 0; k < MATMUL_WARP_WIDTH; k++) {
//             wmma::mma_sync(acc_frag_1[j_a][j_b][1], a_frag[j_a][k][0], b_frag[k][j_b][1], acc_frag_1[j_a][j_b][1]);
//          }

//       }
//    }

//    if (output_to_shmem) {
//       // #pragma unroll
//       for (int j_b = 0; j_b < MATMUL_WARP_WIDTH; j_b++) {
//          // #pragma unroll
//          for (int j_a = 0; j_a < MATMUL_WARP_WIDTH; j_a++) {
//             //accumlator fragments are not supporte for bfloat16, so we cannot directly cast or store the values to shared memory
//             //of type bfloat 16. We need to move the values to the a_fragment which supports bfloat16 and then store it to shared memory
//             //does it matter where we put this?
//             wmma::store_matrix_sync(
//                scratch_real + (out_trans ?
//                j_b * WMMA_M * sqrt_N + j_a * WMMA_N:
//                j_a * WMMA_M * sqrt_N + j_b * WMMA_N),
//                acc_frag_1[j_a][j_b][0], sqrt_N, out_layout
//             );
   
//             wmma::store_matrix_sync(
//                scratch_imag + (out_trans ?
//                j_b * WMMA_M * sqrt_N + j_a * WMMA_N:
//                j_a * WMMA_M * sqrt_N + j_b * WMMA_N),
//                acc_frag_1[j_a][j_b][1], sqrt_N, out_layout
//             );
//          }
//       }
//    }
// }

template <typename ALayout, typename BLayout, bool out_trans, int MATMUL_WARP_WIDTH, bool output_to_shmem>
__device__ __forceinline__ void _complex_matmul_c2r(
    float *scratch_real,
    int sqrt_N,
    int N,
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_K, WMMA_N, __nv_bfloat16, ALayout> a_frag[MATMUL_WARP_WIDTH][MATMUL_WARP_WIDTH][2],
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_K, WMMA_N, __nv_bfloat16, BLayout> b_frag[MATMUL_WARP_WIDTH][MATMUL_WARP_WIDTH][2],
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_K, WMMA_N, float> acc_frag_1[MATMUL_WARP_WIDTH][MATMUL_WARP_WIDTH][2],
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
               scratch_real + (out_trans ?
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
    float *scratch_real,
    int sqrt_N,
    int N,
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_K, WMMA_N, __nv_bfloat16, ALayout> a_frag[MATMUL_WARP_WIDTH][MATMUL_WARP_WIDTH][2],
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_K, WMMA_N, __nv_bfloat16, BLayout> b_frag[MATMUL_WARP_WIDTH][MATMUL_WARP_WIDTH][2],
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_K, WMMA_N, float> acc_frag_1[MATMUL_WARP_WIDTH][MATMUL_WARP_WIDTH][2],
    wmma::layout_t out_layout = wmma::mem_row_major
    )
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

      }
   }

   if (output_to_shmem) {
      // #pragma unroll
      for (int j_b = 0; j_b < MATMUL_WARP_WIDTH; j_b++) {
         // #pragma unroll
         for (int j_a = 0; j_a < MATMUL_WARP_WIDTH; j_a++) {
            //does it matter where we put this?
            wmma::store_matrix_sync(
               scratch_real + (out_trans ?
               j_b * WMMA_M * sqrt_N + j_a * WMMA_N:
               j_a * WMMA_M * sqrt_N + j_b * WMMA_N),
               acc_frag_1[j_a][j_b][0], sqrt_N, out_layout
            );
         }
      }
   }
}

template <typename ALayout, bool a_trans, int MATMUL_WARP_WIDTH, bool a_frag_from_acc>
__device__ __forceinline__ void load_a_frag(
    float *scratch_real,
    float *scratch_imag,
    int sqrt_N,
    int N,
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_K, WMMA_N, float> acc_frag_1[MATMUL_WARP_WIDTH][MATMUL_WARP_WIDTH][2],
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_K, WMMA_N, __nv_bfloat16, ALayout> a_frag[MATMUL_WARP_WIDTH][MATMUL_WARP_WIDTH][2])
{
   int a_idx;

   if (a_frag_from_acc) {
      // load up a_frag's from acc_frag_1
      // #pragma unroll
      for (int j_a = 0; j_a < MATMUL_WARP_WIDTH; j_a++) {
         // #pragma unroll
         for (int j_b = 0; j_b < MATMUL_WARP_WIDTH; j_b++) {
            // #pragma unroll
            for (int k = 0; k < 2; k++) {
               for (int i = 0; i < acc_frag_1[j_a][j_b][k].num_elements; i++) {
                  a_frag[j_a][j_b][k].x[i] = __float2bfloat16(acc_frag_1[j_a][j_b][k].x[i]);
                  a_frag[j_a][j_b][k].x[i + acc_frag_1[j_a][j_b][k].num_elements] = __float2bfloat16(acc_frag_1[j_a][j_b][k].x[i]);
               }
            }
         }
      }
   } else {
      // #pragma unroll
      __nv_bfloat16 tmp_real[2048];
      __nv_bfloat16 tmp_imag[2048];

      for(int i = 0; i < N; i++) {
         tmp_real[i] = __float2bfloat16(scratch_real[i]);
         tmp_imag[i] = __float2bfloat16(scratch_imag[i]);
      }

      __syncthreads();

      for (int j_a = 0; j_a < MATMUL_WARP_WIDTH; j_a++) {
         // #pragma unroll
         for (int k = 0; k < MATMUL_WARP_WIDTH; k++) {
            a_idx = a_trans ? k * WMMA_K * sqrt_N + j_a * WMMA_K : j_a * WMMA_K * sqrt_N + k * WMMA_K;
            wmma::load_matrix_sync(a_frag[j_a][k][0], tmp_real + a_idx, sqrt_N);
            wmma::load_matrix_sync(a_frag[j_a][k][1], tmp_imag + a_idx, sqrt_N);
         }
      }  
   }
}

// template <typename ALayout, bool a_trans, int MATMUL_WARP_WIDTH, bool a_frag_from_acc>
// __device__ __forceinline__ void load_a_frag_256(
//     float *scratch_real,
//     float *scratch_imag,
//     int sqrt_N,
//     int N,
//     wmma::fragment<wmma::accumulator, WMMA_M, WMMA_K, WMMA_N, float> acc_frag_1[MATMUL_WARP_WIDTH][MATMUL_WARP_WIDTH][2],
//     wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_K, WMMA_N, __nv_bfloat16, ALayout> a_frag[MATMUL_WARP_WIDTH][MATMUL_WARP_WIDTH][2])
// {
//    int a_idx;

//    if (a_frag_from_acc) {
//       // load up a_frag's from acc_frag_1
//       // #pragma unroll
//       for (int j_a = 0; j_a < MATMUL_WARP_WIDTH; j_a++) {
//          // #pragma unroll
//          for (int j_b = 0; j_b < MATMUL_WARP_WIDTH; j_b++) {
//             // #pragma unroll
//             for (int k = 0; k < 2; k++) {
//                for (int i = 0; i < acc_frag_1[j_a][j_b][0].num_elements; i++) {
//                   a_frag[j_a][j_b][k].x[i] = __float2bfloat16(acc_frag_1[j_a][j_b][k].x[i]);
//                   a_frag[j_a][j_b][k].x[i + acc_frag_1[j_a][j_b][k].num_elements] = __float2bfloat16(acc_frag_1[j_a][j_b][k].x[i]);
//                }
//             }
//          }
//       }
//    } else {
//       // #pragma unroll
//       for (int j_a = 0; j_a < MATMUL_WARP_WIDTH; j_a++) {
//          // #pragma unroll
//          for (int k = 0; k < MATMUL_WARP_WIDTH; k++) {
//             a_idx = a_trans ? k * WMMA_K * sqrt_N + j_a * WMMA_K : j_a * WMMA_K * sqrt_N + k * WMMA_K;
//             wmma::load_matrix_sync(a_frag[j_a][k][0], reinterpret_cast<__nv_bfloat16*>(scratch_real) + a_idx, 256);
//             wmma::load_matrix_sync(a_frag[j_a][k][1], reinterpret_cast<__nv_bfloat16*>(scratch_imag) + a_idx, 256);
//          }
//       }  
//    }
// }

template <typename ALayout, bool a_trans, int MATMUL_WARP_WIDTH, bool a_frag_from_acc>
__device__ __forceinline__ void load_b_frag_r2c(
    const __nv_bfloat16* b_real,
    int sqrt_N,
    int N,
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_K, WMMA_N, float> acc_frag_1[MATMUL_WARP_WIDTH][MATMUL_WARP_WIDTH][2],
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_K, WMMA_N, __nv_bfloat16, ALayout> b_frag[MATMUL_WARP_WIDTH][MATMUL_WARP_WIDTH][2])
{
   int b_idx;
   // #pragma unroll
   for (int j_a = 0; j_a < MATMUL_WARP_WIDTH; j_a++) {
      // #pragma unroll
      for (int k = 0; k < MATMUL_WARP_WIDTH; k++) {
         b_idx = a_trans ? k * WMMA_K * sqrt_N + j_a * WMMA_K : j_a * WMMA_K * sqrt_N + k * WMMA_K;
         wmma::load_matrix_sync(b_frag[j_a][k][0], b_real + b_idx, sqrt_N);
      }
   }  
}

// template <typename ALayout, bool a_trans, int MATMUL_WARP_WIDTH, bool a_frag_from_acc>
// __device__ __forceinline__ void load_b_frag(
//     float* scratch_real,
//     float* scratch_imag,
//     int sqrt_N,
//     int N,
//     wmma::fragment<wmma::accumulator, WMMA_M, WMMA_K, WMMA_N, float> acc_frag_1[MATMUL_WARP_WIDTH][MATMUL_WARP_WIDTH][2],
//     wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_K, WMMA_N, __nv_bfloat16, ALayout> b_frag[MATMUL_WARP_WIDTH][MATMUL_WARP_WIDTH][2])
// {
//    int b_idx;
//    // #pragma unroll
//    for (int j_a = 0; j_a < MATMUL_WARP_WIDTH; j_a++) {
//       // #pragma unroll
//       for (int k = 0; k < MATMUL_WARP_WIDTH; k++) {
//          b_idx = a_trans ? k * WMMA_K * sqrt_N + j_a * WMMA_K : j_a * WMMA_K * sqrt_N + k * WMMA_K;
//          wmma::load_matrix_sync(b_frag[j_a][k][0], b_real + b_idx, sqrt_N);
//          wmma::load_matrix_sync(b_frag[j_a][k][1], b_imag + b_idx, sqrt_N);
//       }
//    }  
// }

template <typename ALayout, bool a_trans, int MATMUL_WARP_WIDTH, bool a_frag_from_acc>
__device__ __forceinline__ void load_a_frag_r2c(
    const __nv_bfloat16 *a_real,
    int sqrt_N,
    int N,
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_K, WMMA_N, float> acc_frag_1[MATMUL_WARP_WIDTH][MATMUL_WARP_WIDTH][2],
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_K, WMMA_N, __nv_bfloat16, ALayout> a_frag[MATMUL_WARP_WIDTH][MATMUL_WARP_WIDTH][2])
{
   int a_idx;

   if (a_frag_from_acc) {
      // load up a_frag's from acc_frag_1
      // #pragma unroll
      for (int j_a = 0; j_a < MATMUL_WARP_WIDTH; j_a++) {
         // #pragma unroll
         for (int j_b = 0; j_b < MATMUL_WARP_WIDTH; j_b++) {
            // #pragma unroll
            for (int k = 0; k < 1; k++) {
               // #pragma unroll
               for (int i = 0; i < acc_frag_1[j_a][j_b][0].num_elements; i++) {
                  a_frag[j_a][j_b][k].x[i] = __float2bfloat16(acc_frag_1[j_a][j_b][k].x[i]);
                  a_frag[j_a][j_b][k].x[i + acc_frag_1[j_a][j_b][k].num_elements] = __float2bfloat16(acc_frag_1[j_a][j_b][k].x[i]);
               }
            }
         }
      }
   } else {
      // #pragma unroll
      for (int j_a = 0; j_a < MATMUL_WARP_WIDTH; j_a++) {
         // #pragma unroll
         for (int k = 0; k < MATMUL_WARP_WIDTH; k++) {
            a_idx = a_trans ? k * WMMA_K * sqrt_N + j_a * WMMA_K : j_a * WMMA_K * sqrt_N + k * WMMA_K;
            wmma::load_matrix_sync(a_frag[j_a][k][0], a_real + a_idx, sqrt_N);
         }
      }  
   }
}

// template <typename ALayout, bool a_trans, int MATMUL_WARP_WIDTH, bool a_frag_from_acc>
// __device__ __forceinline__ void load_a_frag_r2c_256(
//     const __nv_bfloat16 *a_real,
//     int sqrt_N,
//     int N,
//     wmma::fragment<wmma::accumulator, WMMA_M, WMMA_K, WMMA_N, float> acc_frag_1[MATMUL_WARP_WIDTH][MATMUL_WARP_WIDTH][2],
//     wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_K, WMMA_N, __nv_bfloat16, ALayout> a_frag[MATMUL_WARP_WIDTH][MATMUL_WARP_WIDTH][2])
// {
//    int a_idx;

//    if (a_frag_from_acc) {
//       // load up a_frag's from acc_frag_1
//       // #pragma unroll
//       for (int j_a = 0; j_a < MATMUL_WARP_WIDTH; j_a++) {
//          // #pragma unroll
//          for (int j_b = 0; j_b < MATMUL_WARP_WIDTH; j_b++) {
//             // #pragma unroll
//             for (int k = 0; k < 1; k++) {
//                // #pragma unroll
//                for (int i = 0; i < acc_frag_1[j_a][j_b][0].num_elements; i++) {
//                   a_frag[j_a][j_b][k].x[i] = __float2bfloat16(acc_frag_1[j_a][j_b][k].x[i]);
//                   a_frag[j_a][j_b][k].x[i + acc_frag_1[j_a][j_b][k].num_elements] = __float2bfloat16(acc_frag_1[j_a][j_b][k].x[i]);
//                }
//             }
//          }
//       }
//    } else {
//       // #pragma unroll
//       for (int j_a = 0; j_a < MATMUL_WARP_WIDTH; j_a++) {
//          // #pragma unroll
//          for (int k = 0; k < MATMUL_WARP_WIDTH; k++) {
//             a_idx = a_trans ? k * WMMA_K * sqrt_N + j_a * WMMA_K : j_a * WMMA_K * sqrt_N + k * WMMA_K;
//             wmma::load_matrix_sync(a_frag[j_a][k][0], reinterpret_cast<__nv_bfloat16 *>(a_real) + a_idx, 256);
//          }
//       }  
//    }
// }

template <typename ALayout, typename BLayout, bool a_trans, bool out_trans, int MATMUL_WARP_WIDTH, bool a_frag_from_acc, bool output_to_shmem>
__device__ __forceinline__ void complex_matmul(
    float *scratch_real,
    float *scratch_imag,
    int sqrt_N,
    int N,
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_K, WMMA_N, __nv_bfloat16, BLayout> b_frag[MATMUL_WARP_WIDTH][MATMUL_WARP_WIDTH][2],
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_K, WMMA_N, float> acc_frag_1[MATMUL_WARP_WIDTH][MATMUL_WARP_WIDTH][2],
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_K, WMMA_N, __nv_bfloat16, ALayout> k_frag[MATMUL_WARP_WIDTH][MATMUL_WARP_WIDTH][2],
    wmma::layout_t out_layout = wmma::mem_row_major)
{

   wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_K, WMMA_N, __nv_bfloat16, ALayout> a_frag [MATMUL_WARP_WIDTH][MATMUL_WARP_WIDTH][2];
   load_a_frag<ALayout, a_trans, MATMUL_WARP_WIDTH, a_frag_from_acc>(scratch_real, scratch_imag, sqrt_N, N, acc_frag_1, a_frag);

   // __syncthreads();
   // multiply a_frag by k_frag
   for (int j_a = 0; j_a < MATMUL_WARP_WIDTH; j_a++) {
      for (int k = 0; k < MATMUL_WARP_WIDTH; k++) {
         for (int i = 0; i < acc_frag_1[j_a][k][0].num_elements / 2; i++) {
            complex_mul_bfloat162(
               __nv_bfloat162(a_frag[j_a][k][0].x[2 * i], a_frag[j_a][k][0].x[2 * i + 1]),
               __nv_bfloat162(a_frag[j_a][k][1].x[2 * i], a_frag[j_a][k][1].x[2 * i + 1]),
               __nv_bfloat162(k_frag[j_a][k][0].x[2 * i], k_frag[j_a][k][0].x[2 * i + 1]),
               __nv_bfloat162(k_frag[j_a][k][1].x[2 * i], k_frag[j_a][k][1].x[2 * i + 1]),
               &a_frag[j_a][k][0].x[2 * i], 
               &a_frag[j_a][k][1].x[2 * i],
               &a_frag[j_a][k][0].x[2 * i + 1],
               &a_frag[j_a][k][1].x[2 * i + 1]
            );
         }
      }
   }

   _complex_matmul<ALayout, BLayout, out_trans, MATMUL_WARP_WIDTH, output_to_shmem>(scratch_real, scratch_imag, sqrt_N, N, a_frag, b_frag, acc_frag_1, out_layout);
}

template <typename ALayout, typename BLayout, bool a_trans, bool out_trans, int MATMUL_WARP_WIDTH, bool a_frag_from_acc, bool output_to_shmem>
__device__ __forceinline__ void complex_matmul(
    float *scratch_real,
    float *scratch_imag,
    int sqrt_N,
    int N,
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_K, WMMA_N, __nv_bfloat16, BLayout> b_frag[MATMUL_WARP_WIDTH][MATMUL_WARP_WIDTH][2],
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_K, WMMA_N, float> acc_frag_1[MATMUL_WARP_WIDTH][MATMUL_WARP_WIDTH][2],
    wmma::layout_t out_layout = wmma::mem_row_major)
{
   wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_K, WMMA_N, __nv_bfloat16, ALayout> a_frag[MATMUL_WARP_WIDTH][MATMUL_WARP_WIDTH][2];
   load_a_frag<ALayout, a_trans, MATMUL_WARP_WIDTH, a_frag_from_acc>(scratch_real, scratch_imag, sqrt_N, N, acc_frag_1, a_frag);

   // __syncthreads();
   _complex_matmul<ALayout, BLayout, out_trans, MATMUL_WARP_WIDTH, output_to_shmem>(scratch_real, scratch_imag, sqrt_N, N, a_frag, b_frag, acc_frag_1, out_layout);
}

// template <typename ALayout, typename BLayout, bool b_trans, bool out_trans, int MATMUL_WARP_WIDTH, bool b_frag_from_acc, bool output_to_shmem>
// __device__ __forceinline__ void complex_matmul_load_b(
//     float* scratch_real,
//     float* scratch_imag,
//     int sqrt_N,
//     int N,
//     wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_K, WMMA_N, __nv_bfloat16, ALayout> a_frag[MATMUL_WARP_WIDTH][MATMUL_WARP_WIDTH][2],
//     wmma::fragment<wmma::accumulator, WMMA_M, WMMA_K, WMMA_N, float> acc_frag_1[MATMUL_WARP_WIDTH][MATMUL_WARP_WIDTH][2],
//     wmma::layout_t out_layout = wmma::mem_row_major)
// {
//    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_K, WMMA_N, __nv_bfloat16, BLayout> b_frag[MATMUL_WARP_WIDTH][MATMUL_WARP_WIDTH][2];
//    load_b_frag<BLayout, b_trans, MATMUL_WARP_WIDTH, b_frag_from_acc>(b_real, b_imag, sqrt_N, N, acc_frag_1, b_frag);

//    // __syncthreads();
//    _complex_matmul<ALayout, BLayout, out_trans, MATMUL_WARP_WIDTH, output_to_shmem>(b_real, b_imag, sqrt_N, N, a_frag, b_frag, acc_frag_1, out_layout);
// }

// template <typename ALayout, typename BLayout, bool b_trans, bool out_trans, int MATMUL_WARP_WIDTH, bool b_frag_from_acc, bool output_to_shmem>
// __device__ __forceinline__ void complex_matmul_load_b(
//     float* b_real,
//     float* b_imag,
//     int sqrt_N,
//     int N,
//     wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_K, WMMA_N, __nv_bfloat16, ALayout> a_frag[MATMUL_WARP_WIDTH][MATMUL_WARP_WIDTH][2],
//     wmma::fragment<wmma::accumulator, WMMA_M, WMMA_K, WMMA_N, float> acc_frag_1[MATMUL_WARP_WIDTH][MATMUL_WARP_WIDTH][2],
//     wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_K, WMMA_N, __nv_bfloat16, BLayout> k_frag[MATMUL_WARP_WIDTH][MATMUL_WARP_WIDTH][2],
//     wmma::layout_t out_layout = wmma::mem_row_major)
// {
//    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_K, WMMA_N, __nv_bfloat16, BLayout> b_frag[MATMUL_WARP_WIDTH][MATMUL_WARP_WIDTH][2];
//    load_b_frag<BLayout, b_trans, MATMUL_WARP_WIDTH, b_frag_from_acc>(b_real, b_imag, sqrt_N, N, acc_frag_1, b_frag);

//    // __syncthreads();
//    // multiply b_frag by k_frag
//    for (int j_a = 0; j_a < MATMUL_WARP_WIDTH; j_a++) {
//       for (int k = 0; k < MATMUL_WARP_WIDTH; k++) {
//          for (int i = 0; i < acc_frag_1[j_a][k][0].num_elements / 2; i++) {
//             complex_mul_bfloat162(
//                __nv_bfloat162(b_frag[j_a][k][0].x[2 * i], b_frag[j_a][k][0].x[2 * i + 1]),
//                __nv_bfloat162(b_frag[j_a][k][1].x[2 * i], b_frag[j_a][k][1].x[2 * i + 1]),
//                __nv_bfloat162(k_frag[j_a][k][0].x[2 * i], k_frag[j_a][k][0].x[2 * i + 1]),
//                __nv_bfloat162(k_frag[j_a][k][1].x[2 * i], k_frag[j_a][k][1].x[2 * i + 1]),
//                &b_frag[j_a][k][0].x[2 * i], 
//                &b_frag[j_a][k][1].x[2 * i],
//                &b_frag[j_a][k][0].x[2 * i + 1],
//                &b_frag[j_a][k][1].x[2 * i + 1]
//             );
//          }
//       }
//    }

//    // __syncthreads();
//    _complex_matmul<ALayout, BLayout, out_trans, MATMUL_WARP_WIDTH, output_to_shmem>(b_real, b_imag, sqrt_N, N, a_frag, b_frag, acc_frag_1, out_layout);
// }

template <typename ALayout, typename BLayout, bool a_trans, bool out_trans, int MATMUL_WARP_WIDTH, bool a_frag_from_acc, bool output_to_shmem>
__device__ __forceinline__ void complex_matmul_r2c(
    const __nv_bfloat16 *a_real_input,
    float *scratch_real,
    float *scratch_imag,
    int sqrt_N,
    int N,
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_K, WMMA_N, __nv_bfloat16, BLayout> b_frag[MATMUL_WARP_WIDTH][MATMUL_WARP_WIDTH][2],
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_K, WMMA_N, float> acc_frag_1[MATMUL_WARP_WIDTH][MATMUL_WARP_WIDTH][2],
    wmma::layout_t out_layout = wmma::mem_row_major)
{
   wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_K, WMMA_N, __nv_bfloat16, ALayout> a_frag[MATMUL_WARP_WIDTH][MATMUL_WARP_WIDTH][2];
   load_a_frag_r2c<ALayout, a_trans, MATMUL_WARP_WIDTH, a_frag_from_acc>(a_real_input, sqrt_N, N, acc_frag_1, a_frag);

   _complex_matmul_r2c<ALayout, BLayout, out_trans, MATMUL_WARP_WIDTH, output_to_shmem>(scratch_real, scratch_imag, sqrt_N, N, a_frag, b_frag, acc_frag_1, out_layout);
}

template <typename ALayout, typename BLayout, bool b_trans, bool out_trans, int MATMUL_WARP_WIDTH, bool b_frag_from_acc, bool output_to_shmem>
__device__ __forceinline__ void complex_matmul_r2c_load_b(
    const __nv_bfloat16 *b_real_input,
    float* scratch_real,
    float* scratch_imag,
    int sqrt_N,
    int N,
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_K, WMMA_N, __nv_bfloat16, ALayout> a_frag[MATMUL_WARP_WIDTH][MATMUL_WARP_WIDTH][2],
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_K, WMMA_N, float> acc_frag_1[MATMUL_WARP_WIDTH][MATMUL_WARP_WIDTH][2],
    wmma::layout_t out_layout = wmma::mem_row_major)
{
   wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_K, WMMA_N, __nv_bfloat16, BLayout> b_frag[MATMUL_WARP_WIDTH][MATMUL_WARP_WIDTH][2];
   load_b_frag_r2c<BLayout, b_trans, MATMUL_WARP_WIDTH, b_frag_from_acc>(b_real_input, sqrt_N, N, acc_frag_1, b_frag);

   _complex_matmul_r2c_load_b<ALayout, BLayout, out_trans, MATMUL_WARP_WIDTH, output_to_shmem>(scratch_real, scratch_imag, sqrt_N, N, a_frag, b_frag, acc_frag_1, out_layout);
}

// template <typename ALayout, typename BLayout, bool a_trans, bool out_trans, int MATMUL_WARP_WIDTH, bool a_frag_from_acc, bool output_to_shmem>
// __device__ __forceinline__ void complex_matmul_r2c_256(
//     const __nv_bfloat16 *a_real_input,
//     float *scratch_real,
//     float *scratch_imag,
//     int sqrt_N,
//     int N,
//     wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_K, WMMA_N, __nv_bfloat16, BLayout> b_frag[MATMUL_WARP_WIDTH][MATMUL_WARP_WIDTH][2],
//     wmma::fragment<wmma::accumulator, WMMA_M, WMMA_K, WMMA_N, float> acc_frag_1[MATMUL_WARP_WIDTH][MATMUL_WARP_WIDTH][2],
//     wmma::layout_t out_layout = wmma::mem_row_major)
// {
//    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_K, WMMA_N, __nv_bfloat16, ALayout> a_frag[MATMUL_WARP_WIDTH][MATMUL_WARP_WIDTH][2];
//    load_a_frag_r2c_256<ALayout, a_trans, MATMUL_WARP_WIDTH, a_frag_from_acc>(a_real_input, sqrt_N, N, acc_frag_1, a_frag);

//    // __syncthreads();

//    _complex_matmul_r2c_256<ALayout, BLayout, out_trans, MATMUL_WARP_WIDTH, output_to_shmem>(scratch_real, scratch_imag, sqrt_N, N, a_frag, b_frag, acc_frag_1, out_layout);
// }

template <typename ALayout, typename BLayout, bool a_trans, bool out_trans, int MATMUL_WARP_WIDTH, bool a_frag_from_acc, bool output_to_shmem>
__device__ __forceinline__ void complex_matmul_c2r(
    float *scratch_real,
    float *scratch_imag,
    int sqrt_N,
    int N,
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_K, WMMA_N, __nv_bfloat16, BLayout> b_frag[MATMUL_WARP_WIDTH][MATMUL_WARP_WIDTH][2],
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_K, WMMA_N, float> acc_frag_1[MATMUL_WARP_WIDTH][MATMUL_WARP_WIDTH][2],
    wmma::layout_t out_layout = wmma::mem_row_major)
{
   wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_K, WMMA_N, __nv_bfloat16, ALayout> a_frag[MATMUL_WARP_WIDTH][MATMUL_WARP_WIDTH][2];
   load_a_frag<ALayout, a_trans, MATMUL_WARP_WIDTH, a_frag_from_acc>(scratch_real, scratch_imag, sqrt_N, N, acc_frag_1, a_frag);

   _complex_matmul_c2r<ALayout, BLayout, out_trans, MATMUL_WARP_WIDTH, output_to_shmem>(scratch_real, sqrt_N, N, a_frag, b_frag, acc_frag_1, out_layout);
}

// template <typename ALayout, typename BLayout, bool a_trans, bool out_trans, int MATMUL_WARP_WIDTH, bool a_frag_from_acc, bool output_to_shmem>
// __device__ __forceinline__ void complex_matmul_c2r_256(
//     float *scratch_real,
//     float *scratch_imag,
//     int sqrt_N,
//     int N,
//     wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_K, WMMA_N, __nv_bfloat16, BLayout> b_frag[MATMUL_WARP_WIDTH][MATMUL_WARP_WIDTH][2],
//     wmma::fragment<wmma::accumulator, WMMA_M, WMMA_K, WMMA_N, float> acc_frag_1[MATMUL_WARP_WIDTH][MATMUL_WARP_WIDTH][2],
//     wmma::layout_t out_layout = wmma::mem_row_major)
// {
//    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_K, WMMA_N, __nv_bfloat16, ALayout> a_frag[MATMUL_WARP_WIDTH][MATMUL_WARP_WIDTH][2];
//    load_a_frag_256<ALayout, a_trans, MATMUL_WARP_WIDTH, a_frag_from_acc>(scratch_real, scratch_imag, sqrt_N, N, acc_frag_1, a_frag);
//    // __syncthreads();

//    _complex_matmul_c2r_256<ALayout, BLayout, out_trans, MATMUL_WARP_WIDTH, output_to_shmem>(scratch_real, sqrt_N, N, a_frag, b_frag, acc_frag_1, out_layout);
// }

// template <typename ALayout, typename BLayout, bool a_trans, bool out_trans, int MATMUL_WARP_WIDTH, bool a_frag_from_acc, bool output_to_shmem>
// __device__ __forceinline__ void complex_matmul_c2r_256(
//     float *scratch_real,
//     float *scratch_imag,
//     int sqrt_N,
//     int N,
//     wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_K, WMMA_N, __nv_bfloat16, BLayout> b_frag[MATMUL_WARP_WIDTH][MATMUL_WARP_WIDTH][2],
//     wmma::fragment<wmma::accumulator, WMMA_M, WMMA_K, WMMA_N, float> acc_frag_1[MATMUL_WARP_WIDTH][MATMUL_WARP_WIDTH][2],
//     wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_K, WMMA_N, __nv_bfloat16, ALayout> k_frag[MATMUL_WARP_WIDTH][MATMUL_WARP_WIDTH][2],
//     wmma::layout_t out_layout = wmma::mem_row_major)
// {
//    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_K, WMMA_N, __nv_bfloat16, ALayout> a_frag[MATMUL_WARP_WIDTH][MATMUL_WARP_WIDTH][2];
//    load_a_frag_256<ALayout, a_trans, MATMUL_WARP_WIDTH, a_frag_from_acc>(scratch_real, scratch_imag, sqrt_N, N, acc_frag_1, a_frag);
//    // __syncthreads();

//    // multiply a_frag by k_frag
//    for (int j_a = 0; j_a < MATMUL_WARP_WIDTH; j_a++) {
//       for (int k = 0; k < MATMUL_WARP_WIDTH; k++) {
//          for (int i = 0; i < acc_frag_1[j_a][k][0].num_elements / 2; i++) {
//             complex_mul_bfloat162(
//                __nv_bfloat162(a_frag[j_a][k][0].x[2 * i], a_frag[j_a][k][0].x[2 * i + 1]),
//                __nv_bfloat162(a_frag[j_a][k][1].x[2 * i], a_frag[j_a][k][1].x[2 * i + 1]),
//                __nv_bfloat162(k_frag[j_a][k][0].x[2 * i], k_frag[j_a][k][0].x[2 * i + 1]),
//                __nv_bfloat162(k_frag[j_a][k][1].x[2 * i], k_frag[j_a][k][1].x[2 * i + 1]),
//                &a_frag[j_a][k][0].x[2 * i], 
//                &a_frag[j_a][k][1].x[2 * i],
//                &a_frag[j_a][k][0].x[2 * i + 1],
//                &a_frag[j_a][k][1].x[2 * i + 1]
//             );
//          }
//       }
//    }

//    _complex_matmul_c2r_256<ALayout, BLayout, out_trans, MATMUL_WARP_WIDTH, output_to_shmem>(scratch_real, sqrt_N, N, a_frag, b_frag, acc_frag_1, out_layout);
// }

template <typename ALayout, typename BLayout, bool a_trans, bool out_trans, int MATMUL_WARP_WIDTH, bool a_frag_from_acc, bool output_to_shmem>
__device__ __forceinline__ void complex_matmul_c2r(
    float *scratch_real,
    float *scratch_imag,
    int sqrt_N,
    int N,
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_K, WMMA_N, __nv_bfloat16, BLayout> b_frag[MATMUL_WARP_WIDTH][MATMUL_WARP_WIDTH][2],
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_K, WMMA_N, float> acc_frag_1[MATMUL_WARP_WIDTH][MATMUL_WARP_WIDTH][2],
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_K, WMMA_N, __nv_bfloat16, ALayout> k_frag[MATMUL_WARP_WIDTH][MATMUL_WARP_WIDTH][2],
    wmma::layout_t out_layout = wmma::mem_row_major)
{
   wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_K, WMMA_N, __nv_bfloat16, ALayout> a_frag[MATMUL_WARP_WIDTH][MATMUL_WARP_WIDTH][2];
   load_a_frag<ALayout, a_trans, MATMUL_WARP_WIDTH, a_frag_from_acc>(scratch_real, scratch_imag, sqrt_N, N, acc_frag_1, a_frag);
   // __syncthreads();

   //multiply a_frag by k_frag
   for (int j_a = 0; j_a < MATMUL_WARP_WIDTH; j_a++) {
      for (int k = 0; k < MATMUL_WARP_WIDTH; k++) {
         for (int i = 0; i < acc_frag_1[j_a][k][0].num_elements / 2; i++) {
            complex_mul_bfloat162(
               __nv_bfloat162(a_frag[j_a][k][0].x[2 * i], a_frag[j_a][k][0].x[2 * i + 1]),
               __nv_bfloat162(a_frag[j_a][k][1].x[2 * i], a_frag[j_a][k][1].x[2 * i + 1]),
               __nv_bfloat162(k_frag[j_a][k][0].x[2 * i], k_frag[j_a][k][0].x[2 * i + 1]),
               __nv_bfloat162(k_frag[j_a][k][1].x[2 * i], k_frag[j_a][k][1].x[2 * i + 1]),
               &a_frag[j_a][k][0].x[2 * i], 
               &a_frag[j_a][k][1].x[2 * i],
               &a_frag[j_a][k][0].x[2 * i + 1],
               &a_frag[j_a][k][1].x[2 * i + 1]
            );
         }
      }
   }

   _complex_matmul_c2r<ALayout, BLayout, out_trans, MATMUL_WARP_WIDTH, output_to_shmem>(scratch_real, sqrt_N, N, a_frag, b_frag, acc_frag_1, out_layout);
}

__device__ __forceinline__ void complex_mul(at::BFloat16 a_real, at::BFloat16 a_imag, at::BFloat16 b_real, at::BFloat16 b_imag, at::BFloat16 *c_real, at::BFloat16 *c_imag) {
   __nv_bfloat16 temp_x, temp_y;
   // temp_x = __hsub(__hmul(a_real, b_real), __hmul(a_imag, b_imag));
   // temp_y = __hadd(__hmul(a_imag, b_real), __hmul(a_real, b_imag));
   temp_x = __nv_bfloat16(a_real * b_real - a_imag * b_imag);
   temp_y = __hfma(__nv_bfloat16(a_imag), __nv_bfloat16(b_real), __nv_bfloat16(a_real * b_imag));
   *c_real = temp_x;
   *c_imag = temp_y;
}

__device__ __forceinline__ void complex_mul_float_bfloat16(float a_real, float a_imag, at::BFloat16 b_real, at::BFloat16 b_imag, at::BFloat16 *c_real, at::BFloat16 *c_imag) {
   __nv_bfloat16 temp_x, temp_y;
   // temp_x = __hsub(__hmul(a_real, b_real), __hmul(a_imag, b_imag));
   // temp_y = __hadd(__hmul(a_imag, b_real), __hmul(a_real, b_imag));
   temp_x = __nv_bfloat16(at::BFloat16(a_real) * b_real - at::BFloat16(a_imag) * b_imag);
   temp_y = __hfma(__nv_bfloat16(at::BFloat16(a_imag)), __nv_bfloat16(b_real), __nv_bfloat16(at::BFloat16(a_real) * b_imag));
   *c_real = temp_x;
   *c_imag = temp_y;
}

__device__ __forceinline__ void complex_mul_bfloat162(__nv_bfloat162 a_real, __nv_bfloat162 a_imag, __nv_bfloat162 b_real, __nv_bfloat162 b_imag, __nv_bfloat162 *c_real, __nv_bfloat162 *c_imag) {
   __nv_bfloat162 temp_x, temp_y;

   temp_x = __hsub2(__hmul2(a_real, b_real), __hmul2(a_imag, b_imag));
   temp_y = __hfma2(a_imag, b_real, __hmul2(a_real, b_imag));
   *c_real = temp_x;
   *c_imag = temp_y;
}

__device__ __forceinline__ void complex_mul_bfloat162(__nv_bfloat162 a_real, __nv_bfloat162 a_imag, __nv_bfloat162 b_real, __nv_bfloat162 b_imag, __nv_bfloat16 *c_real_0, __nv_bfloat16 *c_imag_0, __nv_bfloat16 *c_real_1, __nv_bfloat16 *c_imag_1) {
   __nv_bfloat162 temp_x, temp_y;

   temp_x = __hsub2(__hmul2(a_real, b_real), __hmul2(a_imag, b_imag));
   temp_y = __hfma2(a_imag, b_real, __hmul2(a_real, b_imag));
   *c_real_0 = temp_x.x;
   *c_imag_0 = temp_y.x;
   *c_real_1 = temp_x.y;
   *c_imag_1 = temp_y.y;
}

// negates b_imag
__device__ __forceinline__ void complex_mul_conj_bfloat162(__nv_bfloat162 a_real, __nv_bfloat162 a_imag, __nv_bfloat162 b_real, __nv_bfloat162 b_imag, c10::complex<__nv_bfloat16> *c_0, c10::complex<__nv_bfloat16> *c_1) {
   __nv_bfloat162 temp_x, temp_y;

   temp_x = __hfma2(a_real, b_real, __hmul2(a_imag, b_imag));
   // temp_x = __hsub2(__hmul2(a_real, b_real), __hmul2(a_imag, b_imag));
   temp_y = __hsub2(__hmul2(a_imag, b_real), __hmul2(a_real, b_imag));
   // temp_y = __hfma2(a_imag, b_real, __hmul2(a_real, b_imag));
   *c_0 = c10::complex<__nv_bfloat16>(temp_x.x, temp_y.x);
   *c_1 = c10::complex<__nv_bfloat16>(temp_x.y, temp_y.y);
}

__device__ __forceinline__ void complex_mul_conj_bfloat162(__nv_bfloat162 a_real, __nv_bfloat162 a_imag, __nv_bfloat162 b_real, __nv_bfloat162 b_imag, __nv_bfloat162 *c_real, __nv_bfloat162 *c_imag) {
   __nv_bfloat162 temp_x, temp_y;

   temp_x = __hfma2(a_real, b_real, __hmul2(a_imag, b_imag));
   // temp_x = __hsub2(__hmul2(a_real, b_real), __hmul2(a_imag, b_imag));
   temp_y = __hsub2(__hmul2(a_imag, b_real), __hmul2(a_real, b_imag));
   // temp_y = __hfma2(a_imag, b_real, __hmul2(a_real, b_imag));
   *c_real = temp_x;
   *c_imag = temp_y;
}

#endif