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

#ifndef MONARCH_CUDA_LOAD_
#define MONARCH_CUDA_LOAD_

template <typename ALayout, bool a_trans, int MATMUL_WARP_WIDTH, bool a_frag_from_acc>
__device__ __forceinline__ void load_a_frag(
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
    for (int j_a = 0; j_a < MATMUL_WARP_WIDTH; j_a++) {
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
    for (int j_a = 0; j_a < MATMUL_WARP_WIDTH; j_a++) {
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
__device__ __forceinline__ void load_a_frag_256(
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
    for (int j_a = 0; j_a < MATMUL_WARP_WIDTH; j_a++) {
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
    for (int j_a = 0; j_a < MATMUL_WARP_WIDTH; j_a++) {
      // #pragma unroll
      for (int k = 0; k < MATMUL_WARP_WIDTH; k++) {
        a_idx = a_trans ? k * WMMA_K * 256 + j_a * WMMA_K : j_a * WMMA_K * 256 + k * WMMA_K;
        wmma::load_matrix_sync(a_frag[j_a][k][0], a_real + a_idx, 256);
        wmma::load_matrix_sync(a_frag[j_a][k][1], a_imag + a_idx, 256);
      }
    }  
  }
}

template <typename ALayout, bool a_trans, int MATMUL_WARP_WIDTH, bool a_frag_from_acc>
__device__ __forceinline__ void load_a_frag_256(
  const half *a_real,
  const half *a_imag,
  int sqrt_N,
  int N,
  wmma::fragment<wmma::accumulator, WMMA_M, WMMA_K, WMMA_N, half> acc_frag_1[MATMUL_WARP_WIDTH][MATMUL_WARP_WIDTH][2],
  wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_K, WMMA_N, half, ALayout> a_frag[MATMUL_WARP_WIDTH][MATMUL_WARP_WIDTH][2])
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
          for (int i = 0; i < acc_frag_1[j_a][j_b][0].num_elements; i++) {
            a_frag[j_a][j_b][k].x[i] = acc_frag_1[j_a][j_b][k].x[i];
            a_frag[j_a][j_b][k].x[i + acc_frag_1[j_a][j_b][k].num_elements] = acc_frag_1[j_a][j_b][k].x[i];
          }
        }
      }
    }
  } else {
    // #pragma unroll
    for (int j_a = 0; j_a < MATMUL_WARP_WIDTH; j_a++) {
      // #pragma unroll
      for (int k = 0; k < MATMUL_WARP_WIDTH; k++) {
        a_idx = a_trans ? k * WMMA_K * 256 + j_a * WMMA_K : j_a * WMMA_K * 256 + k * WMMA_K;
        wmma::load_matrix_sync(a_frag[j_a][k][0], a_real + a_idx, 256);
        wmma::load_matrix_sync(a_frag[j_a][k][1], a_imag + a_idx, 256);
      }
    }  
  }
}

template <typename ALayout, bool a_trans, int MATMUL_WARP_WIDTH, bool a_frag_from_acc>
__device__ __forceinline__ void load_a_frag_1024(
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
    for (int j_a = 0; j_a < MATMUL_WARP_WIDTH; j_a++) {
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
    for (int j_a = 0; j_a < MATMUL_WARP_WIDTH; j_a++) {
      // #pragma unroll
      for (int k = 0; k < MATMUL_WARP_WIDTH; k++) {
        a_idx = a_trans ? k * WMMA_K * 1024 + j_a * WMMA_K : j_a * WMMA_K * 1024 + k * WMMA_K;
        wmma::load_matrix_sync(a_frag[j_a][k][0], a_real + a_idx, 1024);
        wmma::load_matrix_sync(a_frag[j_a][k][1], a_imag + a_idx, 1024);
      }
    }  
  }
}

template <typename ALayout, bool a_trans, int MATMUL_WARP_WIDTH, bool a_frag_from_acc>
__device__ __forceinline__ void load_a_frag_1024(
  const half *a_real,
  const half *a_imag,
  int sqrt_N,
  int N,
  wmma::fragment<wmma::accumulator, WMMA_M, WMMA_K, WMMA_N, half> acc_frag_1[MATMUL_WARP_WIDTH][MATMUL_WARP_WIDTH][2],
  wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_K, WMMA_N, half, ALayout> a_frag[MATMUL_WARP_WIDTH][MATMUL_WARP_WIDTH][2])
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
          for (int i = 0; i < acc_frag_1[j_a][j_b][0].num_elements; i++) {
            a_frag[j_a][j_b][k].x[i] = acc_frag_1[j_a][j_b][k].x[i];
            a_frag[j_a][j_b][k].x[i + acc_frag_1[j_a][j_b][k].num_elements] = acc_frag_1[j_a][j_b][k].x[i];
          }
        }
      }
    }
  } else {
    // #pragma unroll
    for (int j_a = 0; j_a < MATMUL_WARP_WIDTH; j_a++) {
      // #pragma unroll
      for (int k = 0; k < MATMUL_WARP_WIDTH; k++) {
        a_idx = a_trans ? k * WMMA_K * 1024 + j_a * WMMA_K : j_a * WMMA_K * 1024 + k * WMMA_K;
        wmma::load_matrix_sync(a_frag[j_a][k][0], a_real + a_idx, 1024);
        wmma::load_matrix_sync(a_frag[j_a][k][1], a_imag + a_idx, 1024);
      }
    }  
  }
}

template <typename ALayout, bool a_trans, int MATMUL_WARP_WIDTH, bool a_frag_from_acc>
__device__ __forceinline__ void load_b_frag_r2c(
  const half *b_real,
  int sqrt_N,
  int N,
  wmma::fragment<wmma::accumulator, WMMA_M, WMMA_K, WMMA_N, half> acc_frag_1[MATMUL_WARP_WIDTH][MATMUL_WARP_WIDTH][2],
  wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_K, WMMA_N, half, ALayout> b_frag[MATMUL_WARP_WIDTH][MATMUL_WARP_WIDTH][2])
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

template <typename ALayout, bool a_trans, int MATMUL_WARP_WIDTH, bool a_frag_from_acc>
__device__ __forceinline__ void load_b_frag(
  half *b_real,
  half *b_imag,
  int sqrt_N,
  int N,
  wmma::fragment<wmma::accumulator, WMMA_M, WMMA_K, WMMA_N, half> acc_frag_1[MATMUL_WARP_WIDTH][MATMUL_WARP_WIDTH][2],
  wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_K, WMMA_N, half, ALayout> b_frag[MATMUL_WARP_WIDTH][MATMUL_WARP_WIDTH][2])
{
  int b_idx;
  // #pragma unroll
  for (int j_a = 0; j_a < MATMUL_WARP_WIDTH; j_a++) {
    // #pragma unroll
    for (int k = 0; k < MATMUL_WARP_WIDTH; k++) {
      b_idx = a_trans ? k * WMMA_K * sqrt_N + j_a * WMMA_K : j_a * WMMA_K * sqrt_N + k * WMMA_K;
      wmma::load_matrix_sync(b_frag[j_a][k][0], b_real + b_idx, sqrt_N);
      wmma::load_matrix_sync(b_frag[j_a][k][1], b_imag + b_idx, sqrt_N);
    }
  }  
}

template <typename ALayout, bool a_trans, int MATMUL_WARP_WIDTH, bool a_frag_from_acc>
__device__ __forceinline__ void load_a_frag_r2c(
  const half *a_real,
  int sqrt_N,
  int N,
  wmma::fragment<wmma::accumulator, WMMA_M, WMMA_K, WMMA_N, half> acc_frag_1[MATMUL_WARP_WIDTH][MATMUL_WARP_WIDTH][2],
  wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_K, WMMA_N, half, ALayout> a_frag[MATMUL_WARP_WIDTH][MATMUL_WARP_WIDTH][2])
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
            a_frag[j_a][j_b][k].x[i] = acc_frag_1[j_a][j_b][k].x[i];
            a_frag[j_a][j_b][k].x[i + acc_frag_1[j_a][j_b][k].num_elements] = acc_frag_1[j_a][j_b][k].x[i];
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

template <typename ALayout, bool a_trans, int MATMUL_WARP_WIDTH, bool a_frag_from_acc>
__device__ __forceinline__ void load_a_frag_r2c_256(
  const half *a_real,
  int sqrt_N,
  int N,
  wmma::fragment<wmma::accumulator, WMMA_M, WMMA_K, WMMA_N, half> acc_frag_1[MATMUL_WARP_WIDTH][MATMUL_WARP_WIDTH][2],
  wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_K, WMMA_N, half, ALayout> a_frag[MATMUL_WARP_WIDTH][MATMUL_WARP_WIDTH][2])
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
            a_frag[j_a][j_b][k].x[i] = acc_frag_1[j_a][j_b][k].x[i];
            a_frag[j_a][j_b][k].x[i + acc_frag_1[j_a][j_b][k].num_elements] = acc_frag_1[j_a][j_b][k].x[i];
          }
        }
      }
    }
  } else {
    // #pragma unroll
    for (int j_a = 0; j_a < MATMUL_WARP_WIDTH; j_a++) {
      // #pragma unroll
      for (int k = 0; k < MATMUL_WARP_WIDTH; k++) {
        a_idx = a_trans ? k * WMMA_K * 256 + j_a * WMMA_K : j_a * WMMA_K * 256 + k * WMMA_K;
        wmma::load_matrix_sync(a_frag[j_a][k][0], a_real + a_idx, 256);
      }
    }  
  }
}

template <typename ALayout, bool a_trans, int MATMUL_WARP_WIDTH, bool a_frag_from_acc>
__device__ __forceinline__ void load_a_frag_r2c_1024(
  const half *a_real,
  int sqrt_N,
  int N,
  wmma::fragment<wmma::accumulator, WMMA_M, WMMA_K, WMMA_N, half> acc_frag_1[MATMUL_WARP_WIDTH][MATMUL_WARP_WIDTH][2],
  wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_K, WMMA_N, half, ALayout> a_frag[MATMUL_WARP_WIDTH][MATMUL_WARP_WIDTH][2])
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
            a_frag[j_a][j_b][k].x[i] = acc_frag_1[j_a][j_b][k].x[i];
            a_frag[j_a][j_b][k].x[i + acc_frag_1[j_a][j_b][k].num_elements] = acc_frag_1[j_a][j_b][k].x[i];
          }
        }
      }
    }
  } else {
    // #pragma unroll
    for (int j_a = 0; j_a < MATMUL_WARP_WIDTH; j_a++) {
      // #pragma unroll
      for (int k = 0; k < MATMUL_WARP_WIDTH; k++) {
        a_idx = a_trans ? k * WMMA_K * 1024 + j_a * WMMA_K : j_a * WMMA_K * 1024 + k * WMMA_K;
        wmma::load_matrix_sync(a_frag[j_a][k][0], a_real + a_idx, 1024);
      }
    }  
  }
}

#endif