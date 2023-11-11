// Copyright (c) 2023 Dan Fu, Hermann Kumbong

#include <torch/extension.h>

#include <vector>
#include <stdio.h>
#include <mma.h>
#include <cuda_fp16.h>
#include <cub/block/block_load.cuh>
#include <cub/block/block_store.cuh>
#include "shared/monarch_cuda_shared_fp16_complex_mul.h"
#include "shared/monarch_cuda_shared_fp16_matmuls.h"
#include "shared/monarch_cuda_shared_fp16_load_frags.h"
using namespace nvcuda;

using complex_half_t = typename c10::complex<at::Half>;

#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16
// #define TILE_SIZE 4
// #define SHMEM_SIZE 256 * TILE_SIZE
// #define SEQUENCE_SIZE 256
#define WARP_SIZE 32

#ifndef MONARCH_CUDA_H_
#define MONARCH_CUDA_H_

template <typename ALayout, typename BLayout, bool a_trans, bool out_trans, int MATMUL_WARP_WIDTH, bool a_frag_from_acc, bool output_to_shmem>
__device__ __forceinline__ void complex_matmul(
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
  load_a_frag<ALayout, a_trans, MATMUL_WARP_WIDTH, a_frag_from_acc>(a_real, a_imag, sqrt_N, N, acc_frag_1, a_frag);

  // __syncthreads();
  // multiply a_frag by k_frag
  for (int j_a = 0; j_a < MATMUL_WARP_WIDTH; j_a++) {
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

  _complex_matmul<ALayout, BLayout, out_trans, MATMUL_WARP_WIDTH, output_to_shmem>(a_real, a_imag, sqrt_N, N, a_frag, b_frag, acc_frag_1, out_layout);
}

template <typename ALayout, typename BLayout, bool a_trans, bool out_trans, int MATMUL_WARP_WIDTH, bool a_frag_from_acc, bool output_to_shmem>
__device__ __forceinline__ void complex_matmul(
  half *a_real,
  half *a_imag,
  int sqrt_N,
  int N,
  wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_K, WMMA_N, half, BLayout> b_frag[MATMUL_WARP_WIDTH][MATMUL_WARP_WIDTH][2],
  wmma::fragment<wmma::accumulator, WMMA_M, WMMA_K, WMMA_N, half> acc_frag_1[MATMUL_WARP_WIDTH][MATMUL_WARP_WIDTH][2],
  wmma::layout_t out_layout = wmma::mem_row_major)
{
  wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_K, WMMA_N, half, ALayout> a_frag[MATMUL_WARP_WIDTH][MATMUL_WARP_WIDTH][2];
  load_a_frag<ALayout, a_trans, MATMUL_WARP_WIDTH, a_frag_from_acc>(a_real, a_imag, sqrt_N, N, acc_frag_1, a_frag);

  // __syncthreads();
  _complex_matmul<ALayout, BLayout, out_trans, MATMUL_WARP_WIDTH, output_to_shmem>(a_real, a_imag, sqrt_N, N, a_frag, b_frag, acc_frag_1, out_layout);
}

template <typename ALayout, typename BLayout, bool b_trans, bool out_trans, int MATMUL_WARP_WIDTH, bool b_frag_from_acc, bool output_to_shmem>
__device__ __forceinline__ void complex_matmul_load_b(
  half *b_real,
  half *b_imag,
  int sqrt_N,
  int N,
  wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_K, WMMA_N, half, ALayout> a_frag[MATMUL_WARP_WIDTH][MATMUL_WARP_WIDTH][2],
  wmma::fragment<wmma::accumulator, WMMA_M, WMMA_K, WMMA_N, half> acc_frag_1[MATMUL_WARP_WIDTH][MATMUL_WARP_WIDTH][2],
  wmma::layout_t out_layout = wmma::mem_row_major)
{
  wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_K, WMMA_N, half, BLayout> b_frag[MATMUL_WARP_WIDTH][MATMUL_WARP_WIDTH][2];
  load_b_frag<BLayout, b_trans, MATMUL_WARP_WIDTH, b_frag_from_acc>(b_real, b_imag, sqrt_N, N, acc_frag_1, b_frag);

  // __syncthreads();
  _complex_matmul<ALayout, BLayout, out_trans, MATMUL_WARP_WIDTH, output_to_shmem>(b_real, b_imag, sqrt_N, N, a_frag, b_frag, acc_frag_1, out_layout);
}

template <typename ALayout, typename BLayout, bool b_trans, bool out_trans, int MATMUL_WARP_WIDTH, bool b_frag_from_acc, bool output_to_shmem>
__device__ __forceinline__ void complex_matmul_load_b(
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
  load_b_frag<BLayout, b_trans, MATMUL_WARP_WIDTH, b_frag_from_acc>(b_real, b_imag, sqrt_N, N, acc_frag_1, b_frag);

  // __syncthreads();
  // multiply b_frag by k_frag
  for (int j_a = 0; j_a < MATMUL_WARP_WIDTH; j_a++) {
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
  _complex_matmul<ALayout, BLayout, out_trans, MATMUL_WARP_WIDTH, output_to_shmem>(b_real, b_imag, sqrt_N, N, a_frag, b_frag, acc_frag_1, out_layout);
}

template <typename ALayout, typename BLayout, bool a_trans, bool out_trans, int MATMUL_WARP_WIDTH, bool a_frag_from_acc, bool output_to_shmem>
__device__ __forceinline__ void complex_matmul_r2c(
  const half *a_real_input,
  half *a_real,
  half *a_imag,
  int sqrt_N,
  int N,
  wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_K, WMMA_N, half, BLayout> b_frag[MATMUL_WARP_WIDTH][MATMUL_WARP_WIDTH][2],
  wmma::fragment<wmma::accumulator, WMMA_M, WMMA_K, WMMA_N, half> acc_frag_1[MATMUL_WARP_WIDTH][MATMUL_WARP_WIDTH][2],
  wmma::layout_t out_layout = wmma::mem_row_major)
{
  wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_K, WMMA_N, half, ALayout> a_frag[MATMUL_WARP_WIDTH][MATMUL_WARP_WIDTH][2];
  load_a_frag_r2c<ALayout, a_trans, MATMUL_WARP_WIDTH, a_frag_from_acc>(a_real_input, sqrt_N, N, acc_frag_1, a_frag);

  _complex_matmul_r2c<ALayout, BLayout, out_trans, MATMUL_WARP_WIDTH, output_to_shmem>(a_real, a_imag, sqrt_N, N, a_frag, b_frag, acc_frag_1, out_layout);
}

template <typename ALayout, typename BLayout, bool b_trans, bool out_trans, int MATMUL_WARP_WIDTH, bool b_frag_from_acc, bool output_to_shmem>
__device__ __forceinline__ void complex_matmul_r2c_load_b(
   const half *b_real_input,
   half *b_real,
   half *b_imag,
   int sqrt_N,
   int N,
   wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_K, WMMA_N, half, ALayout> a_frag[MATMUL_WARP_WIDTH][MATMUL_WARP_WIDTH][2],
   wmma::fragment<wmma::accumulator, WMMA_M, WMMA_K, WMMA_N, half> acc_frag_1[MATMUL_WARP_WIDTH][MATMUL_WARP_WIDTH][2],
   wmma::layout_t out_layout = wmma::mem_row_major)
{
  wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_K, WMMA_N, half, BLayout> b_frag[MATMUL_WARP_WIDTH][MATMUL_WARP_WIDTH][2];
  load_b_frag_r2c<BLayout, b_trans, MATMUL_WARP_WIDTH, b_frag_from_acc>(b_real_input, sqrt_N, N, acc_frag_1, b_frag);

  _complex_matmul_r2c_load_b<ALayout, BLayout, out_trans, MATMUL_WARP_WIDTH, output_to_shmem>(b_real, b_imag, sqrt_N, N, a_frag, b_frag, acc_frag_1, out_layout);
}

template <typename ALayout, typename BLayout, bool a_trans, bool out_trans, int MATMUL_WARP_WIDTH, bool a_frag_from_acc, bool output_to_shmem>
__device__ __forceinline__ void complex_matmul_r2c_256(
  const half *a_real_input,
  half *a_real,
  half *a_imag,
  int sqrt_N,
  int N,
  wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_K, WMMA_N, half, BLayout> b_frag[MATMUL_WARP_WIDTH][MATMUL_WARP_WIDTH][2],
  wmma::fragment<wmma::accumulator, WMMA_M, WMMA_K, WMMA_N, half> acc_frag_1[MATMUL_WARP_WIDTH][MATMUL_WARP_WIDTH][2],
  wmma::layout_t out_layout = wmma::mem_row_major)
{
  wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_K, WMMA_N, half, ALayout> a_frag[MATMUL_WARP_WIDTH][MATMUL_WARP_WIDTH][2];
  load_a_frag_r2c_256<ALayout, a_trans, MATMUL_WARP_WIDTH, a_frag_from_acc>(a_real_input, sqrt_N, N, acc_frag_1, a_frag);

  // __syncthreads();

  _complex_matmul_r2c_256<ALayout, BLayout, out_trans, MATMUL_WARP_WIDTH, output_to_shmem>(a_real, a_imag, sqrt_N, N, a_frag, b_frag, acc_frag_1, out_layout);
}

template <typename ALayout, typename BLayout, bool a_trans, bool out_trans, int MATMUL_WARP_WIDTH, bool a_frag_from_acc, bool output_to_shmem>
__device__ __forceinline__ void complex_matmul_r2c_1024(
  const half *a_real_input,
  half *a_real,
  half *a_imag,
  int sqrt_N,
  int N,
  wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_K, WMMA_N, half, BLayout> b_frag[MATMUL_WARP_WIDTH][MATMUL_WARP_WIDTH][2],
  wmma::fragment<wmma::accumulator, WMMA_M, WMMA_K, WMMA_N, half> acc_frag_1[MATMUL_WARP_WIDTH][MATMUL_WARP_WIDTH][2],
  wmma::layout_t out_layout = wmma::mem_row_major)
{
  wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_K, WMMA_N, half, ALayout> a_frag[MATMUL_WARP_WIDTH][MATMUL_WARP_WIDTH][2];
  load_a_frag_r2c_1024<ALayout, a_trans, MATMUL_WARP_WIDTH, a_frag_from_acc>(a_real_input, sqrt_N, N, acc_frag_1, a_frag);

  // __syncthreads();

  _complex_matmul_r2c_1024<ALayout, BLayout, out_trans, MATMUL_WARP_WIDTH, output_to_shmem>(a_real, a_imag, sqrt_N, N, a_frag, b_frag, acc_frag_1, out_layout);
}

template <typename ALayout, typename BLayout, bool a_trans, bool out_trans, int MATMUL_WARP_WIDTH, bool a_frag_from_acc, bool output_to_shmem>
__device__ __forceinline__ void complex_matmul_c2c_1024(
  half *a_real,
  half *a_imag,
  int sqrt_N,
  int N,
  wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_K, WMMA_N, half, BLayout> b_frag[MATMUL_WARP_WIDTH][MATMUL_WARP_WIDTH][2],
  wmma::fragment<wmma::accumulator, WMMA_M, WMMA_K, WMMA_N, half> acc_frag_1[MATMUL_WARP_WIDTH][MATMUL_WARP_WIDTH][2],
  wmma::layout_t out_layout = wmma::mem_row_major)
{
  wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_K, WMMA_N, half, ALayout> a_frag[MATMUL_WARP_WIDTH][MATMUL_WARP_WIDTH][2];
  load_a_frag_1024<ALayout, a_trans, MATMUL_WARP_WIDTH, a_frag_from_acc>(a_real, a_imag, sqrt_N, N, acc_frag_1, a_frag);

  // __syncthreads();

  _complex_matmul_1024<ALayout, BLayout, out_trans, MATMUL_WARP_WIDTH, output_to_shmem>(a_real, a_imag, sqrt_N, N, a_frag, b_frag, acc_frag_1, out_layout);
}

template <typename ALayout, typename BLayout, bool a_trans, bool out_trans, int MATMUL_WARP_WIDTH, bool a_frag_from_acc, bool output_to_shmem>
__device__ __forceinline__ void complex_matmul_c2c_256(
  const half *a_real_inp,
  const half *a_imag_inp,
  half *a_real_out,
  half *a_imag_out,
  int sqrt_N,
  int N,
  wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_K, WMMA_N, half, BLayout> b_frag[MATMUL_WARP_WIDTH][MATMUL_WARP_WIDTH][2],
  wmma::fragment<wmma::accumulator, WMMA_M, WMMA_K, WMMA_N, half> acc_frag_1[MATMUL_WARP_WIDTH][MATMUL_WARP_WIDTH][2],
  wmma::layout_t out_layout = wmma::mem_row_major)
{
  wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_K, WMMA_N, half, ALayout> a_frag[MATMUL_WARP_WIDTH][MATMUL_WARP_WIDTH][2];
  load_a_frag_256<ALayout, a_trans, MATMUL_WARP_WIDTH, a_frag_from_acc>(a_real_inp, a_imag_inp, sqrt_N, N, acc_frag_1, a_frag);

  // __syncthreads();

  _complex_matmul_256<ALayout, BLayout, out_trans, MATMUL_WARP_WIDTH, output_to_shmem>(a_real_out, a_imag_out, sqrt_N, N, a_frag, b_frag, acc_frag_1, out_layout);
}

template <typename ALayout, typename BLayout, bool a_trans, bool out_trans, int MATMUL_WARP_WIDTH, bool a_frag_from_acc, bool output_to_shmem>
__device__ __forceinline__ void complex_matmul_c2c_256(
  half *a_real_inp,
  half *a_imag_inp,
  half *a_real_out,
  half *a_imag_out,
  int sqrt_N,
  int N,
  wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_K, WMMA_N, half, BLayout> b_frag[MATMUL_WARP_WIDTH][MATMUL_WARP_WIDTH][2],
  wmma::fragment<wmma::accumulator, WMMA_M, WMMA_K, WMMA_N, half> acc_frag_1[MATMUL_WARP_WIDTH][MATMUL_WARP_WIDTH][2],
  wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_K, WMMA_N, half, ALayout> k_frag[MATMUL_WARP_WIDTH][MATMUL_WARP_WIDTH][2],
  wmma::layout_t out_layout = wmma::mem_row_major)
{
  wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_K, WMMA_N, half, ALayout> a_frag[MATMUL_WARP_WIDTH][MATMUL_WARP_WIDTH][2];
  load_a_frag_256<ALayout, a_trans, MATMUL_WARP_WIDTH, a_frag_from_acc>(a_real_inp, a_imag_inp, sqrt_N, N, acc_frag_1, a_frag);

  // multiply a_frag by k_frag
  for (int j_a = 0; j_a < MATMUL_WARP_WIDTH; j_a++) {
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

  _complex_matmul_256<ALayout, BLayout, out_trans, MATMUL_WARP_WIDTH, output_to_shmem>(a_real_out, a_imag_out, sqrt_N, N, a_frag, b_frag, acc_frag_1, out_layout);
}

template <typename ALayout, typename BLayout, bool a_trans, bool out_trans, int MATMUL_WARP_WIDTH, bool a_frag_from_acc, bool output_to_shmem>
__device__ __forceinline__ void complex_matmul_c2c_1024(
  const half *a_real_inp,
  const half *a_imag_inp,
  half *a_real_out,
  half *a_imag_out,
  int sqrt_N,
  int N,
  wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_K, WMMA_N, half, BLayout> b_frag[MATMUL_WARP_WIDTH][MATMUL_WARP_WIDTH][2],
  wmma::fragment<wmma::accumulator, WMMA_M, WMMA_K, WMMA_N, half> acc_frag_1[MATMUL_WARP_WIDTH][MATMUL_WARP_WIDTH][2],
  wmma::layout_t out_layout = wmma::mem_row_major)
{
  wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_K, WMMA_N, half, ALayout> a_frag[MATMUL_WARP_WIDTH][MATMUL_WARP_WIDTH][2];
  load_a_frag_1024<ALayout, a_trans, MATMUL_WARP_WIDTH, a_frag_from_acc>(a_real_inp, a_imag_inp, sqrt_N, N, acc_frag_1, a_frag);

  // __syncthreads();

  _complex_matmul_1024<ALayout, BLayout, out_trans, MATMUL_WARP_WIDTH, output_to_shmem>(a_real_out, a_imag_out, sqrt_N, N, a_frag, b_frag, acc_frag_1, out_layout);
}

template <typename ALayout, typename BLayout, bool a_trans, bool out_trans, int MATMUL_WARP_WIDTH, bool a_frag_from_acc, bool output_to_shmem>
__device__ __forceinline__ void complex_matmul_c2c_1024(
  half *a_real_inp,
  half *a_imag_inp,
  half *a_real_out,
  half *a_imag_out,
  int sqrt_N,
  int N,
  wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_K, WMMA_N, half, BLayout> b_frag[MATMUL_WARP_WIDTH][MATMUL_WARP_WIDTH][2],
  wmma::fragment<wmma::accumulator, WMMA_M, WMMA_K, WMMA_N, half> acc_frag_1[MATMUL_WARP_WIDTH][MATMUL_WARP_WIDTH][2],
  wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_K, WMMA_N, half, ALayout> k_frag[MATMUL_WARP_WIDTH][MATMUL_WARP_WIDTH][2],
  wmma::layout_t out_layout = wmma::mem_row_major)
{
  wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_K, WMMA_N, half, ALayout> a_frag[MATMUL_WARP_WIDTH][MATMUL_WARP_WIDTH][2];
  load_a_frag_1024<ALayout, a_trans, MATMUL_WARP_WIDTH, a_frag_from_acc>(a_real_inp, a_imag_inp, sqrt_N, N, acc_frag_1, a_frag);

  // multiply a_frag by k_frag
  for (int j_a = 0; j_a < MATMUL_WARP_WIDTH; j_a++) {
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

  _complex_matmul_1024<ALayout, BLayout, out_trans, MATMUL_WARP_WIDTH, output_to_shmem>(a_real_out, a_imag_out, sqrt_N, N, a_frag, b_frag, acc_frag_1, out_layout);
}

template <typename ALayout, typename BLayout, bool a_trans, bool out_trans, int MATMUL_WARP_WIDTH, bool a_frag_from_acc, bool output_to_shmem>
__device__ __forceinline__ void complex_matmul_c2r(
  half *a_real,
  half *a_imag,
  half *a_real_out,
  int sqrt_N,
  int N,
  wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_K, WMMA_N, half, BLayout> b_frag[MATMUL_WARP_WIDTH][MATMUL_WARP_WIDTH][2],
  wmma::fragment<wmma::accumulator, WMMA_M, WMMA_K, WMMA_N, half> acc_frag_1[MATMUL_WARP_WIDTH][MATMUL_WARP_WIDTH][2],
  wmma::layout_t out_layout = wmma::mem_row_major)
{
  wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_K, WMMA_N, half, ALayout> a_frag[MATMUL_WARP_WIDTH][MATMUL_WARP_WIDTH][2];
  load_a_frag<ALayout, a_trans, MATMUL_WARP_WIDTH, a_frag_from_acc>(a_real, a_imag, sqrt_N, N, acc_frag_1, a_frag);

  _complex_matmul_c2r<ALayout, BLayout, out_trans, MATMUL_WARP_WIDTH, output_to_shmem>(a_real_out, sqrt_N, N, a_frag, b_frag, acc_frag_1, out_layout);
}

template <typename ALayout, typename BLayout, bool a_trans, bool out_trans, int MATMUL_WARP_WIDTH, bool a_frag_from_acc, bool output_to_shmem>
__device__ __forceinline__ void complex_matmul_c2r_256(
  half *a_real,
  half *a_imag,
  half *a_real_out,
  int sqrt_N,
  int N,
  wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_K, WMMA_N, half, BLayout> b_frag[MATMUL_WARP_WIDTH][MATMUL_WARP_WIDTH][2],
  wmma::fragment<wmma::accumulator, WMMA_M, WMMA_K, WMMA_N, half> acc_frag_1[MATMUL_WARP_WIDTH][MATMUL_WARP_WIDTH][2],
  wmma::layout_t out_layout = wmma::mem_row_major)
{
  wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_K, WMMA_N, half, ALayout> a_frag[MATMUL_WARP_WIDTH][MATMUL_WARP_WIDTH][2];
  load_a_frag_256<ALayout, a_trans, MATMUL_WARP_WIDTH, a_frag_from_acc>(a_real, a_imag, sqrt_N, N, acc_frag_1, a_frag);
  // __syncthreads();

  _complex_matmul_c2r_256<ALayout, BLayout, out_trans, MATMUL_WARP_WIDTH, output_to_shmem>(a_real_out, sqrt_N, N, a_frag, b_frag, acc_frag_1, out_layout);
}

template <typename ALayout, typename BLayout, bool a_trans, bool out_trans, int MATMUL_WARP_WIDTH, bool a_frag_from_acc, bool output_to_shmem>
__device__ __forceinline__ void complex_matmul_c2r_256(
  half *a_real,
  half *a_imag,
  half *a_real_out,
  int sqrt_N,
  int N,
  wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_K, WMMA_N, half, BLayout> b_frag[MATMUL_WARP_WIDTH][MATMUL_WARP_WIDTH][2],
  wmma::fragment<wmma::accumulator, WMMA_M, WMMA_K, WMMA_N, half> acc_frag_1[MATMUL_WARP_WIDTH][MATMUL_WARP_WIDTH][2],
  wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_K, WMMA_N, half, ALayout> k_frag[MATMUL_WARP_WIDTH][MATMUL_WARP_WIDTH][2],
  wmma::layout_t out_layout = wmma::mem_row_major)
{
  wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_K, WMMA_N, half, ALayout> a_frag[MATMUL_WARP_WIDTH][MATMUL_WARP_WIDTH][2];
  load_a_frag_256<ALayout, a_trans, MATMUL_WARP_WIDTH, a_frag_from_acc>(a_real, a_imag, sqrt_N, N, acc_frag_1, a_frag);
  // __syncthreads();

  // multiply a_frag by k_frag
  for (int j_a = 0; j_a < MATMUL_WARP_WIDTH; j_a++) {
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

  _complex_matmul_c2r_256<ALayout, BLayout, out_trans, MATMUL_WARP_WIDTH, output_to_shmem>(a_real_out, sqrt_N, N, a_frag, b_frag, acc_frag_1, out_layout);
}

template <typename ALayout, typename BLayout, bool a_trans, bool out_trans, int MATMUL_WARP_WIDTH, bool a_frag_from_acc, bool output_to_shmem>
__device__ __forceinline__ void complex_matmul_c2r_1024(
  half *a_real,
  half *a_imag,
  half *a_real_out,
  int sqrt_N,
  int N,
  wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_K, WMMA_N, half, BLayout> b_frag[MATMUL_WARP_WIDTH][MATMUL_WARP_WIDTH][2],
  wmma::fragment<wmma::accumulator, WMMA_M, WMMA_K, WMMA_N, half> acc_frag_1[MATMUL_WARP_WIDTH][MATMUL_WARP_WIDTH][2],
  wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_K, WMMA_N, half, ALayout> k_frag[MATMUL_WARP_WIDTH][MATMUL_WARP_WIDTH][2],
  wmma::layout_t out_layout = wmma::mem_row_major)
{
  wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_K, WMMA_N, half, ALayout> a_frag[MATMUL_WARP_WIDTH][MATMUL_WARP_WIDTH][2];
  load_a_frag_1024<ALayout, a_trans, MATMUL_WARP_WIDTH, a_frag_from_acc>(a_real, a_imag, sqrt_N, N, acc_frag_1, a_frag);
  // __syncthreads();

  // multiply a_frag by k_frag
  for (int j_a = 0; j_a < MATMUL_WARP_WIDTH; j_a++) {
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

  _complex_matmul_c2r_1024<ALayout, BLayout, out_trans, MATMUL_WARP_WIDTH, output_to_shmem>(a_real_out, sqrt_N, N, a_frag, b_frag, acc_frag_1, out_layout);
}

template <typename ALayout, typename BLayout, bool a_trans, bool out_trans, int MATMUL_WARP_WIDTH, bool a_frag_from_acc, bool output_to_shmem>
__device__ __forceinline__ void complex_matmul_c2r(
  half *a_real,
  half *a_imag,
  half *a_real_out,
  int sqrt_N,
  int N,
  wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_K, WMMA_N, half, BLayout> b_frag[MATMUL_WARP_WIDTH][MATMUL_WARP_WIDTH][2],
  wmma::fragment<wmma::accumulator, WMMA_M, WMMA_K, WMMA_N, half> acc_frag_1[MATMUL_WARP_WIDTH][MATMUL_WARP_WIDTH][2],
  wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_K, WMMA_N, half, ALayout> k_frag[MATMUL_WARP_WIDTH][MATMUL_WARP_WIDTH][2],
  wmma::layout_t out_layout = wmma::mem_row_major)
{
  wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_K, WMMA_N, half, ALayout> a_frag[MATMUL_WARP_WIDTH][MATMUL_WARP_WIDTH][2];
  load_a_frag<ALayout, a_trans, MATMUL_WARP_WIDTH, a_frag_from_acc>(a_real, a_imag, sqrt_N, N, acc_frag_1, a_frag);
  // __syncthreads();

  // multiply a_frag by k_frag
  for (int j_a = 0; j_a < MATMUL_WARP_WIDTH; j_a++) {
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

  _complex_matmul_c2r<ALayout, BLayout, out_trans, MATMUL_WARP_WIDTH, output_to_shmem>(a_real_out, sqrt_N, N, a_frag, b_frag, acc_frag_1, out_layout);
}

#endif