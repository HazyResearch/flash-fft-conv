// Copyright (c) 2023 Dan Fu, Hermann Kumbong

#include <torch/extension.h>

#include <vector>
#include <stdio.h>
#include <mma.h>
#include <cuda_fp16.h>
#include <cub/block/block_load.cuh>
#include <cub/block/block_store.cuh>
#include "monarch_cuda_shared.h"
#include "monarch_cuda_shared_r2r.h"
using namespace nvcuda;

template <int BLOCK_DIM_X, int BLOCK_DIM_Y, int N, int MATMUL_WARP_WIDTH, bool RECOMPUTE, int B_TILE_SIZE, int H_TILE_SIZE>
__global__ void monarch_conv_bwd_cuda_kernel(
    const at::Half *__restrict__ dout,
    const at::Half *__restrict__ a,
    const c10::complex<at::Half> *__restrict__ k_f,
    const c10::complex<at::Half> *__restrict__ b,
    const c10::complex<at::Half> *__restrict__ twiddle_factors_fft,
    const c10::complex<at::Half> *__restrict__ twid_r2r,
    const c10::complex<at::Half> *__restrict__ b_ifft,
    const c10::complex<at::Half> *__restrict__ twiddle_factors_ifft,
    at::Half *dx_out,
    c10::complex<at::Half> *dk_f_out,
    const at::Half *__restrict__ in_gate,
    const at::Half *__restrict__ out_gate,
    at::Half *din_gate,
    at::Half *dout_gate,
    uint B,
    uint H,
    uint signal_size,
    uint sqrt_N)
{

  extern __shared__ at::Half a_real[];
  at::Half *a_imag = &a_real[N];
  at::Half *a_real_2 = &a_real[2 * N];
  at::Half *a_imag_2 = &a_real[3 * N];
  at::Half *b_real = &a_real[4 * N];
  at::Half *b_imag = &a_real[5 * N];
  at::Half *b_real_2 = &a_real[6 * N];
  at::Half *b_imag_2 = &a_real[7 * N];

  const int num_threads = BLOCK_DIM_X * BLOCK_DIM_Y;
  const int thread_id = threadIdx.x + blockDim.x * threadIdx.y;
  // const int thread_id = threadIdx.x;
  const int items_per_thread_input = 2 * N / num_threads;
  const int items_per_thread_kf = N / num_threads;
  // this is for reading in the DFT matrix or twiddle factors
  const int items_per_thread_matrix = N / num_threads;
  // const int warp_id = thread_id / WARP_SIZE;

  // NOTE - we are loading and storing data in a STRIPED FORMAT
  // SEQUENCE_SIZE * TILE_SIZE items, WARP_SIZE * TILE_SIZE threads -> items_per_thread_input
  using BlockLoad_Input = cub::BlockLoad<double, BLOCK_DIM_X, items_per_thread_input / 4, cub::BLOCK_LOAD_STRIPED, BLOCK_DIM_Y>;
  using BlockLoad_Complex_Input = cub::BlockLoad<c10::complex<float>, BLOCK_DIM_X, items_per_thread_input / 2, cub::BLOCK_LOAD_STRIPED, BLOCK_DIM_Y>;
  using BlockLoad_Sequence = cub::BlockLoad<c10::complex<float>, BLOCK_DIM_X, items_per_thread_kf / 2, cub::BLOCK_LOAD_STRIPED, BLOCK_DIM_Y>;
  using BlockLoad_Filter = cub::BlockLoad<complex_half_t, BLOCK_DIM_X, items_per_thread_kf, cub::BLOCK_LOAD_STRIPED, BLOCK_DIM_Y>;
  using BlockLoad_Shared = cub::BlockLoad<c10::complex<float>, BLOCK_DIM_X, items_per_thread_matrix / 2, cub::BLOCK_LOAD_STRIPED, BLOCK_DIM_Y>; // for the DFT / Twiddle, etc
  using BlockStore_Sequence = cub::BlockStore<double, BLOCK_DIM_X, items_per_thread_input / 4, cub::BLOCK_STORE_STRIPED, BLOCK_DIM_Y>;
  using BlockStore_Sequence_Complex = cub::BlockStore<complex_half_t, BLOCK_DIM_X, items_per_thread_kf, cub::BLOCK_STORE_STRIPED, BLOCK_DIM_Y>;

  // index into block blockIdx.x
  int b_offset_signal = blockIdx.x * H * signal_size * B_TILE_SIZE;
  // index into the H
  int h_offset_signal = blockIdx.y * signal_size * H_TILE_SIZE;
  int h_offset_kernel = blockIdx.y * (N + 1) * H_TILE_SIZE;

  complex_half_t a_input_data[items_per_thread_input];    // for storing the input
  complex_half_t kf_input_data[items_per_thread_input];    // for storing the kf
  complex_half_t z_data[items_per_thread_kf];          // for storing the intermediates
  complex_half_t temp[items_per_thread_input];
  at::Half x_input_data[items_per_thread_input];     // for storing the input
  at::Half orig_input_data[items_per_thread_input];     // for storing the input
  at::Half ingate_data[items_per_thread_input];        // for storing the gates
  at::Half outgate_data[items_per_thread_input];        // for storing the gates
  at::Half dingate_data[items_per_thread_input];       // for storing the dgate
  at::Half doutgate_data[items_per_thread_input];       // for storing the dgate
  complex_half_t twid_input_data[items_per_thread_kf];    // for storing the input
  complex_half_t twid_input_data_conj[items_per_thread_kf];    // for storing the input
  complex_half_t b_input_data[items_per_thread_matrix];   // for storing matrices, twiddle factors
  complex_half_t b_input_data_2[items_per_thread_matrix]; // another place for storing matrices, twiddle factors

  // for the dft
  wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_K, WMMA_N, half, wmma::row_major> b_frag_dft[MATMUL_WARP_WIDTH][MATMUL_WARP_WIDTH][2];
  // for the idft
  wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_K, WMMA_N, half, wmma::row_major> b_frag_idft[MATMUL_WARP_WIDTH][MATMUL_WARP_WIDTH][2];
  // for the dft
  wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_K, WMMA_N, half, wmma::col_major> a_frag_dft[MATMUL_WARP_WIDTH][MATMUL_WARP_WIDTH][2];
  // for the idft
  // wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_K, WMMA_N, half, wmma::col_major> a_frag_idft[MATMUL_WARP_WIDTH][MATMUL_WARP_WIDTH][2];
  // for kernels
  // wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_K, WMMA_N, half, wmma::row_major> k_frag[MATMUL_WARP_WIDTH][MATMUL_WARP_WIDTH][2];
  // for twiddles
  wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_K, WMMA_N, half, wmma::row_major> twiddle_dft_frag[MATMUL_WARP_WIDTH][MATMUL_WARP_WIDTH][2];
  // for twiddles
  wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_K, WMMA_N, half, wmma::row_major> twiddle_idft_frag[MATMUL_WARP_WIDTH][MATMUL_WARP_WIDTH][2];

  // loads SEQUENCE_SIZE into b
  BlockLoad_Shared().Load(
      reinterpret_cast<const c10::complex<float> *>(b),
      reinterpret_cast<c10::complex<float>(&)[items_per_thread_matrix / 2]>(b_input_data)); // hopefully this interleaves things correctly

  // loads SEQUENCE_SIZE into b
  BlockLoad_Shared().Load(
      reinterpret_cast<const c10::complex<float> *>(b_ifft),
      reinterpret_cast<c10::complex<float>(&)[items_per_thread_matrix / 2]>(b_input_data_2)); // hopefully this interleaves things correctly

  int a_idx, b_idx;
  __half2 scratch;

  // load the DFT matrix into b_real, b_imag
  // this costs about 60 us
  // #pragma unroll
  for (int i = 0; i < items_per_thread_matrix / 2; i++)
  {
    b_idx = i * num_threads + thread_id;

    scratch = __half2(b_input_data[2 * i].real(), b_input_data[2 * i + 1].real());
    reinterpret_cast<__half2 *>(b_real)[b_idx] = scratch;
    scratch = __half2(b_input_data[2 * i].imag(), b_input_data[2 * i + 1].imag());
    reinterpret_cast<__half2 *>(b_imag)[b_idx] = scratch;

    scratch = __half2(b_input_data_2[2 * i].real(), b_input_data_2[2 * i + 1].real());
    reinterpret_cast<__half2 *>(b_real_2)[b_idx] = scratch;
    scratch = __half2(b_input_data_2[2 * i].imag(), b_input_data_2[2 * i + 1].imag());
    reinterpret_cast<__half2 *>(b_imag_2)[b_idx] = scratch;
  }

  __syncthreads();

  // load into twiddle factors
  // NOTE(danfu): this takes about 60 us
  BlockLoad_Shared().Load(
      reinterpret_cast<const c10::complex<float> *>(twiddle_factors_fft),
      reinterpret_cast<c10::complex<float>(&)[items_per_thread_matrix / 2]>(b_input_data));

  // start loading ifft twiddle factors
  // TODO(danfu): this costs about 60 us
  BlockLoad_Shared().Load(
      reinterpret_cast<const c10::complex<float> *>(twiddle_factors_ifft),
      reinterpret_cast<c10::complex<float>(&)[items_per_thread_matrix / 2]>(b_input_data_2));

  bool a_trans = true;
  bool b_trans = false;
  wmma::fragment<wmma::accumulator, WMMA_M, WMMA_K, WMMA_N, half> acc_frag_1[MATMUL_WARP_WIDTH][MATMUL_WARP_WIDTH][2];

  // load DFT matrix into b_frag
  #pragma unroll
  for (int j_b = 0; j_b < MATMUL_WARP_WIDTH; j_b++)
  {
    // #pragma unroll
    for (int k = 0; k < MATMUL_WARP_WIDTH; k++)
    {
      a_idx = a_trans ? j_b * WMMA_N * sqrt_N + k * WMMA_K : k * WMMA_K * sqrt_N + j_b * WMMA_N;
      b_idx = b_trans ? j_b * WMMA_N * sqrt_N + k * WMMA_K : k * WMMA_K * sqrt_N + j_b * WMMA_N;
      wmma::load_matrix_sync(a_frag_dft[k][j_b][0], reinterpret_cast<half *>(b_real) + a_idx, sqrt_N);
      wmma::load_matrix_sync(b_frag_dft[k][j_b][0], reinterpret_cast<half *>(b_real) + b_idx, sqrt_N);
      wmma::load_matrix_sync(a_frag_dft[k][j_b][1], reinterpret_cast<half *>(b_imag) + a_idx, sqrt_N);
      wmma::load_matrix_sync(b_frag_dft[k][j_b][1], reinterpret_cast<half *>(b_imag) + b_idx, sqrt_N);
    }
  }

  // load iDFT matrix into b_frag_idft
  // #pragma unroll
  for (int j_b = 0; j_b < MATMUL_WARP_WIDTH; j_b++)
  {
    // #pragma unroll
    for (int k = 0; k < MATMUL_WARP_WIDTH; k++)
    {
      // a_idx = a_trans ? j_b * WMMA_N * sqrt_N + k * WMMA_K : k * WMMA_K * sqrt_N + j_b * WMMA_N;
      b_idx = b_trans ? j_b * WMMA_N * sqrt_N + k * WMMA_K : k * WMMA_K * sqrt_N + j_b * WMMA_N;
      // wmma::load_matrix_sync(a_frag_idft[k][j_b][0], reinterpret_cast<half *>(b_real_2) + a_idx, sqrt_N);
      wmma::load_matrix_sync(b_frag_idft[k][j_b][0], reinterpret_cast<half *>(b_real_2) + b_idx, sqrt_N);
      // wmma::load_matrix_sync(a_frag_idft[k][j_b][1], reinterpret_cast<half *>(b_imag_2) + a_idx, sqrt_N);
      wmma::load_matrix_sync(b_frag_idft[k][j_b][1], reinterpret_cast<half *>(b_imag_2) + b_idx, sqrt_N);
    }
  }

  __syncthreads();

  // load twiddles into shared memory
  // load the DFT matrix into b_real, b_imag
  // this costs about 60 us
  // #pragma unroll
  for (int i = 0; i < items_per_thread_matrix / 2; i++)
  {
    b_idx = i * num_threads + thread_id;

    scratch = __half2(b_input_data[2 * i].real(), b_input_data[2 * i + 1].real());
    reinterpret_cast<__half2 *>(b_real)[b_idx] = scratch;
    scratch = __half2(b_input_data[2 * i].imag(), b_input_data[2 * i + 1].imag());
    reinterpret_cast<__half2 *>(b_imag)[b_idx] = scratch;

    scratch = __half2(b_input_data_2[2 * i].real(), b_input_data_2[2 * i + 1].real());
    reinterpret_cast<__half2 *>(b_real_2)[b_idx] = scratch;
    scratch = __half2(b_input_data_2[2 * i].imag(), b_input_data_2[2 * i + 1].imag());
    reinterpret_cast<__half2 *>(b_imag_2)[b_idx] = scratch;
  }

  __syncthreads();

  // load DFT twiddles into twiddle_dft_frag
  // #pragma unroll
  for (int j_b = 0; j_b < MATMUL_WARP_WIDTH; j_b++)
  {
    // #pragma unroll
    for (int k = 0; k < MATMUL_WARP_WIDTH; k++)
    {
      b_idx = b_trans ? j_b * WMMA_N * sqrt_N + k * WMMA_K : k * WMMA_K * sqrt_N + j_b * WMMA_N;
      wmma::load_matrix_sync(twiddle_dft_frag[k][j_b][0], reinterpret_cast<half *>(b_real) + b_idx, sqrt_N);
      wmma::load_matrix_sync(twiddle_dft_frag[k][j_b][1], reinterpret_cast<half *>(b_imag) + b_idx, sqrt_N);
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
      wmma::load_matrix_sync(twiddle_idft_frag[k][j_b][0], reinterpret_cast<half *>(b_real_2) + b_idx, sqrt_N);
      wmma::load_matrix_sync(twiddle_idft_frag[k][j_b][1], reinterpret_cast<half *>(b_imag_2) + b_idx, sqrt_N);
    }
  }

  // load twid into twid_input_data
  BlockLoad_Filter().Load(
    reinterpret_cast<const complex_half_t *>(twid_r2r),
    reinterpret_cast<complex_half_t(&)[items_per_thread_kf]>(twid_input_data)
  );

  negate_twid(&twid_input_data[0], &twid_input_data_conj[0], items_per_thread_kf);

  // #pragma unroll
  for (int h_tile_id = 0; h_tile_id < H_TILE_SIZE; h_tile_id++)
  {

    // start loading k_f
    // NOTE(danfu): this load from HBM costs about 60 us
    
    BlockLoad_Filter().Load(
        reinterpret_cast<const complex_half_t *>(k_f + h_offset_kernel + h_tile_id * (N + 1)),
        reinterpret_cast<complex_half_t(&)[items_per_thread_kf]>(kf_input_data));

    if (thread_id == 0)
    {
      // load in the pivot into the imag position
      kf_input_data[0] = complex_half_t(kf_input_data[0].real(), (k_f + h_offset_kernel + h_tile_id * (N + 1))[N].real());
    }

    for(int i=0; i< items_per_thread_input; i++) {
      temp[i] = complex_half_t(__float2half(0.0f), __float2half(0.0f));
    }

    // __syncthreads();
    // #pragma unroll
    for (int b_tile_id = 0; b_tile_id < B_TILE_SIZE; b_tile_id++)
    {

      int input_offset = h_offset_signal + b_offset_signal + h_tile_id * signal_size + b_tile_id * H * signal_size;

      // load a into x_input_data
      BlockLoad_Input().Load(
        reinterpret_cast<const double *>(a + input_offset),
        reinterpret_cast<double(&)[items_per_thread_input / 4]>(x_input_data),
        signal_size / 4, 0.
      );

      if(in_gate != nullptr) {
        // load in_gate into ingate_data
        BlockLoad_Input().Load(
          reinterpret_cast<const double *>(in_gate + input_offset),
          reinterpret_cast<double(&)[items_per_thread_input / 4]>(ingate_data),
          signal_size / 4, 0.
        );

        // put orig a into orig_input_data, and compute a = in_gate * a
        for (int i = 0; i < items_per_thread_input / 2; i++) {
          reinterpret_cast<__half2 *>(orig_input_data)[i] = reinterpret_cast<__half2 *>(x_input_data)[i];
          reinterpret_cast<__half2 *>(x_input_data)[i] = __hmul2(
            reinterpret_cast<__half2 *>(x_input_data)[i],
            reinterpret_cast<__half2 *>(ingate_data)[i]
          );
        }
      }

      // load a into a_real_2
      load_input(
        &a_real_2[0], &a_imag_2[0], &x_input_data[0],
        items_per_thread_input, num_threads, thread_id);

      __syncthreads();

      // first DFT(x)
      complex_matmul_load_b<wmma::col_major, wmma::row_major, false, false, MATMUL_WARP_WIDTH, false, false>(
          reinterpret_cast<half *>(a_real_2),  // this is the output
          reinterpret_cast<half *>(a_imag_2),  // this is the output
          sqrt_N,
          N,
          a_frag_dft,
          acc_frag_1,
          wmma::mem_row_major);

      // __syncthreads();

      // second DFT(x), with twiddle
      complex_matmul<wmma::row_major, wmma::row_major, false, true, MATMUL_WARP_WIDTH, true, true>(
          reinterpret_cast<half *>(a_real_2),
          reinterpret_cast<half *>(a_imag_2),
          sqrt_N,
          N,
          b_frag_dft,
          acc_frag_1,
          twiddle_dft_frag,
          wmma::mem_col_major);
      
      __syncthreads();

      // load dout into x_input_data
      BlockLoad_Input().Load(
        reinterpret_cast<const double *>(dout + input_offset),
        reinterpret_cast<double(&)[items_per_thread_input / 4]>(x_input_data),
        signal_size / 4, 0.
      );

      // put DFT(x) into a_input_data
      process_zf(
        &a_real_2[0], &a_imag_2[0], &a_input_data[0], &twid_input_data[0],
        items_per_thread_kf, num_threads, thread_id, N);

      if (out_gate != nullptr) { // compute dout_gate

        // multiply by kf, and put it into z_data
        multiply_kf(
          &a_input_data[0], &kf_input_data[0], &z_data[0],
          items_per_thread_kf, num_threads, thread_id);

        // put it into a_real
        store_z_data(
          &a_real[0], &a_imag[0], &z_data[0],
          items_per_thread_kf, num_threads, thread_id);

        __syncthreads();

        // process yf from a_real and put it into z_data
        process_yf(
          &a_real[0], &a_imag[0], &z_data[0], &twid_input_data_conj[0],
          items_per_thread_kf, num_threads, thread_id, N);

        // put it back into a_real
        store_z_data(
          &a_real[0], &a_imag[0], &z_data[0],
          items_per_thread_kf, num_threads, thread_id);

        // compute ifft
        complex_matmul<wmma::col_major, wmma::row_major, true, true, MATMUL_WARP_WIDTH, false, true>(
            reinterpret_cast<half *>(a_real),
            reinterpret_cast<half *>(a_imag),
            sqrt_N,
            N,
            b_frag_idft,
            acc_frag_1,
            // k_frag,
            wmma::mem_col_major);
        // __syncthreads();

        complex_matmul<wmma::row_major, wmma::row_major, false, true, MATMUL_WARP_WIDTH, false, true>(
            reinterpret_cast<half *>(a_real),
            reinterpret_cast<half *>(a_imag),
            sqrt_N,
            N,
            b_frag_idft,
            acc_frag_1,
            twiddle_idft_frag,
            wmma::mem_col_major);

        // put result into doutgate_data
        load_output(
          &a_real[0], &a_imag[0], &doutgate_data[0],
          items_per_thread_input, num_threads, thread_id);

        // load out_gate
        BlockLoad_Input().Load(
          reinterpret_cast<const double *>(out_gate + input_offset),
          reinterpret_cast<double(&)[items_per_thread_input / 4]>(outgate_data),
          signal_size / 4, 0.
        );

        // compute dout_gate = dout_gate * dout
        for (int i = 0; i < items_per_thread_input / 2; i++) {
          reinterpret_cast<__half2 *>(doutgate_data)[i] = __hmul2(
            reinterpret_cast<__half2 *>(x_input_data)[i],
            reinterpret_cast<__half2 *>(doutgate_data)[i]
          );
        }

        // compute dout = dout * out_gate
        for (int i = 0; i < items_per_thread_input / 2; i++) {
          reinterpret_cast<__half2 *>(x_input_data)[i] = __hmul2(
            reinterpret_cast<__half2 *>(x_input_data)[i],
            reinterpret_cast<__half2 *>(outgate_data)[i]
          );
        }

        __syncthreads();
      }

      // put dout from x_input_data into a_real
      load_input(
        &a_real[0], &a_imag[0], &x_input_data[0],
        items_per_thread_input, num_threads, thread_id);

      __syncthreads();

      // first DFT(dout)
      complex_matmul_load_b<wmma::col_major, wmma::row_major, false, false, MATMUL_WARP_WIDTH, false, false>(
          reinterpret_cast<half *>(a_real),  // this is the output
          reinterpret_cast<half *>(a_imag),  // this is the output
          sqrt_N,
          N,
          a_frag_dft,
          acc_frag_1,
          wmma::mem_row_major);

      // second DFT(dout), with twiddle
      complex_matmul<wmma::row_major, wmma::row_major, false, true, MATMUL_WARP_WIDTH, true, true>(
          reinterpret_cast<half *>(a_real),
          reinterpret_cast<half *>(a_imag),
          sqrt_N,
          N,
          b_frag_dft,
          acc_frag_1,
          twiddle_dft_frag,
          wmma::mem_col_major);
      
      __syncthreads();

      // put DFT(dout) into z_data
      process_zf(
        &a_real[0], &a_imag[0], &z_data[0], &twid_input_data[0],
        items_per_thread_kf, num_threads, thread_id, N);

      // DFT(x) = DFT(x) * N is in a_input_data
      for (int i = 0; i < items_per_thread_kf; i++)
      {
        reinterpret_cast<__half2 *>(a_input_data)[i] = __hmul2(
            reinterpret_cast<__half2 *>(a_input_data)[i],
            __half2(__float2half(float(N)), __float2half(float(N))));
      }

      // dk_f = dout * x.conj()
      multiply_kf_conj(
        &z_data[0], &a_input_data[0], &a_input_data[0], items_per_thread_kf, num_threads, thread_id);
      
      if (thread_id == 0) {
        reinterpret_cast<__half2 *>(a_input_data)[0] = __hmul2(
          __half2(__half(a_input_data[0].real()), __half(a_input_data[0].imag())),
          __half2(__float2half(0.5), __float2half(0.5))
        );
      }

      for(int i=0; i< items_per_thread_kf; i++) {
        temp[i] += a_input_data[i];
      }

      // multiply z_data by kf.conj()
      multiply_kf_conj(
        &z_data[0], &kf_input_data[0], &z_data[0],
        items_per_thread_kf, num_threads, thread_id);

      store_z_data(
        &a_real[0], &a_imag[0], &z_data[0],
        items_per_thread_kf, num_threads, thread_id);

      __syncthreads();

      process_yf(
        &a_real[0], &a_imag[0], &z_data[0], &twid_input_data_conj[0],
        items_per_thread_kf, num_threads, thread_id, N);

      store_z_data(
        &a_real[0], &a_imag[0], &z_data[0],
        items_per_thread_kf, num_threads, thread_id);

      __syncthreads();

      // start computing iFFT(dout), and multiply by k_frag
      complex_matmul<wmma::col_major, wmma::row_major, true, true, MATMUL_WARP_WIDTH, false, true>(
          reinterpret_cast<half *>(a_real),
          reinterpret_cast<half *>(a_imag),
          sqrt_N,
          N,
          b_frag_idft,
          acc_frag_1,
          // k_frag,
          wmma::mem_col_major);

      // second iFFT dout, and multiply by twiddle
      complex_matmul<wmma::row_major, wmma::row_major, false, true, MATMUL_WARP_WIDTH, false, true>(
          reinterpret_cast<half *>(a_real),
          reinterpret_cast<half *>(a_imag),
            // reinterpret_cast<half *>(a_real),
          // reinterpret_cast<half *>(out + input_offset),
          sqrt_N,
          N,
          b_frag_idft,
          acc_frag_1,
          twiddle_idft_frag,
          wmma::mem_col_major);

      load_output(
        &a_real[0], &a_imag[0], &x_input_data[0],
        items_per_thread_input, num_threads, thread_id);

      if (in_gate != nullptr) {
        // din_gate = dx * u, du = dx * ingate
        for (int i = 0; i < items_per_thread_input / 2; i++) {
          reinterpret_cast<__half2 *>(dingate_data)[i] = __hmul2(
            reinterpret_cast<__half2 *>(x_input_data)[i],
            reinterpret_cast<__half2 *>(orig_input_data)[i]
          );
          reinterpret_cast<__half2 *>(x_input_data)[i] = __hmul2(
            reinterpret_cast<__half2 *>(x_input_data)[i],
            reinterpret_cast<__half2 *>(ingate_data)[i]
          );
        }
        BlockStore_Sequence().Store(
          reinterpret_cast<double *>(din_gate + input_offset),
          reinterpret_cast<double(&)[items_per_thread_input / 4]>(dingate_data),
          signal_size / 4
        );
      }

      // write to HBM
      BlockStore_Sequence().Store(
        reinterpret_cast<double *>(dx_out + input_offset),
        reinterpret_cast<double(&)[items_per_thread_input / 4]>(x_input_data),
        signal_size / 4
      );

      if (out_gate != nullptr) {
        // write to HBM
        BlockStore_Sequence().Store(
          reinterpret_cast<double *>(dout_gate + input_offset),
          reinterpret_cast<double(&)[items_per_thread_input / 4]>(doutgate_data),
          signal_size / 4
        );
      }

      // __syncthreads();
    } // b_tile_id

    if (thread_id == 0) {
      complex_half_t pivot = complex_half_t(temp[0].imag(), 0.);
      temp[0] = complex_half_t(temp[0].real(), 0.);
      (dk_f_out + h_offset_kernel + blockIdx.x * H * (N + 1) + h_tile_id * (N+1))[N] = pivot;
    }

    // store dk_f
    BlockStore_Sequence_Complex().Store(
        reinterpret_cast<complex_half_t *>(dk_f_out + h_offset_kernel + blockIdx.x * H * (N + 1) + h_tile_id * (N+1)),
        reinterpret_cast<complex_half_t(&)[items_per_thread_kf]>(temp));
  }   // h_tile_id
}