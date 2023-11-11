// Copyright (c) 2023 Dan Fu, Hermann Kumbong

#include <torch/extension.h>

#include <vector>
#include <stdio.h>
#include <mma.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include "shared.h"

using namespace nvcuda;

template <int TILE_H, int K>
__global__ void butterfly_ifft_padded_cuda_kernel_64(
    const __half2 *__restrict__ x_real,
    const __half2 *__restrict__ x_imag,
    const complex_half_t *__restrict__ d_f,
    const __half2 *__restrict__ twiddle_factors_real,
    const __half2 *__restrict__ twiddle_factors_imag,
    __half2 *__restrict__ out_real,
    __half2 *__restrict__ out_gate,
    uint B,
    uint H,
    int M)
{
    const int max_idx = M / 2; //actually should be -1 since indices are 0-based but we are using < instead of <=
    const int out_offset = blockIdx.y * H * M/2 + blockIdx.z * TILE_H * M/2;
    const int in_offset = blockIdx.y * H * 64 * 32 * gridDim.x + blockIdx.z * TILE_H * 64 * 32 * gridDim.x;
    int idx;
    int t_offset;
    int out_t_offset;
    int shared_offset;
    const int N = 64;

    extern __shared__ half x_real_shared[];
    half *x_imag_shared = &x_real_shared[N * N];
    half *d_f_real = &x_imag_shared[N * N];
    half *d_f_imag = &d_f_real[N * N];
    half *twiddles_real_shared = &d_f_imag[N * N];
    half *twiddles_imag_shared = &twiddles_real_shared[N * N];
    half *out_real_shared = &twiddles_imag_shared[N * N];

    half tmp_real, tmp_imag;

    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::col_major> a_frag_real[K][4];
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::col_major> a_frag_imag[K][4];
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> tw_frag_real[4];
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> tw_frag_imag[4];
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag_real[4];
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag_imag[4];
    wmma::fragment<wmma::accumulator, 16, 16, 16, half> acc_frag_real[K];

    // #pragma unroll
    for (int i = threadIdx.y; i < N; i+=blockDim.y)
    {
        idx = i * 32 * gridDim.x + blockIdx.x * 32 + threadIdx.x;
        shared_offset = i * 32 + threadIdx.x;
        reinterpret_cast<__half2 *>(twiddles_real_shared)[shared_offset] = twiddle_factors_real[idx];
        reinterpret_cast<__half2 *>(twiddles_imag_shared)[shared_offset] = twiddle_factors_imag[idx];

        // #pragma unroll
        shared_offset = i * 64 + threadIdx.x;
        d_f_real[shared_offset] = d_f[shared_offset].real();
        d_f_imag[shared_offset] = d_f[shared_offset].imag();

        d_f_real[shared_offset + blockDim.x] = d_f[shared_offset + blockDim.x].real();
        d_f_imag[shared_offset + blockDim.x] = d_f[shared_offset + blockDim.x].imag();
    }

    __syncthreads();

    for (int i = 0; i < 4; i++)
    {
        if(i < K){
#pragma unroll
            for (int j = 0; j < 4; j++)
            {
                wmma::load_matrix_sync(a_frag_real[i][j], d_f_real + j * N * 16 + i * 16, N);
                wmma::load_matrix_sync(a_frag_imag[i][j], d_f_imag + j * N * 16 + i * 16, N);
            }
        }
        wmma::load_matrix_sync(tw_frag_real[i], twiddles_real_shared + i * N * 16 + threadIdx.y * 16, N);
        wmma::load_matrix_sync(tw_frag_imag[i], twiddles_imag_shared + i * N * 16 + threadIdx.y * 16, N);
    }

    for (int t = 0; t < TILE_H; t++)
    {

        out_t_offset = t * M/2;
        t_offset = t * 64 * 32 * gridDim.x;

        for (int i = threadIdx.y; i < N; i+=blockDim.y)
        {
            idx = i * 32 * gridDim.x + blockIdx.x * 32 + threadIdx.x;
            shared_offset = i * 32 + threadIdx.x;
            reinterpret_cast<__half2 *>(x_real_shared)[shared_offset] = x_real[idx + in_offset + t_offset];
            reinterpret_cast<__half2 *>(x_imag_shared)[shared_offset] = x_imag[idx + in_offset + t_offset];
        }

        __syncthreads();

        for (int i = 0; i < 4; i++)
        {
            wmma::load_matrix_sync(b_frag_real[i], x_real_shared + i * N * 16 + threadIdx.y * 16, N);
            wmma::load_matrix_sync(b_frag_imag[i], x_imag_shared + i * N * 16 + threadIdx.y * 16, N);
        }

        for (int j = 0; j < 4; j++)
        {
            for (int k = 0; k < tw_frag_real[j].num_elements; k++)
            {
                tmp_real = __hsub(__hmul(tw_frag_real[j].x[k], b_frag_real[j].x[k]), __hmul(tw_frag_imag[j].x[k], b_frag_imag[j].x[k]));
                tmp_imag = __hadd(__hmul(tw_frag_real[j].x[k], b_frag_imag[j].x[k]), __hmul(tw_frag_imag[j].x[k], b_frag_real[j].x[k]));
                b_frag_real[j].x[k] = tmp_real;
                b_frag_imag[j].x[k] = tmp_imag;
            }
        }

        for (int i = 0; i < K; i++)
        {
            wmma::fill_fragment(acc_frag_real[i], __float2half(0.0f));

// bd
#pragma unroll
            for (int k = 0; k < 4; k++)
            {
                wmma::mma_sync(acc_frag_real[i], a_frag_imag[i][k], b_frag_imag[k], acc_frag_real[i]);
            }

            for (int k = 0; k < acc_frag_real[i].num_elements; k++)
            {
                acc_frag_real[i].x[k] = __hneg(acc_frag_real[i].x[k]);
            }
        }

        for (int i = 0; i < K; i++)
        {
// ac - bd
#pragma unroll
            for (int k = 0; k < 4; k++)
            {
                wmma::mma_sync(acc_frag_real[i], a_frag_real[i][k], b_frag_real[k], acc_frag_real[i]);
            }
        }

#pragma unroll
        for (int i = 0; i < K; i++)
        {
            wmma::store_matrix_sync(out_real_shared + i * N * 16 + threadIdx.y * 16, acc_frag_real[i], N, wmma::mem_row_major);
        }

        __syncthreads();

#pragma unroll
        for (int i = threadIdx.y; i < N; i+=blockDim.y)
        {
            idx = i * 32 * gridDim.x + blockIdx.x * 32 + threadIdx.x;
            shared_offset = i * 32 + threadIdx.x;

            if(idx < max_idx){
                if(out_gate != nullptr)
                    out_real[out_offset + out_t_offset + idx] = __hmul2(reinterpret_cast<__half2 *>(out_real_shared)[shared_offset], out_gate[out_offset + out_t_offset + idx]);
                else
                    out_real[out_offset + out_t_offset + idx] = reinterpret_cast<__half2 *>(out_real_shared)[shared_offset];
            }
        }

        __syncthreads();
    }
}


template <int K>
__global__ void butterfly_ifft_padded_cuda_kernel_32(
    const __half2 *__restrict__ x_real,
    const __half2 *__restrict__ x_imag,
    const complex_half_t *__restrict__ d_f,
    const __half2 *__restrict__ twiddle_factors_real,
    const __half2 *__restrict__ twiddle_factors_imag,
    __half2 *__restrict__ out_real,
    __half2 *__restrict__ out_gate,
    uint B,
    uint H,
    int M)
{
    const int max_idx = M / 2; //actually should be -1 since indices are 0-based but we are using < instead of <=
    const int N  = 32;
    int idx;
    int shared_offset;

    const int out_offset  =  blockIdx.y * H * M / 2 + blockIdx.z * M / 2; 
    const int in_offset = blockIdx.y * H * 32 * 32 * gridDim.x + blockIdx.z * 32 * 32 * gridDim.x;


    __shared__ half x_real_shared[32 * 64];
    __shared__ half x_imag_shared[32 * 64];
    __shared__ half d_f_real[32 * 32];
    __shared__ half d_f_imag[32 * 32];
    __shared__ half twiddles_real_shared[32 * 64];
    __shared__ half twiddles_imag_shared[32 * 64];
    __shared__ half out_real_shared[32 * 64];

    // #pragma unroll
    for (int i = threadIdx.y; i < N; i+=blockDim.y)
    {
        idx = i * 32 * gridDim.x + blockIdx.x * 32 + threadIdx.x;
        int shared_offset = i * 32 + threadIdx.x;

        reinterpret_cast<__half2 *>(x_real_shared)[shared_offset] = x_real[in_offset  + idx];
        reinterpret_cast<__half2 *>(x_imag_shared)[shared_offset] = x_imag[in_offset  + idx];
        reinterpret_cast<__half2 *>(twiddles_real_shared)[shared_offset] = twiddle_factors_real[idx];
        reinterpret_cast<__half2 *>(twiddles_imag_shared)[shared_offset] = twiddle_factors_imag[idx];

        // #pragma unroll
        shared_offset = i * 32 + threadIdx.x;
        d_f_real[shared_offset] = d_f[shared_offset].real();
        d_f_imag[shared_offset] = d_f[shared_offset].imag();  
    }

    __syncthreads();

    if (threadIdx.y < N/16)
    {
        half tmp_real, tmp_imag;

        wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::col_major> a_frag_real[K][2];
        wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::col_major> a_frag_imag[K][2];
        wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> tw_frag_real[2][2];
        wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> tw_frag_imag[2][2];
        wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag_real[2][2];
        wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag_imag[2][2];
        wmma::fragment<wmma::accumulator, 16, 16, 16, half> acc_frag_real[K][2];

        int t = threadIdx.y * 32;

        for (int i = 0; i < 2; i++)
        {
            for (int j = 0; j < 2; j++)
            {
                if(i < K){
                    wmma::load_matrix_sync(a_frag_real[i][j], d_f_real + j * N * 16 + i * 16, N);
                    wmma::load_matrix_sync(a_frag_imag[i][j], d_f_imag + j * N * 16 + i * 16, N);
                }
                wmma::load_matrix_sync(b_frag_real[i][j], x_real_shared + i * 2 * N * 16 + j * 16 + t, 2 * N);
                wmma::load_matrix_sync(b_frag_imag[i][j], x_imag_shared + i * 2 * N * 16 + j * 16 + t, 2 * N);
                wmma::load_matrix_sync(tw_frag_real[i][j], twiddles_real_shared + i * 2 * N * 16 + j * 16 + t, 2 * N);
                wmma::load_matrix_sync(tw_frag_imag[i][j], twiddles_imag_shared + i * 2 * N * 16 + j * 16 + t, 2 * N);
            }
        }

        for (int i = 0; i < 2; i++)
        {
            for (int j = 0; j < 2; j++)
            {
                for (int k = 0; k < tw_frag_real[i][j].num_elements; k++)
                {
                    tmp_real = __hsub(__hmul(tw_frag_real[i][j].x[k], b_frag_real[i][j].x[k]), __hmul(tw_frag_imag[i][j].x[k], b_frag_imag[i][j].x[k]));
                    tmp_imag = __hadd(__hmul(tw_frag_real[i][j].x[k], b_frag_imag[i][j].x[k]), __hmul(tw_frag_imag[i][j].x[k], b_frag_real[i][j].x[k]));
                    b_frag_real[i][j].x[k] = tmp_real;
                    b_frag_imag[i][j].x[k] = tmp_imag;
                }
            }
        }
 
        for (int i = 0; i < K; i++)
        {
            for (int j = 0; j < 2; j++)
            {
                wmma::fill_fragment(acc_frag_real[i][j], __float2half(0.0f));

                // bd
                for (int k = 0; k < 2; k++)
                {
                    wmma::mma_sync(acc_frag_real[i][j], a_frag_imag[i][k], b_frag_imag[k][j], acc_frag_real[i][j]);
                }

                for (int k = 0; k < acc_frag_real[i][j].num_elements; k++)
                {
                    acc_frag_real[i][j].x[k] = __hneg(acc_frag_real[i][j].x[k]);
                }
            }
        }

        for (int i = 0; i < K; i++)
        {
            for (int j = 0; j < 2; j++)
            {
                // ac - bd
                for (int k = 0; k < 2; k++)
                {
                    wmma::mma_sync(acc_frag_real[i][j], a_frag_real[i][k], b_frag_real[k][j], acc_frag_real[i][j]);
                }
            }
        }

        for (int i = 0; i < K; i++)
        {
            for (int j = 0; j < 2; j++)
            {
                wmma::store_matrix_sync(out_real_shared + i * 2 * N * 16 + j * 16 + t, acc_frag_real[i][j], 2 * N, wmma::mem_row_major);
            }
        }
    }

    __syncthreads();

#pragma unroll
    for (int i = threadIdx.y; i < N; i+=blockDim.y)
    {
        idx = i * 32 * gridDim.x + blockIdx.x * 32 + threadIdx.x;
        shared_offset = i * 32 + threadIdx.x;

        if(idx < max_idx){
            if(out_gate != nullptr){
                out_real[idx +  out_offset] = __hmul2(reinterpret_cast<__half2 *>(out_real_shared)[shared_offset], out_gate[idx +  out_offset]);
            }else{
                out_real[idx +  out_offset] = reinterpret_cast<__half2 *>(out_real_shared)[shared_offset];
            }
        }
        
    }
}


template <int TILE_H, int K>
__global__ void butterfly_ifft_padded_cuda_kernel_128(
    const __half2 *__restrict__ x_real,
    const __half2 *__restrict__ x_imag,
    const complex_half_t *__restrict__ d_f,
    const __half2 *__restrict__ twiddle_factors_real,
    const __half2 *__restrict__ twiddle_factors_imag,
    __half2 *__restrict__ out_real,
    __half2 *__restrict__ out_gate,
    uint B,
    uint H,
    int M)
{
    const int max_idx = M / 2; //actually should be -1 since indices are 0-based but we are using < instead of <=
    const int out_offset = blockIdx.y * H * M/2 + blockIdx.z * TILE_H *  M/2;
    const int in_offset = blockIdx.y * H * 128 * 32 * 2 * gridDim.x + blockIdx.z * TILE_H * 128 * 32 *  2 * gridDim.x;
    const int N = 128;
    int idx;
    int t_offset;
    int out_t_offset;
    int shared_offset;


    extern __shared__ half real_shared[];
    half *imag_shared = &real_shared[128 * 128];
    half *real_shared_2 = &imag_shared[128 * 128];
    half *imag_shared_2 = &real_shared_2[128 * 128];

    half tmp_real, tmp_imag;

    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::col_major> a_frag[K][8];
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> tw_frag_real[8];
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> tw_frag_imag[8];
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag_real[8];
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag_imag[8];
    wmma::fragment<wmma::accumulator, 16, 16, 16, half> acc_frag_real[K];

    for (int i = threadIdx.y; i < N; i+=blockDim.y)
    {
        for(int j=0; j< 4; j++){
            shared_offset = i * 128 + threadIdx.x + j * blockDim.x;
            real_shared_2[shared_offset] = d_f[shared_offset].real();
            imag_shared_2[shared_offset] = d_f[shared_offset].imag();
        }
    }

    for (int i = threadIdx.y; i < N; i+=blockDim.y)
    {
        for(int j=0; j< 2; j++){
            idx = i * 32 * 2 * gridDim.x + j * blockDim.x + blockIdx.x * 64 + threadIdx.x;
            shared_offset = i * 64 + threadIdx.x + j * blockDim.x;    
            reinterpret_cast<__half2*>(real_shared)[shared_offset] = twiddle_factors_real[idx];
            reinterpret_cast<__half2*>(imag_shared)[shared_offset] = twiddle_factors_imag[idx];
        }
    }

    __syncthreads();


    for (int i = 0; i < 8; i++){
        wmma::load_matrix_sync(tw_frag_real[i], real_shared + i * 128 * 16 + threadIdx.y * 16, 128);
        wmma::load_matrix_sync(tw_frag_imag[i], imag_shared + i * 128 * 16 + threadIdx.y * 16, 128);
    }

    __syncthreads();

    for (int t = 0; t < TILE_H; t++)
    {

        out_t_offset = t * M/2;
        t_offset = t * 128 * 32 * 2  * gridDim.x;

        for (int i = 0; i < K; i++){
            for (int j = 0; j < 8; j++){
                wmma::load_matrix_sync(a_frag[i][j], imag_shared_2 + j * 128 * 16 + i * 16, 128);
            }
        }

        for (int i = threadIdx.y; i < N; i+=blockDim.y)
        {
            for(int j=0; j< 2; j++){
                idx = i * 32 * 2 * gridDim.x + j * blockDim.x + blockIdx.x * 64 + threadIdx.x;
                shared_offset = i * 64 + threadIdx.x + j * blockDim.x;  
                reinterpret_cast<__half2*>(real_shared)[shared_offset] = x_real[idx + in_offset + t_offset];
                reinterpret_cast<__half2*>(imag_shared)[shared_offset] = x_imag[idx + in_offset + t_offset];
            }
        }

        __syncthreads();

        for (int i = 0; i < 8; i++)
        {
            wmma::load_matrix_sync(b_frag_real[i], real_shared + i * N * 16 + threadIdx.y * 16, N);
            wmma::load_matrix_sync(b_frag_imag[i], imag_shared + i * N * 16 + threadIdx.y * 16, N);
        }


        for (int j = 0; j < 8; j++)
        {
            for (int k = 0; k < tw_frag_real[j].num_elements; k++)
            {
                tmp_real = __hsub(__hmul(tw_frag_real[j].x[k], b_frag_real[j].x[k]), __hmul(tw_frag_imag[j].x[k], b_frag_imag[j].x[k]));
                tmp_imag = __hadd(__hmul(tw_frag_real[j].x[k], b_frag_imag[j].x[k]), __hmul(tw_frag_imag[j].x[k], b_frag_real[j].x[k]));
                b_frag_real[j].x[k] = tmp_real;
                b_frag_imag[j].x[k] = tmp_imag;
            }
        }

        for (int i = 0; i < K; i++)
        {
            wmma::fill_fragment(acc_frag_real[i], __float2half(0.0f));

// bd
#pragma unroll
            for (int k = 0; k < 8; k++)
            {
                wmma::mma_sync(acc_frag_real[i], a_frag[i][k], b_frag_imag[k], acc_frag_real[i]);
            }

            for (int k = 0; k < acc_frag_real[i].num_elements; k++)
            {
                acc_frag_real[i].x[k] = __hneg(acc_frag_real[i].x[k]);
            }
        }

        for (int i = 0; i < K; i++){
            for (int j = 0; j < 8; j++){
                wmma::load_matrix_sync(a_frag[i][j], real_shared_2 + j * 128 * 16 + i * 16, 128);
            }
        }

        for (int i = 0; i < K; i++)
        {
// ac - bd
#pragma unroll
            for (int k = 0; k < 8; k++)
            {
                wmma::mma_sync(acc_frag_real[i], a_frag[i][k], b_frag_real[k], acc_frag_real[i]);
            }
        }

#pragma unroll
        for (int i = 0; i < K; i++)
        {
            //wmma::store_matrix_sync(real_shared + i * N * 16 + threadIdx.y * 16, acc_frag_real[i], N, wmma::mem_row_major);
            wmma::store_matrix_sync(real_shared + i * N * 16 + threadIdx.y * 16, acc_frag_real[i], N, wmma::mem_row_major);
        }

        __syncthreads();

#pragma unroll
        for (int i = threadIdx.y; i < N; i+=blockDim.y)
        {
            for(int j=0; j< 2; j++){
                idx = i * 32 * 2 * gridDim.x + j * blockDim.x + blockIdx.x * 64 + threadIdx.x;
                shared_offset = i * 64 + threadIdx.x + j * blockDim.x;
                if(idx < max_idx){
                    if(out_gate != nullptr){
                        out_real[idx + out_offset + out_t_offset] = __hmul2(reinterpret_cast<__half2*>(real_shared)[shared_offset], out_gate[idx + out_offset + out_t_offset]);
                    }else{
                        out_real[idx + out_offset + out_t_offset] = reinterpret_cast<__half2*>(real_shared)[shared_offset];
                    }
                }
            }
        }

        __syncthreads();
    }
}


__global__ void butterfly_ifft_padded_cuda_kernel_16(
    const __half2 *__restrict__ x_real,
    const __half2 *__restrict__ x_imag,
    const complex_half_t *__restrict__ d_f,
    const __half2 *__restrict__ twiddle_factors_real,
    const __half2 *__restrict__ twiddle_factors_imag,
    __half2 *__restrict__ out_real,
    __half2 *__restrict__ out_gate,
    uint B,
    uint H,
    int M)
{
    const int max_idx = M / 2; //actually should be -1 since indices are 0-based but we are using < instead of <=
    const int N  = 16;
    const int out_offset  =  blockIdx.y * H * M / 2 + blockIdx.z * M / 2; 
    const int offset = blockIdx.y * H * N * blockDim.x * gridDim.x + blockIdx.z * N * blockDim.x * gridDim.x;

    __shared__ half x_real_shared[N * 64];
    __shared__ half x_imag_shared[N * 64];
    __shared__ half d_f_real[N * N];
    __shared__ half d_f_imag[N * N];
    __shared__ half twiddles_real_shared[N * 64];
    __shared__ half twiddles_imag_shared[N * 64];
    __shared__ half out_real_shared[N * 64];

    // #pragma unroll
    for (int i = threadIdx.y; i < N; i++)
    {
        int idx = i * blockDim.x * gridDim.x + blockIdx.x * blockDim.x + threadIdx.x;
        int shared_offset = i * blockDim.x + threadIdx.x;
        reinterpret_cast<__half2 *>(x_real_shared)[shared_offset] = x_real[idx + offset];
        reinterpret_cast<__half2 *>(x_imag_shared)[shared_offset] = x_imag[idx + offset];
        reinterpret_cast<__half2 *>(twiddles_real_shared)[shared_offset] = twiddle_factors_real[idx];
        reinterpret_cast<__half2 *>(twiddles_imag_shared)[shared_offset] = twiddle_factors_imag[idx];

        if(threadIdx.x  < 16 ){
            shared_offset = i * 16 + threadIdx.x;
            d_f_real[shared_offset] = d_f[shared_offset].real();
            d_f_imag[shared_offset] = d_f[shared_offset].imag();
        }
    }

    __syncthreads();

    //check if it is better to have one warp do all the multiplication or split between warps
    if (threadIdx.y < 4)
    {
        half tmp_real, tmp_imag;

        wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::col_major> a_frag_real;
        wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::col_major> a_frag_imag;
        wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> tw_frag_real;
        wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> tw_frag_imag;
        wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag_real;
        wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag_imag;
        wmma::fragment<wmma::accumulator, 16, 16, 16, half> acc_frag_real;

        wmma::load_matrix_sync(a_frag_real, d_f_real, N);
        wmma::load_matrix_sync(a_frag_imag, d_f_imag, N);
        wmma::load_matrix_sync(b_frag_real, x_real_shared + threadIdx.y * 16, 64);
        wmma::load_matrix_sync(b_frag_imag, x_imag_shared + threadIdx.y * 16, 64);
        wmma::load_matrix_sync(tw_frag_real, twiddles_real_shared + threadIdx.y * 16, 64);
        wmma::load_matrix_sync(tw_frag_imag, twiddles_imag_shared + threadIdx.y * 16, 64);
     


        for (int k = 0; k < tw_frag_real.num_elements; k++)
        {
            tmp_real = __hsub(__hmul(tw_frag_real.x[k], b_frag_real.x[k]), __hmul(tw_frag_imag.x[k], b_frag_imag.x[k]));
            tmp_imag = __hadd(__hmul(tw_frag_real.x[k], b_frag_imag.x[k]), __hmul(tw_frag_imag.x[k], b_frag_real.x[k]));
            b_frag_real.x[k] = tmp_real;
            b_frag_imag.x[k] = tmp_imag;
        }
     

        wmma::fill_fragment(acc_frag_real, __float2half(0.0f));
  
        wmma::mma_sync(acc_frag_real, a_frag_imag, b_frag_imag, acc_frag_real);

        for(int k=0; k< acc_frag_real.num_elements; k++){
            acc_frag_real.x[k] = __hneg(acc_frag_real.x[k]);
        }
    

        wmma::mma_sync(acc_frag_real, a_frag_real, b_frag_real, acc_frag_real);

        wmma::store_matrix_sync(out_real_shared + threadIdx.y * 16, acc_frag_real, 64, wmma::mem_row_major);
 
    }

    __syncthreads();

#pragma unroll
    for (int i = threadIdx.y; i < N; i++)
    {
        int idx = i * blockDim.x * gridDim.x + blockIdx.x * blockDim.x + threadIdx.x;
        if(idx < max_idx){
            if(out_gate != nullptr){
                out_real[out_offset + idx] = __hmul2(reinterpret_cast<__half2 *>(out_real_shared)[i * 32 + threadIdx.x], out_gate[out_offset + idx]);
            }
            else{
                out_real[out_offset + idx] = reinterpret_cast<__half2 *>(out_real_shared)[i * 32 + threadIdx.x];
            }
        }
    }
}

torch::Tensor butterfly_ifft_padded_cuda(
    torch::Tensor x_real,
    torch::Tensor x_imag,
    torch::Tensor d_f,
    torch::Tensor twiddle_factors_real,
    torch::Tensor twiddle_factors_imag,
    int fft_size,
    std::optional<at::Tensor> out_gate = std::nullopt
    )
{

    uint B = x_real.size(0);
    uint H = x_real.size(1);
    uint N_M = x_real.size(2);
    const int d_f_size = d_f.size(0);
    // const int TILE_SIZE = 16;

    dim3 gridDim;
    dim3 blockDim;

    // uint N = x_real.size(2);
    gridDim.y = B;

    blockDim.x = 32;
    blockDim.y = 4;
    gridDim.x = 512 / (32 * 1024/ (N_M / d_f_size));
    gridDim.z = H;

    const int TILE_H = 16;
    torch::Tensor out_real = torch::empty({B, H, fft_size}, x_real.options());
    const int K = ceil(fft_size / (1.0 * 16 * (N_M / d_f_size)));

    switch(d_f_size){
        case 16:
            butterfly_ifft_padded_cuda_kernel_16<<<gridDim, blockDim>>>(
                    static_cast<__half2 *>(x_real.data_ptr()),
                    static_cast<__half2 *>(x_imag.data_ptr()),
                    static_cast<complex_half_t *>(d_f.data_ptr()),
                    static_cast<__half2 *>(twiddle_factors_real.data_ptr()),
                    static_cast<__half2 *>(twiddle_factors_imag.data_ptr()),
                    static_cast<__half2 *>(out_real.data_ptr()),
                    out_gate ? static_cast<__half2 *>(out_gate.value().data_ptr()) : nullptr,
                    B,
                    H,
                    fft_size
                );
            break;
        case 32:
            switch (K)
            {
            case 1:
                butterfly_ifft_padded_cuda_kernel_32<1><<<gridDim, blockDim>>>(
                    static_cast<__half2 *>(x_real.data_ptr()),
                    static_cast<__half2 *>(x_imag.data_ptr()),
                    static_cast<complex_half_t *>(d_f.data_ptr()),
                    static_cast<__half2 *>(twiddle_factors_real.data_ptr()),
                    static_cast<__half2 *>(twiddle_factors_imag.data_ptr()),
                    static_cast<__half2 *>(out_real.data_ptr()),
                    out_gate ? static_cast<__half2 *>(out_gate.value().data_ptr()) : nullptr,
                    B,
                    H,
                    fft_size
                );
                break;
            case 2:
                butterfly_ifft_padded_cuda_kernel_32<2><<<gridDim, blockDim>>>(
                    static_cast<__half2 *>(x_real.data_ptr()),
                    static_cast<__half2 *>(x_imag.data_ptr()),
                    static_cast<complex_half_t *>(d_f.data_ptr()),
                    static_cast<__half2 *>(twiddle_factors_real.data_ptr()),
                    static_cast<__half2 *>(twiddle_factors_imag.data_ptr()),
                    static_cast<__half2 *>(out_real.data_ptr()),
                    out_gate ? static_cast<__half2 *>(out_gate.value().data_ptr()) : nullptr,
                    B,
                    H,
                    fft_size
                );
                break;
            default:
                printf("Invalid K: %d\n", K);
                break;
            }
            break;

        case 64:
            gridDim.z = H / TILE_H;
            switch (K)
            {
            case 1:
                cudaFuncSetAttribute(&butterfly_ifft_padded_cuda_kernel_64<TILE_H, 1>, cudaFuncAttributeMaxDynamicSharedMemorySize, 65536);
                butterfly_ifft_padded_cuda_kernel_64<TILE_H, 1><<<gridDim, blockDim, 65536>>>(
                    static_cast<__half2 *>(x_real.data_ptr()),
                    static_cast<__half2 *>(x_imag.data_ptr()),
                    static_cast<complex_half_t *>(d_f.data_ptr()),
                    static_cast<__half2 *>(twiddle_factors_real.data_ptr()),
                    static_cast<__half2 *>(twiddle_factors_imag.data_ptr()),
                    static_cast<__half2 *>(out_real.data_ptr()),
                    out_gate ? static_cast<__half2 *>(out_gate.value().data_ptr()) : nullptr,
                    B,
                    H,
                    fft_size);
                break;

            case 2:
                cudaFuncSetAttribute(&butterfly_ifft_padded_cuda_kernel_64<TILE_H, 2>, cudaFuncAttributeMaxDynamicSharedMemorySize, 65536);
                butterfly_ifft_padded_cuda_kernel_64<TILE_H, 2><<<gridDim, blockDim, 65536>>>(
                    static_cast<__half2 *>(x_real.data_ptr()),
                    static_cast<__half2 *>(x_imag.data_ptr()),
                    static_cast<complex_half_t *>(d_f.data_ptr()),
                    static_cast<__half2 *>(twiddle_factors_real.data_ptr()),
                    static_cast<__half2 *>(twiddle_factors_imag.data_ptr()),
                    static_cast<__half2 *>(out_real.data_ptr()),
                    out_gate ? static_cast<__half2 *>(out_gate.value().data_ptr()) : nullptr,
                    B,
                    H,
                    fft_size);
                break;

            case 3:
                cudaFuncSetAttribute(&butterfly_ifft_padded_cuda_kernel_64<TILE_H, 3>, cudaFuncAttributeMaxDynamicSharedMemorySize, 65536);
                butterfly_ifft_padded_cuda_kernel_64<TILE_H, 3><<<gridDim, blockDim, 65536>>>(
                    static_cast<__half2 *>(x_real.data_ptr()),
                    static_cast<__half2 *>(x_imag.data_ptr()),
                    static_cast<complex_half_t *>(d_f.data_ptr()),
                    static_cast<__half2 *>(twiddle_factors_real.data_ptr()),
                    static_cast<__half2 *>(twiddle_factors_imag.data_ptr()),
                    static_cast<__half2 *>(out_real.data_ptr()),
                    out_gate ? static_cast<__half2 *>(out_gate.value().data_ptr()) : nullptr,
                    B,
                    H,
                    fft_size);
                break;

            case 4:
                cudaFuncSetAttribute(&butterfly_ifft_padded_cuda_kernel_64<TILE_H, 4>, cudaFuncAttributeMaxDynamicSharedMemorySize, 65536);
                butterfly_ifft_padded_cuda_kernel_64<TILE_H, 4><<<gridDim, blockDim, 65536>>>(
                    static_cast<__half2 *>(x_real.data_ptr()),
                    static_cast<__half2 *>(x_imag.data_ptr()),
                    static_cast<complex_half_t *>(d_f.data_ptr()),
                    static_cast<__half2 *>(twiddle_factors_real.data_ptr()),
                    static_cast<__half2 *>(twiddle_factors_imag.data_ptr()),
                    static_cast<__half2 *>(out_real.data_ptr()),
                    out_gate ? static_cast<__half2 *>(out_gate.value().data_ptr()) : nullptr,
                    B,
                    H,
                    fft_size);
                break;
            
            default:
                break;
            }
            
            break;
        case 128:
            blockDim.x = 32;
            blockDim.y = 8;
            gridDim.x = 256 / (32 * 1024/ (N_M / d_f_size));
            gridDim.z = H / TILE_H;

            switch (K)
            {
            case 1:
                cudaFuncSetAttribute(&butterfly_ifft_padded_cuda_kernel_128<TILE_H, 1>, cudaFuncAttributeMaxDynamicSharedMemorySize, 65536 * 2);

                butterfly_ifft_padded_cuda_kernel_128<TILE_H, 1><<<gridDim, blockDim, 65536 * 2>>>(
                    static_cast<__half2 *>(x_real.data_ptr()),
                    static_cast<__half2 *>(x_imag.data_ptr()),
                    static_cast<complex_half_t *>(d_f.data_ptr()),
                    static_cast<__half2 *>(twiddle_factors_real.data_ptr()),
                    static_cast<__half2 *>(twiddle_factors_imag.data_ptr()),
                    static_cast<__half2 *>(out_real.data_ptr()),
                    out_gate ? static_cast<__half2 *>(out_gate.value().data_ptr()) : nullptr,
                    B,
                    H,
                    fft_size);
                break;

            case 2:
                cudaFuncSetAttribute(&butterfly_ifft_padded_cuda_kernel_128<TILE_H, 2>, cudaFuncAttributeMaxDynamicSharedMemorySize, 65536 * 2);

                butterfly_ifft_padded_cuda_kernel_128<TILE_H, 2><<<gridDim, blockDim, 65536 * 2>>>(
                    static_cast<__half2 *>(x_real.data_ptr()),
                    static_cast<__half2 *>(x_imag.data_ptr()),
                    static_cast<complex_half_t *>(d_f.data_ptr()),
                    static_cast<__half2 *>(twiddle_factors_real.data_ptr()),
                    static_cast<__half2 *>(twiddle_factors_imag.data_ptr()),
                    static_cast<__half2 *>(out_real.data_ptr()),
                    out_gate ? static_cast<__half2 *>(out_gate.value().data_ptr()) : nullptr,
                    B,
                    H,
                    fft_size);
                break;

            case 3:
                cudaFuncSetAttribute(&butterfly_ifft_padded_cuda_kernel_128<TILE_H, 3>, cudaFuncAttributeMaxDynamicSharedMemorySize, 65536 * 2);

                butterfly_ifft_padded_cuda_kernel_128<TILE_H, 3><<<gridDim, blockDim, 65536 * 2>>>(
                    static_cast<__half2 *>(x_real.data_ptr()),
                    static_cast<__half2 *>(x_imag.data_ptr()),
                    static_cast<complex_half_t *>(d_f.data_ptr()),
                    static_cast<__half2 *>(twiddle_factors_real.data_ptr()),
                    static_cast<__half2 *>(twiddle_factors_imag.data_ptr()),
                    static_cast<__half2 *>(out_real.data_ptr()),
                    out_gate ? static_cast<__half2 *>(out_gate.value().data_ptr()) : nullptr,
                    B,
                    H,
                    fft_size);
                break;

            case 4:
                cudaFuncSetAttribute(&butterfly_ifft_padded_cuda_kernel_128<TILE_H, 4>, cudaFuncAttributeMaxDynamicSharedMemorySize, 65536 * 2);

                butterfly_ifft_padded_cuda_kernel_128<TILE_H, 4><<<gridDim, blockDim, 65536 * 2>>>(
                    static_cast<__half2 *>(x_real.data_ptr()),
                    static_cast<__half2 *>(x_imag.data_ptr()),
                    static_cast<complex_half_t *>(d_f.data_ptr()),
                    static_cast<__half2 *>(twiddle_factors_real.data_ptr()),
                    static_cast<__half2 *>(twiddle_factors_imag.data_ptr()),
                    static_cast<__half2 *>(out_real.data_ptr()),
                    out_gate ? static_cast<__half2 *>(out_gate.value().data_ptr()) : nullptr,
                    B,
                    H,
                    fft_size);
                break;

            case 5:
                cudaFuncSetAttribute(&butterfly_ifft_padded_cuda_kernel_128<TILE_H, 5>, cudaFuncAttributeMaxDynamicSharedMemorySize, 65536 * 2);

                butterfly_ifft_padded_cuda_kernel_128<TILE_H, 5><<<gridDim, blockDim, 65536 * 2>>>(
                    static_cast<__half2 *>(x_real.data_ptr()),
                    static_cast<__half2 *>(x_imag.data_ptr()),
                    static_cast<complex_half_t *>(d_f.data_ptr()),
                    static_cast<__half2 *>(twiddle_factors_real.data_ptr()),
                    static_cast<__half2 *>(twiddle_factors_imag.data_ptr()),
                    static_cast<__half2 *>(out_real.data_ptr()),
                    out_gate ? static_cast<__half2 *>(out_gate.value().data_ptr()) : nullptr,
                    B,
                    H,
                    fft_size);
                break;

            case 6:
                cudaFuncSetAttribute(&butterfly_ifft_padded_cuda_kernel_128<TILE_H, 6>, cudaFuncAttributeMaxDynamicSharedMemorySize, 65536 * 2);

                butterfly_ifft_padded_cuda_kernel_128<TILE_H, 6><<<gridDim, blockDim, 65536 * 2>>>(
                    static_cast<__half2 *>(x_real.data_ptr()),
                    static_cast<__half2 *>(x_imag.data_ptr()),
                    static_cast<complex_half_t *>(d_f.data_ptr()),
                    static_cast<__half2 *>(twiddle_factors_real.data_ptr()),
                    static_cast<__half2 *>(twiddle_factors_imag.data_ptr()),
                    static_cast<__half2 *>(out_real.data_ptr()),
                    out_gate ? static_cast<__half2 *>(out_gate.value().data_ptr()) : nullptr,
                    B,
                    H,
                    fft_size);
                break;

            case 7:
                cudaFuncSetAttribute(&butterfly_ifft_padded_cuda_kernel_128<TILE_H, 7>, cudaFuncAttributeMaxDynamicSharedMemorySize, 65536 * 2);

                butterfly_ifft_padded_cuda_kernel_128<TILE_H, 7><<<gridDim, blockDim, 65536 * 2>>>(
                    static_cast<__half2 *>(x_real.data_ptr()),
                    static_cast<__half2 *>(x_imag.data_ptr()),
                    static_cast<complex_half_t *>(d_f.data_ptr()),
                    static_cast<__half2 *>(twiddle_factors_real.data_ptr()),
                    static_cast<__half2 *>(twiddle_factors_imag.data_ptr()),
                    static_cast<__half2 *>(out_real.data_ptr()),
                    out_gate ? static_cast<__half2 *>(out_gate.value().data_ptr()) : nullptr,
                    B,
                    H,
                    fft_size);
                break;

            case 8:
                cudaFuncSetAttribute(&butterfly_ifft_padded_cuda_kernel_128<TILE_H, 8>, cudaFuncAttributeMaxDynamicSharedMemorySize, 65536 * 2);

                butterfly_ifft_padded_cuda_kernel_128<TILE_H, 8><<<gridDim, blockDim, 65536 * 2>>>(
                    static_cast<__half2 *>(x_real.data_ptr()),
                    static_cast<__half2 *>(x_imag.data_ptr()),
                    static_cast<complex_half_t *>(d_f.data_ptr()),
                    static_cast<__half2 *>(twiddle_factors_real.data_ptr()),
                    static_cast<__half2 *>(twiddle_factors_imag.data_ptr()),
                    static_cast<__half2 *>(out_real.data_ptr()),
                    out_gate ? static_cast<__half2 *>(out_gate.value().data_ptr()) : nullptr,
                    B,
                    H,
                    fft_size);
                break;
            
            default:
                printf("Invalid K: %d\n", K);
                break;
            }
            break;

        default:
            printf("Invalid d_f_size: %d\n", d_f_size);
            break;
    }
    
    return out_real;
}
