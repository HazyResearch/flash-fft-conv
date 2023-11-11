// Copyright (c) 2023 Dan Fu, Hermann Kumbong

#include <torch/extension.h>

#include <vector>
#include <stdio.h>
#include <mma.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include "shared.h"

using namespace nvcuda;


template <int K>
__global__ void butterfly_cuda_kernel_64(
    const __nv_bfloat162 *__restrict__ x,
    const __nv_bfloat162 *__restrict__ x_gate,
    const __nv_bfloat162 *__restrict__ d_f_real,
    const __nv_bfloat162 *__restrict__ d_f_imag,
    const __nv_bfloat162 *__restrict__ twiddle_factors_real,
    const __nv_bfloat162 *__restrict__ twiddle_factors_imag,
    __nv_bfloat162 *__restrict__ out_real,
    __nv_bfloat162 *__restrict__ out_imag,
    uint B,
    uint H,
    int M)
{
    const int max_idx = M / 2; //actually should be -1 since indices are 0-based but we are using < instead of <=
    const int offset = blockIdx.y * H * M/2 + blockIdx.z * 16 *  M/2;
    const int out_offset = blockIdx.y * H * 64 * 32 * gridDim.x + blockIdx.z * 16 * 64 * 32 * gridDim.x;
    int idx;
    int t_offset;
    int out_t_offset;
    int shared_offset;
    const int N = 64;
    

    extern __shared__ __nv_bfloat16 x_shared[];
    __nv_bfloat16 *d_f_real_shared = &x_shared[K * 16 * N];
    __nv_bfloat16 *d_f_imag_shared = &d_f_real_shared[N * N];
    __nv_bfloat16 *twiddles_real_shared = &d_f_imag_shared[N * N];
    __nv_bfloat16 *twiddles_imag_shared = &twiddles_real_shared[N * N];
    float *out_real_shared = reinterpret_cast<float*>(&twiddles_imag_shared[N * N]);
    float *out_imag_shared = &out_real_shared[N * N];

    // #pragma unroll
    for (int i = threadIdx.y; i < N; i+=blockDim.y)
    {
        idx = i * 32 * gridDim.x + blockIdx.x * 32 + threadIdx.x;
        shared_offset = i * 32 + threadIdx.x;
        reinterpret_cast<__nv_bfloat162 *>(twiddles_real_shared)[shared_offset] = twiddle_factors_real[idx];
        reinterpret_cast<__nv_bfloat162 *>(twiddles_imag_shared)[shared_offset] = twiddle_factors_imag[idx];

        // #pragma unroll
        shared_offset = i * 32 + threadIdx.x;
        reinterpret_cast<__nv_bfloat162 *>(d_f_real_shared)[shared_offset] = d_f_real[shared_offset];
        reinterpret_cast<__nv_bfloat162 *>(d_f_imag_shared)[shared_offset] = d_f_imag[shared_offset];
    }

    float2 tmp_real, tmp_imag;

    wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::col_major> a_frag_real[4];
    wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> tw_frag_real[4];
    wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> tw_frag_imag[4];
    wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::col_major> a_frag_imag[4];
    wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::row_major> b_frag[4][4];
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc_frag_real[4];
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc_frag_imag[4];

    __syncthreads();

    for (int i = 0; i < 4; i++)
    {
        wmma::load_matrix_sync(a_frag_real[i], d_f_real_shared + i * N * 16 + threadIdx.y * 16, N);
        wmma::load_matrix_sync(a_frag_imag[i], d_f_imag_shared + i * N * 16 + threadIdx.y * 16, N);
        wmma::load_matrix_sync(tw_frag_real[i], twiddles_real_shared + threadIdx.y * N * 16 + i * 16, N);
        wmma::load_matrix_sync(tw_frag_imag[i], twiddles_imag_shared + threadIdx.y * N * 16 + i * 16, N);
    }

    for (int t = 0; t < 16; t++)
    {
        t_offset = t * M/2;
        out_t_offset = t * 64 * 32 * gridDim.x;

        for (int i = threadIdx.y; i < N; i+=blockDim.y)
        {
            if(i < K * 16){
                idx = i * 32 * gridDim.x + blockIdx.x * 32 + threadIdx.x;
                shared_offset = i * 32 + threadIdx.x;
                if(x_gate != nullptr){
                    reinterpret_cast<__nv_bfloat162 *>(x_shared)[shared_offset] = idx < max_idx ? __hmul2(x[idx + offset + t_offset], x_gate[idx + offset + t_offset]) : __floats2bfloat162_rn(0.0f, 0.0f);
                }else{
                    reinterpret_cast<__nv_bfloat162 *>(x_shared)[shared_offset] = idx < max_idx ? x[idx + offset + t_offset] : __floats2bfloat162_rn(0.0f, 0.0f);
                }
            }
        }

        __syncthreads();

        for (int i = 0; i < K; i++)
        {
            for (int j = 0; j < 4; j++)
            {
                wmma::load_matrix_sync(b_frag[i][j], x_shared + i * N * 16 + j * 16, N);
            }
        }

#pragma unroll
        for (int j = 0; j < 4; j++)
        {
            wmma::fill_fragment(acc_frag_real[j], 0.0f);

            for (int k = 0; k < K; k++)
            {
                wmma::mma_sync(acc_frag_real[j], a_frag_real[k], b_frag[k][j], acc_frag_real[j]);
            }
        }

#pragma unroll

        for (int j = 0; j < 4; j++)
        {
            wmma::fill_fragment(acc_frag_imag[j], 0.0f);

            for (int k = 0; k < K; k++)
            {
                wmma::mma_sync(acc_frag_imag[j], a_frag_imag[k], b_frag[k][j], acc_frag_imag[j]);
            }
        }

#pragma unroll
        for (int j = 0; j < 4; j++)
        {
            for (int k = 0; k < acc_frag_real[j].num_elements / 2; k++)
            {
                tmp_real = reinterpret_cast<float2 *>(acc_frag_real[j].x)[k];
                tmp_imag = reinterpret_cast<float2 *>(acc_frag_imag[j].x)[k];
                
                reinterpret_cast<float2 *>(acc_frag_real[j].x)[k] = tmp_real * __bfloat1622float2(reinterpret_cast<__nv_bfloat162 *>(tw_frag_real[j].x)[k]) - tmp_imag * __bfloat1622float2(reinterpret_cast<__nv_bfloat162 *>(tw_frag_imag[j].x)[k]);
                reinterpret_cast<float2 *>(acc_frag_imag[j].x)[k] = tmp_real * __bfloat1622float2(reinterpret_cast<__nv_bfloat162 *>(tw_frag_imag[j].x)[k]) + tmp_imag * __bfloat1622float2(reinterpret_cast<__nv_bfloat162 *>(tw_frag_real[j].x)[k]);
            }

            wmma::store_matrix_sync(out_real_shared + threadIdx.y * N * 16 + j * 16, acc_frag_real[j], N, wmma::mem_row_major);
            wmma::store_matrix_sync(out_imag_shared + threadIdx.y * N * 16 + j * 16, acc_frag_imag[j], N, wmma::mem_row_major);
        }

        __syncthreads();

#pragma unroll
        for (int i = threadIdx.y; i < N; i+=blockDim.y)
        {
            idx = i * 32 * gridDim.x + blockIdx.x * 32 + threadIdx.x;
            shared_offset = i * 32 + threadIdx.x;
            out_real[out_offset + out_t_offset + idx] = __float22bfloat162_rn(reinterpret_cast<float2*>(out_real_shared)[shared_offset]);
            out_imag[out_offset + out_t_offset + idx] = __float22bfloat162_rn(reinterpret_cast<float2*>(out_imag_shared)[shared_offset]);
        }

        __syncthreads();
    }
}

template <int K>
__global__ void butterfly_cuda_kernel_32(
    const __nv_bfloat162 *__restrict__ x,
    const __nv_bfloat162 *__restrict__ x_gate,
    const __nv_bfloat16 *__restrict__ d_f_real,
    const __nv_bfloat16 *__restrict__ d_f_imag,
    const __nv_bfloat162 *__restrict__ twiddle_factors_real,
    const __nv_bfloat162 *__restrict__ twiddle_factors_imag,
    __nv_bfloat162 *__restrict__ out_real,
    __nv_bfloat162 *__restrict__ out_imag,
    uint B,
    uint H,
    int M)
{
    const int N  = 32;
    const int max_idx = M / 2; //actually should be -1 since indices are 0-based but we are using < instead of <=
   
    const int offset  =  blockIdx.y * H * M / 2 + blockIdx.z * M / 2; 
    const int out_offset = blockIdx.y * H * 32 * 32 * gridDim.x + blockIdx.z * 32 * 32 * gridDim.x;
    

    __shared__ __nv_bfloat16 x_shared[K * 16 * 64];
    __shared__ __nv_bfloat16 d_f_real_shared[32 * 32];
    __shared__ __nv_bfloat16 d_f_imag_shared[32 * 32];
    __shared__ __nv_bfloat16 twiddles_real_shared[32 * 64];
    __shared__ __nv_bfloat16 twiddles_imag_shared[32 * 64];
    __shared__ float out_real_shared[32 * 64];
    __shared__ float out_imag_shared[32 * 64];

    // #pragma unroll
    for (int i = threadIdx.y; i<32; i+=blockDim.y)
    {
        int idx = i * 32 * gridDim.x + blockIdx.x * 32 + threadIdx.x;
        int shared_offset = i * 32 + threadIdx.x;

        if(i < K * 16){
            if(x_gate != nullptr){
                reinterpret_cast<__nv_bfloat162 *>(x_shared)[shared_offset] = idx < max_idx ? __hmul2(x[idx + offset], x_gate[idx + offset]) : __floats2bfloat162_rn(0.0f, 0.0f);
            }else{
                reinterpret_cast<__nv_bfloat162 *>(x_shared)[shared_offset] = idx < max_idx ? x[idx + offset] : __floats2bfloat162_rn(0.0f, 0.0f);
            }
        }
        reinterpret_cast<__nv_bfloat162 *>(twiddles_real_shared)[shared_offset] = twiddle_factors_real[idx];
        reinterpret_cast<__nv_bfloat162 *>(twiddles_imag_shared)[shared_offset] = twiddle_factors_imag[idx];

        // #pragma unroll
        d_f_real_shared[shared_offset] = d_f_real[shared_offset];
        d_f_imag_shared[shared_offset] = d_f_imag[shared_offset];
    }

    __syncthreads();

    if (threadIdx.y < N / 16)
    {
        float2 tmp_real, tmp_imag;

        wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::col_major> a_frag_real[2][2];
        wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> tw_frag_real[2][2];
        wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> tw_frag_imag[2][2];
        wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::col_major> a_frag_imag[2][2];
        wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::row_major> b_frag[K][2];
        wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc_frag_real[2][2];
        wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc_frag_imag[2][2];

        int t = threadIdx.y * 32;

        for (int i = 0; i < 2; i++)
        {
            for (int j = 0; j < 2; j++)
            {
                wmma::load_matrix_sync(a_frag_real[i][j], d_f_real_shared + j * N * 16 + i * 16, N);
                wmma::load_matrix_sync(a_frag_imag[i][j], d_f_imag_shared + j * N * 16 + i * 16, N);
                if(i < K){
                    wmma::load_matrix_sync(b_frag[i][j], x_shared + i * 2 * N * 16 + j * 16 + t, 2 * N);
                }
                wmma::load_matrix_sync(tw_frag_real[i][j], twiddles_real_shared + i * 2 * N * 16 + j * 16 + t, 2 * N);
                wmma::load_matrix_sync(tw_frag_imag[i][j], twiddles_imag_shared + i * 2 * N * 16 + j * 16 + t, 2 * N);
            }
        }

#pragma unroll
        for (int i = 0; i < 2; i++)
        {
            for (int j = 0; j < 2; j++)
            {
                wmma::fill_fragment(acc_frag_real[i][j], 0.0f);

                for (int k = 0; k < K; k++)
                {
                    wmma::mma_sync(acc_frag_real[i][j], a_frag_real[i][k], b_frag[k][j], acc_frag_real[i][j]);
                }
            }
        }

#pragma unroll
        for (int i = 0; i < 2; i++)
        {
            for (int j = 0; j < 2; j++)
            {
                wmma::fill_fragment(acc_frag_imag[i][j], 0.0f);

                for (int k = 0; k < K; k++)
                {
                    wmma::mma_sync(acc_frag_imag[i][j], a_frag_imag[i][k], b_frag[k][j], acc_frag_imag[i][j]);
                }
            }
        }

#pragma unroll
        for (int i = 0; i < 2; i++)
        {
            for (int j = 0; j < 2; j++)
            {
                 for (int k = 0; k < acc_frag_real[i][j].num_elements / 2; k++)
                {
                    tmp_real = 	reinterpret_cast<float2 *>(acc_frag_real[i][j].x)[k];
                    tmp_imag = 	reinterpret_cast<float2 *>(acc_frag_imag[i][j].x)[k];
                    reinterpret_cast<float2 *>(acc_frag_real[i][j].x)[k] = 	tmp_real  * __bfloat1622float2(reinterpret_cast<__nv_bfloat162 *>(tw_frag_real[i][j].x)[k]) - tmp_imag * __bfloat1622float2(reinterpret_cast<__nv_bfloat162 *>(tw_frag_imag[i][j].x)[k]);
                    reinterpret_cast<float2 *>(acc_frag_imag[i][j].x)[k] =  tmp_real  * __bfloat1622float2(reinterpret_cast<__nv_bfloat162 *>(tw_frag_imag[i][j].x)[k]) + tmp_imag * __bfloat1622float2(reinterpret_cast<__nv_bfloat162 *>(tw_frag_real[i][j].x)[k]);
                }
                wmma::store_matrix_sync(out_real_shared + i * 2 * N * 16 + j * 16 + t, acc_frag_real[i][j], 2 * N, wmma::mem_row_major);
                wmma::store_matrix_sync(out_imag_shared + i * 2 * N * 16 + j * 16 + t, acc_frag_imag[i][j], 2 * N, wmma::mem_row_major);
            }
        }
    }

    __syncthreads();

#pragma unroll
    for (int i = threadIdx.y; i<32; i+=blockDim.y)
    {
        int idx = i * 32 * gridDim.x + blockIdx.x * 32 + threadIdx.x;
        out_real[out_offset + idx] = __float22bfloat162_rn(reinterpret_cast<float2*>(out_real_shared)[i * 32 + threadIdx.x]);
        out_imag[out_offset + idx] = __float22bfloat162_rn(reinterpret_cast<float2*>(out_imag_shared)[i * 32 + threadIdx.x]);
    }
}

template <int K>
__global__ void butterfly_cuda_kernel_128(
    const __nv_bfloat162 *__restrict__ x,
    const __nv_bfloat162 *__restrict__ x_gate,
    const __nv_bfloat162 *__restrict__ d_f_real,
    const __nv_bfloat162 *__restrict__ d_f_imag,
    const __nv_bfloat162 *__restrict__ twiddle_factors_real,
    const __nv_bfloat162 *__restrict__ twiddle_factors_imag,
    __nv_bfloat162 *__restrict__ out_real,
    __nv_bfloat162 *__restrict__ out_imag,
    uint B,
    uint H,
    int M)
{
    const int max_idx = M / 2; //actually should be -1 since indices are 0-based but we are using < instead of <=
    const int offset = blockIdx.y * H * M/2 + blockIdx.z * 16 *  M/2;
    const int out_offset = blockIdx.y * H * 128 * 32 * 2 * gridDim.x + blockIdx.z * 16 * 128 * 32 * 2 * gridDim.x;
    const int N = 128;
    int idx;
    int t_offset;
    int out_t_offset;
    int shared_offset;

    extern __shared__ __nv_bfloat16 shared_real[];
    __nv_bfloat16 *shared_imag = &shared_real[128 * 128];


    wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::col_major> a_frag_real[8];
    wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> tw_frag_real[8];
    wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> tw_frag_imag[8];
    wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::col_major> a_frag_imag[8];
    wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::row_major> b_frag[K][8];
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc_frag_real[8];
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc_frag_imag[8];

    for (int i = threadIdx.y ; i < N; i+=blockDim.y)
    {
        for(int j=0; j< 2; j++){
            shared_offset = i * 64 + threadIdx.x + j * blockDim.x;
            reinterpret_cast<__nv_bfloat162 *>(shared_real)[shared_offset] = d_f_real[shared_offset];
            reinterpret_cast<__nv_bfloat162 *>(shared_imag)[shared_offset] = d_f_imag[shared_offset];
        }
    }

    __syncthreads();


    for (int i = 0; i < 8; i++){
        wmma::load_matrix_sync(a_frag_real[i], shared_real + i * 128 * 16 + threadIdx.y * 16, 128);
        wmma::load_matrix_sync(a_frag_imag[i], shared_imag + i * 128 * 16 + threadIdx.y * 16, 128);
    }


    __syncthreads();



    for (int i = threadIdx.y; i < N; i+=blockDim.y)
    {
        for(int j=0; j< 2; j++){
            idx = i * 32 * 2 * gridDim.x + j * blockDim.x + blockIdx.x * 64 + threadIdx.x;
            shared_offset = i * 64 + threadIdx.x + j * blockDim.x; 
            reinterpret_cast<__nv_bfloat162*>(shared_real)[shared_offset] = twiddle_factors_real[idx];
            reinterpret_cast<__nv_bfloat162*>(shared_imag)[shared_offset] = twiddle_factors_imag[idx];
        }
    }

    __syncthreads();


    for (int i = 0; i < 8; i++){
        wmma::load_matrix_sync(tw_frag_real[i], shared_real + threadIdx.y * 128 * 16 + i * 16, 128);
        wmma::load_matrix_sync(tw_frag_imag[i], shared_imag + threadIdx.y * 128 * 16 + i * 16, 128);
    }

    __syncthreads();


    for(int t=0; t< 16; t++){
        t_offset = t * M/2;
        out_t_offset = t * 128 * 32 * 2 * gridDim.x;

        for (int i = threadIdx.y; i < N; i+=blockDim.y)
        {
            if(i < K * 16){
                for(int j=0; j< 2; j++){
                    idx = i * 32 * 2 * gridDim.x + j * blockDim.x + blockIdx.x * 64 + threadIdx.x;
                    shared_offset = i * 64 + threadIdx.x + j * blockDim.x;
                    if(x_gate != nullptr){
                        reinterpret_cast<__nv_bfloat162*>(shared_real)[shared_offset] = idx < max_idx ?  __hmul2(x[idx + offset + t_offset], x_gate[idx + offset + t_offset]) : __floats2bfloat162_rn(0.0f, 0.0f);
                    }else{  
                        reinterpret_cast<__nv_bfloat162*>(shared_real)[shared_offset] = idx < max_idx ? x[idx + offset + t_offset] : __floats2bfloat162_rn(0.0f, 0.0f);
                    }
                }
            }
        }


        __syncthreads();


        for (int i = 0; i < K; i++)
        {
            for (int j = 0; j < 8; j++)
            {
                wmma::load_matrix_sync(b_frag[i][j], shared_real + i * 128 * 16 + j * 16, 128);
            }
        }

        __syncthreads();

        #pragma unroll
            for (int j = 0; j < 8; j++)
            {
                wmma::fill_fragment(acc_frag_real[j], 0.0f);

                for (int k = 0; k < K; k++)
                {
                    wmma::mma_sync(acc_frag_real[j], a_frag_real[k], b_frag[k][j], acc_frag_real[j]);
                }
            }

    #pragma unroll

            for (int j = 0; j < 8; j++)
            {
                wmma::fill_fragment(acc_frag_imag[j], 0.0f);

                for (int k = 0; k < K; k++)
                {
                    wmma::mma_sync(acc_frag_imag[j], a_frag_imag[k], b_frag[k][j], acc_frag_imag[j]);
                }
            }

            float2 tmp_real, tmp_imag;
    #pragma unroll
            for (int j = 0; j < 8; j++)
            {
                for (int k = 0; k < acc_frag_real[j].num_elements / 2; k++)
                {
                    tmp_real = reinterpret_cast<float2 *>(acc_frag_real[j].x)[k];
                    tmp_imag = reinterpret_cast<float2 *>(acc_frag_imag[j].x)[k];
                
                    reinterpret_cast<float2 *>(acc_frag_real[j].x)[k] = tmp_real * __bfloat1622float2(reinterpret_cast<__nv_bfloat162 *>(tw_frag_real[j].x)[k]) - tmp_imag * __bfloat1622float2(reinterpret_cast<__nv_bfloat162 *>(tw_frag_imag[j].x)[k]);
                    reinterpret_cast<float2 *>(acc_frag_imag[j].x)[k] = tmp_real * __bfloat1622float2(reinterpret_cast<__nv_bfloat162 *>(tw_frag_imag[j].x)[k]) + tmp_imag * __bfloat1622float2(reinterpret_cast<__nv_bfloat162 *>(tw_frag_real[j].x)[k]);
                }
            }

            for (int j = 0; j < 8; j++)
            {
                wmma::store_matrix_sync(reinterpret_cast<float*>(shared_real) + threadIdx.y * 128 * 16 + j * 16, acc_frag_real[j], 128, wmma::mem_row_major);
            }

            __syncthreads();

    #pragma unroll
            for (int i = threadIdx.y; i < N; i+=blockDim.y)
            {
                for(int j=0; j< 2; j++){
                    idx = i * 32 * 2 * gridDim.x + j * blockDim.x + blockIdx.x * 64 + threadIdx.x;
                    shared_offset = i * 64 + threadIdx.x + j * blockDim.x; 
                    out_real[idx + out_offset + out_t_offset] = __float22bfloat162_rn(reinterpret_cast<float2*>(shared_real)[shared_offset]);
                }
            }

            __syncthreads();


            for (int j = 0; j < 8; j++)
            {
                wmma::store_matrix_sync(reinterpret_cast<float*>(shared_real) + threadIdx.y * 128 * 16 + j * 16, acc_frag_imag[j], 128, wmma::mem_row_major);
            }

            __syncthreads();

    #pragma unroll
            for (int i = threadIdx.y; i < N; i+=blockDim.y)
            {
                for(int j=0; j< 2; j++){
                    idx = i * 32 * 2 * gridDim.x + j * blockDim.x + blockIdx.x * 64 + threadIdx.x;
                    shared_offset = i * 64 + threadIdx.x + j * blockDim.x; 
                    out_imag[idx + out_offset + out_t_offset] = __float22bfloat162_rn(reinterpret_cast<float2*>(shared_real)[shared_offset]);
                }
            }
    }
}

template<int K>
__global__ void butterfly_cuda_kernel_16(
    const __nv_bfloat162 *__restrict__ x,
    const __nv_bfloat162 *__restrict__ x_gate,
    const __nv_bfloat16 *__restrict__ d_f_real,
    const __nv_bfloat16 *__restrict__ d_f_imag,
    const __nv_bfloat162 *__restrict__ twiddle_factors_real,
    const __nv_bfloat162 *__restrict__ twiddle_factors_imag,
    __nv_bfloat162 *__restrict__ out_real,
    __nv_bfloat162 *__restrict__ out_imag,
    uint B,
    uint H,
    int M)
{
    const int max_idx = M / 2; //actually should be -1 since indices are 0-based but we are using < instead of <=
    const int N  = 16;
    const int offset  =  blockIdx.y * H * M / 2 + blockIdx.z * M / 2; 
    const int out_offset = blockIdx.y * H * N * blockDim.x * gridDim.x + blockIdx.z * N * blockDim.x * gridDim.x;

    

    __shared__ __nv_bfloat16 x_shared[N * 64];
    __shared__ __nv_bfloat16 d_f_real_shared[N * N];
    __shared__ __nv_bfloat16 d_f_imag_shared[N * N];
    __shared__ __nv_bfloat16 twiddles_real_shared[N * 64];
    __shared__ __nv_bfloat16 twiddles_imag_shared[N * 64];
    __shared__ float out_real_shared[N * 64];
    __shared__ float out_imag_shared[N * 64];

    // #pragma unroll
    for (int i = threadIdx.y; i < N; i++)
    {
        int idx = i * blockDim.x * gridDim.x + blockIdx.x * blockDim.x + threadIdx.x;
        int shared_offset = i * blockDim.x + threadIdx.x;

        if(x_gate != nullptr){
            reinterpret_cast<__nv_bfloat162 *>(x_shared)[shared_offset] = idx < max_idx ? __hmul2(x[idx + offset], x_gate[idx + offset]) : __floats2bfloat162_rn(0.0f, 0.0f);
        }else{
            reinterpret_cast<__nv_bfloat162 *>(x_shared)[shared_offset] = idx < max_idx ? x[idx + offset] : __floats2bfloat162_rn(0.0f, 0.0f);
        }
        reinterpret_cast<__nv_bfloat162 *>(twiddles_real_shared)[shared_offset] = twiddle_factors_real[idx];
        reinterpret_cast<__nv_bfloat162 *>(twiddles_imag_shared)[shared_offset] = twiddle_factors_imag[idx];

        // #pragma unroll
        if(threadIdx.x  < 16 ){
            shared_offset = i * 16 + threadIdx.x;
            d_f_real_shared[shared_offset] = d_f_real[shared_offset];
            d_f_imag_shared[shared_offset] = d_f_imag[shared_offset];
        }
    }

    __syncthreads();

    if (threadIdx.y < 4)
    {
        float2 tmp_real, tmp_imag;

        wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::col_major> a_frag_real;
        wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> tw_frag_real;
        wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> tw_frag_imag;
        wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::col_major> a_frag_imag;
        wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::row_major> b_frag;
        wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc_frag_real;
        wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc_frag_imag;


        wmma::load_matrix_sync(a_frag_real, d_f_real_shared, N);
        wmma::load_matrix_sync(a_frag_imag, d_f_imag_shared, N);
        wmma::load_matrix_sync(b_frag, x_shared + threadIdx.y * 16, 64);
        wmma::load_matrix_sync(tw_frag_real, twiddles_real_shared + threadIdx.y * 16, 64);
        wmma::load_matrix_sync(tw_frag_imag, twiddles_imag_shared + threadIdx.y * 16, 64);
 


        wmma::fill_fragment(acc_frag_real, 0.0f);


        wmma::mma_sync(acc_frag_real, a_frag_real, b_frag, acc_frag_real);



        wmma::fill_fragment(acc_frag_imag, 0.0f);


         wmma::mma_sync(acc_frag_imag, a_frag_imag, b_frag, acc_frag_imag);
    

#pragma unroll
        for (int k = 0; k < acc_frag_real.num_elements / 2; k++)
        {
            tmp_real = 	reinterpret_cast<float2 *>(acc_frag_real.x)[k];
            tmp_imag = 	reinterpret_cast<float2 *>(acc_frag_imag.x)[k];
            reinterpret_cast<float2 *>(acc_frag_real.x)[k] = 	tmp_real  * __bfloat1622float2(reinterpret_cast<__nv_bfloat162 *>(tw_frag_real.x)[k]) - tmp_imag * __bfloat1622float2(reinterpret_cast<__nv_bfloat162 *>(tw_frag_imag.x)[k]);
            reinterpret_cast<float2 *>(acc_frag_imag.x)[k] =  tmp_real  * __bfloat1622float2(reinterpret_cast<__nv_bfloat162 *>(tw_frag_imag.x)[k]) + tmp_imag * __bfloat1622float2(reinterpret_cast<__nv_bfloat162 *>(tw_frag_real.x)[k]);
        }
        wmma::store_matrix_sync(out_real_shared + threadIdx.y * 16, acc_frag_real, 64, wmma::mem_row_major);
        wmma::store_matrix_sync(out_imag_shared + threadIdx.y * 16, acc_frag_imag, 64, wmma::mem_row_major);

    }
    __syncthreads();

#pragma unroll
    for (int i = threadIdx.y; i < N; i++)
    {
        int idx = i * blockDim.x * gridDim.x + blockIdx.x * blockDim.x + threadIdx.x;;
        out_real[out_offset + idx] = __float22bfloat162_rn(reinterpret_cast<float2*>(out_real_shared)[i * 32 + threadIdx.x]);
        out_imag[out_offset + idx] = __float22bfloat162_rn(reinterpret_cast<float2*>(out_imag_shared)[i * 32 + threadIdx.x]);
    }
}

std::vector<torch::Tensor> butterfly_padded_bf16_cuda(
    torch::Tensor x,
    torch::Tensor d_f_real,
    torch::Tensor d_f_imag,
    torch::Tensor twiddle_factors_real,
    torch::Tensor twiddle_factors_imag,
    int M,
    std::optional<at::Tensor> x_gate = std::nullopt
    )
{

    uint B = x.size(0);
    uint H = x.size(1);

    uint d_f_size = d_f_real.size(1);

    uint N = x.size(2);

    //need to make sure that N is less that the M to which we are padding 
    assert(N <= d_f_size * M);

    dim3 gridDim;
    dim3 blockDim;

    gridDim.y = B;
    gridDim.z = H;

    blockDim.x = 32;
    blockDim.y = 4;

    torch::Tensor out_real = torch::empty({B, H, d_f_size * M}, x.options());
    torch::Tensor out_imag = torch::empty({B, H, d_f_size * M}, x.options());

    gridDim.x = 512 / (32 * 1024/ M);

    const int K = ceil(N / (1.0 * 16 * M));

    switch (d_f_size)
    {
        case 16:
            butterfly_cuda_kernel_16<1><<<gridDim, blockDim>>>(
                    static_cast<__nv_bfloat162 *>(x.data_ptr()),
                    x_gate ? static_cast<__nv_bfloat162 *>(x_gate.value().data_ptr()) : nullptr,
                    static_cast<__nv_bfloat16 *>(d_f_real.data_ptr()),
                    static_cast<__nv_bfloat16 *>(d_f_imag.data_ptr()),
                    static_cast<__nv_bfloat162 *>(twiddle_factors_real.data_ptr()),
                    static_cast<__nv_bfloat162 *>(twiddle_factors_imag.data_ptr()),
                    static_cast<__nv_bfloat162 *>(out_real.data_ptr()),
                    static_cast<__nv_bfloat162 *>(out_imag.data_ptr()),
                    B,
                    H,
                    N);
                    break;
        case 32:
            switch(K){
                case 1:
                    butterfly_cuda_kernel_32<1><<<gridDim, blockDim>>>(
                    static_cast<__nv_bfloat162 *>(x.data_ptr()),
                    x_gate ? static_cast<__nv_bfloat162 *>(x_gate.value().data_ptr()) : nullptr,
                    static_cast<__nv_bfloat16 *>(d_f_real.data_ptr()),
                    static_cast<__nv_bfloat16 *>(d_f_imag.data_ptr()),
                    static_cast<__nv_bfloat162 *>(twiddle_factors_real.data_ptr()),
                    static_cast<__nv_bfloat162 *>(twiddle_factors_imag.data_ptr()),
                    static_cast<__nv_bfloat162 *>(out_real.data_ptr()),
                    static_cast<__nv_bfloat162 *>(out_imag.data_ptr()),
                    B,
                    H,
                    N);
                    break;
                case 2:
                    butterfly_cuda_kernel_32<2><<<gridDim, blockDim>>>(
                        static_cast<__nv_bfloat162 *>(x.data_ptr()),
                        x_gate ? static_cast<__nv_bfloat162 *>(x_gate.value().data_ptr()) : nullptr,
                        static_cast<__nv_bfloat16 *>(d_f_real.data_ptr()),
                        static_cast<__nv_bfloat16 *>(d_f_imag.data_ptr()),
                        static_cast<__nv_bfloat162 *>(twiddle_factors_real.data_ptr()),
                        static_cast<__nv_bfloat162 *>(twiddle_factors_imag.data_ptr()),
                        static_cast<__nv_bfloat162 *>(out_real.data_ptr()),
                        static_cast<__nv_bfloat162 *>(out_imag.data_ptr()),
                        B,
                        H,
                        N);
                    break;
                default:
                    printf("Invalid K, df size 32: %d\n", K);
            }
            break;
        case 64:
            gridDim.z = H / 16;

            switch(K){
                case 1:
                    cudaFuncSetAttribute(&butterfly_cuda_kernel_64<1>, cudaFuncAttributeMaxDynamicSharedMemorySize, 78000);
                    butterfly_cuda_kernel_64<1><<<gridDim, blockDim, 78000>>>(
                    static_cast<__nv_bfloat162 *>(x.data_ptr()),
                    x_gate ? static_cast<__nv_bfloat162 *>(x_gate.value().data_ptr()) : nullptr,
                    static_cast<__nv_bfloat162 *>(d_f_real.data_ptr()),
                    static_cast<__nv_bfloat162 *>(d_f_imag.data_ptr()),
                    static_cast<__nv_bfloat162 *>(twiddle_factors_real.data_ptr()),
                    static_cast<__nv_bfloat162 *>(twiddle_factors_imag.data_ptr()),
                    static_cast<__nv_bfloat162 *>(out_real.data_ptr()),
                    static_cast<__nv_bfloat162 *>(out_imag.data_ptr()),
                    B,
                    H,
                    N);
                    break;
                case 2:
                    cudaFuncSetAttribute(&butterfly_cuda_kernel_64<2>, cudaFuncAttributeMaxDynamicSharedMemorySize, 78000);
                    butterfly_cuda_kernel_64<2><<<gridDim, blockDim, 78000>>>(
                    static_cast<__nv_bfloat162 *>(x.data_ptr()),
                    x_gate ? static_cast<__nv_bfloat162 *>(x_gate.value().data_ptr()) : nullptr,
                    static_cast<__nv_bfloat162 *>(d_f_real.data_ptr()),
                    static_cast<__nv_bfloat162 *>(d_f_imag.data_ptr()),
                    static_cast<__nv_bfloat162 *>(twiddle_factors_real.data_ptr()),
                    static_cast<__nv_bfloat162 *>(twiddle_factors_imag.data_ptr()),
                    static_cast<__nv_bfloat162 *>(out_real.data_ptr()),
                    static_cast<__nv_bfloat162 *>(out_imag.data_ptr()),
                    B,
                    H,
                    N);
                    break;
                case 3:
                    cudaFuncSetAttribute(&butterfly_cuda_kernel_64<3>, cudaFuncAttributeMaxDynamicSharedMemorySize, 78000);
                    butterfly_cuda_kernel_64<3><<<gridDim, blockDim, 78000>>>(
                    static_cast<__nv_bfloat162 *>(x.data_ptr()),
                    x_gate ? static_cast<__nv_bfloat162 *>(x_gate.value().data_ptr()) : nullptr,
                    static_cast<__nv_bfloat162 *>(d_f_real.data_ptr()),
                    static_cast<__nv_bfloat162 *>(d_f_imag.data_ptr()),
                    static_cast<__nv_bfloat162 *>(twiddle_factors_real.data_ptr()),
                    static_cast<__nv_bfloat162 *>(twiddle_factors_imag.data_ptr()),
                    static_cast<__nv_bfloat162 *>(out_real.data_ptr()),
                    static_cast<__nv_bfloat162 *>(out_imag.data_ptr()),
                    B,
                    H,
                    N);
                    break;
                case 4:
                    cudaFuncSetAttribute(&butterfly_cuda_kernel_64<4>, cudaFuncAttributeMaxDynamicSharedMemorySize, 78000);
                    butterfly_cuda_kernel_64<4><<<gridDim, blockDim, 78000>>>(
                    static_cast<__nv_bfloat162 *>(x.data_ptr()),
                    x_gate ? static_cast<__nv_bfloat162 *>(x_gate.value().data_ptr()) : nullptr,
                    static_cast<__nv_bfloat162 *>(d_f_real.data_ptr()),
                    static_cast<__nv_bfloat162 *>(d_f_imag.data_ptr()),
                    static_cast<__nv_bfloat162 *>(twiddle_factors_real.data_ptr()),
                    static_cast<__nv_bfloat162 *>(twiddle_factors_imag.data_ptr()),
                    static_cast<__nv_bfloat162 *>(out_real.data_ptr()),
                    static_cast<__nv_bfloat162 *>(out_imag.data_ptr()),
                    B,
                    H,
                    N);
                    break;
                default:
                    printf("Invalid K, df size 64: %d\n", K);
            }   
            break;
        case 128:
            blockDim.x = 32;
            blockDim.y = 8;
            gridDim.x = 256 / (32 * 1024/ M);
            gridDim.z = H / 16;
            switch(K){
                case 1:
                    cudaFuncSetAttribute(&butterfly_cuda_kernel_128<1>, cudaFuncAttributeMaxDynamicSharedMemorySize, 65536);
                    butterfly_cuda_kernel_128<1><<<gridDim, blockDim, 65536>>>(
                        static_cast<__nv_bfloat162 *>(x.data_ptr()),
                        x_gate ? static_cast<__nv_bfloat162 *>(x_gate.value().data_ptr()) : nullptr,
                        static_cast<__nv_bfloat162 *>(d_f_real.data_ptr()),
                        static_cast<__nv_bfloat162 *>(d_f_imag.data_ptr()),
                        static_cast<__nv_bfloat162 *>(twiddle_factors_real.data_ptr()),
                        static_cast<__nv_bfloat162 *>(twiddle_factors_imag.data_ptr()),
                        static_cast<__nv_bfloat162 *>(out_real.data_ptr()),
                        static_cast<__nv_bfloat162 *>(out_imag.data_ptr()),
                        B,
                        H,
                        N);
                        break;
                case 2:
                    cudaFuncSetAttribute(&butterfly_cuda_kernel_128<2>, cudaFuncAttributeMaxDynamicSharedMemorySize, 65536);
                    butterfly_cuda_kernel_128<2><<<gridDim, blockDim, 65536>>>(
                        static_cast<__nv_bfloat162 *>(x.data_ptr()),
                        x_gate ? static_cast<__nv_bfloat162 *>(x_gate.value().data_ptr()) : nullptr,
                        static_cast<__nv_bfloat162 *>(d_f_real.data_ptr()),
                        static_cast<__nv_bfloat162 *>(d_f_imag.data_ptr()),
                        static_cast<__nv_bfloat162 *>(twiddle_factors_real.data_ptr()),
                        static_cast<__nv_bfloat162 *>(twiddle_factors_imag.data_ptr()),
                        static_cast<__nv_bfloat162 *>(out_real.data_ptr()),
                        static_cast<__nv_bfloat162 *>(out_imag.data_ptr()),
                        B,
                        H,
                        N);
                    break;
                case 3:
                    cudaFuncSetAttribute(&butterfly_cuda_kernel_128<3>, cudaFuncAttributeMaxDynamicSharedMemorySize, 65536);

                    butterfly_cuda_kernel_128<3><<<gridDim, blockDim, 65536>>>(
                        static_cast<__nv_bfloat162 *>(x.data_ptr()),
                        x_gate ? static_cast<__nv_bfloat162 *>(x_gate.value().data_ptr()) : nullptr,
                        static_cast<__nv_bfloat162 *>(d_f_real.data_ptr()),
                        static_cast<__nv_bfloat162 *>(d_f_imag.data_ptr()),
                        static_cast<__nv_bfloat162 *>(twiddle_factors_real.data_ptr()),
                        static_cast<__nv_bfloat162 *>(twiddle_factors_imag.data_ptr()),
                        static_cast<__nv_bfloat162 *>(out_real.data_ptr()),
                        static_cast<__nv_bfloat162 *>(out_imag.data_ptr()),
                        B,
                        H,
                        N);
                    break;
                case 4:
                    cudaFuncSetAttribute(&butterfly_cuda_kernel_128<4>, cudaFuncAttributeMaxDynamicSharedMemorySize, 65536);

                    butterfly_cuda_kernel_128<4><<<gridDim, blockDim, 65536>>>(
                        static_cast<__nv_bfloat162 *>(x.data_ptr()),
                        x_gate ? static_cast<__nv_bfloat162 *>(x_gate.value().data_ptr()) : nullptr,
                        static_cast<__nv_bfloat162 *>(d_f_real.data_ptr()),
                        static_cast<__nv_bfloat162 *>(d_f_imag.data_ptr()),
                        static_cast<__nv_bfloat162 *>(twiddle_factors_real.data_ptr()),
                        static_cast<__nv_bfloat162 *>(twiddle_factors_imag.data_ptr()),
                        static_cast<__nv_bfloat162 *>(out_real.data_ptr()),
                        static_cast<__nv_bfloat162 *>(out_imag.data_ptr()),
                        B,
                        H,
                        N);
                    break;
                case 5:
                    cudaFuncSetAttribute(&butterfly_cuda_kernel_128<5>, cudaFuncAttributeMaxDynamicSharedMemorySize, 65536);

                    butterfly_cuda_kernel_128<5><<<gridDim, blockDim, 65536>>>(
                        static_cast<__nv_bfloat162 *>(x.data_ptr()),
                        x_gate ? static_cast<__nv_bfloat162 *>(x_gate.value().data_ptr()) : nullptr,
                        static_cast<__nv_bfloat162 *>(d_f_real.data_ptr()),
                        static_cast<__nv_bfloat162 *>(d_f_imag.data_ptr()),
                        static_cast<__nv_bfloat162 *>(twiddle_factors_real.data_ptr()),
                        static_cast<__nv_bfloat162 *>(twiddle_factors_imag.data_ptr()),
                        static_cast<__nv_bfloat162 *>(out_real.data_ptr()),
                        static_cast<__nv_bfloat162 *>(out_imag.data_ptr()),
                        B,
                        H,
                        N);
                    break;
                case 6:
                    cudaFuncSetAttribute(&butterfly_cuda_kernel_128<6>, cudaFuncAttributeMaxDynamicSharedMemorySize, 65536);

                    butterfly_cuda_kernel_128<6><<<gridDim, blockDim, 65536>>>(
                        static_cast<__nv_bfloat162 *>(x.data_ptr()),
                        x_gate ? static_cast<__nv_bfloat162 *>(x_gate.value().data_ptr()) : nullptr,
                        static_cast<__nv_bfloat162 *>(d_f_real.data_ptr()),
                        static_cast<__nv_bfloat162 *>(d_f_imag.data_ptr()),
                        static_cast<__nv_bfloat162 *>(twiddle_factors_real.data_ptr()),
                        static_cast<__nv_bfloat162 *>(twiddle_factors_imag.data_ptr()),
                        static_cast<__nv_bfloat162 *>(out_real.data_ptr()),
                        static_cast<__nv_bfloat162 *>(out_imag.data_ptr()),
                        B,
                        H,
                        N);
                    break;
                case 7:
                    cudaFuncSetAttribute(&butterfly_cuda_kernel_128<7>, cudaFuncAttributeMaxDynamicSharedMemorySize, 65536);

                    butterfly_cuda_kernel_128<7><<<gridDim, blockDim, 65536>>>(
                        static_cast<__nv_bfloat162 *>(x.data_ptr()),
                        x_gate ? static_cast<__nv_bfloat162 *>(x_gate.value().data_ptr()) : nullptr,
                        static_cast<__nv_bfloat162 *>(d_f_real.data_ptr()),
                        static_cast<__nv_bfloat162 *>(d_f_imag.data_ptr()),
                        static_cast<__nv_bfloat162 *>(twiddle_factors_real.data_ptr()),
                        static_cast<__nv_bfloat162 *>(twiddle_factors_imag.data_ptr()),
                        static_cast<__nv_bfloat162 *>(out_real.data_ptr()),
                        static_cast<__nv_bfloat162 *>(out_imag.data_ptr()),
                        B,
                        H,
                        N);
                    break;
                case 8:
                    cudaFuncSetAttribute(&butterfly_cuda_kernel_128<8>, cudaFuncAttributeMaxDynamicSharedMemorySize, 65536);

                    butterfly_cuda_kernel_128<8><<<gridDim, blockDim, 65536>>>(
                        static_cast<__nv_bfloat162 *>(x.data_ptr()),
                        x_gate ? static_cast<__nv_bfloat162 *>(x_gate.value().data_ptr()) : nullptr,
                        static_cast<__nv_bfloat162 *>(d_f_real.data_ptr()),
                        static_cast<__nv_bfloat162 *>(d_f_imag.data_ptr()),
                        static_cast<__nv_bfloat162 *>(twiddle_factors_real.data_ptr()),
                        static_cast<__nv_bfloat162 *>(twiddle_factors_imag.data_ptr()),
                        static_cast<__nv_bfloat162 *>(out_real.data_ptr()),
                        static_cast<__nv_bfloat162 *>(out_imag.data_ptr()),
                        B,
                        H,
                        N);
                    break;
                default:
                    printf("Invalid K, df size 128: %d\n", K);

            }
            break;

        default:
        printf("Not yet implemented \n");
            break;
    }

    return {out_real, out_imag};
}