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
using complex_bhalf_t = typename c10::complex<at::BFloat16>;

#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16
#define WARP_SIZE 32

#ifndef MONARCH_CUDA_H_
#define MONARCH_CUDA_H_

__device__ __forceinline__ float2 

operator+( float2 lhs, float2 rhs) 

{

    float2 res = { lhs.x + rhs.x , lhs.y + rhs.y };

    return res;

}


__device__ __forceinline__ float2 

operator-( float2 lhs, float2 rhs) 

{

    float2 res = { lhs.x - rhs.x , lhs.y - rhs.y };

    return res;

}

__device__ __forceinline__ float2 

operator*( float2 lhs, float2 rhs) 

{

    float2 res = { lhs.x * rhs.x , lhs.y * rhs.y };

    return res;

}
#endif