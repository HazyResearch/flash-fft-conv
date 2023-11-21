
// Copyright (c) 2023 Dan Fu, Hermann Kumbong

#include <torch/extension.h>
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <vector>

#define DISPATCH_FLOAT_AND_HALF_AND_BF16(INPUT_TYPE, WEIGHT_TYPE, NAME, ...)                     \
  if ((INPUT_TYPE == at::ScalarType::Half) && (WEIGHT_TYPE == at::ScalarType::Half)) {           \
    using input_t = __half;                                                            \
    using weight_t = __half;                                                           \
    __VA_ARGS__();                                                                       \
  } else if((INPUT_TYPE == at::ScalarType::Half) && (WEIGHT_TYPE == at::ScalarType::BFloat16)){    \
    using input_t = __half;                                                            \
    using weight_t = __nv_bfloat16;                                                       \
    __VA_ARGS__();                                                                       \
  } else if((INPUT_TYPE == at::ScalarType::Half) && (WEIGHT_TYPE == at::ScalarType::Float)){   \
    using input_t = __half;                                                            \
    using weight_t = float;                                                       \
    __VA_ARGS__();                                                                       \
  } else if ((INPUT_TYPE == at::ScalarType::BFloat16) && (WEIGHT_TYPE == at::ScalarType::BFloat16)) {    \
    using input_t = __nv_bfloat16;                                                        \
    using weight_t = __nv_bfloat16;                                                       \
    __VA_ARGS__();                                                                       \
  } else if ((INPUT_TYPE == at::ScalarType::BFloat16) && (WEIGHT_TYPE == at::ScalarType::Half)) {      \
    using input_t = __nv_bfloat16;                                                        \
    using weight_t = __half;                                                       \
    __VA_ARGS__();                                                                       \
  } else if ((INPUT_TYPE == at::ScalarType::BFloat16) && (WEIGHT_TYPE == at::ScalarType::Float)) {    \
    using input_t = __nv_bfloat16;                                                        \
    using weight_t = float;                                                       \
    __VA_ARGS__();                                                                       \
  } else if ((INPUT_TYPE == at::ScalarType::Float) && (WEIGHT_TYPE == at::ScalarType::Float))  { \
    using input_t = float;                                                               \
    using weight_t = float;                                                              \
    __VA_ARGS__();                                                                       \
  } else if ((INPUT_TYPE == at::ScalarType::Float) && (WEIGHT_TYPE == at::ScalarType::Half))  {  \
    using input_t = float;                                                               \
    using weight_t = __half;                                                           \
    __VA_ARGS__();                                                                       \
  } else if ((INPUT_TYPE == at::ScalarType::Float) && (WEIGHT_TYPE == at::ScalarType::BFloat16))  {  \
    using input_t = float;                                                               \
    using weight_t = __nv_bfloat16;                                                           \
    __VA_ARGS__();                                                                       \
  } else {                                                                               \
    AT_ERROR(#NAME, " not implemented for input-type '", toString(INPUT_TYPE), "' and weight-type '", toString(WEIGHT_TYPE), "'"); \
  }


#define DISPATCH_FLOAT2_AND_HALF2_AND_BF162(INPUT_TYPE, WEIGHT_TYPE, NAME, ...)                     \
  if ((INPUT_TYPE == at::ScalarType::Half) && (WEIGHT_TYPE == at::ScalarType::Half)) {           \
    using input_t = __half2;                                                            \
    using weight_t = __half2;                                                           \
    __VA_ARGS__();                                                                       \
  } else if((INPUT_TYPE == at::ScalarType::Half) && (WEIGHT_TYPE == at::ScalarType::BFloat16)){    \
    using input_t = __half2;                                                            \
    using weight_t = __nv_bfloat162;                                                       \
    __VA_ARGS__();                                                                       \
  } else if((INPUT_TYPE == at::ScalarType::Half) && (WEIGHT_TYPE == at::ScalarType::Float)){   \
    using input_t = __half2;                                                            \
    using weight_t = float2;                                                       \
    __VA_ARGS__();                                                                       \
  } else if ((INPUT_TYPE == at::ScalarType::BFloat16) && (WEIGHT_TYPE == at::ScalarType::BFloat16)) {    \
    using input_t = __nv_bfloat162;                                                        \
    using weight_t = __nv_bfloat162;                                                       \
    __VA_ARGS__();                                                                       \
  } else if ((INPUT_TYPE == at::ScalarType::BFloat16) && (WEIGHT_TYPE == at::ScalarType::Half)) {      \
    using input_t = __nv_bfloat162;                                                        \
    using weight_t = __half2;                                                       \
    __VA_ARGS__();                                                                       \
  } else if ((INPUT_TYPE == at::ScalarType::BFloat16) && (WEIGHT_TYPE == at::ScalarType::Float)) {    \
    using input_t = __nv_bfloat162;                                                        \
    using weight_t = float2;                                                       \
    __VA_ARGS__();                                                                       \
  } else if ((INPUT_TYPE == at::ScalarType::Float) && (WEIGHT_TYPE == at::ScalarType::Float))  { \
    using input_t = float2;                                                               \
    using weight_t = float2;                                                              \
    __VA_ARGS__();                                                                       \
  } else if ((INPUT_TYPE == at::ScalarType::Float) && (WEIGHT_TYPE == at::ScalarType::Half))  {  \
    using input_t = float2;                                                               \
    using weight_t = __half2;                                                           \
    __VA_ARGS__();                                                                       \
  } else if ((INPUT_TYPE == at::ScalarType::Float) && (WEIGHT_TYPE == at::ScalarType::BFloat16))  {  \
    using input_t = float2;                                                               \
    using weight_t = __nv_bfloat162;                                                           \
    __VA_ARGS__();                                                                       \
  } else {                                                                               \
    AT_ERROR(#NAME, " not implemented for input-type '", toString(INPUT_TYPE), "' and weight-type '", toString(WEIGHT_TYPE), "'"); \
  }

__forceinline__ __device__ float __hfma(const float a, const float b, const float c)
{
    return a * b + c;
}

__forceinline__ __device__ float2 __hfma2(const float2 a, const float2 b, const float2 c)
{
    return make_float2(a.x * b.x + c.x, a.y * b.y + c.y);
}

template<typename T>
__forceinline__ __device__ void set_value(T* dst, T src)
{
    *dst = src;
}

__forceinline__ __device__ void set_value(__half2* dst, float2 src)
{
    *dst = __float22half2_rn(src);
}

__forceinline__ __device__ void set_value(__nv_bfloat162* dst, float2 src)
{
    *dst = __float22bfloat162_rn(src);
}

__forceinline__ __device__ void set_value(float2* dst, __half2 src)
{
    *dst = __half22float2(src);
}

__forceinline__ __device__ void set_value(float2* dst, __nv_bfloat162 src)
{
    *dst = __bfloat1622float2(src);
}

__forceinline__ __device__ void set_value(__half2* dst, __nv_bfloat162 src)
{
    *dst = __float22half2_rn(__bfloat1622float2(src));
}

__forceinline__ __device__ void set_value(__nv_bfloat162* dst, __half2 src)
{
    *dst = __float22bfloat162_rn(__half22float2(src));
}

__forceinline__ __device__ void set_value(__half* dst, float src)
{
    *dst = __float2half(src);
}

__forceinline__ __device__ void set_value(__nv_bfloat16* dst, float src)
{
    *dst = __float2bfloat16(src);
}

__forceinline__ __device__ void set_value(float* dst, __half src)
{
    *dst = __half2float(src);
}

__forceinline__ __device__ void set_value(float* dst, __nv_bfloat16 src)
{
    *dst = __bfloat162float(src);
}

__forceinline__ __device__ void set_value(__half* dst, __nv_bfloat16 src)
{
    *dst = __float2half(__bfloat162float(src));
}

__forceinline__ __device__ void set_value(__nv_bfloat16* dst, __half src)
{
    *dst = __float2bfloat16(__half2float(src));
}
