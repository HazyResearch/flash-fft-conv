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

#ifndef MONARCH_CUDA_FP16_COMPLEX_MUL_
#define MONARCH_CUDA_FP16_COMPLEX_MUL_

__device__ __forceinline__ void complex_mul(at::Half a_real, at::Half a_imag, at::Half b_real, at::Half b_imag, at::Half *c_real, at::Half *c_imag) {
  __half temp_x, temp_y;
  // temp_x = __hsub(__hmul(a_real, b_real), __hmul(a_imag, b_imag));
  // temp_y = __hadd(__hmul(a_imag, b_real), __hmul(a_real, b_imag));
  temp_x = __half(a_real * b_real - a_imag * b_imag);
  temp_y = __hfma(__half(a_imag), __half(b_real), __half(a_real * b_imag));
  *c_real = temp_x;
  *c_imag = temp_y;
}

__device__ __forceinline__ void complex_mul(complex_half_t a, complex_half_t b, complex_half_t *c) {
  __half temp_x, temp_y;
  __half2 temp2;
  // temp_x = __hsub(__hmul(a_real, b_real), __hmul(a_imag, b_imag));
  // temp_y = __hadd(__hmul(a_imag, b_real), __hmul(a_real, b_imag));
  // temp_x = __half(a.real() * b.real() - a.imag() * b.imag());
  temp2 = __hmul2(__half2(a.real(), a.imag()), __half2(b.real(), b.imag()));
  temp_x = __hsub(temp2.x, temp2.y);
  temp_y = __hfma(__half(a.imag()), __half(b.real()), __half(a.real() * b.imag()));
  *c = complex_half_t(temp_x, temp_y);
}

__device__ __forceinline__ void complex_mul_float_half(float a_real, float a_imag, at::Half b_real, at::Half b_imag, at::Half *c_real, at::Half *c_imag) {
  __half temp_x, temp_y;
  // temp_x = __hsub(__hmul(a_real, b_real), __hmul(a_imag, b_imag));
  // temp_y = __hadd(__hmul(a_imag, b_real), __hmul(a_real, b_imag));
  temp_x = __half(at::Half(a_real) * b_real - at::Half(a_imag) * b_imag);
  temp_y = __hfma(__half(at::Half(a_imag)), __half(b_real), __half(at::Half(a_real) * b_imag));
  *c_real = temp_x;
  *c_imag = temp_y;
}

__device__ __forceinline__ void complex_mul_half2(__half2 a_real, __half2 a_imag, __half2 b_real, __half2 b_imag, __half2 *c_real, __half2 *c_imag) {
  __half2 temp_x, temp_y;

  temp_x = __hsub2(__hmul2(a_real, b_real), __hmul2(a_imag, b_imag));
  temp_y = __hfma2(a_imag, b_real, __hmul2(a_real, b_imag));
  *c_real = temp_x;
  *c_imag = temp_y;
}

__device__ __forceinline__ void complex_mul_half2(__half2 a_real, __half2 a_imag, __half2 b_real, __half2 b_imag, complex_half_t *c1, complex_half_t *c2) {
  __half2 temp_x, temp_y;

  temp_x = __hsub2(__hmul2(a_real, b_real), __hmul2(a_imag, b_imag));
  temp_y = __hfma2(a_imag, b_real, __hmul2(a_real, b_imag));
  *c1 = complex_half_t(temp_x.x, temp_y.x);
  *c2 = complex_half_t(temp_x.y, temp_y.y);
}

__device__ __forceinline__ void complex_mul_half2(__half2 a_real, __half2 a_imag, __half2 b_real, __half2 b_imag, __half *c_real_0, __half *c_imag_0, __half *c_real_1, __half *c_imag_1) {
  __half2 temp_x, temp_y;

  temp_x = __hsub2(__hmul2(a_real, b_real), __hmul2(a_imag, b_imag));
  temp_y = __hfma2(a_imag, b_real, __hmul2(a_real, b_imag));
  *c_real_0 = temp_x.x;
  *c_imag_0 = temp_y.x;
  *c_real_1 = temp_x.y;
  *c_imag_1 = temp_y.y;
}

__device__ __forceinline__ void complex_mul_half2(complex_half_t a1, complex_half_t a2, complex_half_t b1, complex_half_t b2, complex_half_t *c1, complex_half_t *c2) {
  __half2 a_real, a_imag, b_real, b_imag;

  a_real = __half2(a1.real(), a2.real());
  a_imag = __half2(a1.imag(), a2.imag());
  b_real = __half2(b1.real(), b2.real());
  b_imag = __half2(b1.imag(), b2.imag());

  complex_mul_half2(a_real, a_imag, b_real, b_imag, c1, c2);
}

__device__ __forceinline__ void complex_mul_conj(complex_half_t a, complex_half_t b, complex_half_t *c) {
  __half temp_x, temp_y;
  __half2 temp2;

  temp_x = __hfma(__half(a.real()), __half(b.real()), __half(a.imag() * b.imag()));
  temp2 = __hmul2(__half2(a.imag(), a.real()), __half2(__half(b.real()), __half(b.imag())));
  temp_y = __hsub(temp2.x, temp2.y);
  *c = complex_half_t(temp_x, temp_y);
}

// negates b_imag
__device__ __forceinline__ void complex_mul_conj_half2(__half2 a_real, __half2 a_imag, __half2 b_real, __half2 b_imag, c10::complex<__half> *c_0, c10::complex<__half> *c_1) {
  __half2 temp_x, temp_y;

  temp_x = __hfma2(a_real, b_real, __hmul2(a_imag, b_imag));
  // temp_x = __hsub2(__hmul2(a_real, b_real), __hmul2(a_imag, b_imag));
  temp_y = __hsub2(__hmul2(a_imag, b_real), __hmul2(a_real, b_imag));
  // temp_y = __hfma2(a_imag, b_real, __hmul2(a_real, b_imag));
  *c_0 = c10::complex<__half>(temp_x.x, temp_y.x);
  *c_1 = c10::complex<__half>(temp_x.y, temp_y.y);
}

// negates b_imag
__device__ __forceinline__ void complex_mul_conj_half2(__half2 a_real, __half2 a_imag, __half2 b_real, __half2 b_imag, complex_half_t *c_0, complex_half_t *c_1) {
  __half2 temp_x, temp_y;

  temp_x = __hfma2(a_real, b_real, __hmul2(a_imag, b_imag));
  // temp_x = __hsub2(__hmul2(a_real, b_real), __hmul2(a_imag, b_imag));
  temp_y = __hsub2(__hmul2(a_imag, b_real), __hmul2(a_real, b_imag));
  // temp_y = __hfma2(a_imag, b_real, __hmul2(a_real, b_imag));
  *c_0 = complex_half_t(temp_x.x, temp_y.x);
  *c_1 = complex_half_t(temp_x.y, temp_y.y);
}

__device__ __forceinline__ void complex_mul_conj_half2(complex_half_t a1, complex_half_t a2, complex_half_t b1, complex_half_t b2, complex_half_t *c1, complex_half_t *c2) {
  __half2 a_real, a_imag, b_real, b_imag;

  a_real = __half2(a1.real(), a2.real());
  a_imag = __half2(a1.imag(), a2.imag());
  b_real = __half2(b1.real(), b2.real());
  b_imag = __half2(b1.imag(), b2.imag());

  complex_mul_conj_half2(a_real, a_imag, b_real, b_imag, c1, c2);
}

// negates b_imag
__device__ __forceinline__ void complex_mul_conj_half2(__half2 a_real, __half2 a_imag, c10::complex<__half> b_0, c10::complex<__half> b_1, c10::complex<__half> *c_0, c10::complex<__half> *c_1) {
  __half2 b_real_h2, b_imag_h2;

  b_real_h2 = __half2(b_0.real(), b_1.real());
  b_imag_h2 = __half2(b_0.imag(), b_1.imag());
  complex_mul_conj_half2(a_real, a_imag, b_real_h2, b_imag_h2, c_0, c_1);
}

__device__ __forceinline__ void complex_mul_conj_half2(__half2 a_real, __half2 a_imag, __half2 b_real, __half2 b_imag, __half2 *c_real, __half2 *c_imag) {
  __half2 temp_x, temp_y;

  temp_x = __hfma2(a_real, b_real, __hmul2(a_imag, b_imag));
  // temp_x = __hsub2(__hmul2(a_real, b_real), __hmul2(a_imag, b_imag));
  temp_y = __hsub2(__hmul2(a_imag, b_real), __hmul2(a_real, b_imag));
  // temp_y = __hfma2(a_imag, b_real, __hmul2(a_real, b_imag));
  *c_real = temp_x;
  *c_imag = temp_y;
}

__device__ __forceinline__ complex_half_t conj(complex_half_t inp) {
  return complex_half_t(inp.real(), -inp.imag());
}

#endif