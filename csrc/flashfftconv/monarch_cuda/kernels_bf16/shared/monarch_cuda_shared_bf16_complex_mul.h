// Copyright (c) 2023 Dan Fu, Hermann Kumbong

#include <torch/extension.h>

#include <cuda_fp16.h>
#include <cuda_bf16.h>
using namespace nvcuda;

#ifndef MONARCH_CUDA_BF16_COMPLEX_MUL_
#define MONARCH_CUDA_BF16_COMPLEX_MUL_

using complex_bfloat16_t = typename c10::complex<at::BFloat16>;

__device__ __forceinline__ void complex_mul(at::BFloat16 a_real, at::BFloat16 a_imag, at::BFloat16 b_real, at::BFloat16 b_imag, at::BFloat16 *c_real, at::BFloat16 *c_imag) {
  __nv_bfloat16 temp_x, temp_y;
  // temp_x = __hsub(__hmul(a_real, b_real), __hmul(a_imag, b_imag));
  // temp_y = __hadd(__hmul(a_imag, b_real), __hmul(a_real, b_imag));
  temp_x = __nv_bfloat16(a_real * b_real - a_imag * b_imag);
  temp_y = __hfma(__nv_bfloat16(a_imag), __nv_bfloat16(b_real), __nv_bfloat16(a_real * b_imag));
  *c_real = temp_x;
  *c_imag = temp_y;
}

__device__ __forceinline__ void complex_mul(complex_bfloat16_t a, complex_bfloat16_t b, complex_bfloat16_t *c) {
  __nv_bfloat16 temp_x, temp_y;
  __nv_bfloat162 temp2;
  // temp_x = __hsub(__hmul(a_real, b_real), __hmul(a_imag, b_imag));
  // temp_y = __hadd(__hmul(a_imag, b_real), __hmul(a_real, b_imag));
  // temp_x = __half(a.real() * b.real() - a.imag() * b.imag());
  temp2 = __hmul2(
    __nv_bfloat162(
      __nv_bfloat16(a.real()),
      __nv_bfloat16(a.imag())
    ),
    __nv_bfloat162(
      __nv_bfloat16(b.real()),
      __nv_bfloat16(b.imag())
    )
  );
  temp_x = __hsub(temp2.x, temp2.y);
  temp_y = __hfma(
    __nv_bfloat16(a.imag()), __nv_bfloat16(b.real()),
    __nv_bfloat16(a.real() * b.imag())
  );
  *c = complex_bfloat16_t(temp_x, temp_y);
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

__device__ __forceinline__ void complex_mul_bfloat162(__nv_bfloat162 a_real, __nv_bfloat162 a_imag, __nv_bfloat162 b_real, __nv_bfloat162 b_imag, complex_bfloat16_t *c1, complex_bfloat16_t *c2) {
  __nv_bfloat162 temp_x, temp_y;

  temp_x = __hsub2(__hmul2(a_real, b_real), __hmul2(a_imag, b_imag));
  temp_y = __hfma2(a_imag, b_real, __hmul2(a_real, b_imag));
  *c1 = complex_bfloat16_t(temp_x.x, temp_y.x);
  *c2 = complex_bfloat16_t(temp_x.y, temp_y.y);
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

__device__ __forceinline__ void complex_mul_bfloat162(complex_bfloat16_t a1, complex_bfloat16_t a2, complex_bfloat16_t b1, complex_bfloat16_t b2, complex_bfloat16_t *c1, complex_bfloat16_t *c2) {
  __nv_bfloat162 a_real, a_imag, b_real, b_imag;

  a_real = __nv_bfloat162(
    __nv_bfloat16(a1.real()),
    __nv_bfloat16(a2.real())
  );
  a_imag = __nv_bfloat162(
    __nv_bfloat16(a1.imag()),
    __nv_bfloat16(a2.imag())
  );
  b_real = __nv_bfloat162(
    __nv_bfloat16(b1.real()),
    __nv_bfloat16(b2.real())
  );
  b_imag = __nv_bfloat162(
    __nv_bfloat16(b1.imag()),
    __nv_bfloat16(b2.imag())
  );

  complex_mul_bfloat162(a_real, a_imag, b_real, b_imag, c1, c2);
}

__device__ __forceinline__ void complex_mul_conj(complex_bfloat16_t a, complex_bfloat16_t b, complex_bfloat16_t *c) {
  __nv_bfloat16 temp_x, temp_y;
  __nv_bfloat162 temp2;

  temp_x = __hfma(__nv_bfloat16(a.real()), __nv_bfloat16(b.real()), __nv_bfloat16(a.imag() * b.imag()));
  temp2 = __hmul2(
    __nv_bfloat162(
      __nv_bfloat16(a.imag()),
      __nv_bfloat16(a.real())
    ),
    __nv_bfloat162(
      __nv_bfloat16(b.real()),
      __nv_bfloat16(b.imag())
    )
  );
  temp_y = __hsub(temp2.x, temp2.y);
  *c = complex_bfloat16_t(temp_x, temp_y);
}

// negates b_imag
__device__ __forceinline__ void complex_mul_conj_bfloat162(
  __nv_bfloat162 a_real, 
  __nv_bfloat162 a_imag,
  __nv_bfloat162 b_real,
  __nv_bfloat162 b_imag,
  c10::complex<__nv_bfloat16> *c_0,
  c10::complex<__nv_bfloat16> *c_1
) {
  __nv_bfloat162 temp_x, temp_y;

  temp_x = __hfma2(a_real, b_real, __hmul2(a_imag, b_imag));
  // temp_x = __hsub2(__hmul2(a_real, b_real), __hmul2(a_imag, b_imag));
  temp_y = __hsub2(__hmul2(a_imag, b_real), __hmul2(a_real, b_imag));
  // temp_y = __hfma2(a_imag, b_real, __hmul2(a_real, b_imag));
  *c_0 = c10::complex<__nv_bfloat16>(temp_x.x, temp_y.x);
  *c_1 = c10::complex<__nv_bfloat16>(temp_x.y, temp_y.y);
}

// negates b_imag
__device__ __forceinline__ void complex_mul_conj_bfloat162(__nv_bfloat162 a_real, __nv_bfloat162 a_imag, __nv_bfloat162 b_real, __nv_bfloat162 b_imag, complex_bfloat16_t *c_0, complex_bfloat16_t *c_1) {
  __nv_bfloat162 temp_x, temp_y;

  temp_x = __hfma2(a_real, b_real, __hmul2(a_imag, b_imag));
  // temp_x = __hsub2(__hmul2(a_real, b_real), __hmul2(a_imag, b_imag));
  temp_y = __hsub2(__hmul2(a_imag, b_real), __hmul2(a_real, b_imag));
  // temp_y = __hfma2(a_imag, b_real, __hmul2(a_real, b_imag));
  *c_0 = complex_bfloat16_t(temp_x.x, temp_y.x);
  *c_1 = complex_bfloat16_t(temp_x.y, temp_y.y);
}

__device__ __forceinline__ void complex_mul_conj_bfloat162(complex_bfloat16_t a1, complex_bfloat16_t a2, complex_bfloat16_t b1, complex_bfloat16_t b2, complex_bfloat16_t *c1, complex_bfloat16_t *c2) {
  __nv_bfloat162 a_real, a_imag, b_real, b_imag;

  a_real = __nv_bfloat162(
    __nv_bfloat16(a1.real()),
    __nv_bfloat16(a2.real())
  );
  a_imag = __nv_bfloat162(
    __nv_bfloat16(a1.imag()),
    __nv_bfloat16(a2.imag())
  );
  b_real = __nv_bfloat162(
    __nv_bfloat16(b1.real()),
    __nv_bfloat16(b2.real())
  );
  b_imag = __nv_bfloat162(
    __nv_bfloat16(b1.imag()),
    __nv_bfloat16(b2.imag())
  );

  complex_mul_conj_bfloat162(a_real, a_imag, b_real, b_imag, c1, c2);
}

__device__ __forceinline__ void complex_mul_conj_bfloat162(
  __nv_bfloat162 a_real, 
  __nv_bfloat162 a_imag, 
  __nv_bfloat162 b_real, 
  __nv_bfloat162 b_imag, 
  __nv_bfloat162 *c_real,
  __nv_bfloat162 *c_imag
) {
  __nv_bfloat162 temp_x, temp_y;

  temp_x = __hfma2(a_real, b_real, __hmul2(a_imag, b_imag));
  // temp_x = __hsub2(__hmul2(a_real, b_real), __hmul2(a_imag, b_imag));
  temp_y = __hsub2(__hmul2(a_imag, b_real), __hmul2(a_real, b_imag));
  // temp_y = __hfma2(a_imag, b_real, __hmul2(a_real, b_imag));
  *c_real = temp_x;
  *c_imag = temp_y;
}

__device__ __forceinline__ void complex_mul_conj_bfloat162(
  __nv_bfloat162 a_real,
  __nv_bfloat162 a_imag,
  c10::complex<__nv_bfloat16> b_0,
  c10::complex<__nv_bfloat16> b_1,
  c10::complex<__nv_bfloat16> *c_0,
  c10::complex<__nv_bfloat16> *c_1) {
  __nv_bfloat162 b_real_h2, b_imag_h2;

  b_real_h2 = __nv_bfloat162(b_0.real(), b_1.real());
  b_imag_h2 = __nv_bfloat162(b_0.imag(), b_1.imag());
  complex_mul_conj_bfloat162(a_real, a_imag, b_real_h2, b_imag_h2, c_0, c_1);
}

__device__ __forceinline__ complex_bfloat16_t conj(complex_bfloat16_t inp) {
  return complex_bfloat16_t(inp.real(), -inp.imag());
}


#endif