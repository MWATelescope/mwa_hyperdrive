// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

#pragma once

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>

#include <cuComplex.h>

#include "types.h"

// If SINGLE is enabled, use single-precision floats everywhere. Otherwise
// default to double-precision.
#ifdef SINGLE
#define FLOAT4  float4
#define SINCOS  sincosf
#define EXP     expf
#define POW     powf
#define FLOOR   floorf
#define COMPLEX cuFloatComplex
#define CUCONJ  cuConjf
#define LOG     logf
#define EXP     expf
#else
#define FLOAT4  double4
#define SINCOS  sincos
#define EXP     exp
#define POW     pow
#define FLOOR   floor
#define COMPLEX cuDoubleComplex
#define CUCONJ  cuConj
#define LOG     log
#define EXP     exp
#endif

const FLOAT VEL_C = 299792458.0;                           // speed of light in a vacuum
const FLOAT LN_2 = 0.6931471805599453;                     // ln(2), or log_e(2)
const FLOAT TAU = 6.283185307179586;                       // 2 * PI
const FLOAT FRAC_PI_2 = 1.5707963267948966;                // PI / 2
const FLOAT SQRT_FRAC_PI_SQ_2_LN_2 = 2.6682231283184983;   // sqrt( PI^2 / (2 ln(2)) )
const FLOAT EXP_CONST = -((FRAC_PI_2 * FRAC_PI_2) / LN_2); // -( (PI/2)^2 / ln(2) )

typedef struct JONES_C {
    // The (0,0) component
    COMPLEX j00;
    // The (0,1) component
    COMPLEX j01;
    // The (1,0) component
    COMPLEX j10;
    // The (1,1) component
    COMPLEX j11;
} JONES_C;

inline __device__ COMPLEX operator+(const COMPLEX a, const COMPLEX b) {
    return COMPLEX{
        .x = a.x + b.x,
        .y = a.y + b.y,
    };
}

inline __device__ COMPLEX operator*(const COMPLEX a, const COMPLEX b) {
    return COMPLEX{
        .x = a.x * b.x - a.y * b.y,
        .y = a.x * b.y + a.y * b.x,
    };
}

inline __device__ void operator*=(COMPLEX &a, const COMPLEX b) {
    a = COMPLEX{
        .x = a.x * b.x - a.y * b.y,
        .y = a.x * b.y + a.y * b.x,
    };
}

inline __device__ COMPLEX operator*(const COMPLEX a, const FLOAT b) {
    return COMPLEX{
        .x = a.x * b,
        .y = a.y * b,
    };
}

inline __device__ void operator+=(COMPLEX &a, const COMPLEX b) {
    a.x += b.x;
    a.y += b.y;
}

inline __device__ JONES operator*(const JONES a, const FLOAT b) {
    return JONES{
        .j00_re = a.j00_re * b,
        .j00_im = a.j00_im * b,
        .j01_re = a.j01_re * b,
        .j01_im = a.j01_im * b,
        .j10_re = a.j10_re * b,
        .j10_im = a.j10_im * b,
        .j11_re = a.j11_re * b,
        .j11_im = a.j11_im * b,
    };
}

inline __device__ JONES operator*(const JONES a, const COMPLEX b) {
    return JONES{
        .j00_re = a.j00_re * b.x - a.j00_im * b.y,
        .j00_im = a.j00_re * b.y + a.j00_im * b.x,
        .j01_re = a.j01_re * b.x - a.j01_im * b.y,
        .j01_im = a.j01_re * b.y + a.j01_im * b.x,
        .j10_re = a.j10_re * b.x - a.j10_im * b.y,
        .j10_im = a.j10_re * b.y + a.j10_im * b.x,
        .j11_re = a.j11_re * b.x - a.j11_im * b.y,
        .j11_im = a.j11_re * b.y + a.j11_im * b.x,
    };
}

inline __device__ void operator+=(JONES &a, const JONES b) {
    a.j00_re += b.j00_re;
    a.j00_im += b.j00_im;
    a.j01_re += b.j01_re;
    a.j01_im += b.j01_im;
    a.j10_re += b.j10_re;
    a.j10_im += b.j10_im;
    a.j11_re += b.j11_re;
    a.j11_im += b.j11_im;
}

inline __device__ JonesF32 operator+(JonesF32 a, JonesF32 b) {
    return JonesF32{
        .j00_re = a.j00_re + b.j00_re,
        .j00_im = a.j00_im + b.j00_im,
        .j01_re = a.j01_re + b.j01_re,
        .j01_im = a.j01_im + b.j01_im,
        .j10_re = a.j10_re + b.j10_re,
        .j10_im = a.j10_im + b.j10_im,
        .j11_re = a.j11_re + b.j11_re,
        .j11_im = a.j11_im + b.j11_im,
    };
}

inline __device__ JonesF64 operator+(JonesF64 a, JonesF64 b) {
    return JonesF64{
        .j00_re = a.j00_re + b.j00_re,
        .j00_im = a.j00_im + b.j00_im,
        .j01_re = a.j01_re + b.j01_re,
        .j01_im = a.j01_im + b.j01_im,
        .j10_re = a.j10_re + b.j10_re,
        .j10_im = a.j10_im + b.j10_im,
        .j11_re = a.j11_re + b.j11_re,
        .j11_im = a.j11_im + b.j11_im,
    };
}

inline __device__ void operator+=(JonesF32 &a, const JonesF64 b) {
    a.j00_re += (float)b.j00_re;
    a.j00_im += (float)b.j00_im;
    a.j01_re += (float)b.j01_re;
    a.j01_im += (float)b.j01_im;
    a.j10_re += (float)b.j10_re;
    a.j10_im += (float)b.j10_im;
    a.j11_re += (float)b.j11_re;
    a.j11_im += (float)b.j11_im;
}

inline __device__ JONES operator/(JONES a, FLOAT b) {
    return JONES{
        .j00_re = a.j00_re / b,
        .j00_im = a.j00_im / b,
        .j01_re = a.j01_re / b,
        .j01_im = a.j01_im / b,
        .j10_re = a.j10_re / b,
        .j10_im = a.j10_im / b,
        .j11_re = a.j11_re / b,
        .j11_im = a.j11_im / b,
    };
}

inline __device__ void operator/=(JONES &a, FLOAT b) {
    a.j00_re /= b;
    a.j00_im /= b;
    a.j01_re /= b;
    a.j01_im /= b;
    a.j10_re /= b;
    a.j10_im /= b;
    a.j11_re /= b;
    a.j11_im /= b;
}

inline __device__ void operator+=(volatile JONES &a, volatile JONES b) {
    a.j00_re += b.j00_re;
    a.j00_im += b.j00_im;
    a.j01_re += b.j01_re;
    a.j01_im += b.j01_im;
    a.j10_re += b.j10_re;
    a.j10_im += b.j10_im;
    a.j11_re += b.j11_re;
    a.j11_im += b.j11_im;
}

inline __device__ void operator-=(JONES &a, JONES b) {
    a.j00_re -= b.j00_re;
    a.j00_im -= b.j00_im;
    a.j01_re -= b.j01_re;
    a.j01_im -= b.j01_im;
    a.j10_re -= b.j10_re;
    a.j10_im -= b.j10_im;
    a.j11_re -= b.j11_re;
    a.j11_im -= b.j11_im;
}

inline __device__ void operator+=(volatile JonesF64 &a, volatile JonesF64 b) {
    a.j00_re += b.j00_re;
    a.j00_im += b.j00_im;
    a.j01_re += b.j01_re;
    a.j01_im += b.j01_im;
    a.j10_re += b.j10_re;
    a.j10_im += b.j10_im;
    a.j11_re += b.j11_re;
    a.j11_im += b.j11_im;
}

inline __device__ void operator+=(JonesF64 &a, JonesF32 b) {
    a.j00_re += (double)b.j00_re;
    a.j00_im += (double)b.j00_im;
    a.j01_re += (double)b.j01_re;
    a.j01_im += (double)b.j01_im;
    a.j10_re += (double)b.j10_re;
    a.j10_im += (double)b.j10_im;
    a.j11_re += (double)b.j11_re;
    a.j11_im += (double)b.j11_im;
}
inline __device__ UVW operator*(const UVW a, const FLOAT b) {
    return UVW{
        .u = a.u * b,
        .v = a.v * b,
        .w = a.w * b,
    };
}

inline __device__ UVW operator/(const UVW a, const FLOAT b) {
    return UVW{
        .u = a.u / b,
        .v = a.v / b,
        .w = a.w / b,
    };
}

inline __device__ ShapeletUV operator*(const ShapeletUV a, const FLOAT b) {
    return ShapeletUV{
        .u = a.u * b,
        .v = a.v * b,
    };
}

/**
 * Multiply a Jones matrix by two beam Jones matrices (i.e. J1 . J . J2^H).
 */
inline __device__ void apply_beam(const JONES *j1, JONES *j, const JONES *j2) {
    // Cast the input Jones matrices to complex forms for convenience.
    JONES_C *j1c = (JONES_C *)j1;
    JONES_C *jc = (JONES_C *)j;
    JONES_C *j2c = (JONES_C *)j2;

    // J1 . J
    JONES_C temp{
        .j00 = j1c->j00 * jc->j00 + j1c->j01 * jc->j10,
        .j01 = j1c->j00 * jc->j01 + j1c->j01 * jc->j11,
        .j10 = j1c->j10 * jc->j00 + j1c->j11 * jc->j10,
        .j11 = j1c->j10 * jc->j01 + j1c->j11 * jc->j11,
    };

    // J2^H
    JONES_C j2h = JONES_C{
        .j00 = CUCONJ(j2c->j00),
        .j01 = CUCONJ(j2c->j10),
        .j10 = CUCONJ(j2c->j01),
        .j11 = CUCONJ(j2c->j11),
    };

    // (J1 . J) . J2^H
    jc->j00 = temp.j00 * j2h.j00 + temp.j01 * j2h.j10;
    jc->j01 = temp.j00 * j2h.j01 + temp.j01 * j2h.j11;
    jc->j10 = temp.j10 * j2h.j00 + temp.j11 * j2h.j10;
    jc->j11 = temp.j10 * j2h.j01 + temp.j11 * j2h.j11;
}

/**
 * Multiply a Jones matrix by two beam Jones matrices (i.e. J1 . J . J2^H).
 */
inline __device__ JONES apply_beam2(const JONES *j1, const JONES *j, const JONES *j2) {
    // Cast the input Jones matrices to complex forms for convenience.
    JONES_C *j1c = (JONES_C *)j1;
    JONES_C *jc = (JONES_C *)j;
    JONES_C *j2c = (JONES_C *)j2;
    JONES_C temp;

    // J1 . J
    temp.j00 = j1c->j00 * jc->j00 + j1c->j01 * jc->j10;
    temp.j01 = j1c->j00 * jc->j01 + j1c->j01 * jc->j11;
    temp.j10 = j1c->j10 * jc->j00 + j1c->j11 * jc->j10;
    temp.j11 = j1c->j10 * jc->j01 + j1c->j11 * jc->j11;

    // J2^H
    JONES_C j2h = JONES_C{
        .j00 = CUCONJ(j2c->j00),
        .j01 = CUCONJ(j2c->j10),
        .j10 = CUCONJ(j2c->j01),
        .j11 = CUCONJ(j2c->j11),
    };

    // (J1 . J) . J2^H
    temp = JONES_C{
        jc->j00 = temp.j00 * j2h.j00 + temp.j01 * j2h.j10,
        jc->j01 = temp.j00 * j2h.j01 + temp.j01 * j2h.j11,
        jc->j10 = temp.j10 * j2h.j00 + temp.j11 * j2h.j10,
        jc->j11 = temp.j10 * j2h.j01 + temp.j11 * j2h.j11,
    };
    return JONES{
        .j00_re = temp.j00.x,
        .j00_im = temp.j00.y,
        .j01_re = temp.j01.x,
        .j01_im = temp.j01.y,
        .j10_re = temp.j10.x,
        .j10_im = temp.j10.y,
        .j11_re = temp.j11.x,
        .j11_im = temp.j11.y,
    };
}

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

/**
 * `gpu_assert` checks that CUDA code successfully returned.
 */
inline void gpu_assert(cudaError_t code, const char *file, int line, bool abort = true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "gpu_assert: %s (%s:%d)\n", cudaGetErrorString(code), file, line);
        if (abort)
            exit(code);
    }
}

// `cudaCheck` wraps `gpu_assert` for general usage.
#define cudaCheck(code)                                                                                                \
    { gpu_assert((code), __FILE__, __LINE__); }

#ifndef NDEBUG
#define cudaSoftCheck(code)                                                                                            \
    { gpu_assert((code), __FILE__, __LINE__); }
#else
// When not debugging, `cudaSoftCheck` is a "no-op". Useful for granting full speed in release builds.
#define cudaSoftCheck(code) (code)
#endif

#ifdef __cplusplus
} // extern "C"
#endif // __cplusplus

inline __device__ FLOAT4 operator+(FLOAT4 a, FLOAT4 b) {
    FLOAT4 t;
    t.x = a.x + b.x;
    t.y = a.y + b.y;
    t.z = a.z + b.z;
    t.w = a.w + b.w;
    return t;
}

inline __device__ FLOAT4 operator-(FLOAT4 a, FLOAT4 b) {
    FLOAT4 t;
    t.x = a.x - b.x;
    t.y = a.y - b.y;
    t.z = a.z - b.z;
    t.w = a.w - b.w;
    return t;
}

inline __device__ FLOAT4 operator*(FLOAT4 a, FLOAT b) {
    FLOAT4 t;
    t.x = a.x * b;
    t.y = a.y * b;
    t.z = a.z * b;
    t.w = a.w * b;
    return t;
}

inline __device__ void operator+=(FLOAT4 &a, FLOAT4 b) {
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
    a.w += b.w;
}

/**
 * Multiply a Jones matrix by a complex number and accumulate in
 * another Jones matrix.
 */
inline __device__ JONES complex_multiply(JONES j, FLOAT real, FLOAT imag) {
    FLOAT4 j_re = FLOAT4{
        .x = j.j00_re,
        .y = j.j01_re,
        .z = j.j10_re,
        .w = j.j11_re,
    };
    FLOAT4 j_im = FLOAT4{
        .x = j.j00_im,
        .y = j.j01_im,
        .z = j.j10_im,
        .w = j.j11_im,
    };
    FLOAT4 new_re = j_re * real - j_im * imag;
    FLOAT4 new_im = j_re * imag + j_im * real;

    return JONES{
        .j00_re = new_re.x,
        .j00_im = new_im.x,
        .j01_re = new_re.y,
        .j01_im = new_im.y,
        .j10_re = new_re.z,
        .j10_im = new_im.z,
        .j11_re = new_re.w,
        .j11_im = new_im.w,
    };
}
