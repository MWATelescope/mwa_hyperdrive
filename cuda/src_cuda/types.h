// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

/**
 * Types to be passed between Rust and CUDA code.
 */

#pragma once

#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

/**
 * The (u,v,w) coordinates of a baseline. There are no units (i.e. these are
 * dimensionless).
 */
typedef struct UVW {
    // u coordinate \[dimensionless\]
    const double u;
    // v coordinate \[dimensionless\]
    const double v;
    // w coordinate \[dimensionless\]
    const double w;
} UVW;

/**
 * The LMN coordinates of a sky-model component.
 */
typedef struct LMN {
    // l coordinate \[dimensionless\]
    const double l;
    // m coordinate \[dimensionless\]
    const double m;
    // n coordinate \[dimensionless\]
    const double n;
} LMN;

/**
 * Parameters describing a Gaussian (also applicable to shapelets).
 */
typedef struct GaussianParams {
    // Major axis size \[radians\]
    const double maj;
    // Minor axis size \[radians\]
    const double min;
    // Position angle \[radians\]
    const double pa;
} GaussianParams;

/**
 * Parameters describing a shapelet coefficient.
 */
typedef struct ShapeletCoeff {
    const size_t n1;
    const size_t n2;
    const double value;
} ShapeletCoeff;

/**
 * (u,v) coordinates for a shapelet. W isn't used, so we're a bit more efficient
 * by not using UVW.
 */
typedef struct ShapeletUV {
    // u coordinate \[dimensionless\]
    const double u;
    // v coordinate \[dimensionless\]
    const double v;
} ShapeletUV;

/**
 * A Jones matrix, single precision. The floats are unpacked into real and imag
 * components because complex numbers don't traverse the FFI boundary well.
 */
typedef struct JonesF32 {
    // Real XX component
    float xx_re;
    // Imag XX component
    float xx_im;
    // Real XY component
    float xy_re;
    // Imag XY component
    float xy_im;
    // Real YX component
    float yx_re;
    // Imag YX component
    float yx_im;
    // Real YY component
    float yy_re;
    // Imag YY component
    float yy_im;
} JonesF32;

/**
 * A Jones matrix, double precision. The floats are unpacked into real and imag
 * components because complex numbers don't traverse the FFI boundary well.
 */
typedef struct JonesF64 {
    // Real XX component
    double xx_re;
    // Imag XX component
    double xx_im;
    // Real XY component
    double xy_re;
    // Imag XY component
    double xy_im;
    // Real YX component
    double yx_re;
    // Imag YX component
    double yx_im;
    // Real YY component
    double yy_re;
    // Imag YY component
    double yy_im;
} JonesF64;

#ifdef __cplusplus
} // extern "C"
#endif // __cplusplus
