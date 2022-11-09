// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

/**
 * Types to be passed between Rust and CUDA code.
 */

#pragma once

#include <stddef.h>

// If SINGLE is enabled, use single-precision floats everywhere. Otherwise
// default to double-precision.
#ifdef SINGLE
#define FLOAT float
#define JONES JonesF32
#else
#define FLOAT double
#define JONES JonesF64
#endif

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

/**
 * The (u,v,w) coordinates of a baseline. They are in units of metres.
 */
typedef struct UVW {
    // u coordinate [metres]
    FLOAT u;
    // v coordinate [metres]
    FLOAT v;
    // w coordinate [metres]
    FLOAT w;
} UVW;

/**
 * The LMN coordinates of a sky-model component (prepared for application in the
   RIME).
 */
typedef struct LmnRime {
    // l coordinate [dimensionless]
    FLOAT l;
    // m coordinate [dimensionless]
    FLOAT m;
    // n coordinate [dimensionless]
    FLOAT n;
} LmnRime;

/**
 * Parameters describing a Gaussian (also applicable to shapelets).
 */
typedef struct GaussianParams {
    // Major axis size [radians]
    FLOAT maj;
    // Minor axis size [radians]
    FLOAT min;
    // Position angle [radians]
    FLOAT pa;
} GaussianParams;

/**
 * Parameters describing a shapelet coefficient.
 */
typedef struct ShapeletCoeff {
    int n1;
    int n2;
    FLOAT value;
} ShapeletCoeff;

/**
 * (u,v) coordinates for a shapelet. W isn't used, so we're a bit more efficient
 * by not using UVW.
 */
typedef struct ShapeletUV {
    // u coordinate [metres]
    FLOAT u;
    // v coordinate [metres]
    FLOAT v;
} ShapeletUV;

/**
 * A Jones matrix, single precision. The floats are unpacked into real and imag
 * components because complex numbers don't traverse the FFI boundary well.
 */
typedef struct JonesF32 {
    // Real part of the (0,0) component
    float j00_re;
    // Imaginary part of the (0,0) component
    float j00_im;
    // Real part of the (0,1) component
    float j01_re;
    // Imaginary part of the (0,1) component
    float j01_im;
    // Real part of the (1,0) component
    float j10_re;
    // Imaginary part of the (1, 0) component
    float j10_im;
    // Real part of the (1,1) component
    float j11_re;
    // Imaginary part of the (1,1) component
    float j11_im;
} JonesF32;

/**
 * A Jones matrix, double precision. The floats are unpacked into real and imag
 * components because complex numbers don't traverse the FFI boundary well.
 */
typedef struct JonesF64 {
    // Real part of the (0,0) component
    double j00_re;
    // Imaginary part of the (0,0) component
    double j00_im;
    // Real part of the (0,1) component
    double j01_re;
    // Imaginary part of the (0,1) component
    double j01_im;
    // Real part of the (1,0) component
    double j10_re;
    // Imaginary part of the (1, 0) component
    double j10_im;
    // Real part of the (1,1) component
    double j11_re;
    // Imaginary part of the (1,1) component
    double j11_im;
} JonesF64;

#ifdef __cplusplus
} // extern "C"
#endif // __cplusplus
