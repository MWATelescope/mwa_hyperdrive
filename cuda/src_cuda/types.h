// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

/**
 * Types to be passed between Rust and CUDA code.
 */

#pragma once

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
 * The LMN coordinates of a sky-model component.
 */
typedef struct LMN {
    // l coordinate [dimensionless]
    FLOAT l;
    // m coordinate [dimensionless]
    FLOAT m;
    // n coordinate [dimensionless]
    FLOAT n;
} LMN;

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
    size_t n1;
    size_t n2;
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

/**
 * All the parameters needed to describe point-source components.
 */
typedef struct Points {
    size_t num_power_law_points;
    LMN *power_law_lmns;
    // Instrumental flux densities calculated at 150 MHz.
    JONES *power_law_fds;
    // Spectral indices.
    FLOAT *power_law_sis;

    size_t num_list_points;
    LMN *list_lmns;
    // Instrumental (i.e. XX, XY, YX, XX).
    JONES *list_fds;
} Points;

/**
 * All the parameters needed to describe Gaussian components.
 */
typedef struct Gaussians {
    size_t num_power_law_gaussians;
    LMN *power_law_lmns;
    // Instrumental flux densities calculated at 150 MHz.
    JONES *power_law_fds;
    // Spectral indices.
    FLOAT *power_law_sis;
    GaussianParams *power_law_gps;

    size_t num_list_gaussians;
    LMN *list_lmns;
    // Instrumental (i.e. XX, XY, YX, XX).
    JONES *list_fds;
    GaussianParams *list_gps;
} Gaussians;

/**
 * All the parameters needed to describe Shapelet components.
 */
typedef struct Shapelets {
    size_t num_power_law_shapelets;
    LMN *power_law_lmns;
    // Instrumental flux densities calculated at 150 MHz.
    JONES *power_law_fds;
    // Spectral indices.
    FLOAT *power_law_sis;
    GaussianParams *power_law_gps;
    ShapeletUV *power_law_shapelet_uvs;
    ShapeletCoeff *power_law_shapelet_coeffs;
    size_t *power_law_num_shapelet_coeffs;

    size_t num_list_shapelets;
    LMN *list_lmns;
    // Instrumental (i.e. XX, XY, YX, XX).
    JONES *list_fds;
    GaussianParams *list_gps;
    ShapeletUV *list_shapelet_uvs;
    ShapeletCoeff *list_shapelet_coeffs;
    size_t *list_num_shapelet_coeffs;
} Shapelets;

#ifdef __cplusplus
} // extern "C"
#endif // __cplusplus
