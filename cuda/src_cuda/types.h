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
 * (RA, Dec.) coordinates. Both have units of radians.
 */
typedef struct RADec {
    // Right Ascension [radians]
    const double ra;
    // Declination [radians]
    const double dec;
} RADec;

/**
 * The (u,v,w) coordinates of a baseline. They are in units of metres.
 */
typedef struct UVW {
    // u coordinate [metres]
    const double u;
    // v coordinate [metres]
    const double v;
    // w coordinate [metres]
    const double w;
} UVW;

/**
 * The LMN coordinates of a sky-model component.
 */
typedef struct LMN {
    // l coordinate [dimensionless]
    const double l;
    // m coordinate [dimensionless]
    const double m;
    // n coordinate [dimensionless]
    const double n;
} LMN;

/**
 * Parameters describing a Gaussian (also applicable to shapelets).
 */
typedef struct GaussianParams {
    // Major axis size [radians]
    const double maj;
    // Minor axis size [radians]
    const double min;
    // Position angle [radians]
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
    // u coordinate [metres]
    const double u;
    // v coordinate [metres]
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

/**
 * All the parameters needed to describe point-source components.
 */
typedef struct Points {
    const size_t num_power_law_points;
    const RADec *power_law_radecs;
    const LMN *power_law_lmns;
    // Instrumental flux densities calculated at 150 MHz.
    const JonesF64 *power_law_fds;
    // Spectral indices.
    const double *power_law_sis;

    const size_t num_list_points;
    const RADec *list_radecs;
    const LMN *list_lmns;
    // Instrumental (i.e. XX, XY, YX, XX).
    const JonesF64 *list_fds;
} Points;

/**
 * All the parameters needed to describe Gaussian components.
 */
typedef struct Gaussians {
    const size_t num_power_law_gaussians;
    const RADec *power_law_radecs;
    const LMN *power_law_lmns;
    // Instrumental flux densities calculated at 150 MHz.
    const JonesF64 *power_law_fds;
    // Spectral indices.
    const double *power_law_sis;
    const GaussianParams *power_law_gps;

    const size_t num_list_gaussians;
    const RADec *list_radecs;
    const LMN *list_lmns;
    // Instrumental (i.e. XX, XY, YX, XX).
    const JonesF64 *list_fds;
    const GaussianParams *list_gps;
} Gaussians;

/**
 * All the parameters needed to describe Shapelet components.
 */
typedef struct Shapelets {
    const size_t num_power_law_shapelets;
    const RADec *power_law_radecs;
    const LMN *power_law_lmns;
    // Instrumental flux densities calculated at 150 MHz.
    const JonesF64 *power_law_fds;
    // Spectral indices.
    const double *power_law_sis;
    const GaussianParams *power_law_gps;
    const ShapeletUV *power_law_shapelet_uvs;
    const ShapeletCoeff *power_law_shapelet_coeffs;
    const size_t *power_law_num_shapelet_coeffs;

    const size_t num_list_shapelets;
    const RADec *list_radecs;
    const LMN *list_lmns;
    // Instrumental (i.e. XX, XY, YX, XX).
    const JonesF64 *list_fds;
    const GaussianParams *list_gps;
    const ShapeletUV *list_shapelet_uvs;
    const ShapeletCoeff *list_shapelet_coeffs;
    const size_t *list_num_shapelet_coeffs;
} Shapelets;

#ifdef __cplusplus
} // extern "C"
#endif // __cplusplus
