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
 * (right ascension, declination) coordinates. Each is in units of radians.
 */
typedef struct RADec {
    // right ascension coordinate [radians]
    FLOAT ra;
    // declination coordinate [radians]
    FLOAT dec;
} RADec;

/**
 * (hour angle, declination) coordinates. Each is in units of radians.
 */
typedef struct HADec {
    // hour angle coordinate [radians]
    FLOAT ha;
    // declination coordinate [radians]
    FLOAT dec;
} HADec;

/**
 * The (x,y,z) coordinates of an antenna/tile/station. They are in units of
 * metres.
 */
typedef struct XYZ {
    // x coordinate [metres]
    FLOAT x;
    // y coordinate [metres]
    FLOAT y;
    // z coordinate [metres]
    FLOAT z;
} XYZ;

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

/**
 * Common things needed to perform modelling. All pointers are to device
 * memory.
 */
typedef struct Addresses {
    const int num_freqs;
    const int num_vis;
    const int num_baselines;
    const int sbf_l;
    const int sbf_n;
    const FLOAT sbf_c;
    const FLOAT sbf_dx;
    const FLOAT *d_freqs;
    const FLOAT *d_shapelet_basis_values;
    const int num_unique_beam_freqs;
    const int *d_tile_map;
    const int *d_freq_map;
    const int *d_tile_index_to_unflagged_tile_index_map;
    JonesF32 *d_vis;
} Addresses;

/**
 * All the parameters needed to describe point-source components.
 */
typedef struct Points {
    const int num_power_laws;
    const LmnRime *power_law_lmns;
    // Instrumental flux densities calculated at 150 MHz.
    const JONES *power_law_fds;
    // Spectral indices.
    const FLOAT *power_law_sis;

    const int num_curved_power_laws;
    const LmnRime *curved_power_law_lmns;
    // Instrumental flux densities calculated at 150 MHz.
    const JONES *curved_power_law_fds;
    // Spectral indices.
    const FLOAT *curved_power_law_sis;
    // Spectral curvatures.
    const FLOAT *curved_power_law_qs;

    const int num_lists;
    const LmnRime *list_lmns;
    // Instrumental (i.e. XX, XY, YX, XX).
    const JONES *list_fds;
} Points;

/**
 * All the parameters needed to describe Gaussian components.
 */
typedef struct Gaussians {
    const int num_power_laws;
    const LmnRime *power_law_lmns;
    // Instrumental flux densities calculated at 150 MHz.
    const JONES *power_law_fds;
    // Spectral indices.
    const FLOAT *power_law_sis;
    const GaussianParams *power_law_gps;

    const int num_curved_power_laws;
    const LmnRime *curved_power_law_lmns;
    // Instrumental flux densities calculated at 150 MHz.
    const JONES *curved_power_law_fds;
    // Spectral indices.
    const FLOAT *curved_power_law_sis;
    // Spectral curvatures.
    const FLOAT *curved_power_law_qs;
    const GaussianParams *curved_power_law_gps;

    const int num_lists;
    const LmnRime *list_lmns;
    // Instrumental (i.e. XX, XY, YX, XX).
    const JONES *list_fds;
    const GaussianParams *list_gps;
} Gaussians;

/**
 * All the parameters needed to describe Shapelet components.
 */
typedef struct Shapelets {
    const int num_power_laws;
    const LmnRime *power_law_lmns;
    // Instrumental flux densities calculated at 150 MHz.
    const JONES *power_law_fds;
    // Spectral indices.
    const FLOAT *power_law_sis;
    const GaussianParams *power_law_gps;
    const ShapeletUV *power_law_shapelet_uvs;
    const ShapeletCoeff *power_law_shapelet_coeffs;
    const int *power_law_num_shapelet_coeffs;

    const int num_curved_power_laws;
    const LmnRime *curved_power_law_lmns;
    // Instrumental flux densities calculated at 150 MHz.
    const JONES *curved_power_law_fds;
    // Spectral indices.
    const FLOAT *curved_power_law_sis;
    // Spectral curvatures.
    const FLOAT *curved_power_law_qs;
    const GaussianParams *curved_power_law_gps;
    const ShapeletUV *curved_power_law_shapelet_uvs;
    const ShapeletCoeff *curved_power_law_shapelet_coeffs;
    const int *curved_power_law_num_shapelet_coeffs;

    const int num_lists;
    const LmnRime *list_lmns;
    // Instrumental (i.e. XX, XY, YX, XX).
    const JONES *list_fds;
    const GaussianParams *list_gps;
    const ShapeletUV *list_shapelet_uvs;
    const ShapeletCoeff *list_shapelet_coeffs;
    const int *list_num_shapelet_coeffs;
} Shapelets;

typedef struct IonoConsts {
    double alpha;
    double beta;
    double gain;
} IonoConsts;

#ifdef __cplusplus
} // extern "C"
#endif // __cplusplus
