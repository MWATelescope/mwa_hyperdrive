// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <stdlib.h>

#include <cuComplex.h>

#include "common.cuh"
#include "memory.h"
#include "model.h"
#include "types.h"

// If SINGLE is enabled, use single-precision floats everywhere. Otherwise
// default to double-precision.
#ifdef SINGLE
#define FLOAT4    float4
#define SINCOS    sincosf
#define EXP       expf
#define POW       powf
#define FLOOR     floorf
#define CUCOMPLEX cuFloatComplex
#define CUCONJ    cuConjf
#else
#define FLOAT4    double4
#define SINCOS    sincos
#define EXP       exp
#define POW       pow
#define FLOOR     floor
#define CUCOMPLEX cuDoubleComplex
#define CUCONJ    cuConj
#endif

/**
 * A Jones matrix using the specified precision and CUDA complex types.
 */
typedef struct JONES_C {
    CUCOMPLEX j00;
    CUCOMPLEX j01;
    CUCOMPLEX j10;
    CUCOMPLEX j11;
} JONES_C;

const FLOAT VEL_C = 299792458.0;
const FLOAT LN_2 = 0.6931471805599453;
const FLOAT FRAC_PI_2 = 1.5707963267948966;
const FLOAT SQRT_FRAC_PI_SQ_2_LN_2 = 2.6682231283184983;
const FLOAT EXP_CONST = -((FRAC_PI_2 * FRAC_PI_2) / LN_2);

inline __device__ JONES operator*(JONES a, FLOAT b) {
    JONES t;
    t.xx_re = a.xx_re * b;
    t.xx_im = a.xx_im * b;
    t.xy_re = a.xy_re * b;
    t.xy_im = a.xy_im * b;
    t.yx_re = a.yx_re * b;
    t.yx_im = a.yx_im * b;
    t.yy_re = a.yy_re * b;
    t.yy_im = a.yy_im * b;
    return t;
}

inline __device__ CUCOMPLEX operator+(CUCOMPLEX a, CUCOMPLEX b) {
    CUCOMPLEX t;
    t.x = a.x + b.x;
    t.y = a.y + b.y;
    return t;
}

inline __device__ CUCOMPLEX operator*(CUCOMPLEX a, CUCOMPLEX b) {
    CUCOMPLEX t;
    t.x = a.x * b.x - a.y * b.y;
    t.y = a.x * b.y + a.y * b.x;
    return t;
}

inline __device__ CUCOMPLEX operator*(CUCOMPLEX a, FLOAT b) {
    CUCOMPLEX t;
    t.x = a.x * b;
    t.y = a.y * b;
    return t;
}

inline __device__ JONES_C operator*(JONES_C a, FLOAT b) {
    JONES_C t;
    t.j00 = a.j00 * b;
    t.j01 = a.j01 * b;
    t.j10 = a.j10 * b;
    t.j11 = a.j11 * b;
    return t;
}

inline __device__ void operator+=(JONES &a, JONES &b) {
    a.xx_re += b.xx_re;
    a.xx_im += b.xx_im;
    a.xy_re += b.xy_re;
    a.xy_im += b.xy_im;
    a.yx_re += b.yx_re;
    a.yx_im += b.yx_im;
    a.yy_re += b.yy_re;
    a.yy_im += b.yy_im;
}

inline __device__ void operator+=(JonesF32 &a, JonesF64 &b) {
    a.xx_re += (float)b.xx_re;
    a.xx_im += (float)b.xx_im;
    a.xy_re += (float)b.xy_re;
    a.xy_im += (float)b.xy_im;
    a.yx_re += (float)b.yx_re;
    a.yx_im += (float)b.yx_im;
    a.yy_re += (float)b.yy_re;
    a.yy_im += (float)b.yy_im;
}

inline __device__ FLOAT4 operator+(FLOAT4 a, FLOAT4 b) {
    FLOAT4 t;
    t.x = a.x - b.x;
    t.y = a.y - b.y;
    t.z = a.z - b.z;
    t.w = a.w - b.w;
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
inline __device__ void complex_multiply(const JONES *fd, FLOAT real, FLOAT imag, JONES *delta_vis) {
    FLOAT4 fd_re = FLOAT4{
        .x = fd->xx_re,
        .y = fd->xy_re,
        .z = fd->yx_re,
        .w = fd->yy_re,
    };
    FLOAT4 fd_im = FLOAT4{
        .x = fd->xx_im,
        .y = fd->xy_im,
        .z = fd->yx_im,
        .w = fd->yy_im,
    };
    FLOAT4 delta_vis_re = fd_re * real - fd_im * imag;
    FLOAT4 delta_vis_im = fd_re * imag + fd_im * real;

    delta_vis->xx_re += delta_vis_re.x;
    delta_vis->xy_re += delta_vis_re.y;
    delta_vis->yx_re += delta_vis_re.z;
    delta_vis->yy_re += delta_vis_re.w;
    delta_vis->xx_im += delta_vis_im.x;
    delta_vis->xy_im += delta_vis_im.y;
    delta_vis->yx_im += delta_vis_im.z;
    delta_vis->yy_im += delta_vis_im.w;
}

/**
 * Multiply a Jones matrix by two beam Jones matrices (i.e. J1 . J . J2^H).
 */
inline __device__ void apply_beam(const JONES_C *j1, JONES *j, const JONES_C *j2) {
    JONES_C *jm = (JONES_C *)j;
    JONES_C temp;

    // J1 . J
    temp.j00 = j1->j00 * jm->j00 + j1->j01 * jm->j10;
    temp.j01 = j1->j00 * jm->j01 + j1->j01 * jm->j11;
    temp.j10 = j1->j10 * jm->j00 + j1->j11 * jm->j10;
    temp.j11 = j1->j10 * jm->j01 + j1->j11 * jm->j11;

    // J2^H
    JONES_C j2h = JONES_C{
        .j00 = CUCONJ(j2->j00),
        .j01 = CUCONJ(j2->j10),
        .j10 = CUCONJ(j2->j01),
        .j11 = CUCONJ(j2->j11),
    };

    // (J1 . J) . J2^H
    jm->j00 = temp.j00 * j2h.j00 + temp.j01 * j2h.j10;
    jm->j01 = temp.j00 * j2h.j01 + temp.j01 * j2h.j11;
    jm->j10 = temp.j10 * j2h.j00 + temp.j11 * j2h.j10;
    jm->j11 = temp.j10 * j2h.j01 + temp.j11 * j2h.j11;
}

inline __device__ void extrap_power_law_fd(const int i_comp, const FLOAT freq, const JONES *d_ref_fds,
                                           const FLOAT *d_sis, JONES *out) {
    const FLOAT flux_ratio = POW(freq / POWER_LAW_FD_REF_FREQ, d_sis[i_comp]);
    *out = d_ref_fds[i_comp] * flux_ratio;
}

/**
 * Kernel for calculating point-source-component visibilities.
 */
__global__ void model_points_kernel(const int num_freqs, const int num_vis, const UVW *d_uvws, const FLOAT *d_freqs,
                                    const Points d_comps, JonesF32 *d_vis) {
    const int i_vis = threadIdx.x + (blockDim.x * blockIdx.x);
    if (i_vis >= num_vis)
        return;

    // Get the indices for this thread. `i_vis` indexes over baselines and
    // frequencies, with frequencies moving faster than baselines.
    const int i_bl = i_vis / num_freqs;
    const int i_freq = i_vis % num_freqs;
    const int i_fd = i_freq * d_comps.num_list_points;
    const FLOAT freq = d_freqs[i_freq];
    const FLOAT one_on_lambda = freq / VEL_C;
    const UVW uvw_m = d_uvws[i_bl]; // metres
    const UVW uvw = UVW{
        .u = uvw_m.u * one_on_lambda,
        .v = uvw_m.v * one_on_lambda,
        .w = uvw_m.w * one_on_lambda,
    };

    FLOAT real, imag;
    JONES fd;
    JONES delta_vis = JONES{
        .xx_re = 0.0,
        .xx_im = 0.0,
        .xy_re = 0.0,
        .xy_im = 0.0,
        .yx_re = 0.0,
        .yx_im = 0.0,
        .yy_re = 0.0,
        .yy_im = 0.0,
    };

    for (int i_comp = 0; i_comp < d_comps.num_power_law_points; i_comp++) {
        const LMN lmn = d_comps.power_law_lmns[i_comp];
        SINCOS(uvw.u * lmn.l + uvw.v * lmn.m + uvw.w * lmn.n, &imag, &real);

        extrap_power_law_fd(i_comp, freq, d_comps.power_law_fds, d_comps.power_law_sis, &fd);
        complex_multiply(&fd, real, imag, &delta_vis);
    }

    for (int i_comp = 0; i_comp < d_comps.num_list_points; i_comp++) {
        const LMN lmn = d_comps.list_lmns[i_comp];
        SINCOS(uvw.u * lmn.l + uvw.v * lmn.m + uvw.w * lmn.n, &imag, &real);

        const JONES *fd = &d_comps.list_fds[i_fd + i_comp];
        complex_multiply(fd, real, imag, &delta_vis);
    }

    d_vis[i_vis] += delta_vis;
}

/**
 * Kernel for calculating Gaussian-source-component visibilities.
 */
__global__ void model_gaussians_kernel(const int num_freqs, const int num_vis, const UVW *d_uvws, const FLOAT *d_freqs,
                                       const Gaussians d_comps, JonesF32 *d_vis) {
    const int i_vis = threadIdx.x + (blockDim.x * blockIdx.x);
    if (i_vis >= num_vis)
        return;

    // Get the indices for this thread. `i_vis` indexes over baselines and
    // frequencies, with frequencies moving faster than baselines.
    const int i_bl = i_vis / num_freqs;
    const int i_freq = i_vis % num_freqs;
    const int i_fd = i_freq * d_comps.num_list_gaussians;
    const FLOAT freq = d_freqs[i_freq];
    const FLOAT one_on_lambda = freq / VEL_C;
    const UVW uvw_m = d_uvws[i_bl]; // metres
    const UVW uvw = UVW{
        .u = uvw_m.u * one_on_lambda,
        .v = uvw_m.v * one_on_lambda,
        .w = uvw_m.w * one_on_lambda,
    };

    FLOAT real, imag;
    FLOAT s_pa, c_pa, k_x, k_y, envelope;
    JONES fd;
    JONES delta_vis = JONES{
        .xx_re = 0.0,
        .xx_im = 0.0,
        .xy_re = 0.0,
        .xy_im = 0.0,
        .yx_re = 0.0,
        .yx_im = 0.0,
        .yy_re = 0.0,
        .yy_im = 0.0,
    };

    for (size_t i_comp = 0; i_comp < d_comps.num_power_law_gaussians; i_comp++) {
        const LMN lmn = d_comps.power_law_lmns[i_comp];
        SINCOS(uvw.u * lmn.l + uvw.v * lmn.m + uvw.w * lmn.n, &imag, &real);

        const GaussianParams g_params = d_comps.power_law_gps[i_comp];
        SINCOS(g_params.pa, &s_pa, &c_pa);
        // Temporary variables for clarity.
        k_x = uvw.u * s_pa + uvw.v * c_pa;
        k_y = uvw.u * c_pa - uvw.v * s_pa;
        envelope = EXP(EXP_CONST *
                       ((g_params.maj * g_params.maj) * (k_x * k_x) + (g_params.min * g_params.min) * (k_y * k_y)));

        // Scale by envelope.
        real *= envelope;
        imag *= envelope;

        extrap_power_law_fd(i_comp, freq, d_comps.power_law_fds, d_comps.power_law_sis, &fd);
        complex_multiply(&fd, real, imag, &delta_vis);
    }

    for (size_t i_comp = 0; i_comp < d_comps.num_list_gaussians; i_comp++) {
        const LMN lmn = d_comps.list_lmns[i_comp];
        SINCOS(uvw.u * lmn.l + uvw.v * lmn.m + uvw.w * lmn.n, &imag, &real);

        const GaussianParams g_params = d_comps.list_gps[i_comp];
        SINCOS(g_params.pa, &s_pa, &c_pa);
        // Temporary variables for clarity.
        k_x = uvw.u * s_pa + uvw.v * c_pa;
        k_y = uvw.u * c_pa - uvw.v * s_pa;
        envelope = EXP(EXP_CONST *
                       ((g_params.maj * g_params.maj) * (k_x * k_x) + (g_params.min * g_params.min) * (k_y * k_y)));

        // Scale by envelope.
        real *= envelope;
        imag *= envelope;

        const JONES *fd = &d_comps.list_fds[i_fd + i_comp];
        complex_multiply(fd, real, imag, &delta_vis);
    }

    d_vis[i_vis] += delta_vis;
}

/**
 * Kernel for calculating shapelet-source-component visibilities.
 *
 * `*_shapelet_coeffs` is actually a flattened array-of-arrays. The size of each
 * sub-array is given by an element of `*_num_shapelet_coeffs`.
 */
__global__ void model_shapelets_kernel(const size_t num_freqs, const size_t num_vis, const UVW *d_uvws,
                                       const FLOAT *d_freqs, const Shapelets d_comps,
                                       const FLOAT *d_shapelet_basis_values, const size_t sbf_l, const size_t sbf_n,
                                       const FLOAT sbf_c, const FLOAT sbf_dx, JonesF32 *d_vis) {
    const int i_vis = threadIdx.x + (blockDim.x * blockIdx.x);
    if (i_vis >= num_vis)
        return;

    // Get the indices for this thread. `i_vis` indexes over baselines and
    // frequencies, with frequencies moving faster than baselines.
    const int i_bl = i_vis / num_freqs;
    const int i_freq = i_vis % num_freqs;
    const int i_fd = i_freq * d_comps.num_list_shapelets;
    const FLOAT freq = d_freqs[i_freq];
    const FLOAT one_on_lambda = freq / VEL_C;
    const UVW uvw_m = d_uvws[i_bl]; // metres
    const UVW uvw = UVW{
        .u = uvw_m.u * one_on_lambda,
        .v = uvw_m.v * one_on_lambda,
        .w = uvw_m.w * one_on_lambda,
    };

    FLOAT real, imag, real2, imag2;
    FLOAT s_pa, c_pa;
    JONES fd;
    JONES delta_vis = JONES{
        .xx_re = 0.0,
        .xx_im = 0.0,
        .xy_re = 0.0,
        .xy_im = 0.0,
        .yx_re = 0.0,
        .yx_im = 0.0,
        .yy_re = 0.0,
        .yy_im = 0.0,
    };

    const FLOAT chewie = SQRT_FRAC_PI_SQ_2_LN_2 / sbf_dx;
    const FLOAT I_POWERS_REAL[4] = {1.0, 0.0, -1.0, 0.0};
    const FLOAT I_POWERS_IMAG[4] = {0.0, 1.0, 0.0, -1.0};

    int coeff_depth = 0;
    int i_uv = i_bl * d_comps.num_power_law_shapelets;
    for (int i_comp = 0; i_comp < d_comps.num_power_law_shapelets; i_comp++) {
        const LMN lmn = d_comps.power_law_lmns[i_comp];
        SINCOS(uvw.u * lmn.l + uvw.v * lmn.m + uvw.w * lmn.n, &imag, &real);

        ShapeletUV s_uv = d_comps.power_law_shapelet_uvs[i_uv + i_comp];
        FLOAT shapelet_u = s_uv.u * one_on_lambda;
        FLOAT shapelet_v = s_uv.v * one_on_lambda;
        const GaussianParams g_params = d_comps.power_law_gps[i_comp];
        SINCOS(g_params.pa, &s_pa, &c_pa);

        // Temporary variables for clarity.
        FLOAT x = shapelet_u * s_pa + shapelet_v * c_pa;
        FLOAT y = shapelet_u * c_pa - shapelet_v * s_pa;
        FLOAT const_x = g_params.maj * chewie;
        FLOAT const_y = -g_params.min * chewie;
        FLOAT x_pos = x * const_x + sbf_c;
        FLOAT y_pos = y * const_y + sbf_c;
        int x_pos_int = (int)FLOOR(x_pos);
        int y_pos_int = (int)FLOOR(y_pos);

        FLOAT envelope_re = 0.0;
        FLOAT envelope_im = 0.0;
        for (int i_coeff = 0; i_coeff < d_comps.power_law_num_shapelet_coeffs[i_comp]; i_coeff++) {
            const ShapeletCoeff coeff = d_comps.power_law_shapelet_coeffs[coeff_depth];

            FLOAT x_low = d_shapelet_basis_values[sbf_l * coeff.n1 + x_pos_int];
            FLOAT x_high = d_shapelet_basis_values[sbf_l * coeff.n1 + x_pos_int + 1];
            FLOAT u_value = x_low + (x_high - x_low) * (x_pos - FLOOR(x_pos));

            FLOAT y_low = d_shapelet_basis_values[sbf_l * coeff.n2 + y_pos_int];
            FLOAT y_high = d_shapelet_basis_values[sbf_l * coeff.n2 + y_pos_int + 1];
            FLOAT v_value = y_low + (y_high - y_low) * (y_pos - FLOOR(y_pos));

            // I_POWER_TABLE stuff. The intention is just to find the
            // appropriate power of i, i.e.:
            // index = (n1 + n2) % 4    (so that index is between 0 and 3 inclusive)
            // i^index, e.g.
            // i^0 =  1.0 + 0.0i
            // i^1 =  0.0 + 1.0i
            // i^2 = -1.0 + 0.0i
            // i^3 =  0.0 - 1.0i
            //
            // The following my attempt at doing this efficiently.
            int i_power_index = (coeff.n1 + coeff.n2) % 4;
            FLOAT i_power_re = I_POWERS_REAL[i_power_index];
            FLOAT i_power_im = I_POWERS_IMAG[i_power_index];

            FLOAT rest = coeff.value * u_value * v_value;
            envelope_re += i_power_re * rest;
            envelope_im += i_power_im * rest;

            coeff_depth++;
        }

        // Scale by envelope.
        real2 = real * envelope_re - imag * envelope_im;
        imag2 = real * envelope_im + imag * envelope_re;

        extrap_power_law_fd(i_comp, freq, d_comps.power_law_fds, d_comps.power_law_sis, &fd);
        complex_multiply(&fd, real2, imag2, &delta_vis);
    }

    coeff_depth = 0;
    i_uv = i_bl * d_comps.num_list_shapelets;
    for (int i_comp = 0; i_comp < d_comps.num_list_shapelets; i_comp++) {
        const LMN lmn = d_comps.list_lmns[i_comp];
        SINCOS(uvw.u * lmn.l + uvw.v * lmn.m + uvw.w * lmn.n, &imag, &real);

        ShapeletUV s_uv = d_comps.list_shapelet_uvs[i_uv + i_comp];
        FLOAT shapelet_u = s_uv.u * one_on_lambda;
        FLOAT shapelet_v = s_uv.v * one_on_lambda;
        const GaussianParams g_params = d_comps.list_gps[i_comp];
        SINCOS(g_params.pa, &s_pa, &c_pa);

        // Temporary variables for clarity.
        FLOAT x = shapelet_u * s_pa + shapelet_v * c_pa;
        FLOAT y = shapelet_u * c_pa - shapelet_v * s_pa;
        FLOAT const_x = g_params.maj * chewie;
        FLOAT const_y = -g_params.min * chewie;
        FLOAT x_pos = x * const_x + sbf_c;
        FLOAT y_pos = y * const_y + sbf_c;
        int x_pos_int = (int)FLOOR(x_pos);
        int y_pos_int = (int)FLOOR(y_pos);

        FLOAT envelope_re = 0.0;
        FLOAT envelope_im = 0.0;
        for (int i_coeff = 0; i_coeff < d_comps.list_num_shapelet_coeffs[i_comp]; i_coeff++) {
            const ShapeletCoeff *coeff = &d_comps.list_shapelet_coeffs[coeff_depth];

            FLOAT x_low = d_shapelet_basis_values[sbf_l * coeff->n1 + x_pos_int];
            FLOAT x_high = d_shapelet_basis_values[sbf_l * coeff->n1 + x_pos_int + 1];
            FLOAT u_value = x_low + (x_high - x_low) * (x_pos - FLOOR(x_pos));

            FLOAT y_low = d_shapelet_basis_values[sbf_l * coeff->n2 + y_pos_int];
            FLOAT y_high = d_shapelet_basis_values[sbf_l * coeff->n2 + y_pos_int + 1];
            FLOAT v_value = y_low + (y_high - y_low) * (y_pos - FLOOR(y_pos));

            // I_POWER_TABLE stuff. The intention is just to find the
            // appropriate power of i, i.e.:
            // index = (n1 + n2) % 4    (so that index is between 0 and 3 inclusive)
            // i^index, e.g.
            // i^0 =  1.0 + 0.0i
            // i^1 =  0.0 + 1.0i
            // i^2 = -1.0 + 0.0i
            // i^3 =  0.0 - 1.0i
            //
            // The following my attempt at doing this efficiently.
            int i_power_index = (coeff->n1 + coeff->n2) % 4;
            FLOAT i_power_re = I_POWERS_REAL[i_power_index];
            FLOAT i_power_im = I_POWERS_IMAG[i_power_index];

            FLOAT rest = coeff->value * u_value * v_value;
            envelope_re += i_power_re * rest;
            envelope_im += i_power_im * rest;

            coeff_depth++;
        }

        // Scale by envelope.
        real2 = real * envelope_re - imag * envelope_im;
        imag2 = real * envelope_im + imag * envelope_re;

        const JONES *fd = &d_comps.list_fds[i_fd + i_comp];
        complex_multiply(fd, real2, imag2, &delta_vis);
    }

    d_vis[i_vis] += delta_vis;
}

/**
 * Kernel for calculating point-source-component visibilities attenuated by the
 * FEE beam.
 */
__global__ void model_points_fee_kernel(int num_freqs, int num_vis, const UVW *d_uvws, const FLOAT *d_freqs,
                                        const Points d_comps, const JONES_C *d_beam_jones, const int *d_tile_map,
                                        const int *d_freq_map, int num_fee_freqs,
                                        const int *d_tile_index_to_unflagged_tile_index_map, JonesF32 *d_vis) {
    // First tile idx  == blockIdx.x
    // Second tile idx == blockIdx.y
    // Frequency       == blockDim.z * blockIdx.z + threadIdx.x
    if (blockIdx.y <= blockIdx.x)
        return;

    // Get the indices for this thread.
    int i_freq = blockDim.x * blockIdx.z + threadIdx.x;
    if (i_freq >= num_freqs)
        return;

    int i_bl = gridDim.x * blockIdx.x - (blockIdx.x * blockIdx.x + blockIdx.x) / 2 + blockIdx.y - blockIdx.x - 1;
    // `i_vis` indexes over baselines and frequencies, with frequencies moving
    // faster than baselines.
    int i_vis = i_bl * num_freqs + i_freq;
    if (i_vis >= num_vis)
        return;

    FLOAT freq = d_freqs[i_freq];
    FLOAT one_on_lambda = freq / VEL_C;
    UVW uvw_m = d_uvws[i_bl]; // metres
    UVW uvw = UVW{
        .u = uvw_m.u * one_on_lambda,
        .v = uvw_m.v * one_on_lambda,
        .w = uvw_m.w * one_on_lambda,
    };

    FLOAT real, imag;
    JONES fd;
    JONES delta_vis = JONES{
        .xx_re = 0.0,
        .xx_im = 0.0,
        .xy_re = 0.0,
        .xy_im = 0.0,
        .yx_re = 0.0,
        .yx_im = 0.0,
        .yy_re = 0.0,
        .yy_im = 0.0,
    };

    // Every element of the beam Jones map is actually a pair of ints; access
    // the right element, then disentangle the upper and lower bits to get the
    // indices.
    int i_j1_row = d_tile_map[d_tile_index_to_unflagged_tile_index_map[blockIdx.x]];
    int i_j2_row = d_tile_map[d_tile_index_to_unflagged_tile_index_map[blockIdx.y]];
    int i_col = d_freq_map[i_freq];
    int num_directions = d_comps.num_power_law_points + d_comps.num_list_points;

    for (int i_comp = 0; i_comp < d_comps.num_power_law_points; i_comp++) {
        extrap_power_law_fd(i_comp, freq, d_comps.power_law_fds, d_comps.power_law_sis, &fd);
        JONES_C j1 = d_beam_jones[((num_directions * num_fee_freqs * i_j1_row) + num_directions * i_col) + i_comp];
        JONES_C j2 = d_beam_jones[((num_directions * num_fee_freqs * i_j2_row) + num_directions * i_col) + i_comp];
        apply_beam(&j1, &fd, &j2);

        const LMN lmn = d_comps.power_law_lmns[i_comp];
        SINCOS(uvw.u * lmn.l + uvw.v * lmn.m + uvw.w * lmn.n, &imag, &real);
        complex_multiply(&fd, real, imag, &delta_vis);
    }

    int i_fd = i_freq * d_comps.num_list_points;
    for (int i_comp = 0; i_comp < d_comps.num_list_points; i_comp++) {
        fd = d_comps.list_fds[i_fd + i_comp];
        JONES_C j1 = d_beam_jones[((num_directions * num_fee_freqs * i_j1_row) + num_directions * i_col) + i_comp +
                                  d_comps.num_power_law_points];
        JONES_C j2 = d_beam_jones[((num_directions * num_fee_freqs * i_j2_row) + num_directions * i_col) + i_comp +
                                  d_comps.num_power_law_points];
        apply_beam(&j1, &fd, &j2);

        const LMN lmn = d_comps.list_lmns[i_comp];
        SINCOS(uvw.u * lmn.l + uvw.v * lmn.m + uvw.w * lmn.n, &imag, &real);
        complex_multiply(&fd, real, imag, &delta_vis);
    }

    d_vis[i_vis] += delta_vis;
}

/**
 * Kernel for calculating Gaussian-source-component visibilities.
 */
__global__ void model_gaussians_fee_kernel(const int num_freqs, const int num_vis, const UVW *d_uvws,
                                           const FLOAT *d_freqs, const Gaussians d_comps, JONES_C *d_beam_jones,
                                           const int *d_tile_map, const int *d_freq_map, int num_fee_freqs,
                                           const int *d_tile_index_to_unflagged_tile_index_map, JonesF32 *d_vis) {
    // First tile idx  == blockIdx.x
    // Second tile idx == blockIdx.y
    // Frequency       == blockDim.z * blockIdx.z + threadIdx.x
    if (blockIdx.y <= blockIdx.x)
        return;

    // Get the indices for this thread.
    int i_freq = blockDim.x * blockIdx.z + threadIdx.x;
    if (i_freq >= num_freqs)
        return;

    int i_bl = gridDim.x * blockIdx.x - (blockIdx.x * blockIdx.x + blockIdx.x) / 2 + blockIdx.y - blockIdx.x - 1;
    // `i_vis` indexes over baselines and frequencies, with frequencies moving
    // faster than baselines.
    int i_vis = i_bl * num_freqs + i_freq;
    if (i_vis >= num_vis)
        return;

    const FLOAT freq = d_freqs[i_freq];
    const FLOAT one_on_lambda = freq / VEL_C;
    const UVW uvw_m = d_uvws[i_bl]; // metres
    const UVW uvw = UVW{
        .u = uvw_m.u * one_on_lambda,
        .v = uvw_m.v * one_on_lambda,
        .w = uvw_m.w * one_on_lambda,
    };

    FLOAT real, imag;
    FLOAT s_pa, c_pa, k_x, k_y, envelope;
    JONES fd;
    JONES delta_vis = JONES{
        .xx_re = 0.0,
        .xx_im = 0.0,
        .xy_re = 0.0,
        .xy_im = 0.0,
        .yx_re = 0.0,
        .yx_im = 0.0,
        .yy_re = 0.0,
        .yy_im = 0.0,
    };

    // Every element of the beam Jones map is actually a pair of ints; access
    // the right element, then disentangle the upper and lower bits to get the
    // indices.
    int i_j1_row = d_tile_map[d_tile_index_to_unflagged_tile_index_map[blockIdx.x]];
    int i_j2_row = d_tile_map[d_tile_index_to_unflagged_tile_index_map[blockIdx.y]];
    int i_col = d_freq_map[i_freq];
    int num_directions = d_comps.num_power_law_gaussians + d_comps.num_list_gaussians;

    for (size_t i_comp = 0; i_comp < d_comps.num_power_law_gaussians; i_comp++) {
        extrap_power_law_fd(i_comp, freq, d_comps.power_law_fds, d_comps.power_law_sis, &fd);
        JONES_C j1 = d_beam_jones[((num_directions * num_fee_freqs * i_j1_row) + num_directions * i_col) + i_comp];
        JONES_C j2 = d_beam_jones[((num_directions * num_fee_freqs * i_j2_row) + num_directions * i_col) + i_comp];
        apply_beam(&j1, &fd, &j2);

        const LMN lmn = d_comps.power_law_lmns[i_comp];
        SINCOS(uvw.u * lmn.l + uvw.v * lmn.m + uvw.w * lmn.n, &imag, &real);

        const GaussianParams g_params = d_comps.power_law_gps[i_comp];
        SINCOS(g_params.pa, &s_pa, &c_pa);
        // Temporary variables for clarity.
        k_x = uvw.u * s_pa + uvw.v * c_pa;
        k_y = uvw.u * c_pa - uvw.v * s_pa;
        envelope = EXP(EXP_CONST *
                       ((g_params.maj * g_params.maj) * (k_x * k_x) + (g_params.min * g_params.min) * (k_y * k_y)));

        // Scale by envelope.
        real *= envelope;
        imag *= envelope;

        complex_multiply(&fd, real, imag, &delta_vis);
    }

    int i_fd = i_freq * d_comps.num_list_gaussians;
    for (size_t i_comp = 0; i_comp < d_comps.num_list_gaussians; i_comp++) {
        fd = d_comps.list_fds[i_fd + i_comp];
        JONES_C j1 = d_beam_jones[((num_directions * num_fee_freqs * i_j1_row) + num_directions * i_col) + i_comp +
                                  d_comps.num_power_law_gaussians];
        JONES_C j2 = d_beam_jones[((num_directions * num_fee_freqs * i_j2_row) + num_directions * i_col) + i_comp +
                                  d_comps.num_power_law_gaussians];
        apply_beam(&j1, &fd, &j2);

        const LMN lmn = d_comps.list_lmns[i_comp];
        SINCOS(uvw.u * lmn.l + uvw.v * lmn.m + uvw.w * lmn.n, &imag, &real);

        const GaussianParams g_params = d_comps.list_gps[i_comp];
        SINCOS(g_params.pa, &s_pa, &c_pa);
        // Temporary variables for clarity.
        k_x = uvw.u * s_pa + uvw.v * c_pa;
        k_y = uvw.u * c_pa - uvw.v * s_pa;
        envelope = EXP(EXP_CONST *
                       ((g_params.maj * g_params.maj) * (k_x * k_x) + (g_params.min * g_params.min) * (k_y * k_y)));

        // Scale by envelope.
        real *= envelope;
        imag *= envelope;

        complex_multiply(&fd, real, imag, &delta_vis);
    }

    d_vis[i_vis] += delta_vis;
}

/**
 * Kernel for calculating shapelet-source-component visibilities.
 *
 * `*_shapelet_coeffs` is actually a flattened array-of-arrays. The size of each
 * sub-array is given by an element of `*_num_shapelet_coeffs`.
 */
__global__ void model_shapelets_fee_kernel(const size_t num_freqs, const size_t num_vis, const UVW *d_uvws,
                                           const FLOAT *d_freqs, const Shapelets d_comps,
                                           const FLOAT *d_shapelet_basis_values, const size_t sbf_l, const size_t sbf_n,
                                           const FLOAT sbf_c, const FLOAT sbf_dx, JONES_C *d_beam_jones,
                                           const int *d_tile_map, const int *d_freq_map, int num_fee_freqs,
                                           const int *d_tile_index_to_unflagged_tile_index_map, JonesF32 *d_vis) {
    // First tile idx  == blockIdx.x
    // Second tile idx == blockIdx.y
    // Frequency       == blockDim.z * blockIdx.z + threadIdx.x
    if (blockIdx.y <= blockIdx.x)
        return;

    // Get the indices for this thread.
    int i_freq = blockDim.x * blockIdx.z + threadIdx.x;
    if (i_freq >= num_freqs)
        return;

    int i_bl = gridDim.x * blockIdx.x - (blockIdx.x * blockIdx.x + blockIdx.x) / 2 + blockIdx.y - blockIdx.x - 1;
    // `i_vis` indexes over baselines and frequencies, with frequencies moving
    // faster than baselines.
    int i_vis = i_bl * num_freqs + i_freq;
    if (i_vis >= num_vis)
        return;

    const FLOAT freq = d_freqs[i_freq];
    const FLOAT one_on_lambda = freq / VEL_C;
    const UVW uvw_m = d_uvws[i_bl]; // metres
    const UVW uvw = UVW{
        .u = uvw_m.u * one_on_lambda,
        .v = uvw_m.v * one_on_lambda,
        .w = uvw_m.w * one_on_lambda,
    };

    FLOAT real, imag, real2, imag2;
    FLOAT s_pa, c_pa;
    JONES fd;
    JONES delta_vis = JONES{
        .xx_re = 0.0,
        .xx_im = 0.0,
        .xy_re = 0.0,
        .xy_im = 0.0,
        .yx_re = 0.0,
        .yx_im = 0.0,
        .yy_re = 0.0,
        .yy_im = 0.0,
    };

    // Every element of the beam Jones map is actually a pair of ints; access
    // the right element, then disentangle the upper and lower bits to get the
    // indices.
    int i_j1_row = d_tile_map[d_tile_index_to_unflagged_tile_index_map[blockIdx.x]];
    int i_j2_row = d_tile_map[d_tile_index_to_unflagged_tile_index_map[blockIdx.y]];
    int i_col = d_freq_map[i_freq];
    int num_directions = d_comps.num_power_law_shapelets + d_comps.num_list_shapelets;

    const FLOAT chewie = SQRT_FRAC_PI_SQ_2_LN_2 / sbf_dx;
    const FLOAT I_POWERS_REAL[4] = {1.0, 0.0, -1.0, 0.0};
    const FLOAT I_POWERS_IMAG[4] = {0.0, 1.0, 0.0, -1.0};

    int coeff_depth = 0;
    int i_uv = i_bl * d_comps.num_power_law_shapelets;
    for (int i_comp = 0; i_comp < d_comps.num_power_law_shapelets; i_comp++) {
        const LMN lmn = d_comps.power_law_lmns[i_comp];
        SINCOS(uvw.u * lmn.l + uvw.v * lmn.m + uvw.w * lmn.n, &imag, &real);

        ShapeletUV s_uv = d_comps.power_law_shapelet_uvs[i_uv + i_comp];
        FLOAT shapelet_u = s_uv.u * one_on_lambda;
        FLOAT shapelet_v = s_uv.v * one_on_lambda;
        const GaussianParams g_params = d_comps.power_law_gps[i_comp];
        SINCOS(g_params.pa, &s_pa, &c_pa);

        // Temporary variables for clarity.
        FLOAT x = shapelet_u * s_pa + shapelet_v * c_pa;
        FLOAT y = shapelet_u * c_pa - shapelet_v * s_pa;
        FLOAT const_x = g_params.maj * chewie;
        FLOAT const_y = -g_params.min * chewie;
        FLOAT x_pos = x * const_x + sbf_c;
        FLOAT y_pos = y * const_y + sbf_c;
        int x_pos_int = (int)FLOOR(x_pos);
        int y_pos_int = (int)FLOOR(y_pos);

        FLOAT envelope_re = 0.0;
        FLOAT envelope_im = 0.0;
        for (int i_coeff = 0; i_coeff < d_comps.power_law_num_shapelet_coeffs[i_comp]; i_coeff++) {
            const ShapeletCoeff coeff = d_comps.power_law_shapelet_coeffs[coeff_depth];

            FLOAT x_low = d_shapelet_basis_values[sbf_l * coeff.n1 + x_pos_int];
            FLOAT x_high = d_shapelet_basis_values[sbf_l * coeff.n1 + x_pos_int + 1];
            FLOAT u_value = x_low + (x_high - x_low) * (x_pos - FLOOR(x_pos));

            FLOAT y_low = d_shapelet_basis_values[sbf_l * coeff.n2 + y_pos_int];
            FLOAT y_high = d_shapelet_basis_values[sbf_l * coeff.n2 + y_pos_int + 1];
            FLOAT v_value = y_low + (y_high - y_low) * (y_pos - FLOOR(y_pos));

            // I_POWER_TABLE stuff. The intention is just to find the
            // appropriate power of i, i.e.:
            // index = (n1 + n2) % 4    (so that index is between 0 and 3 inclusive)
            // i^index, e.g.
            // i^0 =  1.0 + 0.0i
            // i^1 =  0.0 + 1.0i
            // i^2 = -1.0 + 0.0i
            // i^3 =  0.0 - 1.0i
            //
            // The following my attempt at doing this efficiently.
            int i_power_index = (coeff.n1 + coeff.n2) % 4;
            FLOAT i_power_re = I_POWERS_REAL[i_power_index];
            FLOAT i_power_im = I_POWERS_IMAG[i_power_index];

            FLOAT rest = coeff.value * u_value * v_value;
            envelope_re += i_power_re * rest;
            envelope_im += i_power_im * rest;

            coeff_depth++;
        }

        // Scale by envelope.
        real2 = real * envelope_re - imag * envelope_im;
        imag2 = real * envelope_im + imag * envelope_re;

        extrap_power_law_fd(i_comp, freq, d_comps.power_law_fds, d_comps.power_law_sis, &fd);
        JONES_C j1 = d_beam_jones[((num_directions * num_fee_freqs * i_j1_row) + num_directions * i_col) + i_comp];
        JONES_C j2 = d_beam_jones[((num_directions * num_fee_freqs * i_j2_row) + num_directions * i_col) + i_comp];
        apply_beam(&j1, &fd, &j2);
        complex_multiply(&fd, real2, imag2, &delta_vis);
    }

    coeff_depth = 0;
    i_uv = i_bl * d_comps.num_list_shapelets;
    const int i_fd = i_freq * d_comps.num_list_shapelets;
    for (int i_comp = 0; i_comp < d_comps.num_list_shapelets; i_comp++) {
        const LMN lmn = d_comps.list_lmns[i_comp];
        SINCOS(uvw.u * lmn.l + uvw.v * lmn.m + uvw.w * lmn.n, &imag, &real);

        ShapeletUV s_uv = d_comps.list_shapelet_uvs[i_uv + i_comp];
        FLOAT shapelet_u = s_uv.u * one_on_lambda;
        FLOAT shapelet_v = s_uv.v * one_on_lambda;
        const GaussianParams g_params = d_comps.list_gps[i_comp];
        SINCOS(g_params.pa, &s_pa, &c_pa);

        // Temporary variables for clarity.
        FLOAT x = shapelet_u * s_pa + shapelet_v * c_pa;
        FLOAT y = shapelet_u * c_pa - shapelet_v * s_pa;
        FLOAT const_x = g_params.maj * chewie;
        FLOAT const_y = -g_params.min * chewie;
        FLOAT x_pos = x * const_x + sbf_c;
        FLOAT y_pos = y * const_y + sbf_c;
        int x_pos_int = (int)FLOOR(x_pos);
        int y_pos_int = (int)FLOOR(y_pos);

        FLOAT envelope_re = 0.0;
        FLOAT envelope_im = 0.0;
        for (int i_coeff = 0; i_coeff < d_comps.list_num_shapelet_coeffs[i_comp]; i_coeff++) {
            const ShapeletCoeff *coeff = &d_comps.list_shapelet_coeffs[coeff_depth];

            FLOAT x_low = d_shapelet_basis_values[sbf_l * coeff->n1 + x_pos_int];
            FLOAT x_high = d_shapelet_basis_values[sbf_l * coeff->n1 + x_pos_int + 1];
            FLOAT u_value = x_low + (x_high - x_low) * (x_pos - FLOOR(x_pos));

            FLOAT y_low = d_shapelet_basis_values[sbf_l * coeff->n2 + y_pos_int];
            FLOAT y_high = d_shapelet_basis_values[sbf_l * coeff->n2 + y_pos_int + 1];
            FLOAT v_value = y_low + (y_high - y_low) * (y_pos - FLOOR(y_pos));

            // I_POWER_TABLE stuff. The intention is just to find the
            // appropriate power of i, i.e.:
            // index = (n1 + n2) % 4    (so that index is between 0 and 3 inclusive)
            // i^index, e.g.
            // i^0 =  1.0 + 0.0i
            // i^1 =  0.0 + 1.0i
            // i^2 = -1.0 + 0.0i
            // i^3 =  0.0 - 1.0i
            //
            // The following my attempt at doing this efficiently.
            int i_power_index = (coeff->n1 + coeff->n2) % 4;
            FLOAT i_power_re = I_POWERS_REAL[i_power_index];
            FLOAT i_power_im = I_POWERS_IMAG[i_power_index];

            FLOAT rest = coeff->value * u_value * v_value;
            envelope_re += i_power_re * rest;
            envelope_im += i_power_im * rest;

            coeff_depth++;
        }

        // Scale by envelope.
        real2 = real * envelope_re - imag * envelope_im;
        imag2 = real * envelope_im + imag * envelope_re;

        fd = d_comps.list_fds[i_fd + i_comp];
        JONES_C j1 = d_beam_jones[((num_directions * num_fee_freqs * i_j1_row) + num_directions * i_col) + i_comp];
        JONES_C j2 = d_beam_jones[((num_directions * num_fee_freqs * i_j2_row) + num_directions * i_col) + i_comp];
        apply_beam(&j1, &fd, &j2);
        complex_multiply(&fd, real2, imag2, &delta_vis);
    }

    d_vis[i_vis] += delta_vis;
}

extern "C" int model_points(const Points *comps, const Addresses *a, const UVW *d_uvws, const void *d_beam_jones) {
    dim3 gridDim, blockDim;
    if (a->num_unique_beam_freqs == 0) {
        // Thread blocks are distributed by visibility (one visibility per
        // frequency and baseline).
        blockDim.x = 512;
        blockDim.y = 1;
        gridDim.x = (int)ceil((double)a->num_vis / (double)blockDim.x);
        gridDim.y = 1;

        model_points_kernel<<<gridDim, blockDim>>>(a->num_freqs, a->num_vis, d_uvws, a->d_freqs, *comps, a->d_vis);
        cudaCheck(cudaPeekAtLastError());
    } else {
        // Thread blocks are distributed by tile indices (x is tile 0, y is tile
        // 1) and frequency (z).
        blockDim.x = 256;
        blockDim.y = 1;
        blockDim.z = 1;
        gridDim.x = a->num_tiles;
        gridDim.y = a->num_tiles;
        gridDim.z = (int)ceil((double)a->num_freqs / (double)blockDim.x);

        model_points_fee_kernel<<<gridDim, blockDim>>>(
            a->num_freqs, a->num_vis, d_uvws, a->d_freqs, *comps, (JONES_C *)d_beam_jones, a->d_tile_map, a->d_freq_map,
            a->num_unique_beam_freqs, a->d_tile_index_to_unflagged_tile_index_map, a->d_vis);
        cudaCheck(cudaPeekAtLastError());
    }

    return EXIT_SUCCESS;
}

extern "C" int model_gaussians(const Gaussians *comps, const Addresses *a, const UVW *d_uvws,
                               const void *d_beam_jones) {
    dim3 gridDim, blockDim;
    if (a->num_unique_beam_freqs == 0) {
        // Thread blocks are distributed by visibility (one visibility per frequency
        // and baseline).
        blockDim.x = 512;
        gridDim.x = (int)ceil((double)a->num_vis / (double)blockDim.x);

        model_gaussians_kernel<<<gridDim, blockDim>>>(a->num_freqs, a->num_vis, d_uvws, a->d_freqs, *comps, a->d_vis);
        cudaCheck(cudaPeekAtLastError());
    } else {
        // Thread blocks are distributed by tile indices (x is tile 0, y is tile
        // 1) and frequency (z).
        blockDim.x = 256;
        blockDim.y = 1;
        blockDim.z = 1;
        gridDim.x = a->num_tiles;
        gridDim.y = a->num_tiles;
        gridDim.z = (int)ceil((double)a->num_freqs / (double)blockDim.x);

        model_gaussians_fee_kernel<<<gridDim, blockDim>>>(
            a->num_freqs, a->num_vis, d_uvws, a->d_freqs, *comps, (JONES_C *)d_beam_jones, a->d_tile_map, a->d_freq_map,
            a->num_unique_beam_freqs, a->d_tile_index_to_unflagged_tile_index_map, a->d_vis);
        cudaCheck(cudaPeekAtLastError());
    }

    return EXIT_SUCCESS;
}

extern "C" int model_shapelets(const Shapelets *comps, const Addresses *a, const UVW *d_uvws,
                               const void *d_beam_jones) {
    dim3 gridDim, blockDim;
    if (a->num_unique_beam_freqs == 0) {
        // Thread blocks are distributed by visibility (one visibility per frequency
        // and baseline).
        dim3 blocks, threads;
        threads.x = 128;
        threads.y = 1;
        blocks.x = (int)ceil((double)a->num_vis / (double)threads.x);
        blocks.y = 1;

        model_shapelets_kernel<<<blocks, threads>>>(a->num_freqs, a->num_vis, d_uvws, a->d_freqs, *comps,
                                                    a->d_shapelet_basis_values, a->sbf_l, a->sbf_n, a->sbf_c, a->sbf_dx,
                                                    a->d_vis);
        cudaCheck(cudaPeekAtLastError());
    } else {
        // Thread blocks are distributed by tile indices (x is tile 0, y is tile
        // 1) and frequency (z).
        blockDim.x = 256;
        blockDim.y = 1;
        blockDim.z = 1;
        gridDim.x = a->num_tiles;
        gridDim.y = a->num_tiles;
        gridDim.z = (int)ceil((double)a->num_freqs / (double)blockDim.x);

        model_shapelets_fee_kernel<<<gridDim, blockDim>>>(
            a->num_freqs, a->num_vis, d_uvws, a->d_freqs, *comps, a->d_shapelet_basis_values, a->sbf_l, a->sbf_n,
            a->sbf_c, a->sbf_dx, (JONES_C *)d_beam_jones, a->d_tile_map, a->d_freq_map, a->num_unique_beam_freqs,
            a->d_tile_index_to_unflagged_tile_index_map, a->d_vis);
        cudaCheck(cudaPeekAtLastError());
    }

    return EXIT_SUCCESS;
}
