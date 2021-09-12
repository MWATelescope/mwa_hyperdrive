// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <stdlib.h>

#include <cuda_runtime.h>

#include "common.cuh"
#include "memory.h"
#include "model.h"
#include "types.h"

// If SINGLE is enabled, use single-precision floats everywhere. Otherwise
// default to double-precision.
#ifdef SINGLE
#define FLOAT4 float4
#define SINCOS sincosf
#define EXP    expf
#define POW    powf
#define FLOOR  floorf
#else
#define FLOAT4 double4
#define SINCOS sincos
#define EXP    exp
#define POW    pow
#define FLOOR  floor
#endif

const FLOAT VEL_C = 299792458.0;
const FLOAT LN_2 = 0.6931471805599453;
const FLOAT FRAC_PI_2 = 1.5707963267948966;
const FLOAT SQRT_FRAC_PI_SQ_2_LN_2 = 2.6682231283184983;
const FLOAT EXP_CONST = -((FRAC_PI_2 * FRAC_PI_2) / LN_2);

inline __host__ __device__ JONES operator*(JONES a, FLOAT b) {
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

inline __host__ __device__ FLOAT4 operator+(FLOAT4 a, FLOAT4 b) {
    FLOAT4 t;
    t.x = a.x - b.x;
    t.y = a.y - b.y;
    t.z = a.z - b.z;
    t.w = a.w - b.w;
    return t;
}

inline __host__ __device__ FLOAT4 operator-(FLOAT4 a, FLOAT4 b) {
    FLOAT4 t;
    t.x = a.x - b.x;
    t.y = a.y - b.y;
    t.z = a.z - b.z;
    t.w = a.w - b.w;
    return t;
}

inline __host__ __device__ FLOAT4 operator*(FLOAT4 a, FLOAT b) {
    FLOAT4 t;
    t.x = a.x * b;
    t.y = a.y * b;
    t.z = a.z * b;
    t.w = a.w * b;
    return t;
}

inline __host__ __device__ void operator+=(FLOAT4 &a, FLOAT4 b) {
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
    a.w += b.w;
}

/**
 * Correctly multiply a Jones matrix by a complex number and accumulate in
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

inline __device__ void extrap_power_law_fd(const int i_comp, const FLOAT freq, const JONES *d_ref_fds,
                                           const FLOAT *d_sis, JONES *out) {
    const FLOAT flux_ratio = POW(freq / POWER_LAW_FD_REF_FREQ, d_sis[i_comp]);
    *out = d_ref_fds[i_comp] * flux_ratio;
}

/**
 * Kernel for calculating point-source-component visibilities.
 */
__global__ void model_points_kernel(const int num_freqs, const int num_vis, const UVW *d_uvws, const FLOAT *d_freqs,
                                    const Points *d_comps, JonesF32 *d_vis) {
    const int i_vis = threadIdx.x + (blockDim.x * blockIdx.x);
    if (i_vis >= num_vis)
        return;

    // Get the indices for this thread. `i_vis` indexes over baselines and
    // frequencies, with frequencies moving faster than baselines.
    const int i_bl = i_vis / num_freqs;
    const int i_freq = i_vis % num_freqs;
    const int i_fd = i_freq * d_comps->num_list_points;
    const FLOAT freq = d_freqs[i_freq];
    const FLOAT one_on_lambda = 1.0 / (VEL_C / freq);
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

    for (int i_comp = 0; i_comp < d_comps->num_power_law_points; i_comp++) {
        const LMN lmn = d_comps->power_law_lmns[i_comp];
        SINCOS(uvw.u * lmn.l + uvw.v * lmn.m + uvw.w * lmn.n, &imag, &real);

        extrap_power_law_fd(i_comp, freq, d_comps->power_law_fds, d_comps->power_law_sis, &fd);
        complex_multiply(&fd, real, imag, &delta_vis);
    }

    for (int i_comp = 0; i_comp < d_comps->num_list_points; i_comp++) {
        const LMN lmn = d_comps->list_lmns[i_comp];
        SINCOS(uvw.u * lmn.l + uvw.v * lmn.m + uvw.w * lmn.n, &imag, &real);

        const JONES *fd = &d_comps->list_fds[i_fd + i_comp];
        complex_multiply(fd, real, imag, &delta_vis);
    }

    d_vis[i_vis].xx_re += delta_vis.xx_re;
    d_vis[i_vis].xx_im += delta_vis.xx_im;
    d_vis[i_vis].xy_re += delta_vis.xy_re;
    d_vis[i_vis].xy_im += delta_vis.xy_im;
    d_vis[i_vis].yx_re += delta_vis.yx_re;
    d_vis[i_vis].yx_im += delta_vis.yx_im;
    d_vis[i_vis].yy_re += delta_vis.yy_re;
    d_vis[i_vis].yy_im += delta_vis.yy_im;
}

/**
 * Kernel for calculating Gaussian-source-component visibilities.
 */
__global__ void model_gaussians_kernel(const int num_freqs, const int num_vis, const UVW *d_uvws, const FLOAT *d_freqs,
                                       const Gaussians *d_comps, JonesF32 *d_vis) {
    const int i_vis = threadIdx.x + (blockDim.x * blockIdx.x);
    if (i_vis >= num_vis)
        return;

    // Get the indices for this thread. `i_vis` indexes over baselines and
    // frequencies, with frequencies moving faster than baselines.
    const int i_bl = i_vis / num_freqs;
    const int i_freq = i_vis % num_freqs;
    const int i_fd = i_freq * d_comps->num_list_gaussians;
    const FLOAT freq = d_freqs[i_freq];
    const FLOAT one_on_lambda = 1.0 / (VEL_C / freq);
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

    for (size_t i_comp = 0; i_comp < d_comps->num_power_law_gaussians; i_comp++) {
        const LMN lmn = d_comps->power_law_lmns[i_comp];
        SINCOS(uvw.u * lmn.l + uvw.v * lmn.m + uvw.w * lmn.n, &imag, &real);

        const GaussianParams g_params = d_comps->power_law_gps[i_comp];
        SINCOS(g_params.pa, &s_pa, &c_pa);
        // Temporary variables for clarity.
        k_x = uvw.u * s_pa + uvw.v * c_pa;
        k_y = uvw.u * c_pa - uvw.v * s_pa;
        envelope = EXP(EXP_CONST *
                       ((g_params.maj * g_params.maj) * (k_x * k_x) + (g_params.min * g_params.min) * (k_y * k_y)));

        // Scale by envelope.
        real *= envelope;
        imag *= envelope;

        extrap_power_law_fd(i_comp, freq, d_comps->power_law_fds, d_comps->power_law_sis, &fd);
        complex_multiply(&fd, real, imag, &delta_vis);
    }

    for (size_t i_comp = 0; i_comp < d_comps->num_list_gaussians; i_comp++) {
        const LMN lmn = d_comps->list_lmns[i_comp];
        SINCOS(uvw.u * lmn.l + uvw.v * lmn.m + uvw.w * lmn.n, &imag, &real);

        const GaussianParams g_params = d_comps->list_gps[i_comp];
        SINCOS(g_params.pa, &s_pa, &c_pa);
        // Temporary variables for clarity.
        k_x = uvw.u * s_pa + uvw.v * c_pa;
        k_y = uvw.u * c_pa - uvw.v * s_pa;
        envelope = EXP(EXP_CONST *
                       ((g_params.maj * g_params.maj) * (k_x * k_x) + (g_params.min * g_params.min) * (k_y * k_y)));

        // Scale by envelope.
        real *= envelope;
        imag *= envelope;

        const JONES *fd = &d_comps->list_fds[i_fd + i_comp];
        complex_multiply(fd, real, imag, &delta_vis);
    }

    d_vis[i_vis].xx_re += delta_vis.xx_re;
    d_vis[i_vis].xx_im += delta_vis.xx_im;
    d_vis[i_vis].xy_re += delta_vis.xy_re;
    d_vis[i_vis].xy_im += delta_vis.xy_im;
    d_vis[i_vis].yx_re += delta_vis.yx_re;
    d_vis[i_vis].yx_im += delta_vis.yx_im;
    d_vis[i_vis].yy_re += delta_vis.yy_re;
    d_vis[i_vis].yy_im += delta_vis.yy_im;
}

/**
 * Kernel for calculating shapelet-source-component visibilities.
 *
 * `*_shapelet_coeffs` is actually a flattened array-of-arrays. The size of each
 * sub-array is given by an element of `*_num_shapelet_coeffs`.
 */
__global__ void model_shapelets_kernel(const size_t num_freqs, const size_t num_vis, const UVW *d_uvws,
                                       const FLOAT *d_freqs, const Shapelets *d_comps,
                                       const FLOAT *d_shapelet_basis_values, const size_t sbf_l, const size_t sbf_n,
                                       const FLOAT sbf_c, const FLOAT sbf_dx, JonesF32 *d_vis) {
    const int i_vis = threadIdx.x + (blockDim.x * blockIdx.x);
    if (i_vis >= num_vis)
        return;

    // Get the indices for this thread. `i_vis` indexes over baselines and
    // frequencies, with frequencies moving faster than baselines.
    const int i_bl = i_vis / num_freqs;
    const int i_freq = i_vis % num_freqs;
    const int i_fd = i_freq * d_comps->num_list_shapelets;
    const FLOAT freq = d_freqs[i_freq];
    const FLOAT one_on_lambda = 1.0 / (VEL_C / freq);
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
    int i_uv = i_bl * d_comps->num_power_law_shapelets;
    for (int i_comp = 0; i_comp < d_comps->num_power_law_shapelets; i_comp++) {
        const LMN lmn = d_comps->power_law_lmns[i_comp];
        SINCOS(uvw.u * lmn.l + uvw.v * lmn.m + uvw.w * lmn.n, &imag, &real);

        ShapeletUV s_uv = d_comps->power_law_shapelet_uvs[i_uv + i_comp];
        FLOAT shapelet_u = s_uv.u * one_on_lambda;
        FLOAT shapelet_v = s_uv.v * one_on_lambda;
        const GaussianParams g_params = d_comps->power_law_gps[i_comp];
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
        for (int i_coeff = 0; i_coeff < d_comps->power_law_num_shapelet_coeffs[i_comp]; i_coeff++) {
            const ShapeletCoeff coeff = d_comps->power_law_shapelet_coeffs[coeff_depth];

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

        extrap_power_law_fd(i_comp, freq, d_comps->power_law_fds, d_comps->power_law_sis, &fd);
        complex_multiply(&fd, real2, imag2, &delta_vis);
    }

    coeff_depth = 0;
    i_uv = i_bl * d_comps->num_list_shapelets;
    for (int i_comp = 0; i_comp < d_comps->num_list_shapelets; i_comp++) {
        const LMN lmn = d_comps->list_lmns[i_comp];
        SINCOS(uvw.u * lmn.l + uvw.v * lmn.m + uvw.w * lmn.n, &imag, &real);

        ShapeletUV s_uv = d_comps->list_shapelet_uvs[i_uv + i_comp];
        FLOAT shapelet_u = s_uv.u * one_on_lambda;
        FLOAT shapelet_v = s_uv.v * one_on_lambda;
        const GaussianParams g_params = d_comps->list_gps[i_comp];
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
        for (int i_coeff = 0; i_coeff < d_comps->list_num_shapelet_coeffs[i_comp]; i_coeff++) {
            const ShapeletCoeff *coeff = &d_comps->list_shapelet_coeffs[coeff_depth];

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

        const JONES *fd = &d_comps->list_fds[i_fd + i_comp];
        complex_multiply(fd, real2, imag2, &delta_vis);
    }

    d_vis[i_vis].xx_re += delta_vis.xx_re;
    d_vis[i_vis].xx_im += delta_vis.xx_im;
    d_vis[i_vis].xy_re += delta_vis.xy_re;
    d_vis[i_vis].xy_im += delta_vis.xy_im;
    d_vis[i_vis].yx_re += delta_vis.yx_re;
    d_vis[i_vis].yx_im += delta_vis.yx_im;
    d_vis[i_vis].yy_re += delta_vis.yy_re;
    d_vis[i_vis].yy_im += delta_vis.yy_im;
}

extern "C" int model_points(const Points *comps, const Addresses *a) {
    LMN *d_pl_lmns = NULL;
    size_t size_lmns = comps->num_power_law_points * sizeof(LMN);
    cudaSoftCheck(cudaMalloc(&d_pl_lmns, size_lmns));
    cudaSoftCheck(cudaMemcpy(d_pl_lmns, comps->power_law_lmns, size_lmns, cudaMemcpyHostToDevice));

    JONES *d_pl_fds = NULL;
    size_t size_fds = comps->num_power_law_points * sizeof(JONES);
    cudaSoftCheck(cudaMalloc(&d_pl_fds, size_fds));
    cudaSoftCheck(cudaMemcpy(d_pl_fds, comps->power_law_fds, size_fds, cudaMemcpyHostToDevice));

    FLOAT *d_pl_sis = NULL;
    size_t size_sis = comps->num_power_law_points * sizeof(FLOAT);
    cudaSoftCheck(cudaMalloc(&d_pl_sis, size_sis));
    cudaSoftCheck(cudaMemcpy(d_pl_sis, comps->power_law_sis, size_sis, cudaMemcpyHostToDevice));

    LMN *d_list_lmns = NULL;
    size_lmns = comps->num_list_points * sizeof(LMN);
    cudaSoftCheck(cudaMalloc(&d_list_lmns, size_lmns));
    cudaSoftCheck(cudaMemcpy(d_list_lmns, comps->list_lmns, size_lmns, cudaMemcpyHostToDevice));

    JONES *d_list_fds = NULL;
    size_fds = a->num_freqs * comps->num_list_points * sizeof(JONES);
    cudaSoftCheck(cudaMalloc(&d_list_fds, size_fds));
    cudaSoftCheck(cudaMemcpy(d_list_fds, comps->list_fds, size_fds, cudaMemcpyHostToDevice));

    Points *d_comps = NULL;
    cudaSoftCheck(cudaMalloc(&d_comps, sizeof(Points)));
    Points c = Points{
        .num_power_law_points = comps->num_power_law_points,
        .power_law_radecs = NULL, // TODO: For beam code.
        .power_law_lmns = d_pl_lmns,
        .power_law_fds = d_pl_fds,
        .power_law_sis = d_pl_sis,

        .num_list_points = comps->num_list_points,
        .list_radecs = NULL,
        .list_lmns = d_list_lmns,
        .list_fds = d_list_fds,
    };
    cudaSoftCheck(cudaMemcpy(d_comps, &c, sizeof(Points), cudaMemcpyHostToDevice));

    // Thread blocks are distributed by visibility (one visibility per frequency
    // and baseline).
    dim3 blocks, threads;
    threads.x = 512;
    threads.y = 1;
    blocks.x = (int)ceil((double)a->num_vis / (double)threads.x);
    blocks.y = 1;

    model_points_kernel<<<blocks, threads>>>(a->num_freqs, a->num_vis, a->d_uvws, a->d_freqs, d_comps, a->d_vis);
    cudaCheck(cudaPeekAtLastError());

    cudaSoftCheck(cudaFree(d_pl_lmns));
    cudaSoftCheck(cudaFree(d_pl_fds));
    cudaSoftCheck(cudaFree(d_pl_sis));
    cudaSoftCheck(cudaFree(d_list_lmns));
    cudaSoftCheck(cudaFree(d_list_fds));
    cudaSoftCheck(cudaFree(d_comps));

    return EXIT_SUCCESS;
}

extern "C" int model_gaussians(const Gaussians *comps, const Addresses *a) {
    LMN *d_pl_lmns = NULL;
    size_t size_lmns = comps->num_power_law_gaussians * sizeof(LMN);
    cudaSoftCheck(cudaMalloc(&d_pl_lmns, size_lmns));
    cudaSoftCheck(cudaMemcpy(d_pl_lmns, comps->power_law_lmns, size_lmns, cudaMemcpyHostToDevice));

    JONES *d_pl_fds = NULL;
    size_t size_fds = comps->num_power_law_gaussians * sizeof(JONES);
    cudaSoftCheck(cudaMalloc(&d_pl_fds, size_fds));
    cudaSoftCheck(cudaMemcpy(d_pl_fds, comps->power_law_fds, size_fds, cudaMemcpyHostToDevice));

    FLOAT *d_pl_sis = NULL;
    size_t size_sis = comps->num_power_law_gaussians * sizeof(FLOAT);
    cudaSoftCheck(cudaMalloc(&d_pl_sis, size_sis));
    cudaSoftCheck(cudaMemcpy(d_pl_sis, comps->power_law_sis, size_sis, cudaMemcpyHostToDevice));

    GaussianParams *d_pl_gps = NULL;
    size_t size_gps = comps->num_power_law_gaussians * sizeof(GaussianParams);
    cudaSoftCheck(cudaMalloc(&d_pl_gps, size_gps));
    cudaSoftCheck(cudaMemcpy(d_pl_gps, comps->power_law_gps, size_gps, cudaMemcpyHostToDevice));

    LMN *d_list_lmns = NULL;
    size_lmns = comps->num_list_gaussians * sizeof(LMN);
    cudaSoftCheck(cudaMalloc(&d_list_lmns, size_lmns));
    cudaSoftCheck(cudaMemcpy(d_list_lmns, comps->list_lmns, size_lmns, cudaMemcpyHostToDevice));

    JONES *d_list_fds = NULL;
    size_fds = a->num_freqs * comps->num_list_gaussians * sizeof(JONES);
    cudaSoftCheck(cudaMalloc(&d_list_fds, size_fds));
    cudaSoftCheck(cudaMemcpy(d_list_fds, comps->list_fds, size_fds, cudaMemcpyHostToDevice));

    GaussianParams *d_list_gps = NULL;
    size_gps = comps->num_list_gaussians * sizeof(GaussianParams);
    cudaSoftCheck(cudaMalloc(&d_list_gps, size_gps));
    cudaSoftCheck(cudaMemcpy(d_list_gps, comps->list_gps, size_gps, cudaMemcpyHostToDevice));

    Gaussians *d_comps = NULL;
    cudaSoftCheck(cudaMalloc(&d_comps, sizeof(Gaussians)));
    Gaussians c = Gaussians{
        .num_power_law_gaussians = comps->num_power_law_gaussians,
        .power_law_radecs = NULL, // TODO: For beam code.
        .power_law_lmns = d_pl_lmns,
        .power_law_fds = d_pl_fds,
        .power_law_sis = d_pl_sis,
        .power_law_gps = d_pl_gps,

        .num_list_gaussians = comps->num_list_gaussians,
        .list_radecs = NULL,
        .list_lmns = d_list_lmns,
        .list_fds = d_list_fds,
        .list_gps = d_list_gps,
    };
    cudaSoftCheck(cudaMemcpy(d_comps, &c, sizeof(Gaussians), cudaMemcpyHostToDevice));

    // Thread blocks are distributed by visibility (one visibility per frequency
    // and baseline).
    dim3 blocks, threads;
    threads.x = 512;
    threads.y = 1;
    blocks.x = (int)ceil((double)a->num_vis / (double)threads.x);
    blocks.y = 1;

    model_gaussians_kernel<<<blocks, threads>>>(a->num_freqs, a->num_vis, a->d_uvws, a->d_freqs, d_comps, a->d_vis);
    cudaCheck(cudaPeekAtLastError());

    cudaSoftCheck(cudaFree(d_pl_lmns));
    cudaSoftCheck(cudaFree(d_pl_fds));
    cudaSoftCheck(cudaFree(d_pl_sis));
    cudaSoftCheck(cudaFree(d_pl_gps));
    cudaSoftCheck(cudaFree(d_list_lmns));
    cudaSoftCheck(cudaFree(d_list_fds));
    cudaSoftCheck(cudaFree(d_list_gps));
    cudaSoftCheck(cudaFree(d_comps));

    return EXIT_SUCCESS;
}

extern "C" int model_shapelets(const Shapelets *comps, const Addresses *a) {
    LMN *d_pl_lmns = NULL;
    size_t size_lmns = comps->num_power_law_shapelets * sizeof(LMN);
    cudaSoftCheck(cudaMalloc(&d_pl_lmns, size_lmns));
    cudaSoftCheck(cudaMemcpy(d_pl_lmns, comps->power_law_lmns, size_lmns, cudaMemcpyHostToDevice));

    JONES *d_pl_fds = NULL;
    size_t size_fds = comps->num_power_law_shapelets * sizeof(JONES);
    cudaSoftCheck(cudaMalloc(&d_pl_fds, size_fds));
    cudaSoftCheck(cudaMemcpy(d_pl_fds, comps->power_law_fds, size_fds, cudaMemcpyHostToDevice));

    FLOAT *d_pl_sis = NULL;
    size_t size_sis = comps->num_power_law_shapelets * sizeof(FLOAT);
    cudaSoftCheck(cudaMalloc(&d_pl_sis, size_sis));
    cudaSoftCheck(cudaMemcpy(d_pl_sis, comps->power_law_sis, size_sis, cudaMemcpyHostToDevice));

    GaussianParams *d_pl_gps = NULL;
    size_t size_gps = comps->num_power_law_shapelets * sizeof(GaussianParams);
    cudaSoftCheck(cudaMalloc(&d_pl_gps, size_gps));
    cudaSoftCheck(cudaMemcpy(d_pl_gps, comps->power_law_gps, size_gps, cudaMemcpyHostToDevice));

    ShapeletUV *d_pl_shapelet_uvs = NULL;
    size_t num_bls = a->num_vis / a->num_freqs;
    size_t size_shapelet_uvs = num_bls * comps->num_power_law_shapelets * sizeof(ShapeletUV);
    cudaSoftCheck(cudaMalloc(&d_pl_shapelet_uvs, size_shapelet_uvs));
    cudaSoftCheck(
        cudaMemcpy(d_pl_shapelet_uvs, comps->power_law_shapelet_uvs, size_shapelet_uvs, cudaMemcpyHostToDevice));

    size_t total_num_shapelet_coeffs = 0;
    for (size_t i = 0; i < comps->num_power_law_shapelets; i++) {
        total_num_shapelet_coeffs += comps->power_law_num_shapelet_coeffs[i];
    }

    ShapeletCoeff *d_pl_shapelet_coeffs = NULL;
    size_t size_shapelet_coeffs = total_num_shapelet_coeffs * sizeof(ShapeletCoeff);
    cudaSoftCheck(cudaMalloc(&d_pl_shapelet_coeffs, size_shapelet_coeffs));
    cudaSoftCheck(cudaMemcpy(d_pl_shapelet_coeffs, comps->power_law_shapelet_coeffs, size_shapelet_coeffs,
                             cudaMemcpyHostToDevice));

    size_t *d_pl_num_shapelet_coeffs = NULL;
    size_t size_num_shapelet_coeffs = comps->num_power_law_shapelets * sizeof(size_t);
    cudaSoftCheck(cudaMalloc(&d_pl_num_shapelet_coeffs, size_num_shapelet_coeffs));
    cudaSoftCheck(cudaMemcpy(d_pl_num_shapelet_coeffs, comps->power_law_num_shapelet_coeffs, size_num_shapelet_coeffs,
                             cudaMemcpyHostToDevice));

    LMN *d_list_lmns = NULL;
    size_lmns = comps->num_list_shapelets * sizeof(LMN);
    cudaSoftCheck(cudaMalloc(&d_list_lmns, size_lmns));
    cudaSoftCheck(cudaMemcpy(d_list_lmns, comps->list_lmns, size_lmns, cudaMemcpyHostToDevice));

    JONES *d_list_fds = NULL;
    size_fds = a->num_freqs * comps->num_list_shapelets * sizeof(JONES);
    cudaSoftCheck(cudaMalloc(&d_list_fds, size_fds));
    cudaSoftCheck(cudaMemcpy(d_list_fds, comps->list_fds, size_fds, cudaMemcpyHostToDevice));

    GaussianParams *d_list_gps = NULL;
    size_gps = comps->num_list_shapelets * sizeof(GaussianParams);
    cudaSoftCheck(cudaMalloc(&d_list_gps, size_gps));
    cudaSoftCheck(cudaMemcpy(d_list_gps, comps->list_gps, size_gps, cudaMemcpyHostToDevice));

    ShapeletUV *d_list_shapelet_uvs = NULL;
    size_shapelet_uvs = num_bls * comps->num_list_shapelets * sizeof(ShapeletUV);
    cudaSoftCheck(cudaMalloc(&d_list_shapelet_uvs, size_shapelet_uvs));
    cudaSoftCheck(cudaMemcpy(d_list_shapelet_uvs, comps->list_shapelet_uvs, size_shapelet_uvs, cudaMemcpyHostToDevice));

    total_num_shapelet_coeffs = 0;
    for (size_t i = 0; i < comps->num_list_shapelets; i++) {
        total_num_shapelet_coeffs += comps->list_num_shapelet_coeffs[i];
    }

    ShapeletCoeff *d_list_shapelet_coeffs = NULL;
    size_shapelet_coeffs = total_num_shapelet_coeffs * sizeof(ShapeletCoeff);
    cudaSoftCheck(cudaMalloc(&d_list_shapelet_coeffs, size_shapelet_coeffs));
    cudaSoftCheck(
        cudaMemcpy(d_list_shapelet_coeffs, comps->list_shapelet_coeffs, size_shapelet_coeffs, cudaMemcpyHostToDevice));

    size_t *d_list_num_shapelet_coeffs = NULL;
    size_num_shapelet_coeffs = comps->num_list_shapelets * sizeof(size_t);
    cudaSoftCheck(cudaMalloc(&d_list_num_shapelet_coeffs, size_num_shapelet_coeffs));
    cudaSoftCheck(cudaMemcpy(d_list_num_shapelet_coeffs, comps->list_num_shapelet_coeffs, size_num_shapelet_coeffs,
                             cudaMemcpyHostToDevice));

    Shapelets *d_comps = NULL;
    cudaSoftCheck(cudaMalloc(&d_comps, sizeof(Shapelets)));
    Shapelets c = Shapelets{
        .num_power_law_shapelets = comps->num_power_law_shapelets,
        .power_law_radecs = NULL, // TODO: For beam code.
        .power_law_lmns = d_pl_lmns,
        .power_law_fds = d_pl_fds,
        .power_law_sis = d_pl_sis,
        .power_law_gps = d_pl_gps,
        .power_law_shapelet_uvs = d_pl_shapelet_uvs,
        .power_law_shapelet_coeffs = d_pl_shapelet_coeffs,
        .power_law_num_shapelet_coeffs = d_pl_num_shapelet_coeffs,

        .num_list_shapelets = comps->num_list_shapelets,
        .list_radecs = NULL,
        .list_lmns = d_list_lmns,
        .list_fds = d_list_fds,
        .list_gps = d_list_gps,
        .list_shapelet_uvs = d_list_shapelet_uvs,
        .list_shapelet_coeffs = d_list_shapelet_coeffs,
        .list_num_shapelet_coeffs = d_list_num_shapelet_coeffs,
    };
    cudaSoftCheck(cudaMemcpy(d_comps, &c, sizeof(Shapelets), cudaMemcpyHostToDevice));

    // Thread blocks are distributed by visibility (one visibility per frequency
    // and baseline).
    dim3 blocks, threads;
    threads.x = 128;
    threads.y = 1;
    blocks.x = (int)ceil((double)a->num_vis / (double)threads.x);
    blocks.y = 1;

    model_shapelets_kernel<<<blocks, threads>>>(a->num_freqs, a->num_vis, a->d_uvws, a->d_freqs, d_comps,
                                                a->d_shapelet_basis_values, a->sbf_l, a->sbf_n, a->sbf_c, a->sbf_dx,
                                                a->d_vis);
    cudaCheck(cudaPeekAtLastError());

    cudaSoftCheck(cudaFree(d_pl_lmns));
    cudaSoftCheck(cudaFree(d_pl_fds));
    cudaSoftCheck(cudaFree(d_pl_sis));
    cudaSoftCheck(cudaFree(d_pl_gps));
    cudaSoftCheck(cudaFree(d_pl_shapelet_uvs));
    cudaSoftCheck(cudaFree(d_pl_shapelet_coeffs));
    cudaSoftCheck(cudaFree(d_pl_num_shapelet_coeffs));
    cudaSoftCheck(cudaFree(d_list_lmns));
    cudaSoftCheck(cudaFree(d_list_fds));
    cudaSoftCheck(cudaFree(d_list_gps));
    cudaSoftCheck(cudaFree(d_list_shapelet_uvs));
    cudaSoftCheck(cudaFree(d_list_shapelet_coeffs));
    cudaSoftCheck(cudaFree(d_list_num_shapelet_coeffs));
    cudaSoftCheck(cudaFree(d_comps));

    return EXIT_SUCCESS;
}

extern "C" int model_timestep(const size_t num_baselines, const size_t num_freqs, const UVW *uvws, const FLOAT *freqs,
                              const Points *points, const Gaussians *gaussians, const Shapelets *shapelets,
                              const FLOAT *shapelet_basis_values, const size_t sbf_l, const size_t sbf_n,
                              const FLOAT sbf_c, const FLOAT sbf_dx, JonesF32 *vis) {
    int status = 0;

    Addresses a =
        init_model(num_baselines, num_freqs, sbf_l, sbf_n, sbf_c, sbf_dx, uvws, freqs, shapelet_basis_values, vis);

    if (points != NULL && (points->num_power_law_points > 0 || points->num_list_points > 0)) {
        status = model_points(points, &a);
        if (status != EXIT_SUCCESS) {
            return status;
        }
    }

    if (gaussians != NULL && (gaussians->num_power_law_gaussians > 0 || gaussians->num_list_gaussians > 0)) {
        status = model_gaussians(gaussians, &a);
        if (status != EXIT_SUCCESS) {
            return status;
        }
    }

    if (shapelets != NULL && (shapelets->num_power_law_shapelets > 0 || shapelets->num_list_shapelets > 0)) {
        status = model_shapelets(shapelets, &a);
        if (status != EXIT_SUCCESS) {
            return status;
        }
    }

    copy_vis(&a);
    destroy(&a);

    return EXIT_SUCCESS;
}
