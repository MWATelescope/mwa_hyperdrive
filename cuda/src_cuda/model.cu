// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <stdlib.h>

#include <cuda_runtime.h>

#include "common.cuh"
#include "memory.h"
#include "model.h"
#include "types.h"

const double VEL_C = 299792458.0;
const double LN_2 = 0.6931471805599453;
const double FRAC_PI_2 = 1.5707963267948966;
const double SQRT_FRAC_PI_SQ_2_LN_2 = 2.6682231283184983;

#define EXP_CONST -((FRAC_PI_2 * FRAC_PI_2) / LN_2)

inline __host__ __device__ JonesF64 operator*(JonesF64 a, double b) {
    JonesF64 t;
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

inline __host__ __device__ double4 operator+(double4 a, double4 b) {
    double4 t;
    t.x = a.x - b.x;
    t.y = a.y - b.y;
    t.z = a.z - b.z;
    t.w = a.w - b.w;
    return t;
}

inline __host__ __device__ double4 operator-(double4 a, double4 b) {
    double4 t;
    t.x = a.x - b.x;
    t.y = a.y - b.y;
    t.z = a.z - b.z;
    t.w = a.w - b.w;
    return t;
}

inline __host__ __device__ double4 operator*(double4 a, double b) {
    double4 t;
    t.x = a.x * b;
    t.y = a.y * b;
    t.z = a.z * b;
    t.w = a.w * b;
    return t;
}

inline __host__ __device__ void operator*=(double4 &a, double b) {
    a.x *= b;
    a.y *= b;
    a.z *= b;
    a.w *= b;
}

inline __host__ __device__ void operator+=(double4 &a, double4 b) {
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
    a.w += b.w;
}

inline __host__ __device__ void operator-=(double4 &a, double4 b) {
    a.x -= b.x;
    a.y -= b.y;
    a.z -= b.z;
    a.w -= b.w;
}

inline __host__ __device__ float4 operator+(float4 a, float4 b) {
    float4 t;
    t.x = a.x + b.x;
    t.y = a.y + b.y;
    t.z = a.z + b.z;
    t.w = a.w + b.w;
    return t;
}

inline __host__ __device__ float4 operator-(float4 a, float4 b) {
    float4 t;
    t.x = a.x - b.x;
    t.y = a.y - b.y;
    t.z = a.z - b.z;
    t.w = a.w - b.w;
    return t;
}

inline __host__ __device__ float4 operator*(float4 a, float b) {
    float4 t;
    t.x = a.x * b;
    t.y = a.y * b;
    t.z = a.z * b;
    t.w = a.w * b;
    return t;
}

inline __host__ __device__ void operator*=(float4 &a, float b) {
    a.x *= b;
    a.y *= b;
    a.z *= b;
    a.w *= b;
}

inline __host__ __device__ void operator+=(float4 &a, float4 b) {
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
    a.w += b.w;
}

inline __host__ __device__ void operator-=(float4 &a, float4 b) {
    a.x -= b.x;
    a.y -= b.y;
    a.z -= b.z;
    a.w -= b.w;
}

/**
 * Device function for calculating Gaussian-source-component visibilities. `uvw`
 * and `vis` are pointers to the _single_ structs that should be used (i.e.
 * they should not be used as arrays).
 */
__device__ void calc_gaussian_vis(const size_t i_freq, const UVW *uvw, JonesF32 *vis, const double *d_freqs,
                                  const size_t num_gaussians, const LMN *d_gaussian_lmns,
                                  const JonesF64 *d_gaussian_fds, const GaussianParams *d_gaussian_params) {
    double real, imag;
    double s_pa, c_pa;
    const double lambda = VEL_C / d_freqs[i_freq];

    // When indexing `d_point_fds`, `fd_freq` gets the right set of components.
    const size_t fd_freq = i_freq * num_gaussians;

    for (size_t i_comp = 0; i_comp < num_gaussians; i_comp++) {
        const LMN *lmn = &d_gaussian_lmns[i_comp];
        const JonesF64 *fd = &d_gaussian_fds[fd_freq + i_comp];
        const GaussianParams *g_params = &d_gaussian_params[i_comp];

        sincos(g_params->pa, &s_pa, &c_pa);
        // Temporary variables for clarity.
        const double k_x = uvw->u * s_pa + uvw->v * c_pa;
        const double k_y = uvw->u * c_pa - uvw->v * s_pa;
        double envelope = exp(EXP_CONST * ((g_params->maj * g_params->maj) * (k_x * k_x) +
                                           (g_params->min * g_params->min) * (k_y * k_y)));

        // Don't use PI explicitly; CUDA's sincospi does that.
        sincospi(2.0 * (uvw->u * lmn->l + uvw->v * lmn->m + uvw->w * (lmn->n - 1.0)) / lambda, &imag, &real);
        // Scale by envelope.
        real *= envelope;
        imag *= envelope;

        vis->xx_re += real * fd->xx_re - imag * fd->xx_im;
        vis->xx_im += real * fd->xx_im + imag * fd->xx_re;
        vis->xy_re += real * fd->xy_re - imag * fd->xy_im;
        vis->xy_im += real * fd->xy_im + imag * fd->xy_re;
        vis->yx_re += real * fd->yx_re - imag * fd->yx_im;
        vis->yx_im += real * fd->yx_im + imag * fd->yx_re;
        vis->yy_re += real * fd->yy_re - imag * fd->yy_im;
        vis->yy_im += real * fd->yy_im + imag * fd->yy_re;
    }
}

/**
 * Device function for calculating shapelet-source-component visibilities. `uvw`
 * and `vis` are pointers to the _single_ structs that should be used (i.e.
 * they should not be used as arrays). `d_shapelet_coeffs` is actually a
 * flattened array-of-arrays. The size of each sub-array is given by an element
 * of `d_num_shapelet_coeffs`.
 */
__device__ void calc_shapelet_vis(const size_t i_freq, const size_t i_bl, const UVW *uvw, JonesF32 *vis,
                                  const double *d_freqs, const size_t num_shapelets, const LMN *d_shapelet_lmns,
                                  const JonesF64 *d_shapelet_fds, const GaussianParams *d_gaussian_params,
                                  const ShapeletUV *d_shapelet_uvs, const ShapeletCoeff *d_shapelet_coeffs,
                                  const size_t *d_num_shapelet_coeffs, const double *d_shapelet_basis_values,
                                  const size_t sbf_l, const size_t sbf_n, const double sbf_c, const double sbf_dx) {
    double real, imag;
    double s_pa, c_pa;
    size_t coeff_depth = 0;
    const double lambda = VEL_C / d_freqs[i_freq];

    const double I_POWERS_REAL[4] = {1.0, 0.0, -1.0, 0.0};
    const double I_POWERS_IMAG[4] = {0.0, 1.0, 0.0, -1.0};

    // When indexing `d_point_fds`, `fd_freq` gets the right set of components.
    const size_t fd_freq = i_freq * num_shapelets;
    // When indexing `d_shapelet_uvws`, `uvw_bl` gets the right set of
    // components.
    const size_t uvw_bl = i_freq * num_shapelets;

    for (size_t i_comp = 0; i_comp < num_shapelets; i_comp++) {
        const LMN *lmn = &d_shapelet_lmns[i_comp];
        const JonesF64 *fd = &d_shapelet_fds[fd_freq + i_comp];
        const GaussianParams *g_params = &d_gaussian_params[i_comp];
        const ShapeletUV *s_uv = &d_shapelet_uvs[uvw_bl + i_comp];

        sincos(g_params->pa, &s_pa, &c_pa);
        // Temporary variables for clarity.
        double x = s_uv->u * s_pa + s_uv->v * c_pa;
        double y = s_uv->u * c_pa - s_uv->v * s_pa;
        double const_x = g_params->maj * SQRT_FRAC_PI_SQ_2_LN_2 / sbf_dx;
        double const_y = -g_params->min * SQRT_FRAC_PI_SQ_2_LN_2 / sbf_dx;
        double x_pos = x * const_x + sbf_c;
        double y_pos = y * const_y + sbf_c;
        size_t x_pos_int = (size_t)floor(x_pos);
        size_t y_pos_int = (size_t)floor(y_pos);

        double envelope_re = 0.0;
        double envelope_im = 0.0;
        double rest;
        for (size_t i_coeff = 0; i_coeff < d_num_shapelet_coeffs[i_comp]; i_coeff++) {
            const ShapeletCoeff *coeff = &d_shapelet_coeffs[coeff_depth];

            double x_low = d_shapelet_basis_values[sbf_l * coeff->n1 + x_pos_int];
            double x_high = d_shapelet_basis_values[sbf_l * coeff->n1 + x_pos_int + 1];
            double u_value = x_low + (x_high - x_low) * (x_pos - floor(x_pos));

            double y_low = d_shapelet_basis_values[sbf_l * coeff->n2 + y_pos_int];
            double y_high = d_shapelet_basis_values[sbf_l * coeff->n2 + y_pos_int + 1];
            double v_value = y_low + (y_high - y_low) * (y_pos - floor(y_pos));

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
            double i_power_re = I_POWERS_REAL[i_power_index];
            double i_power_im = I_POWERS_IMAG[i_power_index];

            rest = coeff->value * u_value * v_value;
            envelope_re += i_power_re * rest;
            envelope_im += i_power_im * rest;

            coeff_depth++;
        }

        // Don't use PI explicitly; CUDA's sincospi does that.
        sincospi(2.0 * (uvw->u * lmn->l + uvw->v * lmn->m + uvw->w * (lmn->n - 1.0)) / lambda, &imag, &real);
        // Scale by envelope.
        real *= envelope_re;
        imag *= envelope_im;

        vis->xx_re += real * fd->xx_re - imag * fd->xx_im;
        vis->xx_im += real * fd->xx_im + imag * fd->xx_re;
        vis->xy_re += real * fd->xy_re - imag * fd->xy_im;
        vis->xy_im += real * fd->xy_im + imag * fd->xy_re;
        vis->yx_re += real * fd->yx_re - imag * fd->yx_im;
        vis->yx_im += real * fd->yx_im + imag * fd->yx_re;
        vis->yy_re += real * fd->yy_re - imag * fd->yy_im;
        vis->yy_im += real * fd->yy_im + imag * fd->yy_re;
    }
}

/**
 * Correctly multiply a Jones matrix by a complex number and accumulate in
 * another Jones matrix.
 */
inline __device__ void complex_multiply(const JonesF64 *fd, double real, double imag, JonesF64 *delta_vis) {
    double4 fd_re = double4{
        .x = fd->xx_re,
        .y = fd->xy_re,
        .z = fd->yx_re,
        .w = fd->yy_re,
    };
    double4 fd_im = double4{
        .x = fd->xx_im,
        .y = fd->xy_im,
        .z = fd->yx_im,
        .w = fd->yy_im,
    };
    double4 delta_vis_re = double4{
        .x = 0.0,
        .y = 0.0,
        .z = 0.0,
        .w = 0.0,
    };
    double4 delta_vis_im = double4{
        .x = 0.0,
        .y = 0.0,
        .z = 0.0,
        .w = 0.0,
    };

    delta_vis_re += fd_re * real - fd_im * imag;
    delta_vis_im += fd_re * imag + fd_im * real;

    delta_vis->xx_re += delta_vis_re.x;
    delta_vis->xy_re += delta_vis_re.y;
    delta_vis->yx_re += delta_vis_re.z;
    delta_vis->yy_re += delta_vis_re.w;
    delta_vis->xx_im += delta_vis_im.x;
    delta_vis->xy_im += delta_vis_im.y;
    delta_vis->yx_im += delta_vis_im.z;
    delta_vis->yy_im += delta_vis_im.w;
}

inline __device__ void extrap_power_law_fd(const int i_comp, const float freq, const JonesF64 *d_ref_fds,
                                           const double *d_sis, JonesF64 *out) {
    const double flux_ratio = pow(freq / POWER_LAW_FD_REF_FREQ, d_sis[i_comp]);
    *out = d_ref_fds[i_comp] * flux_ratio;
}

/**
 * Kernel for calculating point-source-component visibilities that have power
 * laws.
 */
__global__ void model_power_law_points_kernel(const int num_freqs, const int num_vis, const UVW *d_uvws,
                                              const double *d_freqs, JonesF32 *d_vis, int num_power_law_points,
                                              const LMN *d_lmns, const JonesF64 *d_ref_fds, const double *d_sis) {
    const int i_vis = threadIdx.x + (blockDim.x * blockIdx.x);

    if (i_vis >= num_vis)
        return;

    // Get the indices for this thread. `i_vis` indexes over baselines and
    // frequencies, with frequencies moving faster than baselines.
    const int i_freq = i_vis % num_freqs;
    const int i_bl = i_vis / num_freqs;
    const double freq = d_freqs[i_freq];
    const double one_on_lambda = 1.0 / (VEL_C / freq);
    const UVW uvw_temp = d_uvws[i_bl];
    const UVW uvw = UVW{
        .u = uvw_temp.u * one_on_lambda,
        .v = uvw_temp.v * one_on_lambda,
        .w = uvw_temp.w * one_on_lambda,
    };

    double real, imag;
    JonesF64 fd;
    JonesF64 delta_vis = JonesF64{
        .xx_re = 0.0,
        .xx_im = 0.0,
        .xy_re = 0.0,
        .xy_im = 0.0,
        .yx_re = 0.0,
        .yx_im = 0.0,
        .yy_re = 0.0,
        .yy_im = 0.0,
    };

    for (int i_comp = 0; i_comp < num_power_law_points; i_comp++) {
        const LMN lmn = d_lmns[i_comp];
        sincos(uvw.u * lmn.l + uvw.v * lmn.m + uvw.w * lmn.n, &imag, &real);

        extrap_power_law_fd(i_comp, freq, d_ref_fds, d_sis, &fd);
        complex_multiply(&fd, real, imag, &delta_vis);
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
 * Kernel for calculating point-source-component visibilities that have flux
 * densities described by lists.
 */
__global__ void model_list_points_kernel(const int num_freqs, const int num_vis, const UVW *d_uvws,
                                         const double *d_freqs, JonesF32 *d_vis, int num_list_points, const LMN *d_lmns,
                                         const JonesF64 *d_fds) {
    const int i_vis = threadIdx.x + (blockDim.x * blockIdx.x);

    if (i_vis >= num_vis)
        return;

    // Get the indices for this thread. `i_vis` indexes over baselines and
    // frequencies, with frequencies moving faster than baselines.
    const int i_freq = i_vis % num_freqs;
    const int i_bl = i_vis / num_freqs;
    const int i_fd = i_freq * num_list_points;
    const double freq = d_freqs[i_freq];
    const double one_on_lambda = 1.0 / (VEL_C / freq);
    const UVW uvw_temp = d_uvws[i_bl];
    const UVW uvw = UVW{
        .u = uvw_temp.u * one_on_lambda,
        .v = uvw_temp.v * one_on_lambda,
        .w = uvw_temp.w * one_on_lambda,
    };

    double real, imag;
    JonesF64 delta_vis = JonesF64{
        .xx_re = 0.0,
        .xx_im = 0.0,
        .xy_re = 0.0,
        .xy_im = 0.0,
        .yx_re = 0.0,
        .yx_im = 0.0,
        .yy_re = 0.0,
        .yy_im = 0.0,
    };

    for (int i_comp = 0; i_comp < num_list_points; i_comp++) {
        const LMN lmn = d_lmns[i_comp];
        sincos(uvw.u * lmn.l + uvw.v * lmn.m + uvw.w * lmn.n, &imag, &real);

        const JonesF64 fd = d_fds[i_fd + i_comp];
        complex_multiply(&fd, real, imag, &delta_vis);
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
 * Kernel for calculating Gaussian-source-component visibilities that have power
 * laws.
 */
__global__ void model_power_law_gaussians_kernel(const int num_freqs, const int num_vis, const UVW *d_uvws,
                                                 const double *d_freqs, JonesF32 *d_vis,
                                                 const size_t num_power_law_gaussians, const LMN *d_lmns,
                                                 const JonesF64 *d_ref_fds, const double *d_sis,
                                                 const GaussianParams *d_gaussian_params) {
    const int i_vis = threadIdx.x + (blockDim.x * blockIdx.x);

    if (i_vis >= num_vis)
        return;

    // Get the indices for this thread. `i_vis` indexes over baselines and
    // frequencies, with frequencies moving faster than baselines.
    const int i_freq = i_vis % num_freqs;
    const int i_bl = i_vis / num_freqs;
    const double freq = d_freqs[i_freq];
    const double one_on_lambda = 1.0 / (VEL_C / freq);
    const UVW uvw_temp = d_uvws[i_bl];
    const UVW uvw = UVW{
        .u = uvw_temp.u * one_on_lambda,
        .v = uvw_temp.v * one_on_lambda,
        .w = uvw_temp.w * one_on_lambda,
    };

    double real, imag;
    double s_pa, c_pa, k_x, k_y, envelope;
    JonesF64 fd;
    JonesF64 delta_vis = JonesF64{
        .xx_re = 0.0,
        .xx_im = 0.0,
        .xy_re = 0.0,
        .xy_im = 0.0,
        .yx_re = 0.0,
        .yx_im = 0.0,
        .yy_re = 0.0,
        .yy_im = 0.0,
    };

    for (size_t i_comp = 0; i_comp < num_power_law_gaussians; i_comp++) {
        const LMN lmn = d_lmns[i_comp];
        sincospi(uvw.u * lmn.l + uvw.v * lmn.m + uvw.w * lmn.n, &imag, &real);

        const GaussianParams g_params = d_gaussian_params[i_comp];
        sincos(g_params.pa, &s_pa, &c_pa);
        // Temporary variables for clarity.
        k_x = uvw.u * s_pa + uvw.v * c_pa;
        k_y = uvw.u * c_pa - uvw.v * s_pa;
        envelope = exp(EXP_CONST *
                       ((g_params.maj * g_params.maj) * (k_x * k_x) + (g_params.min * g_params.min) * (k_y * k_y)));

        // Scale by envelope.
        real *= envelope;
        imag *= envelope;

        extrap_power_law_fd(i_comp, freq, d_ref_fds, d_sis, &fd);
        complex_multiply(&fd, real, imag, &delta_vis);
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
 * Kernel for calculating Gaussian-source-component visibilities that have power
 * laws.
 */
__global__ void model_list_gaussians_kernel(const int num_freqs, const int num_vis, const UVW *d_uvws,
                                            const double *d_freqs, JonesF32 *d_vis, const size_t num_list_gaussians,
                                            const LMN *d_lmns, const JonesF64 *d_fds,
                                            const GaussianParams *d_gaussian_params) {
    const int i_vis = threadIdx.x + (blockDim.x * blockIdx.x);

    if (i_vis >= num_vis)
        return;

    // Get the indices for this thread. `i_vis` indexes over baselines and
    // frequencies, with frequencies moving faster than baselines.
    const int i_freq = i_vis % num_freqs;
    const int i_bl = i_vis / num_freqs;
    const int i_fd = i_freq * num_list_gaussians;
    const double freq = d_freqs[i_freq];
    const double one_on_lambda = 1.0 / (VEL_C / freq);
    const UVW uvw_temp = d_uvws[i_bl];
    const UVW uvw = UVW{
        .u = uvw_temp.u * one_on_lambda,
        .v = uvw_temp.v * one_on_lambda,
        .w = uvw_temp.w * one_on_lambda,
    };

    double real, imag;
    double s_pa, c_pa, k_x, k_y, envelope;
    JonesF64 delta_vis = JonesF64{
        .xx_re = 0.0,
        .xx_im = 0.0,
        .xy_re = 0.0,
        .xy_im = 0.0,
        .yx_re = 0.0,
        .yx_im = 0.0,
        .yy_re = 0.0,
        .yy_im = 0.0,
    };

    for (size_t i_comp = 0; i_comp < num_list_gaussians; i_comp++) {
        const LMN lmn = d_lmns[i_comp];
        sincospi(uvw.u * lmn.l + uvw.v * lmn.m + uvw.w * lmn.n, &imag, &real);

        const GaussianParams g_params = d_gaussian_params[i_comp];
        sincos(g_params.pa, &s_pa, &c_pa);
        // Temporary variables for clarity.
        k_x = uvw.u * s_pa + uvw.v * c_pa;
        k_y = uvw.u * c_pa - uvw.v * s_pa;
        envelope = exp(EXP_CONST *
                       ((g_params.maj * g_params.maj) * (k_x * k_x) + (g_params.min * g_params.min) * (k_y * k_y)));

        // Scale by envelope.
        real *= envelope;
        imag *= envelope;

        const JonesF64 fd = d_fds[i_fd + i_comp];
        complex_multiply(&fd, real, imag, &delta_vis);
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
 * `d_shapelet_coeffs` is actually a flattened array-of-arrays. The size of each
 * sub-array is given by an element of `d_num_shapelet_coeffs`.
 */
__global__ void model_shapelets_kernel(const size_t num_freqs, const size_t num_vis, const UVW *d_uvws,
                                       const double *d_freqs, JonesF32 *d_vis, const size_t num_shapelets,
                                       const LMN *d_shapelet_lmns, const JonesF64 *d_shapelet_fds,
                                       const GaussianParams *d_gaussian_params, const ShapeletUV *d_shapelet_uvs,
                                       const ShapeletCoeff *d_shapelet_coeffs, const size_t *d_num_shapelet_coeffs,
                                       const double *d_shapelet_basis_values, const size_t sbf_l, const size_t sbf_n,
                                       const double sbf_c, const double sbf_dx) {
    const int i_vis = threadIdx.x + (blockDim.x * blockIdx.x);

    if (i_vis >= num_vis)
        return;

    // Get the pointers for this thread.
    const UVW *uvw = &d_uvws[i_vis / num_freqs];
    JonesF32 *vis = &d_vis[i_vis];
    // `i_vis` indexes over baselines and frequencies, and there are `num_freqs`
    // frequencies per baseline.
    const size_t i_freq = i_vis % num_freqs;
    const size_t i_bl = i_vis / num_freqs;

    calc_shapelet_vis(i_freq, i_bl, uvw, vis, d_freqs, num_shapelets, d_shapelet_lmns, d_shapelet_fds,
                      d_gaussian_params, d_shapelet_uvs, d_shapelet_coeffs, d_num_shapelet_coeffs,
                      d_shapelet_basis_values, sbf_l, sbf_n, sbf_c, sbf_dx);
}

extern "C" int model_power_law_points(const size_t num_power_law_points, const LMN *lmns, const JonesF64 *ref_fds,
                                      const double *sis, const Addresses *a) {
    LMN *d_lmns = NULL;
    size_t size_lmns = num_power_law_points * sizeof(LMN);
    cudaSoftCheck(cudaMalloc(&d_lmns, size_lmns));
    cudaSoftCheck(cudaMemcpy(d_lmns, lmns, size_lmns, cudaMemcpyHostToDevice));

    JonesF64 *d_ref_fds = NULL;
    size_t size_fds = num_power_law_points * sizeof(JonesF64);
    cudaSoftCheck(cudaMalloc(&d_ref_fds, size_fds));
    cudaSoftCheck(cudaMemcpy(d_ref_fds, ref_fds, size_fds, cudaMemcpyHostToDevice));

    double *d_sis = NULL;
    size_t size_sis = num_power_law_points * sizeof(double);
    cudaSoftCheck(cudaMalloc(&d_sis, size_sis));
    cudaSoftCheck(cudaMemcpy(d_sis, sis, size_sis, cudaMemcpyHostToDevice));

    // Thread blocks are distributed by visibility (one visibility per frequency
    // and baseline).
    dim3 blocks, threads;
    // threads.x = 128;
    threads.x = 512;
    threads.y = 1;
    blocks.x = (int)ceil((double)a->num_vis / (double)threads.x);
    blocks.y = 1;

    model_power_law_points_kernel<<<blocks, threads>>>(a->num_freqs, a->num_vis, a->d_uvws, a->d_freqs, a->d_vis,
                                                       num_power_law_points, d_lmns, d_ref_fds, d_sis);
    cudaCheck(cudaPeekAtLastError());

    cudaSoftCheck(cudaFree(d_lmns));
    cudaSoftCheck(cudaFree(d_ref_fds));
    cudaSoftCheck(cudaFree(d_sis));

    return EXIT_SUCCESS;
}

extern "C" int model_list_points(const size_t num_list_points, const LMN *lmns, const JonesF64 *fds,
                                 const Addresses *a) {
    LMN *d_lmns = NULL;
    size_t size_lmns = num_list_points * sizeof(LMN);
    cudaSoftCheck(cudaMalloc(&d_lmns, size_lmns));
    cudaSoftCheck(cudaMemcpy(d_lmns, lmns, size_lmns, cudaMemcpyHostToDevice));

    JonesF64 *d_fds = NULL;
    size_t size_fds = a->num_freqs * num_list_points * sizeof(JonesF64);
    cudaSoftCheck(cudaMalloc(&d_fds, size_fds));
    cudaSoftCheck(cudaMemcpy(d_fds, fds, size_fds, cudaMemcpyHostToDevice));

    // Thread blocks are distributed by visibility (one visibility per frequency
    // and baseline).
    dim3 blocks, threads;
    // threads.x = 128;
    threads.x = 512;
    threads.y = 1;
    blocks.x = (int)ceil((double)a->num_vis / (double)threads.x);
    blocks.y = 1;

    model_list_points_kernel<<<blocks, threads>>>(a->num_freqs, a->num_vis, a->d_uvws, a->d_freqs, a->d_vis,
                                                  num_list_points, d_lmns, d_fds);
    cudaCheck(cudaPeekAtLastError());

    cudaSoftCheck(cudaFree(d_lmns));
    cudaSoftCheck(cudaFree(d_fds));

    return EXIT_SUCCESS;
}

extern "C" int model_power_law_gaussians(const size_t num_power_law_gaussians, const LMN *lmns, const JonesF64 *ref_fds,
                                         const double *sis, const GaussianParams *gaussian_params, const Addresses *a) {
    LMN *d_lmns = NULL;
    size_t size_lmns = num_power_law_gaussians * sizeof(LMN);
    cudaSoftCheck(cudaMalloc(&d_lmns, size_lmns));
    cudaSoftCheck(cudaMemcpy(d_lmns, lmns, size_lmns, cudaMemcpyHostToDevice));

    JonesF64 *d_ref_fds = NULL;
    size_t size_fds = num_power_law_gaussians * sizeof(JonesF64);
    cudaSoftCheck(cudaMalloc(&d_ref_fds, size_fds));
    cudaSoftCheck(cudaMemcpy(d_ref_fds, ref_fds, size_fds, cudaMemcpyHostToDevice));

    double *d_sis = NULL;
    size_t size_sis = num_power_law_gaussians * sizeof(double);
    cudaSoftCheck(cudaMalloc(&d_sis, size_sis));
    cudaSoftCheck(cudaMemcpy(d_sis, sis, size_sis, cudaMemcpyHostToDevice));

    GaussianParams *d_gaussian_params = NULL;
    size_t size_gaussian_params = num_power_law_gaussians * sizeof(GaussianParams);
    cudaSoftCheck(cudaMalloc(&d_gaussian_params, size_gaussian_params));
    cudaSoftCheck(cudaMemcpy(d_gaussian_params, gaussian_params, size_gaussian_params, cudaMemcpyHostToDevice));

    // Thread blocks are distributed by visibility (one visibility per frequency
    // and baseline).
    dim3 blocks, threads;
    threads.x = 512;
    threads.y = 1;
    blocks.x = (int)ceil((double)a->num_vis / (double)threads.x);
    blocks.y = 1;

    model_power_law_gaussians_kernel<<<blocks, threads>>>(a->num_freqs, a->num_vis, a->d_uvws, a->d_freqs, a->d_vis,
                                                          num_power_law_gaussians, d_lmns, d_ref_fds, d_sis,
                                                          d_gaussian_params);
    cudaCheck(cudaPeekAtLastError());

    cudaSoftCheck(cudaFree(d_lmns));
    cudaSoftCheck(cudaFree(d_ref_fds));
    cudaSoftCheck(cudaFree(d_sis));
    cudaSoftCheck(cudaFree(d_gaussian_params));

    return EXIT_SUCCESS;
}

extern "C" int model_list_gaussians(const size_t num_list_gaussians, const LMN *lmns, const JonesF64 *fds,
                                    const GaussianParams *gaussian_params, const Addresses *a) {
    LMN *d_lmns = NULL;
    size_t size_lmns = num_list_gaussians * sizeof(LMN);
    cudaSoftCheck(cudaMalloc(&d_lmns, size_lmns));
    cudaSoftCheck(cudaMemcpy(d_lmns, lmns, size_lmns, cudaMemcpyHostToDevice));

    JonesF64 *d_fds = NULL;
    size_t size_fds = a->num_freqs * num_list_gaussians * sizeof(JonesF64);
    cudaSoftCheck(cudaMalloc(&d_fds, size_fds));
    cudaSoftCheck(cudaMemcpy(d_fds, fds, size_fds, cudaMemcpyHostToDevice));

    GaussianParams *d_gaussian_params = NULL;
    size_t size_gaussian_params = num_list_gaussians * sizeof(GaussianParams);
    cudaSoftCheck(cudaMalloc(&d_gaussian_params, size_gaussian_params));
    cudaSoftCheck(cudaMemcpy(d_gaussian_params, gaussian_params, size_gaussian_params, cudaMemcpyHostToDevice));

    // Thread blocks are distributed by visibility (one visibility per frequency
    // and baseline).
    dim3 blocks, threads;
    threads.x = 512;
    threads.y = 1;
    blocks.x = (int)ceil((double)a->num_vis / (double)threads.x);
    blocks.y = 1;

    model_list_gaussians_kernel<<<blocks, threads>>>(a->num_freqs, a->num_vis, a->d_uvws, a->d_freqs, a->d_vis,
                                                     num_list_gaussians, d_lmns, d_fds, d_gaussian_params);
    cudaDeviceSynchronize();
    cudaCheck(cudaPeekAtLastError());

    cudaSoftCheck(cudaFree(d_lmns));
    cudaSoftCheck(cudaFree(d_fds));
    cudaSoftCheck(cudaFree(d_gaussian_params));

    return EXIT_SUCCESS;
}

extern "C" int model_shapelets(const size_t num_shapelets, const LMN *shapelet_lmns, const JonesF64 *shapelet_fds,
                               const GaussianParams *gaussian_params, const ShapeletUV *shapelet_uvs,
                               const ShapeletCoeff *shapelet_coeffs, const size_t *num_shapelet_coeffs,
                               const Addresses a) {
    LMN *d_lmns = NULL;
    size_t size_lmns = num_shapelets * sizeof(LMN);
    cudaSoftCheck(cudaMalloc(&d_lmns, size_lmns));
    cudaSoftCheck(cudaMemcpy(d_lmns, shapelet_lmns, size_lmns, cudaMemcpyHostToDevice));

    JonesF64 *d_fds = NULL;
    size_t size_fds = num_shapelets * a.num_freqs * sizeof(JonesF64);
    cudaSoftCheck(cudaMalloc(&d_fds, size_fds));
    cudaSoftCheck(cudaMemcpy(d_fds, shapelet_fds, size_fds, cudaMemcpyHostToDevice));

    GaussianParams *d_gaussian_params = NULL;
    size_t size_gaussian_params = num_shapelets * sizeof(GaussianParams);
    cudaSoftCheck(cudaMalloc(&d_gaussian_params, size_gaussian_params));
    cudaSoftCheck(cudaMemcpy(d_gaussian_params, gaussian_params, size_gaussian_params, cudaMemcpyHostToDevice));

    ShapeletUV *d_shapelet_uvs = NULL;
    size_t size_shapelet_uvs = num_shapelets * sizeof(ShapeletUV);
    cudaSoftCheck(cudaMalloc(&d_shapelet_uvs, size_shapelet_uvs));
    cudaSoftCheck(cudaMemcpy(d_shapelet_uvs, shapelet_uvs, size_shapelet_uvs, cudaMemcpyHostToDevice));

    size_t total_num_shapelet_coeffs = 0;
    for (size_t i = 0; i < num_shapelets; i++) {
        total_num_shapelet_coeffs += num_shapelet_coeffs[i];
    }

    ShapeletCoeff *d_shapelet_coeffs = NULL;
    size_t size_shapelet_coeffs = total_num_shapelet_coeffs * sizeof(ShapeletCoeff);
    cudaSoftCheck(cudaMalloc(&d_shapelet_coeffs, size_shapelet_coeffs));
    cudaSoftCheck(cudaMemcpy(d_shapelet_coeffs, shapelet_coeffs, size_shapelet_coeffs, cudaMemcpyHostToDevice));

    size_t *d_num_shapelet_coeffs = NULL;
    size_t size_num_shapelet_coeffs = num_shapelets * sizeof(size_t);
    cudaSoftCheck(cudaMalloc(&d_num_shapelet_coeffs, size_num_shapelet_coeffs));
    cudaSoftCheck(
        cudaMemcpy(d_num_shapelet_coeffs, num_shapelet_coeffs, size_num_shapelet_coeffs, cudaMemcpyHostToDevice));

    // Thread blocks are distributed by visibility (one visibility per frequency
    // and baseline).
    dim3 blocks, threads;
    threads.x = 32;
    threads.y = 1;
    blocks.x = (int)ceil((double)a.num_vis / (double)threads.x);
    blocks.y = 1;

    model_shapelets_kernel<<<blocks, threads>>>(a.num_freqs, a.num_vis, a.d_uvws, a.d_freqs, a.d_vis, num_shapelets,
                                                d_lmns, d_fds, d_gaussian_params, d_shapelet_uvs, d_shapelet_coeffs,
                                                d_num_shapelet_coeffs, a.d_shapelet_basis_values, a.sbf_l, a.sbf_n,
                                                a.sbf_c, a.sbf_dx);
    cudaCheck(cudaPeekAtLastError());

    cudaSoftCheck(cudaFree(d_lmns));
    cudaSoftCheck(cudaFree(d_fds));
    cudaSoftCheck(cudaFree(d_gaussian_params));

    return EXIT_SUCCESS;
}

extern "C" int
model_timestep(const size_t num_baselines, const size_t num_freqs, const size_t num_power_law_points,
               const size_t num_list_points, const size_t num_power_law_gaussians, const size_t num_list_gaussians,
               const size_t num_shapelets, const UVW *uvws, const double *freqs, const LMN *point_power_law_lmns,
               const JonesF64 *point_power_law_ref_fds, const double *point_power_law_sis, const LMN *point_list_lmns,
               const JonesF64 *point_list_fds, const LMN *gaussian_power_law_lmns,
               const JonesF64 *gaussian_power_law_ref_fds, const double *gaussian_power_law_sis,
               const GaussianParams *gaussian_power_law_gaussian_params, const LMN *gaussian_list_lmns,
               const JonesF64 *gaussian_list_fds, const GaussianParams *gaussian_list_gaussian_params,
               const LMN *shapelet_lmns, const JonesF64 *shapelet_fds, const GaussianParams *shapelet_gaussian_params,
               const ShapeletUV *shapelet_uvs, const ShapeletCoeff *shapelet_coeffs, const size_t *num_shapelet_coeffs,
               const double *shapelet_basis_values, const size_t sbf_l, const size_t sbf_n, const double sbf_c,
               const double sbf_dx, JonesF32 *vis) {
    int status = 0;

    Addresses a =
        init_model(num_baselines, num_freqs, sbf_l, sbf_n, sbf_c, sbf_dx, uvws, freqs, shapelet_basis_values, vis);

    if (num_power_law_points > 0) {
        status = model_power_law_points(num_power_law_points, point_power_law_lmns, point_power_law_ref_fds,
                                        point_power_law_sis, &a);
        if (status != EXIT_SUCCESS) {
            return status;
        }
    } // if (num_power_law_points > 0)
    if (num_list_points > 0) {
        status = model_list_points(num_list_points, point_list_lmns, point_list_fds, &a);
        if (status != EXIT_SUCCESS) {
            return status;
        }
    } // if (num_list_points > 0)

    if (num_power_law_gaussians > 0) {
        status = model_power_law_gaussians(num_power_law_gaussians, gaussian_power_law_lmns, gaussian_power_law_ref_fds,
                                           gaussian_power_law_sis, gaussian_power_law_gaussian_params, &a);
        if (status != EXIT_SUCCESS) {
            return status;
        }
    } // if (num_power_law_gaussians > 0)
    if (num_list_gaussians > 0) {
        status = model_list_gaussians(num_list_gaussians, gaussian_list_lmns, gaussian_list_fds,
                                      gaussian_list_gaussian_params, &a);
        if (status != EXIT_SUCCESS) {
            return status;
        }
    } // if (num_list_gaussians > 0)

    if (num_shapelets > 0) {
        status = model_shapelets(num_shapelets, shapelet_lmns, shapelet_fds, shapelet_gaussian_params, shapelet_uvs,
                                 shapelet_coeffs, num_shapelet_coeffs, a);
        if (status != EXIT_SUCCESS) {
            return status;
        }
    } // if (num_shapelets > 0)

    copy_vis(&a);
    destroy(&a);

    return EXIT_SUCCESS;
}
