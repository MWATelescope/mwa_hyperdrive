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

inline __host__ __device__ double4 operator+(double4 a, double4 b) {
    double4 t;
    t.x = a.x + b.x;
    t.y = a.y + b.y;
    t.z = a.z + b.z;
    t.w = a.w + b.w;
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
 * Kernel for calculating point-source-component visibilities.
 */
// __global__ void model_points_kernel(const size_t num_freqs, const size_t num_vis, const UVW *d_uvws,
//                                     const double *d_freqs, JonesF32 *d_vis, int num_points,
//                                     const LMN *d_point_lmns, const JonesF64 *d_point_fds) {
// __global__ void model_points_kernel(const size_t num_freqs, const size_t num_vis, const UVW *d_uvws,
//                                     const double *d_freqs, JonesF32 *d_vis, int num_points, const double4
//                                     *d_point_lmns, const double4 *d_point_fds_re, const double4 *d_point_fds_im) {
// __global__ void model_points_kernel2(const size_t num_freqs, const size_t num_vis, const UVW *d_uvws,
//                                      const double *d_freqs, JonesF32 *d_vis, int num_points, const double *d_ls,
//                                      const double *d_ms, const double *d_ns, const double2 *d_fds_xx,
//                                      const double2 *d_fds_xy, const double2 *d_fds_yx, const double2 *d_fds_yy) {
//     const int i_vis = threadIdx.x + (blockDim.x * blockIdx.x);

//     if (i_vis >= num_vis)
//         return;

//     // Get the pointers for this thread.
//     const UVW uvw = d_uvws[i_vis / num_freqs];
//     // `i_vis` indexes over baselines and frequencies, and there are `num_freqs`
//     // frequencies per baseline.
//     const size_t i_freq = i_vis % num_freqs;

//     const double lambda = 2.0 / (VEL_C / d_freqs[i_freq]);

//     // When indexing `d_point_fds`, `fd_freq` gets the right set of components.
//     const size_t fd_freq = i_freq * num_points;

//     double4 lmn, fd_re, fd_im, inter;
//     double4 delta_vis_re = double4{
//         .x = 0.0,
//         .y = 0.0,
//         .z = 0.0,
//         .w = 0.0,
//     };
//     double4 delta_vis_im = double4{
//         .x = 0.0,
//         .y = 0.0,
//         .z = 0.0,
//         .w = 0.0,
//     };
//     double real, imag;

//     // // Shared memory witchcraft.
//     // // Every point source component has an LMN (3 doubles or 24 bytes) and 4
//     // // instrumental flux densities (8 doubles or 64 bytes; 88 bytes total). To
//     // // be friendly to all CUDA architectures, we use 48kB of shared memory to
//     // // cache things (558 components).
//     // const int magic = 558;
//     // __shared__ double3 lmns[magic];
//     // __shared__ double4 fds_re[magic], fds_im[magic];
//     // int i_comp = 0;
//     // int i_comp_offset = 0;
//     // for (int chunk = 0; chunk < num_points / magic; chunk++) {
//     //     for (i_comp = 0; i_comp < magic; i_comp++) {
//     //         const LMN temp = d_point_lmns[i_comp + i_comp_offset];
//     //         lmns[i_comp] = double3{
//     //             .x = temp.l,
//     //             .y = temp.m,
//     //             .z = temp.n,
//     //         };
//     //         fds_re[i_comp] = d_point_fds_re[fd_freq + i_comp + i_comp_offset];
//     //         fds_im[i_comp] = d_point_fds_im[fd_freq + i_comp + i_comp_offset];
//     //     }
//     //     i_comp_offset += magic;
//     //     __syncwarp();
//     //     for (i_comp = 0; i_comp < magic; i_comp++) {
//     //         lmn = lmns[i_comp];
//     //         fd_re = fds_re[i_comp];
//     //         fd_im = fds_im[i_comp];

//     //         // Don't use PI explicitly; CUDA's sincospi does that.
//     //         sincospi(2.0 * (uvw.u * lmn.x + uvw.v * lmn.y + uvw.w * (lmn.z - 1.0)) / lambda, &imag, &real);

//     //         inter = fd_re * real - fd_im * imag;
//     //         delta_vis_re += inter;

//     //         inter = fd_re * imag + fd_im * real;
//     //         delta_vis_im += inter;
//     //     }
//     // }

//     double l, m, n;
//     double2 fd_xx, fd_xy, fd_yx, fd_yy;
//     for (int i_comp = 0; i_comp < num_points; i_comp++) {
//         // lmn = d_point_lmns[i_comp];
//         // fd_re = d_point_fds_re[i_comp + fd_freq];
//         // fd_im = d_point_fds_im[i_comp + fd_freq];
//         // fd_re = d_point_fds_re[i_comp];
//         // fd_im = d_point_fds_im[i_comp];

//         l = d_ls[i_comp];
//         m = d_ms[i_comp];
//         n = d_ns[i_comp];

//         fd_xx = d_fds_xx[i_comp];
//         fd_xy = d_fds_xy[i_comp];
//         fd_yx = d_fds_yx[i_comp];
//         fd_yy = d_fds_yy[i_comp];

//         fd_re.x = fd_xx.x;
//         fd_re.y = fd_xy.x;
//         fd_re.z = fd_yx.x;
//         fd_re.w = fd_yy.x;
//         fd_im.x = fd_xx.y;
//         fd_im.y = fd_xy.y;
//         fd_im.z = fd_yx.y;
//         fd_im.w = fd_yy.y;

//         // Don't use PI explicitly; CUDA's sincospi does that.
//         // sincospi(2.0 * (uvw.u * lmn.x + uvw.v * lmn.y + uvw.w * (lmn.z - 1.0)) / lambda, &imag, &real);
//         // sincospi(2.0 * (uvw.u * l + uvw.v * m + uvw.w * (n - 1.0)) / lambda, &imag, &real);
//         // sincospi(lambda * (uvw.u * l + uvw.v * m + uvw.w * (n - 1.0)), &imag, &real);

//         real = 0.0;
//         imag = lambda * (uvw.u * l + uvw.v * m + uvw.w * (n - 1.0));

//         inter = fd_re * real - fd_im * imag;
//         delta_vis_re += inter;

//         inter = fd_re * imag + fd_im * real;
//         delta_vis_im += inter;
//     }

//     d_vis[i_vis].xx_re += (float)delta_vis_re.x;
//     d_vis[i_vis].xy_re += (float)delta_vis_re.y;
//     d_vis[i_vis].yx_re += (float)delta_vis_re.z;
//     d_vis[i_vis].yy_re += (float)delta_vis_re.w;
//     d_vis[i_vis].xx_im += (float)delta_vis_im.x;
//     d_vis[i_vis].xy_im += (float)delta_vis_im.y;
//     d_vis[i_vis].yx_im += (float)delta_vis_im.z;
//     d_vis[i_vis].yy_im += (float)delta_vis_im.w;
// }

// // This effort is pretty good, but I think still unacceptably slow. The standard
// // for loop is close to the right answer, but it isn't right, and it's hideously
// // slow. The shared memory stuff does speed things up, but the results are
// // wrong.
// __global__ void model_points_kernel(const int num_baselines, const int num_freqs, const int num_vis, const UVW
// *d_uvws,
//                                     const double *d_freqs, JonesF32 *d_vis, int num_points, const float4
//                                     *d_point_lmns, const float4 *d_point_fds_re, const float4 *d_point_fds_im) {
//     const int i_vis = threadIdx.x + (blockDim.x * blockIdx.x);

//     if (i_vis >= num_vis)
//         return;

//     // Get the indices for this thread. `i_vis` indexes over baselines and
//     // frequencies, with frequencies moving faster than baselines.
//     const int i_freq = i_vis % num_freqs;
//     const int i_bl = i_vis / num_freqs;
//     const double two_on_lambda = 2.0 / (VEL_C / d_freqs[i_freq]);
//     UVW uvw_temp = d_uvws[i_bl];
//     const float4 uvw = float4{
//         .x = (float)(uvw_temp.u * two_on_lambda),
//         .y = (float)(uvw_temp.v * two_on_lambda),
//         .z = (float)(uvw_temp.w * two_on_lambda),
//         .w = 0.0f,
//     };
//     const int fd_freq = i_freq * num_points;

//     float real, imag;
//     float4 lmn, fd_re, fd_im;
//     float4 delta_vis_re = float4{
//         .x = 0.0f,
//         .y = 0.0f,
//         .z = 0.0f,
//         .w = 0.0f,
//     };
//     float4 delta_vis_im = float4{
//         .x = 0.0f,
//         .y = 0.0f,
//         .z = 0.0f,
//         .w = 0.0f,
//     };

//     for (int i_comp = 0; i_comp < num_points; i_comp++) {
//         lmn = d_point_lmns[i_comp];
//         fd_re = d_point_fds_re[fd_freq + i_comp];
//         fd_im = d_point_fds_im[fd_freq + i_comp];
//         // fd_re = d_point_fds_re[i_comp];
//         // fd_im = d_point_fds_im[i_comp];

//         // Don't use PI explicitly; CUDA's sincospi does that.
//         sincospif(uvw.x * lmn.x + uvw.y * lmn.y + uvw.z * lmn.z, &imag, &real);

//         delta_vis_re += fd_re * real - fd_im * imag;
//         delta_vis_im += fd_re * imag + fd_im * real;
//     }

//     // // Shared memory witchcraft.
//     // // Every point source component has an LMN (4 floats == 16 bytes) and 4
//     // // instrumental flux densities (8 floats == 32 bytes; 48 bytes total). To
//     // // be friendly to all CUDA architectures, we use 48kB of shared memory to
//     // // cache things (1024 components).
//     // const int magic = 1024;
//     // __shared__ float4 lmns[magic], fds_re[magic], fds_im[magic];
//     // int i_comp = 0;
//     // int i_comp_offset = 0;
//     // const int num_chunks = num_points / magic;
//     // const int num_remainder = num_points % magic;
//     // for (int chunk = 0; chunk < num_chunks; chunk++) {
//     //     for (i_comp = 0; i_comp < magic; i_comp++) {
//     //         lmns[i_comp] = d_point_lmns[i_comp + i_comp_offset];
//     //         fds_re[i_comp] = d_point_fds_re[fd_freq + i_comp + i_comp_offset];
//     //         fds_im[i_comp] = d_point_fds_im[fd_freq + i_comp + i_comp_offset];
//     //     }
//     //     i_comp_offset += magic;
//     //     __syncwarp();
//     //     for (i_comp = 0; i_comp < magic; i_comp++) {
//     //         lmn = lmns[i_comp];
//     //         fd_re = fds_re[i_comp];
//     //         fd_im = fds_im[i_comp];

//     //         // Don't use PI explicitly; CUDA's sincospi does that.
//     //         sincospi(uvw.x * lmn.x + uvw.x * lmn.y + uvw.z * lmn.z, &imag, &real);

//     //         delta_vis_re += fd_re * real - fd_im * imag;
//     //         delta_vis_im += fd_re * imag + fd_im * real;
//     //     }
//     // }
//     // // Remainder.
//     // for (i_comp = 0; i_comp < num_remainder; i_comp++) {
//     //     lmns[i_comp] = d_point_lmns[i_comp + i_comp_offset];
//     //     fds_re[i_comp] = d_point_fds_re[fd_freq + i_comp + i_comp_offset];
//     //     fds_im[i_comp] = d_point_fds_im[fd_freq + i_comp + i_comp_offset];
//     // }
//     // __syncwarp();
//     // for (i_comp = 0; i_comp < num_remainder; i_comp++) {
//     //     lmn = lmns[i_comp];
//     //     fd_re = fds_re[i_comp];
//     //     fd_im = fds_im[i_comp];

//     //     // Don't use PI explicitly; CUDA's sincospi does that.
//     //     sincospi(uvw.x * lmn.x + uvw.x * lmn.y + uvw.z * lmn.z, &imag, &real);

//     //     delta_vis_re += fd_re * real - fd_im * imag;
//     //     delta_vis_im += fd_re * imag + fd_im * real;
//     // }

//     d_vis[i_vis].xx_re += delta_vis_re.x;
//     d_vis[i_vis].xy_re += delta_vis_re.y;
//     d_vis[i_vis].yx_re += delta_vis_re.z;
//     d_vis[i_vis].yy_re += delta_vis_re.w;
//     d_vis[i_vis].xx_im += delta_vis_im.x;
//     d_vis[i_vis].xy_im += delta_vis_im.y;
//     d_vis[i_vis].yx_im += delta_vis_im.z;
//     d_vis[i_vis].yy_im += delta_vis_im.w;
// }

inline __device__ void extrap_fd(const int i_comp, const float freq, const float4 *d_fds_re, const float4 *d_fds_im,
                                 const float *d_ref_freqs, const float *d_sis, float4 *out_re, float4 *out_im) {
    const float ref_freq = d_ref_freqs[i_comp];
    const float flux_ratio = powf(freq / ref_freq, d_sis[i_comp]);
    float4 fd_re = d_fds_re[i_comp];
    float4 fd_im = d_fds_im[i_comp];
    *out_re = fd_re * flux_ratio;
    *out_im = fd_im * flux_ratio;
}

// Calculate flux densities on the GPU.
__global__ void model_points_kernel(const int num_baselines, const int num_freqs, const int num_vis, const UVW *d_uvws,
                                    const double *d_freqs, JonesF32 *d_vis, int num_points, const float4 *d_lmns,
                                    const float4 *d_fds_re, const float4 *d_fds_im, const float *d_fd_ref_freqs,
                                    const float *d_fd_sis) {
    const int i_vis = threadIdx.x + (blockDim.x * blockIdx.x);

    if (i_vis >= num_vis)
        return;

    // Get the indices for this thread. `i_vis` indexes over baselines and
    // frequencies, with frequencies moving faster than baselines.
    const int i_freq = i_vis % num_freqs;
    const int i_bl = i_vis / num_freqs;
    const double freq_double = d_freqs[i_freq];
    const double two_on_lambda = 2.0 / (VEL_C / freq_double);
    const float freq = (float)freq_double;
    UVW uvw_temp = d_uvws[i_bl];
    const float4 uvw = float4{
        .x = (float)(uvw_temp.u * two_on_lambda),
        .y = (float)(uvw_temp.v * two_on_lambda),
        .z = (float)(uvw_temp.w * two_on_lambda),
        .w = 0.0f,
    };

    float real, imag;
    float4 lmn, fd_re, fd_im;
    float4 delta_vis_re = float4{
        .x = 0.0f,
        .y = 0.0f,
        .z = 0.0f,
        .w = 0.0f,
    };
    float4 delta_vis_im = float4{
        .x = 0.0f,
        .y = 0.0f,
        .z = 0.0f,
        .w = 0.0f,
    };

    for (int i_comp = 0; i_comp < num_points; i_comp++) {
        lmn = d_lmns[i_comp];

        // Don't use PI explicitly; CUDA's sincospi does that.
        sincospif(uvw.x * lmn.x + uvw.y * lmn.y + uvw.z * lmn.z, &imag, &real);

        extrap_fd(i_comp, freq, d_fds_re, d_fds_im, d_fd_ref_freqs, d_fd_sis, &fd_re, &fd_im);
        delta_vis_re += fd_re * real - fd_im * imag;
        delta_vis_im += fd_re * imag + fd_im * real;
    }

    d_vis[i_vis].xx_re += delta_vis_re.x;
    d_vis[i_vis].xy_re += delta_vis_re.y;
    d_vis[i_vis].yx_re += delta_vis_re.z;
    d_vis[i_vis].yy_re += delta_vis_re.w;
    d_vis[i_vis].xx_im += delta_vis_im.x;
    d_vis[i_vis].xy_im += delta_vis_im.y;
    d_vis[i_vis].yx_im += delta_vis_im.z;
    d_vis[i_vis].yy_im += delta_vis_im.w;
}

/**
 * Kernel for calculating Gaussian-source-component visibilities.
 */
__global__ void model_gaussians_kernel(const size_t num_freqs, const size_t num_vis, const UVW *d_uvws,
                                       const double *d_freqs, JonesF32 *d_vis, const size_t num_gaussians,
                                       const LMN *d_gaussian_lmns, const JonesF64 *d_gaussian_fds,
                                       const GaussianParams *d_gaussian_params) {
    const int i_vis = threadIdx.x + (blockDim.x * blockIdx.x);

    if (i_vis >= num_vis)
        return;

    // Get the pointers for this thread.
    const UVW *uvw = &d_uvws[i_vis / num_freqs];
    JonesF32 *vis = &d_vis[i_vis];
    // `i_vis` indexes over baselines and frequencies, and there are `num_freqs`
    // frequencies per baseline.
    const size_t i_freq = i_vis % num_freqs;

    calc_gaussian_vis(i_freq, uvw, vis, d_freqs, num_gaussians, d_gaussian_lmns, d_gaussian_fds, d_gaussian_params);
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

extern "C" int model_points(const size_t num_points, const LMN *point_lmns, const JonesF64 *point_fds,
                            const Addresses *a) {
    // LMN *d_lmns = NULL;
    // size_t size_lmns = num_points * sizeof(LMN);
    // cudaSoftCheck(cudaMalloc(&d_lmns, size_lmns));
    // cudaSoftCheck(cudaMemcpy(d_lmns, point_lmns, size_lmns, cudaMemcpyHostToDevice));

    // double4 *d_lmns = NULL;
    // size_t size_lmns = num_points * sizeof(double4);
    // cudaSoftCheck(cudaMalloc(&d_lmns, size_lmns));
    // double4 *lmns_padded = (double4 *)malloc(size_lmns);
    // for (size_t i = 0; i < num_points; i++) {
    //     LMN lmn = point_lmns[i];
    //     lmns_padded[i].x = lmn.l;
    //     lmns_padded[i].y = lmn.m;
    //     lmns_padded[i].z = lmn.n;
    //     lmns_padded[i].w = 0.0;
    // }
    // cudaSoftCheck(cudaMemcpy(d_lmns, lmns_padded, size_lmns, cudaMemcpyHostToDevice));
    float4 *d_lmns = NULL;
    size_t size_lmns = num_points * sizeof(float4);
    cudaSoftCheck(cudaMalloc(&d_lmns, size_lmns));
    float4 *lmns_padded = (float4 *)malloc(size_lmns);
    for (size_t i = 0; i < num_points; i++) {
        LMN lmn = point_lmns[i];
        lmns_padded[i].x = lmn.l;
        lmns_padded[i].y = lmn.m;
        lmns_padded[i].z = lmn.n - 1.0f;
        lmns_padded[i].w = 0.0f;
    }
    cudaSoftCheck(cudaMemcpy(d_lmns, lmns_padded, size_lmns, cudaMemcpyHostToDevice));

    // double *d_ls = NULL;
    // double *d_ms = NULL;
    // double *d_ns = NULL;
    // size_t size_lmns = num_points * sizeof(double);
    // cudaSoftCheck(cudaMalloc(&d_ls, size_lmns));
    // cudaSoftCheck(cudaMalloc(&d_ms, size_lmns));
    // cudaSoftCheck(cudaMalloc(&d_ns, size_lmns));
    // double *ls = (double *)malloc(size_lmns);
    // double *ms = (double *)malloc(size_lmns);
    // double *ns = (double *)malloc(size_lmns);
    // for (size_t i = 0; i < num_points; i++) {
    //     ls[i] = point_lmns[i].l;
    //     ms[i] = point_lmns[i].m;
    //     ns[i] = point_lmns[i].n;
    // }
    // cudaSoftCheck(cudaMemcpy(d_ls, ls, size_lmns, cudaMemcpyHostToDevice));
    // cudaSoftCheck(cudaMemcpy(d_ms, ms, size_lmns, cudaMemcpyHostToDevice));
    // cudaSoftCheck(cudaMemcpy(d_ns, ns, size_lmns, cudaMemcpyHostToDevice));
    // float *d_ls = NULL;
    // float *d_ms = NULL;
    // float *d_ns = NULL;
    // size_t size_lmns = num_points * sizeof(float);
    // cudaSoftCheck(cudaMalloc(&d_ls, size_lmns));
    // cudaSoftCheck(cudaMalloc(&d_ms, size_lmns));
    // cudaSoftCheck(cudaMalloc(&d_ns, size_lmns));
    // float *ls = (float *)malloc(size_lmns);
    // float *ms = (float *)malloc(size_lmns);
    // float *ns = (float *)malloc(size_lmns);
    // for (size_t i = 0; i < num_points; i++) {
    //     ls[i] = point_lmns[i].l;
    //     ms[i] = point_lmns[i].m;
    //     ns[i] = point_lmns[i].n;
    // }
    // cudaSoftCheck(cudaMemcpy(d_ls, ls, size_lmns, cudaMemcpyHostToDevice));
    // cudaSoftCheck(cudaMemcpy(d_ms, ms, size_lmns, cudaMemcpyHostToDevice));
    // cudaSoftCheck(cudaMemcpy(d_ns, ns, size_lmns, cudaMemcpyHostToDevice));

    // JonesF64 *d_fds = NULL;
    // size_t size_fds = num_points * a->num_freqs * sizeof(JonesF64);
    // cudaSoftCheck(cudaMalloc(&d_fds, size_fds));
    // cudaSoftCheck(cudaMemcpy(d_fds, point_fds, size_fds, cudaMemcpyHostToDevice));

    // double4 *d_fds_re = NULL;
    // double4 *d_fds_im = NULL;
    // size_t size_fds = num_points * a->num_freqs * sizeof(double4);
    // cudaSoftCheck(cudaMalloc(&d_fds_re, size_fds));
    // cudaSoftCheck(cudaMalloc(&d_fds_im, size_fds));
    // double4 *point_fds_re = (double4 *)malloc(size_fds);
    // double4 *point_fds_im = (double4 *)malloc(size_fds);

    float4 *d_fds_re = NULL;
    float4 *d_fds_im = NULL;
    size_t size_fds = num_points * a->num_freqs * sizeof(float4);
    cudaSoftCheck(cudaMalloc(&d_fds_re, size_fds));
    cudaSoftCheck(cudaMalloc(&d_fds_im, size_fds));
    float4 *point_fds_re = (float4 *)malloc(size_fds);
    float4 *point_fds_im = (float4 *)malloc(size_fds);
    for (size_t i = 0; i < num_points * a->num_freqs; i++) {
        JonesF64 fd = point_fds[i];
        point_fds_re[i].x = fd.xx_re;
        point_fds_re[i].y = fd.xy_re;
        point_fds_re[i].z = fd.yx_re;
        point_fds_re[i].w = fd.yy_re;
        point_fds_im[i].x = fd.xx_im;
        point_fds_im[i].y = fd.xy_im;
        point_fds_im[i].z = fd.yx_im;
        point_fds_im[i].w = fd.yy_im;
    }
    cudaSoftCheck(cudaMemcpy(d_fds_re, point_fds_re, size_fds, cudaMemcpyHostToDevice));
    cudaSoftCheck(cudaMemcpy(d_fds_im, point_fds_im, size_fds, cudaMemcpyHostToDevice));

    // double2 *d_fds_xx = NULL;
    // double2 *d_fds_xy = NULL;
    // double2 *d_fds_yx = NULL;
    // double2 *d_fds_yy = NULL;
    // size_t size_fds = num_points * a->num_freqs * sizeof(double2);
    // cudaSoftCheck(cudaMalloc(&d_fds_xx, size_fds));
    // cudaSoftCheck(cudaMalloc(&d_fds_xy, size_fds));
    // cudaSoftCheck(cudaMalloc(&d_fds_yx, size_fds));
    // cudaSoftCheck(cudaMalloc(&d_fds_yy, size_fds));
    // double2 *point_fds_xx = (double2 *)malloc(size_fds);
    // double2 *point_fds_xy = (double2 *)malloc(size_fds);
    // double2 *point_fds_yx = (double2 *)malloc(size_fds);
    // double2 *point_fds_yy = (double2 *)malloc(size_fds);
    // for (size_t i = 0; i < num_points; i++) {
    //     JonesF64 fd = point_fds[i];
    //     point_fds_xx[i].x = fd.xx_re;
    //     point_fds_xx[i].y = fd.xx_im;
    //     point_fds_xy[i].x = fd.xy_re;
    //     point_fds_xy[i].y = fd.xy_im;
    //     point_fds_yx[i].x = fd.yx_re;
    //     point_fds_yx[i].y = fd.yx_im;
    //     point_fds_yy[i].x = fd.yy_re;
    //     point_fds_yy[i].y = fd.yy_im;
    // }
    // cudaSoftCheck(cudaMemcpy(d_fds_xx, point_fds_xx, size_fds, cudaMemcpyHostToDevice));
    // cudaSoftCheck(cudaMemcpy(d_fds_xy, point_fds_xy, size_fds, cudaMemcpyHostToDevice));
    // cudaSoftCheck(cudaMemcpy(d_fds_yx, point_fds_yx, size_fds, cudaMemcpyHostToDevice));
    // cudaSoftCheck(cudaMemcpy(d_fds_yy, point_fds_yy, size_fds, cudaMemcpyHostToDevice));
    // float2 *d_fds_xx = NULL;
    // float2 *d_fds_xy = NULL;
    // float2 *d_fds_yx = NULL;
    // float2 *d_fds_yy = NULL;
    // size_t size_fds = num_points * a->num_freqs * sizeof(float2);
    // cudaSoftCheck(cudaMalloc(&d_fds_xx, size_fds));
    // cudaSoftCheck(cudaMalloc(&d_fds_xy, size_fds));
    // cudaSoftCheck(cudaMalloc(&d_fds_yx, size_fds));
    // cudaSoftCheck(cudaMalloc(&d_fds_yy, size_fds));
    // float2 *point_fds_xx = (float2 *)malloc(size_fds);
    // float2 *point_fds_xy = (float2 *)malloc(size_fds);
    // float2 *point_fds_yx = (float2 *)malloc(size_fds);
    // float2 *point_fds_yy = (float2 *)malloc(size_fds);
    // for (size_t i = 0; i < num_points; i++) {
    //     JonesF64 fd = point_fds[i];
    //     point_fds_xx[i].x = fd.xx_re;
    //     point_fds_xx[i].y = fd.xx_im;
    //     point_fds_xy[i].x = fd.xy_re;
    //     point_fds_xy[i].y = fd.xy_im;
    //     point_fds_yx[i].x = fd.yx_re;
    //     point_fds_yx[i].y = fd.yx_im;
    //     point_fds_yy[i].x = fd.yy_re;
    //     point_fds_yy[i].y = fd.yy_im;
    // }
    // cudaSoftCheck(cudaMemcpy(d_fds_xx, point_fds_xx, size_fds, cudaMemcpyHostToDevice));
    // cudaSoftCheck(cudaMemcpy(d_fds_xy, point_fds_xy, size_fds, cudaMemcpyHostToDevice));
    // cudaSoftCheck(cudaMemcpy(d_fds_yx, point_fds_yx, size_fds, cudaMemcpyHostToDevice));
    // cudaSoftCheck(cudaMemcpy(d_fds_yy, point_fds_yy, size_fds, cudaMemcpyHostToDevice));

    // Thread blocks are distributed by visibility (one visibility per frequency
    // and baseline).
    dim3 blocks, threads;
    // threads.x = 128;
    threads.x = 512;
    threads.y = 1;
    blocks.x = (int)ceil((double)a->num_vis / (double)threads.x);
    blocks.y = 1;

    // model_points_kernel<<<blocks, threads>>>(a->num_freqs, a->num_vis, a->d_uvws, a->d_freqs, a->d_vis, num_points,
    //                                          d_lmns, d_fds);
    // model_points_kernel<<<blocks, threads>>>(a->num_freqs, a->num_vis, a->d_uvws, a->d_freqs, a->d_vis, num_points,
    //                                          d_lmns, d_fds_re, d_fds_im);

    // model_points_kernel<<<blocks, threads>>>(a->num_vis / a->num_freqs, a->num_freqs, a->num_vis, a->d_uvws,
    // a->d_freqs,
    //                                          a->d_vis, num_points, d_lmns, d_fds_re, d_fds_im);

    // Lying for now.
    float *d_fd_ref_freqs = NULL;
    float *d_fd_sis = NULL;
    size_t size = num_points * sizeof(float);
    cudaSoftCheck(cudaMalloc(&d_fd_ref_freqs, size));
    cudaSoftCheck(cudaMalloc(&d_fd_sis, size));
    float *fd_ref_freqs = (float *)malloc(size);
    float *fd_sis = (float *)malloc(size);
    for (size_t i = 0; i < num_points; i++) {
        fd_ref_freqs[i] = 150e6;
        fd_sis[i] = -0.8;
    }
    cudaSoftCheck(cudaMemcpy(d_fd_ref_freqs, fd_ref_freqs, size, cudaMemcpyHostToDevice));
    cudaSoftCheck(cudaMemcpy(d_fd_sis, fd_sis, size, cudaMemcpyHostToDevice));

    // Calculate flux densities on GPU.
    model_points_kernel<<<blocks, threads>>>(a->num_vis / a->num_freqs, a->num_freqs, a->num_vis, a->d_uvws, a->d_freqs,
                                             a->d_vis, num_points, d_lmns, d_fds_re, d_fds_im, d_fd_ref_freqs,
                                             d_fd_sis);

    // model_points_kernel<<<blocks, threads>>>(a->num_freqs, a->num_vis, a->d_uvws, a->d_freqs, a->d_vis, num_points,
    //                                          d_ls, d_ms, d_ns, d_fds_xx, d_fds_xy, d_fds_yx, d_fds_yy);
    cudaDeviceSynchronize();
    cudaCheck(cudaPeekAtLastError());

    // cudaSoftCheck(cudaFree(d_lmns));
    // cudaSoftCheck(cudaFree(d_fds));
    // cudaSoftCheck(cudaFree(d_fds_re));
    // cudaSoftCheck(cudaFree(d_fds_im));

    return EXIT_SUCCESS;
}

extern "C" int model_gaussians(const size_t num_gaussians, const LMN *gaussian_lmns, const JonesF64 *gaussian_fds,
                               const GaussianParams *gaussian_params, const Addresses a) {
    LMN *d_lmns = NULL;
    size_t size_lmns = num_gaussians * sizeof(LMN);
    cudaSoftCheck(cudaMalloc(&d_lmns, size_lmns));
    cudaSoftCheck(cudaMemcpy(d_lmns, gaussian_lmns, size_lmns, cudaMemcpyHostToDevice));

    JonesF64 *d_fds = NULL;
    size_t size_fds = num_gaussians * a.num_freqs * sizeof(JonesF64);
    cudaSoftCheck(cudaMalloc(&d_fds, size_fds));
    cudaSoftCheck(cudaMemcpy(d_fds, gaussian_fds, size_fds, cudaMemcpyHostToDevice));

    GaussianParams *d_gaussian_params = NULL;
    size_t size_gaussian_params = num_gaussians * sizeof(GaussianParams);
    cudaSoftCheck(cudaMalloc(&d_gaussian_params, size_gaussian_params));
    cudaSoftCheck(cudaMemcpy(d_gaussian_params, gaussian_params, size_gaussian_params, cudaMemcpyHostToDevice));

    // Thread blocks are distributed by visibility (one visibility per frequency
    // and baseline).
    dim3 blocks, threads;
    threads.x = 32;
    threads.y = 1;
    blocks.x = (int)ceil((double)a.num_vis / (double)threads.x);
    blocks.y = 1;

    model_gaussians_kernel<<<blocks, threads>>>(a.num_freqs, a.num_vis, a.d_uvws, a.d_freqs, a.d_vis, num_gaussians,
                                                d_lmns, d_fds, d_gaussian_params);
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
    cudaDeviceSynchronize();
    cudaCheck(cudaPeekAtLastError());

    cudaSoftCheck(cudaFree(d_lmns));
    cudaSoftCheck(cudaFree(d_fds));
    cudaSoftCheck(cudaFree(d_gaussian_params));

    return EXIT_SUCCESS;
}

extern "C" int model_timestep(const size_t num_baselines, const size_t num_freqs, const size_t num_points,
                              const size_t num_gaussians, const size_t num_shapelets, const UVW *uvws,
                              const double *freqs, const LMN *point_lmns, const JonesF64 *point_fds,
                              const LMN *gaussian_lmns, const JonesF64 *gaussian_fds,
                              const GaussianParams *gaussian_gaussian_params, const LMN *shapelet_lmns,
                              const JonesF64 *shapelet_fds, const GaussianParams *shapelet_gaussian_params,
                              const ShapeletUV *shapelet_uvs, const ShapeletCoeff *shapelet_coeffs,
                              const size_t *num_shapelet_coeffs, const double *shapelet_basis_values,
                              const size_t sbf_l, const size_t sbf_n, const double sbf_c, const double sbf_dx,
                              JonesF32 *vis) {
    int status = 0;

    Addresses a =
        init_model(num_baselines, num_freqs, sbf_l, sbf_n, sbf_c, sbf_dx, uvws, freqs, shapelet_basis_values, vis);

    if (num_points > 0) {
        // Yes, pass `a` by value.
        status = model_points(num_points, point_lmns, point_fds, &a);
        if (status != EXIT_SUCCESS) {
            return status;
        }
    } // if (num_points > 0)

    if (num_gaussians > 0) {
        status = model_gaussians(num_gaussians, gaussian_lmns, gaussian_fds, gaussian_gaussian_params, a);
        if (status != EXIT_SUCCESS) {
            return status;
        }
    } // if (num_gaussians > 0)

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
