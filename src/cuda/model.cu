// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <stdlib.h>

#include <cuComplex.h>

#include "common.cuh"
#include "model.h"
#include "types.h"

const int NUM_THREADS_PER_BLOCK_POINTS = 256;
const int NUM_THREADS_PER_BLOCK_GAUSSIANS = 256;
const int NUM_THREADS_PER_BLOCK_SHAPELETS = 32;

inline __device__ JONES extrap_power_law_fd(const FLOAT freq, const JONES ref_flux_density,
                                            const FLOAT spectral_index) {
    const FLOAT flux_ratio = POW(freq / POWER_LAW_FD_REF_FREQ, spectral_index);
    return ref_flux_density * flux_ratio;
}

inline __device__ JONES extrap_curved_power_law_fd(const FLOAT freq, const JONES ref_flux_density,
                                                   const FLOAT spectral_index, const FLOAT q) {
    const FLOAT flux_ratio = POW(freq / POWER_LAW_FD_REF_FREQ, spectral_index);
    const FLOAT log_term = LOG(freq / POWER_LAW_FD_REF_FREQ);
    const FLOAT curved_component = EXP(q * log_term * log_term);
    return ref_flux_density * flux_ratio * curved_component;
}

inline __device__ FLOAT get_gaussian_envelope(const GaussianParams g_params, const UVW uvw) {
    FLOAT s_pa, c_pa;
    SINCOS(g_params.pa, &s_pa, &c_pa);
    // Temporary variables for clarity.
    const FLOAT k_x = uvw.u * s_pa + uvw.v * c_pa;
    const FLOAT k_y = uvw.u * c_pa - uvw.v * s_pa;
    return EXP(EXP_CONST * ((g_params.maj * g_params.maj) * (k_x * k_x) + (g_params.min * g_params.min) * (k_y * k_y)));
}

inline __device__ COMPLEX get_shapelet_envelope(const GaussianParams g_params, const ShapeletUV s_uv,
                                                const int num_coeffs, const ShapeletCoeff *coeffs,
                                                const FLOAT *shapelet_basis_values) {
    const FLOAT I_POWERS_REAL[4] = {1.0, 0.0, -1.0, 0.0};
    const FLOAT I_POWERS_IMAG[4] = {0.0, 1.0, 0.0, -1.0};

    FLOAT s_pa, c_pa;
    SINCOS(g_params.pa, &s_pa, &c_pa);

    // Temporary variables for clarity.
    const FLOAT x = s_uv.u * s_pa + s_uv.v * c_pa;
    const FLOAT y = s_uv.u * c_pa - s_uv.v * s_pa;
    const FLOAT const_x = g_params.maj * SHAPELET_CONST;
    const FLOAT const_y = -g_params.min * SHAPELET_CONST;
    const FLOAT x_pos = x * const_x + SBF_C;
    const FLOAT y_pos = y * const_y + SBF_C;
    // If the positions (indices) are negative, our range of shapelet basis
    // functions/values/whatever is too small for the incoming shapelet UVs. CHJ
    // spotted this when (accidentally) trying to simulate Fornax A with SDC3
    // data; these indices only went negative for the long baselines. Instead of
    // increasing the range of allowed UVs, we're doing the lazy thing of
    // sweeping this issue under the carpet, because (1) MWA data is unlikely to
    // ever have baselines rivalling SKA-Low and (2) SDC3 data doesn't use
    // Fornax A or shapelets.
    int x_pos_int = x_pos < 0 ? 0 : (int)FLOOR(x_pos);
    int y_pos_int = y_pos < 0 ? 0 : (int)FLOOR(y_pos);

    COMPLEX envelope = COMPLEX{
        .x = 0.0,
        .y = 0.0,
    };
    for (int i_coeff = 0; i_coeff < num_coeffs; i_coeff++) {
        const ShapeletCoeff coeff = coeffs[i_coeff];

        FLOAT x_low = shapelet_basis_values[SBF_L * (int)coeff.n1 + x_pos_int];
        FLOAT x_high = shapelet_basis_values[SBF_L * (int)coeff.n1 + x_pos_int + 1];
        FLOAT u_value = x_low + (x_high - x_low) * (x_pos - FLOOR(x_pos));

        FLOAT y_low = shapelet_basis_values[SBF_L * (int)coeff.n2 + y_pos_int];
        FLOAT y_high = shapelet_basis_values[SBF_L * (int)coeff.n2 + y_pos_int + 1];
        FLOAT v_value = y_low + (y_high - y_low) * (y_pos - FLOOR(y_pos));

        FLOAT rest = coeff.value * u_value * v_value;

        // I_POWER_TABLE stuff. The intention is just to find the
        // appropriate power of i, i.e.:
        // index = (n1 + n2) % 4    (so that index is between 0 and 3 inclusive)
        // i^index, e.g.
        // i^0 =  1.0 + 0.0i
        // i^1 =  0.0 + 1.0i
        // i^2 = -1.0 + 0.0i
        // i^3 =  0.0 - 1.0i
        //
        // The following is my attempt at doing this efficiently.
        int i_power_index = (int)((coeff.n1 + coeff.n2) % 4);
        COMPLEX i_power = COMPLEX{
            .x = I_POWERS_REAL[i_power_index],
            .y = I_POWERS_IMAG[i_power_index],
        };

        envelope += i_power * rest;
    }

    return envelope;
}

__global__ void model_points_kernel(const int num_freqs, const int num_baselines, const FLOAT *freqs, const UVW *uvws,
                                    const Points comps, const JONES *__restrict__ beam_jones, const int *tile_map,
                                    const int *__restrict__ freq_map, int num_fee_freqs,
                                    const int *__restrict__ tile_index_to_unflagged_tile_index_map,
                                    JonesF32 *__restrict__ vis_fb) {
    // The 0-indexed number of tiles as a float.
    const float num_tiles = (sqrtf(1.0f + 8.0f * (float)num_baselines) - 1.0f) / 2.0f;
    const int num_directions = comps.num_power_laws + comps.num_curved_power_laws + comps.num_lists;

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < num_baselines * num_freqs; i += gridDim.x * blockDim.x) {
        const int i_bl = i % num_baselines;
        const int i_freq = i / num_baselines;

        const FLOAT freq = freqs[i_freq];
        const UVW uvw = uvws[i_bl] * freq / VEL_C;

        // Get tile indices for this baseline to get the correct beam responses.
        const float tile1f =
            floorf(-0.5f * sqrtf(4.0f * num_tiles * (num_tiles + 1.0f) - 8.0f * i_bl + 1.0f) + num_tiles + 0.5f);
        const int i_tile2 = i_bl - (int)(tile1f * (num_tiles - (tile1f + 1.0f) / 2.0f)) + 1;
        const int i_tile1 = (int)tile1f;

        // `i_j1_row` and `i_j2_row` are indices into beam responses.
        const int i_j1_row = tile_map[tile_index_to_unflagged_tile_index_map[i_tile1]];
        const int i_j2_row = tile_map[tile_index_to_unflagged_tile_index_map[i_tile2]];

        const int i_col = freq_map[i_freq];
        const JONES *j1 = beam_jones + num_directions * (num_fee_freqs * i_j1_row + i_col);
        const JONES *j2 = beam_jones + num_directions * (num_fee_freqs * i_j2_row + i_col);

        COMPLEX geom;
        JONES delta_vis = JONES{
            .j00_re = 0.0,
            .j00_im = 0.0,
            .j01_re = 0.0,
            .j01_im = 0.0,
            .j10_re = 0.0,
            .j10_im = 0.0,
            .j11_re = 0.0,
            .j11_im = 0.0,
        };

        for (int i_comp = 0; i_comp < comps.num_power_laws; i_comp++) {
            // Estimate a flux density from the reference FD and spectral index.
            JONES fd = extrap_power_law_fd(freq, comps.power_law_fds[i_comp], comps.power_law_sis[i_comp]);
            apply_beam(j1++, &fd, j2++);

            // Measurement equation. The 2 PI is already multiplied on the LMN
            // terms (as well as a -1 on the n).
            const LmnRime lmn = comps.power_law_lmns[i_comp];
            SINCOS(uvw.u * lmn.l + uvw.v * lmn.m + uvw.w * lmn.n, &geom.y, &geom.x);
            delta_vis += fd * geom;
        }

        for (int i_comp = 0; i_comp < comps.num_curved_power_laws; i_comp++) {
            JONES fd =
                extrap_curved_power_law_fd(freq, comps.curved_power_law_fds[i_comp], comps.curved_power_law_sis[i_comp],
                                           comps.curved_power_law_qs[i_comp]);
            apply_beam(j1++, &fd, j2++);

            const LmnRime lmn = comps.curved_power_law_lmns[i_comp];
            SINCOS(uvw.u * lmn.l + uvw.v * lmn.m + uvw.w * lmn.n, &geom.y, &geom.x);
            delta_vis += fd * geom;
        }

        for (int i_comp = 0; i_comp < comps.num_lists; i_comp++) {
            JONES fd = comps.list_fds[i_freq * comps.num_lists + i_comp];
            apply_beam(j1++, &fd, j2++);

            const LmnRime lmn = comps.list_lmns[i_comp];
            SINCOS(uvw.u * lmn.l + uvw.v * lmn.m + uvw.w * lmn.n, &geom.y, &geom.x);
            delta_vis += fd * geom;
        }

        // Visibilities are ordered over baselines and frequencies, with
        // baselines moving faster than frequencies.
        vis_fb[i_freq * num_baselines + i_bl] += delta_vis;
    }
}

/**
 * Kernel for calculating Gaussian-source-component visibilities.
 */
__global__ void model_gaussians_kernel(const int num_freqs, const int num_baselines, const FLOAT *freqs,
                                       const UVW *uvws, const Gaussians comps, const JONES *__restrict__ beam_jones,
                                       const int *tile_map, const int *__restrict__ freq_map, const int num_fee_freqs,
                                       const int *__restrict__ tile_index_to_unflagged_tile_index_map,
                                       JonesF32 *__restrict__ vis_fb) {
    // The 0-indexed number of tiles as a float.
    const float num_tiles = (sqrtf(1.0f + 8.0f * (float)num_baselines) - 1.0f) / 2.0f;
    const int num_directions = comps.num_power_laws + comps.num_curved_power_laws + comps.num_lists;

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < num_baselines * num_freqs; i += gridDim.x * blockDim.x) {
        const int i_bl = i % num_baselines;
        const int i_freq = i / num_baselines;

        const FLOAT freq = freqs[i_freq];
        const UVW uvw = uvws[i_bl] * freq / VEL_C;

        // Get tile indices for this baseline to get the correct beam responses.
        const float tile1f =
            floorf(-0.5f * sqrtf(4.0f * num_tiles * (num_tiles + 1.0f) - 8.0f * i_bl + 1.0f) + num_tiles + 0.5f);
        const int i_tile2 = i_bl - (int)(tile1f * (num_tiles - (tile1f + 1.0f) / 2.0f)) + 1;
        const int i_tile1 = (int)tile1f;

        // `i_j1_row` and `i_j2_row` are indices into beam responses.
        const int i_j1_row = tile_map[tile_index_to_unflagged_tile_index_map[i_tile1]];
        const int i_j2_row = tile_map[tile_index_to_unflagged_tile_index_map[i_tile2]];

        const int i_col = freq_map[i_freq];
        const JONES *j1 = beam_jones + num_directions * (num_fee_freqs * i_j1_row + i_col);
        const JONES *j2 = beam_jones + num_directions * (num_fee_freqs * i_j2_row + i_col);

        COMPLEX geom;
        FLOAT envelope;
        JONES delta_vis = JONES{
            .j00_re = 0.0,
            .j00_im = 0.0,
            .j01_re = 0.0,
            .j01_im = 0.0,
            .j10_re = 0.0,
            .j10_im = 0.0,
            .j11_re = 0.0,
            .j11_im = 0.0,
        };

        for (int i_comp = 0; i_comp < comps.num_power_laws; i_comp++) {
            // Estimate a flux density from the reference FD and spectral index.
            JONES fd = extrap_power_law_fd(freq, comps.power_law_fds[i_comp], comps.power_law_sis[i_comp]);
            apply_beam(j1++, &fd, j2++);

            // Measurement equation. The 2 PI is already multiplied on the LMN
            // terms (as well as a -1 on the n).
            const LmnRime lmn = comps.power_law_lmns[i_comp];
            SINCOS(uvw.u * lmn.l + uvw.v * lmn.m + uvw.w * lmn.n, &geom.y, &geom.x);

            // Scale by envelope.
            envelope = get_gaussian_envelope(comps.power_law_gps[i_comp], uvw);
            delta_vis += fd * geom * envelope;
        }

        for (int i_comp = 0; i_comp < comps.num_curved_power_laws; i_comp++) {
            JONES fd =
                extrap_curved_power_law_fd(freq, comps.curved_power_law_fds[i_comp], comps.curved_power_law_sis[i_comp],
                                           comps.curved_power_law_qs[i_comp]);
            apply_beam(j1++, &fd, j2++);

            const LmnRime lmn = comps.curved_power_law_lmns[i_comp];
            SINCOS(uvw.u * lmn.l + uvw.v * lmn.m + uvw.w * lmn.n, &geom.y, &geom.x);

            envelope = get_gaussian_envelope(comps.curved_power_law_gps[i_comp], uvw);
            delta_vis += fd * geom * envelope;
        }

        for (int i_comp = 0; i_comp < comps.num_lists; i_comp++) {
            JONES fd = comps.list_fds[i_freq * comps.num_lists + i_comp];
            apply_beam(j1++, &fd, j2++);

            const LmnRime lmn = comps.list_lmns[i_comp];
            SINCOS(uvw.u * lmn.l + uvw.v * lmn.m + uvw.w * lmn.n, &geom.y, &geom.x);

            envelope = get_gaussian_envelope(comps.list_gps[i_comp], uvw);
            delta_vis += fd * geom * envelope;
        }

        // Visibilities are ordered over baselines and frequencies, with
        // baselines moving faster than frequencies.
        vis_fb[i_freq * num_baselines + i_bl] += delta_vis;
    }
}

/**
 * Kernel for calculating shapelet-source-component visibilities.
 *
 * `*_shapelet_coeffs` is actually a flattened array-of-arrays. The size of each
 * sub-array is given by an element of `*_num_shapelet_coeffs`.
 */
__global__ void model_shapelets_kernel(const int num_freqs, const int num_baselines, const FLOAT *__restrict__ freqs,
                                       const UVW *uvws, const Shapelets comps,
                                       const FLOAT *__restrict__ shapelet_basis_values,
                                       const JONES *__restrict__ beam_jones, const int *__restrict__ tile_map,
                                       const int *__restrict__ freq_map, const int num_fee_freqs,
                                       const int *__restrict__ tile_index_to_unflagged_tile_index_map,
                                       JonesF32 *__restrict__ vis_fb) {
    // The 0-indexed number of tiles as a float.
    const float num_tiles = (sqrtf(1.0f + 8.0f * (float)num_baselines) - 1.0f) / 2.0f;
    const int num_directions = comps.num_power_laws + comps.num_curved_power_laws + comps.num_lists;

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < num_baselines * num_freqs; i += gridDim.x * blockDim.x) {
        const int i_bl = i % num_baselines;
        const int i_freq = i / num_baselines;

        const FLOAT freq = freqs[i_freq];
        const FLOAT one_on_lambda = freq / VEL_C;
        const UVW uvw = uvws[i_bl] * one_on_lambda;

        // Get tile indices for this baseline to get the correct beam responses.
        const float tile1f =
            floorf(-0.5f * sqrtf(4.0f * num_tiles * (num_tiles + 1.0f) - 8.0f * i_bl + 1.0f) + num_tiles + 0.5f);
        const int i_tile2 = i_bl - (int)(tile1f * (num_tiles - (tile1f + 1.0f) / 2.0f)) + 1;
        const int i_tile1 = (int)tile1f;

        // `i_j1_row` and `i_j2_row` are indices into beam responses.
        const int i_j1_row = tile_map[tile_index_to_unflagged_tile_index_map[i_tile1]];
        const int i_j2_row = tile_map[tile_index_to_unflagged_tile_index_map[i_tile2]];

        const int i_col = freq_map[i_freq];
        const JONES *j1 = beam_jones + num_directions * (num_fee_freqs * i_j1_row + i_col);
        const JONES *j2 = beam_jones + num_directions * (num_fee_freqs * i_j2_row + i_col);

        COMPLEX geom;
        JONES delta_vis = JONES{
            .j00_re = 0.0,
            .j00_im = 0.0,
            .j01_re = 0.0,
            .j01_im = 0.0,
            .j10_re = 0.0,
            .j10_im = 0.0,
            .j11_re = 0.0,
            .j11_im = 0.0,
        };

        const ShapeletCoeff *shapelet_coeffs = comps.power_law_shapelet_coeffs;
        for (int i_comp = 0; i_comp < comps.num_power_laws; i_comp++) {
            JONES fd = extrap_power_law_fd(freq, comps.power_law_fds[i_comp], comps.power_law_sis[i_comp]);
            apply_beam(j1++, &fd, j2++);

            const LmnRime lmn = comps.power_law_lmns[i_comp];
            SINCOS(uvw.u * lmn.l + uvw.v * lmn.m + uvw.w * lmn.n, &geom.y, &geom.x);

            // Scale by envelope.
            ShapeletUV s_uv = comps.power_law_shapelet_uvs[i_bl * comps.num_power_laws + i_comp] * one_on_lambda;
            int num_coeffs = comps.power_law_num_shapelet_coeffs[i_comp];
            COMPLEX envelope = get_shapelet_envelope(comps.power_law_gps[i_comp], s_uv, num_coeffs, shapelet_coeffs,
                                                     shapelet_basis_values);
            shapelet_coeffs += num_coeffs;

            delta_vis += fd * geom * envelope;
        }

        shapelet_coeffs = comps.curved_power_law_shapelet_coeffs;
        for (int i_comp = 0; i_comp < comps.num_curved_power_laws; i_comp++) {
            JONES fd =
                extrap_curved_power_law_fd(freq, comps.curved_power_law_fds[i_comp], comps.curved_power_law_sis[i_comp],
                                           comps.curved_power_law_qs[i_comp]);
            apply_beam(j1++, &fd, j2++);

            const LmnRime lmn = comps.curved_power_law_lmns[i_comp];
            SINCOS(uvw.u * lmn.l + uvw.v * lmn.m + uvw.w * lmn.n, &geom.y, &geom.x);

            ShapeletUV s_uv =
                comps.curved_power_law_shapelet_uvs[i_bl * comps.num_curved_power_laws + i_comp] * one_on_lambda;
            int num_coeffs = comps.curved_power_law_num_shapelet_coeffs[i_comp];
            COMPLEX envelope = get_shapelet_envelope(comps.curved_power_law_gps[i_comp], s_uv, num_coeffs,
                                                     shapelet_coeffs, shapelet_basis_values);
            shapelet_coeffs += num_coeffs;

            delta_vis += fd * geom * envelope;
        }

        shapelet_coeffs = comps.list_shapelet_coeffs;
        for (int i_comp = 0; i_comp < comps.num_lists; i_comp++) {
            JONES fd = comps.list_fds[i_freq * comps.num_lists + i_comp];
            apply_beam(j1++, &fd, j2++);

            const LmnRime lmn = comps.list_lmns[i_comp];
            SINCOS(uvw.u * lmn.l + uvw.v * lmn.m + uvw.w * lmn.n, &geom.y, &geom.x);

            ShapeletUV s_uv = comps.list_shapelet_uvs[i_bl * comps.num_lists + i_comp] * one_on_lambda;
            int num_coeffs = comps.list_num_shapelet_coeffs[i_comp];
            COMPLEX envelope =
                get_shapelet_envelope(comps.list_gps[i_comp], s_uv, num_coeffs, shapelet_coeffs, shapelet_basis_values);
            shapelet_coeffs += num_coeffs;

            delta_vis += fd * geom * envelope;
        }

        vis_fb[i_freq * num_baselines + i_bl] += delta_vis;
    }
}

extern "C" const char *model_points(const Points *comps, const Addresses *a, const UVW *d_uvws,
                                    const JONES *d_beam_jones, JonesF32 *d_vis_fb) {
    dim3 gridDim, blockDim;
    // Thread blocks are distributed by cross-correlation baseline.
    blockDim.x = NUM_THREADS_PER_BLOCK_POINTS;
    gridDim.x = (int)ceil((double)(a->num_baselines * a->num_freqs) / (double)blockDim.x);

    model_points_kernel<<<gridDim, blockDim>>>(a->num_freqs, a->num_baselines, a->d_freqs, d_uvws, *comps, d_beam_jones,
                                               a->d_tile_map, a->d_freq_map, a->num_unique_beam_freqs,
                                               a->d_tile_index_to_unflagged_tile_index_map, d_vis_fb);

    cudaError_t error_id;
#ifdef DEBUG
    error_id = cudaDeviceSynchronize();
    if (error_id != cudaSuccess) {
        return cudaGetErrorString(error_id);
    }
#endif
    error_id = cudaGetLastError();
    if (error_id != cudaSuccess) {
        return cudaGetErrorString(error_id);
    }

    return NULL;
}

extern "C" const char *model_gaussians(const Gaussians *comps, const Addresses *a, const UVW *d_uvws,
                                       const JONES *d_beam_jones, JonesF32 *d_vis_fb) {
    dim3 gridDim, blockDim;
    // Thread blocks are distributed by cross-correlation baseline.
    blockDim.x = NUM_THREADS_PER_BLOCK_GAUSSIANS;
    gridDim.x = (int)ceil((double)(a->num_baselines * a->num_freqs) / (double)blockDim.x);

    model_gaussians_kernel<<<gridDim, blockDim>>>(a->num_freqs, a->num_baselines, a->d_freqs, d_uvws, *comps,
                                                  d_beam_jones, a->d_tile_map, a->d_freq_map, a->num_unique_beam_freqs,
                                                  a->d_tile_index_to_unflagged_tile_index_map, d_vis_fb);

    cudaError_t error_id;
#ifdef DEBUG
    error_id = cudaDeviceSynchronize();
    if (error_id != cudaSuccess) {
        return cudaGetErrorString(error_id);
    }
#endif
    error_id = cudaGetLastError();
    if (error_id != cudaSuccess) {
        return cudaGetErrorString(error_id);
    }

    return NULL;
}

extern "C" const char *model_shapelets(const Shapelets *comps, const Addresses *a, const UVW *d_uvws,
                                       const JONES *d_beam_jones, JonesF32 *d_vis_fb) {
    dim3 gridDim, blockDim;
    // Thread blocks are distributed by cross-correlation baseline.
    blockDim.x = NUM_THREADS_PER_BLOCK_SHAPELETS;
    gridDim.x = (int)ceil((double)(a->num_baselines * a->num_freqs) / (double)blockDim.x);

    model_shapelets_kernel<<<gridDim, blockDim>>>(
        a->num_freqs, a->num_baselines, a->d_freqs, d_uvws, *comps, a->d_shapelet_basis_values, d_beam_jones,
        a->d_tile_map, a->d_freq_map, a->num_unique_beam_freqs, a->d_tile_index_to_unflagged_tile_index_map, d_vis_fb);

    cudaError_t error_id;
#ifdef DEBUG
    error_id = cudaDeviceSynchronize();
    if (error_id != cudaSuccess) {
        return cudaGetErrorString(error_id);
    }
#endif
    error_id = cudaGetLastError();
    if (error_id != cudaSuccess) {
        return cudaGetErrorString(error_id);
    }

    return NULL;
}
