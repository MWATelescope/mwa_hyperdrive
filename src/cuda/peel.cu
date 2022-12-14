// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <stdlib.h>

#include <cuComplex.h>

#include "common.cuh"
#include "peel.h"

/**
 * Turn XYZs into UVWs. Multiple sets of XYZs over time can be converted.
 * Expects the device to be parallel over baselines.
 */
__global__ void xyzs_to_uvws_kernel(const XYZ *xyzs, const FLOAT *lmsts, UVW *uvws, RADec pointing_centre,
                                    int num_tiles, int num_baselines, int num_timesteps) {
    const int i_bl = threadIdx.x + (blockDim.x * blockIdx.x);
    if (i_bl >= num_baselines)
        return;

    const float n = (float)(num_tiles - 1);
    const float tile1f = floorf(-0.5 * sqrtf(4.0 * n * (n + 1.0) - 8.0 * i_bl + 1.0) + n + 0.5);
    const int tile2 = (int)(i_bl - tile1f * (n - (tile1f + 1.0) / 2.0) + 1.0);
    const int tile1 = (int)tile1f;

    FLOAT s_ha, c_ha, s_dec, c_dec;
    SINCOS(pointing_centre.dec, &s_dec, &c_dec);

    for (int i_time = 0; i_time < num_timesteps; i_time++) {
        XYZ xyz = xyzs[i_time * num_tiles + tile1];
        const XYZ xyz2 = xyzs[i_time * num_tiles + tile2];
        xyz.x -= xyz2.x;
        xyz.y -= xyz2.y;
        xyz.z -= xyz2.z;

        const FLOAT hour_angle = lmsts[i_time] - pointing_centre.ra;
        SINCOS(hour_angle, &s_ha, &c_ha);

        uvws[i_time * num_baselines + (int)i_bl] = UVW{
            .u = s_ha * xyz.x + c_ha * xyz.y,
            .v = -s_dec * c_ha * xyz.x + s_dec * s_ha * xyz.y + c_dec * xyz.z,
            .w = c_dec * c_ha * xyz.x - c_dec * s_ha * xyz.y + s_dec * xyz.z,
        };
    }
}

/**
 * Kernel for rotating visibilities and averaging them into "low-resolution"
 * visibilities.
 *
 * The visibilities should be ordered in time, frequency and baseline (slowest
 * to fastest). The weights should never be negative; this allows us to avoid
 * special logic when averaging.
 */
__global__ void rotate_average_kernel(const JONES *high_res_vis, const float *high_res_weights, JONES *low_res_vis,
                                      RADec pointing_centre, const int num_timesteps, const int num_tiles,
                                      const int num_baselines, const int num_freqs, const int freq_average_factor,
                                      const FLOAT *lmsts, const XYZ *xyzs, const UVW *uvws_from, UVW *uvws_to,
                                      const FLOAT *lambdas) {
    const int i_bl = threadIdx.x + (blockDim.x * blockIdx.x);
    if (i_bl >= num_baselines)
        return;

    // Prepare an "argument" for later.
    const FLOAT arg = -TAU * (uvws_to[i_bl].w - uvws_from[i_bl].w);

    for (int i_freq = 0; i_freq < num_freqs; i_freq += freq_average_factor) {
        JONES vis_weighted_sum = JONES{
            .j00_re = 0.0,
            .j00_im = 0.0,
            .j01_re = 0.0,
            .j01_im = 0.0,
            .j10_re = 0.0,
            .j10_im = 0.0,
            .j11_re = 0.0,
            .j11_im = 0.0,
        };
        FLOAT weight_sum = 0.0;

        for (int i_time = 0; i_time < num_timesteps; i_time++) {
            for (int i_freq_chunk = i_freq; i_freq_chunk < i_freq + freq_average_factor; i_freq_chunk++) {
                COMPLEX complex;
                SINCOS(arg / lambdas[i_freq_chunk], &complex.y, &complex.x);

                const int step = (i_time * num_freqs + i_freq_chunk) * num_baselines + i_bl;
                const FLOAT weight = high_res_weights[step];
                const JONES rotated_weighted_vis = high_res_vis[step] * weight * complex;

                vis_weighted_sum += rotated_weighted_vis;
                weight_sum += weight;
            }
        }

        // If `weight_sum` is bigger than 0, use it in division, otherwise just
        // divide by 1. We do this so we don't get NaN values, and we don't use
        // if statements in case the compiler optimises this better to avoid
        // warp divergence.
        vis_weighted_sum /= (weight_sum > 0.0) ? weight_sum : 1.0;

        const int low_res_step = (i_freq / freq_average_factor) * num_baselines + i_bl;
        low_res_vis[low_res_step] = vis_weighted_sum;
        // low_res_weights[low_res_step] = weight_sum;
    }
}

/**
 *
 */
__device__ void apply_iono(const JonesF32 *vis, JONES *vis_out, const FLOAT iono_const_alpha,
                           const FLOAT iono_const_beta, const int num_baselines, const int num_freqs, const UVW *uvws,
                           const FLOAT *lambdas_m) {
    const int i_bl = threadIdx.x + (blockDim.x * blockIdx.x);
    // No need to check if this thread should continue; this is a device
    // function.

    const UVW uvw = uvws[i_bl];
    const FLOAT arg = -TAU * (uvw.u * iono_const_alpha + uvw.v * iono_const_beta);

    for (int i_freq = 0; i_freq < num_freqs; i_freq++) {
        COMPLEX complex;
        // The baseline UV is in units of metres, so we need to divide by λ to
        // use it in an exponential. But we're also multiplying by λ², so just
        // multiply by λ.
        SINCOS(arg * lambdas_m[i_freq], &complex.y, &complex.x);

        const int step = i_freq * num_baselines + i_bl;
        // TODO: Yuck
        const int step2 = i_bl * num_freqs + i_freq;
        const JonesF32 asdf = vis[step2] * complex;
        vis_out[step] = JONES{
            .j00_re = (FLOAT)asdf.j00_re,
            .j00_im = (FLOAT)asdf.j00_im,
            .j01_re = (FLOAT)asdf.j01_re,
            .j01_im = (FLOAT)asdf.j01_im,
            .j10_re = (FLOAT)asdf.j10_re,
            .j10_im = (FLOAT)asdf.j10_im,
            .j11_re = (FLOAT)asdf.j11_re,
            .j11_im = (FLOAT)asdf.j11_im,
        };
    }
}

// https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf
// template <unsigned int blockSize> __device__ void warpReduce(volatile JonesF64 *sdata, unsigned int tid) {
__device__ void warpReduce(volatile JonesF64 *sdata, unsigned int tid) {
    sdata[tid] += sdata[tid + 32];
    sdata[tid] += sdata[tid + 16];
    sdata[tid] += sdata[tid + 8];
    sdata[tid] += sdata[tid + 4];
    sdata[tid] += sdata[tid + 2];
    sdata[tid] += sdata[tid + 1];
}
// template <unsigned int blockSize> __global__ void reduce_jones(JonesF64 *data, const int n) {
//     extern __shared__ JonesF64 sdata[];
//     unsigned int tid = threadIdx.x;
//     unsigned int i = blockIdx.x * (blockSize * 2) + tid;
//     unsigned int gridSize = blockSize * 2 * gridDim.x;
//     sdata[tid] = JonesF64{
//         .j00_re = 0.0,
//         .j00_im = 0.0,
//         .j01_re = 0.0,
//         .j01_im = 0.0,
//         .j10_re = 0.0,
//         .j10_im = 0.0,
//         .j11_re = 0.0,
//         .j11_im = 0.0,
//     };

//     while (i < n + gridSize) {
//         sdata[tid] += data[i] + data[i + blockSize];
//         i += gridSize;
//     }
//     __syncthreads();
//     if (blockSize >= 512) {
//         if (tid < 256) {
//             sdata[tid] += sdata[tid + 256];
//         }
//         __syncthreads();
//     }
//     if (blockSize >= 256) {
//         if (tid < 128) {
//             sdata[tid] += sdata[tid + 128];
//         }
//         __syncthreads();
//     }
//     if (blockSize >= 128) {
//         if (tid < 64) {
//             sdata[tid] += sdata[tid + 64];
//         }
//         __syncthreads();
//     }
//     if (tid < 32)
//         warpReduce<blockSize>(sdata, tid);
//     if (tid == 0)
//         data[blockIdx.x] = sdata[0];
//     if (tid == 0) {
//         printf("reduce %f %d\n", sdata[0].j11_im, blockIdx.x);
//     }
// }

// For each frequency, add up all of the data across baselines, dumping the
// results back into the data according to block index.
__global__ void reduce_jones(JonesF64 *data, const int num_baselines) {
    extern __shared__ JonesF64 sdata[];
    const int tid = threadIdx.x;
    const int offset = blockIdx.x * num_baselines;

    sdata[tid] = JonesF64{
        .j00_re = 0.0,
        .j00_im = 0.0,
        .j01_re = 0.0,
        .j01_im = 0.0,
        .j10_re = 0.0,
        .j10_im = 0.0,
        .j11_re = 0.0,
        .j11_im = 0.0,
    };
    int i = tid;
    while (i + blockDim.x < num_baselines) {
        sdata[tid] += data[i + offset] + data[i + offset + blockDim.x];
        i += 2 * blockDim.x;
    }
    __syncthreads();
    // Get the last few values.
    if (i < num_baselines) {
        sdata[tid] += data[i + offset];
    }
    __syncthreads();

    if (blockDim.x >= 512) {
        if (tid < 256) {
            sdata[tid] += sdata[tid + 256];
        }
        __syncthreads();
    }
    if (blockDim.x >= 256) {
        if (tid < 128) {
            sdata[tid] += sdata[tid + 128];
        }
        __syncthreads();
    }
    if (blockDim.x >= 128) {
        if (tid < 64) {
            sdata[tid] += sdata[tid + 64];
        }
        __syncthreads();
    }

    if (tid < 32) {
        warpReduce(sdata, tid);
    }
    if (tid == 0) {
        data[blockIdx.x] = sdata[0];
        // printf("reduce %f %d\n", sdata[0].j11_im, blockIdx.x);
        // printf("");
    }
}

// This is designed to take only a single block and add everything.
__global__ void reduce_jones2(JonesF64 *data, const int n, double *iono_consts) {
    extern __shared__ JonesF64 sdata[];
    int tid = threadIdx.x;

    if (tid < n) {
        sdata[tid] = data[tid];
    }
    __syncthreads();
    for (int i = n - 1; i > 0; i -= 1) {
        if (tid == i) {
            sdata[tid - 1] += sdata[tid];
        }
        __syncthreads();
    }
    if (tid == 0) {
        const double a_uu = sdata[0].j00_re;
        const double a_uv = sdata[0].j00_im;
        const double a_vv = sdata[0].j01_re;
        const double aa_u = sdata[0].j01_im;
        const double aa_v = sdata[0].j10_re;
        // const double s_vm = sdata[0].j10_im;
        // const double s_mm = sdata[0].j11_re;
        // printf("reduce2 %f\n", sdata[0].j11_im);
        const double denom = TAU * (a_uu * a_vv - a_uv * a_uv);
        iono_consts[0] += (aa_u * a_vv - aa_v * a_uv) / denom;
        iono_consts[1] += (aa_v * a_uu - aa_u * a_uv) / denom;
    }
}

/**
 * Kernel for ...
 */
__global__ void iono_loop_kernel(const JONES *vis_residual, const float *vis_weights, const JonesF32 *vis_model,
                                 JONES *vis_model_rotated, const double *iono_consts, JonesF64 *iono_fits,
                                 const int num_iterations, const int num_baselines, const int num_freqs,
                                 const FLOAT *lmsts, const UVW *uvws, const FLOAT *lambdas_m) {
    const int i_bl = threadIdx.x + (blockDim.x * blockIdx.x);
    if (i_bl >= num_baselines)
        return;

    const UVW uvw = uvws[i_bl];

    // Apply the latest iono constants to the model visibilities.
    const double iono_const_alpha = iono_consts[0];
    const double iono_const_beta = iono_consts[1];

    // TODO: Would it be better to avoid the function call?
    // TODO: Use the updated source position for the UVWs?
    apply_iono(vis_model, vis_model_rotated, iono_const_alpha, iono_const_beta, num_baselines, num_freqs, uvws,
               lambdas_m);

    for (int i_freq = 0; i_freq < num_freqs; i_freq++) {
        const double lambda = lambdas_m[i_freq];
        const double lambda_2 = lambda * lambda;

        const double u = (double)uvw.u;
        const double v = (double)uvw.v;

        const int step = i_freq * num_baselines + i_bl;
        const double weight = vis_weights[step];
        const JONES *residual = &vis_residual[step];
        const double residual_i_re = residual->j00_re + residual->j11_re;
        const double residual_i_im = residual->j00_im + residual->j11_im;
        const JONES *model = &vis_model_rotated[step];
        const double model_i_re = model->j00_re + model->j11_re;
        const double model_i_im = model->j00_im + model->j11_im;

        const double mr = model_i_re * (residual_i_im - model_i_im);
        const double mm = model_i_re * model_i_re;

        JonesF64 j = JonesF64{
            .j00_re = lambda_2 * weight * mm * u * u,      // a_uu
            .j00_im = lambda_2 * weight * mm * u * v,      // a_uv
            .j01_re = lambda_2 * weight * mm * v * v,      // a_vv
            .j01_im = -lambda * weight * mr * u,           // aa_u
            .j10_re = -lambda * weight * mr * v,           // aa_v
            .j10_im = weight * model_i_re * residual_i_re, // s_vm
            .j11_re = weight * mm,                         // s_mm
            .j11_im = 1.0,
        };
        iono_fits[step] = j;
    }
}

__global__ void subtract_iono_kernel(JONES *vis_residual, const JonesF32 *vis_model, const double iono_const_alpha,
                                     const double iono_const_beta, const UVW *uvws, const FLOAT *lambdas_m,
                                     const int num_timesteps, const int num_baselines, const int num_freqs) {
    const int i_bl = threadIdx.x + (blockDim.x * blockIdx.x);
    if (i_bl >= num_baselines)
        return;

    for (int i_time = 0; i_time < num_timesteps; i_time++) {
        const UVW uvw = uvws[i_time * num_baselines + i_bl];
        const FLOAT arg = -TAU * (uvw.u * iono_const_alpha + uvw.v * iono_const_beta);
        for (int i_freq = 0; i_freq < num_freqs; i_freq++) {
            const FLOAT lambda = lambdas_m[i_freq];

            COMPLEX complex;
            // The baseline UV is in units of metres, so we need to divide by λ to
            // use it in an exponential. But we're also multiplying by λ², so just
            // multiply by λ.
            SINCOS(arg * lambda, &complex.y, &complex.x);

            const int step = (i_time * num_freqs + i_freq) * num_baselines + i_bl;
            // TODO: Yuck
            const int step2 = (i_time * num_baselines + i_bl) * num_freqs + i_freq;
            JONES r = vis_residual[step];
            const JonesF32 m = vis_model[step2];

            r += m;
            r -= m * complex;
            vis_residual[step] = r;
        }
    }
}

/* Host functions */

extern "C" int xyzs_to_uvws(const XYZ *d_xyzs, const FLOAT *d_lmsts, UVW *d_uvws, RADec pointing_centre, int num_tiles,
                            int num_baselines, int num_timesteps) {
    dim3 gridDim, blockDim;
    // Thread blocks are distributed by baseline indices.
    blockDim.x = 256;
    blockDim.y = 1;
    blockDim.z = 1;
    gridDim.x = (int)ceil((double)num_baselines / (double)blockDim.x);
    gridDim.y = 1;
    gridDim.z = 1;

    xyzs_to_uvws_kernel<<<gridDim, blockDim>>>(d_xyzs, d_lmsts, d_uvws, pointing_centre, num_tiles, num_baselines,
                                               num_timesteps);
    cudaCheck(cudaPeekAtLastError());

    return 0;
}

extern "C" int rotate_average(const JONES *d_high_res_vis, const float *d_high_res_weights, JONES *d_low_res_vis,
                              RADec pointing_centre, const int num_timesteps, const int num_tiles,
                              const int num_baselines, const int num_freqs, const int freq_average_factor,
                              const FLOAT *d_lmsts, const XYZ *d_xyzs, const UVW *d_uvws_from, UVW *d_uvws_to,
                              const FLOAT *d_lambdas) {
    dim3 gridDim, blockDim;
    // Thread blocks are distributed by baseline indices.
    blockDim.x = 256;
    blockDim.y = 1;
    blockDim.z = 1;
    gridDim.x = (int)ceil((double)num_baselines / (double)blockDim.x);
    gridDim.y = 1;
    gridDim.z = 1;

    // Prepare the "to" UVWs.
    xyzs_to_uvws_kernel<<<gridDim, blockDim>>>(d_xyzs, d_lmsts, d_uvws_to, pointing_centre, num_tiles, num_baselines,
                                               num_timesteps);
    rotate_average_kernel<<<gridDim, blockDim>>>(
        d_high_res_vis, d_high_res_weights, d_low_res_vis, pointing_centre, num_timesteps, num_tiles, num_baselines,
        num_freqs, freq_average_factor, d_lmsts, d_xyzs, d_uvws_from, d_uvws_to, d_lambdas);
    cudaCheck(cudaPeekAtLastError());

    return 0;
}

extern "C" int iono_loop(const JONES *d_vis_residual, const float *d_vis_weights, const JonesF32 *d_vis_model,
                         JONES *d_vis_model_rotated, JonesF64 *d_iono_fits, double *iono_const_alpha,
                         double *iono_const_beta, const int num_timesteps, const int num_tiles, const int num_baselines,
                         const int num_freqs, const int num_iterations, const FLOAT *d_lmsts, const UVW *d_uvws,
                         const FLOAT *d_lambdas_m) {
    // Thread blocks are distributed by baseline indices.
    dim3 gridDim, blockDim;
    blockDim.x = 256;
    blockDim.y = 1;
    blockDim.z = 1;
    gridDim.x = (int)ceil((double)num_baselines / (double)blockDim.x);
    gridDim.y = 1;
    gridDim.z = 1;
    // These are used to do ionospheric fit adding.
    dim3 gridDimAdd, blockDimAdd;
    blockDimAdd.x = 256;
    blockDimAdd.y = 1;
    blockDimAdd.z = 1;
    gridDimAdd.x = num_freqs;
    gridDimAdd.y = 1;
    gridDimAdd.z = 1;
    // And one final ionospheric fit adding.
    dim3 gridDimAdd2, blockDimAdd2;
    blockDimAdd2.x = num_freqs;
    blockDimAdd2.y = 1;
    blockDimAdd2.z = 1;
    gridDimAdd2.x = 1;
    gridDimAdd2.y = 1;
    gridDimAdd2.z = 1;

    double *d_iono_consts;
    cudaMalloc(&d_iono_consts, 2 * sizeof(double));
    cudaMemcpy(d_iono_consts, iono_const_alpha, sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_iono_consts + 1, iono_const_beta, sizeof(double), cudaMemcpyHostToDevice);

    for (int iteration = 0; iteration < num_iterations; iteration++) {
        // Do the work for one loop of the iteration.
        iono_loop_kernel<<<gridDim, blockDim>>>(d_vis_residual, d_vis_weights, d_vis_model, d_vis_model_rotated,
                                                d_iono_consts, d_iono_fits, num_tiles, num_baselines, num_freqs,
                                                d_lmsts, d_uvws, d_lambdas_m);
        cudaCheck(cudaPeekAtLastError());
        // Sum the iono fits.
        reduce_jones<<<gridDimAdd, blockDimAdd, blockDimAdd.x * sizeof(JonesF64)>>>(d_iono_fits, num_baselines);
        cudaCheck(cudaPeekAtLastError());
        reduce_jones2<<<gridDimAdd2, blockDimAdd2, num_freqs * sizeof(JonesF64)>>>(d_iono_fits, num_freqs,
                                                                                   d_iono_consts);
        cudaCheck(cudaPeekAtLastError());

        // // Sane?
        // printf("iter %d\n", iteration);
        // cudaMemcpy(iono_const_alpha, d_iono_consts, sizeof(double), cudaMemcpyDeviceToHost);
        // cudaMemcpy(iono_const_beta, d_iono_consts + 1, sizeof(double), cudaMemcpyDeviceToHost);
        // printf("%.4e %.4e\n", *iono_const_alpha, *iono_const_beta);
    }

    cudaMemcpy(iono_const_alpha, d_iono_consts, sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(iono_const_beta, d_iono_consts + 1, sizeof(double), cudaMemcpyDeviceToHost);
    cudaFree(d_iono_consts);
    // printf("%.4e %.4e\n", *iono_const_alpha, *iono_const_beta);

    return 0;
}

extern "C" int subtract_iono(JONES *d_vis_residual, const JonesF32 *d_vis_model, double iono_const_alpha,
                             double iono_const_beta, const UVW *d_uvws, const FLOAT *d_lambdas_m,
                             const int num_timesteps, const int num_baselines, const int num_freqs) {
    // Thread blocks are distributed by baseline indices.
    dim3 gridDim, blockDim;
    blockDim.x = 256;
    blockDim.y = 1;
    blockDim.z = 1;
    gridDim.x = (int)ceil((double)num_baselines / (double)blockDim.x);
    gridDim.y = 1;
    gridDim.z = 1;

    subtract_iono_kernel<<<gridDim, blockDim>>>(d_vis_residual, d_vis_model, iono_const_alpha, iono_const_beta, d_uvws,
                                                d_lambdas_m, num_timesteps, num_baselines, num_freqs);

    return 0;
}
