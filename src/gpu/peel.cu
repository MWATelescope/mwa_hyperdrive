// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <stdlib.h>

#include "common.cuh"
#include "peel.h"

/**
 * Turn XYZs into UVWs. Multiple sets of XYZs over time can be converted.
 * Expects the device to be parallel over baselines.
 * xyzs has shape (num_timesteps, num_tiles).
 * lmsts has shape (num_timesteps).
 * uvws has shape (num_timesteps, num_baselines).
 */
__global__ void xyzs_to_uvws_kernel(const XYZ *xyzs, const FLOAT *lmsts, UVW *uvws, RADec pointing_centre,
                                    int num_tiles, int num_baselines, int num_timesteps)
{
    const int i_bl = threadIdx.x + (blockDim.x * blockIdx.x);
    if (i_bl >= num_baselines)
        return;

    // Find the tile indices from the baseline index. `num_tiles` has to be
    // subtracted by 1 to make it "0 index".
    const float n = (float)(num_tiles - 1);
    const float tile1f = floorf(-0.5 * sqrtf(4.0 * n * (n + 1.0) - 8.0 * i_bl + 1.0) + n + 0.5);
    const int tile2 = (int)(i_bl - tile1f * (n - (tile1f + 1.0) / 2.0) + 1.0);
    const int tile1 = (int)tile1f;

    FLOAT s_ha, c_ha, s_dec, c_dec;
    SINCOS(pointing_centre.dec, &s_dec, &c_dec);

    for (int i_time = 0; i_time < num_timesteps; i_time++)
    {
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
 * Kernel for rotating visibilities.
 *
 * The visibilities should be ordered in time, frequency and baseline (slowest
 * to fastest).
 */
__global__ void rotate_kernel(JonesF32 *vis_fb, const int num_timesteps, const int num_baselines, const int num_freqs,
                              const UVW *uvws_from, const UVW *uvws_to, const FLOAT *lambdas)
{
    const int i_bl = threadIdx.x + (blockDim.x * blockIdx.x);
    if (i_bl >= num_baselines)
        return;

    for (int i_freq = 0; i_freq < num_freqs; i_freq += 1)
    {
        for (int i_time = 0; i_time < num_timesteps; i_time++)
        {
            // Prepare an "argument" for later.
            const FLOAT arg = -TAU * ((double)uvws_to[i_time * num_baselines + i_bl].w -
                                      (double)uvws_from[i_time * num_baselines + i_bl].w);
            COMPLEX complex;
            SINCOS(arg / lambdas[i_freq], &complex.y, &complex.x);

            const int step = (i_time * num_freqs + i_freq) * num_baselines + i_bl;
#ifdef SINGLE
            vis_fb[step] *= complex;
#else
            const JonesF32 vis_single = vis_fb[step];
            JonesF64 vis_double = JonesF64{
                .j00_re = vis_single.j00_re,
                .j00_im = vis_single.j00_im,
                .j01_re = vis_single.j01_re,
                .j01_im = vis_single.j01_im,
                .j10_re = vis_single.j10_re,
                .j10_im = vis_single.j10_im,
                .j11_re = vis_single.j11_re,
                .j11_im = vis_single.j11_im,
            };
            vis_double *= complex;
            vis_fb[step] = JonesF32{
                .j00_re = (float)vis_double.j00_re,
                .j00_im = (float)vis_double.j00_im,
                .j01_re = (float)vis_double.j01_re,
                .j01_im = (float)vis_double.j01_im,
                .j10_re = (float)vis_double.j10_re,
                .j10_im = (float)vis_double.j10_im,
                .j11_re = (float)vis_double.j11_re,
                .j11_im = (float)vis_double.j11_im,
            };
#endif
        }
    }
}

/**
 * Kernel for averaging visibilities.
 *
 * The visibilities should be ordered in time, frequency and baseline (slowest
 * to fastest). The weights should never be negative; this allows us to avoid
 * special logic when averaging.
 */
__global__ void average_kernel(const JonesF32 *high_res_vis, const float *high_res_weights, JonesF32 *low_res_vis,
                               const int num_timesteps, const int num_baselines, const int num_freqs,
                               const int freq_average_factor)
{
    const int i_bl = threadIdx.x + (blockDim.x * blockIdx.x);
    if (i_bl >= num_baselines)
        return;

    for (int i_freq = 0; i_freq < num_freqs; i_freq += freq_average_factor)
    {
        JonesF64 vis_weighted_sum = JonesF64{
            .j00_re = 0.0,
            .j00_im = 0.0,
            .j01_re = 0.0,
            .j01_im = 0.0,
            .j10_re = 0.0,
            .j10_im = 0.0,
            .j11_re = 0.0,
            .j11_im = 0.0,
        };
        double weight_sum = 0.0;

        for (int i_time = 0; i_time < num_timesteps; i_time++)
        {
            for (int i_freq_chunk = i_freq; i_freq_chunk < i_freq + freq_average_factor; i_freq_chunk++)
            {
                const int step = (i_time * num_freqs + i_freq_chunk) * num_baselines + i_bl;
                const double weight = high_res_weights[step];
                const JonesF32 vis_single = high_res_vis[step];
                const JonesF64 vis_double = JonesF64{
                    .j00_re = vis_single.j00_re,
                    .j00_im = vis_single.j00_im,
                    .j01_re = vis_single.j01_re,
                    .j01_im = vis_single.j01_im,
                    .j10_re = vis_single.j10_re,
                    .j10_im = vis_single.j10_im,
                    .j11_re = vis_single.j11_re,
                    .j11_im = vis_single.j11_im,
                };
                const JonesF64 rotated_weighted_vis = vis_double * weight;

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
        low_res_vis[low_res_step] = JonesF32{
            .j00_re = (float)vis_weighted_sum.j00_re,
            .j00_im = (float)vis_weighted_sum.j00_im,
            .j01_re = (float)vis_weighted_sum.j01_re,
            .j01_im = (float)vis_weighted_sum.j01_im,
            .j10_re = (float)vis_weighted_sum.j10_re,
            .j10_im = (float)vis_weighted_sum.j10_im,
            .j11_re = (float)vis_weighted_sum.j11_re,
            .j11_im = (float)vis_weighted_sum.j11_im,
        };
        // low_res_weights[low_res_step] = weight_sum;
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
__global__ void rotate_average_kernel(const JonesF32 *high_res_vis, const float *high_res_weights,
                                      JonesF32 *low_res_vis, RADec pointing_centre, const int num_timesteps,
                                      const int num_tiles, const int num_baselines, const int num_freqs,
                                      const int freq_average_factor, const FLOAT *lmsts, const XYZ *xyzs,
                                      const UVW *uvws_from, UVW *uvws_to, const FLOAT *lambdas)
{
    const int i_bl = threadIdx.x + (blockDim.x * blockIdx.x);
    if (i_bl >= num_baselines)
        return;

    for (int i_freq = 0; i_freq < num_freqs; i_freq += freq_average_factor)
    {
        JonesF64 vis_weighted_sum = JonesF64{
            .j00_re = 0.0,
            .j00_im = 0.0,
            .j01_re = 0.0,
            .j01_im = 0.0,
            .j10_re = 0.0,
            .j10_im = 0.0,
            .j11_re = 0.0,
            .j11_im = 0.0,
        };
        double weight_sum = 0.0;

        for (int i_time = 0; i_time < num_timesteps; i_time++)
        {
            // Prepare an "argument" for later.
            const double arg = -TAU * ((double)uvws_to[i_time * num_baselines + i_bl].w -
                                       (double)uvws_from[i_time * num_baselines + i_bl].w);
            for (int i_freq_chunk = i_freq; i_freq_chunk < i_freq + freq_average_factor; i_freq_chunk++)
            {
                C64 complex;
                sincos(arg / lambdas[i_freq_chunk], &complex.y, &complex.x);

                const int step = (i_time * num_freqs + i_freq_chunk) * num_baselines + i_bl;
                const double weight = high_res_weights[step];
                const JonesF32 vis_single = high_res_vis[step];
                const JonesF64 vis_double = JonesF64{
                    .j00_re = vis_single.j00_re,
                    .j00_im = vis_single.j00_im,
                    .j01_re = vis_single.j01_re,
                    .j01_im = vis_single.j01_im,
                    .j10_re = vis_single.j10_re,
                    .j10_im = vis_single.j10_im,
                    .j11_re = vis_single.j11_re,
                    .j11_im = vis_single.j11_im,
                };
                const JonesF64 rotated_weighted_vis = vis_double * weight * complex;

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
        low_res_vis[low_res_step] = JonesF32{
            .j00_re = (float)vis_weighted_sum.j00_re,
            .j00_im = (float)vis_weighted_sum.j00_im,
            .j01_re = (float)vis_weighted_sum.j01_re,
            .j01_im = (float)vis_weighted_sum.j01_im,
            .j10_re = (float)vis_weighted_sum.j10_re,
            .j10_im = (float)vis_weighted_sum.j10_im,
            .j11_re = (float)vis_weighted_sum.j11_re,
            .j11_im = (float)vis_weighted_sum.j11_im,
        };
        // low_res_weights[low_res_step] = weight_sum;
    }
}

/**
 *
 */
__device__ void apply_iono(const JonesF32 *vis, JonesF32 *vis_out, const IonoConsts *iono_consts,
                           const int num_baselines, const int num_freqs, const UVW *uvws,
                           const FLOAT *lambdas_m)
{
    const int i_bl = threadIdx.x + (blockDim.x * blockIdx.x);
    // No need to check if this thread should continue; this is a device
    // function.

    const UVW uvw = uvws[i_bl];
    const FLOAT arg = -TAU * (uvw.u * iono_consts->alpha + uvw.v * iono_consts->beta);

    for (int i_freq = 0; i_freq < num_freqs; i_freq++)
    {
        COMPLEX complex;
        // The baseline UV is in units of metres, so we need to divide by λ to
        // use it in an exponential. But we're also multiplying by λ², so just
        // multiply by λ.
        SINCOS(arg * lambdas_m[i_freq], &complex.y, &complex.x);

        const int step = i_freq * num_baselines + i_bl;
        // vis_out[step] = vis[step] * complex;
        vis_out[step] = vis[step] * (complex * iono_consts->gain);
        // vis_out[step] = vis[step] * (MAKE_COMPLEX(complex.x / iono_consts->gain, complex.y / iono_consts->gain));
    }
}

/**
 * This is extremely difficult to explain in a comment. This code is derived
 * from this presentation
 * (https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf), but
 * I've had to add `__syncwarp` calls otherwise the sum is incorrect.
 */
template <int BLOCK_SIZE>
__device__ void warp_reduce(volatile JonesF64 *sdata, int tid)
{
    if (BLOCK_SIZE >= 64)
    {
        sdata[tid] += sdata[tid + 32];
        __syncwarp();
    }
    if (BLOCK_SIZE >= 32)
    {
        sdata[tid] += sdata[tid + 16];
        __syncwarp();
    }
    if (BLOCK_SIZE >= 16)
    {
        sdata[tid] += sdata[tid + 8];
        __syncwarp();
    }
    if (BLOCK_SIZE >= 8)
    {
        sdata[tid] += sdata[tid + 4];
        __syncwarp();
    }
    if (BLOCK_SIZE >= 4)
    {
        sdata[tid] += sdata[tid + 2];
        __syncwarp();
    }
    if (BLOCK_SIZE >= 2)
    {
        sdata[tid] += sdata[tid + 1];
    }
}

/**
 * Kernel to add ionospherically-related values (all baselines for a frequency).
 */
template <int BLOCK_SIZE>
__global__ void reduce_baselines(JonesF64 *data, const int num_baselines)
{
    // Every thread has an element of shared memory. This is useful for speeding
    // up accumulation.
    __shared__ JonesF64 sdata[BLOCK_SIZE];
    // tid is "thread ID".
    int tid = threadIdx.x;
    // This thread will start accessing data from this index. It is intended to
    // be targeting a specific frequency (`blockIdx.x`).
    int i = num_baselines * blockIdx.x + tid;

    // Initialise the thread's shared memory.
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

    // Accumulate all the baselines for this frequency. `i` is incremented so
    // that all threads can do coalesced reads.
    while (i < num_baselines * (int)(blockIdx.x + 1))
    {
        sdata[tid] += data[i];
        i += BLOCK_SIZE;
    }
    // The threads may be out of sync because some have a not-too-big index and
    // others have a too-big index. So we sync. The following syncs are done for
    // the same reason.
    __syncthreads();

    if (BLOCK_SIZE >= 512)
    {
        if (tid < 256)
        {
            sdata[tid] += sdata[tid + 256];
        }
        __syncthreads();
    }
    if (BLOCK_SIZE >= 256)
    {
        if (tid < 128)
        {
            sdata[tid] += sdata[tid + 128];
        }
        __syncthreads();
    }
    if (BLOCK_SIZE >= 128)
    {
        if (tid < 64)
        {
            sdata[tid] += sdata[tid + 64];
        }
        __syncthreads();
    }

    // At this point, we need to add the results for the first warp (first 32
    // threads). We no longer need to sync threads because we don't care about
    // thread indices >= 32.
    if (tid < 32)
        warp_reduce<BLOCK_SIZE>(sdata, tid);
    // The first index has the sum; write it out so we can use it later. The
    // index is the frequency.
    if (tid == 0)
        data[blockIdx.x] = sdata[0];
}

/**
 * Kernel to add ionospherically-related values (all frequencies, after all
 * baselines have been added in a per-frequency basis). There should only be one
 * thread block running this kernel.
 */
template <int BLOCK_SIZE>
__global__ void reduce_freqs(
    JonesF64 *data, const FLOAT *lambdas_m, const int num_freqs,
    IonoConsts *iono_consts, const FLOAT convergence)
{
    // Every thread has an element of shared memory. This is useful for speeding
    // up accumulation.
    __shared__ JonesF64 sdata[BLOCK_SIZE];
    // tid is "thread ID".
    int tid = threadIdx.x;

    // Initialise the thread's shared memory.
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

    // Accumulate the per-frequency ionospheric fits. `i_freq` is incremented so
    // that all threads can do coalesced reads.
    for (int i_freq = tid; i_freq < num_freqs; i_freq += BLOCK_SIZE)
    {
        // The data we're accessing here represents all of the ionospheric
        // values for each frequency. These values have not been scaled by λ, so
        // we do that here. When the values were generated, UV were not scaled
        // by λ, so below we use λ² for λ⁴, and λ for λ².
        const double lambda = (double)lambdas_m[i_freq];
        const double lambda_2 = lambda * lambda;

        // Scale the ionospheric values by lambda.
        JonesF64 j = data[i_freq];
        j.j00_re *= lambda_2; // a_uu
        j.j00_im *= lambda_2; // a_uv
        j.j01_re *= lambda_2; // a_vv
        j.j01_im *= -lambda;  // A_u
        j.j10_re *= -lambda;  // A_v

        sdata[tid] += j;
    }
    // The threads may be out of sync because some have a not-too-big index and
    // others have a too-big index. So we sync. The following syncs are done for
    // the same reason.
    __syncthreads();

    if (BLOCK_SIZE >= 512)
    {
        if (tid < 256)
        {
            sdata[tid] += sdata[tid + 256];
        }
        __syncthreads();
    }
    if (BLOCK_SIZE >= 256)
    {
        if (tid < 128)
        {
            sdata[tid] += sdata[tid + 128];
        }
        __syncthreads();
    }
    if (BLOCK_SIZE >= 128)
    {
        if (tid < 64)
        {
            sdata[tid] += sdata[tid + 64];
        }
        __syncthreads();
    }

    if (tid < 32)
        warp_reduce<BLOCK_SIZE>(sdata, tid);
    if (tid == 0)
    {
        const double a_uu = sdata[0].j00_re;
        const double a_uv = sdata[0].j00_im;
        const double a_vv = sdata[0].j01_re;
        const double A_u = sdata[0].j01_im;
        const double A_v = sdata[0].j10_re;
        const double s_vm = sdata[0].j10_im;
        const double s_mm = sdata[0].j11_re;

#ifdef DEBUG
        // Not necessary, but might be useful for checking things.
        data[0] = sdata[0];
#endif

        const double denom = TAU * (a_uu * a_vv - a_uv * a_uv);
        iono_consts->alpha += convergence * (A_u * a_vv - A_v * a_uv) / denom;
        iono_consts->beta += convergence * (A_v * a_uu - A_u * a_uv) / denom;
        // warning: gain is actually needed for model real to line up with vis real.
        iono_consts->gain *= 1. + convergence * (s_vm / s_mm - 1);
        // printf(
        //     "iono: a %+6.4e b %+6.4e a_uu %+6.4e a_uv %+6.4e a_vv %+6.4e A_u %+6.4e A_v %+6.4e denom %+6.4e s_vm %+6.4e s_mm %+6.4e s_vm/s_mm %+6.4e nf %d \n",
        //     iono_consts->alpha, iono_consts->beta,
        //     a_uu, a_uv, a_vv, A_u, A_v, denom,
        //     // iono_consts->gain,
        //     s_vm, s_mm, s_vm / s_mm,
        //     num_freqs
        // );
    }
}

/**
 * Kernel for ...
 */
__global__ void iono_loop_kernel(const JonesF32 *vis_residual, const float *vis_weights, const JonesF32 *vis_model,
                                 JonesF32 *vis_model_rotated, const IonoConsts *iono_consts, JonesF64 *iono_fits,
                                 const int num_baselines, const int num_freqs, const UVW *uvws,
                                 const FLOAT *lambdas_m)
{
    const int i_bl = threadIdx.x + (blockDim.x * blockIdx.x);
    if (i_bl >= num_baselines)
        return;

    // Normally, we would divide by λ to get dimensionless UV. However, UV
    // are only used to determine a_uu, a_uv, a_vv, which are also scaled by
    // lambda. So... don't divide by λ.
    const double u = uvws[i_bl].u;
    const double v = uvws[i_bl].v;

    // Apply the latest iono constants to the model visibilities.
    // TODO: Would it be better to avoid the function call?
    // TODO: Use the updated source position for the UVWs?
    apply_iono(vis_model, vis_model_rotated, iono_consts, num_baselines, num_freqs, uvws,
               lambdas_m);

    for (int i_freq = 0; i_freq < num_freqs; i_freq++)
    {
        const int step = i_freq * num_baselines + i_bl;
        const double weight = (double)vis_weights[step];
        const JonesF32 *residual = vis_residual + step;
        const double residual_i_re = (double)residual->j00_re + (double)residual->j11_re;
        const double residual_i_im = (double)residual->j00_im + (double)residual->j11_im;
        const JonesF32 *model = vis_model_rotated + step;
        const double model_i_re = (double)model->j00_re + (double)model->j11_re;
        const double model_i_im = (double)model->j00_im + (double)model->j11_im;

        const double mr = model_i_re * (residual_i_im - model_i_im);
        const double mm = model_i_re * model_i_re;

        iono_fits[step] = JonesF64{
            // Rather than multiplying by λ here, do it later, when all these
            // values are added together for a single frequency. This means
            // we'll have higher precision overall and fewer multiplies.
            .j00_re = weight * mm * u * u,                 // a_uu
            .j00_im = weight * mm * u * v,                 // a_uv
            .j01_re = weight * mm * v * v,                 // a_vv
            .j01_im = weight * mr * u,                     // A_u
            .j10_re = weight * mr * v,                     // A_v
            .j10_im = weight * model_i_re * residual_i_re, // s_vm
            .j11_re = weight * mm,                         // s_mm
            .j11_im = 1.0,
        };
    }
}

__global__ void subtract_iono_kernel(JonesF32 *vis_residual, const JonesF32 *vis_model, const IonoConsts iono_consts,
                                     const IonoConsts old_iono_consts,
                                     const UVW *uvws, const FLOAT *lambdas_m,
                                     const int num_timesteps, const int num_baselines, const int num_freqs)
{
    const int i_bl = threadIdx.x + (blockDim.x * blockIdx.x);
    if (i_bl >= num_baselines)
        return;

    for (int i_time = 0; i_time < num_timesteps; i_time++)
    {
        const UVW uvw = uvws[i_time * num_baselines + i_bl];
        const FLOAT arg = -TAU * (uvw.u * iono_consts.alpha + uvw.v * iono_consts.beta);
        const FLOAT old_arg = -TAU * (uvw.u * old_iono_consts.alpha + uvw.v * old_iono_consts.beta);
        for (int i_freq = 0; i_freq < num_freqs; i_freq++)
        {
            const FLOAT lambda = lambdas_m[i_freq];

            COMPLEX complex;
            // The baseline UV is in units of metres, so we need to divide by λ to
            // use it in an exponential. But we're also multiplying by λ², so just
            // multiply by λ.
            SINCOS(arg * lambda, &complex.y, &complex.x);
            COMPLEX old_complex;
            SINCOS(old_arg * lambda, &old_complex.y, &old_complex.x);

            const int step = (i_time * num_freqs + i_freq) * num_baselines + i_bl;
            JonesF32 r = vis_residual[step];
            const JonesF32 m = vis_model[step];

            // Promoting the Jones matrices makes things demonstrably more
            // precise.
            JonesF64 r2 = JonesF64{
                .j00_re = r.j00_re,
                .j00_im = r.j00_im,
                .j01_re = r.j01_re,
                .j01_im = r.j01_im,
                .j10_re = r.j10_re,
                .j10_im = r.j10_im,
                .j11_re = r.j11_re,
                .j11_im = r.j11_im,
            };
            JonesF64 m2 = JonesF64{
                .j00_re = m.j00_re,
                .j00_im = m.j00_im,
                .j01_re = m.j01_re,
                .j01_im = m.j01_im,
                .j10_re = m.j10_re,
                .j10_im = m.j10_im,
                .j11_re = m.j11_re,
                .j11_im = m.j11_im,
            };

            r2 += m2 * (old_complex * old_iono_consts.gain);
            r2 -= m2 * (complex * iono_consts.gain);
            // r2 += m2 * MAKE_COMPLEX(old_complex.x - complex.x, old_complex.y - complex.y) * (old_iono_consts.gain / iono_consts.gain);
            // r2 += m2 * MAKE_COMPLEX(old_complex.x - complex.x, old_complex.y - complex.y) * (iono_consts.gain / old_iono_consts.gain);
            vis_residual[step] = JonesF32{
                .j00_re = (float)r2.j00_re,
                .j00_im = (float)r2.j00_im,
                .j01_re = (float)r2.j01_re,
                .j01_im = (float)r2.j01_im,
                .j10_re = (float)r2.j10_re,
                .j10_im = (float)r2.j10_im,
                .j11_re = (float)r2.j11_re,
                .j11_im = (float)r2.j11_im,
            };
        }
    }
}

__global__ void add_model_kernel(JonesF32 *vis_residual, const JonesF32 *vis_model, const IonoConsts iono_consts,
                                 const FLOAT *lambdas_m, const UVW *uvws,
                                 const int num_timesteps, const int num_freqs, const int num_baselines)
{
    const int i_bl = threadIdx.x + (blockDim.x * blockIdx.x);
    if (i_bl >= num_baselines)
        return;

    for (int i_time = 0; i_time < num_timesteps; i_time++)
    {
        const UVW uvw = uvws[i_time * num_baselines + i_bl];
        const FLOAT arg = -TAU * (uvw.u * iono_consts.alpha + uvw.v * iono_consts.beta);

        for (int i_freq = 0; i_freq < num_freqs; i_freq++)
        {
            COMPLEX complex;
            if (iono_consts.alpha == 0.0 && iono_consts.beta == 0.0)
            {
                complex.x = 1.0;
                complex.y = 0.0;
            }
            else
            {
                // The baseline UV is in units of metres, so we need to divide
                // by λ to use it in an exponential. But we're also multiplying
                // by λ², so just multiply by λ.
                SINCOS(arg * lambdas_m[i_freq], &complex.y, &complex.x);
            }

            const int step = (i_time * num_freqs + i_freq) * num_baselines + i_bl;
            vis_residual[step] += vis_model[step] * (complex * iono_consts.gain);
        }
    }
}

/* Host functions */

extern "C" const char *xyzs_to_uvws(const XYZ *d_xyzs, const FLOAT *d_lmsts, UVW *d_uvws, RADec pointing_centre,
                                    int num_tiles, int num_baselines, int num_timesteps)
{
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

    gpuError_t error_id;
#ifdef DEBUG
    error_id = gpuDeviceSynchronize();
    if (error_id != gpuSuccess)
    {
        return gpuGetErrorString(error_id);
    }
#endif
    error_id = gpuGetLastError();
    if (error_id != gpuSuccess)
    {
        return gpuGetErrorString(error_id);
    }

    return NULL;
}

extern "C" const char *rotate(JonesF32 *d_vis_fb, const int num_timesteps, const int num_baselines, const int num_freqs,
                              const UVW *d_uvws_from, const UVW *d_uvws_to, const FLOAT *d_lambdas)
{
    dim3 gridDim, blockDim;
    // Thread blocks are distributed by baseline indices.
    blockDim.x = 256;
    gridDim.x = (int)ceil((double)num_baselines / (double)blockDim.x);

    rotate_kernel<<<gridDim, blockDim>>>(d_vis_fb, num_timesteps, num_baselines, num_freqs, d_uvws_from, d_uvws_to,
                                         d_lambdas);

    gpuError_t error_id;
#ifdef DEBUG
    error_id = gpuDeviceSynchronize();
    if (error_id != gpuSuccess)
    {
        return gpuGetErrorString(error_id);
    }
#endif
    error_id = gpuGetLastError();
    if (error_id != gpuSuccess)
    {
        return gpuGetErrorString(error_id);
    }

    return NULL;
}

extern "C" const char *average(const JonesF32 *d_high_res_vis, const float *d_high_res_weights, JonesF32 *d_low_res_vis,
                               const int num_timesteps, const int num_baselines, const int num_freqs,
                               const int freq_average_factor)
{
    dim3 gridDim, blockDim;
    // Thread blocks are distributed by baseline indices.
    blockDim.x = 256;
    gridDim.x = (int)ceil((double)num_baselines / (double)blockDim.x);

    average_kernel<<<gridDim, blockDim>>>(d_high_res_vis, d_high_res_weights, d_low_res_vis, num_timesteps,
                                          num_baselines, num_freqs, freq_average_factor);
    gpuError_t error_id;
#ifdef DEBUG
    error_id = gpuDeviceSynchronize();
    if (error_id != gpuSuccess)
    {
        return gpuGetErrorString(error_id);
    }
#endif
    error_id = gpuGetLastError();
    if (error_id != gpuSuccess)
    {
        return gpuGetErrorString(error_id);
    }

    return NULL;
}

extern "C" const char *iono_loop(const JonesF32 *d_vis_residual, const float *d_vis_weights,
                                 const JonesF32 *d_vis_model, JonesF32 *d_vis_model_rotated, JonesF64 *d_iono_fits,
                                 IonoConsts *iono_consts,
                                 const int num_baselines, const int num_freqs, const int num_iterations,
                                 const UVW *d_uvws, const FLOAT *d_lambdas_m,
                                 const FLOAT convergence)
{
    // Thread blocks are distributed by baseline indices.
    dim3 gridDim, blockDim;
    blockDim.x = 256;
    gridDim.x = (int)ceil((double)num_baselines / (double)blockDim.x);
    // These are used to do add ionospheric fits (all baselines per frequency).
    dim3 gridDimAdd, blockDimAdd;
    const int NUM_ADD_THREADS = 256;
    blockDimAdd.x = NUM_ADD_THREADS;
    gridDimAdd.x = num_freqs;
    // These are used to accumulate the per-frequency ionospheric fits.
    dim3 gridDimAdd2, blockDimAdd2;
    const int NUM_ADD_THREADS2 = 256;
    blockDimAdd2.x = NUM_ADD_THREADS2;
    gridDimAdd2.x = 1;

    IonoConsts *d_iono_consts;
    gpuMalloc(&d_iono_consts, sizeof(IonoConsts));
    gpuMemcpy(d_iono_consts, iono_consts, sizeof(IonoConsts), gpuMemcpyHostToDevice);

    for (int iteration = 0; iteration < num_iterations; iteration++)
    {
        // Do the work for one loop of the iteration.
        iono_loop_kernel<<<gridDim, blockDim>>>(d_vis_residual, d_vis_weights, d_vis_model, d_vis_model_rotated,
                                                d_iono_consts, d_iono_fits, num_baselines, num_freqs, d_uvws,
                                                d_lambdas_m);

        gpuError_t error_id;
#ifdef DEBUG
        error_id = gpuDeviceSynchronize();
        if (error_id != gpuSuccess)
        {
            return gpuGetErrorString(error_id);
        }
#endif
        error_id = gpuGetLastError();
        if (error_id != gpuSuccess)
        {
            return gpuGetErrorString(error_id);
        }

        // Sum the iono fits.
        reduce_baselines<NUM_ADD_THREADS><<<gridDimAdd, blockDimAdd>>>(d_iono_fits, num_baselines);
#ifdef DEBUG
        error_id = gpuDeviceSynchronize();
        if (error_id != gpuSuccess)
        {
            return gpuGetErrorString(error_id);
        }
#endif
        error_id = gpuGetLastError();
        if (error_id != gpuSuccess)
        {
            return gpuGetErrorString(error_id);
        }

        reduce_freqs<NUM_ADD_THREADS2>
            <<<gridDimAdd2, blockDimAdd2>>>(d_iono_fits, d_lambdas_m, num_freqs, d_iono_consts, convergence);
#ifdef DEBUG
        error_id = gpuDeviceSynchronize();
        if (error_id != gpuSuccess)
        {
            return gpuGetErrorString(error_id);
        }
#endif
        error_id = gpuGetLastError();
        if (error_id != gpuSuccess)
        {
            return gpuGetErrorString(error_id);
        }

        gpuMemcpy(iono_consts, d_iono_consts, sizeof(IonoConsts), gpuMemcpyDeviceToHost);
        if (iono_consts->gain < 0.0)
        {
            break;
        }
    }

    gpuMemcpy(iono_consts, d_iono_consts, sizeof(IonoConsts), gpuMemcpyDeviceToHost);
    gpuFree(d_iono_consts);

    return NULL;
}

extern "C" const char *subtract_iono(JonesF32 *d_vis_residual, const JonesF32 *d_vis_model,
                                     const IonoConsts iono_consts, IonoConsts old_iono_consts, const UVW *d_uvws,
                                     const FLOAT *d_lambdas_m, const int num_timesteps, const int num_baselines,
                                     const int num_freqs)
{
    // Thread blocks are distributed by baseline indices.
    dim3 gridDim, blockDim;
    blockDim.x = 256;
    gridDim.x = (int)ceil((double)num_baselines / (double)blockDim.x);

    subtract_iono_kernel<<<gridDim, blockDim>>>(d_vis_residual, d_vis_model, iono_consts, old_iono_consts, d_uvws,
                                                d_lambdas_m, num_timesteps, num_baselines, num_freqs);

    gpuError_t error_id;
#ifdef DEBUG
    error_id = gpuDeviceSynchronize();
    if (error_id != gpuSuccess)
    {
        return gpuGetErrorString(error_id);
    }
#endif
    error_id = gpuGetLastError();
    if (error_id != gpuSuccess)
    {
        return gpuGetErrorString(error_id);
    }

    return NULL;
}

extern "C" const char *add_model(JonesF32 *d_vis_residual, const JonesF32 *d_vis_model, const IonoConsts iono_consts,
                                 const FLOAT *d_lambdas_m, const UVW *d_uvws, const int num_timesteps,
                                 const int num_freqs, const int num_baselines)
{
    // Thread blocks are distributed by baseline indices.
    dim3 gridDim, blockDim;
    blockDim.x = 256;
    gridDim.x = (int)ceil((double)num_baselines / (double)blockDim.x);

    add_model_kernel<<<gridDim, blockDim>>>(d_vis_residual, d_vis_model, iono_consts, d_lambdas_m,
                                            d_uvws, num_timesteps, num_freqs, num_baselines);

    gpuError_t error_id;
#ifdef DEBUG
    error_id = gpuDeviceSynchronize();
    if (error_id != gpuSuccess)
    {
        return gpuGetErrorString(error_id);
    }
#endif
    error_id = gpuGetLastError();
    if (error_id != gpuSuccess)
    {
        return gpuGetErrorString(error_id);
    }

    return NULL;
}
