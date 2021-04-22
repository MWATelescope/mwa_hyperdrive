// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

// Derived from WODEN (https://github.com/JLBLine/WODEN, commit 854d9c8).

#include <assert.h>
#include <stdio.h>

#include <cuda_runtime.h>

#include "hyperdrive.h"

// `gpu_assert` checks that CUDA code successfully returned.
inline void gpu_assert(cudaError_t code, const char *file, int line, bool abort = true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "gpu_assert: %s (%s:%d)\n", cudaGetErrorString(code), file, line);
        if (abort)
            exit(code);
    }
}
// `cudaCheck` wraps `gpu_assert` for general usage.
#define cudaCheck(code)                                                                                                \
    { gpu_assert((code), __FILE__, __LINE__); }

#ifndef NDEBUG
// When not debugging, `cudaSoftCheck` is a "no-op". Useful for granting full speed in release builds.
#define cudaSoftCheck(code) (code)
#else
#define cudaSoftCheck(code)                                                                                            \
    { gpu_assert((code), __FILE__, __LINE__); }
#endif

/// Kernel for calculating point-source visibilities. Uses `atomicAdd` to fold
/// the values over source components.
__global__ void calc_point_vis(const unsigned int n_baselines, const unsigned int n_points, const unsigned int n_vis,
                               const UVW_c *d_uvw, const LMN_c *d_lmn, const float *d_point_fd, float *d_sum_vis_real,
                               float *d_sum_vis_imag) {
    const int i_vis = threadIdx.x + (blockDim.x * blockIdx.x);
    const int i_comp = threadIdx.y + (blockDim.y * blockIdx.y);

    if ((i_vis >= n_vis) || (i_comp >= n_points))
        return;

    // All present frequencies change every `n_baselines`.
    const int i_freq = i_vis / n_baselines;

    const float u = d_uvw[i_vis].u;
    const float v = d_uvw[i_vis].v;
    const float w = d_uvw[i_vis].w;
    const float l = d_lmn[i_comp].l;
    const float m = d_lmn[i_comp].m;
    const float n = d_lmn[i_comp].n;
    const float flux_density = d_point_fd[i_freq * n_points + i_comp];

    // Calculate -2 * PI * (u * l + v * m + w * (n - 1)). We don't use PI
    // explicitly; CUDA's sincospif does that.
    // Not sure why, but we get an exact match with OSKAR sims and correct
    // location on sky through wsclean without negative in front of 2pi.
    float real;
    float imag;
    sincospif(2 * (u * l + v * m + w * (n - 1.0f)), &imag, &real);

    atomicAdd(&d_sum_vis_real[i_vis], real * flux_density);
    atomicAdd(&d_sum_vis_imag[i_vis], imag * flux_density);
}

/// Generate visibilities for the given source list.
///
/// Currently only takes a single source containing point sources.
extern "C" int vis_gen(const UVW_c *uvw, const Source_c *src, Vis_c *vis, unsigned int n_channels,
                       unsigned int n_baselines) {
    // Sanity checks.
    // TODO: Check other source component types when they are being handled.
    if (src->n_points == 0) {
        fprintf(stderr, "No point sources provided; nothing to do!\n");
        return 1;
    }

    UVW_c *d_uvw = NULL;
    size_t size_baselines = n_baselines * n_channels * sizeof(UVW_c);
    cudaSoftCheck(cudaMalloc(&d_uvw, size_baselines));
    cudaSoftCheck(cudaMemcpy(d_uvw, uvw, size_baselines, cudaMemcpyHostToDevice));

    LMN_c *d_lmn = NULL;
    size_t size_lmn = src->n_points * sizeof(LMN_c);
    cudaSoftCheck(cudaMalloc(&d_lmn, size_lmn));
    cudaSoftCheck(cudaMemcpy(d_lmn, src->point_lmn, size_lmn, cudaMemcpyHostToDevice));

    float *d_point_fd = NULL;
    size_t size_points = src->n_points * sizeof(float);
    size_t size_fds = size_points * n_channels;
    cudaSoftCheck(cudaMalloc(&d_point_fd, size_fds));
    cudaSoftCheck(cudaMemcpy(d_point_fd, src->point_fd, size_fds, cudaMemcpyHostToDevice));

    float *d_sum_vis_real = NULL;
    float *d_sum_vis_imag = NULL;
    size_t size_visibilities = vis->n_vis * sizeof(float);
    cudaSoftCheck(cudaMalloc(&d_sum_vis_real, size_visibilities));
    cudaSoftCheck(cudaMalloc(&d_sum_vis_imag, size_visibilities));

    dim3 blocks, threads;

    // Generate visibilities for the point sources.
    if (src->n_points > 0) {
        // Thread blocks are distributed by visibility (one visibility per
        // frequency and baseline; y) and point source component (y);
        threads.x = 64;
        threads.y = 16;
        blocks.x = (int)ceilf((float)vis->n_vis / (float)threads.x);
        blocks.y = (int)ceilf((float)src->n_points / (float)threads.y);

#ifndef NDEBUG
        printf("num. visibilities (vis->n_vis): %u\n", vis->n_vis);
        printf("num. point source components (src->n_points): %u\n", src->n_points);
        printf("num. x blocks (n_baselines * n_channels): %u\n", blocks.x);
        printf("num. y blocks (src->n_points): %u\n", blocks.y);
#endif

        calc_point_vis<<<blocks, threads>>>(n_baselines, src->n_points, vis->n_vis, d_uvw, d_lmn, d_point_fd,
                                            d_sum_vis_real, d_sum_vis_imag);
        cudaCheck(cudaPeekAtLastError());
        cudaSoftCheck(cudaFree(d_point_fd));
    } // if (num_points > 0)

    // Copy the results into host memory.
    cudaSoftCheck(cudaMemcpy(vis->real, d_sum_vis_real, size_visibilities, cudaMemcpyDeviceToHost));
    cudaSoftCheck(cudaMemcpy(vis->imag, d_sum_vis_imag, size_visibilities, cudaMemcpyDeviceToHost));

    // Clean up.
    cudaSoftCheck(cudaFree(d_uvw));
    cudaSoftCheck(cudaFree(d_lmn));
    cudaSoftCheck(cudaFree(d_sum_vis_real));
    cudaSoftCheck(cudaFree(d_sum_vis_imag));

    return 0;
}
