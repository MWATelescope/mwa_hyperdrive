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
                               const float *d_u, const float *d_v, const float *d_w, const float *d_l, const float *d_m,
                               const float *d_n, const float *d_point_fd, float *d_sum_vis_real,
                               float *d_sum_vis_imag) {
    const int i_vis = threadIdx.x + (blockDim.x * blockIdx.x);
    const int i_comp = threadIdx.y + (blockDim.y * blockIdx.y);

    if ((i_vis >= n_vis) || (i_comp >= n_points))
        return;

    // All present frequencies change every `n_baselines`.
    const int i_freq = i_vis / n_baselines;

    const float u = d_u[i_vis];
    const float v = d_v[i_vis];
    const float w = d_w[i_vis];
    const float l = d_l[i_comp];
    const float m = d_m[i_comp];
    const float n = d_n[i_comp];
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
extern "C" int vis_gen(const UVW_s *uvw, const Source_s *src, Visibilities_s *vis) {
    // Sanity checks.
    // TODO: Check other source component types when they are being handled.
    if (src->n_points == 0) {
        fprintf(stderr, "No point sources provided; nothing to do!\n");
        return 1;
    }

    float *d_u = NULL;
    float *d_v = NULL;
    float *d_w = NULL;
    size_t size_baselines = uvw->n_vis * sizeof(float);
    cudaSoftCheck(cudaMalloc(&d_u, size_baselines));
    cudaSoftCheck(cudaMalloc(&d_v, size_baselines));
    cudaSoftCheck(cudaMalloc(&d_w, size_baselines));
    cudaSoftCheck(cudaMemcpy(d_u, uvw->u, size_baselines, cudaMemcpyHostToDevice));
    cudaSoftCheck(cudaMemcpy(d_v, uvw->v, size_baselines, cudaMemcpyHostToDevice));
    cudaSoftCheck(cudaMemcpy(d_w, uvw->w, size_baselines, cudaMemcpyHostToDevice));

    float *d_l = NULL;
    float *d_m = NULL;
    float *d_n = NULL;
    size_t size_points = src->n_points * sizeof(float);
    cudaSoftCheck(cudaMalloc(&d_l, size_points));
    cudaSoftCheck(cudaMalloc(&d_m, size_points));
    cudaSoftCheck(cudaMalloc(&d_n, size_points));
    cudaSoftCheck(cudaMemcpy(d_l, src->point_l, size_points, cudaMemcpyHostToDevice));
    cudaSoftCheck(cudaMemcpy(d_m, src->point_m, size_points, cudaMemcpyHostToDevice));
    cudaSoftCheck(cudaMemcpy(d_n, src->point_n, size_points, cudaMemcpyHostToDevice));

    float *d_point_fd = NULL;
    size_t size_fds = size_points * src->n_channels;
    cudaSoftCheck(cudaMalloc(&d_point_fd, size_fds));
    cudaSoftCheck(cudaMemcpy(d_point_fd, src->point_fd, size_fds, cudaMemcpyHostToDevice));

    float *d_sum_vis_real = NULL;
    float *d_sum_vis_imag = NULL;
    size_t size_visibilities = vis->n_visibilities * sizeof(float);
    cudaSoftCheck(cudaMalloc(&d_sum_vis_real, size_visibilities));
    cudaSoftCheck(cudaMalloc(&d_sum_vis_imag, size_visibilities));

    dim3 blocks, threads;

    // Generate visibilities for the point sources.
    if (src->n_points > 0) {
        // Thread blocks are distributed by visibility (one visibility per
        // frequency and baseline; y) and point source component (y);
        threads.x = 64;
        threads.y = 16;
        blocks.x = (int)ceilf((float)uvw->n_vis / (float)threads.x);
        blocks.y = (int)ceilf((float)src->n_points / (float)threads.y);

#ifndef NDEBUG
        printf("num. visibilities (uvw->n_elem): %u\n", uvw->n_vis);
        printf("num. point source components (src->n_points): %u\n", src->n_points);
        printf("num. x blocks (uvw->n_elem): %u\n", blocks.x);
        printf("num. y blocks (src->n_points): %u\n", blocks.y);
#endif

        calc_point_vis<<<blocks, threads>>>(uvw->n_baselines, src->n_points, uvw->n_vis, d_u, d_v, d_w, d_l, d_m, d_n,
                                            d_point_fd, d_sum_vis_real, d_sum_vis_imag);
        cudaCheck(cudaPeekAtLastError());
        cudaSoftCheck(cudaFree(d_point_fd));
    } // if (num_points > 0)

    // Copy the results into host memory.
    cudaSoftCheck(cudaMemcpy(vis->real, d_sum_vis_real, size_visibilities, cudaMemcpyDeviceToHost));
    cudaSoftCheck(cudaMemcpy(vis->imag, d_sum_vis_imag, size_visibilities, cudaMemcpyDeviceToHost));

    // Clean up.
    cudaSoftCheck(cudaFree(d_u));
    cudaSoftCheck(cudaFree(d_v));
    cudaSoftCheck(cudaFree(d_w));
    cudaSoftCheck(cudaFree(d_l));
    cudaSoftCheck(cudaFree(d_m));
    cudaSoftCheck(cudaFree(d_n));
    cudaSoftCheck(cudaFree(d_sum_vis_real));
    cudaSoftCheck(cudaFree(d_sum_vis_imag));

    return 0;
}
