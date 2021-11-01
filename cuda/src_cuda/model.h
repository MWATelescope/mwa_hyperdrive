// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

#pragma once

#include <stdlib.h>

#include "memory.h"
#include "types.h"

const FLOAT POWER_LAW_FD_REF_FREQ = 150e6; // Hz

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

/**
 * Generate sky-model visibilities for a single timestep given multiple
 * sky-model point-source components. See the documentation of `model_timestep`
 * for more info.
 */
int model_points(const Points *points, const Addresses *a, const UVW *d_uvws, const void *d_beam_jones);

/**
 * Generate sky-model visibilities for a single timestep given multiple
 * sky-model Gaussian components. See the documentation of `model_timestep` for
 * more info.
 */
int model_gaussians(const Gaussians *gaussians, const Addresses *a, const UVW *d_uvws, const void *d_beam_jones);

/**
 * Generate sky-model visibilities for a single timestep given multiple
 * sky-model shapelet components. See the documentation of `model_timestep` for
 * more info.
 */
int model_shapelets(const Shapelets *shapelets, const Addresses *a, const UVW *d_uvws, const void *d_beam_jones);

/**
 * Generate sky-model visibilities for a single timestep given multiple
 * sky-model sources.
 *
 * `uvws` has one element per baseline. `freqs` has one element per...
 * frequency.
 *
 * `points`, `gaussians` and `shapelets` contain coordinates and flux densities
 * for their respective component types. The components are further split into
 * "power law" and "list" types; this is done for efficiency. For the list
 * types, the flux densities ("fds") are two-dimensional arrays, of which the
 * first axis corresponds to frequency and the second component.
 *
 * `*_shapelet_uvs` are special UVWs (without the Ws) calculated just for the
 * shapelets. These are two-dimensional arrays; the first axis corresponds to
 * baselines and the second a shapelet component.
 *
 * `*_shapelet_coeffs` is a flattened array-of-arrays. The length of each
 * sub-array is indicated by `*_num_shapelet_coeffs` (which has a length equal
 * to `*_num_shapelets`).
 *
 * `vis` is a two-dimensional array, of which the first axis corresponds to
 * baselines and the second frequency. It is the only argument that should be
 * mutated and should be completely full of zeros before this function is
 * called.
 */
int model_timestep_no_beam(int num_baselines, int num_freqs, UVW *uvws, FLOAT *freqs, Points *points,
                           Gaussians *gaussians, Shapelets *shapelets, FLOAT *shapelet_basis_values, int sbf_l,
                           int sbf_n, FLOAT sbf_c, FLOAT sbf_dx, JonesF32 *vis);

int model_timestep_fee_beam(int num_baselines, int num_freqs, int num_tiles, UVW *uvws, FLOAT *freqs, Points *points,
                            Gaussians *gaussians, Shapelets *shapelets, FLOAT *shapelet_basis_values, int sbf_l,
                            int sbf_n, FLOAT sbf_c, FLOAT sbf_dx, void *d_beam_coeffs, int num_beam_coeffs,
                            int num_unique_fee_tiles, int num_unique_fee_freqs, uint64_t *d_beam_jones_map,
                            void *d_beam_norm_jones, JonesF32 *vis);

#ifdef __cplusplus
} // extern "C"
#endif // __cplusplus
