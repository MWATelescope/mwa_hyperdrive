// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

#pragma once

#include <stdlib.h>

#include "memory.h"
#include "types.h"

const double POWER_LAW_FD_REF_FREQ = 150e6;

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

/**
 * Generate sky-model visibilities for a single timestep given multiple
 * sky-model point-sources with power-law flux densities. See the documentation
 * of `model_timestep` for more info.
 */
int model_power_law_points(const size_t num_power_law_points, const LMN *lmns, const JonesF64 *ref_fds,
                           const double *sis, const Addresses *a);

/**
 * Generate sky-model visibilities for a single timestep given multiple
 * sky-model point-sources with flux-density lists. See the documentation of
 * `model_timestep` for more info.
 */
int model_list_points(const size_t num_points, const LMN *point_lmns, const JonesF64 *point_fds, const Addresses *a);

/**
 * Generate sky-model visibilities for a single timestep given multiple
 * sky-model Gaussian-sources with power-law flux densities. See the
 * documentation of `model_timestep` for more info.
 */
int model_power_law_gaussians(const size_t num_power_law_gaussians, const LMN *lmns, const JonesF64 *ref_fds,
                              const double *sis, const GaussianParams *gaussian_params, const Addresses *a);

/**
 * Generate sky-model visibilities for a single timestep given multiple
 * sky-model Gaussian-sources with flux-density lists. See the
 * documentation of `model_timestep` for more info.
 */
int model_list_gaussians(const size_t num_power_law_gaussians, const LMN *lmns, const JonesF64 *fds,
                         const GaussianParams *gaussian_params, const Addresses *a);

/**
 * Generate sky-model visibilities for a single timestep given multiple
 * sky-model shapelet-sources with power-law flux densities. See the
 * documentation of `model_timestep` for more info.
 */
int model_power_law_shapelets(const size_t num_shapelets, const LMN *shapelet_lmns, const JonesF64 *shapelet_fds,
                              const GaussianParams *gaussian_params, const ShapeletUV *shapelet_uvs,
                              const ShapeletCoeff *shapelet_coeffs, const size_t *num_shapelet_coeffs,
                              const Addresses a);

/**
 * Generate sky-model visibilities for a single timestep given multiple
 * sky-model shapelet-sources with flux-density lists. See the documentation of
 * `model_timestep` for more info.
 */
int model_list_shapelets(const size_t num_shapelets, const LMN *shapelet_lmns, const JonesF64 *shapelet_fds,
                         const GaussianParams *gaussian_params, const ShapeletUV *shapelet_uvs,
                         const ShapeletCoeff *shapelet_coeffs, const size_t *num_shapelet_coeffs, const Addresses a);

/**
 * Generate sky-model visibilities for a single timestep given multiple
 * sky-model sources.
 *
 * `uvws` has one element per baseline. `freqs` has one element per...
 * frequency.
 *
 * `point_lmns`, `gaussian_lmns` and `shapelet_lmns` are the LMN coordinates for
 * each sky-model component type.
 *
 * `point_fds` etc. are two-dimensional arrays, of which the first axis
 * corresponds to frequency and the second component.
 *
 * `gaussian_gaussian_params` applies to Gaussian components, whereas
 * `shapelet_gaussian_params` are for shapelet components.
 *
 * `shapelet_uvs` are special UVWs (without the Ws) calculated just for the
 * shapelets. These are two-dimensional arrays; the first axis corresponds to
 * baselines and the second a shapelet component.
 *
 * `shapelet_coeffs` is a flattened array-of-arrays. The length of each
 * sub-array is indicated by `num_shapelet_coeffs` (which has a length equal to
 * `num_shapelets`)
 *
 * `vis` is a two-dimensional array, of which the first axis corresponds to
 * baselines and the second frequency. It is the only argument that should be
 * mutated and should be completely full of zeros before this function is
 * called.
 */
int model_timestep(const size_t num_baselines, const size_t num_freqs, const size_t num_power_law_points,
                   const size_t num_list_points, const size_t num_power_law_gaussians, const size_t num_list_gaussians,
                   const size_t num_shapelets, const UVW *uvws, const double *freqs, const LMN *point_power_law_lmns,
                   const JonesF64 *point_power_law_ref_fds, const double *point_power_law_sis,
                   const LMN *point_list_lmns, const JonesF64 *point_list_fds, const LMN *gaussian_power_law_lmns,
                   const JonesF64 *gaussian_power_law_ref_fds, const double *gaussian_power_law_sis,
                   const GaussianParams *gaussian_power_law_gaussian_params, const LMN *gaussian_list_lmns,
                   const JonesF64 *gaussian_list_fds, const GaussianParams *gaussian_list_gaussian_params,
                   const LMN *shapelet_lmns, const JonesF64 *shapelet_fds,
                   const GaussianParams *shapelet_gaussian_params, const ShapeletUV *shapelet_uvs,
                   const ShapeletCoeff *shapelet_coeffs, const size_t *num_shapelet_coeffs,
                   const double *shapelet_basis_values, const size_t sbf_l, const size_t sbf_n, const double sbf_c,
                   const double sbf_dx, JonesF32 *vis);

#ifdef __cplusplus
} // extern "C"
#endif // __cplusplus
