// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

#pragma once

#include "types.h"

const FLOAT POWER_LAW_FD_REF_FREQ = 150e6; // Hz

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

/**
 * Common things needed to perform modelling. All pointers are to device
 * memory.
 */
typedef struct Addresses {
    const int num_freqs;
    const int num_vis;
    const int num_baselines;
    const int sbf_l;
    const int sbf_n;
    const FLOAT sbf_c;
    const FLOAT sbf_dx;
    const FLOAT *d_freqs;
    const FLOAT *d_shapelet_basis_values;
    const int num_unique_beam_freqs;
    const int *d_tile_map;
    const int *d_freq_map;
    const int *d_tile_index_to_unflagged_tile_index_map;
    JonesF32 *d_vis;
} Addresses;

/**
 * All the parameters needed to describe point-source components.
 */
typedef struct Points {
    const int num_power_laws;
    const LmnRime *power_law_lmns;
    // Instrumental flux densities calculated at 150 MHz.
    const JONES *power_law_fds;
    // Spectral indices.
    const FLOAT *power_law_sis;

    const int num_curved_power_laws;
    const LmnRime *curved_power_law_lmns;
    // Instrumental flux densities calculated at 150 MHz.
    const JONES *curved_power_law_fds;
    // Spectral indices.
    const FLOAT *curved_power_law_sis;
    // Spectral curvatures.
    const FLOAT *curved_power_law_qs;

    const int num_lists;
    const LmnRime *list_lmns;
    // Instrumental (i.e. XX, XY, YX, XX).
    const JONES *list_fds;
} Points;

/**
 * All the parameters needed to describe Gaussian components.
 */
typedef struct Gaussians {
    const int num_power_laws;
    const LmnRime *power_law_lmns;
    // Instrumental flux densities calculated at 150 MHz.
    const JONES *power_law_fds;
    // Spectral indices.
    const FLOAT *power_law_sis;
    const GaussianParams *power_law_gps;

    const int num_curved_power_laws;
    const LmnRime *curved_power_law_lmns;
    // Instrumental flux densities calculated at 150 MHz.
    const JONES *curved_power_law_fds;
    // Spectral indices.
    const FLOAT *curved_power_law_sis;
    // Spectral curvatures.
    const FLOAT *curved_power_law_qs;
    const GaussianParams *curved_power_law_gps;

    const int num_lists;
    const LmnRime *list_lmns;
    // Instrumental (i.e. XX, XY, YX, XX).
    const JONES *list_fds;
    const GaussianParams *list_gps;
} Gaussians;

/**
 * All the parameters needed to describe Shapelet components.
 */
typedef struct Shapelets {
    const int num_power_laws;
    const LmnRime *power_law_lmns;
    // Instrumental flux densities calculated at 150 MHz.
    const JONES *power_law_fds;
    // Spectral indices.
    const FLOAT *power_law_sis;
    const GaussianParams *power_law_gps;
    const ShapeletUV *power_law_shapelet_uvs;
    const ShapeletCoeff *power_law_shapelet_coeffs;
    const int *power_law_num_shapelet_coeffs;

    const int num_curved_power_laws;
    const LmnRime *curved_power_law_lmns;
    // Instrumental flux densities calculated at 150 MHz.
    const JONES *curved_power_law_fds;
    // Spectral indices.
    const FLOAT *curved_power_law_sis;
    // Spectral curvatures.
    const FLOAT *curved_power_law_qs;
    const GaussianParams *curved_power_law_gps;
    const ShapeletUV *curved_power_law_shapelet_uvs;
    const ShapeletCoeff *curved_power_law_shapelet_coeffs;
    const int *curved_power_law_num_shapelet_coeffs;

    const int num_lists;
    const LmnRime *list_lmns;
    // Instrumental (i.e. XX, XY, YX, XX).
    const JONES *list_fds;
    const GaussianParams *list_gps;
    const ShapeletUV *list_shapelet_uvs;
    const ShapeletCoeff *list_shapelet_coeffs;
    const int *list_num_shapelet_coeffs;
} Shapelets;

/**
 * Generate sky-model visibilities for a single timestep given multiple
 * sky-model point-source components.
 *
 * `points` contains coordinates and flux densities for their respective
 * component types. The components are further split into "power law", "curved
 * power law" and "list" types; this is done for efficiency. For the list types,
 * the flux densities ("fds") are two-dimensional arrays, of which the first
 * axis corresponds to frequency and the second component.
 *
 * `a` is the populated `Addresses` struct needed to do any sky modelling.
 *
 * `d_uvws` has one element per baseline.
 *
 * `d_beam_jones` is the beam response used for each unique tile, unique
 * frequency and direction. The metadata within `a` allows disambiguation of
 * which tile and frequency should use which set of responses.
 */
const char *model_points(const Points *points, const Addresses *a, const UVW *d_uvws, const JONES *d_beam_jones);

/**
 * Generate sky-model visibilities for a single timestep given multiple
 * sky-model Gaussian components. See the documentation of `model_points` for
 * more info.
 */
const char *model_gaussians(const Gaussians *gaussians, const Addresses *a, const UVW *d_uvws,
                            const JONES *d_beam_jones);

/**
 * Generate sky-model visibilities for a single timestep given multiple
 * sky-model shapelet components. See the documentation of `model_points` for
 * more info.
 */
const char *model_shapelets(const Shapelets *shapelets, const Addresses *a, const UVW *d_uvws,
                            const JONES *d_beam_jones);

#ifdef __cplusplus
} // extern "C"
#endif // __cplusplus
