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
const char *model_points(const Points *comps, const Addresses *a, const UVW *d_uvws, const JONES *d_beam_jones,
                         JonesF32 *d_vis_fb);

/**
 * Generate sky-model visibilities for a single timestep given multiple
 * sky-model Gaussian components. See the documentation of `model_points` for
 * more info.
 */
const char *model_gaussians(const Gaussians *comps, const Addresses *a, const UVW *d_uvws, const JONES *d_beam_jones,
                            JonesF32 *d_vis_fb);

/**
 * Generate sky-model visibilities for a single timestep given multiple
 * sky-model shapelet components. See the documentation of `model_points` for
 * more info.
 */
const char *model_shapelets(const Shapelets *comps, const Addresses *a, const UVW *d_uvws, const JONES *d_beam_jones,
                            JonesF32 *d_vis_fb);

#ifdef __cplusplus
} // extern "C"
#endif // __cplusplus
