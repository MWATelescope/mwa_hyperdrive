// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

#pragma once

#include "types.h"

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

const char *xyzs_to_uvws(const XYZ *d_xyzs, const FLOAT *d_lmsts, UVW *d_uvws, RADec pointing_centre, int num_tiles,
                         int num_baselines, int num_timesteps);

const char *rotate_average(const JonesF32 *d_high_res_vis, const float *d_high_res_weights, JonesF32 *d_low_res_vis,
                           RADec pointing_centre, const int num_timesteps, const int num_tiles, const int num_baselines,
                           const int num_freqs, const int freq_average_factor, const FLOAT *d_lmsts, const XYZ *d_xyzs,
                           const UVW *d_uvws_from, UVW *d_uvws_to, const FLOAT *d_lambdas);

const char *iono_loop(const JonesF32 *d_vis_residual, const float *d_vis_weights, const JonesF32 *d_vis_model,
                      JonesF32 *d_vis_model_rotated, JonesF64 *d_iono_fits, IonoConsts *d_iono_consts,
                      const int num_timesteps, const int num_tiles, const int num_baselines, const int num_freqs,
                      const int num_iterations, const FLOAT *d_lmsts, const UVW *d_uvws, const FLOAT *d_lambdas_m);

const char *subtract_iono(JonesF32 *d_vis_residual, const JonesF32 *d_vis_model, const IonoConsts *d_iono_consts,
                          const IonoConsts *d_old_iono_consts, const UVW *d_uvws, const FLOAT *d_lambdas_m,
                          const int num_timesteps, const int num_baselines, const int num_freqs);

const char *add_model(JonesF32 *d_vis_residual, const JonesF32 *d_vis_model, const IonoConsts *d_iono_consts,
                      const FLOAT *d_lambdas_m, const UVW *d_uvws, const int num_timesteps, const int num_baselines,
                      const int num_freqs);

#ifdef __cplusplus
} // extern "C"
#endif // __cplusplus
