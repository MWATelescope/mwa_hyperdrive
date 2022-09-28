// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

#pragma once

#include "types.h"

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

/**
 * Common things needed to perform modelling. All pointers are to device
 * memory.
 */
typedef struct Addresses {
    int num_freqs;
    int num_vis;
    int num_tiles;
    int sbf_l;
    int sbf_n;
    FLOAT sbf_c;
    FLOAT sbf_dx;
    const FLOAT *d_freqs;
    const FLOAT *d_shapelet_basis_values;
    int num_unique_beam_freqs;
    const int *d_tile_map;
    const int *d_freq_map;
    const int *d_tile_index_to_unflagged_tile_index_map;
    JonesF32 *d_vis;
} Addresses;

#ifdef __cplusplus
} // extern "C"
#endif // __cplusplus
