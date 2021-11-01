// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

#pragma once

#include <stdlib.h>

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
    FLOAT *d_freqs;
    FLOAT *d_shapelet_basis_values;
    int num_unique_beam_freqs;
    const uint64_t *d_beam_jones_map;
    JonesF32 *d_vis;
} Addresses;

/**
 * Function to allocate necessary arrays (UVWs, frequencies and visibilities)
 * for modelling on the device.
 */
Addresses init_model(int num_baselines, int num_freqs, int num_tiles, int sbf_l, int sbf_n, FLOAT sbf_c, FLOAT sbf_dx,
                     UVW *uvws, FLOAT *freqs, FLOAT *shapelet_basis_values, void *d_fee_coeffs, int num_fee_beam_coeffs,
                     int num_unique_fee_tiles, int num_unique_fee_freqs, uint64_t *d_beam_jones_map,
                     void *d_beam_norm_jones, JonesF32 *vis);

/**
 * Copy the device visibilities to the host. It is assumed that this operation
 * always succeeds so not status int is returned.
 */
void copy_vis(const Addresses *addresses);

/**
 * Set all of the visibilities to zero.
 */
void clear_vis(Addresses *a);

/**
 * Deallocate necessary arrays (UVWs, frequencies and visibilities) on the
 * device.
 */
void destroy(Addresses *addresses);

#ifdef __cplusplus
} // extern "C"
#endif // __cplusplus
