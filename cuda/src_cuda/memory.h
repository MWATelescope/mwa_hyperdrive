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
 * The return type of `allocate_init`. All pointers are to device memory, except
 * `host_vis`.
 */
typedef struct Addresses {
    const size_t num_freqs;
    const size_t num_vis;
    const size_t sbf_l;
    const size_t sbf_n;
    const double sbf_c;
    const double sbf_dx;
    UVW *d_uvws;
    double *d_freqs;
    double *d_shapelet_basis_values;
    JonesF32 *d_vis;
    JonesF32 *host_vis;
} Addresses;

/**
 * Function to allocate necessary arrays (UVWs, frequencies and visibilities)
 * for modelling on the device.
 */
Addresses init_model(const size_t num_baselines, const size_t num_freqs, const size_t sbf_l, const size_t sbf_n,
                     const double sbf_c, const double sbf_dx, const UVW *uvws, const double *freqs,
                     const double *shapelet_basis_values, JonesF32 *vis);

/**
 * Copy the device visibilities to the host. It is assumed that this operation
 * always succeeds so not status int is returned.
 */
void copy_vis(const Addresses *addresses);

/**
 * Set all of the visibilities to zero.
 */
void clear_vis(const Addresses *a);

/**
 * Deallocate necessary arrays (UVWs, frequencies and visibilities) on the
 * device.
 */
void destroy(const Addresses *addresses);

#ifdef __cplusplus
} // extern "C"
#endif // __cplusplus
