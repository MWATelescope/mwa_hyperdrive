// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

/**
 * Utilities for CUDA/HIP devices.
 */

#pragma once

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

/**
 * A "watered-down" version of the CUDA example "deviceQuery".
 *
 * See the full example at:
   https://github.com/NVIDIA/cuda-samples/blob/master/Samples/1_Utilities/deviceQuery/deviceQuery.cpp
 *
 * As this code contains code derived from an official NVIDIA example, legally,
 * a copyright, list of conditions and disclaimer must be distributed with this
 * code. This should be found in the root of the mwa_hyperdrive git repo, file
 * LICENSE-NVIDIA.
 */
const char *get_gpu_device_info(int device, char name[256], int *device_major, int *device_minor,
                                size_t *total_global_mem, int *driver_version, int *runtime_version);

/**
 * Zero the beam-response Jones matrices of every direction below the horizon.
 *
 * `d_jones` has shape (`num_tiles`, `num_freqs`, `num_directions`), slowest to
 * fastest (the layout hyperbeam's GPU beam code writes), and `d_zas` holds
 * `num_directions` zenith angles [radians]. Both are in the GPU precision
 * (`float` with `SINGLE`, otherwise `double`), so each Jones matrix is 8
 * floats; they are passed as `void` pointers so that this declaration is the
 * same for both precisions. A direction is below the horizon if its zenith
 * angle is greater than pi/2.
 */
const char *zero_jones_below_horizon(const void *d_zas, int num_directions, int num_tiles, int num_freqs,
                                     void *d_jones);

#ifdef __cplusplus
} // extern "C"
#endif // __cplusplus
