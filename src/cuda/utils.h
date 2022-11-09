// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

/**
 * Utilities for CUDA devices.
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
const char *get_cuda_device_info(int device, char name[256], int *device_major, int *device_minor,
                                 size_t *total_global_mem, int *driver_version, int *runtime_version);

#ifdef __cplusplus
} // extern "C"
#endif // __cplusplus
