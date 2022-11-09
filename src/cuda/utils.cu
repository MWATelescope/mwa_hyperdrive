// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

// "Homegrown" CUDA utilities.
//
// As this code contains code derived from an official NVIDIA example
// (https://github.com/NVIDIA/cuda-samples/blob/master/Samples/1_Utilities/deviceQuery/deviceQuery.cpp),
// legally, a copyright, list of conditions and disclaimer must be distributed
// with this code. This should be found in the "cuda" directory of the
// mwa_hyperdrive git repo, file LICENSE-NVIDIA.

#include <cuda.h>

extern "C" const char *get_cuda_device_info(int device, char name[256], int *device_major, int *device_minor,
                                            size_t *total_global_mem, int *driver_version, int *runtime_version) {
    cudaError_t error_id = cudaSetDevice(device);
    if (error_id != cudaSuccess)
        return cudaGetErrorString(error_id);

    cudaDeviceProp device_prop;
    error_id = cudaGetDeviceProperties(&device_prop, device);
    if (error_id != cudaSuccess)
        return cudaGetErrorString(error_id);

    memcpy(name, device_prop.name, 256);
    *device_major = device_prop.major;
    *device_minor = device_prop.minor;
    *total_global_mem = device_prop.totalGlobalMem;

    error_id = cudaDriverGetVersion(driver_version);
    if (error_id != cudaSuccess)
        return cudaGetErrorString(error_id);

    error_id = cudaRuntimeGetVersion(runtime_version);
    if (error_id != cudaSuccess)
        return cudaGetErrorString(error_id);

    return NULL;
}
