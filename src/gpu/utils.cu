// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

// "Homegrown" GPU utilities.
//
// As this code contains code derived from an official NVIDIA example
// (https://github.com/NVIDIA/cuda-samples/blob/master/Samples/1_Utilities/deviceQuery/deviceQuery.cpp),
// legally, a copyright, list of conditions and disclaimer must be distributed
// with this code. This should be found in the root directory of the
// mwa_hyperdrive git repo, file LICENSE-NVIDIA.

// HIP-specific defines.
#if __HIPCC__
#define gpuDeviceProp          hipDeviceProp_t
#define gpuError_t             hipError_t
#define gpuDriverGetVersion    hipDriverGetVersion
#define gpuGetDeviceProperties hipGetDeviceProperties
#define gpuGetErrorString      hipGetErrorString
#define gpuRuntimeGetVersion   hipRuntimeGetVersion
#define gpuSetDevice           hipSetDevice
#define gpuSuccess             hipSuccess

// CUDA-specific defines.
#elif __CUDACC__
#define gpuDeviceProp          cudaDeviceProp
#define gpuError_t             cudaError_t
#define gpuDriverGetVersion    cudaDriverGetVersion
#define gpuGetDeviceProperties cudaGetDeviceProperties
#define gpuGetErrorString      cudaGetErrorString
#define gpuRuntimeGetVersion   cudaRuntimeGetVersion
#define gpuSetDevice           cudaSetDevice
#define gpuSuccess             cudaSuccess
#endif // __HIPCC__

#ifdef __CUDACC__
#include <cuda.h>
#elif __HIPCC__
#include <hip/hip_complex.h>
#include <hip/hip_runtime.h>
#endif

extern "C" const char *get_gpu_device_info(int device, char name[256], int *device_major, int *device_minor,
                                           size_t *total_global_mem, int *driver_version, int *runtime_version) {
    gpuError_t error_id = gpuSetDevice(device);
    if (error_id != gpuSuccess)
        return gpuGetErrorString(error_id);

    gpuDeviceProp device_prop;
    error_id = gpuGetDeviceProperties(&device_prop, device);
    if (error_id != gpuSuccess)
        return gpuGetErrorString(error_id);

    memcpy(name, device_prop.name, 256);
    *device_major = device_prop.major;
    *device_minor = device_prop.minor;
    *total_global_mem = device_prop.totalGlobalMem;

    error_id = gpuDriverGetVersion(driver_version);
    if (error_id != gpuSuccess)
        return gpuGetErrorString(error_id);

    error_id = gpuRuntimeGetVersion(runtime_version);
    if (error_id != gpuSuccess)
        return gpuGetErrorString(error_id);

    return NULL;
}
