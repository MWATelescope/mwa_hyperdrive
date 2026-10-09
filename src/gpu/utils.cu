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
#define gpuDeviceSynchronize   hipDeviceSynchronize
#define gpuError_t             hipError_t
#define gpuDriverGetVersion    hipDriverGetVersion
#define gpuGetDeviceProperties hipGetDeviceProperties
#define gpuGetErrorString      hipGetErrorString
#define gpuGetLastError        hipGetLastError
#define gpuRuntimeGetVersion   hipRuntimeGetVersion
#define gpuSetDevice           hipSetDevice
#define gpuSuccess             hipSuccess

// CUDA-specific defines.
#elif __CUDACC__
#define gpuDeviceProp          cudaDeviceProp
#define gpuDeviceSynchronize   cudaDeviceSynchronize
#define gpuError_t             cudaError_t
#define gpuDriverGetVersion    cudaDriverGetVersion
#define gpuGetDeviceProperties cudaGetDeviceProperties
#define gpuGetErrorString      cudaGetErrorString
#define gpuGetLastError        cudaGetLastError
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

#include "types.h"
#include "utils.h"

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

/**
 * See `zero_jones_below_horizon` for the layout of `jones`. One thread per
 * Jones matrix; consecutive threads handle consecutive directions, so the
 * reads of `zas` and the writes of `jones` are coalesced.
 */
__global__ void zero_jones_below_horizon_kernel(const FLOAT *zas, const int num_directions, const size_t num_jones,
                                                JONES *jones) {
    // pi/2, in the GPU precision.
    const FLOAT horizon_za = 1.5707963267948966;
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < num_jones;
         i += (size_t)gridDim.x * blockDim.x) {
        const int i_direction = (int)(i % (size_t)num_directions);
        if (zas[i_direction] > horizon_za) {
            jones[i] = JONES{};
        }
    }
}

extern "C" const char *zero_jones_below_horizon(const void *d_zas, int num_directions, int num_tiles, int num_freqs,
                                                void *d_jones) {
    if (num_directions <= 0 || num_tiles <= 0 || num_freqs <= 0) {
        return NULL;
    }
    const size_t num_jones = (size_t)num_tiles * (size_t)num_freqs * (size_t)num_directions;

    dim3 gridDim, blockDim;
    blockDim.x = 256;
    // Cap the grid size; the kernel strides over anything beyond it.
    size_t num_blocks = (num_jones + blockDim.x - 1) / blockDim.x;
    if (num_blocks > 65535) {
        num_blocks = 65535;
    }
    gridDim.x = (unsigned int)num_blocks;
    zero_jones_below_horizon_kernel<<<gridDim, blockDim>>>((const FLOAT *)d_zas, num_directions, num_jones,
                                                           (JONES *)d_jones);

    gpuError_t error_id;
#ifdef DEBUG
    error_id = gpuDeviceSynchronize();
    if (error_id != gpuSuccess) {
        return gpuGetErrorString(error_id);
    }
#endif
    error_id = gpuGetLastError();
    if (error_id != gpuSuccess) {
        return gpuGetErrorString(error_id);
    }

    return NULL;
}
