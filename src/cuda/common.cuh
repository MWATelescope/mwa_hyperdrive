// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <assert.h>
#include <stdio.h>

#pragma once

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

/**
 * `gpu_assert` checks that CUDA code successfully returned.
 */
inline void gpu_assert(cudaError_t code, const char *file, int line, bool abort = true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "gpu_assert: %s (%s:%d)\n", cudaGetErrorString(code), file, line);
        if (abort)
            exit(code);
    }
}

// `cudaCheck` wraps `gpu_assert` for general usage.
#define cudaCheck(code)                                                                                                \
    { gpu_assert((code), __FILE__, __LINE__); }

#ifndef NDEBUG
#define cudaSoftCheck(code)                                                                                            \
    { gpu_assert((code), __FILE__, __LINE__); }
#else
// When not debugging, `cudaSoftCheck` is a "no-op". Useful for granting full speed in release builds.
#define cudaSoftCheck(code) (code)
#endif

#ifdef __cplusplus
} // extern "C"
#endif // __cplusplus
