// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! CUDA code to be used by hyperdrive, contained within its own crate.

#![allow(non_snake_case)]
#![allow(clippy::upper_case_acronyms)]

mod utils;

pub(crate) use utils::{get_device_info, CudaDeviceInfo, CudaDriverInfo};

// Import Rust bindings to the CUDA code specific to the precision we're using,
// and set corresponding compile-time types.
#[cfg(feature = "cuda-single")]
include!("model_single.rs");
#[cfg(feature = "cuda-single")]
include!("memory_single.rs");

#[cfg(not(feature = "cuda-single"))]
include!("model_double.rs");
#[cfg(not(feature = "cuda-single"))]
include!("memory_double.rs");

// Set a compile-time variable type.
cfg_if::cfg_if! {
    if #[cfg(feature = "cuda-single")] {
        /// f32 (using the "cuda-single" feature)
        pub(crate) type CudaFloat = f32;
        pub(crate) type CudaJones = JonesF32;
    } else if #[cfg(all(feature = "cuda", not(feature = "cuda-single")))] {
        /// f64 (using the "cuda" feature and not "cuda-single")
        pub(crate) type CudaFloat = f64;
        pub(crate) type CudaJones = JonesF64;
    }
}
