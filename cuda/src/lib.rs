// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! CUDA code to be used by hyperdrive, contained within its own crate.

#![allow(non_camel_case_types)]
#![allow(non_snake_case)]

mod utils;

pub use utils::{get_device_info, CudaDeviceInfo, CudaDriverInfo};

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
