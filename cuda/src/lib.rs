// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! CUDA code to be used by hyperdrive, contained within its own crate.

#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
// This is needed only because tests inside bindgen-produced files (model_*.rs
// and memory_*.rs) trigger the warning.
#![allow(deref_nullptr)]
// Link hyperdrive_cu produced from build.rs
#![link(name = "hyperdrive_cu", kind = "static")]

pub use mwa_hyperdrive_beam::{CudaError, DevicePointer};

// Import Rust bindings to the CUDA code specific to the precision we're using,
// and set corresponding compile-time types.
mwa_hyperdrive_common::cfg_if::cfg_if! {
    if #[cfg(feature = "cuda-single")] {
        include!("model_single.rs");
        include!("memory_single.rs");
    } else {
        include!("model_double.rs");
        include!("memory_double.rs");
    }
}
