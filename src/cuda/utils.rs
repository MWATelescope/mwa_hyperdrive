// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Utilities for CUDA devices.
//!
//! We assume that everything is UTF-8.

include!("utils_bindings.rs");

use std::ffi::{CStr, CString};

use cuda_runtime_sys::*;
use log::trace;
use marlu::cuda_runtime_sys;

#[derive(Debug, Clone)]
pub(crate) struct CudaDriverInfo {
    /// Formatted CUDA driver version, e.g. "11.7".
    pub(crate) driver_version: String,
    /// Formatted CUDA runtime version, e.g. "11.7".
    pub(crate) runtime_version: String,
}

#[derive(Debug, Clone)]
pub(crate) struct CudaDeviceInfo {
    pub(crate) name: String,
    pub(crate) capability: String,
    /// \[MebiBytes (MiB)\]
    pub(crate) total_global_mem: usize,
}

/// Get CUDA device and driver information. At present, this function only
/// returns information on "device 0".
pub(crate) fn get_device_info() -> Result<(CudaDeviceInfo, CudaDriverInfo), String> {
    unsafe {
        // TODO: Always assume we're using device 0 for now.
        let device = 0;
        let name = CString::from_vec_unchecked(vec![1; 256]).into_raw();
        let mut device_major = 0;
        let mut device_minor = 0;
        let mut total_global_mem = 0;
        let mut driver_version = 0;
        let mut runtime_version = 0;
        let error_id = get_cuda_device_info(
            device,
            name,
            &mut device_major,
            &mut device_minor,
            &mut total_global_mem,
            &mut driver_version,
            &mut runtime_version,
        );
        cuda_error_to_rust_error(error_id)?;

        let device_info = CudaDeviceInfo {
            name: CString::from_raw(name)
                .to_str()
                .expect("CUDA device name isn't UTF-8")
                .to_string(),
            capability: format!("{}.{}", device_major, device_minor),
            total_global_mem: total_global_mem / 1048576,
        };

        let driver_version = format!("{}.{}", driver_version / 1000, (driver_version % 100) / 10);
        let runtime_version = format!(
            "{}.{}",
            runtime_version / 1000,
            (runtime_version % 100) / 10
        );

        Ok((
            device_info,
            CudaDriverInfo {
                driver_version,
                runtime_version,
            },
        ))
    }
}

/// Convert a `cudaError_t` cast to an `i32` to a Rust error.
fn cuda_error_to_rust_error(error_id: i32) -> Result<(), String> {
    if error_id == cudaError::cudaSuccess as i32 {
        Ok(())
    } else {
        unsafe {
            trace!("CUDA error ID: {error_id}");
            let error_str = get_cuda_error(error_id);
            trace!("CUDA error str address: {:#x}", error_str as usize);
            Err(CStr::from_ptr(error_str)
                .to_str()
                .expect("CUDA error string isn't UTF-8")
                .to_string())
        }
    }
}
