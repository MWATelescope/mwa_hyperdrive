// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Utilities for CUDA devices.
//!
//! We assume that everything is UTF-8.

include!("utils_bindings.rs");

use std::{
    ffi::{CStr, CString},
    panic::Location,
};

use super::CudaError;

#[derive(Debug, Clone)]
pub(crate) struct CudaDriverInfo {
    /// Formatted CUDA driver version, e.g. "11.7".
    pub(crate) driver_version: Box<str>,
    /// Formatted CUDA runtime version, e.g. "11.7".
    pub(crate) runtime_version: Box<str>,
}

#[derive(Debug, Clone)]
pub(crate) struct CudaDeviceInfo {
    pub(crate) name: Box<str>,
    pub(crate) capability: Box<str>,
    /// \[MebiBytes (MiB)\]
    pub(crate) total_global_mem: usize,
}

/// Get CUDA device and driver information. At present, this function only
/// returns information on "device 0".
#[track_caller]
pub(crate) fn get_device_info() -> Result<(CudaDeviceInfo, CudaDriverInfo), CudaError> {
    unsafe {
        // TODO: Always assume we're using device 0 for now.
        let device = 0;
        let name = CString::from_vec_unchecked(vec![1; 256]).into_raw();
        let mut device_major = 0;
        let mut device_minor = 0;
        let mut total_global_mem = 0;
        let mut driver_version = 0;
        let mut runtime_version = 0;
        let error_message_ptr = get_cuda_device_info(
            device,
            name,
            &mut device_major,
            &mut device_minor,
            &mut total_global_mem,
            &mut driver_version,
            &mut runtime_version,
        );
        if !error_message_ptr.is_null() {
            // Get the CUDA error message behind the pointer.
            let error_message = CStr::from_ptr(error_message_ptr)
                .to_str()
                .unwrap_or("<cannot read CUDA error string>");
            let location = Location::caller();
            return Err(CudaError::Generic {
                msg: error_message.into(),
                file: location.file(),
                line: location.line(),
            });
        }

        let device_info = CudaDeviceInfo {
            name: CString::from_raw(name)
                .to_str()
                .expect("CUDA device name isn't UTF-8")
                .to_string()
                .into_boxed_str(),
            capability: format!("{device_major}.{device_minor}").into_boxed_str(),
            total_global_mem: total_global_mem / 1048576,
        };

        let driver_version = format!("{}.{}", driver_version / 1000, (driver_version / 10) % 100);
        let runtime_version = format!(
            "{}.{}",
            runtime_version / 1000,
            (runtime_version / 10) % 100
        );

        Ok((
            device_info,
            CudaDriverInfo {
                driver_version: driver_version.into_boxed_str(),
                runtime_version: runtime_version.into_boxed_str(),
            },
        ))
    }
}
