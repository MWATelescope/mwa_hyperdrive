// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Utilities for CUDA devices.
//!
//! We assume that everything is UTF-8.

include!("utils_bindings.rs");

use std::ffi::{CStr, CString};

use super::CudaError;

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
            // Get the CUDA error message associated with the enum variant.
            let error_message = CStr::from_ptr(error_message_ptr)
                .to_str()
                .unwrap_or("<cannot read CUDA error string>");
            let our_error_str = format!(
                "{}:{}: get_cuda_device_info: {error_message}",
                file!(),
                line!()
            );
            return Err(CudaError::Generic(our_error_str));
        }

        let device_info = CudaDeviceInfo {
            name: CString::from_raw(name)
                .to_str()
                .expect("CUDA device name isn't UTF-8")
                .to_string(),
            capability: format!("{device_major}.{device_minor}"),
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
