// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Utilities for CUDA/HIP devices.
//!
//! We assume that everything is UTF-8.

include!("utils_bindings.rs");

use std::{
    ffi::{CStr, CString},
    panic::Location,
};

use super::GpuError;

#[derive(Debug, Clone)]
pub(crate) struct GpuDriverInfo {
    /// Formatted CUDA/HIP driver version, e.g. "11.7".
    pub(crate) driver_version: Box<str>,
    /// Formatted CUDA/HIP runtime version, e.g. "11.7".
    pub(crate) runtime_version: Box<str>,
}

#[derive(Debug, Clone)]
pub(crate) struct GpuDeviceInfo {
    pub(crate) name: Box<str>,
    pub(crate) capability: Box<str>,
    /// \[MebiBytes (MiB)\]
    pub(crate) total_global_mem: usize,
}

/// Get CUDA/HIP device and driver information. At present, this function only
/// returns information on "device 0".
pub(crate) fn get_device_info() -> Result<(GpuDeviceInfo, GpuDriverInfo), GpuError> {
    unsafe {
        // TODO: Always assume we're using device 0 for now.
        let device = 0;
        let name = CString::from_vec_unchecked(vec![1; 256]).into_raw();
        let mut device_major = 0;
        let mut device_minor = 0;
        let mut total_global_mem = 0;
        let mut driver_version = 0;
        let mut runtime_version = 0;
        let error_message_ptr = get_gpu_device_info(
            device,
            name,
            &mut device_major,
            &mut device_minor,
            &mut total_global_mem,
            &mut driver_version,
            &mut runtime_version,
        );
        if !error_message_ptr.is_null() {
            // Get the CUDA/HIP error message behind the pointer.
            let error_message = CStr::from_ptr(error_message_ptr).to_str();
            #[cfg(feature = "cuda")]
            let error_message = error_message.unwrap_or("<cannot read CUDA error string>");
            #[cfg(feature = "hip")]
            let error_message = error_message.unwrap_or("<cannot read HIP error string>");
            let location = Location::caller();
            return Err(GpuError::Generic {
                msg: error_message.into(),
                file: location.file(),
                line: location.line(),
            });
        }

        let device_info = GpuDeviceInfo {
            name: CString::from_raw(name)
                .to_str()
                .expect("GPU device name isn't UTF-8")
                .to_string()
                .into_boxed_str(),
            capability: format!("{device_major}.{device_minor}").into_boxed_str(),
            total_global_mem: total_global_mem / 1048576,
        };

        #[cfg(feature = "cuda")]
        let (driver_version, runtime_version) = {
            let d = format!("{}.{}", driver_version / 1000, (driver_version / 10) % 100);
            let r = format!(
                "{}.{}",
                runtime_version / 1000,
                (runtime_version / 10) % 100
            );
            (d, r)
        };
        #[cfg(feature = "hip")]
        let (driver_version, runtime_version) = {
            // This isn't documented, but is the only thing that makes sense to
            // me.
            let d = format!(
                "{}.{}",
                driver_version / 10_000_000,
                (driver_version / 10_000) % 100
            );
            let r = format!(
                "{}.{}",
                runtime_version / 10_000_000,
                (runtime_version / 10_000) % 100
            );
            (d, r)
        };

        Ok((
            device_info,
            GpuDriverInfo {
                driver_version: driver_version.into_boxed_str(),
                runtime_version: runtime_version.into_boxed_str(),
            },
        ))
    }
}
