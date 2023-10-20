// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! GPU code to be used by hyperdrive.

#![allow(non_snake_case)]
#![allow(clippy::upper_case_acronyms)]
#![allow(clippy::excessive_precision)]

#[cfg(test)]
mod tests;
mod utils;

use std::{
    ffi::{c_void, CStr},
    panic::Location,
    ptr::null_mut,
};

use thiserror::Error;

pub(crate) use utils::get_device_info;

// Import Rust bindings to the CUDA/HIP code specific to the precision we're
// using, and set corresponding compile-time types.
cfg_if::cfg_if! {
    if #[cfg(feature = "gpu-single")] {
        /// f32 (using the "gpu-single" feature)
        pub(crate) type GpuFloat = f32;
        pub(crate) type GpuJones = JonesF32;

        include!("types_single.rs");
        include!("model_single.rs");
        include!("peel_single.rs");
    } else if #[cfg(all(any(feature = "cuda", feature = "hip"), not(feature = "gpu-single")))] {
        /// f64 (not using "gpu-single")
        pub(crate) type GpuFloat = f64;
        pub(crate) type GpuJones = JonesF64;

        include!("types_double.rs");
        include!("model_double.rs");
        include!("peel_double.rs");
    }
}

// Import CUDA/HIP functions into the same names.
#[cfg(feature = "cuda")]
use cuda_runtime_sys::{
    cudaDeviceSynchronize as gpuDeviceSynchronize, cudaError::cudaSuccess as gpuSuccess,
    cudaFree as gpuFree, cudaGetErrorString as gpuGetErrorString,
    cudaGetLastError as gpuGetLastError, cudaMalloc as gpuMalloc, cudaMemcpy as gpuMemcpy,
    cudaMemcpyKind::cudaMemcpyDeviceToDevice as gpuMemcpyDeviceToDevice,
    cudaMemcpyKind::cudaMemcpyDeviceToHost as gpuMemcpyDeviceToHost,
    cudaMemcpyKind::cudaMemcpyHostToDevice as gpuMemcpyHostToDevice,
};
#[cfg(feature = "hip")]
use hip_sys::hiprt::{
    hipDeviceSynchronize as gpuDeviceSynchronize, hipError_t::hipSuccess as gpuSuccess,
    hipFree as gpuFree, hipGetErrorString as gpuGetErrorString, hipGetLastError as gpuGetLastError,
    hipMalloc as gpuMalloc, hipMemcpy as gpuMemcpy,
    hipMemcpyKind::hipMemcpyDeviceToDevice as gpuMemcpyDeviceToDevice,
    hipMemcpyKind::hipMemcpyDeviceToHost as gpuMemcpyDeviceToHost,
    hipMemcpyKind::hipMemcpyHostToDevice as gpuMemcpyHostToDevice,
};

// Ensure that the shapelet constants are the same in the Rust code and GPU
// code.
static_assertions::const_assert_eq!(crate::model::shapelets::SBF_L as i32, SBF_L);
static_assertions::const_assert_eq!(crate::model::shapelets::SBF_N as i32, SBF_N);
static_assertions::const_assert_eq!(crate::model::shapelets::SBF_C as GpuFloat, SBF_C);
static_assertions::const_assert_eq!(crate::model::shapelets::SBF_DX as GpuFloat, SBF_DX);

macro_rules! gpu_kernel_call {
    ($gpu_fn:path, $($args:expr),* $(,)?) => {{
        #[allow(unused_unsafe)]
        unsafe {
            let error_message_ptr = $gpu_fn($($args),*);
            if error_message_ptr.is_null() {
                Ok(())
            } else {
                // Get the GPU error message behind the pointer.
                let error_message = std::ffi::CStr::from_ptr(error_message_ptr).to_str();
                #[cfg(feature = "cuda")]
                let error_message = error_message.unwrap_or("<cannot read CUDA error string>");
                #[cfg(feature = "hip")]
                let error_message = error_message.unwrap_or("<cannot read HIP error string>");
                let our_error_message = format!("{}: {error_message}", stringify!($gpu_fn));
                Err(GpuError::Kernel {
                    msg: our_error_message.into(),
                    file: file!(),
                    line: line!(),
                })
            }
        }
    }};
}
pub(crate) use gpu_kernel_call;

#[derive(Clone, Copy)]
pub(crate) enum GpuCall {
    Malloc,
    CopyToDevice,
    CopyFromDevice,
}

/// Run [`gpuGetLastError`] and [`gpuDeviceSynchronize`]. If either of these
/// calls return an error, it is converted to a Rust error and returned from
/// this function. The single argument describes what the just-performed
/// operation was and makes the returned error a helpful one.
///
/// # Safety
///
/// This function interfaces directly with the CUDA/HIP API. Rust errors attempt
/// to catch problems but there are no guarantees.
#[track_caller]
unsafe fn check_for_errors(gpu_call: GpuCall) -> Result<(), GpuError> {
    // Only do a device sync if we're in debug mode, for performance.
    let debug_mode = matches!(std::env::var("DEBUG").as_deref(), Ok("true"));
    if debug_mode {
        let code = gpuDeviceSynchronize();
        if code != gpuSuccess {
            let c_str = CStr::from_ptr(gpuGetErrorString(code));
            let msg = c_str.to_str();
            #[cfg(feature = "cuda")]
            let msg = msg.unwrap_or("<cannot read CUDA error string>");
            #[cfg(feature = "hip")]
            let msg = msg.unwrap_or("<cannot read HIP error string>");
            let location = Location::caller();
            return Err(match gpu_call {
                GpuCall::Malloc => GpuError::Malloc {
                    msg: msg.into(),
                    file: location.file(),
                    line: location.line(),
                },
                GpuCall::CopyToDevice => GpuError::CopyToDevice {
                    msg: msg.into(),
                    file: location.file(),
                    line: location.line(),
                },
                GpuCall::CopyFromDevice => GpuError::CopyFromDevice {
                    msg: msg.into(),
                    file: location.file(),
                    line: location.line(),
                },
            });
        }
    }

    let code = gpuGetLastError();
    if code != gpuSuccess {
        let c_str = CStr::from_ptr(gpuGetErrorString(code));
        let msg = c_str.to_str();
        #[cfg(feature = "cuda")]
        let msg = msg.unwrap_or("<cannot read CUDA error string>");
        #[cfg(feature = "hip")]
        let msg = msg.unwrap_or("<cannot read HIP error string>");
        let location = Location::caller();
        return Err(match gpu_call {
            GpuCall::Malloc => GpuError::Malloc {
                msg: msg.into(),
                file: location.file(),
                line: location.line(),
            },
            GpuCall::CopyToDevice => GpuError::CopyToDevice {
                msg: msg.into(),
                file: location.file(),
                line: location.line(),
            },
            GpuCall::CopyFromDevice => GpuError::CopyFromDevice {
                msg: msg.into(),
                file: location.file(),
                line: location.line(),
            },
        });
    }

    Ok(())
}

/// A Rust-managed pointer to CUDA device memory. When this is dropped,
/// [`gpuFree`] is called on the pointer.
#[derive(Debug)]
pub(crate) struct DevicePointer<T> {
    pub(crate) ptr: *mut T,

    /// The number of bytes allocated against `ptr`.
    size: usize,
}

impl<T> Drop for DevicePointer<T> {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe {
                gpuFree(self.ptr.cast());
            }
        }
    }
}

impl<T> DevicePointer<T> {
    /// Get a const pointer to the device memory.
    pub(crate) fn get(&self) -> *const T {
        self.ptr as *const T
    }

    /// Get a mutable pointer to the device memory.
    pub(crate) fn get_mut(&mut self) -> *mut T {
        self.ptr
    }

    /// The the number of bytes allocated in this [`DevicePointer`].
    pub(crate) fn get_size(&self) -> usize {
        self.size
    }

    /// Get the number of elements allocated against the buffer.
    pub(crate) fn get_num_elements(&self) -> usize {
        self.size / std::mem::size_of::<T>()
    }

    /// Allocate a number of bytes on the device.
    #[track_caller]
    pub(crate) fn malloc(size: usize) -> Result<DevicePointer<T>, GpuError> {
        if size == 0 {
            Ok(Self::default())
        } else {
            let mut d_ptr = std::ptr::null_mut();
            unsafe {
                gpuMalloc(&mut d_ptr, size);
                check_for_errors(GpuCall::Malloc)?;
            }
            Ok(Self {
                ptr: d_ptr.cast(),
                size,
            })
        }
    }

    /// Re-allocate a number of bytes on the device. Nothing is done if `size`
    /// is smaller than `self.size`. Note that unlike `libc`'s `remalloc`, if a
    /// new buffer is created, the original bytes are not preserved.
    #[track_caller]
    pub(crate) fn realloc(&mut self, size: usize) -> Result<(), GpuError> {
        if size <= self.size {
            return Ok(());
        }

        // CUDA/HIP don't provide a realloc, so just make a new `DevicePointer`
        // and swap it with the old one; the old buffer will be dropped.
        let mut new = Self::malloc(size)?;
        std::mem::swap(self, &mut new);
        Ok(())
    }

    /// Copy a slice of data to the device. Any type is allowed, and the returned
    /// pointer is to the device memory.
    #[track_caller]
    pub(crate) fn copy_to_device(v: &[T]) -> Result<DevicePointer<T>, GpuError> {
        let size = std::mem::size_of_val(v);
        unsafe {
            let mut d_ptr = Self::malloc(size)?;
            gpuMemcpy(
                d_ptr.get_mut().cast(),
                v.as_ptr().cast(),
                size,
                gpuMemcpyHostToDevice,
            );
            check_for_errors(GpuCall::CopyToDevice)?;
            Ok(d_ptr)
        }
    }

    /// Copy a slice of data from the device. There must be an equal number of
    /// bytes in the `DevicePointer` and `v`. The contents of `v` are
    /// overwritten.
    #[track_caller]
    pub fn copy_from_device(&self, v: &mut [T]) -> Result<(), GpuError> {
        let location = Location::caller();
        if self.ptr.is_null() {
            return Err(GpuError::CopyFromDevice {
                msg: "Attempted to copy data from a null device pointer".into(),
                file: location.file(),
                line: location.line(),
            });
        }

        let size = std::mem::size_of_val(v);
        if size != self.size {
            return Err(GpuError::CopyFromDevice {
                msg: format!(
                    "Device buffer size {} is not equal to provided buffer size {size} (length {})",
                    self.size,
                    v.len()
                )
                .into(),
                file: location.file(),
                line: location.line(),
            });
        }

        unsafe {
            gpuMemcpy(
                v.as_mut_ptr().cast(),
                self.ptr.cast(),
                size,
                gpuMemcpyDeviceToHost,
            );
            check_for_errors(GpuCall::CopyFromDevice)
        }
    }

    /// Overwrite the device memory allocated against this [`DevicePointer`]
    /// with new memory. If the amount of memory associated with `v` exceeds
    /// what is already allocated against the pointer, then the buffer is freed
    /// and another is created to fit `v` (i.e. re-alloc).
    #[track_caller]
    pub(crate) fn overwrite(&mut self, v: &[T]) -> Result<(), GpuError> {
        // Nothing to do if the collection is empty.
        if v.is_empty() {
            return Ok(());
        }

        let size = std::mem::size_of_val(v);
        self.realloc(size)?;
        unsafe {
            gpuMemcpy(
                self.get_mut() as *mut c_void,
                v.as_ptr().cast(),
                size,
                gpuMemcpyHostToDevice,
            );
            check_for_errors(GpuCall::CopyToDevice)
        }
    }

    /// Copy the contents of a [`DevicePointer`] to another one. The other one is realloc'd if necessary.
    #[track_caller]
    pub(crate) fn copy_to(&self, other: &mut DevicePointer<T>) -> Result<(), GpuError> {
        // Nothing to do if self is empty.
        if self.size == 0 {
            return Ok(());
        }

        other.realloc(self.size)?;
        unsafe {
            gpuMemcpy(
                other.get_mut().cast(),
                self.get().cast(),
                self.size,
                gpuMemcpyDeviceToDevice,
            );
            check_for_errors(GpuCall::CopyToDevice)
        }
    }

    /// Clear all of the bytes in the buffer by writing zeros.
    pub(crate) fn clear(&mut self) {
        #[cfg(feature = "cuda")]
        use cuda_runtime_sys::cudaMemset as gpuMemset;
        #[cfg(feature = "hip")]
        use hip_sys::hiprt::hipMemset as gpuMemset;

        unsafe {
            if self.size > 0 {
                gpuMemset(self.get_mut().cast(), 0, self.size);
            }
        }
    }
}

impl<T: Default> DevicePointer<T> {
    /// Copy a slice of data from the device. There must be an equal number of
    /// bytes in the `DevicePointer` and `v`.
    #[track_caller]
    pub fn copy_from_device_new(&self) -> Result<Vec<T>, GpuError> {
        if self.ptr.is_null() {
            let location = Location::caller();
            return Err(GpuError::CopyFromDevice {
                msg: "Attempted to copy data from a null device pointer".into(),
                file: location.file(),
                line: location.line(),
            });
        }

        let mut v: Vec<T> = Vec::default();
        v.resize_with(self.size / std::mem::size_of::<T>(), || T::default());

        unsafe {
            gpuMemcpy(
                v.as_mut_ptr().cast(),
                self.ptr.cast(),
                self.size,
                gpuMemcpyDeviceToHost,
            );
            check_for_errors(GpuCall::CopyFromDevice)?;
        }

        Ok(v)
    }
}

impl<T> Default for DevicePointer<T> {
    fn default() -> Self {
        Self {
            ptr: null_mut(),
            size: 0,
        }
    }
}

#[derive(Error, Debug)]
pub enum GpuError {
    #[cfg(feature = "cuda")]
    #[error("{file}:{line}: cudaMemcpy to device failed: {msg}")]
    CopyToDevice {
        msg: Box<str>,
        file: &'static str,
        line: u32,
    },

    #[cfg(feature = "hip")]
    #[error("{file}:{line}: hipMemcpy to device failed: {msg}")]
    CopyToDevice {
        msg: Box<str>,
        file: &'static str,
        line: u32,
    },

    #[cfg(feature = "cuda")]
    #[error("{file}:{line}: cudaMemcpy from device failed: {msg}")]
    CopyFromDevice {
        msg: Box<str>,
        file: &'static str,
        line: u32,
    },

    #[cfg(feature = "hip")]
    #[error("{file}:{line}: hipMemcpy from device failed: {msg}")]
    CopyFromDevice {
        msg: Box<str>,
        file: &'static str,
        line: u32,
    },

    #[cfg(feature = "cuda")]
    #[error("{file}:{line}: cudaMalloc error: {msg}")]
    Malloc {
        msg: Box<str>,
        file: &'static str,
        line: u32,
    },

    #[cfg(feature = "hip")]
    #[error("{file}:{line}: hipMalloc error: {msg}")]
    Malloc {
        msg: Box<str>,
        file: &'static str,
        line: u32,
    },

    #[cfg(feature = "cuda")]
    #[error("{file}:{line}: CUDA kernel error: {msg}")]
    Kernel {
        msg: Box<str>,
        file: &'static str,
        line: u32,
    },

    #[cfg(feature = "hip")]
    #[error("{file}:{line}: HIP kernel error: {msg}")]
    Kernel {
        msg: Box<str>,
        file: &'static str,
        line: u32,
    },

    #[cfg(feature = "cuda")]
    #[error("{file}:{line}: {msg}")]
    Generic {
        msg: Box<str>,
        file: &'static str,
        line: u32,
    },

    #[cfg(feature = "hip")]
    #[error("{file}:{line}: {msg}")]
    Generic {
        msg: Box<str>,
        file: &'static str,
        line: u32,
    },
}

// Suppress warnings for unused GPU shapelet consts.
mod unused {
    #[allow(unused)]
    fn unused() {
        use super::{SBF_C, SBF_DX, SBF_L, SBF_N};

        dbg!(SBF_L);
        dbg!(SBF_N);
        dbg!(SBF_C);
        dbg!(SBF_DX);
    }
}
