// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! CUDA code to be used by hyperdrive.

#![allow(non_snake_case)]
#![allow(clippy::upper_case_acronyms)]

#[cfg(test)]
mod tests;
mod utils;

use std::{
    ffi::{c_void, CStr},
    panic::Location,
    ptr::null_mut,
};

use thiserror::Error;

pub(crate) use utils::{get_device_info, CudaDeviceInfo, CudaDriverInfo};

// Import Rust bindings to the CUDA code specific to the precision we're using,
// and set corresponding compile-time types.
cfg_if::cfg_if! {
    if #[cfg(feature = "cuda-single")] {
        /// f32 (using the "cuda-single" feature)
        pub(crate) type CudaFloat = f32;
        pub(crate) type CudaJones = JonesF32;

        include!("types_single.rs");
        include!("model_single.rs");
    } else if #[cfg(all(feature = "cuda", not(feature = "cuda-single")))] {
        /// f64 (using the "cuda" feature and not "cuda-single")
        pub(crate) type CudaFloat = f64;
        pub(crate) type CudaJones = JonesF64;

        include!("types_double.rs");
        include!("model_double.rs");
    }
}

// Ensure that the shapelet constants are the same in the Rust code and CUDA
// code.
static_assertions::const_assert_eq!(crate::model::shapelets::SBF_L as i32, SBF_L);
static_assertions::const_assert_eq!(crate::model::shapelets::SBF_N as i32, SBF_N);
static_assertions::const_assert_eq!(crate::model::shapelets::SBF_C as CudaFloat, SBF_C);
static_assertions::const_assert_eq!(crate::model::shapelets::SBF_DX as CudaFloat, SBF_DX);

macro_rules! cuda_kernel_call {
    ($cuda_fn:path, $($args:expr),* $(,)?) => {{
        #[allow(unused_unsafe)]
        unsafe {
            let error_message_ptr = $cuda_fn($($args),*);
            if error_message_ptr.is_null() {
                Ok(())
            } else {
                // Get the CUDA error message behind the pointer.
                let error_message = std::ffi::CStr::from_ptr(error_message_ptr).to_str().unwrap_or("<cannot read CUDA error string>");
                let our_error_message = format!("{}: {error_message}", stringify!($cuda_fn));
                Err(CudaError::Kernel {
                    msg: our_error_message.into(),
                    file: file!(),
                    line: line!(),
                })
            }
        }
    }};
}
pub(crate) use cuda_kernel_call;

#[derive(Clone, Copy)]
pub(crate) enum CudaCall {
    Malloc,
    CopyToDevice,
    CopyFromDevice,
}

/// Run [`cuda_runtime_sys::cudaGetLastError`] and
/// [`cuda_runtime_sys::cudaDeviceSynchronize`]. If either of these calls return
/// an error, it is converted to a Rust error and returned from this function.
/// The single argument describes what the just-performed operation was and
/// makes the returned error a helpful one.
///
/// # Safety
///
/// This function interfaces directly with the CUDA API. Rust errors attempt to
/// catch problems but there are no guarantees.
#[track_caller]
unsafe fn check_for_errors(cuda_call: CudaCall) -> Result<(), CudaError> {
    // Only do a device sync if we're in debug mode, for performance.
    let debug_mode = matches!(std::env::var("DEBUG").as_deref(), Ok("true"));
    if debug_mode {
        let code = cuda_runtime_sys::cudaDeviceSynchronize();
        if code != cuda_runtime_sys::cudaError::cudaSuccess {
            let c_str = CStr::from_ptr(cuda_runtime_sys::cudaGetErrorString(code));
            let msg = c_str.to_str().unwrap_or("<cannot read CUDA error string>");
            let location = Location::caller();
            return Err(match cuda_call {
                CudaCall::Malloc => CudaError::Malloc {
                    msg: msg.into(),
                    file: location.file(),
                    line: location.line(),
                },
                CudaCall::CopyToDevice => CudaError::CopyToDevice {
                    msg: msg.into(),
                    file: location.file(),
                    line: location.line(),
                },
                CudaCall::CopyFromDevice => CudaError::CopyFromDevice {
                    msg: msg.into(),
                    file: location.file(),
                    line: location.line(),
                },
            });
        }
    }

    let code = cuda_runtime_sys::cudaGetLastError();
    if code != cuda_runtime_sys::cudaError::cudaSuccess {
        let c_str = CStr::from_ptr(cuda_runtime_sys::cudaGetErrorString(code));
        let msg = c_str.to_str().unwrap_or("<cannot read CUDA error string>");
        let location = Location::caller();
        return Err(match cuda_call {
            CudaCall::Malloc => CudaError::Malloc {
                msg: msg.into(),
                file: location.file(),
                line: location.line(),
            },
            CudaCall::CopyToDevice => CudaError::CopyToDevice {
                msg: msg.into(),
                file: location.file(),
                line: location.line(),
            },
            CudaCall::CopyFromDevice => CudaError::CopyFromDevice {
                msg: msg.into(),
                file: location.file(),
                line: location.line(),
            },
        });
    }

    Ok(())
}

/// A Rust-managed pointer to CUDA device memory. When this is dropped,
/// [`cuda_runtime_sys::cudaFree`] is called on the pointer.
#[derive(Debug)]
pub(crate) struct DevicePointer<T> {
    ptr: *mut T,

    /// The number of bytes allocated against `ptr`.
    size: usize,
}

impl<T> Drop for DevicePointer<T> {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe {
                cuda_runtime_sys::cudaFree(self.ptr.cast());
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

    /// Allocate a number of bytes on the device.
    #[track_caller]
    pub(crate) fn malloc(size: usize) -> Result<DevicePointer<T>, CudaError> {
        if size == 0 {
            Ok(Self::default())
        } else {
            let mut d_ptr = std::ptr::null_mut();
            unsafe {
                cuda_runtime_sys::cudaMalloc(&mut d_ptr, size);
                check_for_errors(CudaCall::Malloc)?;
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
    pub(crate) fn realloc(&mut self, size: usize) -> Result<(), CudaError> {
        if size <= self.size {
            return Ok(());
        }

        // CUDA doesn't provide a realloc, so just make a new `DevicePointer`
        // and swap it with the old one; the old buffer will be dropped.
        let mut new = Self::malloc(size)?;
        std::mem::swap(self, &mut new);
        Ok(())
    }

    /// Copy a slice of data to the device. Any type is allowed, and the returned
    /// pointer is to the device memory.
    #[track_caller]
    pub(crate) fn copy_to_device(v: &[T]) -> Result<DevicePointer<T>, CudaError> {
        let size = std::mem::size_of_val(v);
        unsafe {
            let mut d_ptr = Self::malloc(size)?;
            cuda_runtime_sys::cudaMemcpy(
                d_ptr.get_mut().cast(),
                v.as_ptr().cast(),
                size,
                cuda_runtime_sys::cudaMemcpyKind::cudaMemcpyHostToDevice,
            );
            check_for_errors(CudaCall::CopyToDevice)?;
            Ok(d_ptr)
        }
    }

    /// Copy a slice of data from the device. There must be an equal number of
    /// bytes in the `DevicePointer` and `v`. The contents of `v` are
    /// overwritten.
    #[track_caller]
    pub fn copy_from_device(&self, v: &mut [T]) -> Result<(), CudaError> {
        let location = Location::caller();
        if self.ptr.is_null() {
            return Err(CudaError::CopyFromDevice {
                msg: "Attempted to copy data from a null device pointer".into(),
                file: location.file(),
                line: location.line(),
            });
        }

        let size = std::mem::size_of_val(v);
        if size != self.size {
            return Err(CudaError::CopyFromDevice {
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
            cuda_runtime_sys::cudaMemcpy(
                v.as_mut_ptr().cast(),
                self.ptr.cast(),
                size,
                cuda_runtime_sys::cudaMemcpyKind::cudaMemcpyDeviceToHost,
            );
            check_for_errors(CudaCall::CopyFromDevice)
        }
    }

    /// Overwrite the device memory allocated against this [`DevicePointer`]
    /// with new memory. If the amount of memory associated with `v` exceeds
    /// what is already allocated against the pointer, then the buffer is freed
    /// and another is created to fit `v` (i.e. re-alloc).
    #[track_caller]
    pub(crate) fn overwrite(&mut self, v: &[T]) -> Result<(), CudaError> {
        // Nothing to do if the collection is empty.
        if v.is_empty() {
            return Ok(());
        }

        let size = std::mem::size_of_val(v);
        self.realloc(size)?;
        unsafe {
            cuda_runtime_sys::cudaMemcpy(
                self.get_mut() as *mut c_void,
                v.as_ptr().cast(),
                size,
                cuda_runtime_sys::cudaMemcpyKind::cudaMemcpyHostToDevice,
            );
            check_for_errors(CudaCall::CopyToDevice)
        }
    }

    /// Clear all of the bytes in the buffer by writing zeros.
    #[cfg(test)]
    pub(crate) fn clear(&mut self) {
        unsafe {
            if self.size > 0 {
                cuda_runtime_sys::cudaMemset(self.get_mut().cast(), 0, self.size);
            }
        }
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
pub enum CudaError {
    #[error("{file}:{line}: cudaMemcpy to device failed: {msg}")]
    CopyToDevice {
        msg: Box<str>,
        file: &'static str,
        line: u32,
    },

    #[error("{file}:{line}: cudaMemcpy from device failed: {msg}")]
    CopyFromDevice {
        msg: Box<str>,
        file: &'static str,
        line: u32,
    },

    #[error("{file}:{line}: cudaMalloc error: {msg}")]
    Malloc {
        msg: Box<str>,
        file: &'static str,
        line: u32,
    },

    #[error("{file}:{line}: CUDA kernel error: {msg}")]
    Kernel {
        msg: Box<str>,
        file: &'static str,
        line: u32,
    },

    #[error("{file}:{line}: {msg}")]
    Generic {
        msg: Box<str>,
        file: &'static str,
        line: u32,
    },
}

// Suppress warnings for unused CUDA shapelet consts.
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
