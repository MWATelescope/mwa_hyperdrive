// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! CUDA code to be used by hyperdrive, contained within its own crate.

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
        include!("peel_single.rs");
    } else if #[cfg(all(feature = "cuda", not(feature = "cuda-single")))] {
        /// f64 (using the "cuda" feature and not "cuda-single")
        pub(crate) type CudaFloat = f64;
        pub(crate) type CudaJones = JonesF64;

        include!("types_double.rs");
        include!("model_double.rs");
        include!("peel_double.rs");
    }
}

impl Default for UVW {
    fn default() -> Self {
        Self {
            u: 0.0,
            v: 0.0,
            w: 0.0,
        }
    }
}

#[derive(Clone, Copy)]
pub(crate) enum CudaCall {
    Malloc,
    CopyToDevice,
    CopyFromDevice,
}

/// Run [`cuda_runtime_sys::cudaPeekAtLastError`] and
/// [`cuda_runtime_sys::cudaDeviceSynchronize`]. If either of these calls return
/// an error, it is converted to a Rust error and returned from this function.
/// The single argument describes what the just-performed operation was and
/// makes the returned error a helpful one.
///
/// # Safety
///
/// This function interfaces directly with the CUDA API. Rust errors attempt to
/// catch problems but there are no guarantees.
pub(crate) unsafe fn peek_and_sync(cuda_call: CudaCall) -> Result<(), CudaError> {
    let code = cuda_runtime_sys::cudaPeekAtLastError();
    if code != cuda_runtime_sys::cudaError::cudaSuccess {
        let c_str = CStr::from_ptr(cuda_runtime_sys::cudaGetErrorString(code));
        let s = c_str.to_str().unwrap().to_string();
        return Err(match cuda_call {
            CudaCall::Malloc => CudaError::Malloc(s),
            CudaCall::CopyToDevice => CudaError::CopyToDevice(s),
            CudaCall::CopyFromDevice => CudaError::CopyFromDevice(s),
        });
    }

    let code = cuda_runtime_sys::cudaDeviceSynchronize();
    if code != cuda_runtime_sys::cudaError::cudaSuccess {
        let c_str = CStr::from_ptr(cuda_runtime_sys::cudaGetErrorString(code));
        let s = c_str.to_str().unwrap().to_string();
        return Err(match cuda_call {
            CudaCall::Malloc => CudaError::Malloc(s),
            CudaCall::CopyToDevice => CudaError::CopyToDevice(s),
            CudaCall::CopyFromDevice => CudaError::CopyFromDevice(s),
        });
    }

    Ok(())
}

/// A Rust-managed pointer to CUDA device memory. When this is dropped,
/// [`cuda_runtime_sys::cudaFree`] is called on the pointer.
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
    pub(crate) fn malloc(size: usize) -> Result<DevicePointer<T>, CudaError> {
        let mut d_ptr = std::ptr::null_mut();
        unsafe {
            cuda_runtime_sys::cudaMalloc(&mut d_ptr, size);
            peek_and_sync(CudaCall::Malloc)?;
        }
        Ok(Self {
            ptr: d_ptr.cast(),
            size,
        })
    }

    /// Re-allocate a number of bytes on the device. Nothing is done if `size`
    /// is smaller than `self.size`.
    pub(crate) fn realloc(&mut self, size: usize) -> Result<(), CudaError> {
        if size <= self.size {
            return Ok(());
        }

        // CUDA doesn't provide a realloc, so we have to free the original
        // pointer and make a new one.
        unsafe {
            if !self.get().is_null() {
                cuda_runtime_sys::cudaFree(self.get_mut().cast());
            }

            let mut d_ptr = std::ptr::null_mut();
            cuda_runtime_sys::cudaMalloc(&mut d_ptr, size);
            peek_and_sync(CudaCall::Malloc)?;
            self.ptr = d_ptr.cast();
            self.size = size;
        }
        Ok(())
    }

    /// Copy a slice of data to the device. Any type is allowed, and the returned
    /// pointer is to the device memory.
    pub(crate) fn copy_to_device(v: &[T]) -> Result<DevicePointer<T>, CudaError> {
        let size = v.len() * std::mem::size_of::<T>();
        unsafe {
            let mut d_ptr = Self::malloc(size)?;
            cuda_runtime_sys::cudaMemcpy(
                d_ptr.get_mut().cast(),
                v.as_ptr().cast(),
                size,
                cuda_runtime_sys::cudaMemcpyKind::cudaMemcpyHostToDevice,
            );
            peek_and_sync(CudaCall::CopyToDevice)?;
            Ok(d_ptr)
        }
    }

    /// Copy a slice of data from the device. There must be an equal number of
    /// bytes in the `DevicePointer` and `v`.
    #[track_caller]
    pub fn copy_from_device(&self, v: &mut [T]) -> Result<(), CudaError> {
        if self.ptr.is_null() {
            let loc = Location::caller();
            return Err(CudaError::CopyFromDevice(format!(
                "{}:{}:{}: Attempted to copy data from a null device pointer",
                loc.file(),
                loc.line(),
                loc.column()
            )));
        }

        let size = v.len() * std::mem::size_of::<T>();
        if size != self.size {
            return Err(CudaError::CopyFromDevice(format!(
                "Device buffer size {} is not equal to provided buffer size {size} (length {})",
                self.size,
                v.len()
            )));
        }

        unsafe {
            cuda_runtime_sys::cudaMemcpy(
                v.as_mut_ptr().cast(),
                self.ptr.cast(),
                size,
                cuda_runtime_sys::cudaMemcpyKind::cudaMemcpyDeviceToHost,
            );
            peek_and_sync(CudaCall::CopyFromDevice)
        }
    }

    /// Overwrite the device memory allocated against this [`DevicePointer`]
    /// with new memory. If the amount of memory associated with `v` exceeds
    /// what is already allocated against the pointer, then the buffer is freed
    /// and another is created to fit `v` (i.e. re-alloc).
    pub(crate) fn overwrite(&mut self, v: &[T]) -> Result<(), CudaError> {
        // Nothing to do if the collection is empty.
        if v.is_empty() {
            return Ok(());
        }

        let size = v.len() * std::mem::size_of::<T>();
        self.realloc(size)?;
        unsafe {
            cuda_runtime_sys::cudaMemcpy(
                self.get_mut() as *mut c_void,
                v.as_ptr().cast(),
                size,
                cuda_runtime_sys::cudaMemcpyKind::cudaMemcpyHostToDevice,
            );
            peek_and_sync(CudaCall::CopyToDevice)
        }
    }

    /// Clear all of the bytes in the buffer by writing zeros.
    pub(crate) fn clear(&mut self) {
        unsafe {
            if self.size > 0 {
                cuda_runtime_sys::cudaMemset(self.get_mut().cast(), 0, self.size);
            }
        }
    }
}

impl<T: Default> DevicePointer<T> {
    /// Copy a slice of data from the device. There must be an equal number of
    /// bytes in the `DevicePointer` and `v`.
    #[cfg(test)]
    #[track_caller]
    pub fn copy_from_device_new(&self) -> Result<Vec<T>, CudaError> {
        if self.ptr.is_null() {
            let loc = Location::caller();
            return Err(CudaError::CopyFromDevice(format!(
                "{}:{}:{}: Attempted to copy data from a null device pointer",
                loc.file(),
                loc.line(),
                loc.column()
            )));
        }

        let mut v: Vec<T> = Vec::default();
        v.resize_with(self.size / std::mem::size_of::<T>(), || T::default());

        unsafe {
            cuda_runtime_sys::cudaMemcpy(
                v.as_mut_ptr().cast(),
                self.ptr.cast(),
                self.size,
                cuda_runtime_sys::cudaMemcpyKind::cudaMemcpyDeviceToHost,
            );
            peek_and_sync(CudaCall::CopyFromDevice)?;
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
pub enum CudaError {
    // #[error("When overwriting, the new amount of memory did not equal the old amount")]
    // SizeMismatch,
    #[error("cudaMemcpy to device failed: {0}")]
    CopyToDevice(String),

    #[error("cudaMemcpy from device failed: {0}")]
    CopyFromDevice(String),

    #[error("cudaMalloc error: {0}")]
    Malloc(String),

    #[error("CUDA kernel error: {0}")]
    Kernel(String),

    #[error("{0}")]
    Generic(String),
}
