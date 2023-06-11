// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

use super::*;
use approx::assert_abs_diff_eq;
use ndarray::Array1;
use serial_test::serial;

#[test]
fn copy_to_and_from_device_succeeds() {
    const LEN: usize = 100;
    let heap = vec![0_u32; LEN];
    let d_ptr = DevicePointer::copy_to_device(&heap).unwrap();
    let mut heap2 = vec![1_u32; LEN];
    d_ptr.copy_from_device(&mut heap2).unwrap();
    assert_abs_diff_eq!(Array1::from(heap), Array1::from(heap2));

    let stack = [0_u32; LEN];
    let d_ptr = DevicePointer::copy_to_device(&stack).unwrap();
    let mut stack2 = [1_u32; LEN];
    d_ptr.copy_from_device(&mut stack2).unwrap();
    assert_abs_diff_eq!(Array1::from(stack.to_vec()), Array1::from(stack2.to_vec()));
}

#[test]
#[serial]
fn gpu_malloc_huge_fails() {
    let size = 1024_usize.pow(4); // 1 TB;
    let result: Result<DevicePointer<u8>, GpuError> = DevicePointer::malloc(size);
    assert!(result.is_err());
    let err = result.unwrap_err().to_string();
    #[cfg(feature = "cuda")]
    assert!(err.ends_with("cudaMalloc error: out of memory"), "{err}");
    #[cfg(feature = "hip")]
    assert!(
        err.contains("hipMalloc error"),
        "Error string wasn't expected; got: {err}"
    );
}

#[test]
fn copy_from_non_existent_pointer_fails() {
    let d_ptr: DevicePointer<u8> = DevicePointer {
        ptr: std::ptr::null_mut::<u8>(),
        size: 1,
    };
    let mut dest = [0; 100];
    let result = d_ptr.copy_from_device(&mut dest);
    assert!(result.is_err());
    let err = result.unwrap_err().to_string();
    #[cfg(feature = "cuda")]
    assert!(err.contains("cudaMemcpy from device failed"));
    #[cfg(feature = "hip")]
    assert!(err.contains("hipMemcpy from device failed"));
    assert!(err.contains("Attempted to copy data from a null device pointer"));
}

#[test]
fn clear_works() {
    let buffer = [1; 10];
    let mut d_ptr = DevicePointer::copy_to_device(&buffer).unwrap();
    // Check that the memory was definitely written correctly.
    let mut copy = [2; 10];
    d_ptr.copy_from_device(&mut copy).unwrap();
    assert_eq!(&buffer, &copy);

    // Now clear and check it's all zeros.
    d_ptr.clear();
    d_ptr.copy_from_device(&mut copy).unwrap();
    let expected = [0; 10];
    assert_eq!(&expected, &copy);
}
