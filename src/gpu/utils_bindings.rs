/* automatically generated by rust-bindgen 0.68.1 */

extern "C" {
    #[doc = " A \"watered-down\" version of the CUDA example \"deviceQuery\".\n\n See the full example at:\nhttps://github.com/NVIDIA/cuda-samples/blob/master/Samples/1_Utilities/deviceQuery/deviceQuery.cpp\n\n As this code contains code derived from an official NVIDIA example, legally,\n a copyright, list of conditions and disclaimer must be distributed with this\n code. This should be found in the root of the mwa_hyperdrive git repo, file\n LICENSE-NVIDIA."]
    pub fn get_gpu_device_info(
        device: ::std::os::raw::c_int,
        name: *mut ::std::os::raw::c_char,
        device_major: *mut ::std::os::raw::c_int,
        device_minor: *mut ::std::os::raw::c_int,
        total_global_mem: *mut usize,
        driver_version: *mut ::std::os::raw::c_int,
        runtime_version: *mut ::std::os::raw::c_int,
    ) -> *const ::std::os::raw::c_char;
}