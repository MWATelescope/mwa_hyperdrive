// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

use std::env;
use std::path::{Path, PathBuf};

// Use the "built" crate to generate some useful build-time information,
// including the git hash and compiler version.
fn write_built(out_dir: &Path) {
    let mut opts = built::Options::default();
    opts.set_compiler(true)
        .set_git(true)
        .set_time(true)
        .set_ci(false)
        .set_env(false)
        .set_dependencies(false)
        .set_features(false)
        .set_cfg(false);
    built::write_built_file_with_opts(
        &opts,
        env::var("CARGO_MANIFEST_DIR").unwrap().as_ref(),
        &out_dir.join("built.rs"),
    )
    .expect("Failed to acquire build-time information");
}

fn main() {
    println!("cargo:rerun-if-changed=build.rs");

    #[cfg(all(feature = "cuda", feature = "hip"))]
    compile_error!("Both 'cuda' and 'hip' features are enabled; only one can be used.");
    #[cfg(all(not(feature = "cuda"), not(feature = "hip"), feature = "gpu-single"))]
    compile_error!(
        "The 'gpu-single' feature must be used with either of the 'cuda' or 'hip' features."
    );

    let out_dir = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR env. variable not defined!"));
    write_built(&out_dir);

    #[cfg(any(feature = "cuda", feature = "hip"))]
    gpu::build_and_link();
}

#[cfg(any(feature = "cuda", feature = "hip"))]
mod gpu {
    use std::{env, path::PathBuf};

    /// Search for any C/C++/CUDA/HIP files, populate the provided buffer with
    /// them, and have rerun-if-changed on all of them.
    #[cfg(any(feature = "cuda", feature = "hip"))]
    fn get_gpu_files<P: AsRef<std::path::Path>>(dir: P, files: &mut Vec<PathBuf>) {
        for path in std::fs::read_dir(dir).expect("dir exists") {
            let path = path.expect("is readable").path();
            if path.is_dir() {
                get_gpu_files(&path, files)
            }

            match path.extension().and_then(|os_str| os_str.to_str()) {
                Some("cu") => {
                    println!("cargo:rerun-if-changed={}", path.display());
                    files.push(path);
                }
                Some("h" | "cuh") => println!("cargo:rerun-if-changed={}", path.display()),
                _ => (),
            }
        }
    }

    pub(super) fn build_and_link() {
        let mut gpu_files = vec![];
        get_gpu_files("src/gpu", &mut gpu_files);

        #[cfg(feature = "cuda")]
        let mut gpu_target = {
            const DEFAULT_CUDA_ARCHES: &[u16] = &[60, 70, 80];
            const DEFAULT_CUDA_SMS: &[u16] = &[60, 70, 75, 80, 86];

            fn parse_and_validate_compute(c: &str, var: &str) -> Vec<u16> {
                let mut out = vec![];
                for compute in c.trim().split(',') {
                    // Check that there's only two numeric characters.
                    if compute.len() != 2 {
                        panic!("When parsing {var}, found '{compute}', which is not a two-digit number!")
                    }

                    match compute.parse() {
                        Ok(p) => out.push(p),
                        Err(_) => {
                            panic!("'{compute}', part of {var}, couldn't be parsed into a number!")
                        }
                    }
                }
                out
            }

            // Attempt to read HYPERDRIVE_CUDA_COMPUTE. HYPERBEAM_CUDA_COMPUTE can be
            // used instead, too.
            println!("cargo:rerun-if-env-changed=HYPERDRIVE_CUDA_COMPUTE");
            println!("cargo:rerun-if-env-changed=HYPERBEAM_CUDA_COMPUTE");
            let (arches, sms) = match (
                env::var("HYPERDRIVE_CUDA_COMPUTE"),
                env::var("HYPERBEAM_CUDA_COMPUTE"),
            ) {
                // When a user-supplied variable exists, use it as the CUDA arch and
                // compute level.
                (Ok(c), _) | (Err(_), Ok(c)) => {
                    let compute = parse_and_validate_compute(&c, "HYPERDRIVE_CUDA_COMPUTE");
                    let sms = compute.clone();
                    (compute, sms)
                }
                (Err(_), Err(_)) => {
                    // Print out all of the default arches and computes as a
                    // warning.
                    println!("cargo:warning=No HYPERDRIVE_CUDA_COMPUTE; Passing arch=compute_{DEFAULT_CUDA_ARCHES:?} and code=sm_{DEFAULT_CUDA_SMS:?} to nvcc");
                    (DEFAULT_CUDA_ARCHES.to_vec(), DEFAULT_CUDA_SMS.to_vec())
                }
            };

            let mut cuda_target = cc::Build::new();
            cuda_target.cuda(true).cudart("shared"); // We handle linking cudart statically

            // If $CXX is not set but $CUDA_PATH is, search for
            // $CUDA_PATH/bin/g++ and if it exists, set that as $CXX.
            if env::var_os("CXX").is_none() {
                // Unlike above, we care about $CUDA_PATH being unicode.
                if let Ok(cuda_path) = env::var("CUDA_PATH") {
                    // Look for the g++ that CUDA wants.
                    let compiler = std::path::PathBuf::from(cuda_path).join("bin/g++");
                    if compiler.exists() {
                        println!("cargo:warning=Setting $CXX to {}", compiler.display());
                        env::set_var("CXX", compiler.into_os_string());
                    }
                }
            }

            // Loop over each arch and sm
            for arch in arches {
                for &sm in &sms {
                    if sm < arch {
                        continue;
                    }

                    cuda_target.flag("-gencode");
                    cuda_target.flag(&format!("arch=compute_{arch},code=sm_{sm}"));
                }
            }

            cuda_target
        };

        #[cfg(feature = "hip")]
        let mut gpu_target = {
            println!("cargo:rerun-if-env-changed=HIP_PATH");
            let mut hip_path = match env::var_os("HIP_PATH") {
                Some(p) => {
                    println!(
                        "cargo:warning=HIP_PATH set from env {}",
                        p.to_string_lossy()
                    );
                    std::path::PathBuf::from(p)
                }
                None => {
                    let hip_path = hip_sys::hiprt::get_hip_path();
                    println!(
                        "cargo:warning=HIP_PATH set from hip_sys {}",
                        hip_path.display()
                    );
                    hip_path
                }
            };

            // It seems that various ROCm releases change where hipcc is...
            let mut compiler = hip_path.join("bin/hipcc");
            if !compiler.exists() {
                // Try the dir above, which might be the ROCm dir.
                hip_path = hip_path.parent().unwrap().into();
                compiler = hip_path.join("bin/hipcc");
                if !compiler.exists() {
                    panic!(
                        "Couldn't find hipcc in either {} or {}",
                        hip_sys::hiprt::get_hip_path().display(),
                        hip_path.parent().unwrap().display()
                    );
                }
            }
            if !hip_path.join("include/hip/hip_runtime_api.h").exists() {
                panic!(
                    "Couldn't find include/hip/hip_runtime_api.h in {}",
                    hip_path.display()
                );
            }

            let mut hip_target = cc::Build::new();
            hip_target
                .compiler(compiler)
                // .include(hip_path.join("include/hip"))
                // .include(hip_path.join("../include/hip"))
                ;

            println!("cargo:rerun-if-env-changed=HYPERBEAM_HIP_ARCH");
            println!("cargo:rerun-if-env-changed=HYPERDRIVE_HIP_ARCH");
            let arches: Vec<String> = match (
                env::var("HYPERBEAM_HIP_ARCH"),
                env::var("HYPERDRIVE_HIP_ARCH"),
            ) {
                // When a user-supplied variable exists, use it as the CUDA arch and
                // compute level.
                (Ok(c), _) | (Err(_), Ok(c)) => {
                    vec![c]
                }
                _ => {
                    // Print out all of the default arches and computes as a
                    // warning.
                    println!("cargo:warning=No offload arch found, try HYPERBEAM_HIP_ARCH");
                    vec![]
                }
            };

            for arch in arches {
                hip_target.flag(&format!("--offload-arch={arch}"));
            }

            match env::var("DEBUG").as_deref() {
                Ok("false") => (),
                _ => {
                    hip_target
                        .flag("-ggdb")
                        .flag("-O1") // <- don't use -O0 https://github.com/ROCm/HIP/issues/3183
                        .flag("-gmodules");
                }
            };

            hip_target
        };

        // The DEBUG env. variable is set by cargo. If running "cargo build
        // --release", DEBUG is "false", otherwise "true". C/C++/CUDA like
        // the compile option "NDEBUG" to be defined when using assert.h, so
        // if appropriate, define that here. We also define "DEBUG" so that
        // can be used.
        match env::var("DEBUG").as_deref() {
            Ok("false") | Ok("0") => {
                gpu_target.define("NDEBUG", "");
            }
            _ => {
                gpu_target.define("DEBUG", "").flag("-v");
            }
        };

        // If we're told to, use single-precision floats. The default in the GPU
        // code is to use double-precision.
        #[cfg(feature = "gpu-single")]
        gpu_target.define("SINGLE", None);

        // Break in case of emergency.
        // gpu_target.debug(true);

        for f in gpu_files {
            gpu_target.file(f);
        }
        gpu_target.compile("hyperdrive_gpu");
    }
}
