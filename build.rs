// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

fn main() {
    println!("cargo:rerun-if-changed=build.rs");

    #[cfg(all(feature = "cuda", feature = "hip"))]
    compile_error!("Both 'cuda' and 'hip' features are enabled; only one can be used.");
    #[cfg(all(not(feature = "cuda"), not(feature = "hip"), feature = "gpu-single"))]
    compile_error!(
        "The 'gpu-single' feature must be used with either of the 'cuda' or 'hip' features."
    );

    built::write_built_file().expect("Failed to acquire build-time information");

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

    /// Returns the minimum (major, minor) nvcc version required to compile for
    /// a given SM/arch target. Used to filter the default arch/SM lists so that
    /// targets unsupported by the installed toolkit are skipped gracefully
    /// rather than producing a cryptic nvcc error or a silent runtime failure.
    #[cfg(feature = "cuda")]
    fn min_nvcc_version_for_sm(sm: u16) -> (u32, u32) {
        match sm {
            0..=61 => (8, 0),   // Kepler / Maxwell / Pascal (sm_60, sm_61)
            62..=75 => (10, 0), // Volta (sm_70, sm_72), Turing (sm_75)
            76..=85 => (11, 0), // Ampere A100 (sm_80)
            86 => (11, 1),      // Ampere RTX 30-series (sm_86)
            87 => (11, 4),      // Ampere Jetson Orin (sm_87)
            88..=90 => (11, 8), // Ada Lovelace (sm_89), Hopper (sm_90)
            _ => (12, 8),       // Blackwell (sm_100, sm_110, sm_120) and beyond
        }
    }

    /// Queries the installed nvcc for its (major, minor) version, or returns
    /// None if nvcc is not found or the output cannot be parsed.
    #[cfg(feature = "cuda")]
    fn get_nvcc_version() -> Option<(u32, u32)> {
        let output = std::process::Command::new("nvcc")
            .arg("--version")
            .output()
            .ok()?;
        let text = String::from_utf8_lossy(&output.stdout);
        // Version line: "Cuda compilation tools, release 12.9, V12.9.86"
        for line in text.lines() {
            if let Some(idx) = line.find("release ") {
                let rest = &line[idx + 8..];
                let ver = rest.split(',').next()?.trim();
                let mut parts = ver.splitn(2, '.');
                let major: u32 = parts.next()?.parse().ok()?;
                let minor: u32 = parts.next()?.parse().ok()?;
                return Some((major, minor));
            }
        }
        None
    }

    pub(super) fn build_and_link() {
        let mut gpu_files = vec![];
        get_gpu_files("src/gpu", &mut gpu_files);

        #[cfg(feature = "cuda")]
        let mut gpu_target = {
            // sm_89 = Ada Lovelace (RTX 4000), sm_90 = Hopper, sm_100 = Blackwell HPC,
            // sm_110 = future Blackwell, sm_120 = Blackwell GeForce (RTX 5000).
            // Entries requiring a newer nvcc than is installed are filtered out below.
            const DEFAULT_CUDA_ARCHES: &[u16] = &[60, 70, 80, 86, 87, 89, 90, 100, 110, 120];
            const DEFAULT_CUDA_SMS: &[u16] = &[60, 61, 70, 75, 80, 86, 87, 89, 90, 100, 110, 120];

            fn parse_and_validate_compute(c: &str, var: &str) -> Vec<u16> {
                let mut out = vec![];
                for compute in c.trim().split(',') {
                    // Check that there are only two or three numeric characters
                    // (e.g. 86 for Ampere, 120 for Blackwell).
                    if compute.len() < 2 || compute.len() > 3 {
                        panic!("When parsing {var}, found '{compute}', which is not a two- or three-digit number!")
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

            // Gate arches and SMs against the installed nvcc version, skipping
            // any that require a newer toolkit than is present. Panics if
            // nothing survives — a clear message here is better than a cryptic
            // nvcc error or a silent "named symbol not found" at runtime.
            let nvcc_ver = get_nvcc_version().unwrap_or_else(|| {
                println!("cargo:warning=Could not determine nvcc version; skipping SM compatibility gating");
                (u32::MAX, u32::MAX)
            });
            println!(
                "cargo:warning=Detected nvcc version {}.{}",
                nvcc_ver.0, nvcc_ver.1
            );

            let filter_by_nvcc = |targets: Vec<u16>, label: &str| -> Vec<u16> {
                targets
                    .into_iter()
                    .filter(|&sm| {
                        let min = min_nvcc_version_for_sm(sm);
                        if nvcc_ver < min {
                            println!(
                                "cargo:warning=Skipping {label}={sm}: requires nvcc >= {}.{} \
                                 (have {}.{})",
                                min.0, min.1, nvcc_ver.0, nvcc_ver.1
                            );
                            false
                        } else {
                            true
                        }
                    })
                    .collect()
            };

            let arches = filter_by_nvcc(arches, "arch");
            let sms = filter_by_nvcc(sms, "sm");

            if arches.is_empty() {
                panic!(
                    "No CUDA arches remain after nvcc version gating (nvcc {}.{}). \
                     Either set HYPERDRIVE_CUDA_COMPUTE to a supported value or \
                     upgrade your CUDA toolkit.",
                    nvcc_ver.0, nvcc_ver.1
                );
            }

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

            // Capture the highest arch before the loop consumes the vec;
            // needed for the PTX fallback below.
            let max_arch = arches.iter().copied().max();

            // Loop over each arch and sm
            for arch in arches {
                for &sm in &sms {
                    if sm < arch {
                        continue;
                    }

                    cuda_target.flag("-gencode");
                    cuda_target.flag(format!("arch=compute_{arch},code=sm_{sm}"));
                }
            }

            // PTX fallback for the highest arch: allows JIT compilation on GPUs
            // newer than any explicitly targeted SM, avoiding another "named
            // symbol not found" when the next architecture generation arrives.
            if let Some(max) = max_arch {
                cuda_target.flag("-gencode");
                cuda_target.flag(format!("arch=compute_{max},code=compute_{max}"));
            }

            match env::var("DEBUG").as_deref() {
                Ok("false") => (),
                _ => {
                    cuda_target.flag("-G");
                }
            };

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
            println!("cargo:warning={compiler:?}");
            hip_target.compiler(compiler);

            println!("cargo:rerun-if-env-changed=HIP_FLAGS");
            if let Some(p) = env::var_os("HIP_FLAGS") {
                let s: String = p.to_string_lossy().into();
                println!("cargo:warning=HIP_FLAGS set from env {s}",);
                hip_target.flag(&s);
            }

            println!("cargo:rerun-if-env-changed=ROCM_VER");
            println!("cargo:rerun-if-env-changed=ROCM_PATH");
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
                println!("cargo:warning={gpu_target:?}");
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
