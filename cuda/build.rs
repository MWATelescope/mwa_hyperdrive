// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

use std::env;

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
            Err(_) => panic!("'{compute}', part of {var}, couldn't be parsed into a number!"),
        }
    }
    out
}

fn main() {
    println!("cargo:rerun-if-changed=build.rs");

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

    // Compile all CUDA source files into a single library. Find .cu, .h and
    // .cuh files; if any of them change, tell cargo to recompile.
    let mut cuda_files = vec![];
    for entry in std::fs::read_dir("src_cuda").expect("src_cuda directory doesn't exist!") {
        let entry = entry.expect("Couldn't access file in src_cuda directory");
        let path = entry.path();
        // Skip this entry if it isn't a file.
        if !path.is_file() {
            continue;
        }

        match path.extension().and_then(|os_str| os_str.to_str()) {
            // Track this file if it's extension is .cu
            Some("cu") => {
                println!("cargo:rerun-if-changed={}", path.display());
                // Add this .cu file to be compiled later.
                cuda_files.push(path);
            }

            Some("h" | "cuh") => {
                println!("cargo:rerun-if-changed={}", path.display());
            }

            _ => (),
        }
    }

    let mut cuda_target = cc::Build::new();
    cuda_target
        .cuda(true)
        .cudart("shared") // We handle linking cudart statically
        .define(
            // The DEBUG env. variable is set by cargo. If running "cargo build
            // --release", DEBUG is "false", otherwise "true". C/C++/CUDA like
            // the compile option "NDEBUG" to be defined when using assert.h, so
            // if appropriate, define that here. We also define "DEBUG" so that
            // can be used.
            match env::var("DEBUG").as_deref() {
                Ok("false") => "NDEBUG",
                _ => "DEBUG",
            },
            None,
        );

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

    // If we're told to, use single-precision floats. The default in the CUDA
    // code is to use double-precision.
    #[cfg(feature = "cuda-single")]
    cuda_target.define("SINGLE", None);

    cuda_target
        .file("src_cuda/model.cu")
        .file("src_cuda/utils.cu")
        .compile("hyperdrive_cu");
}
