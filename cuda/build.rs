// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

use std::env;

// This code is adapted from pkg-config-rs
// (https://github.com/rust-lang/pkg-config-rs).
#[allow(clippy::if_same_then_else, clippy::needless_bool)]
fn infer_static(name: &str) -> bool {
    if env::var(&format!("{}_STATIC", name.to_uppercase())).is_ok() {
        true
    } else if env::var(&format!("{}_DYNAMIC", name.to_uppercase())).is_ok() {
        false
    } else if env::var("PKG_CONFIG_ALL_STATIC").is_ok() {
        true
    } else if env::var("PKG_CONFIG_ALL_DYNAMIC").is_ok() {
        false
    } else {
        false
    }
}

fn main() {
    // Attempt to read HYPERDRIVE_CUDA_COMPUTE.
    let compute = match env::var("HYPERDRIVE_CUDA_COMPUTE") {
        Ok(c) => c,
        Err(e) => panic!(
            "Problem reading env. variable HYPERDRIVE_CUDA_COMPUTE ! {}",
            e
        ),
    };
    // Check that there's only two numeric characters.
    if compute.parse::<u16>().is_err() {
        panic!("HYPERDRIVE_CUDA_COMPUTE couldn't be parsed into a number!")
    }
    if compute.len() != 2 {
        panic!("HYPERDRIVE_CUDA_COMPUTE is not a two-digit number!")
    }

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

        // Continue if this file's extension is .cu
        match path.extension().and_then(|os_str| os_str.to_str()) {
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
        .flag("-cudart=static")
        .flag("-gencode")
        // Using the specified HYPERDRIVE_CUDA_COMPUTE
        .flag(&format!("arch=compute_{c},code=sm_{c}", c = compute))
        .define(
            // The DEBUG env. variable is set by cargo. If running "cargo build
            // --release", DEBUG is "false", otherwise "true". C/C++/CUDA like
            // the compile option "NDEBUG" to be defined when using assert.h, so
            // if appropriate, define that here. We also define "DEBUG" so that
            // could be used.
            match env::var("DEBUG").as_deref() {
                Ok("false") => "NDEBUG",
                _ => "DEBUG",
            },
            None,
        )
        .files(&cuda_files)
        .compile("hyperdrive_cu");

    // Link CUDA. If the library path manually specified, search there.
    if let Ok(lib_dir) = env::var("CUDA_LIB") {
        println!("cargo:rustc-link-search=native={}", lib_dir);
    }

    if infer_static("cuda") {
        // CUDA ships its static library as cudart_static.a, not cudart.a
        println!("cargo:rustc-link-lib=static=cudart_static");
    } else {
        println!("cargo:rustc-link-lib=cudart");
    }
}
