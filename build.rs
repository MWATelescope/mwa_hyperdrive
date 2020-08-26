// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

use std::env;
use std::path::{Path, PathBuf};

fn bind_erfa() {
    match pkg_config::probe_library("erfa") {
        Ok(lib) => {
            // Find erfa.h
            let mut erfa_include: Option<_> = None;
            for mut inc_path in lib.include_paths {
                inc_path.push("erfa.h");
                if inc_path.exists() {
                    erfa_include = Some(inc_path.to_str().unwrap().to_string());
                    break;
                }
            }

            bindgen::builder()
                .header(erfa_include.expect("Couldn't find erfa.h in pkg-config include dirs"))
                .whitelist_function("eraSeps")
                .whitelist_function("eraHd2ae")
                .whitelist_function("eraAe2hd")
                .generate()
                .expect("Unable to generate bindings")
                .write_to_file(PathBuf::from(env::var("OUT_DIR").unwrap()).join("erfa.rs"))
                .expect("Couldn't write bindings");
        }
        Err(_) => panic!("Couldn't find the ERFA library via pkg-config"),
    };
}

// Use the "built" crate to generate some useful build-time information,
// including the git hash and compiler version.
fn write_built() {
    use built::*;
    let mut opts = Options::default();
    opts.set_compiler(true)
        .set_git(true)
        .set_ci(false)
        .set_env(false)
        .set_dependencies(false)
        .set_features(false)
        .set_time(true)
        .set_cfg(false);
    built::write_built_file_with_opts(
        &opts,
        env::var("CARGO_MANIFEST_DIR").unwrap().as_ref(),
        &Path::new(&env::var("OUT_DIR").unwrap()).join("built.rs"),
    )
    .expect("Failed to acquire build-time information");
}

fn main() {
    write_built();

    bind_erfa();

    // Build bindings to C/C++ functions.
    let bindings = bindgen::Builder::default()
        .header("src_cuda/vis_gen.h")
        // Invalidate the built crate whenever any of the included header files
        // changed.
        .parse_callbacks(Box::new(bindgen::CargoCallbacks))
        .generate()
        .expect("Unable to generate bindings");

    // Write the bindings to the $OUT_DIR/bindings.rs file.
    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
    bindings
        .write_to_file(out_path.join("bindings.rs"))
        .expect("Couldn't write bindings!");

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

    // Compile src_cuda/vis_gen.cu
    println!("cargo:rerun-if-changed=src_cuda/vis_gen.cu");
    cc::Build::new()
        .cuda(true)
        .flag("-cudart=static")
        .flag("-gencode")
        // Using the specified HYPERDRIVE_CUDA_COMPUTE
        .flag(format!("arch=compute_{c},code=sm_{c}", c = compute).as_str())
        .define(
            // The DEBUG env. variable is set by cargo. If running "cargo build
            // --release", DEBUG is "false", otherwise "true". C/C++/CUDA like
            // the compile option "NDEBUG" to be defined when using assert.h, so
            // if appropriate, define that here. "DEBUG" is also defined, and
            // could also be used.
            if env::var("DEBUG").unwrap() == "false" {
                "NDEBUG"
            } else {
                "DEBUG"
            },
            None,
        )
        .file("src_cuda/vis_gen.cu")
        .compile("hyperdrive_cu");

    // Use the following search paths when linking.
    // CUDA could be installed in a couple of places, and use "lib" or "lib64";
    // specify all combinations.
    for path in &["/usr/local/cuda", "/opt/cuda"] {
        for lib_path in &["lib", "lib64"] {
            println!("cargo:rustc-link-search=native={}/{}", path, lib_path);
        }
    }
    // Link with the dynamic cudart library
    println!("cargo:rustc-link-lib=cudart");
}
