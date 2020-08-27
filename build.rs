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
}
