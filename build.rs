// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

use std::env;
use std::{
    fs::File,
    path::{Path, PathBuf},
};

fn bind_erfa(out_dir: &Path) {
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
                .write_to_file(&out_dir.join("erfa.rs"))
                .expect("Couldn't write bindings");
        }
        Err(_) => panic!("Couldn't find the ERFA library via pkg-config"),
    };
}

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
    let out_dir = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR env. variable not defined!"));

    // This block of code forces hyperdrive to recompile its binaries everytime
    // we do a release build.
    if env::var("DEBUG").unwrap() == "false" {
        let p = Path::new(&out_dir).join("rebuild_stamp");
        File::create(&p).unwrap();
        println!("cargo:rerun-if-changed={}", p.display());
    }

    write_built(&out_dir);
    bind_erfa(&out_dir);
}
