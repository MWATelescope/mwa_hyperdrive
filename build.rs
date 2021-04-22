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
    let out_dir = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR env. variable not defined!"));

    write_built(&out_dir);
}
