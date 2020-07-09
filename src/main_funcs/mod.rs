// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

pub(crate) mod simulate_vis;
pub(crate) mod verify;

// Re-exports.
pub(crate) use simulate_vis::*;
pub(crate) use verify::*;

use std::fs::File;
use std::io::Read;

use anyhow::bail;

// Add build-time information from the "built" crate.
include!(concat!(env!("OUT_DIR"), "/built.rs"));

// Not sure how to format this string nicely without the "lazy_static" crate.
use lazy_static::lazy_static;
lazy_static! {
    /// A formatted string detailing which git commit of hyperdrive was used,
    /// what compiler version was used, and when the executable was built.
    pub static ref HYPERDRIVE_VERSION: String =
        format!(r#"{ver}
Compiled on git commit hash: {hash}{dirty}
                head ref:    {head_ref}
         at: {time}
         with compiler: {compiler}"#,
                ver = env!("CARGO_PKG_VERSION"),
                hash = GIT_COMMIT_HASH.unwrap_or("<no git info>"),
                dirty = match GIT_DIRTY {
                    Some(true) => " (dirty)",
                    _ => "",
                },
                head_ref = GIT_HEAD_REF.unwrap_or("<no git info>"),
                time = BUILT_TIME_UTC,
                compiler = RUSTC_VERSION,
        );
    // These lines prevent warnings about unused built variables.
    static ref _RUSTDOC_VERSION: &'static str = RUSTDOC_VERSION;
    static ref _GIT_VERSION: Option<&'static str> = GIT_VERSION;
}
