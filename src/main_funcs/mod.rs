// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

pub(crate) mod simulate_vis;
pub(crate) mod verify_srclist;

// Re-exports.
use std::fs::File;
use std::io::Read;
use std::path::PathBuf;

use anyhow::bail;

pub(crate) use simulate_vis::*;
pub(crate) use verify_srclist::*;

// Add build-time information from the "built" crate.
include!(concat!(env!("OUT_DIR"), "/built.rs"));

// Not sure how to format this string nicely without the "lazy_static" crate.
use lazy_static::lazy_static;
lazy_static! {
    /// A formatted string detailing which git commit of hyperdrive was used,
    /// what compiler version was used, and when the executable was built.
    pub static ref HYPERDRIVE_VERSION: String = format!(
        "Compiled on git commit {git} with {compiler} on {time}",
        git = GIT_VERSION.unwrap_or("<no git info>"),
        compiler = RUSTC_VERSION,
        time = BUILT_TIME_UTC
    );
    // Ignore the RUSTDOC_VERSION; this line prevents a warning about
    // `RUSTDOC_VERSION` being unused.
    static ref _DOC: &'static str = RUSTDOC_VERSION;
}
