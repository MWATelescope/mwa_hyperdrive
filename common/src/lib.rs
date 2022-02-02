// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Code to be shared between hyperdrive-related binaries.

mod logging;

pub use logging::*;

pub fn is_a_tty() -> bool {
    atty::is(atty::Stream::Stdout) || atty::is(atty::Stream::Stderr)
}

// Re-exports.
pub use {
    cfg_if, chrono, clap, fern, indicatif, itertools, lazy_static, log, marlu,
    marlu::{c32, c64, erfa_sys, hifitime, mwalib, ndarray, num_traits, rayon, Complex},
    serde_json, thiserror, toml,
};
