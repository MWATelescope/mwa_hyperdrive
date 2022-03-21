// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Calibration software for the Murchison Widefield Array (MWA) radio
//! telescope.
//!
//! <https://mwatelescope.github.io/mwa_hyperdrive/index.html>

pub mod averaging;
pub mod calibrate;
pub mod constants;
pub(crate) mod context;
pub(crate) mod error;
pub(crate) mod filenames;
pub(crate) mod flagging;
pub(crate) mod glob;
mod help_texts;
pub mod math;
pub(crate) mod messages;
pub mod metafits;
pub mod model;
pub(crate) mod pfb_gains;
pub mod solutions;
pub(crate) mod time;
pub(crate) mod unit_parsing;
pub mod utilities;
pub mod vis_io;
pub mod vis_utils;

mwa_hyperdrive_common::cfg_if::cfg_if! {
    if #[cfg(test)] {
        mod jones_test;
        mod tests;
    }
}

// Re-exports.
pub use error::HyperdriveError;
