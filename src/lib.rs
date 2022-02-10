// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Calibration software for the Murchison Widefield Array (MWA) radio
//! telescope.

pub mod calibrate;
pub mod constants;
pub(crate) mod context;
pub mod data_formats;
pub(crate) mod error;
pub(crate) mod flagging;
pub(crate) mod glob;
pub mod math;
pub mod model;
pub(crate) mod pfb_gains;
pub(crate) mod shapelets;
pub mod simulate_vis;
pub(crate) mod time;
pub(crate) mod unit_parsing;

mwa_hyperdrive_common::cfg_if::cfg_if! {
    if #[cfg(test)] {
        mod jones_test;
        mod tests;
    }
}

// Re-exports.
pub use error::HyperdriveError;
