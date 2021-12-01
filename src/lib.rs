// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Calibration software for the Murchison Widefield Array (MWA) radio
//! telescope.

pub mod calibrate;
pub mod constants;
pub mod model;
pub mod simulate_vis;

pub(crate) mod context;
pub mod data_formats;
pub(crate) mod error;
pub(crate) mod flagging;
pub(crate) mod glob;
pub(crate) mod math;
pub(crate) mod pfb_gains;
pub(crate) mod shapelets;
pub(crate) mod unit_parsing;

#[cfg(test)]
pub(crate) mod tests;

// Re-exports.
pub(crate) use constants::*;
pub use error::HyperdriveError;
pub(crate) use mwa_rust_core::*;
