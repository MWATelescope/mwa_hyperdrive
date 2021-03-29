// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

/*!
Calibration software for the Murchison Widefield Array (MWA) radio telescope.
 */

pub mod calibrate;
pub mod visibility_gen;

pub(crate) mod constants;
pub(crate) mod context;
pub(crate) mod data_formats;
pub(crate) mod flagging;
pub(crate) mod glob;
pub(crate) mod math;

// Re-exports.
pub(crate) use constants::*;
pub(crate) use mwa_hyperdrive_core::*;
