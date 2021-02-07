// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

/*!
Calibration software for the Murchison Widefield Array (MWA) radio telescope.
 */

pub mod calibrate;
pub mod constants;
pub mod context;
pub mod flagging;
pub mod glob;
pub(crate) mod math;
pub mod visibility_gen;

// Re-exports.
pub use constants::*;
pub use context::Context;
pub use flagging::cotter::CotterFlags;
pub(crate) use math::*;
pub use mwa_hyperdrive_core::*;
pub use mwa_hyperdrive_srclist::flux_density::FluxDensity;

pub use std::fs::File;
pub use std::io::Read;
pub use std::path::{Path, PathBuf};

// External re-exports.
pub use log::{debug, error, info, trace, warn};
pub use mwalib::*;
pub use num::complex::Complex64 as c64;
