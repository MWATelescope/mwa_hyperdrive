// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

/*!
Calibration software for the Murchison Widefield Array (MWA) radio telescope.
 */

pub mod constants;
pub mod context;
pub mod coord;
pub mod flagging;
pub mod foreign;
pub mod instrument;
pub(crate) mod math;
pub mod sourcelist;
pub mod visibility_gen;

// Re-exports.
pub use constants::*;
pub use context::Context;
pub use coord::*;
pub use foreign::*;
pub use instrument::PrimaryBeam;
pub(crate) use math::*;
pub use sourcelist::flux_density::FluxDensity;
pub use sourcelist::source::{Source, SourceComponent};

pub use std::path::{Path, PathBuf};

// External re-exports.
pub use fitsio::FitsFile;
pub use log::{debug, error, info, trace, warn};
pub use mwalib::*;
use num::Complex;
