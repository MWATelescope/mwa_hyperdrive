// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Calibration software for the Murchison Widefield Array (MWA) radio
//! telescope.
//!
//! <https://mwatelescope.github.io/mwa_hyperdrive/index.html>

// TODO: #![warn(clippy::too_many_lines)]

mod averaging;
mod beam;
mod cli;
mod constants;
mod context;
mod di_calibrate;
mod flagging;
mod io;
mod math;
mod metafits;
mod misc;
pub mod model;
mod params;
mod solutions;
pub mod srclist;
mod unit_parsing;

#[cfg(any(feature = "cuda", feature = "hip"))]
mod gpu;

#[cfg(test)]
mod tests;

use crossbeam_utils::atomic::AtomicCell;
lazy_static::lazy_static! {
    /// Are progress bars being drawn? This should only ever be enabled by CLI
    /// code.
    static ref PROGRESS_BARS: AtomicCell<bool> = AtomicCell::new(false);

    /// What device (GPU or CPU) are we using for modelling and beam responses?
    /// This should only ever be changed from its default by CLI code.
    static ref MODEL_DEVICE: AtomicCell<model::ModelDevice> = {
        cfg_if::cfg_if! {
            if #[cfg(any(feature = "cuda", feature = "hip"))] {
                AtomicCell::new(ModelDevice::Gpu)
            } else {
                AtomicCell::new(ModelDevice::Cpu)
            }
        }
    };
}

// Re-exports.
pub use averaging::{Chanblock, Timeblock};
pub use beam::{create_beam_object, Delays};
#[doc(hidden)]
pub use cli::Hyperdrive;
pub use cli::HyperdriveError;
pub use context::Polarisations;
pub use di_calibrate::calibrate_timeblocks;
pub use io::read::{CrossData, MsReader, RawDataCorrections, RawDataReader, UvfitsReader};
pub use math::TileBaselineFlags;
pub use model::ModelDevice;
pub use solutions::CalibrationSolutions;
