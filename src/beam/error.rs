// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Errors associated with beam calculations.

use thiserror::Error;

use super::BEAM_TYPES_COMMA_SEPARATED;

#[derive(Error, Debug)]
pub enum BeamError {
    #[error(
        "Unrecognised beam model '{0}'; supported beam models are: {}",
        *BEAM_TYPES_COMMA_SEPARATED
    )]
    Unrecognised(String),

    #[error(
        "Tried to set up a '{0}' beam, which requires MWA dipole delays, but none are available"
    )]
    NoDelays(&'static str),

    #[error(
        "The specified MWA dipole delays aren't valid; there should be 16 values between 0 and 32"
    )]
    BadDelays,

    #[error("There are dipole delays specified for {num_rows} tiles, but when creating the beam object, {num_tiles} was specified as the number of tiles; refusing to continue")]
    InconsistentDelays { num_rows: usize, num_tiles: usize },

    #[error("The number of delays per tile ({delays}) didn't match the number of gains per tile ({gains})")]
    DelayGainsDimensionMismatch { delays: usize, gains: usize },

    #[error("Got tile index {got}, but the biggest tile index is {max}")]
    BadTileIndex { got: usize, max: usize },

    #[error("hyperbeam FEE error: {0}")]
    HyperbeamFee(#[from] mwa_hyperbeam::fee::FEEBeamError),

    #[error("hyperbeam init FEE error: {0}")]
    HyperbeamInitFee(#[from] mwa_hyperbeam::fee::InitFEEBeamError),

    #[error("hyperbeam analytic error: {0}")]
    HyperbeamAnalytic(#[from] mwa_hyperbeam::analytic::AnalyticBeamError),

    #[cfg(any(feature = "cuda", feature = "hip"))]
    #[error(transparent)]
    Gpu(#[from] crate::gpu::GpuError),
}
