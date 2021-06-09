// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Error type for all hyperdrive-related errors.

use thiserror::Error;

#[derive(Error, Debug)]
pub enum HyperdriveError {
    #[error("Requested GPU processing, but the CUDA feature was not enabled when hyperdrive was compiled.")]
    NoGpuCompiled,

    #[error("{0}")]
    InvalidArgs(#[from] crate::calibrate::params::InvalidArgsError),

    #[error("{0}")]
    CalibrateArgs(#[from] crate::calibrate::args::CalibrateArgsError),

    #[error("{0}")]
    Calibrate(#[from] crate::calibrate::CalibrateError),

    #[error("{0}")]
    SimulateVis(#[from] crate::simulate_vis::SimulateVisError),
}
