// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

/*!
Error type for all calibration-related errors.
 */

use thiserror::Error;

use super::params::InvalidArgsError;
use mwa_hyperdrive_core::{mwa_hyperbeam, EstimateError};

#[derive(Error, Debug)]
pub enum CalibrateError {
    #[error("{0}")]
    InvalidArgs(#[from] InvalidArgsError),

    #[error("{0}")]
    Estimate(#[from] EstimateError),

    #[error("{0}")]
    Hyperbeam(#[from] mwa_hyperbeam::fee::FEEBeamError),
}
