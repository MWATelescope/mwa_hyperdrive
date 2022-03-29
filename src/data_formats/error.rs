// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Errors from building a `InputData` trait instance.

use birli::{marlu::SelectionError, BirliError};
use thiserror::Error;

use mwa_hyperdrive_common::thiserror;

#[derive(Error, Debug)]
pub enum ReadInputDataError {
    #[error("The supplied mwaf files don't have flags for timestep {timestep} (GPS time {gps})")]
    MwafFlagsMissingForTimestep { timestep: usize, gps: f64 },

    #[error("The specified PFB gains ('{pfb_flavour}') cannot be applied by Birli,\nprobably because this observation's frequency is incompatible ({freq_res_hz} Hz):\n{birli_error}")]
    PfbRefuse {
        pfb_flavour: String,
        freq_res_hz: f64,
        birli_error: BirliError,
    },

    #[error("Output {array_type} array did not have expected {expected_len} elements on axis {axis_num}")]
    BadArraySize {
        array_type: &'static str,
        expected_len: usize,
        axis_num: usize,
    },

    #[error(transparent)]
    Birli(#[from] BirliError),

    #[error(transparent)]
    MS(#[from] super::ms::MSError),

    #[error(transparent)]
    Uvfits(#[from] super::uvfits::UvfitsReadError),

    #[error(transparent)]
    SelectionError(#[from] SelectionError),

    #[error(transparent)]
    ObsContext(#[from] crate::context::ObsContextError),
}
