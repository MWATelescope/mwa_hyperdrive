// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Errors from building a `InputData` trait instance.

use birli::BirliError;
use marlu::SelectionError;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum VisReadError {
    #[error("The supplied mwaf files don't have flags for timestep {timestep} (GPS time {gps})")]
    MwafFlagsMissingForTimestep { timestep: usize, gps: f64 },

    #[error("Output {array_type} array did not have expected {expected_len} elements on axis {axis_num}")]
    BadArraySize {
        array_type: &'static str,
        expected_len: usize,
        axis_num: usize,
    },

    #[error(transparent)]
    InputFile(#[from] crate::filenames::InputFileError),

    #[error(transparent)]
    Raw(#[from] super::raw::RawReadError),

    #[error(transparent)]
    Birli(#[from] BirliError),

    #[error(transparent)]
    MS(#[from] super::ms::MsReadError),

    #[error(transparent)]
    Uvfits(#[from] super::uvfits::UvfitsReadError),

    #[error(transparent)]
    SelectionError(#[from] SelectionError),
}
