// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Errors associated with interacting with CASA measurement sets.

use std::path::PathBuf;

use marlu::mwalib::MwalibError;
use thiserror::Error;

use mwa_hyperdrive_common::{marlu, thiserror};

#[derive(Error, Debug)]
pub enum MsReadError {
    #[error("Supplied file path {0} does not exist or is not readable!")]
    BadFile(PathBuf),

    #[error("The main table of the measurement set contains no rows!")]
    Empty,

    #[error("Couldn't work out the good start and end times of the measurement set; are all visibilities flagged?")]
    AllFlagged,

    #[error("No timesteps were in file {file}")]
    NoTimesteps { file: String },

    #[error("{0}")]
    GeneralMS(#[from] MSError),

    // // TODO: Kill failure
    // #[error("{0}")]
    // Casacore(#[from] rubbl_casatables::CasacoreError),
    #[error("{0}")]
    Glob(#[from] crate::glob::GlobError),

    #[error("{0}")]
    Mwalib(#[from] MwalibError),
}

#[derive(Error, Debug)]
pub enum MSError {
    #[error("Specified table name {0} does not exist")]
    TableDoesntExist(String),

    #[error("When reading in measurment set, ERFA function eraGd2gc failed to convert geocentric coordinates to geodetic. Is something wrong with your ANTENNA/POSITION column?")]
    Geodetic2Geocentric,

    #[error("Row {row_index} did not have 3 UVW elements!")]
    NotThreeUVW { row_index: u64 },

    #[error("MS {array_type} from {row_index} did not have expected {expected_len} elements on axis {axis_num}!")]
    BadArraySize {
        array_type: &'static str,
        row_index: u64,
        expected_len: usize,
        axis_num: usize,
    },

    #[error("Error when trying to interface with measurement set: {0}")]
    RubblError(String),
}
