// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Errors associated with interacting with CASA measurement sets.

use std::path::PathBuf;

use marlu::mwalib::MwalibError;
use thiserror::Error;

#[derive(Error, Debug)]
pub(crate) enum MsReadError {
    #[error("Supplied file path {0} does not exist or is not readable!")]
    BadFile(PathBuf),

    #[error("The main table of the measurement set contains no rows!")]
    Empty,

    #[error("The SPECTRAL_WINDOW table contained no channel frequencies")]
    NoChannelFreqs,

    #[error("The SPECTRAL_WINDOW table contained no channel widths")]
    NoChanWidths,

    #[error("The SPECTRAL_WINDOW table contains unequal channel widths")]
    ChanWidthsUnequal,

    #[error("Couldn't work out the good start and end times of the measurement set; are all visibilities flagged?")]
    AllFlagged,

    #[error("No timesteps were in file {file}")]
    NoTimesteps { file: String },

    #[error("When reading in measurement set, ERFA function eraGd2gc failed to convert geocentric coordinates to geodetic. Is something wrong with your ANTENNA/POSITION column?")]
    Geodetic2Geocentric,

    #[error("MS {array_type} from {row_index} did not have expected {expected_len} elements on axis {axis_num}!")]
    BadArraySize {
        array_type: &'static str,
        row_index: u64,
        expected_len: usize,
        axis_num: usize,
    },

    #[error("Error when trying to interface with measurement set: {0}")]
    RubblError(String),

    // // TODO: Kill failure
    // #[error(transparent)]
    // Casacore(#[from] rubbl_casatables::CasacoreError),
    #[error(transparent)]
    Glob(#[from] crate::glob::GlobError),

    #[error(transparent)]
    Mwalib(#[from] MwalibError),
}
