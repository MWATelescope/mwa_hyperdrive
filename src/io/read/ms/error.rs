// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Errors associated with interacting with CASA measurement sets.

use std::path::PathBuf;

use marlu::rubbl_casatables;
use thiserror::Error;

#[derive(Error, Debug)]
pub(crate) enum MsReadError {
    #[error("Supplied file path {0} does not exist or is not readable!")]
    BadFile(PathBuf),

    #[error("The main table of the measurement set contains no rows!")]
    MainTableEmpty,

    #[error("The antenna table of the measurement set contains no rows!")]
    AntennaTableEmpty,

    #[error("The SPECTRAL_WINDOW table contained no channel frequencies")]
    NoChannelFreqs,

    #[error("The SPECTRAL_WINDOW table contained no channel widths")]
    NoChanWidths,

    #[error("The SPECTRAL_WINDOW table contains unequal channel widths")]
    ChanWidthsUnequal,

    #[error("No timesteps were in file {file}")]
    NoTimesteps { file: String },

    #[error("MS {array_type} from {row_index} did not have expected {expected_len} elements on axis {axis_num}!")]
    BadArraySize {
        array_type: &'static str,
        row_index: u64,
        expected_len: usize,
        axis_num: usize,
    },

    #[error("There were different numbers of antenna names and antenna XYZs; there must be an equal number for both")]
    MismatchNumNamesNumXyzs,

    #[error("There were different numbers of main table antennas ({main}) antenna XYZs ({xyzs}); there must be an equal number for both")]
    MismatchNumMainTableNumXyzs { main: usize, xyzs: usize },

    #[error("Found a negative antenna number ({0}); all antenna numbers must be positive")]
    AntennaNumNegative(i32),

    #[error("Found an antenna number ({0}), but this is bigger than the total number of antennas in the antenna table.")]
    AntennaNumTooBig(i32),

    #[error("Found {num} of dipole delays in the MWA_TILE_POINTING table, but this must be 16")]
    WrongNumDipoleDelays { num: usize },

    #[error("Found a dipole delay '{delay}' in the MWA_TILE_POINTING table; values must be between 0 and 32")]
    InvalidDelay { delay: i32 },

    #[error("Found a MWA_SUBBAND number '{num}'; values must not be negative")]
    NegativeSubband { num: i32 },

    #[error("Error when trying to interface with measurement set: {0}")]
    Table(#[from] rubbl_casatables::TableError),

    #[error("Error from casacore: {0}")]
    Casacore(#[from] rubbl_casatables::CasacoreError),

    #[error(transparent)]
    Glob(#[from] crate::io::GlobError),

    #[error(transparent)]
    Mwalib(#[from] mwalib::MwalibError),
}
