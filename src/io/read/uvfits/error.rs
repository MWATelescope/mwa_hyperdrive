// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Errors when interfacing with uvfits files.

use std::{borrow::Cow, path::PathBuf};

use hifitime::{Duration, Epoch};
use thiserror::Error;

#[derive(Error, Debug)]
pub enum UvfitsReadError {
    #[error("Supplied file path {0} does not exist or is not readable!")]
    BadFile(PathBuf),

    #[error("Supplied file path {0} does not contain any data")]
    Empty(PathBuf),

    #[error("No timesteps were in file {file}")]
    NoTimesteps { file: PathBuf },

    #[error("The timestamps in the uvfits aren't separated by a regular interval; we think the time resolution is {what_we_think_is_the_time_res}, but we found a gap {gap_found} in timestamp pair {pair}")]
    IrregularTimestamps {
        what_we_think_is_the_time_res: Duration,
        gap_found: Duration,
        pair: usize,
    },

    #[error("uvfits timestep {timestep} was expected to have a timestamp {expected_timestamp}, but {got} was on row {uvfits_row}")]
    MismatchedTimestamps {
        timestep: usize,
        expected_timestamp: Epoch,
        got: Epoch,
        uvfits_row: usize,
    },

    #[error("No antenna names were in the ANNAME column")]
    AnnameEmpty,

    #[error("The TIMSYS ({0}) wasn't UTC, IAT or TAI; this is unsupported")]
    UnknownTimsys(String),

    #[error("Expected to find key {key} in header of HDU {hdu}")]
    MissingKey { key: &'static str, hdu: usize },

    #[error("Found an index for ANTENNA1, but not ANTENNA2; cannot continue")]
    Antenna1ButNotAntenna2,

    #[error("Found an index for ANTENNA2, but not ANTENNA1; cannot continue")]
    Antenna2ButNotAntenna1,

    #[error("None of BASELINE, ANTENNA1 or ANTENNA2 were specified; cannot continue")]
    NoBaselineInfo,

    #[error("There are {0} floats per polarisation; this is unsupported. The uvfits standard enforces only 2 or 3 floats per polarisation")]
    WrongFloatsPerPolCount(u8),

    #[error("The shape of the visibility data is unsupported; we expect COMPLEX to be NAXIS2 (got {complex}), STOKES to be NAXIS3 (got {stokes}), and FREQ to be NAXIS4 (got {freq}")]
    WrongDataOrder { complex: u8, stokes: u8, freq: u8 },

    #[error(
        "STOKES {key} indicates a polarisation type '{value}', which is currently unsupported"
    )]
    UnsupportedPolType { key: Cow<'static, str>, value: i8 },

    #[error(
        "STOKES {crval} and {naxis} indicates a polarisation type '{pol_type}' along with '{num_pols}' polarisations; this is currently unsupported"
    )]
    UnsupportedPols {
        crval: Cow<'static, str>,
        naxis: Cow<'static, str>,
        pol_type: i8,
        num_pols: u8,
    },

    #[error("Could not parse key {key}'s value {value} into a number: {parse_error}")]
    Parse {
        key: Cow<'static, str>,
        value: String,
        parse_error: String,
    },

    #[error("When attempting to read uvfits baseline metadata, cfitsio gave an error: {0}")]
    Metadata(fitsio::errors::Error),

    #[error("When attempting to read uvfits row {row_num}, cfitsio gave an error: {err}")]
    ReadVis {
        row_num: usize,
        err: fitsio::errors::Error,
    },

    /// A generic error associated with fitsio.
    #[error(transparent)]
    Fitsio(#[from] fitsio::errors::Error),

    /// A error from interacting with a fits file. This particular error wraps
    /// those of `rust-fitsio`.
    #[error(transparent)]
    Fits(#[from] crate::io::read::fits::FitsError),

    /// mwalib error.
    #[error(transparent)]
    Mwalib(#[from] Box<mwalib::MwalibError>),

    /// An error when converting a Rust string to a C string.
    #[error(transparent)]
    BadString(#[from] std::ffi::NulError),

    #[error(transparent)]
    Glob(#[from] crate::io::GlobError),

    /// An IO error.
    #[error(transparent)]
    IO(#[from] std::io::Error),
}
