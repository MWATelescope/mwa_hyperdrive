// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Errors when interfacing with uvfits files.

use std::{borrow::Cow, path::PathBuf};

use thiserror::Error;

#[derive(Error, Debug)]
pub(crate) enum FitsError {
    /// An error associated the fitsio crate.
    #[error(transparent)]
    Fitsio(#[from] fitsio::errors::Error),

    /// An IO error.
    #[error(transparent)]
    IO(#[from] std::io::Error),
}

#[derive(Error, Debug)]
pub(crate) enum UvfitsReadError {
    #[error("Supplied file path {0} does not exist or is not readable!")]
    BadFile(PathBuf),

    #[error("Supplied file path {0} does not contain any data")]
    Empty(PathBuf),

    #[error("No timesteps were in file {file}")]
    NoTimesteps { file: PathBuf },

    #[error("No antenna names were in the ANNAME column")]
    AnnameEmpty,

    #[error("The Earth position/location of the array responsible for these uvfits visibilities cannot be assumed or determined. Please specify manually or supply a metafits file.")]
    NoArrayPos,

    #[error("The TIMSYS ({0}) wasn't UTC, IAT or TAI; this is unsupported")]
    UnknownTimsys(String),

    #[error("Expected to find key {key} in header of HDU {hdu}")]
    MissingKey { key: &'static str, hdu: usize },

    #[error("BASELINE is specified as well as ANTENNA1/ANTENNA2; this is unsupported")]
    BaselineAndAntennas,

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

    #[error("Could not parse key {key}'s value {value} into a number: {parse_error}")]
    Parse {
        key: Cow<'static, str>,
        value: String,
        parse_error: String,
    },

    #[error("When attempting to read uvfits baseline metadata, cfitsio gave an error: {0}")]
    Metadata(fitsio::errors::Error),

    #[error("When attempting to read uvfits column {col_name} from HDU {hdu_num}, cfitsio gave an error: {err}")]
    ReadCellArray {
        col_name: &'static str,
        hdu_num: usize,
        err: fitsio::errors::Error,
    },

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
    Mwalib(#[from] mwalib::MwalibError),

    /// An error when converting a Rust string to a C string.
    #[error(transparent)]
    BadString(#[from] std::ffi::NulError),

    #[error(transparent)]
    Glob(#[from] crate::io::GlobError),

    /// An IO error.
    #[error(transparent)]
    IO(#[from] std::io::Error),
}
