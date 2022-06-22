// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Errors when interfacing with uvfits files.

use std::path::PathBuf;

use mwalib::*;
use thiserror::Error;

use mwa_hyperdrive_common::{mwalib, thiserror};

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
    NoTimesteps { file: String },

    #[error("No antenna names were in the ANNAME column")]
    AnnameEmpty,

    #[error("The TIMSYS ({0}) wasn't UTC, IAT or TAI; this is unsupported")]
    UnknownTimsys(String),

    #[error("Expected to find key {key} in header of HDU {hdu}")]
    MissingKey { key: &'static str, hdu: usize },

    #[error("Could not parse key {key}'s value {value} into a number")]
    Parse { key: String, value: String },

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
    Fits(#[from] mwalib::FitsError),

    /// mwalib error.
    #[error(transparent)]
    Mwalib(#[from] mwalib::MwalibError),

    /// An error when converting a Rust string to a C string.
    #[error(transparent)]
    BadString(#[from] std::ffi::NulError),

    #[error(transparent)]
    Glob(#[from] crate::glob::GlobError),

    /// An IO error.
    #[error(transparent)]
    IO(#[from] std::io::Error),
}
