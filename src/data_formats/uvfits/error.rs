// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Errors when interfacing with uvfits files.

use std::path::PathBuf;

use mwalib::*;
use thiserror::Error;

use mwa_hyperdrive_common::{mwalib, thiserror};

#[derive(Error, Debug)]
pub enum FitsError {
    /// An error associated the fitsio crate.
    #[error("{0}")]
    Fitsio(#[from] fitsio::errors::Error),

    /// An IO error.
    #[error("{0}")]
    IO(#[from] std::io::Error),
}

#[derive(Error, Debug)]
pub enum UvfitsReadError {
    #[error("No metafits file supplied - this is required when reading from uvfits files")]
    NoMetafits,

    #[error("Supplied file path {0} does not exist or is not readable!")]
    BadFile(PathBuf),

    #[error("Supplied file path {0} does not contain any data")]
    Empty(PathBuf),

    #[error("Expected to find key {key} in header of HDU {hdu}")]
    MissingKey { key: &'static str, hdu: usize },

    #[error("Could not parse key {key}'s value {value} into a number")]
    Parse { key: String, value: String },

    /// An error associated with ERFA.
    #[error("{source_file}:{source_line} Call to ERFA function {function} returned status code {status}")]
    Erfa {
        source_file: &'static str,
        source_line: u32,
        status: i32,
        function: &'static str,
    },

    #[error("No timesteps were in file {file}")]
    NoTimesteps { file: String },

    /// An error associated with fitsio.
    #[error("{0}")]
    Fitsio(#[from] fitsio::errors::Error),

    /// A error from interacting with a fits file. This particular error wraps
    /// those of `rust-fitsio`.
    #[error("{0}")]
    Fits(#[from] mwalib::FitsError),

    /// mwalib error.
    #[error("{0}")]
    Mwalib(#[from] mwalib::MwalibError),

    /// An error when converting a Rust string to a C string.
    #[error("{0}")]
    BadString(#[from] std::ffi::NulError),

    #[error("{0}")]
    Glob(#[from] crate::glob::GlobError),

    /// An IO error.
    #[error("{0}")]
    IO(#[from] std::io::Error),
}

#[derive(Error, Debug)]
pub enum UvfitsWriteError {
    #[error("Tried to write to row number {row_num}, but only {num_rows} rows are expected")]
    BadRowNum { row_num: usize, num_rows: usize },

    #[error("Expected {total} uvfits rows to be written, but only {current} were written")]
    NotEnoughRowsWritten { current: usize, total: usize },

    /// An error associated with ERFA.
    #[error("{source_file}:{source_line} Call to ERFA function {function} returned status code {status}")]
    Erfa {
        source_file: &'static str,
        source_line: u32,
        status: i32,
        function: &'static str,
    },

    /// An error associated with fitsio.
    #[error("{0}")]
    Fitsio(#[from] fitsio::errors::Error),

    /// An error when converting a Rust string to a C string.
    #[error("{0}")]
    BadString(#[from] std::ffi::NulError),

    /// An IO error.
    #[error("{0}")]
    IO(#[from] std::io::Error),
}
