// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Errors when interfacing with uvfits files.

use thiserror::Error;

use mwa_hyperdrive_core::mwalib::fitsio;

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
pub enum UvfitsError {
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
