// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Errors associated with reading or writing calibration solutions.

use marlu::mwalib;
use mwalib::{fitsio, FitsError};
use thiserror::Error;

use super::rts::RtsReadSolsError;
use mwa_hyperdrive_common::{marlu, thiserror};

#[derive(Error, Debug)]
pub enum ReadSolutionsError {
    #[error("Tried to read calibration solutions file with an unsupported extension '{ext}'!")]
    UnsupportedExt { ext: String },

    #[error(
        "When reading {file}, expected MWAOCAL as the first 7 characters, got '{got}' instead!"
    )]
    AndreBinaryStr { file: String, got: String },

    #[error(
        "When reading {file}, expected a value {expected} in the header, but got '{got}' instead!"
    )]
    AndreBinaryVal {
        file: String,
        expected: &'static str,
        got: String,
    },

    #[error("When interfacing with RTS calibration solutions, a metafits file is required")]
    RtsMetafitsRequired,

    #[error("In file {file} key {key}, could not parse '{got}' as a number!")]
    Parse {
        file: String,
        key: &'static str,
        got: String,
    },

    #[error("{0}")]
    Rts(#[from] RtsReadSolsError),

    #[error("{0}")]
    Fits(#[from] FitsError),

    #[error("IO error: {0}")]
    IO(#[from] std::io::Error),
}

#[derive(Error, Debug)]
pub enum WriteSolutionsError {
    #[error("Tried to write calibration solutions file with an unsupported extension '{ext}'!")]
    UnsupportedExt { ext: String },

    #[error("cfitsio error: {0}")]
    Fitsio(#[from] fitsio::errors::Error),

    #[error("{0}")]
    Fits(#[from] FitsError),

    #[error("IO error: {0}")]
    IO(#[from] std::io::Error),
}
