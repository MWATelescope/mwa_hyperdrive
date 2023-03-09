// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Errors associated with reading or writing calibration solutions.

use thiserror::Error;

use super::rts::RtsReadSolsError;

#[derive(Error, Debug)]
pub(crate) enum SolutionsReadError {
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

    #[error("Based on the dimensions of the solutions, expected {thing} to have {expected} elements, but it had {actual} instead!")]
    BadShape {
        /// What was it that wasn't sensible? Baseline weights length, chanblock
        /// frequency length, etc.
        thing: &'static str,
        expected: usize,
        actual: usize,
    },

    #[error(transparent)]
    ParsePfbFlavour(#[from] crate::io::read::pfb_gains::PfbParseError),

    #[error("When interfacing with RTS calibration solutions, a metafits file is required")]
    RtsMetafitsRequired,

    #[error(transparent)]
    Rts(#[from] RtsReadSolsError),

    #[error(transparent)]
    Fits(#[from] mwalib::FitsError),

    #[error(transparent)]
    Fitsio(#[from] fitsio::errors::Error),

    #[error(transparent)]
    IO(#[from] std::io::Error),
}

#[derive(Error, Debug)]
pub(crate) enum SolutionsWriteError {
    #[error("Tried to write calibration solutions file with an unsupported extension '{ext}'!")]
    UnsupportedExt { ext: String },

    #[error(transparent)]
    Fitsio(#[from] fitsio::errors::Error),

    #[error(transparent)]
    Fits(#[from] mwalib::FitsError),

    #[error(transparent)]
    IO(#[from] std::io::Error),
}
