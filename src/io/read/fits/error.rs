// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Errors associated with reading in fits files.

use std::path::Path;

use thiserror::Error;

#[derive(Error, Debug)]
pub enum FitsError {
    /// Error when opening a fits file.
    #[error(
        "{source_file}:{source_line}:{source_column}: Couldn't open {fits_filename}: {fits_error}"
    )]
    Open {
        fits_error: Box<fitsio::errors::Error>,
        fits_filename: Box<Path>,
        source_file: &'static str,
        source_line: u32,
        source_column: u32,
    },

    /// Error describing a key that couldn't be found in a fits header.
    #[error("{source_file}:{source_line}:{source_column}: {fits_filename} HDU {hdu_num}: Couldn't find key {key}")]
    MissingKey {
        key: Box<str>,
        fits_filename: Box<Path>,
        hdu_num: usize,
        source_file: &'static str,
        source_line: u32,
        source_column: u32,
    },

    /// Error describing a HDU that couldn't be used as an image (e.g. `HduInfo::ImageInfo`).
    #[error("{source_file}:{source_line}:{source_column}: {fits_filename} HDU {hdu_num}: Tried to use as an image, but not an image")]
    NotImage {
        fits_filename: Box<Path>,
        hdu_num: usize,
        source_file: &'static str,
        source_line: u32,
        source_column: u32,
    },

    /// Failure to read a long string.
    #[error("{source_file}:{source_line}:{source_column}: {fits_filename} HDU {hdu_num}: Couldn't read a long string from {key}")]
    LongString {
        key: Box<str>,
        fits_filename: Box<Path>,
        hdu_num: usize,
        source_file: &'static str,
        source_line: u32,
        source_column: u32,
    },

    /// A generic error associated with the fitsio crate.
    #[error(
        "{source_file}:{source_line}:{source_column}: {fits_filename} HDU '{hdu_description}': {fits_error}"
    )]
    Fitsio {
        fits_error: Box<fitsio::errors::Error>,
        fits_filename: Box<Path>,
        hdu_description: Box<str>,
        source_file: &'static str,
        source_line: u32,
        source_column: u32,
    },

    /// An error associated with parsing a string into another type.
    #[error("{source_file}:{source_line}:{source_column}: Couldn't parse {key} in {fits_filename} HDU {hdu_num}")]
    Parse {
        key: Box<str>,
        fits_filename: Box<Path>,
        hdu_num: usize,
        source_file: &'static str,
        source_line: u32,
        source_column: u32,
    },
}
