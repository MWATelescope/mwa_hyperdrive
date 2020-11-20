// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

use thiserror::Error;

use crate::*;

#[derive(Error, Debug)]
/// Error type associated with mwaf files.
pub enum MwafError {
    /// Error to describe some kind of inconsistent state within an mwaf file.
    #[error("Inconsistent mwaf file (file: {file}, expected: {expected}, found: {found})")]
    Inconsistent {
        file: String,
        expected: String,
        found: String,
    },

    #[error("{0}")]
    FitsError(#[from] FitsError),
}

#[derive(Error, Debug)]
/// Error type associated with merging the contents of mwaf files.
pub enum MwafMergeError {
    /// Error to describe some kind of inconsistent state within an mwaf file.
    #[error(
        r#"Inconsistent mwaf contents (first gpubox num: {gpubox1}, second gpubox num: {gpubox2}
expected: {expected}, found: {found})"#
    )]
    Inconsistent {
        gpubox1: u8,
        gpubox2: u8,
        expected: String,
        found: String,
    },

    /// Error to say that no structs were provided.
    #[error("No mwaf files were provided")]
    NoFilesGiven,

    /// Other errors associated with a single mwaf file.
    #[error("{0}")]
    MwafError(#[from] MwafError),
}
