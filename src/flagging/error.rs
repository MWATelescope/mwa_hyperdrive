// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Errors associated with reading AOFlagger flag files.

use std::path::PathBuf;

#[derive(thiserror::Error, Debug)]
/// Error type associated with mwaf files.
pub enum MwafError {
    #[error("mwaf file '{file:?}' has an unhandled version '{version}'")]
    UnhandledVersion { file: PathBuf, version: String },

    #[error("mwaf file '{file:?}' was written by Birli, but the SOFTWARE key didn't report the Birli version")]
    BirliVersion { file: PathBuf },

    #[error(transparent)]
    FitsError(#[from] crate::io::read::fits::FitsError),
}

#[derive(thiserror::Error, Debug)]
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
    #[error(transparent)]
    MwafError(#[from] MwafError),
}
