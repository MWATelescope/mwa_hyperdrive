// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

/*!
Error-handling code associated with reading from raw MWA files.
 */

use std::path::PathBuf;

use thiserror::Error;

use crate::flagging::error::MwafMergeError;
use mwa_hyperdrive_core::mwalib;

#[derive(Error, Debug)]
pub enum NewRawError {
    #[error("Supplied file path {0} does not exist or is not readable!")]
    BadFile(PathBuf),

    #[error("No metafits file supplied")]
    NoMetafits,

    #[error("No gpubox files supplied")]
    NoGpuboxes,

    #[error("gpubox file {0} does not have a corresponding mwaf file specified")]
    GpuboxFileMissingMwafFile(usize),

    #[error("The lone gpubox entry is neither a file nor a glob pattern that matched any files")]
    SingleGpuboxNotAFileOrGlob,

    #[error("The lone mwaf entry is neither a file nor a glob pattern that matched any files")]
    SingleMwafNotAFileOrGlob,

    #[error("Got a tile flag {got}, but the biggest possible antenna index is {max}!")]
    InvalidTileFlag { got: usize, max: usize },

    #[error("All of this observation's tiles are flagged; cannot continue.")]
    AllTilesFlagged,

    #[error("No fine-channel flags were specified, and no rule is in place for automatically flagging observations with a fine-channel resolution of {0} Hz")]
    UnhandledFreqResolutionForFlags(u32),

    #[error("{0}")]
    MwafMerge(#[from] MwafMergeError),

    #[error("{0}")]
    Glob(#[from] crate::glob::GlobError),

    #[error("mwalib error: {0}")]
    Mwalib(#[from] mwalib::MwalibError),
}
