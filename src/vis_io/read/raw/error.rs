// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Error-handling code associated with reading from raw MWA files.

use thiserror::Error;

use crate::flagging::MwafMergeError;

#[derive(Error, Debug)]
pub(crate) enum RawReadError {
    #[error("gpubox file {0} does not have a corresponding mwaf file specified")]
    GpuboxFileMissingMwafFile(usize),

    #[error("Got a tile flag {got}, but the biggest possible antenna index is {max}!")]
    InvalidTileFlag { got: usize, max: usize },

    #[error("All of this observation's tiles are flagged; cannot continue.")]
    AllTilesFlagged,

    #[error("All of this observation's coarse channels are deemed bad; cannot continue")]
    NoGoodCoarseChannels,

    #[error("No fine-channel flags were specified, and no rule is in place for automatically flagging observations with a fine-channel resolution of {0} Hz")]
    UnhandledFreqResolutionForFlags(u32),

    #[error("The raw MWA data contains no timesteps")]
    NoTimesteps,

    #[error("Attempted to read in MWA VCS data; this is unsupported")]
    Vcs,

    #[error(transparent)]
    MwafMerge(#[from] MwafMergeError),

    #[error(transparent)]
    Glob(#[from] crate::glob::GlobError),

    #[error("mwalib error: {0}")]
    Mwalib(#[from] mwalib::MwalibError),
}
