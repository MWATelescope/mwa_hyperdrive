// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Error-handling code associated with reading from raw MWA files.

use thiserror::Error;

use crate::flagging::MwafMergeError;

#[derive(Error, Debug)]
pub enum RawReadError {
    #[error("gpubox file {0} does not have a corresponding mwaf file specified")]
    GpuboxFileMissingMwafFile(usize),

    #[error("Got a tile flag {got}, but the biggest possible antenna index is {max}!")]
    InvalidTileFlag { got: usize, max: usize },

    #[error("All of this observation's coarse channels are deemed bad; cannot continue")]
    NoGoodCoarseChannels,

    #[error("The raw MWA data contains no timesteps")]
    NoTimesteps,

    #[error("Attempted to read in MWA VCS data; this is unsupported")]
    Vcs,

    #[error(transparent)]
    MwafMerge(#[from] MwafMergeError),

    #[error(transparent)]
    Glob(#[from] crate::io::GlobError),

    #[error("mwalib error: {0}")]
    Mwalib(#[from] mwalib::MwalibError),
}
