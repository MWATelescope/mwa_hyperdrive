// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

use thiserror::Error;

use crate::mwalib;

/// Errors associated with setting up a `CalibrateParams` struct.
#[derive(Error, Debug)]
pub enum CalibrateParamsError {}

#[derive(Error, Debug)]
pub enum InvalidArgsError {
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

    #[error("No sky-model source list file supplied")]
    NoSourceList,

    #[error("The number of specified sources was 0, or the size of the source list was 0")]
    NoSources,

    #[error("After vetoing sources, none were left. Decrease the veto threshold, or supply more sources")]
    NoSourcesAfterVeto,

    #[error("Cannot use {got}s as the calibration time resolution; this must be a multiple of the native resolution ({native}s)")]
    InvalidTimeResolution { got: f64, native: f64 },

    #[error("Cannot use {got}s as the calibration frequency resolution; this must be a multiple of the native resolution ({native}s)")]
    InvalidFreqResolution { got: f64, native: f64 },

    #[error("{0}")]
    Glob(#[from] crate::glob::GlobError),

    #[error("{0}")]
    MwafMerge(#[from] crate::flagging::error::MwafMergeError),

    #[error("{0}")]
    Veto(#[from] crate::calibrate::veto::VetoError),

    #[error("{0}")]
    SourceList(#[from] mwa_hyperdrive_srclist::read::SourceListError),

    #[error("mwalib error: {0}")]
    Mwalib(#[from] mwalib::MwalibError),

    #[error("hyperbeam error: {0}")]
    Hyperbeam(#[from] mwa_hyperbeam::fee::FEEBeamError),

    #[error("{0}")]
    IO(#[from] std::io::Error),
}
