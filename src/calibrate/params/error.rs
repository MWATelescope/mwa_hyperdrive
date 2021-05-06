// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

use std::path::PathBuf;

use thiserror::Error;

use mwa_hyperdrive_core::mwa_hyperbeam;

/// Errors associated with setting up a `CalibrateParams` struct.
#[derive(Error, Debug)]
pub enum InvalidArgsError {
    // TODO: List supported combinations.
    #[error("No input data was given!")]
    NoInputData,

    #[error("Either no input data was given, or an invalid combination of formats was given.")]
    InvalidDataInput,

    #[error("Supplied file path {0} does not exist or is not readable!")]
    BadFile(PathBuf),

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

    #[error("Got a tile flag {got}, but the biggest possible antenna index is {max}!")]
    InvalidTileFlag { got: usize, max: usize },

    #[error("Cannot write calibration solutions to a file type '{ext}'. Supported formats are fits and bin")]
    OutputSolutionsFileType { ext: String },

    #[error("Cannot write to the specified calibration solutions file '{file}'. Do you have write permissions set?")]
    OutputSolutionsFileNotWritable { file: String },

    #[error("{0}")]
    InputFile(#[from] super::filenames::InputFileError),

    #[error("{0}")]
    Glob(#[from] crate::glob::GlobError),

    #[error("{0}")]
    RawData(#[from] crate::data_formats::raw::NewRawError),

    #[error("{0}")]
    MS(#[from] crate::data_formats::ms::NewMSError),

    #[error("{0}")]
    Veto(#[from] crate::calibrate::veto::VetoError),

    #[error("{0}")]
    SourceList(#[from] mwa_hyperdrive_srclist::read::SourceListError),

    #[error("hyperbeam init error: {0}")]
    HyperbeamInit(#[from] mwa_hyperbeam::fee::InitFEEBeamError),

    #[error("hyperbeam error: {0}")]
    Hyperbeam(#[from] mwa_hyperbeam::fee::FEEBeamError),

    #[error("{0}")]
    IO(#[from] std::io::Error),
}
