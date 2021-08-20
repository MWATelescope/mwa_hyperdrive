// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Errors associated with calibration arguments.

use std::path::PathBuf;

use thiserror::Error;

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

    #[error(
        "The specified MWA dipole delays aren't valid; there should be 16 values between 0 and 32"
    )]
    BadDelays,

    #[error("The number of specified sources was 0, or the size of the source list was 0")]
    NoSources,

    #[error("After vetoing sources, none were left. Decrease the veto threshold, or supply more sources")]
    NoSourcesAfterVeto,

    #[error("A single timestep was supplied multiple times; this is invalid")]
    DuplicateTimesteps,

    #[error("Timestep {got} was requested but it isn't available; the last timestep is {last}")]
    UnavailableTimestep { got: usize, last: usize },

    #[error("Got a tile flag {got}, but the biggest possible antenna index is {max}!")]
    InvalidTileFlag { got: usize, max: usize },

    #[error("Cannot write calibration solutions to a file type '{ext}'. Supported formats are: fits, bin")]
    OutputSolutionsFileType { ext: String },

    #[error(
        "Cannot write sky-model visibilities to a file type '{ext}'. Supported formats are: uvfits"
    )]
    ModelFileType { ext: String },

    #[error("Cannot write to the specified file '{file}'. Do you have write permissions set?")]
    FileNotWritable { file: String },

    #[error("{0}")]
    InputFile(#[from] super::filenames::InputFileError),

    #[error("{0}")]
    Glob(#[from] crate::glob::GlobError),

    #[error("{0}")]
    RawData(#[from] crate::data_formats::raw::NewRawError),

    #[error("{0}")]
    MS(#[from] crate::data_formats::ms::NewMSError),

    #[error("{0}")]
    Uvfits(#[from] crate::data_formats::uvfits::UvfitsReadError),

    #[error("{0}")]
    Veto(#[from] mwa_hyperdrive_srclist::VetoError),

    #[error("{0}")]
    SourceList(#[from] mwa_hyperdrive_srclist::read::SourceListError),

    #[error("{0}")]
    Beam(#[from] mwa_hyperdrive_beam::BeamError),

    #[error("{0}")]
    IO(#[from] std::io::Error),
}
