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

    #[error("No calibration output was specified. There must be at least one calibration solution file or calibrated visibility file.")]
    NoOutput,

    #[error(
        "Couldn't create directory '{0}' for output files. Do you have write permissions set?"
    )]
    NewDirectory(PathBuf),

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

    #[error(
        "Cannot write visibilities to a file type '{ext}'. Supported formats are: {}", *crate::calibrate::args::VIS_OUTPUT_EXTENSIONS
    )]
    VisFileType { ext: String },

    #[error("Cannot write calibration outputs to a file type '{ext}'.\nSupported formats are: {} (calibration solutions)\n                     : {} (visibility files)", *crate::calibrate::args::CAL_SOLUTION_EXTENSIONS, *crate::calibrate::args::VIS_OUTPUT_EXTENSIONS)]
    CalibrationOutputFile { ext: String },

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
