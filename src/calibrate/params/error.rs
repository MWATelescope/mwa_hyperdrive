// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Errors associated with calibration arguments.

use std::path::PathBuf;

use thiserror::Error;
use vec1::Vec1;

use super::filenames::SUPPORTED_INPUT_FILE_COMBINATIONS;
use mwa_hyperdrive_common::{thiserror, vec1};

/// Errors associated with setting up [super::CalibrateParams].
#[derive(Error, Debug)]
pub enum InvalidArgsError {
    #[error("No input data was given!")]
    NoInputData,

    #[error(
        "An invalid combination of formats was given. Supported:\n{}",
        SUPPORTED_INPUT_FILE_COMBINATIONS
    )]
    InvalidDataInput,

    #[error("Multiple metafits files were specified: {0:?}\nThis is unsupported.")]
    MultipleMetafits(Vec1<PathBuf>),

    #[error("Multiple measurement sets were specified: {0:?}\nThis is currently unsupported.")]
    MultipleMeasurementSets(Vec1<PathBuf>),

    #[error("Multiple uvfits files were specified: {0:?}\nThis is currently unsupported.")]
    MultipleUvfits(Vec1<PathBuf>),

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

    #[error("The data either contains no tiles or all tiles are flagged")]
    NoTiles,

    #[error("The data either contains no frequency channels or all channels are flagged")]
    NoChannels,

    #[error("The data either contains no timesteps or no timesteps are being used")]
    NoTimesteps,

    #[error("The number of specified sources was 0, or the size of the source list was 0")]
    NoSources,

    #[error("After vetoing sources, none were left. Decrease the veto threshold, or supply more sources")]
    NoSourcesAfterVeto,

    #[error("A single timestep was supplied multiple times; this is invalid")]
    DuplicateTimesteps,

    #[error("Timestep {got} was requested but it isn't available; the last timestep is {last}")]
    UnavailableTimestep { got: usize, last: usize },

    #[error("Got a tile flag {got}, but the biggest possible antenna index is {max}")]
    InvalidTileFlag { got: usize, max: usize },

    #[error("Bad flag value: '{0}' is neither an integer or an available antenna name. Run with extra verbosity to see all tile names.")]
    BadTileFlag(String),

    #[error(
        "Cannot write visibilities to a file type '{ext}'. Supported formats are: {}", *crate::calibrate::args::VIS_OUTPUT_EXTENSIONS
    )]
    VisFileType { ext: String },

    #[error("Cannot write calibration outputs to a file type '{ext}'.\nSupported formats are: {} (calibration solutions)\n                     : {} (visibility files)", *crate::calibrate::args::CAL_SOLUTION_EXTENSIONS, *crate::calibrate::args::VIS_OUTPUT_EXTENSIONS)]
    CalibrationOutputFile { ext: String },

    #[error("Could not parse PFB flavour '{0}'.\nSupported flavours are: {}", *crate::pfb_gains::PFB_FLAVOURS)]
    ParsePfbFlavour(String),

    #[error("Error when parsing time average factor: {0}")]
    ParseCalTimeAverageFactor(crate::unit_parsing::UnitParseError),

    #[error("Error when parsing freq. average factor: {0}")]
    ParseCalFreqAverageFactor(crate::unit_parsing::UnitParseError),

    #[error("Calibration time average factor isn't an integer")]
    CalTimeFactorNotInteger,

    #[error("Calibration freq. average factor isn't an integer")]
    CalFreqFactorNotInteger,

    #[error("Calibration time resolution isn't a multiple of input data's: {out} seconds vs {inp} seconds")]
    CalTimeResNotMulitple { out: f64, inp: f64 },

    #[error("Calibration freq. resolution isn't a multiple of input data's: {out} Hz vs {inp} Hz")]
    CalFreqResNotMulitple { out: f64, inp: f64 },

    #[error("Calibration time average factor cannot be 0")]
    CalTimeFactorZero,

    #[error("Calibration freq. average factor cannot be 0")]
    CalFreqFactorZero,

    #[error("Error when parsing output vis. time average factor: {0}")]
    ParseOutputVisTimeAverageFactor(crate::unit_parsing::UnitParseError),

    #[error("Error when parsing output vis. freq. average factor: {0}")]
    ParseOutputVisFreqAverageFactor(crate::unit_parsing::UnitParseError),

    #[error("Output vis. time average factor isn't an integer")]
    OutputVisTimeFactorNotInteger,

    #[error("Output vis. freq. average factor isn't an integer")]
    OutputVisFreqFactorNotInteger,

    #[error("Output vis. time average factor cannot be 0")]
    OutputVisTimeAverageFactorZero,

    #[error("Output vis. freq. average factor cannot be 0")]
    OutputVisFreqAverageFactorZero,

    #[error("Output vis. time resolution isn't a multiple of input data's: {out} seconds vs {inp} seconds")]
    OutputVisTimeResNotMulitple { out: f64, inp: f64 },

    #[error("Output vis. freq. resolution isn't a multiple of input data's: {out} Hz vs {inp} Hz")]
    OutputVisFreqResNotMulitple { out: f64, inp: f64 },

    #[error("Output vis. time resolution cannot be 0")]
    OutputVisTimeResZero,

    #[error("Output vis. freq. resolution cannot be 0")]
    OutputVisFreqResZero,

    #[error("Error when parsing minimum UVW cutoff: {0}")]
    ParseUvwMin(crate::unit_parsing::UnitParseError),

    #[error("Error when parsing maximum UVW cutoff: {0}")]
    ParseUvwMax(crate::unit_parsing::UnitParseError),

    #[error("Cannot write to the specified file '{file}'. Do you have write permissions set?")]
    FileNotWritable { file: String },

    #[error("{0}")]
    InputFile(String),

    #[error("{0}")]
    Glob(#[from] crate::glob::GlobError),

    #[error("{0}")]
    RawData(#[from] crate::data_formats::RawReadError),

    #[error("{0}")]
    MS(#[from] crate::data_formats::MsReadError),

    #[error("{0}")]
    Uvfits(#[from] crate::data_formats::UvfitsReadError),

    #[error("{0}")]
    Veto(#[from] mwa_hyperdrive_srclist::VetoError),

    #[error("Error when trying to read source list: {0}")]
    SourceList(#[from] mwa_hyperdrive_srclist::read::SourceListError),

    #[error("{0}")]
    Beam(#[from] mwa_hyperdrive_beam::BeamError),

    #[error("{0}")]
    IO(#[from] std::io::Error),
}
