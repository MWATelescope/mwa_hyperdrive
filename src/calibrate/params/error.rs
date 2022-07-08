// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Errors associated with calibration arguments.

use std::path::PathBuf;

use thiserror::Error;
use vec1::Vec1;

use crate::filenames::SUPPORTED_INPUT_FILE_COMBINATIONS;
use mwa_hyperdrive_common::{thiserror, vec1};

/// Errors associated with setting up [super::CalibrateParams].
#[derive(Error, Debug)]
pub(crate) enum InvalidArgsError {
    #[error("No input data was given!")]
    NoInputData,

    #[error("{0}\n\nSupported combinations of file formats:\n{SUPPORTED_INPUT_FILE_COMBINATIONS}")]
    InvalidDataInput(&'static str),

    #[error("Multiple metafits files were specified: {0:?}\nThis is unsupported.")]
    MultipleMetafits(Vec1<PathBuf>),

    #[error("Multiple measurement sets were specified: {0:?}\nThis is currently unsupported.")]
    MultipleMeasurementSets(Vec1<PathBuf>),

    #[error("Multiple uvfits files were specified: {0:?}\nThis is currently unsupported.")]
    MultipleUvfits(Vec1<PathBuf>),

    #[error("No calibration output was specified. There must be at least one calibration solution file.")]
    NoOutput,

    #[error("No sky-model source list file supplied")]
    NoSourceList,

    #[error("Tried to create a beam object, but MWA dipole delay information isn't available!")]
    NoDelays,

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

    #[error("Duplicate timesteps were specified; this is invalid")]
    DuplicateTimesteps,

    #[error("Timestep {got} was specified but it isn't available; the last timestep is {last}")]
    UnavailableTimestep { got: usize, last: usize },

    #[error(
        "Cannot write visibilities to a file type '{ext}'. Supported formats are: {}", *crate::help_texts::VIS_OUTPUT_EXTENSIONS
    )]
    VisFileType { ext: String },

    #[error(transparent)]
    TileFlag(#[from] crate::context::InvalidTileFlag),

    #[error("Cannot write calibration solutions to a file type '{ext}'.\nSupported formats are: {}", *crate::calibrate::args::CAL_SOLUTION_EXTENSIONS)]
    CalibrationOutputFile { ext: String },

    #[error(transparent)]
    ParsePfbFlavour(#[from] crate::pfb_gains::PfbParseError),

    #[error("Error when parsing time average factor: {0}")]
    ParseCalTimeAverageFactor(crate::unit_parsing::UnitParseError),

    #[error("Error when parsing freq. average factor: {0}")]
    ParseCalFreqAverageFactor(crate::unit_parsing::UnitParseError),

    #[error("Calibration time average factor isn't an integer")]
    CalTimeFactorNotInteger,

    #[error("Calibration freq. average factor isn't an integer")]
    CalFreqFactorNotInteger,

    #[error("Calibration time resolution isn't a multiple of input data's: {out} seconds vs {inp} seconds")]
    CalTimeResNotMultiple { out: f64, inp: f64 },

    #[error("Calibration freq. resolution isn't a multiple of input data's: {out} Hz vs {inp} Hz")]
    CalFreqResNotMultiple { out: f64, inp: f64 },

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
    OutputVisTimeResNotMultiple { out: f64, inp: f64 },

    #[error("Output vis. freq. resolution isn't a multiple of input data's: {out} Hz vs {inp} Hz")]
    OutputVisFreqResNotMultiple { out: f64, inp: f64 },

    #[error("Error when parsing minimum UVW cutoff: {0}")]
    ParseUvwMin(crate::unit_parsing::UnitParseError),

    #[error("Error when parsing maximum UVW cutoff: {0}")]
    ParseUvwMax(crate::unit_parsing::UnitParseError),

    #[error("Array position specified as {pos:?}, not [<Longitude>, <Latitude>, <Height>]")]
    BadArrayPosition { pos: Vec<f64> },

    #[cfg(feature = "cuda")]
    #[error("CUDA error: {0}")]
    CudaError(String),

    #[error(transparent)]
    Glob(#[from] crate::glob::GlobError),

    #[error(transparent)]
    VisRead(#[from] crate::vis_io::read::VisReadError),

    #[error(transparent)]
    FileWrite(#[from] crate::vis_io::write::FileWriteError),

    #[error(transparent)]
    Veto(#[from] mwa_hyperdrive_srclist::VetoError),

    #[error("Error when trying to read source list: {0}")]
    SourceList(#[from] mwa_hyperdrive_srclist::read::SourceListError),

    #[error(transparent)]
    Beam(#[from] mwa_hyperdrive_beam::BeamError),

    #[error(transparent)]
    IO(#[from] std::io::Error),
}
