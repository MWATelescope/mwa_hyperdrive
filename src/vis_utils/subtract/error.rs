// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Error type for all errors related to vis-subtract.

use std::path::PathBuf;

use thiserror::Error;
use vec1::Vec1;

use crate::{
    filenames::SUPPORTED_CALIBRATED_INPUT_FILE_COMBINATIONS, help_texts::VIS_OUTPUT_EXTENSIONS,
};
use mwa_hyperdrive_common::{thiserror, vec1};

#[derive(Error, Debug)]
pub(crate) enum VisSubtractError {
    #[error("Specified source {name} is not in the input source list; can't subtract it")]
    MissingSource { name: String },

    #[error("No sources were specified for subtraction. Did you want to subtract all sources? See the \"invert\" option.")]
    NoSources,

    #[error("No sources were left after removing specified sources from the source list.")]
    AllSourcesFiltered,

    #[error("After vetoing sources, none were left. Decrease the veto threshold, or supply more sources")]
    NoSourcesAfterVeto,

    #[error("Tried to create a beam object, but MWA dipole delay information isn't available!")]
    NoDelays,

    #[error(
        "The specified MWA dipole delays aren't valid; there should be 16 values between 0 and 32"
    )]
    BadDelays,

    #[error("No input data was given!")]
    NoInputData,

    #[error(
        "{0}\n\nSupported combinations of file formats:\n{SUPPORTED_CALIBRATED_INPUT_FILE_COMBINATIONS}",
    )]
    InvalidDataInput(&'static str),

    #[error("The data either contains no timesteps or no timesteps are being used")]
    NoTimesteps,

    #[error("Duplicate timesteps were specified; this is invalid")]
    DuplicateTimesteps,

    #[error("Timestep {got} was specified but it isn't available; the last timestep is {last}")]
    UnavailableTimestep { got: usize, last: usize },

    #[error(
        "An invalid output format was specified ({0}). Supported:\n{}",
        *VIS_OUTPUT_EXTENSIONS,
    )]
    InvalidOutputFormat(PathBuf),

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

    #[error("Multiple metafits files were specified: {0:?}\nThis is unsupported.")]
    MultipleMetafits(Vec1<PathBuf>),

    #[error("Multiple measurement sets were specified: {0:?}\nThis is unsupported.")]
    MultipleMeasurementSets(Vec1<PathBuf>),

    #[error("Multiple uvfits files were specified: {0:?}\nThis is unsupported.")]
    MultipleUvfits(Vec1<PathBuf>),

    #[error("Array position specified as {pos:?}, not [<Longitude>, <Latitude>, <Height>]")]
    BadArrayPosition { pos: Vec<f64> },

    #[error(transparent)]
    Veto(#[from] mwa_hyperdrive_srclist::VetoError),

    #[error(transparent)]
    VisRead(#[from] crate::vis_io::read::VisReadError),

    #[error(transparent)]
    Glob(#[from] crate::glob::GlobError),

    #[error(transparent)]
    VisWrite(#[from] crate::vis_io::write::VisWriteError),

    #[error(transparent)]
    FileWrite(#[from] crate::vis_io::write::FileWriteError),

    #[error(transparent)]
    SourceList(#[from] mwa_hyperdrive_srclist::read::SourceListError),

    #[error(transparent)]
    Beam(#[from] mwa_hyperdrive_beam::BeamError),

    #[error(transparent)]
    IO(#[from] std::io::Error),
}
