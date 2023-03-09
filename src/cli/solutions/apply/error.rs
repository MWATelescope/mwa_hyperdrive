// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

use std::path::PathBuf;

use thiserror::Error;
use vec1::Vec1;

use crate::{filenames::SUPPORTED_INPUT_FILE_COMBINATIONS, io::write::VIS_OUTPUT_EXTENSIONS};

#[derive(Error, Debug)]
pub(crate) enum SolutionsApplyError {
    #[error("The input data and the solutions have different numbers of tiles (data: {data}, solutions: {solutions}); cannot continue")]
    TileCountMismatch { data: usize, solutions: usize },

    #[error("Multiple metafits files were specified: {0:?}\nThis is unsupported.")]
    MultipleMetafits(Vec1<PathBuf>),

    #[error("Multiple measurement sets were specified: {0:?}\nThis is unsupported.")]
    MultipleMeasurementSets(Vec1<PathBuf>),

    #[error("Multiple uvfits files were specified: {0:?}\nThis is unsupported.")]
    MultipleUvfits(Vec1<PathBuf>),

    #[error("No input data was given!\n\nSupported combinations of file formats:\n{SUPPORTED_INPUT_FILE_COMBINATIONS}")]
    NoInputData,

    #[error("{0}\n\nSupported combinations of file formats:\n{SUPPORTED_INPUT_FILE_COMBINATIONS}")]
    InvalidDataInput(&'static str),

    #[error(
        "An invalid output format was specified ({0}). Supported:\n{}",
        *VIS_OUTPUT_EXTENSIONS,
    )]
    InvalidOutputFormat(PathBuf),

    #[error("The data either contains no tiles or all tiles are flagged")]
    NoTiles,

    #[error(transparent)]
    TileFlag(#[from] crate::context::InvalidTileFlag),

    #[error("The data either contains no timesteps or no timesteps are being used")]
    NoTimesteps,

    #[error("No output was specified. There must be at least one visibility output.")]
    NoOutput,

    #[error("Duplicate timesteps were specified; this is invalid")]
    DuplicateTimesteps,

    #[error("Timestep {got} was specified but it isn't available; the last timestep is {last}")]
    UnavailableTimestep { got: usize, last: usize },

    #[error("Array position specified as {pos:?}, not [<Longitude>, <Latitude>, <Height>]")]
    BadArrayPosition { pos: Vec<f64> },

    #[error(transparent)]
    ParsePfbFlavour(#[from] crate::io::read::pfb_gains::PfbParseError),

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

    #[error(transparent)]
    SolutionsRead(#[from] crate::solutions::SolutionsReadError),

    #[error(transparent)]
    VisRead(#[from] crate::io::read::VisReadError),

    #[error(transparent)]
    FileWrite(#[from] crate::io::write::FileWriteError),

    #[error(transparent)]
    VisWrite(#[from] crate::io::write::VisWriteError),

    #[error(transparent)]
    IO(#[from] std::io::Error),
}
