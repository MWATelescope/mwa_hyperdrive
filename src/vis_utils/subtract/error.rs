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
use mwa_hyperdrive_common::{marlu, thiserror, vec1};

#[derive(Error, Debug)]
pub enum VisSubtractError {
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

    #[error(transparent)]
    InputFiles(#[from] crate::filenames::InputFileError),

    #[error(
        "An invalid combination of formats was given. Supported:\n{}",
        SUPPORTED_CALIBRATED_INPUT_FILE_COMBINATIONS
    )]
    InvalidDataInput,

    #[error(
        "An invalid output format was specified ({0}). Supported:\n{}",
        *VIS_OUTPUT_EXTENSIONS,
    )]
    InvalidOutputFormat(String),

    #[error("Multiple metafits files were specified: {0:?}\nThis is unsupported.")]
    MultipleMetafits(Vec1<PathBuf>),

    #[error("Multiple measurement sets were specified: {0:?}\nThis is unsupported.")]
    MultipleMeasurementSets(Vec1<PathBuf>),

    #[error("Multiple uvfits files were specified: {0:?}\nThis is unsupported.")]
    MultipleUvfits(Vec1<PathBuf>),

    #[error(transparent)]
    SourceList(#[from] mwa_hyperdrive_srclist::read::SourceListError),

    #[error(transparent)]
    Veto(#[from] mwa_hyperdrive_srclist::VetoError),

    #[error(transparent)]
    MS(#[from] crate::data_formats::MsReadError),

    #[error(transparent)]
    UvfitsRead(#[from] crate::data_formats::UvfitsReadError),

    #[error(transparent)]
    MsWrite(#[from] marlu::io::MeasurementSetWriteError),

    #[error(transparent)]
    UvfitsWrite(#[from] marlu::UvfitsWriteError),

    #[error(transparent)]
    Read(#[from] crate::data_formats::ReadInputDataError),

    #[error(transparent)]
    MarluIO(#[from] marlu::io::error::IOError),

    #[error(transparent)]
    Beam(#[from] mwa_hyperdrive_beam::BeamError),

    #[error(transparent)]
    Model(#[from] crate::model::ModelError),

    #[error(transparent)]
    Glob(#[from] crate::glob::GlobError),

    #[error(transparent)]
    IO(#[from] std::io::Error),
}
