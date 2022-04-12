// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

use std::path::PathBuf;

use thiserror::Error;

use crate::{help_texts::CAL_SOL_EXTENSIONS, solutions::ReadSolutionsError};
use mwa_hyperdrive_common::{mwalib, thiserror};

#[derive(Error, Debug)]
pub enum SolutionsPlotError {
    #[error("hyperdrive was not compiled with the \"plotting\" feature.\nYou need to compile hyperdrive from source with this feature to plot solutions.")]
    NoPlottingFeature,

    #[error("No solutions files supplied!")]
    NoInputs,

    #[error(
        "An invalid calibration solutions file format was specified ({0}).\nSupported formats: {}",
        *CAL_SOL_EXTENSIONS,
    )]
    InvalidSolsFormat(PathBuf),

    #[error("Your metafits file had no antenna names!")]
    MetafitsNoAntennaNames,

    #[cfg(feature = "plotting")]
    #[error("Error from the plotters library: {0}")]
    Draw(#[from] super::plotting::DrawError),

    #[error(transparent)]
    ReadSolutions(#[from] ReadSolutionsError),

    #[error("mwalib error: {0}")]
    Mwalib(#[from] mwalib::MwalibError),

    #[error(transparent)]
    IO(#[from] std::io::Error),
}
