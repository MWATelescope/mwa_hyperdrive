// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Error type for all calibration-related errors.

use mwalib::fitsio;
use thiserror::Error;

use super::{args::CalibrateArgsFileError, params::InvalidArgsError};
use crate::{
    model::ModelError,
    solutions::{ReadSolutionsError, WriteSolutionsError},
    vis_io::read::{UvfitsReadError, VisReadError},
};
use mwa_hyperdrive_common::{mwalib, thiserror};

#[derive(Error, Debug)]
pub enum CalibrateError {
    #[error("Insufficient memory available to perform calibration; need {need_gib} GiB of memory.\nYou could try using fewer timesteps and channels.")]
    InsufficientMemory { need_gib: usize },

    #[error(
        "Timestep {timestep} wasn't available in the timestamps list; this is a programmer error"
    )]
    TimestepUnavailable { timestep: usize },

    #[error("{0}")]
    ArgFile(#[from] CalibrateArgsFileError),

    #[error("{0}")]
    InvalidArgs(#[from] InvalidArgsError),

    #[error("{0}")]
    Read(#[from] VisReadError),

    #[error("{0}\n\nSee for more info: https://github.com/MWATelescope/mwa_hyperdrive/wiki/Calibration-solutions")]
    ReadSolutions(#[from] ReadSolutionsError),

    #[error("{0}\n\nSee for more info: https://github.com/MWATelescope/mwa_hyperdrive/wiki/Calibration-solutions")]
    WriteSolutions(#[from] WriteSolutionsError),

    #[error("{0}")]
    Model(#[from] ModelError),

    #[error("cfitsio error: {0}")]
    Fitsio(#[from] fitsio::errors::Error),

    #[error("Error when reading uvfits: {0}")]
    UviftsRead(#[from] UvfitsReadError),

    #[error(transparent)]
    VisWrite(#[from] crate::vis_io::write::VisWriteError),

    #[error("IO error: {0}")]
    IO(#[from] std::io::Error),
}
