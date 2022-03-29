// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Error type for all calibration-related errors.

use birli::marlu::io::error::{IOError as MarluIOError, UvfitsWriteError as MarluUvfitsWriteError};
use mwalib::fitsio;
use thiserror::Error;

use super::{
    args::CalibrateArgsFileError,
    params::InvalidArgsError,
    solutions::{ReadSolutionsError, WriteSolutionsError},
};
use crate::{
    data_formats::{ReadInputDataError, UvfitsReadError, UvfitsWriteError},
    model::ModelError,
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
    Read(#[from] ReadInputDataError),

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

    #[error("Error when writing uvfits: {0}")]
    UviftsWrite(#[from] UvfitsWriteError),

    #[error("Error when writing uvfits using Marlu: {0}")]
    MarluUviftsWrite(#[from] MarluUvfitsWriteError),

    #[error("Error when using Marlu for IO: {0}")]
    MarluIO(#[from] MarluIOError),

    #[error("IO error: {0}")]
    IO(#[from] std::io::Error),

    #[error("Observation Context Error:")]
    ObsContext(#[from] crate::context::ObsContextError),
}
