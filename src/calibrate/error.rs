// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Error type for all calibration-related errors.

use mwalib::fitsio;
use thiserror::Error;

use mwa_hyperdrive_common::{mwalib, thiserror};

#[derive(Error, Debug)]
pub(crate) enum CalibrateError {
    #[error("Insufficient memory available to perform calibration; need {need_gib} GiB of memory.\nYou could try using fewer timesteps and channels.")]
    InsufficientMemory { need_gib: usize },

    #[error(
        "Timestep {timestep} wasn't available in the timestamps list; this is a programmer error"
    )]
    TimestepUnavailable { timestep: usize },

    #[error(transparent)]
    ArgFile(#[from] super::args::CalibrateArgsFileError),

    #[error(transparent)]
    InvalidArgs(#[from] super::params::InvalidArgsError),

    #[error(transparent)]
    SolutionsRead(#[from] crate::solutions::SolutionsReadError),

    #[error(transparent)]
    SolutionsWrite(#[from] crate::solutions::SolutionsWriteError),

    #[error(transparent)]
    Beam(#[from] mwa_hyperdrive_beam::BeamError),

    #[error(transparent)]
    Fitsio(#[from] fitsio::errors::Error),

    #[error(transparent)]
    VisRead(#[from] crate::vis_io::read::VisReadError),

    #[error(transparent)]
    VisWrite(#[from] crate::vis_io::write::VisWriteError),

    #[error(transparent)]
    IO(#[from] std::io::Error),
}
