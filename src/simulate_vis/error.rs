// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

/*!
Error type for all simulate-vis-related errors.
 */

use thiserror::Error;

use mwa_hyperdrive_common::{mwalib, thiserror};

#[derive(Error, Debug)]
pub enum SimulateVisError {
    #[error("Right Ascension was not within 0 to 360!")]
    RaInvalid,

    #[error("Declination was not within -90 to 90!")]
    DecInvalid,

    #[error("One of RA and Dec was specified, but none or both are required!")]
    OnlyOneRAOrDec,

    #[error("Number of fine channels cannot be 0!")]
    FineChansZero,

    #[error("The fine channel resolution cannot be 0 or negative!")]
    FineChansWidthTooSmall,

    #[error("Number of timesteps cannot be 0!")]
    ZeroTimeSteps,

    #[error(
        "The specified MWA dipole delays aren't valid; there should be 16 values between 0 and 32"
    )]
    BadDelays,

    #[error("{0}")]
    SourceList(#[from] mwa_hyperdrive_srclist::read::SourceListError),

    #[error("{0}")]
    Beam(#[from] mwa_hyperdrive_beam::BeamError),

    #[error("{0}")]
    Model(#[from] crate::model::ModelError),

    #[error("{0}")]
    Uvfits(#[from] crate::data_formats::uvfits::UvfitsWriteError),

    #[error("{0}")]
    Glob(#[from] crate::glob::GlobError),

    #[error("{0}")]
    Mwalib(#[from] mwalib::MwalibError),

    #[error("{0}")]
    IO(#[from] std::io::Error),
}
