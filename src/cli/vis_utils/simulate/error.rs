// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Error type for all errors related to vis-simulate.

use std::path::PathBuf;

use thiserror::Error;

use crate::io::write::VIS_OUTPUT_EXTENSIONS;

#[derive(Error, Debug)]
pub(crate) enum VisSimulateError {
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

    #[error(
        "An invalid output format was specified ({0}). Supported:\n{}",
        *VIS_OUTPUT_EXTENSIONS,
    )]
    InvalidOutputFormat(PathBuf),

    #[error("Array position specified as {pos:?}, not [<Longitude>, <Latitude>, <Height>]")]
    BadArrayPosition { pos: Vec<f64> },

    #[error("After vetoing sources, none were left. Decrease the veto threshold, or supply more sources")]
    NoSourcesAfterVeto,

    #[error(transparent)]
    FileWrite(#[from] crate::io::write::FileWriteError),

    #[error(transparent)]
    AverageFactor(#[from] crate::averaging::AverageFactorError),

    #[error(transparent)]
    SourceList(#[from] crate::srclist::ReadSourceListError),

    #[error(transparent)]
    Veto(#[from] crate::srclist::VetoError),

    #[error(transparent)]
    Beam(#[from] crate::beam::BeamError),

    #[error(transparent)]
    VisWrite(#[from] crate::io::write::VisWriteError),

    #[error(transparent)]
    Glob(#[from] crate::io::GlobError),

    #[error(transparent)]
    Mwalib(#[from] mwalib::MwalibError),

    #[error(transparent)]
    Model(#[from] crate::model::ModelError),

    #[error(transparent)]
    IO(#[from] std::io::Error),

    #[cfg(feature = "cuda")]
    #[error(transparent)]
    Cuda(#[from] crate::cuda::CudaError),
}
