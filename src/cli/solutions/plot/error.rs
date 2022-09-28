// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

use thiserror::Error;

use crate::solutions::SolutionsReadError;

#[derive(Error, Debug)]
pub(crate) enum SolutionsPlotError {
    #[cfg(not(feature = "plotting"))]
    #[error("hyperdrive was not compiled with the \"plotting\" feature.\nYou need to compile hyperdrive from source with this feature to plot solutions.\nSee this page for instructions: https://MWATelescope.github.io/mwa_hyperdrive/installation/from_source.html")]
    NoPlottingFeature,

    #[cfg(feature = "plotting")]
    #[error("No solutions files supplied!")]
    NoInputs,

    #[cfg(feature = "plotting")]
    #[error(
        "An invalid calibration solutions file format was specified ({0}).\nSupported formats: {}",
        *crate::solutions::CAL_SOLUTION_EXTENSIONS,
    )]
    InvalidSolsFormat(std::path::PathBuf),

    #[cfg(feature = "plotting")]
    #[error("Your metafits file had no antenna names!")]
    MetafitsNoAntennaNames,

    #[cfg(feature = "plotting")]
    #[error("Error from the plotters library: {0}")]
    Draw(#[from] super::plotting::DrawError),

    #[error(transparent)]
    SolutionsRead(#[from] SolutionsReadError),

    #[error(transparent)]
    Mwalib(#[from] mwalib::MwalibError),

    #[error(transparent)]
    IO(#[from] std::io::Error),
}
