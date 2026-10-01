// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

use thiserror::Error;

use crate::solutions::SolutionsReadError;

#[derive(Error, Debug)]
pub(crate) enum SolutionsPlotError {
    #[error("No solutions files supplied!")]
    NoInputs,

    #[error(
        "An invalid calibration solutions file format was specified ({_0:?}).\nSupported formats: {}",
        *crate::solutions::CAL_SOLUTION_EXTENSIONS,
    )]
    InvalidSolsFormat(std::path::PathBuf),

    #[error("Your metafits file had no antenna names!")]
    MetafitsNoAntennaNames,

    #[error("While writing a plot: {0}")]
    Png(#[from] rizzma::skia::PngError),

    #[error(transparent)]
    SolutionsRead(#[from] SolutionsReadError),

    #[error(transparent)]
    Mwalib(#[from] mwalib::MwalibError),

    #[error(transparent)]
    IO(#[from] std::io::Error),
}
