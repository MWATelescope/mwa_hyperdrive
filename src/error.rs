// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Error type for all hyperdrive-related errors.

use thiserror::Error;

use mwa_hyperdrive_common::thiserror;
use mwa_hyperdrive_srclist::SrclistError;

#[derive(Error, Debug)]
pub enum HyperdriveError {
    #[error("Requested GPU processing, but the CUDA feature was not enabled when hyperdrive was compiled.")]
    NoGpuCompiled,

    #[error(transparent)]
    InvalidArgs(#[from] crate::calibrate::params::InvalidArgsError),

    #[error(transparent)]
    ArgsFile(#[from] crate::calibrate::args::CalibrateArgsFileError),

    #[error(transparent)]
    Calibrate(#[from] crate::calibrate::CalibrateError),

    #[error(transparent)]
    VisSimulate(#[from] crate::vis_utils::simulate::VisSimulateError),

    #[error(transparent)]
    VisSubtract(#[from] crate::vis_utils::subtract::VisSubtractError),

    #[error(transparent)]
    SolutionsApply(#[from] crate::solutions::apply::SolutionsApplyError),

    #[error(transparent)]
    SolutionsConvert(#[from] crate::solutions::convert::SolutionsConvertError),

    #[error(transparent)]
    SolutionsPlot(#[from] crate::solutions::plot::SolutionsPlotError),

    #[error("{0}\n\nSee for more info: https://github.com/MWATelescope/mwa_hyperdrive/wiki/Source-lists")]
    Srclist(#[from] SrclistError),
}
