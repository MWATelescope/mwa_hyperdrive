// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

use std::path::PathBuf;

use thiserror::Error;

#[derive(Error, Debug)]
pub(crate) enum CrosstalkPlotError {
    #[error("No crosstalk score files supplied!")]
    NoInputs,

    #[error("{file}: no '{column}' column; is this a crosstalk-qa CSV?")]
    MissingColumn { file: PathBuf, column: String },

    #[error("{file}: line {line} has too few fields or an unreadable value")]
    BadRow { file: PathBuf, line: usize },

    #[error("None of the supplied files have any XX or YY scores for tiles in the metafits")]
    NoScores,

    #[error("While encoding a plot: {0}")]
    Png(#[from] rizzma::skia::PngError),

    #[error(transparent)]
    Mwalib(#[from] mwalib::MwalibError),

    #[error(transparent)]
    IO(#[from] std::io::Error),
}
