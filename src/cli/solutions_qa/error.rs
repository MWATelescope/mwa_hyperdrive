// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

use std::path::PathBuf;

use thiserror::Error;

use crate::solutions::SolutionsReadError;

#[derive(Error, Debug)]
pub(crate) enum SolutionsQaError {
    #[error("No calibration solutions files supplied!")]
    NoInputs,

    #[error("{file}: the solutions have {num_sol_tiles} tiles and no tile names, but the metafits has {num_metafits_tiles} tiles, so they can't be matched")]
    TileCountMismatch {
        file: PathBuf,
        num_sol_tiles: usize,
        num_metafits_tiles: usize,
    },

    #[error("None of the tiles in the supplied solutions are in the metafits")]
    NoTiles,

    #[error(transparent)]
    SolutionsRead(#[from] SolutionsReadError),

    #[error("While encoding a plot: {0}")]
    Png(#[from] rizzma::skia::PngError),

    #[error(transparent)]
    Mwalib(#[from] mwalib::MwalibError),

    #[error(transparent)]
    IO(#[from] std::io::Error),
}
