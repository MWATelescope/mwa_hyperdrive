// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

/*!
Error type for all simulate-vis-related errors.
 */

use thiserror::Error;

#[derive(Error, Debug)]
pub enum SimulateVisError {
    #[error("{0}")]
    Param(#[from] super::ParamError),

    #[error("{0}")]
    VisibilityGen(#[from] crate::visibility_gen::VisibilityGenerationError),

    #[error("{0}")]
    ReadSourceList(#[from] mwa_hyperdrive_srclist::error::ReadSourceListError),

    #[error("{0}")]
    IO(#[from] std::io::Error),
}
