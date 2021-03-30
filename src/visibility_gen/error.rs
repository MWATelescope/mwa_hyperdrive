// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

/*!
Error type for all visibility-generation-related errors.
 */

use thiserror::Error;

use mwa_hyperdrive_core::EstimateError;

#[derive(Error, Debug)]
pub enum VisibilityGenerationError {
    #[error("{0}")]
    Estimate(#[from] EstimateError),

    #[error("{0}")]
    IO(#[from] std::io::Error),
}
