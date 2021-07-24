// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Error type for all sky-model-related errors.

use thiserror::Error;

use mwa_hyperdrive_core::beam::BeamError;

#[derive(Error, Debug)]
pub enum ModelError {
    #[error("{0}")]
    Beam(#[from] BeamError),
}
