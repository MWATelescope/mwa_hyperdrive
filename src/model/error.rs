// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Error type for all sky-model-related errors.

use thiserror::Error;

use mwa_hyperdrive_common::thiserror;

#[derive(Error, Debug)]
pub enum ModelError {
    #[error("{0}")]
    Beam(#[from] mwa_hyperdrive_beam::BeamError),

    #[cfg(feature = "cuda")]
    #[error("{0}")]
    Cuda(#[from] mwa_hyperdrive_beam::CudaError),
}
