// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Errors associated with sky modelling.

use thiserror::Error;

#[derive(Error, Debug)]
pub enum ModelError {
    #[error(transparent)]
    Beam(#[from] crate::beam::BeamError),

    #[cfg(any(feature = "cuda", feature = "hip"))]
    #[error(transparent)]
    Gpu(#[from] crate::gpu::GpuError),
}
