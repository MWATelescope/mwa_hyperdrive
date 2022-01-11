// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Error code associated with beam calculations.

use thiserror::Error;

use mwa_hyperdrive_common::{marlu, thiserror};

#[derive(Error, Debug)]
pub enum BeamError {
    #[error("Tried to create a beam object, but MWA dipole delay information isn't available!")]
    NoDelays,

    #[error("The number of delays per tile ({delays}) didn't match the number of gains per tile ({gains})")]
    DelayGainsDimensionMismatch { delays: usize, gains: usize },

    #[error("hyperbeam error: {0}")]
    Hyperbeam(#[from] mwa_hyperbeam::fee::FEEBeamError),

    #[error("hyperbeam init error: {0}")]
    HyperbeamInit(#[from] mwa_hyperbeam::fee::InitFEEBeamError),

    #[cfg(feature = "cuda")]
    #[error("CUDA error: {0}")]
    Cuda(#[from] marlu::cuda::CudaError),
}
