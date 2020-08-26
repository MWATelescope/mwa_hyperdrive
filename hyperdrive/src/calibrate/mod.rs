// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

/*!
Code to handle calibration.
 */

pub mod args;

use serde::Serialize;

use crate::*;

#[derive(Debug, Default, Serialize)]
pub struct CalibrateParams {
    /// Path to the metafits file.
    pub metafits: PathBuf,

    /// Paths to gpubox files.
    pub gpuboxes: Vec<PathBuf>,

    /// Optional paths to mwaf files.
    pub mwafs: Option<Vec<PathBuf>>,
}

pub fn calibrate(_params: &CalibrateParams) -> Result<(), anyhow::Error> {
    todo!();
}
