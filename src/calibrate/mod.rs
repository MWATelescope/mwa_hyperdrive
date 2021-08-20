// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Code to handle calibration.

pub mod args;
mod di;
mod error;
pub(crate) mod params;
pub mod solutions;

pub use di::di_calibrate;
pub use error::CalibrateError;

// fn calibrate(mut params: CalibrateParams) -> Result<(), CalibrateError> {
//     di_calibrate(&params)?;
//     Ok(())
// }
