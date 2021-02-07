// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

/*!
Code to handle calibration.
 */

pub mod args;
pub mod params;
pub mod veto;

use params::CalibrateParams;

pub fn calibrate(mut params: CalibrateParams) -> Result<(), anyhow::Error> {
    // How much time is available?
    //
    // Assume we're doing a DI step. How much data gets averaged together? Does
    // this depend on baseline length?

    todo!();
}
