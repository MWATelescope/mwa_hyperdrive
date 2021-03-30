// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

/*!
Code to predict visibilities, given a sky model and array.
 */

use mwa_hyperdrive_core::{mwa_hyperbeam, AzEl, RADec, SourceList, LMN, UVW};

/// `uvw`: The UVW coordinates of each baseline.
/// `lmn`: The LMN coordinates of all sky-model source coordinates.
pub(crate) fn predict(
    beam: &mwa_hyperbeam::fee::FEEBeam,
    source_list: SourceList,
    times: &[f64],
    uvw: &[UVW],
    lmn: &[LMN],
) {
}
