// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

/*!
Functions that help to populate the `CalibrateParams` struct.

In an effort to keep the mod.rs file associated with calibration parameters
clean, little functions that can be broken out can sit here.
 */

use crate::mwalib::{MetafitsContext, Pol};

/// With a mwalib metafits context, pull out the tile flags. The returned ints
/// correspond to *antenna numbers*, not *input numbers*.
pub(super) fn get_metafits_tile_flags(context: &MetafitsContext) -> Vec<usize> {
    // Filter avoids pulling out the same tile twice, as there are two RF inputs
    // per tile.
    context
        .rf_inputs
        .iter()
        .filter(|rf| rf.pol == Pol::Y)
        .map(|rf| rf.ant as usize)
        .collect()
}
