// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Code to handle reading from MWA metafits files.
//!
//! Anything here is to supplement mwalib.

use mwalib::{MetafitsContext, Pol};
use ndarray::prelude::*;

use mwa_hyperdrive_common::{mwalib, ndarray};

/// Get the delays for each tile's dipoles.
pub(crate) fn get_dipole_delays(context: &mwalib::MetafitsContext) -> Array2<u32> {
    let mut all_tile_delays = Array2::zeros((context.num_ants, 16));
    for (i, mut tile_delays) in all_tile_delays.outer_iter_mut().enumerate() {
        let rf_input_1 = &context.rf_inputs[2 * i];
        let rf_input_2 = &context.rf_inputs[2 * i + 1];
        let delays: Vec<_> = rf_input_1
            .dipole_delays
            .iter()
            .zip(rf_input_2.dipole_delays.iter())
            .map(|(&d_1, &d_2)| {
                // The delays should be the same, modulo some being 32 (i.e.
                // that dipole's component is dead). This code will pick the
                // smaller delay of the two (delays are always <=32). If both
                // are 32, there's nothing else that can be done.
                d_1.min(d_2)
            })
            .collect();
        tile_delays.assign(&Array1::from(delays));
    }
    all_tile_delays
}

/// Get the gains for each tile's dipoles. If a dipole is "alive", its gain is
/// one, otherwise it is "dead" and has a gain of zero.
pub fn get_dipole_gains(context: &MetafitsContext) -> Array2<f64> {
    let mut dipole_gains = Array2::zeros((
        context.num_ants,
        context.rf_inputs[0].dipole_gains.len() * 2,
    ));
    for (i, mut dipole_gains_for_one_tile) in dipole_gains.outer_iter_mut().enumerate() {
        let rf_input_1 = &context.rf_inputs[2 * i];
        let rf_input_2 = &context.rf_inputs[2 * i + 1];
        let (rf_input_x, rf_input_y) = if rf_input_1.pol == Pol::X {
            (rf_input_1, rf_input_2)
        } else {
            (rf_input_2, rf_input_1)
        };
        dipole_gains_for_one_tile
            .slice_mut(s![..16])
            .assign(&ArrayView1::from(&rf_input_x.dipole_gains));
        dipole_gains_for_one_tile
            .slice_mut(s![16..])
            .assign(&ArrayView1::from(&rf_input_y.dipole_gains));
    }
    dipole_gains
}
