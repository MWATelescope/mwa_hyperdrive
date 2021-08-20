// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Code to handle reading from MWA metafits files.
//!
//! Anything here is to supplement mwalib.

use std::path::Path;

use log::warn;
use ndarray::prelude::*;

use mwa_rust_core::mwalib;
use mwalib::{MetafitsContext, MwalibError, Pol};

/// Populate an `<Option<MetafitsContext>>` if it isn't already populated and
/// return a reference to the context inside the `Option`.
///
/// Why is this useful? Some functions _may_ need information from a metafits
/// file. Rather than unconditionally creating an mwalib context if a metafits
/// file is provided, this function allows the caller to only call mwalib if it
/// is necessary.
pub(crate) fn populate_metafits_context<T: AsRef<Path>>(
    mwalib: &mut Option<MetafitsContext>,
    metafits: T,
    mwa_version: Option<mwalib::MWAVersion>,
) -> Result<&MetafitsContext, MwalibError> {
    match mwalib.as_mut() {
        None => {
            let c = MetafitsContext::new(&metafits, mwa_version)?;
            *mwalib = Some(c);
        }
        Some(_) => (),
    };
    // The mwalib context is always populated at this point; get the reference
    // from inside the Option.
    let context = mwalib.as_ref().unwrap();
    Ok(context)
}

/// MWA metafits files may have delays listed as all 32. This is code for "bad
/// observation, don't use". But, this has been a headache for researchers in
/// the past. When this situation is encountered, issue a warning, but then get
/// the actual observation's delays by iterating over each MWA tile's delays.
pub(crate) fn get_true_delays(context: &MetafitsContext) -> Vec<u32> {
    if !context.delays.iter().any(|&d| d == 32) {
        return context.delays.clone();
    }
    warn!("Metafits dipole delays contained 32s.");
    warn!("This may indicate that the observation's data shouldn't be used.");
    warn!("Proceeding to work out the true delays anyway...");

    // Individual dipoles in MWA tiles might be dead (i.e. delay of 32). To get
    // the true delays, iterate over all tiles until all values are non-32.
    let mut delays = [32; 16];
    for rf in &context.rf_inputs {
        for (mwalib_delay, true_delay) in rf.dipole_delays.iter().zip(delays.iter_mut()) {
            if *mwalib_delay != 32 {
                *true_delay = *mwalib_delay;
            }
        }

        // Are all delays non-32?
        if delays.iter().all(|&d| d != 32) {
            break;
        }
    }
    delays.to_vec()
}

/// Get the gains for each tile's dipoles. If a dipole is "alive", its gain is
/// one, otherwise it is "dead" and has a gain of zero.
pub(crate) fn get_dipole_gains(context: &MetafitsContext) -> Array2<f64> {
    let mut dipole_gains = Array2::from_elem(
        (
            context.rf_inputs.len() / 2,
            context.rf_inputs[0].dipole_gains.len(),
        ),
        1.0,
    );
    for (mut dipole_gains_for_one_tile, rf_input) in dipole_gains.outer_iter_mut().zip(
        context
            .rf_inputs
            .iter()
            .filter(|rf_input| rf_input.pol == Pol::Y),
    ) {
        dipole_gains_for_one_tile.assign(&ArrayView1::from(&rf_input.dipole_gains));
    }
    dipole_gains
}
