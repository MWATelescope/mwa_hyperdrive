// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Code to handle reading from MWA metafits files.
//!
//! Anything here is to supplement mwalib.

use log::warn;

use mwa_hyperdrive_core::mwalib::MetafitsContext;

/// MWA metafits files may have delays listed as all 32. This is code for "bad observation, don't use". But, this has been a headache for researchers in the past. When this situation is encountered, issue a warning, but then get the actual observation's delays by iterating over each MWA tile's delays.
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
