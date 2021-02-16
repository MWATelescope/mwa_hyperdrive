// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

/*!
Useful constants.

All constants *must* be double precision. `hyperdrive` should do as many
calculations as possible in double precision before converting to a lower
precision, if it is ever required.
 */

pub use std::f64::consts::{FRAC_PI_2, PI, TAU};

/// Sources with beam-attenuated flux densities less than this value are
/// discarded from sky-model source lists.
pub const DEFAULT_VETO_THRESHOLD: f64 = 0.01;

/// Sources with elevations less than this value are discarded from sky-model
/// source lists.
pub const ELEVATION_LIMIT: f64 = 0.0;

/// Sources that are separated by more than this value (degrees) from the
/// pointing are discarded from sky-model source lists.
pub const CUTOFF_DISTANCE: f64 = 30.0;

/// Alan Levine's gains from PFB simulations. Taken from RTS source code.
pub const LEVINE_GAINS_40KHZ: [f64; 32] = [
    0.51735311934047334,
    0.59251439019439012,
    0.70695099259495631,
    0.82467941813344192,
    0.91743238101078828,
    0.97399249233715968,
    0.99882351784428292,
    1.0041872682882493,
    1.0021295484391897,
    1.0000974383045906,
    1.0004197495080835,
    1.0020927020996839,
    1.0032018583576889,
    1.0027668031914465,
    1.001305418352239,
    1.0001674256814668,
    1.0003506058381628,
    1.0016962975293491,
    1.0030147335641364,
    1.0030573420014388,
    1.0016582119173054,
    1.0001394672444315,
    1.0004004241051296,
    1.0028377901921051,
    1.0039523509152424,
    0.9949679743767017,
    0.96320539409670669,
    0.89751138048775558,
    0.79674361345958533,
    0.67664334604801912,
    0.56869884824103156,
    0.5082890508180502,
];

/// Gains from empirical averaging of RTS BP solution points using "Anish" PFB
/// gains for 1062363808 and backing out corrections to flatten average coarse
/// channel.
pub const EMPIRICAL_GAINS_40KHZ: [f64; 32] = [
    0.5, 0.5, 0.67874855, 0.83576969, 0.95187049, 1.0229769, 1.05711736, 1.06407012, 1.06311151,
    1.06089592, 1.0593481, 1.06025714, 1.06110822, 1.05893943, 1.05765503, 1.05601938, 0.5,
    1.05697461, 1.05691842, 1.05688129, 1.05623901, 1.05272663, 1.05272112, 1.05551337, 1.05724941,
    1.0519857, 1.02483081, 0.96454596, 0.86071928, 0.71382954, 0.5, 0.5,
];

pub use mwa_hyperdrive_core::constants::*;
