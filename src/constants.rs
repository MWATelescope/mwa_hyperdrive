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

pub use mwa_hyperdrive_core::constants::*;
