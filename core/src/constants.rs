// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

/*!
Useful constants.

All constants *must* be double precision. `hyperdrive` should do as many
calculations as possible in double precision before converting to a lower
precision, if it is ever required.
 */

use std::f64::consts::{PI, TAU};

/// When a spectral index must be assumed, this value is used.
pub const DEFAULT_SPEC_INDEX: f64 = -0.8;
/// The smallest allowable spectral index.
pub const SPEC_INDEX_CAP: f64 = -2.0;
/// The sign of the lm->uv Fourier transform.
pub const R2C_SIGN: f64 = -1.0;

// Things that should never change.

/// Speed of light [metres/second]
pub const VEL_C: f64 = erfa_sys::ERFA_CMPS;

/// Seconds per day
pub const DAYSEC: f64 = erfa_sys::ERFA_DAYSEC;
/// Seconds of time to radians.
pub const DS2R: f64 = erfa_sys::ERFA_DS2R;
/// Hour angle to radians (15 / 180 * PI).
pub const DH2R: f64 = 15.0 / 180.0 * PI;
/// Ratio of a solar day to a sidereal day (24/23.9344696).
pub const SOLAR2SIDEREAL: f64 = 24.0 / 23.9344696;
/// Earth rotation rate in radians per UT1 second.
pub const EARTH_ROTATION_RATE: f64 = SOLAR2SIDEREAL * TAU / DAYSEC;
/// Radius of the Earth [metres]
// This was taken from the RTS. The IAU value is 6378100; maybe the extra 40m
// are for the MWA's location? TODO: Check.
pub const EARTH_RADIUS: f64 = 6378140.0;

/// MWA latitude [degrees]
pub const MWA_LAT_DEG: f64 = -26.703_319;
/// MWA latitude [radians]
pub const MWA_LAT_RAD: f64 = mwalib::MWA_LATITUDE_RADIANS;
// TODO: Take the sin and cos of MWA_LAT_RAD when Rust's const functionality is
// more mature.
/// sin(MWA latitude)
pub const MWA_LAT_SIN: f64 = -0.44937075493668055;
/// cos(MWA latitude)
pub static MWA_LAT_COS: f64 = 0.8933453557318344;
