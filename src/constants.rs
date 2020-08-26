// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

/*!
Useful constants.

All constants *must* be double precision. `hyperdrive` should do as many
calculations as possible in double precision before converting to a lower
precision, if it is ever required.

When declaring constants, it would be nice to use the `constant` keyword, but
that prevents the usage of the same constants in other constants, e.g.

MWA_LAT_DEG: f64 = -26.703319;
MWA_LAT_RAD: f64 = MWA_LAT_DEG.to_radians();

The lazy_static crate means that all constants become references (must be
de-referenced with a `*`), but, we can re-use constants to help minimise the
number re-calculations and errors.
 */

pub use std::f64::consts::PI;

use lazy_static::lazy_static;

lazy_static! {
/// When a spectral index must be assumed, this value is used.
pub static ref DEFAULT_SPEC_INDEX: f64 = -0.8;
/// The smallest allowable spectral index.
pub static ref SPEC_INDEX_CAP: f64 = -2.0;
/// The sign of the lm->uv Fourier transform.
pub static ref R2C_SIGN: f64 = -1.0;

// Things that should never change.

/// 2 * PI
pub static ref PI2: f64 = 2.0 * PI;
/// PI / 2
pub static ref PIBY2: f64 = PI / 2.0;

/// Speed of light [metres/second]
pub static ref VEL_C: f64 = 299_792_458.0;

/// Seconds per day
pub static ref DAYSEC: f64 = 86400.0;
/// Seconds of time to radians. Taken from ERFA.
pub static ref DS2R: f64 = 7.272_205_216_643_04e-5;
/// Hour angle to radians (15 / 180 * PI).
pub static ref DH2R: f64 = 0.261_799_387_799_149_46;
/// Ratio of a solar day to a sidereal day (24/23.9344696).
pub static ref SOLAR2SIDEREAL: f64 = 1.002_737_811_911_354_6;
/// Earth rotation rate in radians per UT1 second. Taken from ERFA.
pub static ref EARTH_ROTATION_RATE: f64 = *SOLAR2SIDEREAL * *PI2 / *DAYSEC;
/// Radius of the Earth [metres]
// This was taken from the RTS. The IAU value is 6378100; maybe the extra 40m
// are for the MWA's location? TODO: Check.
pub static ref EARTH_RADIUS: f64 = 6378140.0;

/// MWA latitude [degrees]
pub static ref MWA_LAT_DEG: f64 = -26.703_319;
/// MWA latitude [radians]
pub static ref MWA_LAT_RAD: f64 = MWA_LAT_DEG.to_radians();
/// cos(MWA latitude)
pub static ref MWA_LAT_COS: f64 = MWA_LAT_RAD.cos();
/// sin(MWA latitude)
pub static ref MWA_LAT_SIN: f64 = MWA_LAT_RAD.sin();
}
