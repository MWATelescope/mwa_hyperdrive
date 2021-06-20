// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Useful constants.
//!
//! All constants *must* be double precision. `hyperdrive` should do as many
//! calculations as possible in double precision before converting to a lower
//! precision.

use std::f64::consts::PI;

/// When a spectral index must be assumed, this value is used.
pub const DEFAULT_SPEC_INDEX: f64 = -0.8;
/// The smallest allowable spectral index.
pub const SPEC_INDEX_CAP: f64 = -2.0;

// Things that should never change.

/// Speed of light \[metres/second\]
pub const VEL_C: f64 = erfa_sys::ERFA_CMPS;

/// Seconds per day (86400)
pub const DAYSEC: f64 = erfa_sys::ERFA_DAYSEC;
/// Seconds of time to radians (7.272205216643039903848712e-5).
pub const DS2R: f64 = erfa_sys::ERFA_DS2R;
/// Hour angle to radians (15 / 180 * PI).
pub const DH2R: f64 = 15.0 / 180.0 * PI;
/// Ratio of a solar day to a sidereal day (24/23.9344696 = 1.002737909).
pub const SOLAR2SIDEREAL: f64 = 24.0 / 23.9344696;

/// MWA latitude \[radians\]
pub const MWA_LONG_RAD: f64 = mwalib::MWA_LONGITUDE_RADIANS;
/// MWA latitude \[degrees\]
pub const MWA_LONG_DEG: f64 = MWA_LONG_RAD * 180.0 / PI;
/// MWA latitude \[radians\]
pub const MWA_LAT_RAD: f64 = mwalib::MWA_LATITUDE_RADIANS;
/// MWA latitude \[degrees\]
pub const MWA_LAT_DEG: f64 = MWA_LAT_RAD * 180.0 / PI;

/// MWA height (a.k.a. altitude) \[metres\]
pub const MWA_HEIGHT_M: f64 = mwalib::MWA_ALTITUDE_METRES;

// cotter's constants. Useful for being more precise when converting geocentric
// XYZ to geodetic XYZ!
/// cotter's MWA longitude on Earth in radians. Use [MWA_LONG_RAD] unless you
/// know what you're doing.
pub const COTTER_MWA_LONGITUDE_RADIANS: f64 = 2.0362897754687257;
/// cotter's MWA latitude on Earth in radians. Use [MWA_LAT_RAD] unless you know
/// what you're doing.
pub const COTTER_MWA_LATITUDE_RADIANS: f64 = -0.46606083776035967;
/// cotter's MWA altitude in metres. Use [MWA_HEIGHT_M] unless you know what
/// you're doing.
pub const COTTER_MWA_HEIGHT_METRES: f64 = 377.0;
