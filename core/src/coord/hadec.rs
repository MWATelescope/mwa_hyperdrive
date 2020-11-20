// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

/*!
Handle (hour angle, declination) coordinates.
 */

use super::azel::AzEl;
use super::radec::RADec;

/// A struct containing an Hour Angle and Declination. All units are in radians.
#[derive(Clone, Copy, Debug)]
pub struct HADec {
    /// Hour angle [radians]
    pub ha: f64,
    /// Declination [radians]
    pub dec: f64,
}

impl HADec {
    /// Make a new `HADec` struct from values in radians.
    pub fn new(ha: f64, dec: f64) -> Self {
        Self { ha, dec }
    }

    /// Make a new `HADec` struct from values in degrees.
    pub fn new_degrees(ha: f64, dec: f64) -> Self {
        Self::new(ha.to_radians(), dec.to_radians())
    }

    /// Given a local sidereal time, make a new `RADec` struct from a `HADec`.
    pub fn to_radec(&self, lst: f64) -> RADec {
        RADec {
            ra: lst - self.ha,
            dec: self.dec,
        }
    }

    /// Given a local sidereal time, make a new `HADec` struct from a `RADec`.
    pub fn from_radec(radec: &RADec, lst: f64) -> Self {
        Self {
            ha: lst - radec.ra,
            dec: radec.dec,
        }
    }

    /// Convert the equatorial coordinates to horizon coordinates (azimuth and
    /// elevation), given the local latitude on Earth.
    ///
    /// Uses ERFA.
    pub fn to_azel(&self, latitude: f64) -> AzEl {
        let mut az = 0.0;
        let mut el = 0.0;
        unsafe { erfa_sys::eraHd2ae(self.ha, self.dec, latitude, &mut az, &mut el) }
        AzEl::new(az, el)
    }

    /// Convert the equatorial coordinates to horizon coordinates (azimuth and
    /// elevation) for the MWA's location.
    pub fn to_azel_mwa(&self) -> AzEl {
        Self::to_azel(&self, crate::constants::MWA_LAT_RAD)
    }

    /// Calculate the distance between two sets of coordinates.
    ///
    /// Uses ERFA.
    pub fn separation(&self, b: &Self) -> f64 {
        unsafe { erfa_sys::eraSeps(self.ha, self.dec, b.ha, b.dec) }
    }
}

impl std::fmt::Display for HADec {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "({}°, {}°)", self.ha.to_degrees(), self.dec.to_degrees())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::*;

    #[test]
    fn to_azel() {
        let hd = HADec::new_degrees(1.0, -35.0);
        let ae = hd.to_azel_mwa();
        assert_abs_diff_eq!(ae.az, 3.240305654530152, epsilon = 1e-10);
        assert_abs_diff_eq!(ae.el, 1.425221581624331, epsilon = 1e-10);
    }

    #[test]
    fn to_azel2() {
        let hd = HADec::new_degrees(23.0, -35.0);
        let ae = hd.to_azel_mwa();
        assert_abs_diff_eq!(ae.az, 4.215504972991079, epsilon = 1e-10);
        assert_abs_diff_eq!(ae.el, 1.1981324538790032, epsilon = 1e-10);
    }

    #[test]
    fn separation() {
        let hd1 = HADec::new_degrees(1.0, -35.0);
        let hd2 = HADec::new_degrees(23.0, -35.0);
        let result = hd1.separation(&hd2);
        assert_abs_diff_eq!(result, 0.31389018251593337, epsilon = 1e-10);
    }

    #[test]
    fn separation2() {
        let hd1 = HADec::new_degrees(1.0, -35.0);
        let hd2 = HADec::new_degrees(1.1, -35.0);
        let result = hd1.separation(&hd2);
        assert_abs_diff_eq!(result, 0.0014296899650293985, epsilon = 1e-10);
    }

    #[test]
    fn separation3() {
        let hd1 = HADec::new_degrees(1.0, -35.0);
        let hd2 = HADec::new_degrees(4.0, 35.0);
        let result = hd1.separation(&hd2);
        assert_abs_diff_eq!(result, 1.222708915934097, epsilon = 1e-10);
    }

    #[test]
    fn separation4() {
        let hd1 = HADec::new_degrees(2.0, -35.0);
        let hd2 = HADec::new_degrees(2.0, -35.0);
        let result = hd1.separation(&hd2);
        assert_abs_diff_eq!(result, 0.0, epsilon = 1e-10);
    }
}
