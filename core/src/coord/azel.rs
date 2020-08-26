// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

/*!
Handle (azimuth, elevation) coordinates.
*/

use super::hadec::HADec;

/// A struct containing an Azimuth and Elevation. All units are in radians.
#[derive(Clone, Copy, Debug)]
pub struct AzEl {
    /// Hour angle [radians]
    pub az: f64,
    /// Declination [radians]
    pub el: f64,
}

impl AzEl {
    /// Make a new `AzEl` struct from values in radians.
    pub fn new(az: f64, el: f64) -> Self {
        Self { az, el }
    }

    /// Make a new `AzEl` struct from values in degrees.
    pub fn new_degrees(az: f64, el: f64) -> Self {
        Self::new(az.to_radians(), el.to_radians())
    }

    /// Convert the horizon coordinates to equatorial coordinates (Hour Angle
    /// and Declination), given the local latitude on Earth.
    ///
    /// Uses ERFA.
    pub fn to_hadec(&self, latitude: f64) -> HADec {
        let mut ha = 0.0;
        let mut dec = 0.0;
        unsafe { erfa_sys::eraAe2hd(self.az, self.el, latitude, &mut ha, &mut dec) }
        HADec::new(ha, dec)
    }

    /// Convert the horizon coordinates to equatorial coordinates (Hour Angle
    /// and Declination) for the MWA's location.
    pub fn to_hadec_mwa(&self) -> HADec {
        Self::to_hadec(&self, crate::constants::MWA_LAT_RAD)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::*;

    #[test]
    fn to_hadec() {
        let ae = AzEl::new_degrees(45.0, 30.0);
        let hd = ae.to_hadec(-0.497600);
        assert_abs_diff_eq!(hd.ha, -0.6968754873551053, epsilon = 1e-10);
        assert_abs_diff_eq!(hd.dec, 0.3041176697804004, epsilon = 1e-10);
    }

    #[test]
    fn to_hadec2() {
        let ae = AzEl::new(0.261700, 0.785400);
        let hd = ae.to_hadec(-0.897600);
        assert_abs_diff_eq!(hd.ha, -0.185499449332533, epsilon = 1e-10);
        assert_abs_diff_eq!(hd.dec, -0.12732312479328656, epsilon = 1e-10);
    }
}
