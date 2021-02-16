// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

/*!
Handle (right ascension, declination) coordinates.
 */

use serde::{Deserialize, Serialize};

use super::hadec::HADec;
use super::lmn::LMN;
use super::pointing_centre::PointingCentre;

/// A struct containing a Right Ascension and Declination. All units are in
/// radians.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct RADec {
    /// Right ascension [radians]
    pub ra: f64,
    /// Declination [radians]
    pub dec: f64,
}

impl RADec {
    /// Make a new `RADec` struct from values in radians.
    pub fn new(ra: f64, dec: f64) -> Self {
        Self { ra, dec }
    }

    /// Make a new `RADec` struct from values in degrees.
    pub fn new_degrees(ra: f64, dec: f64) -> Self {
        Self::new(ra.to_radians(), dec.to_radians())
    }

    /// Given a local sidereal time, make a new `HADec` struct from a `RADec`.
    pub fn to_hadec(&self, lst_rad: f64) -> HADec {
        HADec {
            ha: lst_rad - self.ra,
            dec: self.dec,
        }
    }

    /// Given a local sidereal time, make a new `RADec` struct from a `HADec`.
    pub fn from_hadec(hadec: &HADec, lst_rad: f64) -> Self {
        Self {
            ra: lst_rad - hadec.ha,
            dec: hadec.dec,
        }
    }

    /// Get the (l,m,n) direction cosines from these coordinates. All arguments
    /// are in radians.
    ///
    /// Derived using "Coordinate transformations" on page 388 of Synthesis
    /// Imaging in Radio Astronomy II.
    pub fn to_lmn(&self, pc: &PointingCentre) -> LMN {
        let pc_radec = pc.hadec.to_radec(pc.lst);
        let d_ra = self.ra - pc_radec.ra;
        let (s_d_ra, c_d_ra) = d_ra.sin_cos();
        let (s_dec, c_dec) = self.dec.sin_cos();
        let (pc_s_dec, pc_c_dec) = pc_radec.dec.sin_cos();
        LMN {
            l: c_dec * s_d_ra,
            m: s_dec * pc_c_dec - c_dec * pc_s_dec * c_d_ra,
            n: s_dec * pc_s_dec + c_dec * pc_c_dec * c_d_ra,
        }
    }

    /// Calculate the distance between two sets of coordinates (radians).
    ///
    /// Uses ERFA.
    pub fn separation(&self, b: &Self) -> f64 {
        unsafe { erfa_sys::eraSeps(self.ra, self.dec, b.ra, b.dec) }
    }
}

impl std::fmt::Display for RADec {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "({}°, {}°)", self.ra.to_degrees(), self.dec.to_degrees())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::*;

    #[test]
    fn test_to_lmn() {
        let radec = RADec::new_degrees(62.0, -27.5);
        let pc = PointingCentre::new_from_ra(0.0, 60_f64.to_radians(), (-27.0_f64).to_radians());
        let lmn = radec.to_lmn(&pc);
        let expected = LMN {
            l: 0.03095623164758603,
            m: -0.008971846102111436,
            n: 0.9994804738961642,
        };
        assert_abs_diff_eq!(lmn.l, expected.l, epsilon = 1e-10);
        assert_abs_diff_eq!(lmn.m, expected.m, epsilon = 1e-10);
        assert_abs_diff_eq!(lmn.n, expected.n, epsilon = 1e-10);
    }
}
