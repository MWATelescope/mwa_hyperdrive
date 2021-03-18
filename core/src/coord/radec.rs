// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

/*!
Handle (right ascension, declination) coordinates.
TODO: Should RA always be positive?
 */

use std::f64::consts::*;

use log::warn;
use serde::{Deserialize, Serialize};

use super::hadec::HADec;
use super::lmn::LMN;

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

    /// From a collection of `RADec` coordinates and weights, find the average
    /// `RADec` position. The lengths of both collection must be the same to get
    /// sensible results. Not providing any `RADec` coordinates will make this
    /// function panic.
    ///
    /// This function accounts for Right Ascension coordinates that range over
    /// 360 degrees.
    pub fn weighted_average(radec: &[Self], weights: &[f64]) -> RADec {
        // Accounting for the 360 degree branch cut.
        let any_less_than_90 = radec.iter().any(|c| (0.0..FRAC_PI_4).contains(&c.ra));
        let any_between_90_270 = radec
            .iter()
            .any(|c| (FRAC_PI_4..3.0 * FRAC_PI_4).contains(&c.ra));
        let any_greater_than_270 = radec.iter().any(|c| (3.0 * FRAC_PI_4..TAU).contains(&c.ra));
        let new_cutoff = match (any_less_than_90, any_between_90_270, any_greater_than_270) {
            // User is misusing the code!
            (false, false, false) => panic!("No RADec coordinates were provided"),

            // Easy ones.
            (true, false, false) => 0.0,
            (false, true, false) => 0.0,
            (false, false, true) => 0.0,
            (false, true, true) => 0.0,
            (true, true, false) => 0.0,

            // Surrounding 0 or 360.
            (true, false, true) => PI,

            // Danger zone.
            (true, true, true) => {
                warn!("Attempting to find the average RADec over a collection of coordinates that span many RAs!");
                0.0
            }
        };

        // Don't forget the cos(dec) term!
        let average_dec = {
            let (dec_sum, count) = radec
                .iter()
                .fold((0.0, 0), |acc, c| (acc.0 + c.dec, acc.1 + 1));
            // If count == 1, then we set the "average_dec" to 0. This way,
            // cos(0) = 1, and when we multiply the RA by cos(dec), we'll get
            // the unaltered RA; seeing as we only have one RA, we shouldn't
            // alter it.
            if count == 1 {
                0.0
            } else {
                dec_sum / count as f64
            }
        };

        let mut ra_sum = 0.0;
        let mut dec_sum = 0.0;
        let mut weight_sum = 0.0;
        for (c, w) in radec.iter().zip(weights.iter()) {
            let ra = if c.ra > new_cutoff {
                c.ra - 2.0 * new_cutoff
            } else {
                c.ra
            };
            ra_sum += ra * w;
            dec_sum += c.dec * w;
            weight_sum += w;
        }
        ra_sum *= average_dec.cos();
        let mut weighted_pos = RADec::new(ra_sum / weight_sum, dec_sum / weight_sum);
        // Keep the RA positive.
        if weighted_pos.ra < 0.0 {
            weighted_pos.ra += TAU;
        }

        weighted_pos
    }

    /// Get the (l,m,n) direction cosines from these coordinates. All arguments
    /// are in radians.
    ///
    /// Derived using "Coordinate transformations" on page 388 of Synthesis
    /// Imaging in Radio Astronomy II.
    pub fn to_lmn(&self, pointing: &RADec) -> LMN {
        let d_ra = self.ra - pointing.ra;
        let (s_d_ra, c_d_ra) = d_ra.sin_cos();
        let (s_dec, c_dec) = self.dec.sin_cos();
        let (pc_s_dec, pc_c_dec) = pointing.dec.sin_cos();
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
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_to_lmn() {
        let radec = RADec::new_degrees(62.0, -27.5);
        let pointing = RADec::new_degrees(60.0, -27.0);
        let lmn = radec.to_lmn(&pointing);
        let expected = LMN {
            l: 0.03095623164758603,
            m: -0.008971846102111436,
            n: 0.9994804738961642,
        };
        assert_abs_diff_eq!(lmn.l, expected.l, epsilon = 1e-10);
        assert_abs_diff_eq!(lmn.m, expected.m, epsilon = 1e-10);
        assert_abs_diff_eq!(lmn.n, expected.n, epsilon = 1e-10);
    }

    #[test]
    fn test_weighted_pos() {
        // Only the src variable matters here; the rest of the variables aren't
        // used when constructing `RankedSource`.

        // Simple case: both components have a weight of 1.0.
        let c1 = RADec::new_degrees(10.0, 9.0);
        let w1 = 1.0;
        let c2 = RADec::new_degrees(11.0, 10.0);
        let w2 = 1.0;
        let weighted_pos = RADec::weighted_average(&[c1, c2], &[w1, w2]);
        assert_abs_diff_eq!(
            weighted_pos.ra,
            10.5_f64.to_radians() * 9.5_f64.to_radians().cos()
        );
        assert_abs_diff_eq!(weighted_pos.dec, 9.5_f64.to_radians());

        // Complex case: both components have different weights.
        let w1 = 3.0;
        let weighted_pos = RADec::weighted_average(&[c1, c2], &[w1, w2]);
        assert_abs_diff_eq!(
            weighted_pos.ra,
            10.25_f64.to_radians() * 9.5_f64.to_radians().cos()
        );
        assert_abs_diff_eq!(weighted_pos.dec, 9.25_f64.to_radians());
    }

    #[test]
    // This time, make the coordinates go across the 360 degree branch cut.
    fn test_weighted_pos2() {
        let c1 = RADec::new_degrees(10.0, 9.0);
        let w1 = 1.0;
        let c2 = RADec::new_degrees(359.0, 10.0);
        let w2 = 1.0;
        let weighted_pos = RADec::weighted_average(&[c1, c2], &[w1, w2]);
        assert_abs_diff_eq!(
            weighted_pos.ra,
            4.5_f64.to_radians() * 9.5_f64.to_radians().cos()
        );
        assert_abs_diff_eq!(weighted_pos.dec, 9.5_f64.to_radians());
    }

    #[test]
    fn test_weighted_pos_single() {
        let c = RADec::new(0.5, 0.75);
        let w = 1.0;
        let weighted_pos = RADec::weighted_average(&[c], &[w]);
        assert_abs_diff_eq!(weighted_pos.ra, 0.5);
        assert_abs_diff_eq!(weighted_pos.dec, 0.75);
    }

    #[test]
    #[should_panic]
    fn test_weighted_pos_empty() {
        let _weighted_pos = RADec::weighted_average(&[], &[1.0]);
    }
}
