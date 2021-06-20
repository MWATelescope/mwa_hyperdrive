// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Handle UVW coordinates.

use super::hadec::HADec;
use super::xyz::XyzBaseline;

use rayon::prelude::*;

/// The (u,v,w) coordinates of a baseline. All units are in terms of wavelength,
/// with units of metres.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
#[allow(clippy::upper_case_acronyms)]
pub struct UVW {
    /// u-coordinate \[meters\]
    pub u: f64,
    /// v-coordinate \[meters\]
    pub v: f64,
    /// w-coordinate \[meters\]
    pub w: f64,
}

impl UVW {
    /// Convert an [XyzBaseline]s to [UVW], given the phase centre. This
    /// function is less convenient than `from_xyz`, but may be better in tight
    /// loops as the `sin` and `cos` of the phase centre don't need to be
    /// uselessly re-calculated.
    ///
    /// This is Equation 4.1 of: Interferometry and Synthesis in Radio
    /// Astronomy, Third Edition, Section 4: Geometrical Relationships,
    /// Polarimetry, and the Measurement Equation.
    #[inline]
    pub fn from_xyz_inner(xyz: &XyzBaseline, s_ha: f64, c_ha: f64, s_dec: f64, c_dec: f64) -> Self {
        Self {
            u: s_ha * xyz.x + c_ha * xyz.y,
            v: -s_dec * c_ha * xyz.x + s_dec * s_ha * xyz.y + c_dec * xyz.z,
            w: c_dec * c_ha * xyz.x - c_dec * s_ha * xyz.y + s_dec * xyz.z,
        }
    }

    /// Convert an [XyzBaseline]s to [UVW], given the phase centre.
    ///
    /// This is Equation 4.1 of: Interferometry and Synthesis in Radio
    /// Astronomy, Third Edition, Section 4: Geometrical Relationships,
    /// Polarimetry, and the Measurement Equation.
    pub fn from_xyz(xyz: &XyzBaseline, phase_centre: &HADec) -> Self {
        let (s_ha, c_ha) = phase_centre.ha.sin_cos();
        let (s_dec, c_dec) = phase_centre.dec.sin_cos();
        Self::from_xyz_inner(xyz, s_ha, c_ha, s_dec, c_dec)
    }

    /// Convert all [XyzBaseline]s to [UVW], given the phase centre.
    ///
    /// This is Equation 4.1 of: Interferometry and Synthesis in Radio
    /// Astronomy, Third Edition, Section 4: Geometrical Relationships,
    /// Polarimetry, and the Measurement Equation.
    pub fn get_baselines(xyzs: &[XyzBaseline], phase_centre: &HADec) -> Vec<Self> {
        let (s_ha, c_ha) = phase_centre.ha.sin_cos();
        let (s_dec, c_dec) = phase_centre.dec.sin_cos();
        xyzs.iter()
            .map(|xyz| Self::from_xyz_inner(xyz, s_ha, c_ha, s_dec, c_dec))
            .collect()
    }

    /// Convert all [XyzBaseline]s to [UVW], given the phase centre. Processing
    /// is done in parallel on the CPU.
    ///
    /// This is Equation 4.1 of: Interferometry and Synthesis in Radio
    /// Astronomy, Third Edition, Section 4: Geometrical Relationships,
    /// Polarimetry, and the Measurement Equation.
    pub fn get_baselines_parallel(xyzs: &[XyzBaseline], phase_centre: &HADec) -> Vec<Self> {
        let (s_ha, c_ha) = phase_centre.ha.sin_cos();
        let (s_dec, c_dec) = phase_centre.dec.sin_cos();
        xyzs.par_iter()
            .map(|xyz| Self::from_xyz_inner(xyz, s_ha, c_ha, s_dec, c_dec))
            .collect()
    }
}

impl std::ops::Mul<f64> for UVW {
    type Output = Self;

    fn mul(self, rhs: f64) -> Self {
        UVW {
            u: self.u * rhs,
            v: self.v * rhs,
            w: self.w * rhs,
        }
    }
}

impl std::ops::Div<f64> for UVW {
    type Output = Self;

    fn div(self, rhs: f64) -> Self {
        UVW {
            u: self.u / rhs,
            v: self.v / rhs,
            w: self.w / rhs,
        }
    }
}

#[cfg(test)]
impl approx::AbsDiffEq for UVW {
    type Epsilon = f64;

    fn default_epsilon() -> f64 {
        f64::EPSILON
    }

    #[inline]
    fn abs_diff_eq(&self, other: &Self, epsilon: f64) -> bool {
        (self.u - other.u) <= epsilon
            && (self.v - other.v) <= epsilon
            && (self.w - other.w) <= epsilon
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::XyzGeodetic;
    use approx::*;

    #[test]
    fn test_uvw_mul() {
        let uvw = UVW {
            u: 1.0,
            v: 2.0,
            w: 3.0,
        } * 3.0;
        assert_abs_diff_eq!(uvw.u, 3.0);
        assert_abs_diff_eq!(uvw.v, 6.0);
        assert_abs_diff_eq!(uvw.w, 9.0);
    }

    #[test]
    fn test_uvw_div() {
        let uvw = UVW {
            u: 3.0,
            v: 6.0,
            w: 9.0,
        } / 3.0;
        assert_abs_diff_eq!(uvw.u, 1.0);
        assert_abs_diff_eq!(uvw.v, 2.0);
        assert_abs_diff_eq!(uvw.w, 3.0);
    }

    #[test]
    fn get_uvw_baselines_test() {
        let xyz = vec![
            XyzGeodetic {
                x: 289.5692922664971,
                y: -585.6749877929688,
                z: -259.3106530519151,
            },
            XyzGeodetic {
                x: 750.5194624923599,
                y: -565.4390258789063,
                z: 665.2348852011041,
            },
        ];
        let xyz_bl = XyzGeodetic::get_baselines(&xyz);
        let phase = HADec::new(6.0163, -0.453121);
        let uvw = UVW::get_baselines(&xyz_bl, &phase);
        let expected = UVW {
            u: 102.04605530570603,
            v: -1028.2293398297727,
            w: 0.18220641926160397,
        };
        assert_abs_diff_eq!(uvw[0].u, expected.u, epsilon = 1e-10);
        assert_abs_diff_eq!(uvw[0].v, expected.v, epsilon = 1e-10);
        assert_abs_diff_eq!(uvw[0].w, expected.w, epsilon = 1e-10);

        let uvw_parallel = UVW::get_baselines_parallel(&xyz_bl, &phase);
        for (serial, parallel) in uvw.into_iter().zip(uvw_parallel.into_iter()) {
            assert_abs_diff_eq!(serial, parallel, epsilon = 1e-10);
        }
    }
}
