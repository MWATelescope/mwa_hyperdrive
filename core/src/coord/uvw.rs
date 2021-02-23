// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

/*!
Handle UVW coordinates.
 */

use super::hadec::HADec;
use super::xyz::XyzBaseline;

use rayon::prelude::*;

/// The (u,v,w) coordinates of a baseline. All units are in terms of wavelength,
/// with units of metres.
#[derive(Clone, Copy, Debug)]
pub struct UVW {
    /// u-coordinate [meters]
    pub u: f64,
    /// v-coordinate [meters]
    pub v: f64,
    /// w-coordinate [meters]
    pub w: f64,
}

impl UVW {
    /// Convert all `XyzBaseline`s to `UVW`, given the pointing centre.
    /// Processing is done in parallel on the CPU.
    ///
    /// This is Equation 4.1 of: Interferometry and Synthesis in Radio
    /// Astronomy, Third Edition, Section 4: Geometrical Relationships,
    /// Polarimetry, and the Measurement Equation.
    pub fn get_baselines(xyz: &[XyzBaseline], pointing: &HADec) -> Vec<Self> {
        let (pc_s_ha, pc_c_ha) = pointing.ha.sin_cos();
        let (pc_s_dec, pc_c_dec) = pointing.dec.sin_cos();
        xyz.par_iter()
            .map(|l| Self {
                u: pc_s_ha * l.x + pc_c_ha * l.y,
                v: pc_s_dec * pc_s_ha * l.y + pc_c_dec * l.z - pc_s_dec * pc_c_ha * l.x,
                w: pc_c_dec * pc_c_ha * l.x - pc_c_dec * pc_s_ha * l.y + pc_s_dec * l.z,
            })
            .collect()
    }

    /// Copy the contents of a slice of `UVW` structs to U, V, and W
    /// vectors.
    pub fn split(uvw: &[Self]) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let length = uvw.len();
        let mut u = Vec::with_capacity(length);
        let mut v = Vec::with_capacity(length);
        let mut w = Vec::with_capacity(length);
        for elem in uvw {
            u.push(elem.u as f32);
            v.push(elem.v as f32);
            w.push(elem.w as f32);
        }
        // Ensure that the capacity of the vectors matches their length.
        u.shrink_to_fit();
        v.shrink_to_fit();
        w.shrink_to_fit();
        (u, v, w)
    }

    /// Convert a vector of rust `UVW` structs to U, V, and W vectors. Useful
    /// for FFI.
    pub fn decompose(mut uvw: Vec<Self>) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let length = uvw.len();
        let mut u = Vec::with_capacity(length);
        let mut v = Vec::with_capacity(length);
        let mut w = Vec::with_capacity(length);
        for elem in uvw.drain(..) {
            u.push(elem.u as f32);
            v.push(elem.v as f32);
            w.push(elem.w as f32);
        }
        // Ensure that the capacity of the vectors matches their length.
        u.shrink_to_fit();
        v.shrink_to_fit();
        w.shrink_to_fit();
        (u, v, w)
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
mod tests {
    use super::*;
    use crate::coord::xyz::XYZ;
    use approx::*;

    #[test]
    fn get_uvw_baselines_test() {
        let xyz = vec![
            XYZ {
                x: 289.5692922664971,
                y: -585.6749877929688,
                z: -259.3106530519151,
            },
            XYZ {
                x: 750.5194624923599,
                y: -565.4390258789063,
                z: 665.2348852011041,
            },
        ];
        let xyz_bl = XYZ::get_xyz_baselines(&xyz);
        let pointing = HADec::new(6.0163, -0.453121);
        let uvw = UVW::get_baselines(&xyz_bl, &pointing);
        let expected = UVW {
            u: 102.04605530570603,
            v: -1028.2293398297727,
            w: 0.18220641926160397,
        };
        assert_abs_diff_eq!(uvw[0].u, expected.u, epsilon = 1e-10);
        assert_abs_diff_eq!(uvw[0].v, expected.v, epsilon = 1e-10);
        assert_abs_diff_eq!(uvw[0].w, expected.w, epsilon = 1e-10);
    }
}
