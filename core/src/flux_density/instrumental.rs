// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

/*!
Instrumental Stokes flux densities.
 */

use super::FluxDensity;
use crate::c64;

#[derive(Clone, Copy, Debug, Default, PartialEq)]
// TODO: Should this include a frequency?
pub struct InstrumentalStokes {
    /// XX \[Jy\]
    pub xx: c64,
    /// XY \[Jy\]
    pub xy: c64,
    /// YX \[Jy\]
    pub yx: c64,
    /// YY \[Jy\]
    pub yy: c64,
}

impl InstrumentalStokes {
    pub fn to_array(self) -> [c64; 4] {
        [self.xx, self.xy, self.yx, self.yy]
    }
}

impl From<FluxDensity> for InstrumentalStokes {
    fn from(fd: FluxDensity) -> Self {
        Self {
            xx: c64::new(fd.i + fd.q, 0.0),
            xy: c64::new(fd.u, fd.v),
            yx: c64::new(fd.u, -fd.v),
            yy: c64::new(fd.i - fd.q, 0.0),
        }
    }
}

impl From<[c64; 4]> for InstrumentalStokes {
    fn from(arr: [c64; 4]) -> Self {
        Self {
            xx: arr[0],
            xy: arr[1],
            yx: arr[2],
            yy: arr[3],
        }
    }
}

impl std::ops::Mul<c64> for InstrumentalStokes {
    type Output = Self;

    fn mul(self, rhs: c64) -> Self {
        InstrumentalStokes {
            xx: self.xx * rhs,
            xy: self.xy * rhs,
            yx: self.yx * rhs,
            yy: self.yy * rhs,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_from_fd() {
        let fd = FluxDensity {
            freq: 170e6,
            i: 0.058438801501144624,
            q: -0.3929914018344019,
            u: -0.3899498110659575,
            v: -0.058562589895788,
        };
        let result = InstrumentalStokes::from(fd);
        assert_abs_diff_eq!(result.xx, c64::new(fd.i + fd.q, 0.0));
        assert_abs_diff_eq!(result.xy, c64::new(fd.u, fd.v));
        assert_abs_diff_eq!(result.yx, c64::new(fd.u, -fd.v));
        assert_abs_diff_eq!(result.yy, c64::new(fd.i - fd.q, 0.0));
    }

    #[test]
    fn test_from_array() {
        let arr = [
            c64::new(0.058438801501144624, 0.0),
            c64::new(-0.3929914018344019, 0.0),
            c64::new(-0.3899498110659575, 0.0),
            c64::new(-0.058562589895788, 0.0),
        ];
        let result = InstrumentalStokes::from(arr);
        assert_abs_diff_eq!(result.xx, arr[0]);
        assert_abs_diff_eq!(result.xy, arr[1]);
        assert_abs_diff_eq!(result.yx, arr[2]);
        assert_abs_diff_eq!(result.yy, arr[3]);
    }
}
