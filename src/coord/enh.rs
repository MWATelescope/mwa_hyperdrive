// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

/*!
Handle East, North and Height coordinates (typically associated with MWA tiles).
*/

use super::xyz::XYZ;
use crate::constants::*;

/// East, North and Height coordinates of an MWA tile.
#[derive(Debug)]
pub struct ENH {
    /// East [metres]
    pub e: f64,
    /// North [metres]
    pub n: f64,
    /// Height [metres]
    pub h: f64,
}

impl ENH {
    /// Convert coords in local topocentric East, North, Height units to 'local'
    /// XYZ units. Local means Z points north, X points through the equator from
    /// the geocenter along the local meridian and Y is East. This is like the
    /// absolute system except that zero longitude is now the local meridian
    /// rather than prime meridian. Latitude is geodetic, in radians. This is
    /// what you want for constructing the local antenna positions in a UVFITS
    /// antenna table.
    ///
    /// Taken from the third edition of Interferometry and Synthesis in Radio
    /// Astronomy, chapter 4: Geometrical Relationships, Polarimetry, and the
    /// Measurement Equation.
    pub fn to_xyz(&self, latitude: f64) -> XYZ {
        let (s_lat, c_lat) = latitude.sin_cos();
        XYZ {
            x: -self.n * s_lat + self.h * c_lat,
            y: self.e,
            z: self.n * c_lat + self.h * s_lat,
        }
    }

    /// Convert ENH coordinates to local XYZ for the MWA's latitude.
    pub fn to_xyz_mwa(&self) -> XYZ {
        self.to_xyz(*MWA_LAT_RAD)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::*;

    #[test]
    fn convert_enh_to_xyz_test() {
        let enh = ENH {
            n: -101.530,
            e: -585.675,
            h: 375.212,
        };
        let xyz = ENH::to_xyz_mwa(&enh);
        assert_abs_diff_eq!(xyz.x, 289.5692867016053, epsilon = 1e-10);
        assert_abs_diff_eq!(xyz.y, -585.675, epsilon = 1e-10);
        assert_abs_diff_eq!(xyz.z, -259.3106516191025, epsilon = 1e-10);
    }
}
