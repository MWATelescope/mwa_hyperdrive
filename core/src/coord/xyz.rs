// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Handle (x,y,z) coordinates of an antenna (a.k.a. tile or station).
//!
//! This coordinate system is discussed at length in Interferometry and
//! Synthesis in Radio Astronomy, Third Edition, Section 4: Geometrical
//! Relationships, Polarimetry, and the Measurement Equation.

use super::enh::ENH;
use crate::{constants::MWA_LAT_RAD, HADec, UVW};

/// The (x,y,z) coordinates of an antenna (a.k.a. tile or station). All units
/// are in metres.
///
/// This coordinate system is discussed at length in Interferometry and
/// Synthesis in Radio Astronomy, Third Edition, Section 4: Geometrical
/// Relationships, Polarimetry, and the Measurement Equation.
#[derive(Clone, Debug, Default)]
#[allow(clippy::upper_case_acronyms)]
pub struct XYZ {
    /// x-coordinate \[meters\]
    pub x: f64,
    /// y-coordinate \[meters\]
    pub y: f64,
    /// z-coordinate \[meters\]
    pub z: f64,
}

impl XYZ {
    /// Convert local XYZ coordinates at a latitude to ENH coordinates.
    pub fn to_enh(&self, latitude: f64) -> ENH {
        let s_lat = latitude.sin();
        let c_lat = latitude.cos();
        ENH {
            e: self.y,
            n: -self.x * s_lat + self.z * c_lat,
            h: self.x * c_lat + self.z * s_lat,
        }
    }

    /// Convert local XYZ coordinates at the MWA's latitude to ENH coordinates.
    pub fn to_enh_mwa(&self) -> ENH {
        self.to_enh(MWA_LAT_RAD)
    }

    /// Rotate the XYZ coordinates by a given angle (longitude in radians). This
    /// allows conversion to geodetic XYZ from geocentric XYZ, or vice versa.
    pub fn rotate(&self, angle: f64) -> Self {
        let (s_rot, c_rot) = angle.sin_cos();

        let xtemp = self.x * c_rot - self.y * s_rot;
        let y = self.x * s_rot + self.y * c_rot;
        let x = xtemp;
        Self { x, y, z: self.z }
    }

    /// Rotate the XYZ coordinates at the MWA's longitude. From the supplied
    /// sign, this allows conversion to geodetic XYZ from geocentric XYZ, or
    /// vice versa.
    ///
    /// If sign is positive or zero, then the rotation is forward, otherwise
    /// backward.
    pub fn rotate_mwa(&self, sign: i8) -> XYZ {
        let angle = match sign.signum() {
            0 | 1 => 1.0,
            -1 => -1.0,
            _ => unreachable!(),
        } * mwalib::MWA_LONGITUDE_RADIANS;
        self.rotate(angle)
    }

    /// For each XYZ pair, calculate a baseline.
    pub fn get_baselines(xyz: &[Self]) -> Vec<XyzBaseline> {
        // Assume that the length of `xyz` is the number of tiles.
        let n_tiles = xyz.len();
        let n_baselines = (n_tiles * (n_tiles - 1)) / 2;
        let mut diffs = Vec::with_capacity(n_baselines);
        for i in 0..n_tiles {
            for j in i + 1..n_tiles {
                diffs.push(xyz[i].clone() - &xyz[j]);
            }
        }
        diffs
    }

    /// For each tile listed in an mwalib context, calculate a `XYZ` geocentric
    /// coordinate.
    ///
    /// Note that the RF inputs are ordered by antenna number, **not** the
    /// "input"; e.g. in the metafits file, Tile104 is often the first tile
    /// listed ("input" 0), Tile103 second ("input" 2), so the first baseline
    /// would naively be between Tile104 and Tile103.
    pub fn get_tiles_mwalib(context: &mwalib::MetafitsContext) -> Vec<Self> {
        let mut xyz = Vec::with_capacity(context.num_rf_inputs / 2);
        for rf in &context.rf_inputs {
            // There is an RF input for both tile polarisations. The ENH
            // coordinates are the same for both polarisations of a tile; ignore
            // the RF input if it's associated with Y.
            if rf.pol == mwalib::Pol::Y {
                continue;
            }

            let enh = ENH {
                e: rf.east_m,
                n: rf.north_m,
                h: rf.height_m,
            };
            xyz.push(enh.to_xyz_mwa());
        }
        xyz
    }

    /// For each tile listed in an mwalib context, calculate a `XyzBaseline`.
    ///
    /// Note that the RF inputs are ordered by antenna number, **not** the
    /// "input"; e.g. in the metafits file, Tile104 is often the first tile
    /// listed ("input" 0), Tile103 second ("input" 2), so the first baseline
    /// would naively be between Tile104 and Tile103.
    pub fn get_baselines_mwalib(context: &mwalib::MetafitsContext) -> Vec<XyzBaseline> {
        Self::get_baselines(&Self::get_tiles_mwalib(context))
    }

    // This is what cotter does to precessed XYZ tile positions.
    // TODO: Check that this matches doing what we used to do (i.e. XYZ -> XyzBaseline -> UVW).
    pub fn to_uvw(xyzs: &[Self], phase_centre: &HADec) -> Vec<UVW> {
        let (s_ha, c_ha) = phase_centre.ha.sin_cos();
        let (s_dec, c_dec) = phase_centre.dec.sin_cos();

        // Get a UVW for each tile.
        let tile_uvws: Vec<UVW> = xyzs
            .iter()
            .map(|xyz| {
                let bl = XyzBaseline {
                    x: xyz.x,
                    y: xyz.y,
                    z: xyz.z,
                };
                UVW::from_xyz_inner(&bl, s_ha, c_ha, s_dec, c_dec)
            })
            .collect();

        // Take the difference of every pair of UVWs.
        let num_tiles = xyzs.len();
        let num_baselines = (num_tiles * (num_tiles - 1)) / 2;
        let mut bl_uvws = Vec::with_capacity(num_baselines);
        for i in 0..num_tiles {
            for j in i + 1..num_tiles {
                let tile_1 = tile_uvws[i];
                let tile_2 = tile_uvws[j];
                let uvw_bl = UVW {
                    u: tile_1.u - tile_2.u,
                    v: tile_1.v - tile_2.v,
                    w: tile_1.w - tile_2.w,
                };
                bl_uvws.push(uvw_bl);
            }
        }
        bl_uvws
    }
}

impl std::ops::Sub<XYZ> for XYZ {
    type Output = XyzBaseline;

    fn sub(self, rhs: Self) -> XyzBaseline {
        XyzBaseline {
            x: self.x - rhs.x,
            y: self.y - rhs.y,
            z: self.z - rhs.z,
        }
    }
}

impl std::ops::Sub<&XYZ> for XYZ {
    type Output = XyzBaseline;

    fn sub(self, rhs: &Self) -> XyzBaseline {
        XyzBaseline {
            x: self.x - rhs.x,
            y: self.y - rhs.y,
            z: self.z - rhs.z,
        }
    }
}

/// The (x,y,z) coordinates of a baseline. All units are in metres.
///
/// This coordinate system is discussed at length in Interferometry and
/// Synthesis in Radio Astronomy, Third Edition, Section 4: Geometrical
/// Relationships, Polarimetry, and the Measurement Equation.
#[derive(Clone, Debug)]
pub struct XyzBaseline {
    /// x-coordinate \[meters\]
    pub x: f64,
    /// y-coordinate \[meters\]
    pub y: f64,
    /// z-coordinate \[meters\]
    pub z: f64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::*;

    #[test]
    fn get_xyz_baselines_test() {
        let xyz = vec![
            XYZ {
                x: 289.5692922664971,
                y: -585.6749877929688,
                z: -259.3106530519151,
            },
            XYZ {
                x: 520.0443773794285,
                y: -575.5570068359375,
                z: 202.96211607459455,
            },
            XYZ {
                x: 120.0443773794285,
                y: -375.5570068359375,
                z: 2.96211607459455,
            },
            XYZ {
                x: -230.47508511293142,
                y: -10.11798095703125,
                z: -462.2727691265096,
            },
        ];

        let expected = vec![
            XyzBaseline {
                x: -230.47508511293142,
                y: -10.11798095703125,
                z: -462.2727691265096,
            },
            XyzBaseline {
                x: 169.52491488706858,
                y: -210.11798095703125,
                z: -262.2727691265097,
            },
            XyzBaseline {
                x: 520.0443773794285,
                y: -575.5570068359375,
                z: 202.96211607459452,
            },
            XyzBaseline {
                x: 400.0,
                y: -200.0,
                z: 200.0,
            },
            XyzBaseline {
                x: 750.5194624923599,
                y: -565.4390258789063,
                z: 665.2348852011041,
            },
            XyzBaseline {
                x: 350.51946249235993,
                y: -365.43902587890625,
                z: 465.2348852011042,
            },
        ];

        let diffs = XYZ::get_baselines(&xyz);
        assert_eq!(diffs.len(), 6);
        for (exp, diff) in expected.iter().zip(diffs.iter()) {
            assert_abs_diff_eq!(exp.x, diff.x, epsilon = 1e-10);
            assert_abs_diff_eq!(exp.y, diff.y, epsilon = 1e-10);
            assert_abs_diff_eq!(exp.z, diff.z, epsilon = 1e-10);
        }
    }
}
