// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

use fitsio::FitsFile;
use rayon::prelude::*;

use crate::constants::*;

/// East, North and Height coordinates of an MWA tile.
#[derive(Debug)]
struct ENH {
    /// East [metres]
    e: f64,
    /// North [metres]
    n: f64,
    /// Height [metres]
    h: f64,
}

impl ENH {
    /// Convert ENH coordinates to local XYZ. Helper function.
    ///
    /// Taken from the third edition of Interferometry and Synthesis in Radio
    /// Astronomy, chapter 4: Geometrical Relationships, Polarimetry, and the
    /// Measurement Equation.
    fn to_xyz(&self) -> XYZ {
        XYZ {
            x: -self.n * *MWA_LAT_SIN + self.h * *MWA_LAT_COS,
            y: self.e,
            z: self.n * *MWA_LAT_COS + self.h * *MWA_LAT_SIN,
        }
    }
}

/// The (l,m,n) direction-cosine coordinates of a point. All units are in
/// radians.
///
/// This coordinate system is discussed at length in Interferometry and
/// Synthesis in Radio Astronomy, Third Edition, Section 4: Geometrical
/// Relationships, Polarimetry, and the Measurement Equation.
#[derive(Debug)]
pub struct LMN {
    /// l-coordinate [radians]
    pub l: f64,
    /// m-coordinate [radians]
    pub m: f64,
    /// n-coordinate [radians]
    pub n: f64,
}

impl LMN {
    /// Convert a vector of `LMN` structs to L, M, and N vectors. Useful for
    /// FFI.
    pub fn decompose(mut lmn: Vec<Self>) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let length = lmn.len();
        let mut l = Vec::with_capacity(length);
        let mut m = Vec::with_capacity(length);
        let mut n = Vec::with_capacity(length);
        for elem in lmn.drain(..) {
            l.push(elem.l as f32);
            m.push(elem.m as f32);
            n.push(elem.n as f32);
        }
        // Ensure that the capacity of the vectors matches their length.
        l.shrink_to_fit();
        m.shrink_to_fit();
        n.shrink_to_fit();
        (l, m, n)
    }
}

/// The pointing centre coordinates.
#[derive(Clone, Copy, Debug)]
pub struct PC {
    /// Local sidereal time [radians]
    pub lst: f64,
    /// Hour angle [radians]
    pub ha: f64,
    /// sin(ha)
    pub s_ha: f64,
    /// cos(ha)
    pub c_ha: f64,
    /// Right ascension [radians]
    pub ra: f64,
    /// Declination [radians]
    pub dec: f64,
    /// sin(dec)
    pub s_dec: f64,
    /// cos(dec)
    pub c_dec: f64,
}

impl PC {
    /// Generate a `PC` using an hour angle `ha` and declination `dec`.
    ///
    /// As the pointing centre struct saves sine and cosine values, this `new`
    /// function exists to ease reduce programmer effort.
    pub fn new_from_ha(lst: f64, ha: f64, dec: f64) -> Self {
        Self {
            lst,
            ha,
            s_ha: ha.sin(),
            c_ha: ha.cos(),
            ra: lst - ha,
            dec,
            s_dec: dec.sin(),
            c_dec: dec.cos(),
        }
    }

    /// Similar to `PC::new_from_ra`, but takes a right ascension `ra` instead
    /// of an hour angle.
    pub fn new_from_ra(lst: f64, ra: f64, dec: f64) -> Self {
        let ha = lst - ra;
        Self {
            lst,
            ha,
            s_ha: ha.sin(),
            c_ha: ha.cos(),
            ra,
            dec,
            s_dec: dec.sin(),
            c_dec: dec.cos(),
        }
    }

    /// Given a new LST, update a pointing centre struct.
    pub fn update(&mut self, lst: f64) {
        self.ha += lst - self.lst;
        self.s_ha = self.ha.sin();
        self.c_ha = self.ha.cos();
        self.lst = lst;
    }
}

/// A struct containing a Right Ascension and Declination. As the sine and
/// cosine of these coordinates is often used, these are also stored in the
/// struct. All units are in radians.
#[derive(Clone, Copy, Debug)]
pub struct RADec {
    /// Right ascension [radians]
    pub ra: f64,
    /// sin(ha)
    pub s_ra: f64,
    /// cos(ha)
    pub c_ra: f64,
    /// Declination [radians]
    pub dec: f64,
    /// sin(dec)
    pub s_dec: f64,
    /// cos(dec)
    pub c_dec: f64,
}

impl RADec {
    /// As the pointing centre struct saves sine and cosine values, this `new`
    /// function exists to ease reduce programmer effort.
    pub fn new(ra: f64, dec: f64) -> Self {
        Self {
            ra,
            s_ra: ra.sin(),
            c_ra: ra.cos(),
            dec,
            s_dec: dec.sin(),
            c_dec: dec.cos(),
        }
    }

    /// Get the (l,m,n) direction cosines from these coordinates. All arguments
    /// are in radians.
    ///
    /// Derived using "Coordinate transformations" on page 388 of Synthesis
    /// Imaging in Radio Astronomy II.
    pub fn to_lmn(&self, pc: &PC) -> LMN {
        let d_ra = self.ra - pc.ra;
        let s_d_ra = d_ra.sin();
        let c_d_ra = d_ra.cos();
        LMN {
            l: self.c_dec * s_d_ra,
            m: self.s_dec * pc.c_dec - self.c_dec * pc.s_dec * c_d_ra,
            n: self.s_dec * pc.s_dec + self.c_dec * pc.c_dec * c_d_ra,
        }
    }
}

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
    /// Convert all XYZ baselines to UVW, given a pointing centre
    /// struct. Processing is done in parallel on the CPU.
    ///
    /// This is Equation 4.1 of: Interferometry and Synthesis in Radio
    /// Astronomy, Third Edition, Section 4: Geometrical Relationships,
    /// Polarimetry, and the Measurement Equation.
    pub fn get_baselines(xyz: &[XyzBaseline], pc: &PC) -> Vec<Self> {
        xyz.par_iter()
            .map(|l| Self {
                u: pc.s_ha * l.x + pc.c_ha * l.y,
                v: pc.s_dec * pc.s_ha * l.y + pc.c_dec * l.z - pc.s_dec * pc.c_ha * l.x,
                w: pc.c_dec * pc.c_ha * l.x - pc.c_dec * pc.s_ha * l.y + pc.s_dec * l.z,
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

/// The (x,y,z) coordinates of an antenna (a.k.a. station), or a baseline. All
/// units are in metres.
///
/// This coordinate system is discussed at length in Interferometry and
/// Synthesis in Radio Astronomy, Third Edition, Section 4: Geometrical
/// Relationships, Polarimetry, and the Measurement Equation.
#[derive(Clone, Copy, Debug)]
pub struct XYZ {
    /// x-coordinate [meters]
    pub x: f64,
    /// y-coordinate [meters]
    pub y: f64,
    /// z-coordinate [meters]
    pub z: f64,
}

impl XYZ {
    /// Convert coords in local topocentric East, North, Height units to 'local'
    /// XYZ units. Local means Z point north, X points through the equator from
    /// the geocenter along the local meridian and Y is East. This is like the
    /// absolute system except that zero longitude is now the local meridian
    /// rather than prime meridian. Latitude is geodetic, in radians. This is
    /// what you want for constructing the local antenna positions in a UVFITS
    /// antenna table.
    ///
    /// Assumes that the array used is the MWA.
    fn get_xyz_metafits(metafits: &mut FitsFile) -> Result<Vec<Self>, fitsio::errors::Error> {
        // We assume this is MWA data, and that the second HDU (index 1 here)
        // contains the ENH data. The data is stored as f32.
        let hdu = metafits.hdu(1)?;
        let east: Vec<f32> = hdu.read_col(metafits, "East")?;
        let north: Vec<f32> = hdu.read_col(metafits, "North")?;
        let height: Vec<f32> = hdu.read_col(metafits, "Height")?;

        // Convert the coordinates. Do this as f64 for accuracy. We only need to
        // take every second value, because every pair of rows corresponds to a
        // single tile (the values are the same).
        let xyz: Vec<_> = east
            .iter()
            .step_by(2)
            .zip(north.iter().step_by(2))
            .zip(height.iter().step_by(2))
            .map(|((e, n), h)| {
                ENH {
                    e: *e as f64,
                    n: *n as f64,
                    h: *h as f64,
                }
                .to_xyz()
            })
            .collect();
        Ok(xyz)
    }

    /// For each XYZ pair, calculate a baseline.
    pub fn get_xyz_baselines(xyz: &[Self]) -> Vec<XyzBaseline> {
        // Assume that the length of `xyz` is the number of tiles.
        let n_tiles = xyz.len();
        let n_baselines = n_tiles / 2 * (n_tiles - 1);
        let mut diffs = Vec::with_capacity(n_baselines);
        for i in 0..n_tiles {
            for j in i + 1..n_tiles {
                diffs.push(xyz[i] - xyz[j]);
            }
        }
        diffs
    }

    /// For each XYZ pair listed in a metafits file, calculate a baseline.
    ///
    /// Note that the baselines are ordered according to the metafits;
    /// i.e. Tile104 is often the first tile listed, so the first baseline is
    /// Tile104 and Tile103.
    pub fn get_baselines_metafits(
        metafits: &mut FitsFile,
    ) -> Result<Vec<XyzBaseline>, fitsio::errors::Error> {
        let xyz = Self::get_xyz_metafits(metafits)?;
        Ok(XYZ::get_xyz_baselines(&xyz))
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

/// The (x,y,z) coordinates of a baseline. All units are in metres.
///
/// This coordinate system is discussed at length in Interferometry and
/// Synthesis in Radio Astronomy, Third Edition, Section 4: Geometrical
/// Relationships, Polarimetry, and the Measurement Equation.
#[derive(Clone, Copy, Debug)]
pub struct XyzBaseline {
    /// x-coordinate [meters]
    pub x: f64,
    /// y-coordinate [meters]
    pub y: f64,
    /// z-coordinate [meters]
    pub z: f64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use float_cmp::approx_eq;

    #[test]
    fn test_to_lmn() {
        let radec = RADec::new(62.0_f64.to_radians(), (-27.5_f64).to_radians());
        let pc = PC::new_from_ra(0.0, 60_f64.to_radians(), (-27.0_f64).to_radians());
        let lmn = radec.to_lmn(&pc);
        let expected = LMN {
            l: 0.03095623164758603,
            m: -0.008971846102111436,
            n: 0.9994804738961642,
        };
        assert!(
            approx_eq!(f64, lmn.l, expected.l, epsilon = 1e-10),
            "lmn.l ({}) didn't equal expected value ({})!",
            lmn.l,
            expected.l
        );
        assert!(
            approx_eq!(f64, lmn.m, expected.m, epsilon = 1e-10),
            "lmn.m ({}) didn't equal expected value ({})!",
            lmn.m,
            expected.m
        );
        assert!(
            approx_eq!(f64, lmn.n, expected.n, epsilon = 1e-10),
            "lmn.n ({}) didn't equal expected value ({})!",
            lmn.n,
            expected.n
        )
    }

    #[test]
    fn convert_enh_to_xyz_test() {
        let enh = ENH {
            n: -101.530,
            e: -585.675,
            h: 375.212,
        };
        let xyz = ENH::to_xyz(&enh);
        assert!(approx_eq!(f64, xyz.x, 289.5692867016053, epsilon = 1e-10));
        assert!(approx_eq!(f64, xyz.y, -585.675, epsilon = 1e-10));
        assert!(approx_eq!(f64, xyz.z, -259.3106516191025, epsilon = 1e-10));
    }

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

        let diffs = XYZ::get_xyz_baselines(&xyz);
        for (exp, diff) in expected.iter().zip(diffs.iter()) {
            assert!(approx_eq!(f64, exp.x, diff.x, epsilon = 1e-10));
            assert!(approx_eq!(f64, exp.y, diff.y, epsilon = 1e-10));
            assert!(approx_eq!(f64, exp.z, diff.z, epsilon = 1e-10));
        }
    }

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
        let pc = PC::new_from_ha(6.07181, 6.0163, -0.453121);
        let uvw = UVW::get_baselines(&xyz_bl, &pc);
        println!("{:?}", uvw);
        let expected = UVW {
            u: 102.04605530570603,
            v: -1028.2293398297727,
            w: 0.18220641926160397,
        };
        assert!(approx_eq!(f64, uvw[0].u, expected.u, epsilon = 1e-10));
        assert!(approx_eq!(f64, uvw[0].v, expected.v, epsilon = 1e-10));
        assert!(approx_eq!(f64, uvw[0].w, expected.w, epsilon = 1e-10));
    }
}
