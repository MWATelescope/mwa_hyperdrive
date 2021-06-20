// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Handle (x,y,z) coordinates of an antenna (a.k.a. tile or station), geodetic
//! or geocentric.
//!
//! hyperdrive prefers to keep track of [XyzGeodetic] coordinates, as these are
//! what are needed to calculate [UVW]s.
//!
//! This coordinate system is discussed at length in Interferometry and
//! Synthesis in Radio Astronomy, Third Edition, Section 4: Geometrical
//! Relationships, Polarimetry, and the Measurement Equation.

use rayon::prelude::*;
use thiserror::Error;

use crate::{
    constants::{MWA_HEIGHT_M, MWA_LAT_RAD, MWA_LONG_RAD},
    HADec, ENH, UVW,
};

/// The geodetic (x,y,z) coordinates of an antenna (a.k.a. tile or station). All
/// units are in metres.
///
/// This coordinate system is discussed at length in Interferometry and
/// Synthesis in Radio Astronomy, Third Edition, Section 4: Geometrical
/// Relationships, Polarimetry, and the Measurement Equation.
#[derive(Clone, Debug, Default)]
pub struct XyzGeodetic {
    /// x-coordinate \[meters\]
    pub x: f64,
    /// y-coordinate \[meters\]
    pub y: f64,
    /// z-coordinate \[meters\]
    pub z: f64,
}

impl XyzGeodetic {
    /// Convert [XyzGeodetic] coordinates at a latitude to [ENH] coordinates.
    pub fn to_enh(&self, latitude: f64) -> ENH {
        let (s_lat, c_lat) = latitude.sin_cos();
        ENH {
            e: self.y,
            n: -self.x * s_lat + self.z * c_lat,
            h: self.x * c_lat + self.z * s_lat,
        }
    }

    /// Convert [XyzGeodetic] coordinates at the MWA's latitude to [ENH]
    /// coordinates.
    pub fn to_enh_mwa(&self) -> ENH {
        self.to_enh(MWA_LAT_RAD)
    }

    /// For each [XyzGeodetic] pair, calculate an [XyzBaseline].
    pub fn get_baselines(xyz: &[Self]) -> Vec<XyzBaseline> {
        // Assume that the length of `xyz` is the number of tiles.
        let num_tiles = xyz.len();
        let num_baselines = (num_tiles * (num_tiles - 1)) / 2;
        let mut diffs = Vec::with_capacity(num_baselines);
        for i in 0..num_tiles {
            for j in i + 1..num_tiles {
                diffs.push(xyz[i].clone() - &xyz[j]);
            }
        }
        diffs
    }

    /// For each tile listed in an mwalib context, calculate a [XyzGeodetic]
    /// coordinate.
    ///
    /// Note that the RF inputs are ordered by antenna number, **not** the
    /// "input"; e.g. in the metafits file, Tile104 is often the first tile
    /// listed ("input" 0), Tile103 second ("input" 2), so the first baseline
    /// would naively be between Tile104 and Tile103.
    pub fn get_tiles_mwalib(context: &mwalib::MetafitsContext) -> Vec<Self> {
        context
            .rf_inputs
            .iter()
            // There is an RF input for both tile polarisations. The ENH
            // coordinates are the same for both polarisations of a tile; ignore
            // the RF input if it's associated with Y.
            .filter(|rf| matches!(rf.pol, mwalib::Pol::Y))
            .map(|rf| {
                ENH {
                    e: rf.east_m,
                    n: rf.north_m,
                    h: rf.height_m,
                }
                .to_xyz_mwa()
            })
            .collect()
    }

    /// For each tile listed in an mwalib context, calculate an [XyzBaseline].
    ///
    /// Note that the RF inputs are ordered by antenna number, **not** the
    /// "input"; e.g. in the metafits file, Tile104 is often the first tile
    /// listed ("input" 0), Tile103 second ("input" 2), so the first baseline
    /// would naively be between Tile104 and Tile103.
    pub fn get_baselines_mwalib(context: &mwalib::MetafitsContext) -> Vec<XyzBaseline> {
        Self::get_baselines(&Self::get_tiles_mwalib(context))
    }

    fn to_geocentric_inner(
        &self,
        geocentric_vector: &XyzGeocentric,
        sin_longitude: f64,
        cos_longitude: f64,
    ) -> XyzGeocentric {
        let xtemp = self.x * cos_longitude - self.y * sin_longitude;
        let y = self.x * sin_longitude + self.y * cos_longitude;
        let x = xtemp;

        XyzGeocentric {
            x: x + geocentric_vector.x,
            y: y + geocentric_vector.y,
            z: self.z + geocentric_vector.z,
        }
    }

    /// Convert a [XyzGeodetic] coordinate to [XyzGeocentric].
    pub fn to_geocentric(
        &self,
        longitude_rad: f64,
        latitude_rad: f64,
        height_metres: f64,
    ) -> Result<XyzGeocentric, ErfaError> {
        let (sin_longitude, cos_longitude) = longitude_rad.sin_cos();
        let geocentric_vector =
            XyzGeocentric::get_geocentric_vector(longitude_rad, latitude_rad, height_metres)?;
        Ok(XyzGeodetic::to_geocentric_inner(
            self,
            &geocentric_vector,
            sin_longitude,
            cos_longitude,
        ))
    }

    /// Convert a [XyzGeodetic] coordinate to [XyzGeocentric], using the MWA's
    /// location.
    pub fn to_geocentric_mwa(&self) -> Result<XyzGeocentric, ErfaError> {
        self.to_geocentric(MWA_LONG_RAD, MWA_LAT_RAD, MWA_HEIGHT_M)
    }
}

/// Convert [XyzGeodetic] coordinates to [UVW]s without having to form
/// [XyzBaseline]s first.
pub fn xyzs_to_uvws(xyzs: &[XyzGeodetic], phase_centre: &HADec) -> Vec<UVW> {
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

/// Convert many [XyzGeodetic] coordinates to [XyzGeocentric].
pub fn geodetic_to_geocentric(
    geodetics: &[XyzGeodetic],
    longitude_rad: f64,
    latitude_rad: f64,
    height_metres: f64,
) -> Result<Vec<XyzGeocentric>, ErfaError> {
    let (sin_longitude, cos_longitude) = longitude_rad.sin_cos();
    let geocentric_vector =
        XyzGeocentric::get_geocentric_vector(longitude_rad, latitude_rad, height_metres)?;
    let geocentrics = geodetics
        .iter()
        .map(|gd| gd.to_geocentric_inner(&geocentric_vector, sin_longitude, cos_longitude))
        .collect();
    Ok(geocentrics)
}

/// Convert many [XyzGeodetic] coordinates to [XyzGeocentric], using the MWA's
/// location.
pub fn geodetic_to_geocentric_mwa(
    geodetics: &[XyzGeodetic],
) -> Result<Vec<XyzGeocentric>, ErfaError> {
    geodetic_to_geocentric(geodetics, MWA_LONG_RAD, MWA_LAT_RAD, MWA_HEIGHT_M)
}

impl std::ops::Sub<XyzGeodetic> for XyzGeodetic {
    type Output = XyzBaseline;

    fn sub(self, rhs: Self) -> XyzBaseline {
        XyzBaseline {
            x: self.x - rhs.x,
            y: self.y - rhs.y,
            z: self.z - rhs.z,
        }
    }
}

impl std::ops::Sub<&XyzGeodetic> for XyzGeodetic {
    type Output = XyzBaseline;

    fn sub(self, rhs: &Self) -> XyzBaseline {
        XyzBaseline {
            x: self.x - rhs.x,
            y: self.y - rhs.y,
            z: self.z - rhs.z,
        }
    }
}

/// The geodetic (x,y,z) coordinates of a baseline. All units are in metres.
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

/// The geocentric (x,y,z) coordinates of an antenna (a.k.a. tile or station).
/// All units are in metres.
///
/// This coordinate system is discussed at length in Interferometry and
/// Synthesis in Radio Astronomy, Third Edition, Section 4: Geometrical
/// Relationships, Polarimetry, and the Measurement Equation.
#[derive(Clone, Debug, Default)]
pub struct XyzGeocentric {
    /// x-coordinate \[meters\]
    pub x: f64,
    /// y-coordinate \[meters\]
    pub y: f64,
    /// z-coordinate \[meters\]
    pub z: f64,
}

impl XyzGeocentric {
    /// Get a geocentric coordinate vector With the given geodetic coordinates
    /// (longitude, latitude and height). The ellipsoid model is WGS84.
    fn get_geocentric_vector(
        longitude_rad: f64,
        latitude_rad: f64,
        height_metres: f64,
    ) -> Result<XyzGeocentric, ErfaError> {
        let mut geocentric_vector: [f64; 3] = [0.0; 3];
        let status = unsafe {
            erfa_sys::eraGd2gc(
                erfa_sys::ERFA_WGS84 as i32,    // ellipsoid identifier (Note 1)
                longitude_rad,                  // longitude (radians, east +ve)
                latitude_rad,                   // latitude (geodetic, radians, Note 3)
                height_metres,                  // height above ellipsoid (geodetic, Notes 2,3)
                geocentric_vector.as_mut_ptr(), // geocentric vector (Note 2)
            )
        };
        if status != 0 {
            return Err(ErfaError {
                source_file: file!(),
                source_line: line!(),
                status,
                function: "eraGd2gc",
            });
        }
        Ok(XyzGeocentric {
            x: geocentric_vector[0],
            y: geocentric_vector[1],
            z: geocentric_vector[2],
        })
    }

    // TODO: Account for northing and eastings. Australia drifts by ~7cm/year,
    // and the ellipsoid model probably need to be changed too!
    #[inline]
    fn to_geodetic_inner(
        &self,
        geocentric_vector: &Self,
        sin_longitude: f64,
        cos_longitude: f64,
    ) -> XyzGeodetic {
        let geodetic = XyzGeodetic {
            x: self.x - geocentric_vector.x,
            y: self.y - geocentric_vector.y,
            z: self.z - geocentric_vector.z,
        };

        let xtemp = geodetic.x * cos_longitude - geodetic.y * sin_longitude;
        let y = geodetic.x * sin_longitude + geodetic.y * cos_longitude;
        let x = xtemp;
        XyzGeodetic {
            x,
            y,
            z: geodetic.z,
        }
    }

    /// Convert a [XyzGeocentric] coordinate to [XyzGeodetic].
    pub fn to_geodetic(
        &self,
        longitude_rad: f64,
        latitude_rad: f64,
        height_metres: f64,
    ) -> Result<XyzGeodetic, ErfaError> {
        let geocentric_vector =
            XyzGeocentric::get_geocentric_vector(longitude_rad, latitude_rad, height_metres)?;
        let (sin_longitude, cos_longitude) = (-longitude_rad).sin_cos();
        let geodetic = XyzGeocentric::to_geodetic_inner(
            &self,
            &geocentric_vector,
            sin_longitude,
            cos_longitude,
        );
        Ok(geodetic)
    }

    /// Convert a [XyzGeocentric] coordinate to [XyzGeodetic], using the MWA's
    /// location.
    pub fn to_geodetic_mwa(&self) -> Result<XyzGeodetic, ErfaError> {
        self.to_geodetic(MWA_LONG_RAD, MWA_LAT_RAD, MWA_HEIGHT_M)
    }
}

/// Convert many [XyzGeocentric] coordinates to [XyzGeodetic].
pub fn geocentric_to_geodetic(
    geocentrics: &[XyzGeocentric],
    longitude_rad: f64,
    latitude_rad: f64,
    height_metres: f64,
) -> Result<Vec<XyzGeodetic>, ErfaError> {
    let geocentric_vector =
        XyzGeocentric::get_geocentric_vector(longitude_rad, latitude_rad, height_metres)?;
    let (sin_longitude, cos_longitude) = (-longitude_rad).sin_cos();
    let geodetics = geocentrics
        .iter()
        .map(|gc| gc.to_geodetic_inner(&geocentric_vector, sin_longitude, cos_longitude))
        .collect();
    Ok(geodetics)
}

/// Convert many [XyzGeocentric] coordinates to [XyzGeodetic], using the MWA's
/// location.
pub fn geocentric_to_geodetic_mwa(
    geocentrics: &[XyzGeocentric],
) -> Result<Vec<XyzGeodetic>, ErfaError> {
    geocentric_to_geodetic(geocentrics, MWA_LONG_RAD, MWA_LAT_RAD, MWA_HEIGHT_M)
}

/// Convert many [XyzGeocentric] coordinates to [XyzGeodetic]. The calculations
/// are done in parallel.
pub fn geocentric_to_geodetic_parallel(
    geocentrics: &[XyzGeocentric],
    longitude_rad: f64,
    latitude_rad: f64,
    height_metres: f64,
) -> Result<Vec<XyzGeodetic>, ErfaError> {
    let geocentric_vector =
        XyzGeocentric::get_geocentric_vector(longitude_rad, latitude_rad, height_metres)?;
    let (sin_longitude, cos_longitude) = (-longitude_rad).sin_cos();
    let geodetics = geocentrics
        .par_iter()
        .map(|gc| {
            XyzGeocentric::to_geodetic_inner(gc, &geocentric_vector, sin_longitude, cos_longitude)
        })
        .collect();
    Ok(geodetics)
}

#[derive(Error, Debug)]
#[error(
    "{source_file}:{source_line} Call to ERFA function {function} returned status code {status}"
)]
pub struct ErfaError {
    source_file: &'static str,
    source_line: u32,
    status: i32,
    function: &'static str,
}

#[cfg(test)]
mod tests {
    use crate::constants::{
        COTTER_MWA_HEIGHT_METRES, COTTER_MWA_LATITUDE_RADIANS, COTTER_MWA_LONGITUDE_RADIANS,
    };

    use super::*;
    use approx::*;

    #[test]
    fn get_xyz_baselines_test() {
        let xyz = vec![
            XyzGeodetic {
                x: 289.5692922664971,
                y: -585.6749877929688,
                z: -259.3106530519151,
            },
            XyzGeodetic {
                x: 520.0443773794285,
                y: -575.5570068359375,
                z: 202.96211607459455,
            },
            XyzGeodetic {
                x: 120.0443773794285,
                y: -375.5570068359375,
                z: 2.96211607459455,
            },
            XyzGeodetic {
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

        let diffs = XyzGeodetic::get_baselines(&xyz);
        assert_eq!(diffs.len(), 6);
        for (exp, diff) in expected.iter().zip(diffs.iter()) {
            assert_abs_diff_eq!(exp.x, diff.x, epsilon = 1e-10);
            assert_abs_diff_eq!(exp.y, diff.y, epsilon = 1e-10);
            assert_abs_diff_eq!(exp.z, diff.z, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_geocentric_to_geodetic_and_back() {
        // These geodetic XYZ positions are taken from a uvfits made from cotter
        // for Tile011.
        let uvfits_xyz = XyzGeodetic {
            x: 4.56250049e+02,
            y: -1.49785004e+02,
            z: 6.80459899e+01,
        };
        // These geocentric XYZ positions are taken from a MS made from cotter
        // for Tile011.
        let ms_xyz = XyzGeocentric {
            x: -2559524.23682043,
            y: 5095846.67363471,
            z: -2848988.72758185,
        };

        // Check the conversion of geocentric to geodetic.
        let result = ms_xyz.to_geodetic_mwa();
        assert!(result.is_ok());
        let local_xyz = result.unwrap();

        // cotter's MWA coordinates are a little off of what is in mwalib.
        // Verify that the transformation isn't quite right.
        assert_abs_diff_ne!(uvfits_xyz.x, local_xyz.x, epsilon = 1e-1);
        assert_abs_diff_ne!(uvfits_xyz.y, local_xyz.y, epsilon = 1e-1);
        assert_abs_diff_ne!(uvfits_xyz.z, local_xyz.z, epsilon = 1e-1);

        // Now verify cotter's ms XYZ with the constants it uses.
        let result = ms_xyz.to_geodetic(
            COTTER_MWA_LONGITUDE_RADIANS,
            COTTER_MWA_LATITUDE_RADIANS,
            COTTER_MWA_HEIGHT_METRES,
        );
        assert!(result.is_ok());
        let local_xyz = result.unwrap();
        assert_abs_diff_eq!(uvfits_xyz.x, local_xyz.x, epsilon = 1e-6);
        assert_abs_diff_eq!(uvfits_xyz.y, local_xyz.y, epsilon = 1e-6);
        assert_abs_diff_eq!(uvfits_xyz.z, local_xyz.z, epsilon = 1e-6);

        // Now check the conversion of geodetic to geocentric.
        let result = uvfits_xyz.to_geocentric_mwa();
        assert!(result.is_ok());
        let geocentric_xyz = result.unwrap();
        // cotter's MWA coordinates are a little off of what is in mwalib.
        // Verify that the transformation isn't quite right.
        assert_abs_diff_ne!(ms_xyz.x, geocentric_xyz.x, epsilon = 1e-1);
        assert_abs_diff_ne!(ms_xyz.y, geocentric_xyz.y, epsilon = 1e-1);
        assert_abs_diff_ne!(ms_xyz.z, geocentric_xyz.z, epsilon = 1e-1);

        // Now verify cotter's ms XYZ with the constants it uses.
        let result = uvfits_xyz.to_geocentric(
            COTTER_MWA_LONGITUDE_RADIANS,
            COTTER_MWA_LATITUDE_RADIANS,
            COTTER_MWA_HEIGHT_METRES,
        );
        assert!(result.is_ok());
        let geocentric_xyz = result.unwrap();
        assert_abs_diff_eq!(ms_xyz.x, geocentric_xyz.x, epsilon = 1e-6);
        assert_abs_diff_eq!(ms_xyz.y, geocentric_xyz.y, epsilon = 1e-6);
        assert_abs_diff_eq!(ms_xyz.z, geocentric_xyz.z, epsilon = 1e-6);
    }

    #[test]
    fn test_batch_geodetic_methods() {
        let gds = vec![
            XyzGeodetic {
                x: 4.56250049e+02,
                y: -1.49785004e+02,
                z: 6.80459899e+01,
            },
            XyzGeodetic {
                x: 4.57250049e+02,
                y: -1.50785004e+02,
                z: 6.70459899e+01,
            },
            XyzGeodetic {
                x: 4.46250049e+02,
                y: -1.48785004e+02,
                z: 6.82459899e+01,
            },
        ];

        let result = geodetic_to_geocentric_mwa(&gds);
        assert!(result.is_ok());
        let batch = result.unwrap();

        let result: Vec<Result<_, _>> = gds.iter().map(|gd| gd.to_geocentric_mwa()).collect();
        let result: Result<Vec<_>, _> = result.into_iter().collect();
        assert!(result.is_ok());
        let individual = result.unwrap();

        for (b, i) in batch.iter().zip(individual.iter()) {
            assert_abs_diff_eq!(b.x, i.x, epsilon = 1e-10);
            assert_abs_diff_eq!(b.y, i.y, epsilon = 1e-10);
            assert_abs_diff_eq!(b.z, i.z, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_batch_geocentric_methods() {
        let gcs = vec![
            XyzGeocentric {
                x: -2559524.23682043,
                y: 5095846.67363471,
                z: -2848988.72758185,
            },
            XyzGeocentric {
                x: -2559526.23682043,
                y: 5095856.67363471,
                z: -2848968.72758185,
            },
            XyzGeocentric {
                x: -2559534.23682043,
                y: 5095849.67363471,
                z: -2848998.72758185,
            },
        ];

        let result = geocentric_to_geodetic_mwa(&gcs);
        assert!(result.is_ok());
        let batch = result.unwrap();

        let result: Vec<Result<_, _>> = gcs.iter().map(|gc| gc.to_geodetic_mwa()).collect();
        let result: Result<Vec<_>, _> = result.into_iter().collect();
        assert!(result.is_ok());
        let individual = result.unwrap();

        for (b, i) in batch.iter().zip(individual.iter()) {
            assert_abs_diff_eq!(b.x, i.x, epsilon = 1e-10);
            assert_abs_diff_eq!(b.y, i.y, epsilon = 1e-10);
            assert_abs_diff_eq!(b.z, i.z, epsilon = 1e-10);
        }
    }
}
