// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Code to handle precession.

use std::f64::consts::TAU;

use hifitime::Epoch;
use rayon::prelude::*;

use mwa_hyperdrive_core::{HADec, RADec, XyzGeodetic};

#[derive(Debug)]
pub(crate) struct PrecessionInfo {
    /// Bias procession rotation matrix.
    rotation_matrix: [[f64; 3]; 3],

    /// The precessed phase centre in the J2000 epoch.
    pub(crate) hadec_j2000: HADec,

    /// The LMST of the current epoch.
    pub(crate) lmst: f64,

    /// The precessed LMST in the J2000 epoch.
    pub(crate) lmst_j2000: f64,

    /// The precessed array latitude in the J2000 epoch.
    pub(crate) array_latitude_j2000: f64,
}

impl PrecessionInfo {
    // Blatently stolen from cotter.
    pub(crate) fn precess_xyz_parallel(&self, xyzs: &[XyzGeodetic]) -> Vec<XyzGeodetic> {
        let (sep, cep) = self.lmst.sin_cos();
        let (s2000, c2000) = self.lmst_j2000.sin_cos();
        let mut out = Vec::with_capacity(xyzs.len());

        xyzs.par_iter()
            .map(|xyz| {
                // rotate to frame with x axis at zero RA
                let xpr = cep * xyz.x - sep * xyz.y;
                let ypr = sep * xyz.x + cep * xyz.y;
                let zpr = xyz.z;

                let rmat = &self.rotation_matrix;
                let xpr2 = (rmat[0][0]) * xpr + (rmat[0][1]) * ypr + (rmat[0][2]) * zpr;
                let ypr2 = (rmat[1][0]) * xpr + (rmat[1][1]) * ypr + (rmat[1][2]) * zpr;
                let zpr2 = (rmat[2][0]) * xpr + (rmat[2][1]) * ypr + (rmat[2][2]) * zpr;

                // rotate back to frame with xp pointing out at lmst2000
                XyzGeodetic {
                    x: c2000 * xpr2 + s2000 * ypr2,
                    y: -s2000 * xpr2 + c2000 * ypr2,
                    z: zpr2,
                }
            })
            .collect_into_vec(&mut out);
        out
    }
}

pub(crate) fn get_unprecessed_lmst(time: &Epoch, array_longitude_rad: f64) -> f64 {
    unsafe {
        let gmst = pal_sys::palGmst(time.as_mjd_utc_days());
        (gmst + array_longitude_rad) % TAU
    }
}

// This function is very similar to cotter's `PrepareTimestepUVW`.
pub(crate) fn precess_time(
    phase_centre: &RADec,
    time: &Epoch,
    array_longitude_rad: f64,
    array_latitude_rad: f64,
) -> PrecessionInfo {
    let mjd = time.as_mjd_utc_days();

    unsafe {
        // Note that we explicitly use the LMST because we're handling nutation
        // ourselves.
        let lmst = {
            let gmst = pal_sys::palGmst(mjd);
            (gmst + array_longitude_rad) % TAU
        };

        let j2000 = 2000.0;
        let radec_aber = aber_radec_rad(j2000, mjd, phase_centre);
        let mut rotation_matrix = [[0.0; 3]; 3];
        pal_sys::palPrenut(j2000, mjd, rotation_matrix.as_mut_ptr());

        // Transpose the rotation matrix.
        let mut rotation_matrix = {
            let mut new = [[0.0; 3]; 3];
            let old = rotation_matrix;
            for i in 0..3 {
                for j in 0..3 {
                    new[j][i] = old[i][j];
                }
            }
            new
        };

        let precessed = hadec_j2000(&mut rotation_matrix, lmst, array_latitude_rad, &radec_aber);

        PrecessionInfo {
            rotation_matrix,
            hadec_j2000: precessed.hadec,
            lmst,
            lmst_j2000: precessed.lmst,
            array_latitude_j2000: precessed.latitude,
        }
    }
}

// Blatently stolen from cotter.
unsafe fn aber_radec_rad(eq: f64, mjd: f64, radec: &RADec) -> RADec {
    let mut v1 = [0.0; 3];
    let mut v2 = [0.0; 3];

    pal_sys::palDcs2c(radec.ra, radec.dec, v1.as_mut_ptr());
    stelaber(eq, mjd, &mut v1, &mut v2);
    let mut ra2 = 0.0;
    let mut dec2 = 0.0;
    pal_sys::palDcc2s(v2.as_mut_ptr(), &mut ra2, &mut dec2);
    ra2 = pal_sys::palDranrm(ra2);

    RADec::new(ra2, dec2)
}

// Blatently stolen from cotter.
unsafe fn stelaber(eq: f64, mjd: f64, v1: &mut [f64; 3], v2: &mut [f64; 3]) {
    let mut amprms = [0.0; 21];
    let mut v1n = [0.0; 3];
    let mut v2un = [0.0; 3];
    let mut abv = [0.0; 3];

    pal_sys::palMappa(eq, mjd, amprms.as_mut_ptr());

    /* Unpack scalar and vector parameters */
    let ab1 = &amprms[11];
    abv[0] = amprms[8];
    abv[1] = amprms[9];
    abv[2] = amprms[10];

    let mut w = 0.0;
    pal_sys::palDvn(v1.as_mut_ptr(), v1n.as_mut_ptr(), &mut w);

    /* Aberration (normalization omitted) */
    let p1dv = pal_sys::palDvdv(v1n.as_mut_ptr(), abv.as_mut_ptr());
    w = 1.0 + p1dv / (ab1 + 1.0);
    v2un[0] = ab1 * v1n[0] + w * abv[0];
    v2un[1] = ab1 * v1n[1] + w * abv[1];
    v2un[2] = ab1 * v1n[2] + w * abv[2];

    /* Normalize */
    pal_sys::palDvn(v2un.as_mut_ptr(), v2.as_mut_ptr(), &mut w);
}

/// The return type of `hadec_j2000`. All values are in radians.
struct HADecJ2000 {
    hadec: HADec,
    latitude: f64,
    lmst: f64,
}

unsafe fn hadec_j2000(
    rotation_matrix: &mut [[f64; 3]; 3],
    lmst: f64,
    lat_rad: f64,
    radec: &RADec,
) -> HADecJ2000 {
    let (new_lmst, new_lat) = rotate_radec(rotation_matrix, lmst, lat_rad);
    HADecJ2000 {
        hadec: HADec::new(pal_sys::palDranrm(new_lmst - radec.ra), radec.dec),
        latitude: new_lat,
        lmst: new_lmst,
    }
}

// Blatently stolen from cotter.
unsafe fn rotate_radec(rotation_matrix: &mut [[f64; 3]; 3], ra: f64, dec: f64) -> (f64, f64) {
    let mut v1 = [0.0; 3];
    let mut v2 = [0.0; 3];

    pal_sys::palDcs2c(ra, dec, v1.as_mut_ptr());
    pal_sys::palDmxv(
        rotation_matrix.as_mut_ptr(),
        v1.as_mut_ptr(),
        v2.as_mut_ptr(),
    );
    let mut ra2 = 0.0;
    let mut dec2 = 0.0;
    pal_sys::palDcc2s(v2.as_mut_ptr(), &mut ra2, &mut dec2);
    ra2 = pal_sys::palDranrm(ra2);

    (ra2, dec2)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::{assert_abs_diff_eq, assert_abs_diff_ne};
    use mwa_hyperdrive_core::mwalib::{MWA_LATITUDE_RADIANS, MWA_LONGITUDE_RADIANS};
    use std::str::FromStr;

    fn gps_to_epoch(gps: u32) -> Epoch {
        // https://en.wikipedia.org/wiki/Global_Positioning_System#Timekeeping
        // The difference between GPS and TAI time is always 19s, but hifitime
        // wants the number of TAI seconds since 1900. GPS time starts at 1980
        // Jan 5.
        let tai = gps as f64 + 19.0 + crate::constants::HIFITIME_GPS_FACTOR;
        Epoch::from_tai_seconds(tai)
    }

    #[test]
    fn test_no_precession_at_j2000() {
        let j2000_epoch = Epoch::from_str("2000-01-01T11:58:55.816 UTC").unwrap();

        let phase_centre = RADec::new_degrees(0.0, -27.0);
        let eor0 = precess_time(
            &phase_centre,
            &j2000_epoch,
            MWA_LONGITUDE_RADIANS,
            MWA_LATITUDE_RADIANS,
        );
        assert_abs_diff_eq!(eor0.rotation_matrix[0][0], 1.0, epsilon = 1e-4);
        assert_abs_diff_eq!(eor0.rotation_matrix[1][1], 1.0, epsilon = 1e-4);
        assert_abs_diff_eq!(eor0.rotation_matrix[2][2], 1.0, epsilon = 1e-4);
        assert_abs_diff_eq!(eor0.rotation_matrix[0][1], 0.0, epsilon = 1e-4);
        assert_abs_diff_eq!(eor0.rotation_matrix[0][2], 0.0, epsilon = 1e-4);
        assert_abs_diff_eq!(eor0.rotation_matrix[1][0], 0.0, epsilon = 1e-4);
        assert_abs_diff_eq!(eor0.rotation_matrix[1][2], 0.0, epsilon = 1e-4);
        assert_abs_diff_eq!(eor0.rotation_matrix[2][0], 0.0, epsilon = 1e-4);
        assert_abs_diff_eq!(eor0.rotation_matrix[2][1], 0.0, epsilon = 1e-4);

        assert_abs_diff_eq!(eor0.lmst_j2000 - eor0.hadec_j2000.ha, 0.0, epsilon = 1e-4);
        assert_abs_diff_eq!(
            eor0.array_latitude_j2000,
            MWA_LATITUDE_RADIANS,
            epsilon = 1e-4
        );
        assert_abs_diff_eq!(eor0.lmst, eor0.lmst_j2000, epsilon = 1e-4);

        assert_abs_diff_eq!(eor0.lmst, 0.6433259676052971, epsilon = 1e-4);

        let phase_centre = RADec::new_degrees(60.0, -30.0);
        let eor1 = precess_time(
            &phase_centre,
            &j2000_epoch,
            MWA_LONGITUDE_RADIANS,
            MWA_LATITUDE_RADIANS,
        );
        assert_abs_diff_eq!(eor1.rotation_matrix[0][0], 1.0, epsilon = 1e-4);
        assert_abs_diff_eq!(eor1.rotation_matrix[1][1], 1.0, epsilon = 1e-4);
        assert_abs_diff_eq!(eor1.rotation_matrix[2][2], 1.0, epsilon = 1e-4);
        assert_abs_diff_eq!(eor1.rotation_matrix[0][1], 0.0, epsilon = 1e-4);
        assert_abs_diff_eq!(eor1.rotation_matrix[0][2], 0.0, epsilon = 1e-4);
        assert_abs_diff_eq!(eor1.rotation_matrix[1][0], 0.0, epsilon = 1e-4);
        assert_abs_diff_eq!(eor1.rotation_matrix[1][2], 0.0, epsilon = 1e-4);
        assert_abs_diff_eq!(eor1.rotation_matrix[2][0], 0.0, epsilon = 1e-4);
        assert_abs_diff_eq!(eor1.rotation_matrix[2][1], 0.0, epsilon = 1e-4);

        assert_abs_diff_eq!(
            eor1.lmst_j2000 - eor1.hadec_j2000.ha,
            -5.235898085317921,
            epsilon = 1e-4
        );
        assert_abs_diff_eq!(
            eor1.array_latitude_j2000,
            MWA_LATITUDE_RADIANS,
            epsilon = 1e-4
        );
        assert_abs_diff_eq!(eor1.lmst, eor1.lmst_j2000, epsilon = 1e-4);

        assert_abs_diff_eq!(eor1.lmst, 0.6433259676052971, epsilon = 1e-4);
    }

    #[test]
    fn test_precess_1065880128_to_j2000() {
        let epoch = gps_to_epoch(1065880128);
        let phase_centre = RADec::new_degrees(0.0, -27.0);

        let p = precess_time(
            &phase_centre,
            &epoch,
            MWA_LONGITUDE_RADIANS,
            MWA_LATITUDE_RADIANS,
        );
        // How do I know this is right? Good question! ... I don't.
        assert_abs_diff_eq!(p.rotation_matrix[0][0], 1.0, epsilon = 1e-5);
        assert_abs_diff_eq!(p.rotation_matrix[1][1], 1.0, epsilon = 1e-5);
        assert_abs_diff_eq!(p.rotation_matrix[2][2], 1.0, epsilon = 1e-5);
        assert_abs_diff_ne!(p.rotation_matrix[0][1], 0.0, epsilon = 1e-5);
        assert_abs_diff_ne!(p.rotation_matrix[0][2], 0.0, epsilon = 1e-5);
        assert_abs_diff_ne!(p.rotation_matrix[1][0], 0.0, epsilon = 1e-5);
        assert_abs_diff_ne!(p.rotation_matrix[1][2], 0.0, epsilon = 1e-5);
        assert_abs_diff_ne!(p.rotation_matrix[2][0], 0.0, epsilon = 1e-5);
        assert_abs_diff_ne!(p.rotation_matrix[2][1], 0.0, epsilon = 1e-5);

        assert_abs_diff_eq!(p.hadec_j2000.ha, 6.0714305189419715, epsilon = 1e-10);
        assert_abs_diff_eq!(p.hadec_j2000.dec, -0.47122418312765446, epsilon = 1e-10);
        assert_abs_diff_eq!(p.lmst, 6.0747789094260245, epsilon = 1e-10);
        assert_abs_diff_eq!(p.lmst_j2000, 6.071524853456497, epsilon = 1e-10);
        assert_abs_diff_eq!(p.array_latitude_j2000, -0.467396549790915, epsilon = 1e-10);
        assert_abs_diff_ne!(p.array_latitude_j2000, MWA_LATITUDE_RADIANS, epsilon = 1e-5);

        let pc_hadec = phase_centre.to_hadec(p.lmst);
        let ha_diff_arcmin = (pc_hadec.ha - p.hadec_j2000.ha).to_degrees() * 60.0;
        let dec_diff_arcmin = (pc_hadec.dec - p.hadec_j2000.dec).to_degrees() * 60.0;
        assert_abs_diff_eq!(ha_diff_arcmin, 11.510918573880216, epsilon = 1e-5); // 11.5 arcmin!
        assert_abs_diff_eq!(dec_diff_arcmin, -0.05058613713495692, epsilon = 1e-5);
    }

    #[test]
    fn test_precess_1099334672_to_j2000() {
        let epoch = gps_to_epoch(1099334672);
        let phase_centre = RADec::new_degrees(60.0, -30.0);

        let p = precess_time(
            &phase_centre,
            &epoch,
            MWA_LONGITUDE_RADIANS,
            MWA_LATITUDE_RADIANS,
        );
        // How do I know this is right? Good question! ... I don't.
        assert_abs_diff_eq!(p.rotation_matrix[0][0], 1.0, epsilon = 1e-5);
        assert_abs_diff_eq!(p.rotation_matrix[1][1], 1.0, epsilon = 1e-5);
        assert_abs_diff_eq!(p.rotation_matrix[2][2], 1.0, epsilon = 1e-5);
        assert_abs_diff_ne!(p.rotation_matrix[0][1], 0.0, epsilon = 1e-5);
        assert_abs_diff_ne!(p.rotation_matrix[0][2], 0.0, epsilon = 1e-5);
        assert_abs_diff_ne!(p.rotation_matrix[1][0], 0.0, epsilon = 1e-5);
        assert_abs_diff_ne!(p.rotation_matrix[1][2], 0.0, epsilon = 1e-5);
        assert_abs_diff_ne!(p.rotation_matrix[2][0], 0.0, epsilon = 1e-5);
        assert_abs_diff_ne!(p.rotation_matrix[2][1], 0.0, epsilon = 1e-5);

        assert_abs_diff_eq!(p.hadec_j2000.ha, 0.409885996082088, epsilon = 1e-10);
        assert_abs_diff_eq!(p.hadec_j2000.dec, -0.5235637661235192, epsilon = 1e-10);
        assert_abs_diff_eq!(p.lmst, 1.4598017673520172, epsilon = 1e-10);
        assert_abs_diff_eq!(p.lmst_j2000, 1.4571918352968762, epsilon = 1e-10);
        assert_abs_diff_eq!(p.array_latitude_j2000, -0.4661807836570052, epsilon = 1e-10);
        assert_abs_diff_ne!(p.array_latitude_j2000, MWA_LATITUDE_RADIANS, epsilon = 1e-5);

        let pc_hadec = phase_centre.to_hadec(p.lmst);
        let ha_diff_arcmin = (pc_hadec.ha - p.hadec_j2000.ha).to_degrees() * 60.0;
        let dec_diff_arcmin = (pc_hadec.dec - p.hadec_j2000.dec).to_degrees() * 60.0;
        assert_abs_diff_eq!(ha_diff_arcmin, 9.344552279378359, epsilon = 1e-5);
        assert_abs_diff_eq!(dec_diff_arcmin, -0.12035370887056628, epsilon = 1e-5);
    }
}
