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

pub(crate) fn get_lmst(time: Epoch, array_longitude_rad: f64) -> f64 {
    let gmst = pal::palGmst(time.as_mjd_utc_days());
    (gmst + array_longitude_rad) % TAU
}

// This function is very similar to cotter's `PrepareTimestepUVW`.
pub(crate) fn precess_time(
    phase_centre: RADec,
    time: Epoch,
    array_longitude_rad: f64,
    array_latitude_rad: f64,
) -> PrecessionInfo {
    let mjd = time.as_mjd_utc_days();

    // Note that we explicitly use the LMST because we're handling nutation
    // ourselves.
    let lmst = get_lmst(time, array_longitude_rad);

    let j2000 = 2000.0;
    let radec_aber = aber_radec_rad(j2000, mjd, phase_centre);
    let mut rotation_matrix = [[0.0; 3]; 3];
    pal::palPrenut(j2000, mjd, rotation_matrix.as_mut_ptr());

    // Transpose the rotation matrix.
    let mut rotation_matrix = {
        let mut new = [[0.0; 3]; 3];
        let old = rotation_matrix;
        for (i, old) in old.iter().enumerate() {
            for (j, new) in new.iter_mut().enumerate() {
                new[i] = old[j];
            }
        }
        new
    };

    let precessed = hadec_j2000(&mut rotation_matrix, lmst, array_latitude_rad, radec_aber);

    PrecessionInfo {
        rotation_matrix,
        hadec_j2000: precessed.hadec,
        lmst,
        lmst_j2000: precessed.lmst,
        array_latitude_j2000: precessed.latitude,
    }
}

// Blatently stolen from cotter.
fn aber_radec_rad(eq: f64, mjd: f64, radec: RADec) -> RADec {
    let mut v1 = [0.0; 3];
    let mut v2 = [0.0; 3];

    pal::palDcs2c(radec.ra, radec.dec, v1.as_mut_ptr());
    stelaber(eq, mjd, &mut v1, &mut v2);
    let mut ra2 = 0.0;
    let mut dec2 = 0.0;
    pal::palDcc2s(v2.as_mut_ptr(), &mut ra2, &mut dec2);
    ra2 = pal::palDranrm(ra2);

    RADec::new(ra2, dec2)
}

// Blatently stolen from cotter.
fn stelaber(eq: f64, mjd: f64, v1: &mut [f64; 3], v2: &mut [f64; 3]) {
    let mut amprms = [0.0; 21];
    let mut v1n = [0.0; 3];
    let mut v2un = [0.0; 3];
    let mut abv = [0.0; 3];

    pal::palMappa(eq, mjd, amprms.as_mut_ptr());

    /* Unpack scalar and vector parameters */
    let ab1 = &amprms[11];
    abv[0] = amprms[8];
    abv[1] = amprms[9];
    abv[2] = amprms[10];

    let mut w = 0.0;
    pal::palDvn(v1.as_mut_ptr(), v1n.as_mut_ptr(), &mut w);

    /* Aberration (normalization omitted) */
    let p1dv = pal::palDvdv(v1n.as_mut_ptr(), abv.as_mut_ptr());
    w = 1.0 + p1dv / (ab1 + 1.0);
    v2un[0] = ab1 * v1n[0] + w * abv[0];
    v2un[1] = ab1 * v1n[1] + w * abv[1];
    v2un[2] = ab1 * v1n[2] + w * abv[2];

    /* Normalize */
    pal::palDvn(v2un.as_mut_ptr(), v2.as_mut_ptr(), &mut w);
}

/// The return type of `hadec_j2000`. All values are in radians.
struct HADecJ2000 {
    hadec: HADec,
    latitude: f64,
    lmst: f64,
}

fn hadec_j2000(
    rotation_matrix: &mut [[f64; 3]; 3],
    lmst: f64,
    lat_rad: f64,
    radec: RADec,
) -> HADecJ2000 {
    let (new_lmst, new_lat) = rotate_radec(rotation_matrix, lmst, lat_rad);
    HADecJ2000 {
        hadec: HADec::new(pal::palDranrm(new_lmst - radec.ra), radec.dec),
        latitude: new_lat,
        lmst: new_lmst,
    }
}

// Blatently stolen from cotter.
fn rotate_radec(rotation_matrix: &mut [[f64; 3]; 3], ra: f64, dec: f64) -> (f64, f64) {
    let mut v1 = [0.0; 3];
    let mut v2 = [0.0; 3];

    pal::palDcs2c(ra, dec, v1.as_mut_ptr());
    pal::palDmxv(
        rotation_matrix.as_mut_ptr(),
        v1.as_mut_ptr(),
        v2.as_mut_ptr(),
    );
    let mut ra2 = 0.0;
    let mut dec2 = 0.0;
    pal::palDcc2s(v2.as_mut_ptr(), &mut ra2, &mut dec2);
    ra2 = pal::palDranrm(ra2);

    (ra2, dec2)
}

mod pal {
    #![allow(non_snake_case)]

    //! So that we don't require starlink PAL, write the equivalent functions
    //! here. PAL depends on ERFA and luckily all the needed PAL functions just
    //! use ERFA. Code is derived from https://github.com/Starlink/pal commit
    //! 7af65f0 and is licensed under the LGPL-3.0.
    //!
    //! Why not just depend on PAL? It's a huge pain. (1) The C library is
    //! either named libstarlink-pal or libpal, (2) it depends on ERFA so
    //! statically compiling it in a -sys crate is much harder than it should
    //! be, (3) this code changes so slowly that we're unlikely to be
    //! out-of-date.

    use mwa_hyperdrive_core::erfa_sys::*;

    // TODO: ERFA_AULT isn't showing up in erfa_sys for some reason.
    const ERFA_AULT: f64 = ERFA_DAU / ERFA_CMPS;

    pub(super) fn palGmst(ut1: f64) -> f64 {
        unsafe { eraGmst06(ERFA_DJM0, ut1, ERFA_DJM0, ut1) }
    }

    pub(super) fn palDcs2c(a: f64, b: f64, v: *mut f64) {
        unsafe { eraS2c(a, b, v) }
    }

    pub(super) fn palDcc2s(v: *mut f64, a: &mut f64, b: &mut f64) {
        unsafe { eraC2s(v, a, b) }
    }

    pub(super) fn palDranrm(angle: f64) -> f64 {
        unsafe { eraAnp(angle) }
    }

    pub(super) fn palDvn(v: *mut f64, uv: *mut f64, vm: *mut f64) {
        unsafe { eraPn(v, vm, uv) }
    }

    pub(super) fn palDvdv(va: *mut f64, vb: *mut f64) -> f64 {
        unsafe { eraPdp(va, vb) }
    }

    pub(super) fn palDmxv(dm: *mut [f64; 3], va: *mut f64, vb: *mut f64) {
        unsafe { eraRxp(dm, va, vb) }
    }

    pub(super) fn palEvp(
        date: f64,
        deqx: f64,
        dvb: *mut f64,
        dpb: *mut f64,
        dvh: *mut f64,
        dph: *mut f64,
    ) {
        unsafe {
            /* Local Variables; */
            let mut pvh = [[0.0; 3]; 2];
            let mut pvb = [[0.0; 3]; 2];
            let mut d1 = 0.0;
            let mut d2 = 0.0;
            let mut r = [[0.0; 3]; 3];

            /* BCRS PV-vectors. */
            eraEpv00(2400000.5, date, pvh.as_mut_ptr(), pvb.as_mut_ptr());

            /* Was precession to another equinox requested? */
            if deqx > 0.0 {
                /* Yes: compute precession matrix from J2000.0 to deqx. */
                eraEpj2jd(deqx, &mut d1, &mut d2);
                eraPmat06(d1, d2, r.as_mut_ptr());

                /* Rotate the PV-vectors. */
                eraRxpv(r.as_mut_ptr(), pvh.as_mut_ptr(), pvh.as_mut_ptr());
                eraRxpv(r.as_mut_ptr(), pvb.as_mut_ptr(), pvb.as_mut_ptr());
            }

            /* Return the required vectors. */
            let dvb = std::slice::from_raw_parts_mut(dvb, 3);
            let dpb = std::slice::from_raw_parts_mut(dpb, 3);
            let dvh = std::slice::from_raw_parts_mut(dvh, 3);
            let dph = std::slice::from_raw_parts_mut(dph, 3);
            for i in 0..3 {
                dvh[i] = pvh[1][i] / ERFA_DAYSEC;
                dvb[i] = pvb[1][i] / ERFA_DAYSEC;
                dph[i] = pvh[0][i];
                dpb[i] = pvb[0][i];
            }
        }
    }

    pub(super) fn palPrenut(epoch: f64, date: f64, rmatpn: *mut [f64; 3]) {
        unsafe {
            /* Local Variables: */
            let mut bpa = 0.0;
            let mut bpia = 0.0;
            let mut bqa = 0.0;
            let mut chia = 0.0;
            let mut d1 = 0.0;
            let mut d2 = 0.0;
            let mut eps0 = 0.0;
            let mut epsa = 0.0;
            let mut gam = 0.0;
            let mut oma = 0.0;
            let mut pa = 0.0;
            let mut phi = 0.0;
            let mut pia = 0.0;
            let mut psi = 0.0;
            let mut psia = 0.0;
            let mut r1 = [[0.0; 3]; 3];
            let mut r2 = [[0.0; 3]; 3];
            let mut thetaa = 0.0;
            let mut za = 0.0;
            let mut zetaa = 0.0;

            /* Specified Julian epoch as a 2-part JD. */
            eraEpj2jd(epoch, &mut d1, &mut d2);

            /* P matrix, from specified epoch to J2000.0. */
            eraP06e(
                d1,
                d2,
                &mut eps0,
                &mut psia,
                &mut oma,
                &mut bpa,
                &mut bqa,
                &mut pia,
                &mut bpia,
                &mut epsa,
                &mut chia,
                &mut za,
                &mut zetaa,
                &mut thetaa,
                &mut pa,
                &mut gam,
                &mut phi,
                &mut psi,
            );
            eraIr(r1.as_mut_ptr());
            eraRz(-chia, r1.as_mut_ptr());
            eraRx(oma, r1.as_mut_ptr());
            eraRz(psia, r1.as_mut_ptr());
            eraRx(-eps0, r1.as_mut_ptr());

            /* NPB matrix, from J2000.0 to date. */
            eraPnm06a(ERFA_DJM0, date, r2.as_mut_ptr());

            /* NPB matrix, from specified epoch to date. */
            eraRxr(r2.as_mut_ptr(), r1.as_mut_ptr(), rmatpn);
        }
    }

    pub(super) fn palMappa(eq: f64, date: f64, amprms: *mut f64) {
        unsafe {
            /* Local constants */

            /*  Gravitational radius of the Sun x 2 (2*mu/c**2, AU) */
            let gr2: f64 = 2.0 * 9.87063e-9;

            /* Local Variables; */
            let mut ebd = [0.0; 3];
            let mut ehd = [0.0; 3];
            let mut eh = [0.0; 3];
            let mut e = 0.0;
            let mut vn = [0.0; 3];
            let mut vm = 0.0;

            /* Initialise so that unsused values are returned holding zero */
            let amprms = std::slice::from_raw_parts_mut(amprms, 21);
            amprms.fill(0.0);

            /* Time interval for proper motion correction. */
            amprms[0] = eraEpj(ERFA_DJM0, date) - eq;

            /* Get Earth barycentric and heliocentric position and velocity. */
            palEvp(
                date,
                eq,
                ebd.as_mut_ptr(),
                &mut amprms[1],
                ehd.as_mut_ptr(),
                eh.as_mut_ptr(),
            );

            /* Heliocentric direction of Earth (normalized) and modulus. */
            eraPn(eh.as_mut_ptr(), &mut e, &mut amprms[4]);

            /* Light deflection parameter */
            amprms[7] = gr2 / e;

            /* Aberration parameters. */
            for i in 0..3 {
                amprms[i + 8] = ebd[i] * ERFA_AULT;
            }
            eraPn(&mut amprms[8], &mut vm, vn.as_mut_ptr());
            amprms[11] = (1.0 - vm * vm).sqrt();

            /* NPB matrix. */
            palPrenut(eq, date, amprms.as_mut_ptr().add(12) as _);
        }
    }
}

#[cfg(test)]
mod tests {
    use approx::{assert_abs_diff_eq, assert_abs_diff_ne};
    use std::str::FromStr;

    use super::*;
    use crate::time::gps_to_epoch;
    use mwa_hyperdrive_core::constants::{MWA_LAT_RAD, MWA_LONG_RAD};

    // astropy doesn't exactly agree with the numbers below, I think because the
    // LST listed in MWA metafits files doesn't agree with what astropy thinks
    // it should be. But, it's all very close.
    #[test]
    fn test_get_lst() {
        let epoch = gps_to_epoch(1090008642.0);
        assert_abs_diff_eq!(
            get_lmst(epoch, MWA_LONG_RAD),
            6.262087947389409,
            epsilon = 1e-10
        );

        let epoch = gps_to_epoch(1090008643.0);
        assert_abs_diff_eq!(
            get_lmst(epoch, MWA_LONG_RAD),
            6.2621608685650045,
            epsilon = 1e-10
        );

        let epoch = gps_to_epoch(1090008647.0);
        assert_abs_diff_eq!(
            get_lmst(epoch, MWA_LONG_RAD),
            6.262452553175729,
            epsilon = 1e-10
        );

        let epoch = gps_to_epoch(1090008644.0);
        assert_abs_diff_eq!(
            get_lmst(epoch, MWA_LONG_RAD),
            6.262233789694743,
            epsilon = 1e-10
        );
    }

    #[test]
    fn test_no_precession_at_j2000() {
        let j2000_epoch = Epoch::from_str("2000-01-01T11:58:55.816 UTC").unwrap();

        let phase_centre = RADec::new_degrees(0.0, -27.0);
        let eor0 = precess_time(phase_centre, j2000_epoch, MWA_LONG_RAD, MWA_LAT_RAD);
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
        assert_abs_diff_eq!(eor0.array_latitude_j2000, MWA_LAT_RAD, epsilon = 1e-4);
        assert_abs_diff_eq!(eor0.lmst, eor0.lmst_j2000, epsilon = 1e-4);

        assert_abs_diff_eq!(eor0.lmst, 0.6433259676052971, epsilon = 1e-4);

        let phase_centre = RADec::new_degrees(60.0, -30.0);
        let eor1 = precess_time(phase_centre, j2000_epoch, MWA_LONG_RAD, MWA_LAT_RAD);
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
        assert_abs_diff_eq!(eor1.array_latitude_j2000, MWA_LAT_RAD, epsilon = 1e-4);
        assert_abs_diff_eq!(eor1.lmst, eor1.lmst_j2000, epsilon = 1e-4);

        assert_abs_diff_eq!(eor1.lmst, 0.6433259676052971, epsilon = 1e-4);
    }

    #[test]
    fn test_precess_1065880128_to_j2000() {
        let epoch = gps_to_epoch(1065880128.0);
        let phase_centre = RADec::new_degrees(0.0, -27.0);

        let p = precess_time(phase_centre, epoch, MWA_LONG_RAD, MWA_LAT_RAD);
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
        assert_abs_diff_ne!(p.array_latitude_j2000, MWA_LAT_RAD, epsilon = 1e-5);

        let pc_hadec = phase_centre.to_hadec(p.lmst);
        let ha_diff_arcmin = (pc_hadec.ha - p.hadec_j2000.ha).to_degrees() * 60.0;
        let dec_diff_arcmin = (pc_hadec.dec - p.hadec_j2000.dec).to_degrees() * 60.0;
        assert_abs_diff_eq!(ha_diff_arcmin, 11.510918573880216, epsilon = 1e-5); // 11.5 arcmin!
        assert_abs_diff_eq!(dec_diff_arcmin, -0.05058613713495692, epsilon = 1e-5);
    }

    #[test]
    fn test_precess_1099334672_to_j2000() {
        let epoch = gps_to_epoch(1099334672.0);
        let phase_centre = RADec::new_degrees(60.0, -30.0);

        let p = precess_time(phase_centre, epoch, MWA_LONG_RAD, MWA_LAT_RAD);
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
        assert_abs_diff_ne!(p.array_latitude_j2000, MWA_LAT_RAD, epsilon = 1e-5);

        let pc_hadec = phase_centre.to_hadec(p.lmst);
        let ha_diff_arcmin = (pc_hadec.ha - p.hadec_j2000.ha).to_degrees() * 60.0;
        let dec_diff_arcmin = (pc_hadec.dec - p.hadec_j2000.dec).to_degrees() * 60.0;
        assert_abs_diff_eq!(ha_diff_arcmin, 9.344552279378359, epsilon = 1e-5);
        assert_abs_diff_eq!(dec_diff_arcmin, -0.12035370887056628, epsilon = 1e-5);
    }
}
