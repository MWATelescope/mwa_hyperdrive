// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! The array geometry used for modelling at a timestep: tile positions, local
//! sidereal time and array latitude, in the frame selected by [`UvwFrame`]
//! (see `marlu::precession::precess_time_in_frame`).

use std::borrow::Cow;

use hifitime::{Duration, Epoch};
use log::debug;
use marlu::{
    precession::{get_lmst, precess_time_in_frame},
    RADec, UvwFrame, XyzGeodetic,
};

/// Where and when the tiles are, for one timestep.
pub(crate) struct Geometry<'a> {
    /// Local sidereal time [radians] to use with the phase centre.
    pub(crate) lst: f64,
    /// Tile positions consistent with `lst`.
    pub(crate) xyzs: Cow<'a, [XyzGeodetic]>,
    /// Array latitude consistent with `xyzs` [radians].
    pub(crate) latitude: f64,
}

/// Get the geometry to model with at `timestamp`. `precess` is ignored for
/// frames that never precess ([`UvwFrame::Oskar`]).
#[allow(clippy::too_many_arguments)]
pub(crate) fn geometry_at(
    frame: UvwFrame,
    precess: bool,
    array_longitude_rad: f64,
    array_latitude_rad: f64,
    phase_centre: RADec,
    timestamp: Epoch,
    dut1: Duration,
    tile_xyzs: &[XyzGeodetic],
) -> Geometry<'_> {
    if precess || !frame.precesses() {
        let precession_info = precess_time_in_frame(
            frame,
            array_longitude_rad,
            array_latitude_rad,
            phase_centre,
            timestamp,
            dut1,
        );
        debug!(
            "Modelling GPS timestamp {} in the {frame} frame, LST {}°, J2000 LST {}°",
            timestamp.to_gpst_seconds(),
            precession_info.lmst.to_degrees(),
            precession_info.lmst_j2000.to_degrees()
        );
        Geometry {
            lst: precession_info.lmst_j2000,
            xyzs: Cow::from(precession_info.precess_xyz(tile_xyzs)),
            latitude: precession_info.array_latitude_j2000,
        }
    } else {
        let lst = get_lmst(array_longitude_rad, timestamp, dut1);
        debug!(
            "Modelling GPS timestamp {}, LMST {}°",
            timestamp.to_gpst_seconds(),
            lst.to_degrees()
        );
        Geometry {
            lst,
            xyzs: Cow::from(tile_xyzs),
            latitude: array_latitude_rad,
        }
    }
}

#[cfg(test)]
mod tests {
    use approx::assert_abs_diff_eq;

    use super::*;

    #[test]
    fn frames_differ_only_when_precessing() {
        let xyzs = [XyzGeodetic {
            x: 1000.0,
            y: 2000.0,
            z: 3000.0,
        }];
        let pc = RADec::from_radians(0.0, -0.5);
        let t = Epoch::from_gpst_seconds(1090008640.0);
        let a = geometry_at(
            UvwFrame::Casacore,
            false,
            2.0,
            -0.46,
            pc,
            t,
            Duration::default(),
            &xyzs,
        );
        let b = geometry_at(
            UvwFrame::Hyperdrive,
            false,
            2.0,
            -0.46,
            pc,
            t,
            Duration::default(),
            &xyzs,
        );
        assert_abs_diff_eq!(a.lst, b.lst);
        assert_abs_diff_eq!(a.xyzs[0].x, b.xyzs[0].x);

        let c = geometry_at(
            UvwFrame::Casacore,
            true,
            2.0,
            -0.46,
            pc,
            t,
            Duration::default(),
            &xyzs,
        );
        let h = geometry_at(
            UvwFrame::Hyperdrive,
            true,
            2.0,
            -0.46,
            pc,
            t,
            Duration::default(),
            &xyzs,
        );
        // Opposite baseline sign, and ~20 arcsec of frame rotation.
        assert!((c.xyzs[0].x + h.xyzs[0].x).abs() < 1.0);
        assert!((c.xyzs[0].x + h.xyzs[0].x).abs() > 1e-6);

        // The OSKAR frame never precesses, whatever `precess` says.
        let o1 = geometry_at(
            UvwFrame::Oskar,
            true,
            2.0,
            -0.46,
            pc,
            t,
            Duration::default(),
            &xyzs,
        );
        let o2 = geometry_at(
            UvwFrame::Oskar,
            false,
            2.0,
            -0.46,
            pc,
            t,
            Duration::default(),
            &xyzs,
        );
        assert_abs_diff_eq!(o1.lst, o2.lst);
        assert_abs_diff_eq!(o1.xyzs[0].z, 3000.0);
        // ... but uses the apparent rather than mean sidereal time.
        assert!((o1.lst - b.lst).abs() > 1e-7 && (o1.lst - b.lst).abs() < 1e-3);
    }
}
