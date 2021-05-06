// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

/*!
Code to help interface with CASA measurement sets.
 */

use std::path::Path;

use hifitime::Epoch;
use ndarray::prelude::*;
use rubbl_casatables::{Table, TableOpenMode};

use super::error::*;
use crate::constants::*;
use mwa_hyperdrive_core::{erfa_sys, mwalib, XYZ};

/// Open a measurement set table read only. If `table` is `None`, then open the
/// base table.
pub(super) fn read_table(ms: &Path, table: Option<&str>) -> Result<Table, MSError> {
    match Table::open(
        &format!("{}/{}", ms.display(), table.unwrap_or("")),
        TableOpenMode::Read,
    ) {
        Ok(t) => Ok(t),
        Err(e) => Err(MSError::RubblError(e.to_string())),
    }
}

/// Convert a casacore time to a `hifitime` [Epoch].
///
/// casacore uses seconds since 1858-11-17T00:00:00 (MJD epoch).
pub(super) fn casacore_utc_to_epoch(utc_seconds: f64) -> hifitime::Epoch {
    // It appears that casacore does not count the number of leap seconds when
    // giving out the number of UTC seconds. This needs to be accounted for.
    // Because I don't have direct access to a table of leap seconds, and don't
    // want to constantly maintain one, I'm making a compromise; the method
    // below will be off by 1s if the supplied `utc_seconds` is near a leap
    // second.
    let num_leap_seconds = {
        let naive_obs_epoch = Epoch::from_tai_seconds(utc_seconds - MJD_TAI_EPOCH_DIFF);
        utc_seconds - MJD_TAI_EPOCH_DIFF - naive_obs_epoch.as_utc_seconds()
    };
    Epoch::from_tai_seconds(utc_seconds - MJD_TAI_EPOCH_DIFF + num_leap_seconds)
}

/// casacore's antenna positions are geodetic [XYZ] coordinates, but we use
/// geocentric [XYZ] coordinates in hyperdrive. This function converts the
/// casacore positions.
pub(super) fn casacore_positions_to_local_xyz(pos: ArrayView2<f64>) -> Result<Vec<XYZ>, MSError> {
    let mut mwa_xyz: [f64; 3] = [0.0; 3];
    let status = unsafe {
        erfa_sys::eraGd2gc(
            erfa_sys::ERFA_WGS84 as i32,   // ellipsoid identifier (Note 1)
            mwalib::MWA_LONGITUDE_RADIANS, // longitude (radians, east +ve)
            mwalib::MWA_LATITUDE_RADIANS,  // latitude (geodetic, radians, Note 3)
            mwalib::MWA_ALTITUDE_METRES,   // height above ellipsoid (geodetic, Notes 2,3)
            mwa_xyz.as_mut_ptr(),          // geocentric vector (Note 2)
        )
    };
    if status != 0 {
        return Err(MSError::Geodetic2Geocentric);
    }

    let xyz = pos
        .outer_iter()
        .map(|geodetic| {
            let geocentric = XYZ {
                x: geodetic[0] - mwa_xyz[0],
                y: geodetic[1] - mwa_xyz[1],
                z: geodetic[2] - mwa_xyz[2],
            };
            geocentric.rotate_mwa(-1)
        })
        .collect();
    Ok(xyz)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tests::*;

    #[test]
    fn hifitime_behaves_as_expected() {
        // This UTC time is taken from the 1065880128 observation.
        let utc = 4888561714.0;
        let epoch = casacore_utc_to_epoch(utc);
        assert_abs_diff_eq!(epoch.as_utc_seconds(), 3590833714.0, epsilon = 1e-10);
        assert_abs_diff_eq!(
            epoch.as_gpst_seconds() - HIFITIME_GPS_FACTOR,
            1065880130.0,
            epsilon = 1e-10
        );
    }
}
