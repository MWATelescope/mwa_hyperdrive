// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Code to help interface with CASA measurement sets.

use std::path::Path;

use hifitime::Epoch;
use rubbl_casatables::{Table, TableOpenMode};

use super::error::*;
use crate::constants::*;
use mwa_hyperdrive_core::{xyz, XyzGeocentric, XyzGeodetic};

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

/// Convert a casacore time to a `hifitime` [Epoch]. This function is especially
/// useful because casacore apparently doesn't account for leap seconds.
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

/// casacore's antenna positions are [XyzGeocentric] coordinates, but we use
/// [XyzGeodetic] coordinates in hyperdrive. This function converts the casacore
/// positions.
pub(super) fn casacore_positions_to_local_xyz(
    pos: &[XyzGeocentric],
    longitude_rad: f64,
    latitude_rad: f64,
    height_metres: f64,
) -> Result<Vec<XyzGeodetic>, MSError> {
    xyz::geocentric_to_geodetic(pos, longitude_rad, latitude_rad, height_metres)
        .map_err(|_| MSError::Geodetic2Geocentric)
}

/// casacore's antenna positions are [XyzGeocentric] coordinates, but we use
/// [XyzGeodetic] coordinates in hyperdrive. This function converts the casacore
/// positions, assuming we're at the MWA coordinates.
pub(super) fn casacore_positions_to_local_xyz_mwa(
    pos: &[XyzGeocentric],
) -> Result<Vec<XyzGeodetic>, MSError> {
    casacore_positions_to_local_xyz(pos, MWA_LONG_RAD, MWA_LAT_RAD, MWA_HEIGHT_M)
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
