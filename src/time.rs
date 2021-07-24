// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Code to help handle time.

use hifitime::Epoch;

use crate::constants::*;

/// Get the GPS time from a `hifitime` [Epoch]. This function is mostly useful
/// because `hifitime` doesn't count GPS seconds from 1980 Jan 5.
pub(crate) fn epoch_as_gps_seconds(e: Epoch) -> f64 {
    e.as_gpst_seconds() - HIFITIME_GPS_FACTOR
}

/// Convert a Julian date to a `hifitime` [Epoch]. This function is especially
/// useful to account for leap seconds.
pub(crate) fn jd_to_epoch(jd_days: f64) -> hifitime::Epoch {
    // This isn't really the number of leap seconds; it's the number of leap
    // seconds divided by the number of seconds in a day. Perfect for adding to
    // a JD.
    let num_leap_seconds = {
        let naive_obs_epoch = Epoch::from_tai_days(jd_days);
        jd_days - naive_obs_epoch.as_utc_days()
    };
    Epoch::from_jde_tai(jd_days + num_leap_seconds)
}

/// Convert a modified Julian date to a `hifitime` [Epoch]. This function is
/// especially useful to account for leap seconds.
pub(crate) fn mjd_to_epoch(mjd_days: f64) -> hifitime::Epoch {
    // This isn't really the number of leap seconds; it's the number of leap
    // seconds divided by the number of seconds in a day. Perfect for adding to
    // an MJD.
    let num_leap_seconds = {
        let naive_obs_epoch = Epoch::from_tai_days(mjd_days);
        mjd_days - naive_obs_epoch.as_utc_days()
    };
    Epoch::from_mjd_tai(mjd_days + num_leap_seconds)
}

/// Convert a GPS time to a `hifitime` [Epoch].
pub(crate) fn gps_to_epoch(gps: f64) -> Epoch {
    // https://en.wikipedia.org/wiki/Global_Positioning_System#Timekeeping
    // The difference between GPS and TAI time is always 19s, but hifitime
    // wants the number of TAI seconds since 1900. GPS time starts at 1980
    // Jan 5.
    let tai = gps + 19.0 + crate::constants::HIFITIME_GPS_FACTOR;
    Epoch::from_tai_seconds(tai)
}

/// Convert a casacore time to a `hifitime` [Epoch]. This function is especially
/// useful because casacore apparently doesn't account for leap seconds.
///
/// casacore uses seconds since 1858-11-17T00:00:00 (MJD epoch).
pub(crate) fn casacore_utc_to_epoch(utc_seconds: f64) -> hifitime::Epoch {
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

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn hifitime_behaves_as_expected_gps() {
        let gps = 1065880128.0;
        let epoch = gps_to_epoch(gps);
        assert_abs_diff_eq!(epoch_as_gps_seconds(epoch), gps);
    }

    #[test]
    fn hifitime_behaves_as_expected_utc() {
        // This UTC time is taken from the 1065880128 observation.
        let utc = 4888561714.0;
        let epoch = casacore_utc_to_epoch(utc);
        assert_abs_diff_eq!(epoch.as_utc_seconds(), 3590833714.0, epsilon = 1e-10);
        assert_abs_diff_eq!(epoch_as_gps_seconds(epoch), 1065880130.0, epsilon = 1e-10);
    }
}
