// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Helper functions around time.

use hifitime::Epoch;

use mwa_hyperdrive_common::hifitime::{self, Duration};

/// Some timestamps may be read in ever so slightly off from their true values
/// because of float errors. This function checks if a supplied [Epoch], when
/// represented as GPS seconds, is really close to a neat value in the
/// hundredths. If so, the value is rounded and returned.
///
/// e.g. The GPS time 1090008639.999405 should be 1090008634.0. Other examples
/// of usage are in the tests alongside this function.
pub(crate) fn round_hundredths_of_a_second(e: Epoch) -> Epoch {
    let e_gps = e.as_gpst_seconds() * 100.0;
    if (e_gps.round() - e_gps).abs() < 0.1 {
        Epoch::from_gpst_seconds(e_gps.round() / 100.0)
    } else {
        e
    }
}

/// Some timestamps may be read in ever so slightly off from their true values
/// because of float errors. This function checks if a supplied [Epoch], when
/// represented as GPS seconds, is really close to a neat value in the
/// hundredths. If so, the value is rounded and returned.
///
/// e.g. The GPS time 1090008639.999405 should be 1090008634.0. Other examples
/// of usage are in the tests alongside this function.
pub(crate) fn round_hundredths_of_a_second_duration(e: Duration) -> Duration {
    let (centuries, nanos) = e.to_parts();
    let hundredth_ns = 10_000_000;
    Duration::from_parts(centuries, (nanos / hundredth_ns) * hundredth_ns)
}

#[cfg(test)]
mod tests {
    use approx::assert_abs_diff_eq;
    use mwa_hyperdrive_common::hifitime::Unit;

    use super::*;

    #[test]
    fn test_round_seconds() {
        let e = Epoch::from_gpst_seconds(1090008639.999405);
        assert_abs_diff_eq!(
            round_hundredths_of_a_second(e).as_gpst_seconds(),
            1090008640.0
        );

        let e = Epoch::from_gpst_seconds(1090008640.251);
        assert_abs_diff_eq!(
            round_hundredths_of_a_second(e).as_gpst_seconds(),
            1090008640.25
        );

        let e = Epoch::from_gpst_seconds(1090008640.24999);
        assert_abs_diff_eq!(
            round_hundredths_of_a_second(e).as_gpst_seconds(),
            1090008640.25
        );

        // No rounding.
        let e = Epoch::from_gpst_seconds(1090008640.26);
        assert_abs_diff_eq!(
            round_hundredths_of_a_second(e).as_gpst_seconds(),
            1090008640.26
        );
    }

    #[test]
    fn test_round_duration() {
        let half_day = Duration::from_f64(0.5, Unit::Day);
        let millis = Duration::from_f64(1., Unit::Millisecond);
        assert_eq!(
            half_day,
            round_hundredths_of_a_second_duration(half_day + millis)
        );
        assert_eq!(
            half_day,
            round_hundredths_of_a_second_duration(half_day + 4 * millis)
        );
        assert_eq!(
            half_day + 10 * millis,
            round_hundredths_of_a_second_duration(half_day + 11 * millis)
        );
    }
}
