// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

use approx::assert_abs_diff_eq;
use marlu::{c64, time::gps_to_epoch, Jones};
use ndarray::prelude::*;

use super::*;
use crate::jones_test::TestJones;
use mwa_hyperdrive_common::{marlu, ndarray};

fn make_solutions() -> CalibrationSolutions {
    let num_timeblocks = 2;
    let num_tiles = 128;
    let num_chanblocks = 768;
    let di_jones = Array1::range(
        1.0,
        (num_timeblocks * num_tiles * num_chanblocks + 1) as f64,
        1.0,
    );
    let mut di_jones = di_jones
        .into_shape((num_timeblocks, num_tiles, num_chanblocks))
        .unwrap()
        .mapv(|v| {
            Jones::from([
                c64::new(1.0, 2.0),
                c64::new(3.0, 4.0),
                c64::new(5.0, 6.0),
                c64::new(7.0, 8.0),
            ]) * v
        });
    // Sprinkle some flags.
    let flagged_tiles = 3..5;
    let flagged_chanblocks = 5..8;
    di_jones
        .slice_mut(s![.., flagged_tiles.clone(), ..])
        .fill(Jones::nan());
    di_jones
        .slice_mut(s![.., .., flagged_chanblocks.clone()])
        .fill(Jones::nan());

    CalibrationSolutions {
        di_jones,
        flagged_tiles: flagged_tiles.into_iter().collect(),
        flagged_chanblocks: flagged_chanblocks.into_iter().collect(),
        average_timestamps: vec![gps_to_epoch(1065880134.0), gps_to_epoch(1065880146.0)],
        start_timestamps: vec![gps_to_epoch(1065880128.0), gps_to_epoch(1065880140.0)],
        end_timestamps: vec![gps_to_epoch(1065880140.0), gps_to_epoch(1065880152.0)],
        obsid: Some(1065880128),
    }
}

#[test]
fn test_write_and_read_ao_solutions() {
    let sols = make_solutions();
    let tmp_file = tempfile::NamedTempFile::new().expect("Couldn't make tmp file");
    let result = ao::write(&sols, tmp_file.path());
    assert!(result.is_ok());
    result.unwrap();

    let result = ao::read(tmp_file.path());
    assert!(result.is_ok());
    let sols_from_disk = result.unwrap();

    assert_eq!(sols.di_jones.dim(), sols_from_disk.di_jones.dim());
    // Can't use assert_abs_diff_eq on the whole array, because it rejects NaN
    // equality.
    sols.di_jones
        .mapv(TestJones::from)
        .into_iter()
        .zip(sols_from_disk.di_jones.mapv(TestJones::from).into_iter())
        .for_each(|(expected, result)| {
            if expected.any_nan() {
                assert!(result.any_nan());
            } else {
                assert_abs_diff_eq!(expected, result);
            }
        });

    // TODO: Test the timestamps.
}

#[test]
fn test_write_and_read_hyperdrive_solutions() {
    let sols = make_solutions();
    let tmp_file = tempfile::NamedTempFile::new().expect("Couldn't make tmp file");
    let result = hyperdrive::write(&sols, tmp_file.path());
    assert!(result.is_ok());
    result.unwrap();

    let result = hyperdrive::read(tmp_file.path());
    assert!(result.is_ok());
    let sols_from_disk = result.unwrap();

    assert_eq!(sols.di_jones.dim(), sols_from_disk.di_jones.dim());
    // Can't use assert_abs_diff_eq on the whole array, because it rejects NaN
    // equality.
    sols.di_jones
        .mapv(TestJones::from)
        .into_iter()
        .zip(sols_from_disk.di_jones.mapv(TestJones::from).into_iter())
        .for_each(|(expected, result)| {
            if expected.any_nan() {
                assert!(result.any_nan());
            } else {
                assert_abs_diff_eq!(expected, result);
            }
        });

    // TODO: Test the timestamps.
}
