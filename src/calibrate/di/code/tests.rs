// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Direction-independent calibration tests.

use approx::{assert_abs_diff_eq, assert_abs_diff_ne};
use hifitime::Epoch;
use marlu::Jones;
use ndarray::prelude::*;
use vec1::vec1;

use super::{calibrate, calibrate_timeblocks, CalVis, IncompleteSolutions};
use crate::{
    calibrate::{params::CalibrateParams, Chanblock, Timeblock},
    jones_test::TestJones,
    math::is_prime,
};
use mwa_hyperdrive_common::{hifitime, marlu, ndarray, vec1};

/// Make some data "four times as bright as the model". The solutions should
/// then be all "twos". As data and model visibilities are given per baseline
/// and solutions are given per tile, the per tile values should be the sqrt of
/// the multiplicative factor used.
#[test]
fn test_calibrate_trivial() {
    let num_timesteps = 1;
    let num_timeblocks = 1;
    let timeblock_length = 1;
    let num_tiles = 5;
    let num_baselines = num_tiles * (num_tiles - 1) / 2;
    let num_chanblocks = 1;

    let vis_shape = (num_timesteps, num_baselines, num_chanblocks);
    let vis_data: Array3<Jones<f32>> = Array3::from_elem(vis_shape, Jones::identity() * 4.0);
    let vis_model: Array3<Jones<f32>> = Array3::from_elem(vis_shape, Jones::identity());
    let mut di_jones = Array3::from_elem(
        (num_timeblocks, num_tiles, num_chanblocks),
        Jones::<f64>::identity(),
    );

    for timeblock in 0..num_timeblocks {
        let time_range_start = timeblock * timeblock_length;
        let time_range_end = ((timeblock + 1) * timeblock_length).min(vis_data.dim().0);

        let mut di_jones_rev = di_jones.slice_mut(s![timeblock, .., ..]).reversed_axes();

        for (chanblock_index, mut di_jones_rev) in (0..num_chanblocks)
            .into_iter()
            .zip(di_jones_rev.outer_iter_mut())
        {
            let range = s![
                time_range_start..time_range_end,
                ..,
                chanblock_index..chanblock_index + 1
            ];
            let vis_data_slice = vis_data.slice(range);
            let vis_model_slice = vis_model.slice(range);
            let result = calibrate(
                vis_data_slice,
                vis_model_slice,
                di_jones_rev.view_mut(),
                20,
                1e-8,
                1e-5,
            );

            assert!(result.converged);
            assert_eq!(result.num_iterations, 10);
            assert_eq!(result.num_failed, 0);
            assert!(result.max_precision < 1e-13);
            // The solutions should be 2 * identity.
            let expected = Array1::from_elem(di_jones_rev.len(), Jones::identity() * 2.0);

            let di_jones_rev = di_jones_rev.mapv(TestJones::from);
            let expected = expected.mapv(TestJones::from);
            assert_abs_diff_eq!(di_jones_rev, expected, epsilon = 1e-14);
        }
    }

    let di_jones = di_jones.mapv(TestJones::from);
    let expected = Array3::from_elem(di_jones.dim(), Jones::identity() * 2.0).mapv(TestJones::from);
    assert_abs_diff_eq!(di_jones, expected, epsilon = 1e-14);
}

/// As above, but make one Jones matrix much "bigger" than the rest. This should
/// make the calibration solutions not match what we expected, but when it's
/// flagged via the weights, things go back to normal.
#[test]
fn test_calibrate_trivial_with_flags() {
    let num_timesteps = 1;
    let num_timeblocks = 1;
    let timeblock_length = 1;
    let num_tiles = 5;
    let num_baselines = num_tiles * (num_tiles - 1) / 2;
    let num_chanblocks = 2;

    let vis_shape = (num_timesteps, num_baselines, num_chanblocks);
    let mut vis_data: Array3<Jones<f32>> = Array3::from_elem(vis_shape, Jones::identity());
    // Make the first chanblock's data be 4x identity.
    vis_data
        .slice_mut(s![.., .., 0])
        .fill(Jones::identity() * 4.0);
    // Make the second chanblock's data be 9x identity.
    vis_data
        .slice_mut(s![.., .., 1])
        .fill(Jones::identity() * 9.0);
    // Inject some wicked RFI.
    let bad_vis = vis_data.get_mut((0, 0, 0)).unwrap();
    *bad_vis = Jones::identity() * 9000.0;
    let mut vis_weights: Array3<f32> = Array3::ones(vis_shape);
    let mut vis_model: Array3<Jones<f32>> = Array3::from_elem(vis_shape, Jones::identity());
    let mut di_jones = Array3::from_elem(
        (num_timeblocks, num_tiles, num_chanblocks),
        Jones::<f64>::identity(),
    );

    for timeblock in 0..num_timeblocks {
        let time_range_start = timeblock * timeblock_length;
        let time_range_end = ((timeblock + 1) * timeblock_length).min(vis_data.dim().0);

        let mut di_jones_rev = di_jones.slice_mut(s![timeblock, .., ..]).reversed_axes();

        for (chanblock_index, mut di_jones_rev) in (0..num_chanblocks)
            .into_iter()
            .zip(di_jones_rev.outer_iter_mut())
        {
            let range = s![
                time_range_start..time_range_end,
                ..,
                chanblock_index..chanblock_index + 1
            ];
            let vis_data_slice = vis_data.slice(range);
            let vis_model_slice = vis_model.slice(range);
            let result = calibrate(
                vis_data_slice,
                vis_model_slice,
                di_jones_rev.view_mut(),
                20,
                1e-8,
                1e-5,
            );

            assert!(result.converged);
            assert_eq!(result.num_failed, 0);
            let expected = if chanblock_index == 0 {
                // The solutions should be 2 * identity, but they won't be.
                Array1::from_elem(di_jones_rev.len(), Jones::identity() * 2.0)
            } else {
                // The solutions should be 3 * identity.
                Array1::from_elem(di_jones_rev.len(), Jones::identity() * 3.0)
            };
            let di_jones_rev = di_jones_rev.mapv(TestJones::from);
            let expected = expected.mapv(TestJones::from);
            if timeblock == 0 && chanblock_index == 0 {
                assert_abs_diff_ne!(di_jones_rev, expected);
            } else {
                assert_abs_diff_eq!(di_jones_rev, expected);
            }
        }
    }

    // Fix the weight and repeat. We have to set the corresponding visibilities
    // to 0 (this is normally done before the visibilities are returned via
    // `CalVis`).
    let bad_weight = vis_weights.get_mut((0, 0, 0)).unwrap();
    *bad_weight = -1.0;
    let bad_data = vis_data.get_mut((0, 0, 0)).unwrap();
    *bad_data = Jones::default();
    let bad_model = vis_model.get_mut((0, 0, 0)).unwrap();
    *bad_model = Jones::default();
    di_jones.fill(Jones::identity());
    for timeblock in 0..num_timeblocks {
        let time_range_start = timeblock * timeblock_length;
        let time_range_end = ((timeblock + 1) * timeblock_length).min(vis_data.dim().0);

        let mut di_jones_rev = di_jones.slice_mut(s![timeblock, .., ..]).reversed_axes();

        for (chanblock_index, mut di_jones_rev) in (0..num_chanblocks)
            .into_iter()
            .zip(di_jones_rev.outer_iter_mut())
        {
            let range = s![
                time_range_start..time_range_end,
                ..,
                chanblock_index..chanblock_index + 1
            ];
            let vis_data_slice = vis_data.slice(range);
            let vis_model_slice = vis_model.slice(range);
            let result = calibrate(
                vis_data_slice,
                vis_model_slice,
                di_jones_rev.view_mut(),
                20,
                1e-8,
                1e-5,
            );

            assert!(result.converged);
            if chanblock_index == 0 {
                assert_eq!(result.num_iterations, 10);
            } else {
                assert_eq!(result.num_iterations, 12);
            }
            assert_eq!(result.num_failed, 0);
            assert!(result.max_precision < 1e-13);
            let expected = if chanblock_index == 0 {
                // The solutions should be 2 * identity.
                Array1::from_elem(di_jones_rev.len(), Jones::identity() * 2.0)
            } else {
                // The solutions should be 3 * identity.
                Array1::from_elem(di_jones_rev.len(), Jones::identity() * 3.0)
            };

            let di_jones_rev = di_jones_rev.mapv(TestJones::from);
            let expected = expected.mapv(TestJones::from);
            assert_abs_diff_eq!(di_jones_rev, expected, epsilon = 1e-14);
        }
    }

    let di_jones = di_jones.mapv(TestJones::from);
    let mut expected = Array3::from_elem(di_jones.dim(), Jones::identity());
    expected
        .slice_mut(s![.., .., 0])
        .fill(Jones::identity() * 2.0);
    // Make the second chanblock's data be 9x identity.
    expected
        .slice_mut(s![.., .., 1])
        .fill(Jones::identity() * 3.0);
    assert_abs_diff_eq!(di_jones, expected.mapv(TestJones::from), epsilon = 1e-14);
}

/// Test that converting [IncompleteSolutions] to [CalibrationSolutions] does
/// what's expected.
#[test]
fn incomplete_to_complete_trivial() {
    let timeblocks = [Timeblock {
        index: 0,
        range: 0..1,
        start: Epoch::from_gpst_seconds(1065880128.0),
        end: Epoch::from_gpst_seconds(1065880130.0),
        average: Epoch::from_gpst_seconds(1065880129.0),
    }];
    let chanblocks = [
        Chanblock {
            chanblock_index: 0,
            unflagged_index: 0,
            _freq: 150e6,
        },
        Chanblock {
            chanblock_index: 1,
            unflagged_index: 1,
            _freq: 151e6,
        },
        Chanblock {
            chanblock_index: 2,
            unflagged_index: 2,
            _freq: 152e6,
        },
    ];
    let flagged_tiles = [];
    let flagged_chanblock_indices = [];
    let num_timeblocks = timeblocks.len();
    let num_tiles = 5;
    let num_baselines = num_tiles * (num_tiles - 1) / 2;
    let num_chanblocks = chanblocks.len();
    let baseline_weights = vec![1.0; num_baselines];

    let incomplete_di_jones: Vec<Jones<f64>> = (0..num_tiles * num_chanblocks)
        .into_iter()
        .map(|i| Jones::identity() * (i + 1) as f64 * if is_prime(i) { 1.0 } else { 0.5 })
        .collect();
    let incomplete_di_jones = Array3::from_shape_vec(
        (num_timeblocks, num_tiles, num_chanblocks),
        incomplete_di_jones,
    )
    .unwrap();
    let incomplete = IncompleteSolutions {
        di_jones: incomplete_di_jones.clone(),
        timeblocks: &timeblocks,
        chanblocks: &chanblocks,
        _baseline_weights: &baseline_weights,
        _max_iterations: 50,
        _stop_threshold: 1e-8,
        _min_threshold: 1e-4,
    };

    let complete = incomplete.into_cal_sols(
        num_tiles + flagged_tiles.len(),
        &flagged_tiles,
        &flagged_chanblock_indices,
        Some(1065880128),
    );

    // The "complete" solutions should have inverted Jones matrices.
    let expected = incomplete_di_jones.mapv(|v| v.inv());

    assert_abs_diff_eq!(
        complete.di_jones.mapv(TestJones::from),
        expected.mapv(TestJones::from)
    );

    assert!(complete.flagged_tiles.is_empty());
    assert!(complete.flagged_chanblocks.is_empty());
}

// Make the first chanblock flagged. Everything should then just be "shifted one
// over".
#[test]
fn incomplete_to_complete_flags_simple() {
    let timeblocks = [Timeblock {
        index: 0,
        range: 0..1,
        start: Epoch::from_gpst_seconds(1065880128.0),
        end: Epoch::from_gpst_seconds(1065880130.0),
        average: Epoch::from_gpst_seconds(1065880129.0),
    }];
    let chanblocks = [
        Chanblock {
            chanblock_index: 1,
            unflagged_index: 0,
            _freq: 151e6,
        },
        Chanblock {
            chanblock_index: 2,
            unflagged_index: 1,
            _freq: 152e6,
        },
        Chanblock {
            chanblock_index: 3,
            unflagged_index: 2,
            _freq: 153e6,
        },
    ];
    let flagged_tiles = [];
    let flagged_chanblock_indices = [0];
    let num_timeblocks = timeblocks.len();
    let num_tiles = 5;
    let num_baselines = num_tiles * (num_tiles - 1) / 2;
    let num_chanblocks = chanblocks.len();
    let baseline_weights = vec![1.0; num_baselines];
    let total_num_tiles = num_tiles + flagged_tiles.len();

    let di_jones: Vec<Jones<f64>> = (0..num_tiles * num_chanblocks)
        .into_iter()
        .map(|i| Jones::identity() * (i + 1) as f64 * if is_prime(i) { 1.0 } else { 0.5 })
        .collect();
    let incomplete_di_jones =
        Array3::from_shape_vec((num_timeblocks, num_tiles, num_chanblocks), di_jones).unwrap();
    let incomplete = IncompleteSolutions {
        di_jones: incomplete_di_jones.clone(),
        timeblocks: &timeblocks,
        chanblocks: &chanblocks,
        _baseline_weights: &baseline_weights,
        _max_iterations: 50,
        _stop_threshold: 1e-8,
        _min_threshold: 1e-4,
    };

    let complete = incomplete.into_cal_sols(
        total_num_tiles,
        &flagged_tiles,
        &flagged_chanblock_indices,
        Some(1065880128),
    );

    // The first chanblock is all flagged.
    for j in complete.di_jones.slice(s![.., .., 0]).iter() {
        assert!(j.any_nan());
    }
    // All others are not.
    for j in complete.di_jones.slice(s![.., .., 1..]).iter() {
        assert!(!j.any_nan());
    }
    assert_eq!(
        complete.di_jones.slice(s![.., .., 1..]).dim(),
        incomplete_di_jones.dim()
    );
    assert_abs_diff_eq!(
        complete
            .di_jones
            .slice(s![.., .., 1..])
            .mapv(TestJones::from),
        incomplete_di_jones.mapv(|v| TestJones::from(v.inv()))
    );

    assert!(complete.flagged_tiles.is_empty());
    assert_eq!(complete.flagged_chanblocks.len(), 1);
    assert!(complete.flagged_chanblocks.contains(&0));
}

// Same as above, but make the last chanblock flagged.
#[test]
fn incomplete_to_complete_flags_simple2() {
    let timeblocks = [Timeblock {
        index: 0,
        range: 0..1,
        start: Epoch::from_gpst_seconds(1065880128.0),
        end: Epoch::from_gpst_seconds(1065880130.0),
        average: Epoch::from_gpst_seconds(1065880129.0),
    }];
    let chanblocks = [
        Chanblock {
            chanblock_index: 0,
            unflagged_index: 0,
            _freq: 151e6,
        },
        Chanblock {
            chanblock_index: 1,
            unflagged_index: 1,
            _freq: 152e6,
        },
        Chanblock {
            chanblock_index: 2,
            unflagged_index: 2,
            _freq: 153e6,
        },
    ];
    let flagged_tiles = [];
    let flagged_chanblock_indices = [3];
    let num_timeblocks = timeblocks.len();
    let num_tiles = 5;
    let num_baselines = num_tiles * (num_tiles - 1) / 2;
    let num_chanblocks = chanblocks.len();
    let baseline_weights = vec![1.0; num_baselines];
    let total_num_tiles = num_tiles + flagged_tiles.len();

    let incomplete_di_jones: Vec<Jones<f64>> = (0..num_tiles * num_chanblocks)
        .into_iter()
        .map(|i| Jones::identity() * (i + 1) as f64 * if is_prime(i) { 1.0 } else { 0.5 })
        .collect();
    let incomplete_di_jones = Array3::from_shape_vec(
        (num_timeblocks, num_tiles, num_chanblocks),
        incomplete_di_jones,
    )
    .unwrap();
    let incomplete = IncompleteSolutions {
        di_jones: incomplete_di_jones.clone(),
        timeblocks: &timeblocks,
        chanblocks: &chanblocks,
        _baseline_weights: &baseline_weights,
        _max_iterations: 50,
        _stop_threshold: 1e-8,
        _min_threshold: 1e-4,
    };

    let complete = incomplete.into_cal_sols(
        total_num_tiles,
        &flagged_tiles,
        &flagged_chanblock_indices,
        Some(1065880128),
    );

    // The last chanblock is all flagged.
    for j in complete.di_jones.slice(s![.., .., -1]).iter() {
        assert!(j.any_nan());
    }
    // All others are not.
    for j in complete.di_jones.slice(s![.., .., ..-1]).iter() {
        assert!(!j.any_nan());
    }
    assert_eq!(
        complete.di_jones.slice(s![.., .., ..-1]).dim(),
        incomplete_di_jones.dim()
    );
    assert_abs_diff_eq!(
        complete
            .di_jones
            .slice(s![.., .., ..-1])
            .mapv(TestJones::from),
        incomplete_di_jones.mapv(|v| TestJones::from(v.inv()))
    );

    assert!(complete.flagged_tiles.is_empty());
    assert_eq!(complete.flagged_chanblocks.len(), 1);
    assert!(complete.flagged_chanblocks.contains(&3));
}

#[test]
fn incomplete_to_complete_flags_complex() {
    let timeblocks = [Timeblock {
        index: 0,
        range: 0..1,
        start: Epoch::from_gpst_seconds(1065880128.0),
        end: Epoch::from_gpst_seconds(1065880130.0),
        average: Epoch::from_gpst_seconds(1065880129.0),
    }];
    let chanblocks = [
        Chanblock {
            chanblock_index: 0,
            unflagged_index: 0,
            _freq: 150e6,
        },
        Chanblock {
            chanblock_index: 2,
            unflagged_index: 1,
            _freq: 152e6,
        },
        Chanblock {
            chanblock_index: 3,
            unflagged_index: 2,
            _freq: 153e6,
        },
    ];
    let flagged_tiles = [2];
    let flagged_chanblock_indices = [1];
    let num_timeblocks = timeblocks.len();
    let num_tiles = 5;
    let num_baselines = num_tiles * (num_tiles - 1) / 2;
    let num_chanblocks = chanblocks.len();
    let baseline_weights = vec![1.0; num_baselines];
    let total_num_tiles = num_tiles + flagged_tiles.len();
    let total_num_chanblocks = num_chanblocks + flagged_chanblock_indices.len();

    // Cower at my evil, awful code.
    let mut primes = vec1![2];
    while primes.len() < num_tiles * num_chanblocks {
        let next = (*primes.last() + 1..)
            .into_iter()
            .find(|&i| is_prime(i))
            .unwrap();
        primes.push(next);
    }
    let incomplete_di_jones = Array3::from_shape_vec(
        (num_timeblocks, num_tiles, num_chanblocks),
        primes
            .iter()
            .map(|&i| Jones::identity() * i as f64)
            .collect(),
    )
    .unwrap();
    let incomplete = IncompleteSolutions {
        di_jones: incomplete_di_jones,
        timeblocks: &timeblocks,
        chanblocks: &chanblocks,
        _baseline_weights: &baseline_weights,
        _max_iterations: 50,
        _stop_threshold: 1e-8,
        _min_threshold: 1e-4,
    };

    let complete = incomplete.into_cal_sols(
        total_num_tiles,
        &flagged_tiles,
        &flagged_chanblock_indices,
        Some(1065880128),
    );

    // For programmer sanity, enforce here that this test only ever has one
    // timeblock.
    assert_eq!(complete.di_jones.dim().0, 1);

    let mut i_unflagged_tile = 0;
    for i_tile in 0..total_num_tiles {
        let sub_array = complete.di_jones.slice(s![0, i_tile, ..]);
        let mut i_unflagged_chanblock = 0;

        if flagged_tiles.contains(&i_tile) {
            assert!(sub_array.iter().all(|j| j.any_nan()));
        } else {
            for i_chan in 0..total_num_chanblocks {
                if flagged_chanblock_indices.contains(&(i_chan as u16)) {
                    assert!(sub_array[i_chan].any_nan());
                } else {
                    assert_abs_diff_eq!(
                        TestJones::from(sub_array[i_chan]),
                        TestJones::from(
                            Jones::identity()
                                / primes[i_unflagged_tile * num_chanblocks + i_unflagged_chanblock]
                                    as f64
                        )
                    );

                    i_unflagged_chanblock += 1;
                }
            }

            i_unflagged_tile += 1;
        }
    }

    assert_eq!(complete.flagged_tiles.len(), 1);
    assert!(complete.flagged_tiles.contains(&2));
    assert_eq!(complete.flagged_chanblocks.len(), 1);
    assert!(complete.flagged_chanblocks.contains(&1));
}

/// Given calibration parameters and visibilities, this function tests that
/// everything matches an expected quality. The values may change over time but
/// they should be consistent with whatever tests use this test code.
pub(crate) fn test_1090008640_quality(params: CalibrateParams, cal_vis: CalVis) {
    let (_, cal_results) = calibrate_timeblocks(
        cal_vis.vis_data.view(),
        cal_vis.vis_model.view(),
        &params.timeblocks,
        &params.fences.first().chanblocks,
        &params.baseline_weights,
        50,
        1e-8,
        1e-4,
        false,
        false,
    );

    // Only one timeblock.
    assert_eq!(cal_results.dim().0, 1);

    let mut count_50 = 0;
    let mut count_42 = 0;
    let mut chanblocks_42 = vec![];
    let mut fewest_iterations = usize::MAX;
    for cal_result in cal_results {
        match cal_result.num_iterations {
            50 => {
                count_50 += 1;
                fewest_iterations = fewest_iterations.min(cal_result.num_iterations);
            }
            42 => {
                count_42 += 1;
                chanblocks_42.push(cal_result.chanblock.unwrap());
                fewest_iterations = fewest_iterations.min(cal_result.num_iterations);
            }
            0 => panic!("0 iterations? Something is wrong."),
            _ => {
                if cal_result.num_iterations % 2 == 1 {
                    panic!("An odd number of iterations shouldn't be possible; at the time of writing, only even numbers are allowed.");
                }
                fewest_iterations = fewest_iterations.min(cal_result.num_iterations);
            }
        }

        assert!(
            cal_result.converged,
            "Chanblock {} did not converge",
            cal_result.chanblock.unwrap()
        );
        assert_eq!(cal_result.num_failed, 0);
        assert!(cal_result.max_precision < 1e8);
    }

    let expected_count_50 = 14;
    let expected_count_42 = 1;
    let expected_chanblocks_42 = vec![13];
    let expected_fewest_iterations = 40;
    if count_50 != expected_count_50
        || count_42 != expected_count_42
        || chanblocks_42 != expected_chanblocks_42
        || fewest_iterations != expected_fewest_iterations
    {
        panic!(
            r#"
Calibration quality has changed. This test expects:
  {} chanblocks with 50 iterations (got {}),
  {} chanblocks with 42 iterations (got {}),
  chanblocks {:?} to need 42 iterations (got {:?}), and
  no chanblocks to finish in less than {} iterations (got {}).
"#,
            expected_count_50,
            count_50,
            expected_count_42,
            count_42,
            expected_chanblocks_42,
            chanblocks_42,
            fewest_iterations,
            expected_fewest_iterations
        );
    }
}
