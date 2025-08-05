// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Direction-independent calibration tests.

use std::{
    collections::{HashMap, HashSet},
    num::NonZeroUsize,
    path::PathBuf,
};

use approx::{assert_abs_diff_eq, assert_abs_diff_ne};
use hifitime::{Duration, Epoch};
use indicatif::{ProgressBar, ProgressDrawTarget};
use marlu::Jones;
use ndarray::prelude::*;
use vec1::{vec1, Vec1};

use super::{calibrate, calibrate_timeblocks, DiCalParams, IncompleteSolutions};
use crate::{
    averaging::{channels_to_chanblocks, timesteps_to_timeblocks, Chanblock, Spw, Timeblock},
    beam::NoBeam,
    context::Polarisations,
    di_calibrate::calibrate_timeblock,
    io::read::{RawDataCorrections, RawDataReader},
    math::{is_prime, TileBaselineFlags},
    params::{InputVisParams, ModellingParams},
    solutions::CalSolutionType,
    srclist::SourceList,
};

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

    let vis_shape = (num_timesteps, num_chanblocks, num_baselines);
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

        for (chanblock_index, mut di_jones_rev) in
            (0..num_chanblocks).zip(di_jones_rev.outer_iter_mut())
        {
            let range = s![
                time_range_start..time_range_end,
                chanblock_index..chanblock_index + 1,
                ..
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
                Polarisations::default(),
            );

            assert!(result.converged);
            assert_eq!(result.num_iterations, 10);
            assert_eq!(result.num_failed, 0);
            assert!(result.max_precision < 1e-13);
            // The solutions should be 2 * identity.
            let expected = Array1::from_elem(di_jones_rev.len(), Jones::identity() * 2.0);

            assert_abs_diff_eq!(di_jones_rev, expected, epsilon = 1e-14);
        }
    }

    let expected = Array3::from_elem(di_jones.dim(), Jones::identity() * 2.0);
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

    let vis_shape = (num_timesteps, num_chanblocks, num_baselines);
    let mut vis_data: Array3<Jones<f32>> = Array3::from_elem(vis_shape, Jones::identity());
    // Make the first chanblock's data be 4x identity.
    vis_data
        .slice_mut(s![.., 0, ..])
        .fill(Jones::identity() * 4.0);
    // Make the second chanblock's data be 9x identity.
    vis_data
        .slice_mut(s![.., 1, ..])
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

        for (chanblock_index, mut di_jones_rev) in
            (0..num_chanblocks).zip(di_jones_rev.outer_iter_mut())
        {
            let range = s![
                time_range_start..time_range_end,
                chanblock_index..chanblock_index + 1,
                ..
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
                Polarisations::default(),
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

        for (chanblock_index, mut di_jones_rev) in
            (0..num_chanblocks).zip(di_jones_rev.outer_iter_mut())
        {
            let range = s![
                time_range_start..time_range_end,
                chanblock_index..chanblock_index + 1,
                ..
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
                Polarisations::default(),
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

            assert_abs_diff_eq!(di_jones_rev, expected, epsilon = 1e-14);
        }
    }

    let mut expected = Array3::from_elem(di_jones.dim(), Jones::identity());
    expected
        .slice_mut(s![.., .., 0])
        .fill(Jones::identity() * 2.0);
    // Make the second chanblock's data be 9x identity.
    expected
        .slice_mut(s![.., .., 1])
        .fill(Jones::identity() * 3.0);
    assert_abs_diff_eq!(di_jones, expected, epsilon = 1e-14);
}

/// The majority of parameters don't matter for these tests.
fn get_default_params() -> DiCalParams {
    let e = Epoch::from_gpst_seconds(1090008640.0);
    DiCalParams {
        input_vis_params: InputVisParams {
            vis_reader: Box::new(
                RawDataReader::new(
                    &PathBuf::from("test_files/1090008640/1090008640.metafits"),
                    &[PathBuf::from(
                        "test_files/1090008640/1090008640_20140721201027_gpubox01_00.fits",
                    )],
                    None,
                    RawDataCorrections::default(),
                    None,
                )
                .unwrap(),
            ),
            solutions: None,
            timeblocks: vec1![Timeblock {
                index: 0,
                range: 0..1,
                timestamps: vec1![e],
                timesteps: vec1![0],
                median: e,
            }],
            time_res: Duration::from_seconds(2.0),
            spw: Spw {
                chanblocks: vec![Chanblock {
                    chanblock_index: 0,
                    unflagged_index: 0,
                    freq: 10.0,
                }],
                flagged_chan_indices: HashSet::new(),
                flagged_chanblock_indices: HashSet::new(),
                chans_per_chanblock: NonZeroUsize::new(1).unwrap(),
                freq_res: 1.0,
                first_freq: 10.0,
            },
            tile_baseline_flags: TileBaselineFlags {
                tile_to_unflagged_cross_baseline_map: HashMap::new(),
                tile_to_unflagged_auto_index_map: HashMap::new(),
                unflagged_cross_baseline_to_tile_map: HashMap::new(),
                unflagged_auto_index_to_tile_map: HashMap::new(),
                flagged_tiles: HashSet::new(),
            },
            using_autos: false,
            ignore_weights: false,
            dut1: Duration::default(),
        },
        beam: Box::new(NoBeam { num_tiles: 1 }),
        source_list: SourceList::new(),
        cal_timeblocks: vec1![Timeblock {
            index: 0,
            range: 0..1,
            timestamps: vec1![e],
            timesteps: vec1![0],
            median: e,
        }],
        uvw_min: 0.0,
        uvw_max: f64::INFINITY,
        freq_centroid: 150e6,
        baseline_weights: Vec1::try_from_vec(vec![1.0; 8128]).unwrap(),
        max_iterations: 50,
        stop_threshold: 1e-6,
        min_threshold: 1e-3,
        output_solution_files: vec1![(PathBuf::from("asdf.fits"), CalSolutionType::Fits)],
        output_model_vis_params: None,
        modelling_params: ModellingParams {
            apply_precession: true,
        },
    }
}

/// Test that converting [IncompleteSolutions] to [CalibrationSolutions] does
/// what's expected.
#[test]
fn incomplete_to_complete_trivial() {
    let mut params = get_default_params();
    params.input_vis_params.timeblocks = vec1![Timeblock {
        index: 0,
        range: 0..1,
        timestamps: vec1![Epoch::from_gpst_seconds(1065880128.0)],
        timesteps: vec1![0],
        median: Epoch::from_gpst_seconds(1065880128.0),
    }];
    params.input_vis_params.spw.chanblocks = vec![
        Chanblock {
            chanblock_index: 0,
            unflagged_index: 0,
            freq: 150e6,
        },
        Chanblock {
            chanblock_index: 1,
            unflagged_index: 1,
            freq: 151e6,
        },
        Chanblock {
            chanblock_index: 2,
            unflagged_index: 2,
            freq: 152e6,
        },
    ];
    params.input_vis_params.spw.flagged_chanblock_indices = HashSet::new();
    params.input_vis_params.tile_baseline_flags.flagged_tiles = HashSet::new();
    let num_timeblocks = params.input_vis_params.timeblocks.len();
    let num_tiles = params
        .input_vis_params
        .get_obs_context()
        .get_total_num_tiles();
    let num_chanblocks = params.input_vis_params.spw.chanblocks.len();

    let incomplete_di_jones: Vec<Jones<f64>> = (0..num_tiles * num_chanblocks)
        .map(|i| Jones::identity() * (i + 1) as f64 * if is_prime(i) { 1.0 } else { 0.5 })
        .collect();
    let incomplete_di_jones = Array3::from_shape_vec(
        (num_timeblocks, num_tiles, num_chanblocks),
        incomplete_di_jones,
    )
    .unwrap();
    let incomplete = IncompleteSolutions {
        di_jones: incomplete_di_jones.clone(),
        timeblocks: &params.input_vis_params.timeblocks,
        chanblocks: &params.input_vis_params.spw.chanblocks,
        max_iterations: 50,
        stop_threshold: 1e-8,
        min_threshold: 1e-4,
    };

    let complete = incomplete.into_cal_sols(&params, None);

    // The "complete" solutions should have inverted Jones matrices.
    let expected = incomplete_di_jones.mapv(|v| v.inv());

    assert_abs_diff_eq!(complete.di_jones, expected);

    assert!(complete.flagged_tiles.is_empty());
    assert!(complete.flagged_chanblocks.is_empty());
}

// Make the first chanblock flagged. Everything should then just be "shifted one
// over".
#[test]
fn incomplete_to_complete_flags_simple() {
    let mut params = get_default_params();
    params.input_vis_params.timeblocks = vec1![Timeblock {
        index: 0,
        range: 0..1,
        timestamps: vec1![Epoch::from_gpst_seconds(1065880128.0)],
        timesteps: vec1![0],
        median: Epoch::from_gpst_seconds(1065880128.0)
    }];
    params.input_vis_params.spw.chanblocks = vec![
        Chanblock {
            chanblock_index: 1,
            unflagged_index: 0,
            freq: 151e6,
        },
        Chanblock {
            chanblock_index: 2,
            unflagged_index: 1,
            freq: 152e6,
        },
        Chanblock {
            chanblock_index: 3,
            unflagged_index: 2,
            freq: 153e6,
        },
    ];
    params.input_vis_params.spw.flagged_chanblock_indices = HashSet::from([0]);
    params.input_vis_params.tile_baseline_flags.flagged_tiles = HashSet::new();
    let num_timeblocks = params.input_vis_params.timeblocks.len();
    let total_num_tiles = params
        .input_vis_params
        .get_obs_context()
        .get_total_num_tiles();
    let num_tiles = total_num_tiles
        - params
            .input_vis_params
            .tile_baseline_flags
            .flagged_tiles
            .len();
    let num_chanblocks = params.input_vis_params.spw.chanblocks.len();

    let di_jones: Vec<Jones<f64>> = (0..num_tiles * num_chanblocks)
        .map(|i| Jones::identity() * (i + 1) as f64 * if is_prime(i) { 1.0 } else { 0.5 })
        .collect();
    let incomplete_di_jones =
        Array3::from_shape_vec((num_timeblocks, num_tiles, num_chanblocks), di_jones).unwrap();
    let incomplete = IncompleteSolutions {
        di_jones: incomplete_di_jones.clone(),
        timeblocks: &params.input_vis_params.timeblocks,
        chanblocks: &params.input_vis_params.spw.chanblocks,
        max_iterations: 50,
        stop_threshold: 1e-8,
        min_threshold: 1e-4,
    };

    let complete = incomplete.into_cal_sols(&params, None);

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
        complete.di_jones.slice(s![.., .., 1..]),
        incomplete_di_jones.mapv(|v| v.inv())
    );

    assert!(complete.flagged_tiles.is_empty());
    assert_eq!(complete.flagged_chanblocks.len(), 1);
    assert!(complete.flagged_chanblocks.contains(&0));
}

// Same as above, but make the last chanblock flagged.
#[test]
fn incomplete_to_complete_flags_simple2() {
    let mut params = get_default_params();
    params.input_vis_params.timeblocks = vec1![Timeblock {
        index: 0,
        range: 0..1,
        timestamps: vec1![Epoch::from_gpst_seconds(1065880128.0)],
        timesteps: vec1![0],
        median: Epoch::from_gpst_seconds(1065880128.0)
    }];
    params.input_vis_params.spw.chanblocks = vec![
        Chanblock {
            chanblock_index: 0,
            unflagged_index: 0,
            freq: 151e6,
        },
        Chanblock {
            chanblock_index: 1,
            unflagged_index: 1,
            freq: 152e6,
        },
        Chanblock {
            chanblock_index: 2,
            unflagged_index: 2,
            freq: 153e6,
        },
    ];
    params.input_vis_params.spw.flagged_chanblock_indices = HashSet::from([3]);
    let num_timeblocks = params.input_vis_params.timeblocks.len();
    let num_chanblocks = params.input_vis_params.spw.chanblocks.len();
    let total_num_tiles = params
        .input_vis_params
        .get_obs_context()
        .get_total_num_tiles();
    let num_tiles = total_num_tiles
        - params
            .input_vis_params
            .tile_baseline_flags
            .flagged_tiles
            .len();

    let incomplete_di_jones: Vec<Jones<f64>> = (0..num_tiles * num_chanblocks)
        .map(|i| Jones::identity() * (i + 1) as f64 * if is_prime(i) { 1.0 } else { 0.5 })
        .collect();
    let incomplete_di_jones = Array3::from_shape_vec(
        (num_timeblocks, num_tiles, num_chanblocks),
        incomplete_di_jones,
    )
    .unwrap();
    let incomplete = IncompleteSolutions {
        di_jones: incomplete_di_jones.clone(),
        timeblocks: &params.input_vis_params.timeblocks,
        chanblocks: &params.input_vis_params.spw.chanblocks,
        max_iterations: 50,
        stop_threshold: 1e-8,
        min_threshold: 1e-4,
    };

    let complete = incomplete.into_cal_sols(&params, None);

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
        complete.di_jones.slice(s![.., .., ..-1]),
        incomplete_di_jones.mapv(|v| v.inv())
    );

    assert!(complete.flagged_tiles.is_empty());
    assert_eq!(complete.flagged_chanblocks.len(), 1);
    assert!(complete.flagged_chanblocks.contains(&3));
}

#[test]
fn incomplete_to_complete_flags_complex() {
    let mut params = get_default_params();
    params.input_vis_params.timeblocks = vec1![Timeblock {
        index: 0,
        range: 0..1,
        timestamps: vec1![Epoch::from_gpst_seconds(1065880128.0)],
        timesteps: vec1![0],
        median: Epoch::from_gpst_seconds(1065880128.0)
    }];
    params.input_vis_params.spw.chanblocks = vec![
        Chanblock {
            chanblock_index: 0,
            unflagged_index: 0,
            freq: 150e6,
        },
        Chanblock {
            chanblock_index: 2,
            unflagged_index: 1,
            freq: 152e6,
        },
        Chanblock {
            chanblock_index: 3,
            unflagged_index: 2,
            freq: 153e6,
        },
    ];
    params.input_vis_params.spw.flagged_chanblock_indices = HashSet::from([1]);
    params.input_vis_params.tile_baseline_flags.flagged_tiles = HashSet::from([2]);
    let num_timeblocks = params.input_vis_params.timeblocks.len();
    let num_chanblocks = params.input_vis_params.spw.chanblocks.len();
    let total_num_tiles = params
        .input_vis_params
        .get_obs_context()
        .get_total_num_tiles();
    let num_tiles = total_num_tiles
        - params
            .input_vis_params
            .tile_baseline_flags
            .flagged_tiles
            .len();
    let total_num_chanblocks =
        num_chanblocks + params.input_vis_params.spw.flagged_chanblock_indices.len();

    // Cower at my evil, awful code.
    let mut primes = vec1![2];
    while primes.len() < num_tiles * num_chanblocks {
        let next = (*primes.last() + 1..).find(|&i| is_prime(i)).unwrap();
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
        timeblocks: &params.input_vis_params.timeblocks,
        chanblocks: &params.input_vis_params.spw.chanblocks,
        max_iterations: 50,
        stop_threshold: 1e-8,
        min_threshold: 1e-4,
    };

    let complete = incomplete.into_cal_sols(&params, None);

    // For programmer sanity, enforce here that this test only ever has one
    // timeblock.
    assert_eq!(complete.di_jones.dim().0, 1);

    let mut i_unflagged_tile = 0;
    for i_tile in 0..total_num_tiles {
        let sub_array = complete.di_jones.slice(s![0, i_tile, ..]);
        let mut i_unflagged_chanblock = 0;

        if params
            .input_vis_params
            .tile_baseline_flags
            .flagged_tiles
            .contains(&i_tile)
        {
            assert!(sub_array.iter().all(|j| j.any_nan()));
        } else {
            for i_chan in 0..total_num_chanblocks {
                if params
                    .input_vis_params
                    .spw
                    .flagged_chanblock_indices
                    .contains(&(i_chan as u16))
                {
                    assert!(sub_array[i_chan].any_nan());
                } else {
                    assert_abs_diff_eq!(
                        sub_array[i_chan],
                        Jones::identity()
                            / primes[i_unflagged_tile * num_chanblocks + i_unflagged_chanblock]
                                as f64
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

#[test]
fn test_multiple_timeblocks_behave() {
    let timestamps = vec1![
        Epoch::from_gpst_seconds(1090008640.0),
        Epoch::from_gpst_seconds(1090008642.0),
        Epoch::from_gpst_seconds(1090008644.0),
    ];
    let num_timesteps = timestamps.len();
    let num_tiles = 5;
    let num_baselines = num_tiles * (num_tiles - 1) / 2;
    let num_chanblocks = 1;

    let vis_shape = (num_timesteps, num_chanblocks, num_baselines);
    let vis_data: Array3<Jones<f32>> = Array3::from_elem(vis_shape, Jones::identity() * 4.0);
    let vis_model: Array3<Jones<f32>> = Array3::from_elem(vis_shape, Jones::identity());

    let timeblocks = timesteps_to_timeblocks(
        &timestamps,
        Duration::from_seconds(2.0),
        NonZeroUsize::new(1).unwrap(),
        None,
    );
    let spws = channels_to_chanblocks(
        &[150000000],
        40e3 as u64,
        NonZeroUsize::new(1).unwrap(),
        &HashSet::new(),
    );

    let (incomplete_sols, _) = calibrate_timeblocks(
        vis_data.view(),
        vis_model.view(),
        &timeblocks,
        &spws.first().unwrap().chanblocks,
        10,
        1e-8,
        1e-4,
        Polarisations::default(),
        false,
    );

    // The solutions for all timeblocks should be the same.
    let expected = Array3::from_elem(
        (num_timesteps, num_tiles, num_chanblocks),
        Jones::identity() * 2.0,
    );
    assert_abs_diff_eq!(incomplete_sols.di_jones, expected, epsilon = 1e-14);
}

/// Test that multi-timeblock calibration works correctly with identity initial guesses
/// instead of copying all-timesteps initial guesses.
#[test]
fn test_multiple_timeblocks_with_identity_initial_guesses() {
    let timestamps = vec1![
        Epoch::from_gpst_seconds(1090008640.0),
        Epoch::from_gpst_seconds(1090008642.0),
        Epoch::from_gpst_seconds(1090008644.0),
        Epoch::from_gpst_seconds(1090008646.0),
        Epoch::from_gpst_seconds(1090008648.0),
        Epoch::from_gpst_seconds(1090008650.0),
    ];
    let num_timesteps = timestamps.len();
    let num_tiles = 5;
    let num_baselines = num_tiles * (num_tiles - 1) / 2;
    let num_chanblocks = 1;

    let vis_shape = (num_timesteps, num_chanblocks, num_baselines);
    let vis_data: Array3<Jones<f32>> = Array3::from_elem(vis_shape, Jones::identity() * 4.0);
    let vis_model: Array3<Jones<f32>> = Array3::from_elem(vis_shape, Jones::identity());

    // Create timeblocks with 2 timesteps each (3 timeblocks total)
    let timeblocks = timesteps_to_timeblocks(
        &timestamps,
        Duration::from_seconds(2.0),
        NonZeroUsize::new(2).unwrap(),
        None,
    );
    let spws = channels_to_chanblocks(
        &[150000000],
        40e3 as u64,
        NonZeroUsize::new(1).unwrap(),
        &HashSet::new(),
    );

    let (incomplete_sols, results) = calibrate_timeblocks(
        vis_data.view(),
        vis_model.view(),
        &timeblocks,
        &spws.first().unwrap().chanblocks,
        10,
        1e-8,
        1e-4,
        Polarisations::default(),
        false,
    );

    // All timeblocks should converge
    let num_timeblocks = timeblocks.len();
    let num_chanblocks = spws.first().unwrap().chanblocks.len();
    let expected_converged = num_timeblocks * num_chanblocks;
    let actual_converged = results.iter().filter(|r| r.converged).count();
    assert_eq!(
        actual_converged, expected_converged,
        "All timeblocks should converge"
    );

    // The solutions for all timeblocks should be the same (2 * identity)
    let expected = Array3::from_elem(
        (num_timeblocks, num_tiles, num_chanblocks),
        Jones::identity() * 2.0,
    );
    assert_abs_diff_eq!(incomplete_sols.di_jones, expected, epsilon = 1e-14);
}

#[test]
fn test_chanblocks_without_data_have_nan_solutions() {
    let timestamps = vec1![
        Epoch::from_gpst_seconds(1090008640.0),
        Epoch::from_gpst_seconds(1090008642.0),
        Epoch::from_gpst_seconds(1090008644.0),
    ];
    let num_timesteps = timestamps.len();
    let num_tiles = 5;
    let num_baselines = num_tiles * (num_tiles - 1) / 2;
    let freqs = [150000000];
    let num_chanblocks = freqs.len();

    let vis_shape = (num_timesteps, num_chanblocks, num_baselines);
    let vis_data: Array3<Jones<f32>> = Array3::zeros(vis_shape);
    let vis_model: Array3<Jones<f32>> = Array3::zeros(vis_shape);

    let timeblocks = timesteps_to_timeblocks(
        &timestamps,
        Duration::from_seconds(2.0),
        NonZeroUsize::new(1).unwrap(),
        None,
    );
    let fences = channels_to_chanblocks(
        &freqs,
        40e3 as u64,
        NonZeroUsize::new(1).unwrap(),
        &HashSet::new(),
    );

    let (incomplete_sols, results) = calibrate_timeblocks(
        vis_data.view(),
        vis_model.view(),
        &timeblocks,
        &fences[0].chanblocks,
        10,
        1e-8,
        1e-4,
        Polarisations::default(),
        false,
    );
    // All solutions are NaN, because all data and model Jones matrices were 0.
    assert!(incomplete_sols.di_jones.into_iter().all(|j| j.any_nan()));
    // Everything took 1 iteration.
    assert!(results.iter().all(|r| r.num_iterations == 1));

    // If we make the visibilities exactly the same and non zero, then we should
    // still see 1 iteration, but with non-NaN solutions.
    let vis_data: Array3<Jones<f32>> = Array3::from_elem(vis_shape, Jones::identity());
    let vis_model: Array3<Jones<f32>> = Array3::from_elem(vis_shape, Jones::identity());

    let (incomplete_sols, results) = calibrate_timeblocks(
        vis_data.view(),
        vis_model.view(),
        &timeblocks,
        &fences[0].chanblocks,
        10,
        1e-8,
        1e-4,
        Polarisations::default(),
        false,
    );
    assert!(incomplete_sols.di_jones.into_iter().all(|j| !j.any_nan()));
    assert!(results.iter().all(|r| r.num_iterations == 1));
}

#[test]
fn test_recalibrating_failed_chanblocks() {
    let timestamps = vec1![Epoch::from_gpst_seconds(1090008640.0),];
    let num_timesteps = timestamps.len();
    let num_tiles = 5;
    let num_baselines = num_tiles * (num_tiles - 1) / 2;
    let freqs = [150000000, 150040000, 150080000];
    let num_chanblocks = freqs.len();

    let vis_shape = (num_timesteps, num_chanblocks, num_baselines);
    let vis_data: Array3<Jones<f32>> = Array3::from_elem(vis_shape, Jones::identity() * 4.0);
    let vis_model: Array3<Jones<f32>> = Array3::from_elem(vis_shape, Jones::identity());

    let timeblocks = timesteps_to_timeblocks(
        &timestamps,
        Duration::from_seconds(2.0),
        NonZeroUsize::new(1).unwrap(),
        None,
    );
    let fences = channels_to_chanblocks(
        &freqs,
        40000,
        NonZeroUsize::new(1).unwrap(),
        &HashSet::new(),
    );

    // Unlike `calibrate_timeblocks`, `calibrate_timeblock` takes in calibration
    // solutions. These are initially set to identity by `calibrate_timeblocks`;
    // here, we set the middle of 3 chanblocks to 0, so that a guess can be made
    // at what its solution should be after it fails.
    let mut di_jones = Array3::from_elem((1, num_tiles, num_chanblocks), Jones::identity());
    di_jones.slice_mut(s![0, .., 1]).fill(Jones::default());
    let pb = ProgressBar::with_draw_target(Some(num_chanblocks as _), ProgressDrawTarget::hidden());
    let results = calibrate_timeblock(
        vis_data.view(),
        vis_model.view(),
        di_jones.view_mut(),
        &timeblocks[0],
        &fences[0].chanblocks,
        10,
        1e-8,
        1e-4,
        Polarisations::default(),
        pb.clone(),
        false,
    );
    // For reasons I don't understand (it's late), chanblocks 0 and 2 need 10
    // iterations. Chanblock 1 only needs 2, which makes sense, because its been
    // feed the correct calibration solution from its neighbours.
    assert_eq!(results[0].num_iterations, 10);
    assert_eq!(results[1].num_iterations, 2);
    assert_eq!(results[2].num_iterations, 10);

    // Ensure that everything breaks when all initial calibration solutions are
    // zeros.
    let mut di_jones = Array3::from_elem((1, num_tiles, num_chanblocks), Jones::default());
    let results = calibrate_timeblock(
        vis_data.view(),
        vis_model.view(),
        di_jones.view_mut(),
        &timeblocks[0],
        &fences[0].chanblocks,
        10,
        1e-8,
        1e-4,
        Polarisations::default(),
        pb,
        false,
    );
    for result in results {
        assert!(!result.converged);
    }
}
