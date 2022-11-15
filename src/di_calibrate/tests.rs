// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Direction-independent calibration tests.

use std::{
    collections::{HashMap, HashSet},
    path::PathBuf,
};

use approx::{assert_abs_diff_eq, assert_abs_diff_ne};
use clap::Parser;
use hifitime::{Duration, Epoch};
use indicatif::{ProgressBar, ProgressDrawTarget};
use marlu::{
    constants::{MWA_HEIGHT_M, MWA_LAT_DEG, MWA_LONG_DEG},
    Jones, LatLngHeight,
};
use mwalib::{
    _get_fits_col, _get_required_fits_key, _open_fits, _open_hdu, fits_open, fits_open_hdu,
    get_fits_col, get_required_fits_key,
};
use ndarray::prelude::*;
use serial_test::serial;
use tempfile::TempDir;
use vec1::{vec1, Vec1};

use super::{calibrate, calibrate_timeblocks, CalVis, IncompleteSolutions};
use crate::{
    averaging::{channels_to_chanblocks, timesteps_to_timeblocks, Chanblock, Fence, Timeblock},
    beam::create_no_beam_object,
    cli::di_calibrate::DiCalParams,
    di_calibrate::calibrate_timeblock,
    math::{is_prime, TileBaselineFlags},
    solutions::CalSolutionType,
    srclist::SourceList,
    tests::reduced_obsids::get_reduced_1090008640,
    vis_io::read::{MsReader, RawDataCorrections, RawDataReader, VisRead},
    CalibrationSolutions, DiCalArgs, VisSimulateArgs,
};

#[test]
#[serial]
fn test_1090008640_di_calibrate_writes_solutions() {
    let tmp_dir = TempDir::new().expect("couldn't make tmp dir");
    let args = get_reduced_1090008640(true, false);
    let data = args.data.unwrap();
    let metafits = &data[0];
    let gpufits = &data[1];
    let sols = tmp_dir.path().join("sols.fits");
    let cal_model = tmp_dir.path().join("hyp_model.uvfits");

    #[rustfmt::skip]
    let cal_args = DiCalArgs::parse_from([
        "di-calibrate",
        "--data", metafits, gpufits,
        "--source-list", &args.source_list.unwrap(),
        "--outputs", &format!("{}", sols.display()),
        "--model-filenames", &format!("{}", cal_model.display()),
        "--no-progress-bars",
    ]);

    // Run di-cal and check that it succeeds
    let result = cal_args.run(false);
    assert!(result.is_ok(), "result={:?} not ok", result.err().unwrap());

    // check solutions file has been created, is readable
    assert!(sols.exists(), "sols file not written");
    let sol_data = CalibrationSolutions::read_solutions_from_ext(sols, metafits.into()).unwrap();
    assert_eq!(sol_data.obsid, Some(1090008640));
}

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
        input_data: Box::new(
            RawDataReader::new(
                &"test_files/1090008640/1090008640.metafits",
                &["test_files/1090008640/1090008640_20140721201027_gpubox01_00.fits"],
                None,
                RawDataCorrections::default(),
            )
            .unwrap(),
        ),
        raw_data_corrections: None,
        beam: create_no_beam_object(1),
        source_list: SourceList::new(),
        uvw_min: 0.0,
        uvw_max: f64::INFINITY,
        freq_centroid: 150e6,
        baseline_weights: Vec1::try_from_vec(vec![1.0; 8128]).unwrap(),
        timeblocks: vec1![Timeblock {
            index: 0,
            range: 0..1,
            timestamps: vec1![e],
            median: e
        }],
        timesteps: vec1![0],
        freq_average_factor: 1,
        fences: vec1![Fence {
            chanblocks: vec![Chanblock {
                chanblock_index: 0,
                unflagged_index: 0,
                _freq: 10.0
            }],
            flagged_chanblock_indices: vec![],
            _first_freq: 10.0,
            _freq_res: Some(1.0)
        }],
        unflagged_fine_chan_freqs: vec![0.0],
        flagged_fine_chans: HashSet::new(),
        tile_baseline_flags: TileBaselineFlags {
            tile_to_unflagged_cross_baseline_map: HashMap::new(),
            tile_to_unflagged_auto_index_map: HashMap::new(),
            unflagged_cross_baseline_to_tile_map: HashMap::new(),
            unflagged_auto_index_to_tile_map: HashMap::new(),
            flagged_tiles: HashSet::new(),
        },
        unflagged_tile_xyzs: vec![],
        array_position: LatLngHeight {
            longitude_rad: 0.0,
            latitude_rad: 0.0,
            height_metres: 0.0,
        },
        dut1: Duration::from_seconds(0.0),
        apply_precession: false,
        max_iterations: 50,
        stop_threshold: 1e-6,
        min_threshold: 1e-3,
        output_solutions_filenames: vec![(CalSolutionType::Fits, PathBuf::from("asdf.fits"))],
        model_files: None,
        output_model_time_average_factor: 1,
        output_model_freq_average_factor: 1,
        no_progress_bars: true,
        modeller_info: crate::model::ModellerInfo::Cpu,
    }
}

/// Test that converting [IncompleteSolutions] to [CalibrationSolutions] does
/// what's expected.
#[test]
fn incomplete_to_complete_trivial() {
    let mut params = get_default_params();
    params.timeblocks = vec1![Timeblock {
        index: 0,
        range: 0..1,
        timestamps: vec1![Epoch::from_gpst_seconds(1065880128.0)],
        median: Epoch::from_gpst_seconds(1065880128.0),
    }];
    params.fences.first_mut().chanblocks = vec![
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
    params.fences.first_mut().flagged_chanblock_indices = vec![];
    params.tile_baseline_flags.flagged_tiles = HashSet::new();
    let num_timeblocks = params.timeblocks.len();
    let num_tiles = params.get_total_num_tiles();
    let num_chanblocks = params.fences.first().chanblocks.len();

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
        timeblocks: &params.timeblocks,
        chanblocks: &params.fences.first().chanblocks,
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
    params.timeblocks = vec1![Timeblock {
        index: 0,
        range: 0..1,
        timestamps: vec1![Epoch::from_gpst_seconds(1065880128.0)],
        median: Epoch::from_gpst_seconds(1065880128.0)
    }];
    params.fences.first_mut().chanblocks = vec![
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
    params.fences.first_mut().flagged_chanblock_indices = vec![0];
    params.tile_baseline_flags.flagged_tiles = HashSet::new();
    let num_timeblocks = params.timeblocks.len();
    let total_num_tiles = params.get_total_num_tiles();
    let num_tiles = total_num_tiles - params.tile_baseline_flags.flagged_tiles.len();
    let num_chanblocks = params.fences.first().chanblocks.len();

    let di_jones: Vec<Jones<f64>> = (0..num_tiles * num_chanblocks)
        .map(|i| Jones::identity() * (i + 1) as f64 * if is_prime(i) { 1.0 } else { 0.5 })
        .collect();
    let incomplete_di_jones =
        Array3::from_shape_vec((num_timeblocks, num_tiles, num_chanblocks), di_jones).unwrap();
    let incomplete = IncompleteSolutions {
        di_jones: incomplete_di_jones.clone(),
        timeblocks: &params.timeblocks,
        chanblocks: &params.fences.first().chanblocks,
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
    params.timeblocks = vec1![Timeblock {
        index: 0,
        range: 0..1,
        timestamps: vec1![Epoch::from_gpst_seconds(1065880128.0)],
        median: Epoch::from_gpst_seconds(1065880128.0)
    }];
    params.fences.first_mut().chanblocks = vec![
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
    params.tile_baseline_flags.flagged_tiles = HashSet::new();
    params.fences.first_mut().flagged_chanblock_indices = vec![3];
    let num_timeblocks = params.timeblocks.len();
    let num_chanblocks = params.fences.first().chanblocks.len();
    let total_num_tiles = params.get_total_num_tiles();
    let num_tiles = total_num_tiles - params.tile_baseline_flags.flagged_tiles.len();

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
        timeblocks: &params.timeblocks,
        chanblocks: &params.fences.first().chanblocks,
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
    params.timeblocks = vec1![Timeblock {
        index: 0,
        range: 0..1,
        timestamps: vec1![Epoch::from_gpst_seconds(1065880128.0)],
        median: Epoch::from_gpst_seconds(1065880128.0)
    }];
    params.fences.first_mut().chanblocks = vec![
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
    params.fences.first_mut().flagged_chanblock_indices = vec![1];
    params.tile_baseline_flags.flagged_tiles = HashSet::from([2]);
    let num_timeblocks = params.timeblocks.len();
    let num_chanblocks = params.fences.first().chanblocks.len();
    let total_num_tiles = params.get_total_num_tiles();
    let num_tiles = total_num_tiles - params.tile_baseline_flags.flagged_tiles.len();
    let total_num_chanblocks =
        num_chanblocks + params.fences.first().flagged_chanblock_indices.len();

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
        timeblocks: &params.timeblocks,
        chanblocks: &params.fences.first().chanblocks,
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

        if params.tile_baseline_flags.flagged_tiles.contains(&i_tile) {
            assert!(sub_array.iter().all(|j| j.any_nan()));
        } else {
            for i_chan in 0..total_num_chanblocks {
                if params
                    .fences
                    .first()
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
fn test_1090008640_di_calibrate_uses_array_position() {
    let tmp_dir = TempDir::new().expect("couldn't make tmp dir");
    let args = get_reduced_1090008640(true, false);
    let data = args.data.unwrap();
    let metafits = &data[0];
    let gpufits = &data[1];
    let sols = tmp_dir.path().join("sols.fits");
    let cal_model = tmp_dir.path().join("hyp_model.uvfits");

    // with non-default array position
    let exp_lat_deg = MWA_LAT_DEG - 1.;
    let exp_long_deg = MWA_LONG_DEG - 1.;
    let exp_height_m = MWA_HEIGHT_M - 1.;

    #[rustfmt::skip]
    let cal_args = DiCalArgs::parse_from([
        "di-calibrate",
        "--data", metafits, gpufits,
        "--source-list", args.source_list.as_ref().unwrap(),
        "--outputs", &format!("{}", sols.display()),
        "--model-filenames", &format!("{}", cal_model.display()),
        "--array-position",
            &format!("{exp_long_deg}"),
            &format!("{exp_lat_deg}"),
            &format!("{exp_height_m}"),
        "--no-progress-bars",
    ]);

    let pos = cal_args.array_position.unwrap();

    assert_abs_diff_eq!(pos[0], exp_long_deg);
    assert_abs_diff_eq!(pos[1], exp_lat_deg);
    assert_abs_diff_eq!(pos[2], exp_height_m);
}

#[test]
fn test_1090008640_di_calibrate_array_pos_requires_3_args() {
    let tmp_dir = TempDir::new().expect("couldn't make tmp dir");
    let args = get_reduced_1090008640(true, false);
    let data = args.data.unwrap();
    let metafits = &data[0];
    let gpufits = &data[1];
    let sols = tmp_dir.path().join("sols.fits");
    let cal_model = tmp_dir.path().join("hyp_model.uvfits");

    // no height specified
    let exp_lat_deg = MWA_LAT_DEG - 1.;
    let exp_long_deg = MWA_LONG_DEG - 1.;

    #[rustfmt::skip]
    let result = DiCalArgs::try_parse_from([
        "di-calibrate",
        "--data", metafits, gpufits,
        "--source-list", args.source_list.as_ref().unwrap(),
        "--outputs", &format!("{}", sols.display()),
        "--model-filenames", &format!("{}", cal_model.display()),
        "--array-position",
            &format!("{exp_long_deg}"),
            &format!("{exp_lat_deg}"),
    ]);

    assert!(result.is_err());
    assert!(matches!(
        result.err().unwrap().kind(),
        clap::ErrorKind::WrongNumberOfValues
    ));
}

#[test]
/// Generate a model with "vis-simulate" (in uvfits), then feed it to
/// "di-calibrate" and write out the model used for calibration (as uvfits). The
/// visibilities should be exactly the same.
fn test_1090008640_calibrate_model_uvfits() {
    let num_timesteps = 2;
    let num_chans = 10;

    let temp_dir = TempDir::new().expect("couldn't make tmp dir");
    let model = temp_dir.path().join("model.uvfits");
    let args = get_reduced_1090008640(false, false);
    let metafits = &args.data.as_ref().unwrap()[0];
    let srclist = args.source_list.unwrap();
    #[rustfmt::skip]
    let sim_args = VisSimulateArgs::parse_from([
        "vis-simulate",
        "--metafits", metafits,
        "--source-list", &srclist,
        "--output-model-files", &format!("{}", model.display()),
        "--num-timesteps", &format!("{num_timesteps}"),
        "--num-fine-channels", &format!("{num_chans}"),
        "--veto-threshold", "0.0", // Don't complicate things with vetoing
        "--no-progress-bars",
    ]);

    // Run vis-simulate and check that it succeeds
    let result = sim_args.run(false);
    assert!(result.is_ok(), "result={:?} not ok", result.err().unwrap());

    let sols = temp_dir.path().join("sols.fits");
    let cal_model = temp_dir.path().join("cal_model.uvfits");

    #[rustfmt::skip]
    let cal_args = DiCalArgs::parse_from([
        "di-calibrate",
        "--data", &format!("{}", model.display()), metafits,
        "--source-list", &srclist,
        "--outputs", &format!("{}", sols.display()),
        "--model-filenames", &format!("{}", cal_model.display()),
        // If we don't give the array position, there's a bit flip in the results. grrr
        "--array-position", "116.67081523611111", "-26.703319405555554", "377.827",
        "--veto-threshold", "0.0", // Don't complicate things with vetoing
        "--no-progress-bars",
    ]);

    // Run di-cal and check that it succeeds
    let result = cal_args.into_params().unwrap().calibrate();
    assert!(result.is_ok(), "result={:?} not ok", result.err().unwrap());
    let sols = result.unwrap();

    let mut uvfits_m = fits_open!(&model).unwrap();
    let hdu_m = fits_open_hdu!(&mut uvfits_m, 0).unwrap();
    let gcount_m: String = get_required_fits_key!(&mut uvfits_m, &hdu_m, "GCOUNT").unwrap();
    let pcount_m: String = get_required_fits_key!(&mut uvfits_m, &hdu_m, "PCOUNT").unwrap();
    let floats_per_pol_m: String = get_required_fits_key!(&mut uvfits_m, &hdu_m, "NAXIS2").unwrap();
    let num_pols_m: String = get_required_fits_key!(&mut uvfits_m, &hdu_m, "NAXIS3").unwrap();
    let num_fine_freq_chans_m: String =
        get_required_fits_key!(&mut uvfits_m, &hdu_m, "NAXIS4").unwrap();
    let jd_zero_m: String = get_required_fits_key!(&mut uvfits_m, &hdu_m, "PZERO5").unwrap();
    let ptype4_m: String = get_required_fits_key!(&mut uvfits_m, &hdu_m, "PTYPE4").unwrap();

    let mut uvfits_c = fits_open!(&cal_model).unwrap();
    let hdu_c = fits_open_hdu!(&mut uvfits_c, 0).unwrap();
    let gcount_c: String = get_required_fits_key!(&mut uvfits_c, &hdu_c, "GCOUNT").unwrap();
    let pcount_c: String = get_required_fits_key!(&mut uvfits_c, &hdu_c, "PCOUNT").unwrap();
    let floats_per_pol_c: String = get_required_fits_key!(&mut uvfits_c, &hdu_c, "NAXIS2").unwrap();
    let num_pols_c: String = get_required_fits_key!(&mut uvfits_c, &hdu_c, "NAXIS3").unwrap();
    let num_fine_freq_chans_c: String =
        get_required_fits_key!(&mut uvfits_c, &hdu_c, "NAXIS4").unwrap();
    let jd_zero_c: String = get_required_fits_key!(&mut uvfits_c, &hdu_c, "PZERO5").unwrap();
    let ptype4_c: String = get_required_fits_key!(&mut uvfits_c, &hdu_c, "PTYPE4").unwrap();

    let pcount: usize = pcount_m.parse().unwrap();
    assert_eq!(pcount, 7);
    assert_eq!(gcount_m, gcount_c);
    assert_eq!(pcount_m, pcount_c);
    assert_eq!(floats_per_pol_m, floats_per_pol_c);
    assert_eq!(num_pols_m, num_pols_c);
    assert_eq!(num_fine_freq_chans_m, num_fine_freq_chans_c);
    assert_eq!(jd_zero_m, jd_zero_c);
    assert_eq!(ptype4_m, ptype4_c);

    let hdu_m = fits_open_hdu!(&mut uvfits_m, 1).unwrap();
    let tile_names_m: Vec<String> = get_fits_col!(&mut uvfits_m, &hdu_m, "ANNAME").unwrap();
    let hdu_c = fits_open_hdu!(&mut uvfits_c, 1).unwrap();
    let tile_names_c: Vec<String> = get_fits_col!(&mut uvfits_c, &hdu_c, "ANNAME").unwrap();
    for (tile_m, tile_c) in tile_names_m.into_iter().zip(tile_names_c.into_iter()) {
        assert_eq!(tile_m, tile_c);
    }

    // Test visibility values.
    fits_open_hdu!(&mut uvfits_m, 0).unwrap();
    let mut group_params_m = Array1::zeros(pcount);
    let mut vis_m = Array1::zeros(num_chans * 4 * 3);
    fits_open_hdu!(&mut uvfits_c, 0).unwrap();
    let mut group_params_c = group_params_m.clone();
    let mut vis_c = vis_m.clone();

    let mut status = 0;
    for i_row in 0..gcount_m.parse::<i64>().unwrap() {
        unsafe {
            // ffggpe = fits_read_grppar_flt
            fitsio_sys::ffggpe(
                uvfits_m.as_raw(),           /* I - FITS file pointer                       */
                1 + i_row,                   /* I - group to read (1 = 1st group)           */
                1,                           /* I - first vector element to read (1 = 1st)  */
                group_params_m.len() as i64, /* I - number of values to read                */
                group_params_m.as_mut_ptr(), /* O - array of values that are returned       */
                &mut status,                 /* IO - error status                           */
            );
            assert_eq!(status, 0, "Status wasn't 0");
            assert_abs_diff_ne!(group_params_m, group_params_c);
            // ffggpe = fits_read_grppar_flt
            fitsio_sys::ffggpe(
                uvfits_c.as_raw(),           /* I - FITS file pointer                       */
                1 + i_row,                   /* I - group to read (1 = 1st group)           */
                1,                           /* I - first vector element to read (1 = 1st)  */
                group_params_c.len() as i64, /* I - number of values to read                */
                group_params_c.as_mut_ptr(), /* O - array of values that are returned       */
                &mut status,                 /* IO - error status                           */
            );
            assert_eq!(status, 0, "Status wasn't 0");
            assert_abs_diff_eq!(group_params_m, group_params_c);

            // ffgpve = fits_read_img_flt
            fitsio_sys::ffgpve(
                uvfits_m.as_raw(),  /* I - FITS file pointer                       */
                1 + i_row,          /* I - group to read (1 = 1st group)           */
                1,                  /* I - first vector element to read (1 = 1st)  */
                vis_m.len() as i64, /* I - number of values to read                */
                0.0,                /* I - value for undefined pixels              */
                vis_m.as_mut_ptr(), /* O - array of values that are returned       */
                &mut 0,             /* O - set to 1 if any values are null; else 0 */
                &mut status,        /* IO - error status                           */
            );
            assert_abs_diff_ne!(vis_m, vis_c);
            // ffgpve = fits_read_img_flt
            fitsio_sys::ffgpve(
                uvfits_c.as_raw(),  /* I - FITS file pointer                       */
                1 + i_row,          /* I - group to read (1 = 1st group)           */
                1,                  /* I - first vector element to read (1 = 1st)  */
                vis_c.len() as i64, /* I - number of values to read                */
                0.0,                /* I - value for undefined pixels              */
                vis_c.as_mut_ptr(), /* O - array of values that are returned       */
                &mut 0,             /* O - set to 1 if any values are null; else 0 */
                &mut status,        /* IO - error status                           */
            );
            assert_eq!(status, 0, "Status wasn't 0");
            assert_abs_diff_eq!(vis_m, vis_c);
        };
    }

    // Inspect the solutions; they should all be close to identity.
    assert_abs_diff_eq!(
        sols.di_jones,
        Array3::from_elem(sols.di_jones.dim(), Jones::identity()),
        epsilon = 1e-15
    );
}

#[test]
#[serial]
/// Generate a model with "vis-simulate" (in a measurement set), then feed it to
/// "di-calibrate" and write out the model used for calibration (into a
/// measurement set). The visibilities should be exactly the same.
fn test_1090008640_calibrate_model_ms() {
    let num_timesteps = 2;
    let num_chans = 10;

    let temp_dir = TempDir::new().expect("couldn't make tmp dir");
    let model = temp_dir.path().join("model.ms");
    let args = get_reduced_1090008640(false, false);
    let metafits = &args.data.as_ref().unwrap()[0];
    let srclist = args.source_list.unwrap();
    #[rustfmt::skip]
    let sim_args = VisSimulateArgs::parse_from([
        "vis-simulate",
        "--metafits", metafits,
        "--source-list", &srclist,
        "--output-model-files", &format!("{}", model.display()),
        "--num-timesteps", &format!("{num_timesteps}"),
        "--num-fine-channels", &format!("{num_chans}"),
        "--no-progress-bars"
    ]);

    // Run vis-simulate and check that it succeeds
    let result = sim_args.run(false);
    assert!(result.is_ok(), "result={:?} not ok", result.err().unwrap());

    let sols = temp_dir.path().join("sols.fits");
    let cal_model = temp_dir.path().join("cal_model.ms");
    #[rustfmt::skip]
    let cal_args = DiCalArgs::parse_from([
        "di-calibrate",
        "--data", &format!("{}", model.display()), metafits,
        "--source-list", &srclist,
        "--outputs", &format!("{}", sols.display()),
        "--model-filenames", &format!("{}", cal_model.display()),
        "--no-progress-bars"
    ]);

    // Run di-cal and check that it succeeds
    let result = cal_args.into_params().unwrap().calibrate();
    assert!(result.is_ok(), "result={:?} not ok", result.err().unwrap());
    let sols = result.unwrap();

    let array_pos = LatLngHeight::mwa();
    let ms_m = MsReader::new(&model, Some(metafits), Some(array_pos)).unwrap();
    let ctx_m = ms_m.get_obs_context();
    let ms_c = MsReader::new(&cal_model, Some(metafits), Some(array_pos)).unwrap();
    let ctx_c = ms_c.get_obs_context();
    assert_eq!(ctx_m.all_timesteps, ctx_c.all_timesteps);
    assert_eq!(ctx_m.all_timesteps.len(), num_timesteps);
    assert_eq!(ctx_m.timestamps, ctx_c.timestamps);
    assert_eq!(ctx_m.fine_chan_freqs, ctx_c.fine_chan_freqs);
    let m_flags = ctx_m.get_tile_flags(false, None).unwrap();
    let c_flags = ctx_c.get_tile_flags(false, None).unwrap();
    for m in &m_flags {
        assert!(c_flags.contains(m));
    }
    assert_eq!(ctx_m.tile_xyzs, ctx_c.tile_xyzs);
    assert_eq!(ctx_m.flagged_fine_chans, ctx_c.flagged_fine_chans);

    let flagged_fine_chans_set: HashSet<usize> = ctx_m.flagged_fine_chans.iter().cloned().collect();
    let tile_baseline_flags = TileBaselineFlags::new(ctx_m.tile_xyzs.len(), m_flags);
    let max_baseline_idx = tile_baseline_flags
        .tile_to_unflagged_cross_baseline_map
        .values()
        .max()
        .unwrap();
    let data_shape = (
        ctx_m.fine_chan_freqs.len() - ctx_m.flagged_fine_chans.len(),
        max_baseline_idx + 1,
    );
    let mut vis_m = Array2::<Jones<f32>>::zeros(data_shape);
    let mut vis_c = Array2::<Jones<f32>>::zeros(data_shape);
    let mut weight_m = Array2::<f32>::zeros(data_shape);
    let mut weight_c = Array2::<f32>::zeros(data_shape);

    for &timestep in &ctx_m.all_timesteps {
        ms_m.read_crosses(
            vis_m.view_mut(),
            weight_m.view_mut(),
            timestep,
            &tile_baseline_flags,
            &flagged_fine_chans_set,
        )
        .unwrap();
        ms_c.read_crosses(
            vis_c.view_mut(),
            weight_c.view_mut(),
            timestep,
            &tile_baseline_flags,
            &flagged_fine_chans_set,
        )
        .unwrap();

        // Unlike the equivalent uvfits test, we have to use an epsilon here.
        // This is due to the MS antenna positions being in geocentric
        // coordinates and not geodetic like uvfits; in the process of
        // converting from geocentric to geodetic, small float errors are
        // introduced. If a metafits' positions are used instead, the results
        // are *exactly* the same, but we should trust the MS's positions, so
        // these errors must remain.
        assert_abs_diff_eq!(vis_m, vis_c, epsilon = 1e-4);
        assert_abs_diff_eq!(weight_m, weight_c);
    }

    // Inspect the solutions; they should all be close to identity.
    assert_abs_diff_eq!(
        sols.di_jones,
        Array3::from_elem(sols.di_jones.dim(), Jones::identity()),
        epsilon = 2e-8
    );
}

#[test]
fn test_multiple_timeblocks_behave() {
    let timestamps = vec1![
        Epoch::from_gpst_seconds(1090008640.0),
        Epoch::from_gpst_seconds(1090008642.0),
        Epoch::from_gpst_seconds(1090008644.0),
    ];
    let num_timesteps = timestamps.len();
    let timesteps_to_use = Vec1::try_from_vec((0..num_timesteps).collect()).unwrap();
    let num_tiles = 5;
    let num_baselines = num_tiles * (num_tiles - 1) / 2;
    let num_chanblocks = 1;

    let vis_shape = (num_timesteps, num_chanblocks, num_baselines);
    let vis_data: Array3<Jones<f32>> = Array3::from_elem(vis_shape, Jones::identity() * 4.0);
    let vis_model: Array3<Jones<f32>> = Array3::from_elem(vis_shape, Jones::identity());

    let timeblocks = timesteps_to_timeblocks(&timestamps, 1, &timesteps_to_use);
    let fences = channels_to_chanblocks(&[150000000], Some(40e3), 1, &HashSet::new());

    let (incomplete_sols, _) = calibrate_timeblocks(
        vis_data.view(),
        vis_model.view(),
        &timeblocks,
        &fences.first().unwrap().chanblocks,
        10,
        1e-8,
        1e-4,
        false,
        false,
    );

    // The solutions for all timeblocks should be the same.
    let expected = Array3::from_elem(
        (num_timesteps, num_tiles, num_chanblocks),
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
    let timesteps_to_use = Vec1::try_from_vec((0..num_timesteps).collect()).unwrap();
    let num_tiles = 5;
    let num_baselines = num_tiles * (num_tiles - 1) / 2;
    let freqs = [150000000];
    let num_chanblocks = freqs.len();

    let vis_shape = (num_timesteps, num_chanblocks, num_baselines);
    let vis_data: Array3<Jones<f32>> = Array3::zeros(vis_shape);
    let vis_model: Array3<Jones<f32>> = Array3::zeros(vis_shape);

    let timeblocks = timesteps_to_timeblocks(&timestamps, 1, &timesteps_to_use);
    let fences = channels_to_chanblocks(&freqs, Some(40e3), 1, &HashSet::new());

    let (incomplete_sols, results) = calibrate_timeblocks(
        vis_data.view(),
        vis_model.view(),
        &timeblocks,
        &fences[0].chanblocks,
        10,
        1e-8,
        1e-4,
        false,
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
        false,
        false,
    );
    assert!(incomplete_sols.di_jones.into_iter().all(|j| !j.any_nan()));
    assert!(results.iter().all(|r| r.num_iterations == 1));
}

#[test]
fn test_recalibrating_failed_chanblocks() {
    let timestamps = vec1![Epoch::from_gpst_seconds(1090008640.0),];
    let num_timesteps = timestamps.len();
    let timesteps_to_use = Vec1::try_from_vec((0..num_timesteps).collect()).unwrap();
    let num_tiles = 5;
    let num_baselines = num_tiles * (num_tiles - 1) / 2;
    let freqs = [150000000, 150040000, 150080000];
    let num_chanblocks = freqs.len();

    let vis_shape = (num_timesteps, num_chanblocks, num_baselines);
    let vis_data: Array3<Jones<f32>> = Array3::from_elem(vis_shape, Jones::identity() * 4.0);
    let vis_model: Array3<Jones<f32>> = Array3::from_elem(vis_shape, Jones::identity());

    let timeblocks = timesteps_to_timeblocks(&timestamps, 1, &timesteps_to_use);
    let fences = channels_to_chanblocks(&freqs, Some(40e3), 1, &HashSet::new());

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
        pb,
        false,
    );
    for result in results {
        assert!(!result.converged);
    }
}

/// Given calibration parameters and visibilities, this function tests that
/// everything matches an expected quality. The values may change over time but
/// they should be consistent with whatever tests use this test code.
pub(crate) fn test_1090008640_quality(params: DiCalParams, cal_vis: CalVis) {
    let (_, cal_results) = calibrate_timeblocks(
        cal_vis.vis_data_tfb.view(),
        cal_vis.vis_model_tfb.view(),
        &params.timeblocks,
        &params.fences.first().chanblocks,
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
    let mut fewest_iterations = u32::MAX;
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
  {expected_count_50} chanblocks with 50 iterations (got {count_50}),
  {expected_count_42} chanblocks with 42 iterations (got {count_42}),
  chanblocks {expected_chanblocks_42:?} to need 42 iterations (got {chanblocks_42:?}), and
  no chanblocks to finish in less than {expected_fewest_iterations} iterations (got {fewest_iterations}).
"#
        );
    }
}
