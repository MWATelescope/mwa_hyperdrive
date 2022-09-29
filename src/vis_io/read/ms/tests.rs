// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

use std::collections::HashSet;

use approx::assert_abs_diff_eq;
use birli::marlu::XyzGeodetic;
use hifitime::{Duration, Unit};
use marlu::{Jones, MeasurementSetWriter, ObsContext as MarluObsContext, VisContext, VisWrite};
use serial_test::serial; // Need to test serially because casacore is a steaming pile.
use tempfile::tempdir;

use super::*;
use crate::{
    di_calibrate::{get_cal_vis, tests::test_1090008640_quality},
    math::TileBaselineFlags,
    tests::reduced_obsids::get_reduced_1090008640_ms,
};

#[test]
#[serial]
fn test_1090008640_cross_vis() {
    let args = get_reduced_1090008640_ms();
    let ms_reader = if let [metafits, ms] = &args.data.unwrap()[..] {
        match MsReader::new(ms, Some(metafits)) {
            Ok(m) => m,
            Err(e) => panic!("{}", e),
        }
    } else {
        panic!("There weren't 2 elements in args.data");
    };

    let obs_context = &ms_reader.obs_context;
    let total_num_tiles = obs_context.get_total_num_tiles();
    let num_baselines = (total_num_tiles * (total_num_tiles - 1)) / 2;
    let num_chans = obs_context.num_fine_chans_per_coarse_chan;
    let tile_baseline_flags = TileBaselineFlags::new(total_num_tiles, HashSet::new());

    assert_abs_diff_eq!(
        obs_context.timestamps.first().as_gpst_seconds(),
        1090008658.0
    );

    let mut vis = Array2::zeros((num_baselines, num_chans));
    let mut vis_weights = Array2::zeros((num_baselines, num_chans));
    let result = ms_reader.read_crosses(
        vis.view_mut(),
        vis_weights.view_mut(),
        *obs_context.all_timesteps.first(),
        &tile_baseline_flags,
        &HashSet::new(),
    );
    assert!(result.is_ok(), "{}", result.unwrap_err());

    // These values are exactly the same as the raw data when all corrections
    // (except the PFB gains) are turned on. See the
    // read_1090008640_cross_vis_with_corrections test.
    assert_abs_diff_eq!(
        vis[(0, 0)],
        Jones::from([
            c32::new(-1.2564129e2, -1.497961e1),
            c32::new(8.207059e1, -1.4936417e2),
            c32::new(-7.306871e1, 2.36177e2),
            c32::new(-5.5305626e1, -2.3209404e1)
        ])
    );
    assert_abs_diff_eq!(
        vis[(10, 16)],
        Jones::from([
            c32::new(-4.138127e1, -2.638188e2),
            c32::new(5.220332e2, -2.6055228e2),
            c32::new(4.854074e2, -1.9634505e2),
            c32::new(1.6101791e1, -4.4489478e2),
        ])
    );

    // PFB gains will affect weights, but these weren't in Birli when it made
    // this MS; all but one weight are 8.0 (it's flagged).
    assert_abs_diff_eq!(vis_weights[(11, 2)], -8.0);
    // Undo the flag and test all values.
    vis_weights[(11, 2)] = 8.0;
    assert_abs_diff_eq!(vis_weights, Array2::ones(vis_weights.dim()) * 8.0);
}

#[test]
#[serial]
fn read_1090008640_auto_vis() {
    let args = get_reduced_1090008640_ms();
    let ms_reader = if let [metafits, ms] = &args.data.unwrap()[..] {
        match MsReader::new(ms, Some(metafits)) {
            Ok(m) => m,
            Err(e) => panic!("{}", e),
        }
    } else {
        panic!("There weren't 2 elements in args.data");
    };

    let obs_context = &ms_reader.obs_context;
    let total_num_tiles = obs_context.get_total_num_tiles();
    let num_chans = obs_context.num_fine_chans_per_coarse_chan;
    let tile_baseline_flags = TileBaselineFlags::new(total_num_tiles, HashSet::new());

    assert_abs_diff_eq!(
        obs_context.timestamps.first().as_gpst_seconds(),
        1090008658.0
    );

    let mut vis = Array2::zeros((total_num_tiles, num_chans));
    let mut vis_weights = Array2::zeros((total_num_tiles, num_chans));
    let result = ms_reader.read_autos(
        vis.view_mut(),
        vis_weights.view_mut(),
        *obs_context.all_timesteps.first(),
        &tile_baseline_flags,
        &HashSet::new(),
    );
    assert!(result.is_ok(), "{}", result.unwrap_err());

    assert_abs_diff_eq!(
        vis[(0, 0)],
        Jones::from([
            5.3557855e4,
            4.3092007e-7,
            -7.420802e2,
            1.3296518e2,
            -7.420802e2,
            -1.3296518e2,
            5.084874e4,
            -6.6132475e-7
        ])
    );
    assert_abs_diff_eq!(
        vis[(0, 2)],
        Jones::from([
            7.1403125e4,
            -1.3957654e-6,
            -1.0667509e3,
            1.01154564e2,
            -1.0667509e3,
            -1.01154564e2,
            6.7373945e4,
            -7.004146e-7
        ])
    );
    assert_abs_diff_eq!(
        vis[(0, 16)],
        Jones::from([
            1.07272586e5,
            1.9233863e-8,
            -1.0756711e3,
            -4.4097336e1,
            -1.0756711e3,
            4.4097336e1,
            1.0141891e5,
            -9.5129735e-7
        ])
    );
    assert_abs_diff_eq!(
        vis[(10, 16)],
        Jones::from([
            1.0766406e5,
            1.5415758e-6,
            -1.334196e3,
            -1.8926495e3,
            -1.334196e3,
            1.8926495e3,
            1.21874336e5,
            2.0712553e-6
        ])
    );

    assert_abs_diff_eq!(vis_weights, Array2::from_elem(vis_weights.dim(), 8.0));
}

#[test]
#[serial]
fn read_1090008640_auto_vis_with_flags() {
    let args = get_reduced_1090008640_ms();
    let ms_reader = if let [metafits, ms] = &args.data.unwrap()[..] {
        match MsReader::new(ms, Some(metafits)) {
            Ok(m) => m,
            Err(e) => panic!("{}", e),
        }
    } else {
        panic!("There weren't 2 elements in args.data");
    };

    let obs_context = &ms_reader.obs_context;
    let total_num_tiles = obs_context.get_total_num_tiles();
    let num_chans = obs_context.num_fine_chans_per_coarse_chan;
    let tile_flags = HashSet::from([1, 9]);
    let num_unflagged_tiles = total_num_tiles - tile_flags.len();
    let chan_flags = HashSet::from([1]);
    let num_unflagged_chans = num_chans - chan_flags.len();
    let tile_baseline_flags = TileBaselineFlags::new(total_num_tiles, tile_flags);

    assert_abs_diff_eq!(
        obs_context.timestamps.first().as_gpst_seconds(),
        1090008658.0
    );

    let mut vis = Array2::zeros((num_unflagged_tiles, num_unflagged_chans));
    let mut vis_weights = Array2::zeros((num_unflagged_tiles, num_unflagged_chans));
    let result = ms_reader.read_autos(
        vis.view_mut(),
        vis_weights.view_mut(),
        *obs_context.all_timesteps.first(),
        &tile_baseline_flags,
        &chan_flags,
    );
    assert!(result.is_ok(), "{}", result.unwrap_err());

    // Use the same values as the test above, adjusting only the indices.
    assert_abs_diff_eq!(
        vis[(0, 0)],
        Jones::from([
            5.3557855e4,
            4.3092007e-7,
            -7.420802e2,
            1.3296518e2,
            -7.420802e2,
            -1.3296518e2,
            5.084874e4,
            -6.6132475e-7
        ])
    );
    assert_abs_diff_eq!(
        // Channel 2 -> 1
        vis[(0, 1)],
        Jones::from([
            7.1403125e4,
            -1.3957654e-6,
            -1.0667509e3,
            1.01154564e2,
            -1.0667509e3,
            -1.01154564e2,
            6.7373945e4,
            -7.004146e-7
        ])
    );
    assert_abs_diff_eq!(
        // Channel 16 -> 15
        vis[(0, 15)],
        Jones::from([
            1.07272586e5,
            1.9233863e-8,
            -1.0756711e3,
            -4.4097336e1,
            -1.0756711e3,
            4.4097336e1,
            1.0141891e5,
            -9.5129735e-7
        ])
    );
    assert_abs_diff_eq!(
        // Two flagged tiles before tile 10; use index 8. Channel 16 -> 15.
        vis[(8, 15)],
        Jones::from([
            1.0766406e5,
            1.5415758e-6,
            -1.334196e3,
            -1.8926495e3,
            -1.334196e3,
            1.8926495e3,
            1.21874336e5,
            2.0712553e-6
        ])
    );

    assert_abs_diff_eq!(vis_weights, Array2::from_elem(vis_weights.dim(), 8.0));
}

#[test]
#[serial]
fn read_1090008640_cross_and_auto_vis() {
    let args = get_reduced_1090008640_ms();
    let ms_reader = if let [metafits, ms] = &args.data.unwrap()[..] {
        match MsReader::new(ms, Some(metafits)) {
            Ok(m) => m,
            Err(e) => panic!("{}", e),
        }
    } else {
        panic!("There weren't 2 elements in args.data");
    };

    let obs_context = &ms_reader.obs_context;
    let total_num_tiles = obs_context.get_total_num_tiles();
    let num_baselines = (total_num_tiles * (total_num_tiles - 1)) / 2;
    let num_chans = obs_context.num_fine_chans_per_coarse_chan;
    let tile_baseline_flags = TileBaselineFlags::new(total_num_tiles, HashSet::new());

    assert_abs_diff_eq!(
        obs_context.timestamps.first().as_gpst_seconds(),
        1090008658.0
    );

    assert_abs_diff_eq!(
        obs_context.timestamps.first().as_gpst_seconds(),
        1090008658.0
    );

    let mut cross_vis = Array2::zeros((num_baselines, num_chans));
    let mut cross_vis_weights = Array2::zeros((num_baselines, num_chans));
    let mut auto_vis = Array2::zeros((total_num_tiles, num_chans));
    let mut auto_vis_weights = Array2::zeros((total_num_tiles, num_chans));
    let result = ms_reader.read_crosses_and_autos(
        cross_vis.view_mut(),
        cross_vis_weights.view_mut(),
        auto_vis.view_mut(),
        auto_vis_weights.view_mut(),
        *obs_context.all_timesteps.first(),
        &tile_baseline_flags,
        &HashSet::new(),
    );
    assert!(result.is_ok(), "{}", result.unwrap_err());

    assert_abs_diff_eq!(
        cross_vis[(0, 0)],
        Jones::from([
            c32::new(-1.2564129e2, -1.497961e1),
            c32::new(8.207059e1, -1.4936417e2),
            c32::new(-7.306871e1, 2.36177e2),
            c32::new(-5.5305626e1, -2.3209404e1)
        ])
    );
    assert_abs_diff_eq!(
        cross_vis[(10, 16)],
        Jones::from([
            c32::new(-4.138127e1, -2.638188e2),
            c32::new(5.220332e2, -2.6055228e2),
            c32::new(4.854074e2, -1.9634505e2),
            c32::new(1.6101791e1, -4.4489478e2),
        ])
    );

    assert_abs_diff_eq!(cross_vis_weights[(11, 2)], -8.0);
    cross_vis_weights[(11, 2)] = 8.0;
    assert_abs_diff_eq!(
        cross_vis_weights,
        Array2::ones(cross_vis_weights.dim()) * 8.0
    );

    assert_abs_diff_eq!(
        auto_vis[(0, 0)],
        Jones::from([
            5.3557855e4,
            4.3092007e-7,
            -7.420802e2,
            1.3296518e2,
            -7.420802e2,
            -1.3296518e2,
            5.084874e4,
            -6.6132475e-7
        ])
    );
    assert_abs_diff_eq!(
        auto_vis[(0, 2)],
        Jones::from([
            7.1403125e4,
            -1.3957654e-6,
            -1.0667509e3,
            1.01154564e2,
            -1.0667509e3,
            -1.01154564e2,
            6.7373945e4,
            -7.004146e-7
        ])
    );
    assert_abs_diff_eq!(
        auto_vis[(0, 16)],
        Jones::from([
            1.07272586e5,
            1.9233863e-8,
            -1.0756711e3,
            -4.4097336e1,
            -1.0756711e3,
            4.4097336e1,
            1.0141891e5,
            -9.5129735e-7
        ])
    );
    assert_abs_diff_eq!(
        auto_vis[(10, 16)],
        Jones::from([
            1.0766406e5,
            1.5415758e-6,
            -1.334196e3,
            -1.8926495e3,
            -1.334196e3,
            1.8926495e3,
            1.21874336e5,
            2.0712553e-6
        ])
    );

    assert_abs_diff_eq!(
        auto_vis_weights,
        Array2::from_elem(auto_vis_weights.dim(), 8.0)
    );
}

#[test]
#[serial]
fn test_1090008640_calibration_quality() {
    let mut args = get_reduced_1090008640_ms();
    let temp_dir = tempdir().expect("Couldn't make temp dir");
    args.outputs = Some(vec![temp_dir.path().join("hyp_sols.fits")]);
    // To be consistent with other data quality tests, add these flags.
    args.fine_chan_flags = Some(vec![0, 1, 2, 16, 30, 31]);

    let result = args.into_params();
    let params = match result {
        Ok(r) => r,
        Err(e) => panic!("{}", e),
    };

    let cal_vis = get_cal_vis(&params, false).expect("Couldn't read data and generate a model");
    test_1090008640_quality(params, cal_vis);
}

#[test]
#[serial]
fn test_timestep_reading() {
    let temp_dir = tempdir().expect("Couldn't make temp dir");
    let vis_path = temp_dir.path().join("vis.ms");

    let num_timesteps = 10;
    let num_channels = 10;
    let ant_pairs = vec![(0, 1), (0, 2), (1, 2)];

    let obsid = 1090000000;

    let vis_ctx = VisContext {
        num_sel_timesteps: num_timesteps,
        start_timestamp: Epoch::from_gpst_seconds(obsid as f64),
        int_time: Duration::from_f64(1., Unit::Second),
        num_sel_chans: num_channels,
        start_freq_hz: 128_000_000.,
        freq_resolution_hz: 10_000.,
        sel_baselines: ant_pairs,
        avg_time: 1,
        avg_freq: 1,
        num_vis_pols: 4,
    };

    let shape = vis_ctx.sel_dims();

    let vis_data = Array3::<Jones<f32>>::from_shape_fn(shape, |(t, c, b)| {
        let (ant1, ant2) = vis_ctx.sel_baselines[b];
        Jones::from([t as f32, c as f32, ant1 as f32, ant2 as f32, 0., 0., 0., 0.])
    });

    let weight_data = Array3::<f32>::from_elem(shape, 1.);

    let phase_centre = RADec::new_degrees(0., -27.);
    let array_pos = LatLngHeight::new_mwa();
    #[rustfmt::skip]
    let tile_xyzs = vec![
        XyzGeodetic { x: 0., y: 0., z: 0., },
        XyzGeodetic { x: 1., y: 0., z: 0., },
        XyzGeodetic { x: 0., y: 1., z: 0., },
    ];
    let tile_names = vec!["tile_0_0", "tile_1_0", "tile_0_1"];

    let marlu_obs_ctx = MarluObsContext {
        sched_start_timestamp: Epoch::from_gpst_seconds(obsid as f64),
        sched_duration: ((num_timesteps + 1) as f64 * vis_ctx.int_time),
        name: Some(format!("MWA obsid {}", obsid)),
        phase_centre,
        pointing_centre: Some(phase_centre),
        array_pos,
        ant_positions_enh: tile_xyzs
            .iter()
            .map(|xyz| xyz.to_enh(array_pos.latitude_rad))
            .collect(),
        ant_names: tile_names.iter().map(|&s| String::from(s)).collect(),
        field_name: None,
        project_id: None,
        observer: None,
    };
    let (s_lat, c_lat) = array_pos.latitude_rad.sin_cos();
    let ant_positions_xyz = marlu_obs_ctx
        .ant_positions_enh
        .iter()
        .map(|enh| enh.to_xyz_inner(s_lat, c_lat))
        .collect();
    let mut writer = MeasurementSetWriter::new(
        &vis_path,
        phase_centre,
        array_pos,
        ant_positions_xyz,
        Duration::from_total_nanoseconds(0),
    );
    writer.initialize(&vis_ctx, &marlu_obs_ctx, None).unwrap();

    writer
        .write_vis(vis_data.view(), weight_data.view(), &vis_ctx, false)
        .unwrap();

    let ms_reader = MsReader::new::<&PathBuf, &PathBuf>(&vis_path, None).unwrap();
    let ms_ctx = ms_reader.get_obs_context();

    let expected_timestamps = (0..num_timesteps)
        .map(|t| Epoch::from_gpst_seconds((obsid + t) as f64 + 0.5))
        .collect::<Vec<_>>();
    assert_eq!(
        ms_ctx
            .timestamps
            .iter()
            .map(|t| t.as_gpst_seconds())
            .collect::<Vec<_>>(),
        expected_timestamps
            .iter()
            .map(|t| t.as_gpst_seconds())
            .collect::<Vec<_>>()
    );
}

#[test]
#[serial]
fn test_trunc_data() {
    let metafits: Option<&str> = None;
    let expected_num_tiles = 128;
    let expected_unavailable_tiles = (2..128).into_iter().collect::<Vec<usize>>();

    let result = MsReader::new(
        "test_files/1090008640/1090008640_cotter_trunc_autos.ms",
        metafits,
    );
    assert!(result.is_ok(), "{:?}", result.err());
    let reader = result.unwrap();
    assert!(reader.obs_context.autocorrelations_present);
    assert_eq!(reader.obs_context.get_total_num_tiles(), expected_num_tiles);
    assert_eq!(reader.obs_context.get_num_unflagged_tiles(), 2);
    assert_eq!(
        &reader.obs_context.unavailable_tiles,
        &expected_unavailable_tiles
    );
    assert_eq!(
        &reader.obs_context.flagged_tiles,
        &expected_unavailable_tiles
    );
    assert_eq!(&reader.obs_context.all_timesteps, &[0, 1, 2]);
    assert_eq!(&reader.obs_context.unflagged_timesteps, &[2]);

    let result = MsReader::new(
        "test_files/1090008640/1090008640_cotter_trunc_noautos.ms",
        metafits,
    );
    assert!(result.is_ok(), "{:?}", result.err());
    let reader = result.unwrap();
    assert!(!reader.obs_context.autocorrelations_present);
    assert_eq!(reader.obs_context.get_total_num_tiles(), expected_num_tiles);
    assert_eq!(reader.obs_context.get_num_unflagged_tiles(), 2);
    assert_eq!(
        &reader.obs_context.unavailable_tiles,
        &expected_unavailable_tiles
    );
    assert_eq!(
        &reader.obs_context.flagged_tiles,
        &expected_unavailable_tiles
    );
    assert_eq!(&reader.obs_context.all_timesteps, &[0, 1, 2]);
    assert_eq!(&reader.obs_context.unflagged_timesteps, &[2]);

    let result = MsReader::new("test_files/1090008640/1090008640_birli_trunc.ms", metafits);
    assert!(result.is_ok(), "{:?}", result.err());
    let reader = result.unwrap();
    assert!(reader.obs_context.autocorrelations_present);
    assert_eq!(reader.obs_context.get_total_num_tiles(), expected_num_tiles);
    assert_eq!(reader.obs_context.get_num_unflagged_tiles(), 2);
    assert_eq!(
        &reader.obs_context.unavailable_tiles,
        &expected_unavailable_tiles
    );
    assert_eq!(
        &reader.obs_context.flagged_tiles,
        &expected_unavailable_tiles
    );
    assert_eq!(&reader.obs_context.all_timesteps, &[0, 1, 2]);
    assert_eq!(&reader.obs_context.unflagged_timesteps, &[1, 2]);

    // Test that attempting to use all tiles still results in only 2 tiles being available.
    let mut args = get_reduced_1090008640_ms();
    let temp_dir = tempdir().expect("Couldn't make temp dir");
    match args.data.as_mut() {
        Some(d) => d[1] = "test_files/1090008640/1090008640_birli_trunc.ms".to_string(),
        None => unreachable!(),
    }
    args.outputs = Some(vec![temp_dir.path().join("hyp_sols.fits")]);
    args.ignore_input_data_tile_flags = true;
    let result = args.into_params();
    assert!(result.is_ok(), "{:?}", result.err());
    let params = result.unwrap();

    assert_eq!(
        params.tile_baseline_flags.flagged_tiles.len(),
        expected_unavailable_tiles.len()
    );
}
