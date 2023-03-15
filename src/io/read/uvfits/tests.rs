// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

use std::{collections::HashSet, ffi::CString, path::PathBuf};

use approx::{assert_abs_diff_eq, assert_abs_diff_ne};
use fitsio::errors::check_status as fits_check_status;
use hifitime::Duration;
use itertools::Itertools;
use marlu::{c32, LatLngHeight, RADec, UvfitsWriter, VisContext, VisWrite, XyzGeodetic};
use ndarray::prelude::*;
use tempfile::{tempdir, NamedTempFile};

use super::*;
use crate::{
    di_calibrate::{get_cal_vis, tests::test_1090008640_quality},
    math::TileBaselineFlags,
    tests::reduced_obsids::get_reduced_1090008640_uvfits,
};

// TODO(dev): move these to Marlu
fn write_then_read_uvfits(autos: bool) {
    let output = NamedTempFile::new().expect("Couldn't create temporary file");
    let phase_centre = RADec::from_degrees(0.0, -27.0);
    let timesteps = [Epoch::from_gpst_seconds(1065880128.0)];
    let num_timesteps = timesteps.len();
    let num_tiles = 128;
    let autocorrelations_present = autos;
    let fine_chan_width_hz = 80000.0;
    let num_chans = 16;
    let fine_chan_freqs_hz: Vec<f64> = (0..num_chans)
        .map(|i| 150e6 + fine_chan_width_hz * i as f64)
        .collect();

    let (tile_names, xyzs): (Vec<String>, Vec<XyzGeodetic>) = (0..num_tiles)
        .map(|i| {
            (
                format!("Tile{i}"),
                XyzGeodetic {
                    x: 1.0 * i as f64,
                    y: 2.0 * i as f64,
                    z: 3.0 * i as f64,
                },
            )
        })
        .unzip();

    let flagged_tiles = HashSet::new();

    let all_ant_pairs: Vec<(usize, usize)> = (0..num_tiles)
        .flat_map(|tile1| {
            let start_tile2 = if autocorrelations_present {
                tile1
            } else {
                tile1 + 1
            };
            (start_tile2..num_tiles)
                .map(|tile2| (tile1, tile2))
                .collect::<Vec<_>>()
        })
        .collect();

    // XXX(dev): test unflagged?
    // let unflagged_ant_pairs: Vec<(usize, usize)> = all_ant_pairs
    //     .iter()
    //     .filter(|(tile1, tile2)| !flagged_tiles.contains(tile1) && !flagged_tiles.contains(tile2))
    //     .cloned()
    //     .collect();

    let sel_ant_pairs = if autocorrelations_present {
        all_ant_pairs
    } else {
        all_ant_pairs
            .into_iter()
            .filter(|&(tile1, tile2)| tile1 != tile2)
            .collect()
    };
    let num_cross_baselines = (num_tiles * (num_tiles - 1)) / 2;

    let flagged_fine_chans = HashSet::new();
    let tile_baseline_flags = TileBaselineFlags::new(num_tiles, flagged_tiles);

    let array_pos = LatLngHeight::mwa();

    // Just in case this gets accidentally changed.
    assert_eq!(
        num_timesteps, 1,
        "num_timesteps should always be 1 for this test"
    );

    let vis_ctx = VisContext {
        num_sel_timesteps: num_timesteps,
        start_timestamp: timesteps[0],
        int_time: Duration::from_seconds(1.),
        num_sel_chans: num_chans,
        start_freq_hz: fine_chan_freqs_hz[0],
        freq_resolution_hz: fine_chan_width_hz,
        sel_baselines: sel_ant_pairs,
        avg_time: 1,
        avg_freq: 1,
        num_vis_pols: 4,
    };

    let sel_vis = Array3::from_shape_fn(
        vis_ctx.avg_dims(),
        |(timestep_idx, ant_pair_idx, chan_idx)| {
            let (tile1, tile2) = vis_ctx.sel_baselines[ant_pair_idx];
            Jones::from([
                c32::new(tile1 as _, tile2 as _),
                c32::new(chan_idx as _, timestep_idx as _),
                c32::new(1.0, 1.0),
                c32::new(1.0, 1.0),
            ])
        },
    );

    let sel_weights = Array3::from_shape_fn(vis_ctx.avg_dims(), |(_, ant_pair_idx, chan_idx)| {
        (chan_idx + num_chans * ant_pair_idx) as f32
    });

    let result = UvfitsWriter::from_marlu(
        output.path(),
        &vis_ctx,
        array_pos,
        phase_centre,
        Duration::from_seconds(0.0),
        None,
        tile_names,
        xyzs,
        true,
        None,
    );
    assert!(result.is_ok(), "Failed to create new uvfits file");
    let mut output_writer = result.unwrap();

    let result = output_writer.write_vis(sel_vis.view(), sel_weights.view(), &vis_ctx);
    assert!(
        result.is_ok(),
        "Failed to write visibilities to uvfits file: {:?}",
        result.unwrap_err()
    );
    result.unwrap();

    let result = output_writer.finalise();
    assert!(
        result.is_ok(),
        "Failed to finish writing uvfits file: {:?}",
        result.unwrap_err()
    );
    result.unwrap();

    // Inspect the file for sanity's sake!
    let metafits: Option<&str> = None;
    let result = UvfitsReader::new(output.path(), metafits);
    assert!(
        result.is_ok(),
        "Failed to read the just-created uvfits file"
    );
    let uvfits = result.unwrap();

    let mut cross_vis_read = Array2::zeros((num_chans, num_cross_baselines));
    let mut cross_weights_read = Array2::zeros((num_chans, num_cross_baselines));
    let mut auto_vis_read = Array2::zeros((num_chans, num_tiles));
    let mut auto_weights_read = Array2::zeros((num_chans, num_tiles));

    for ((timestep, vis_written), weights_written) in (0..num_timesteps)
        .zip_eq(sel_vis.outer_iter())
        .zip_eq(sel_weights.outer_iter())
    {
        let result = uvfits.read_crosses(
            cross_vis_read.view_mut(),
            cross_weights_read.view_mut(),
            timestep,
            &tile_baseline_flags,
            &flagged_fine_chans,
        );
        assert!(
            result.is_ok(),
            "Failed to read crosses from the just-created uvfits file: {:?}",
            result.unwrap_err()
        );
        result.unwrap();

        vis_written
            .axis_iter(Axis(1))
            .zip_eq(weights_written.axis_iter(Axis(1)))
            .zip_eq(vis_ctx.sel_baselines.iter())
            .filter(|((_, _), (tile1, tile2))| tile1 != tile2)
            .zip_eq(cross_vis_read.axis_iter(Axis(1)))
            .zip_eq(cross_weights_read.axis_iter(Axis(1)))
            .for_each(
                |((((vis_written, weights_written), _), cross_vis_read), cross_weights_read)| {
                    assert_abs_diff_eq!(cross_vis_read, vis_written);
                    assert_abs_diff_eq!(cross_weights_read, weights_written);
                },
            );

        if autocorrelations_present {
            let result = uvfits.read_autos(
                auto_vis_read.view_mut(),
                auto_weights_read.view_mut(),
                timestep,
                &tile_baseline_flags,
                &flagged_fine_chans,
            );
            assert!(
                result.is_ok(),
                "Failed to read autos from the just-created uvfits file: {:?}",
                result.unwrap_err()
            );
            result.unwrap();

            vis_written
                .axis_iter(Axis(1))
                .zip_eq(weights_written.axis_iter(Axis(1)))
                .zip_eq(vis_ctx.sel_baselines.iter())
                .filter(|((_, _), (tile1, tile2))| tile1 == tile2)
                .zip_eq(auto_vis_read.axis_iter(Axis(1)))
                .zip_eq(auto_weights_read.axis_iter(Axis(1)))
                .for_each(
                    |((((vis_written, weights_written), _), auto_vis_read), auto_weights_read)| {
                        assert_abs_diff_eq!(auto_vis_read, vis_written);
                        assert_abs_diff_eq!(auto_weights_read, weights_written);
                    },
                );
        }
    }
}

#[test]
fn uvfits_io_works_for_cross_correlations() {
    write_then_read_uvfits(false)
}

#[test]
fn uvfits_io_works_for_auto_correlations() {
    write_then_read_uvfits(true)
}

#[test]
fn test_1090008640_cross_vis() {
    let args = get_reduced_1090008640_uvfits();
    let uvfits_reader = if let [metafits, uvfits] = &args.data.unwrap()[..] {
        match UvfitsReader::new(uvfits, Some(metafits)) {
            Ok(u) => u,
            Err(e) => panic!("{}", e),
        }
    } else {
        panic!("There weren't 2 elements in args.data");
    };

    let obs_context = &uvfits_reader.obs_context;
    let total_num_tiles = obs_context.tile_xyzs.len();
    let num_baselines = (total_num_tiles * (total_num_tiles - 1)) / 2;
    let num_chans = obs_context.num_fine_chans_per_coarse_chan.unwrap().get();
    let tile_baseline_flags = TileBaselineFlags::new(total_num_tiles, HashSet::new());

    assert_abs_diff_eq!(
        obs_context.timestamps.first().to_gpst_seconds(),
        1090008658.0
    );

    let mut vis = Array2::zeros((num_chans, num_baselines));
    let mut vis_weights = Array2::zeros((num_chans, num_baselines));
    let result = uvfits_reader.read_crosses(
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
        vis[(16, 10)],
        Jones::from([
            c32::new(-4.138127e1, -2.638188e2),
            c32::new(5.220332e2, -2.6055228e2),
            c32::new(4.854074e2, -1.9634505e2),
            c32::new(1.6101791e1, -4.4489478e2),
        ])
    );

    // PFB gains will affect weights, but these weren't in Birli when it made
    // this MS; all but one weight are 8.0 (it's flagged).
    assert_abs_diff_eq!(vis_weights[(2, 11)], -8.0);
    // Undo the flag and test all values.
    vis_weights[(2, 11)] = 8.0;
    assert_abs_diff_eq!(vis_weights, Array2::ones(vis_weights.dim()) * 8.0);
}

#[test]
fn test_1090008640_auto_vis() {
    let args = get_reduced_1090008640_uvfits();
    let uvfits_reader = if let [metafits, uvfits] = &args.data.unwrap()[..] {
        match UvfitsReader::new(uvfits, Some(metafits)) {
            Ok(u) => u,
            Err(e) => panic!("{}", e),
        }
    } else {
        panic!("There weren't 2 elements in args.data");
    };

    let obs_context = &uvfits_reader.obs_context;
    let total_num_tiles = obs_context.get_total_num_tiles();
    let num_chans = obs_context.num_fine_chans_per_coarse_chan.unwrap().get();
    let tile_baseline_flags = TileBaselineFlags::new(total_num_tiles, HashSet::new());

    assert_abs_diff_eq!(
        obs_context.timestamps.first().to_gpst_seconds(),
        1090008658.0
    );

    let mut vis = Array2::zeros((num_chans, total_num_tiles));
    let mut vis_weights = Array2::zeros((num_chans, total_num_tiles));
    let result = uvfits_reader.read_autos(
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
        vis[(2, 0)],
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
        vis[(16, 0)],
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
        vis[(16, 10)],
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
fn test_1090008640_auto_vis_with_flags() {
    let args = get_reduced_1090008640_uvfits();
    let uvfits_reader = if let [metafits, uvfits] = &args.data.unwrap()[..] {
        match UvfitsReader::new(uvfits, Some(metafits)) {
            Ok(u) => u,
            Err(e) => panic!("{}", e),
        }
    } else {
        panic!("There weren't 2 elements in args.data");
    };

    let obs_context = &uvfits_reader.obs_context;
    let total_num_tiles = obs_context.get_total_num_tiles();
    let num_chans = obs_context.num_fine_chans_per_coarse_chan.unwrap().get();
    let flagged_tiles = HashSet::from([1, 9]);
    let num_unflagged_tiles = total_num_tiles - flagged_tiles.len();
    let chan_flags = HashSet::from([1]);
    let num_unflagged_chans = num_chans - chan_flags.len();
    let tile_baseline_flags = TileBaselineFlags::new(total_num_tiles, flagged_tiles);

    assert_abs_diff_eq!(
        obs_context.timestamps.first().to_gpst_seconds(),
        1090008658.0
    );

    let mut vis = Array2::zeros((num_unflagged_chans, num_unflagged_tiles));
    let mut vis_weights = Array2::zeros((num_unflagged_chans, num_unflagged_tiles));
    let result = uvfits_reader.read_autos(
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
        vis[(1, 0)],
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
        vis[(15, 0)],
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
        vis[(15, 8)],
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
fn read_1090008640_cross_and_auto_vis() {
    let args = get_reduced_1090008640_uvfits();
    let uvfits_reader = if let [metafits, uvfits] = &args.data.unwrap()[..] {
        match UvfitsReader::new(uvfits, Some(metafits)) {
            Ok(u) => u,
            Err(e) => panic!("{}", e),
        }
    } else {
        panic!("There weren't 2 elements in args.data");
    };

    let obs_context = &uvfits_reader.obs_context;
    let total_num_tiles = obs_context.get_total_num_tiles();
    let num_baselines = (total_num_tiles * (total_num_tiles - 1)) / 2;
    let num_chans = obs_context.num_fine_chans_per_coarse_chan.unwrap().get();
    let tile_baseline_flags = TileBaselineFlags::new(total_num_tiles, HashSet::new());

    assert_abs_diff_eq!(
        obs_context.timestamps.first().to_gpst_seconds(),
        1090008658.0
    );

    let mut cross_vis = Array2::zeros((num_chans, num_baselines));
    let mut cross_vis_weights = Array2::zeros((num_chans, num_baselines));
    let mut auto_vis = Array2::zeros((num_chans, total_num_tiles));
    let mut auto_vis_weights = Array2::zeros((num_chans, total_num_tiles));
    let result = uvfits_reader.read_crosses_and_autos(
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
        cross_vis[(16, 10)],
        Jones::from([
            c32::new(-4.138127e1, -2.638188e2),
            c32::new(5.220332e2, -2.6055228e2),
            c32::new(4.854074e2, -1.9634505e2),
            c32::new(1.6101791e1, -4.4489478e2),
        ])
    );

    assert_abs_diff_eq!(cross_vis_weights[(2, 11)], -8.0);
    cross_vis_weights[(2, 11)] = 8.0;
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
        auto_vis[(2, 0)],
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
        auto_vis[(16, 0)],
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
        auto_vis[(16, 10)],
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
fn test_1090008640_calibration_quality() {
    let mut args = get_reduced_1090008640_uvfits();
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
fn test_timestep_reading() {
    let temp_dir = tempdir().expect("Couldn't make temp dir");
    let vis_path = temp_dir.path().join("vis.uvfits");
    // uncomment this to write to tmp instead
    // let vis_path = PathBuf::from("/tmp/vis.uvfits");

    let num_timesteps = 10;
    let num_channels = 10;
    let ant_pairs = vec![(0, 1), (0, 2), (1, 2)];

    let obsid = 1090000000;

    let vis_ctx = VisContext {
        num_sel_timesteps: num_timesteps,
        start_timestamp: Epoch::from_gpst_seconds(obsid as f64),
        int_time: Duration::from_seconds(1.),
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

    let phase_centre = RADec::from_degrees(0., -27.);
    let array_pos = LatLngHeight::mwa();
    #[rustfmt::skip]
    let tile_xyzs = vec![
        XyzGeodetic { x: 0., y: 0., z: 0., },
        XyzGeodetic { x: 1., y: 0., z: 0., },
        XyzGeodetic { x: 0., y: 1., z: 0., },
    ];
    let tile_names = vec!["tile_0_0".into(), "tile_1_0".into(), "tile_0_1".into()];

    let mut writer = UvfitsWriter::from_marlu(
        &vis_path,
        &vis_ctx,
        array_pos,
        phase_centre,
        Duration::from_seconds(0.0),
        Some(&format!("synthesized test data {obsid}")),
        tile_names,
        tile_xyzs,
        true,
        None,
    )
    .unwrap();

    writer
        .write_vis(vis_data.view(), weight_data.view(), &vis_ctx)
        .unwrap();

    writer.finalise().unwrap();

    let uvfits_reader = UvfitsReader::new::<&PathBuf, &PathBuf>(&vis_path, None).unwrap();
    let uvfits_ctx = uvfits_reader.get_obs_context();

    let expected_timestamps = (0..num_timesteps)
        .map(|t| Epoch::from_gpst_seconds((obsid + t) as f64 + 0.5))
        .collect::<Vec<_>>();
    assert_eq!(
        uvfits_ctx
            .timestamps
            .iter()
            .map(|t| t.to_gpst_seconds())
            .collect::<Vec<_>>(),
        expected_timestamps
            .iter()
            .map(|t| t.to_gpst_seconds())
            .collect::<Vec<_>>()
    );
}

#[test]
fn test_map_metafits_antenna_order() {
    // First, check the delays and gains of the existing test data. Because this
    // uvfits file has its tiles in the same order as the "metafits order", the
    // delays and gains are already correct without re-ordering.
    let metafits_path = "test_files/1090008640/1090008640.metafits";
    let uvfits = UvfitsReader::new(
        "test_files/1090008640/1090008640.uvfits",
        Some(metafits_path),
    )
    .unwrap();
    let obs_context = uvfits.get_obs_context();
    let delays = match obs_context.dipole_delays.as_ref() {
        Some(Delays::Full(d)) => d,
        _ => unreachable!(),
    };
    // All delays should be 0.
    assert_eq!(delays, Array2::from_elem(delays.dim(), 0));
    // Keep the true gains for later.
    let gains = match obs_context.dipole_gains.as_ref() {
        Some(g) => g,
        _ => unreachable!(),
    };

    // Test that the dipole delays/gains get mapped correctly. As the test
    // uvfits file is already in the same order as the metafits file, the
    // easiest thing to do is to modify the metafits file.
    let metafits = tempfile::NamedTempFile::new().expect("couldn't make a temp file");
    std::fs::copy(metafits_path, metafits.path()).unwrap();
    unsafe {
        let metafits = CString::new(metafits.path().display().to_string())
            .unwrap()
            .into_raw();
        let mut fptr = std::ptr::null_mut();
        let mut status = 0;

        // ffopen = fits_open_file
        fitsio_sys::ffopen(
            &mut fptr,   /* O - FITS file pointer                   */
            metafits,    /* I - full name of file to open           */
            1,           /* I - 0 = open readonly; 1 = read/write   */
            &mut status, /* IO - error status                       */
        );
        fits_check_status(status).unwrap();
        drop(CString::from_raw(metafits));
        // ffmahd = fits_movabs_hdu
        fitsio_sys::ffmahd(
            fptr,                 /* I - FITS file pointer             */
            2,                    /* I - number of the HDU to move to  */
            std::ptr::null_mut(), /* O - type of extension, 0, 1, or 2 */
            &mut status,          /* IO - error status                 */
        );
        fits_check_status(status).unwrap();

        // Swap Tile011 (rows 87 and 88) with Tile017 (rows 91 and 92), as
        // Tile017 has a dead dipole but Tile011 doesn't.
        let mut tile_name = CString::new("Tile017").unwrap().into_raw();
        // ffpcls = fits_write_col_str
        fitsio_sys::ffpcls(
            fptr,           /* I - FITS file pointer                       */
            4,              /* I - number of column to write (1 = 1st col) */
            87,             /* I - first row to write (1 = 1st row)        */
            1,              /* I - first vector element to write (1 = 1st) */
            1,              /* I - number of strings to write              */
            &mut tile_name, /* I - array of pointers to strings            */
            &mut status,    /* IO - error status                           */
        );
        fits_check_status(status).unwrap();
        fitsio_sys::ffpcls(
            fptr,           /* I - FITS file pointer                       */
            4,              /* I - number of column to write (1 = 1st col) */
            88,             /* I - first row to write (1 = 1st row)        */
            1,              /* I - first vector element to write (1 = 1st) */
            1,              /* I - number of strings to write              */
            &mut tile_name, /* I - array of pointers to strings            */
            &mut status,    /* IO - error status                           */
        );
        fits_check_status(status).unwrap();
        drop(CString::from_raw(tile_name));

        let mut tile_name = CString::new("Tile011").unwrap().into_raw();
        fitsio_sys::ffpcls(
            fptr,           /* I - FITS file pointer                       */
            4,              /* I - number of column to write (1 = 1st col) */
            91,             /* I - first row to write (1 = 1st row)        */
            1,              /* I - first vector element to write (1 = 1st) */
            1,              /* I - number of strings to write              */
            &mut tile_name, /* I - array of pointers to strings            */
            &mut status,    /* IO - error status                           */
        );
        fits_check_status(status).unwrap();
        fitsio_sys::ffpcls(
            fptr,           /* I - FITS file pointer                       */
            4,              /* I - number of column to write (1 = 1st col) */
            92,             /* I - first row to write (1 = 1st row)        */
            1,              /* I - first vector element to write (1 = 1st) */
            1,              /* I - number of strings to write              */
            &mut tile_name, /* I - array of pointers to strings            */
            &mut status,    /* IO - error status                           */
        );
        fits_check_status(status).unwrap();
        drop(CString::from_raw(tile_name));

        // ffclos = fits_close_file
        fitsio_sys::ffclos(fptr, &mut status);
        fits_check_status(status).unwrap();
    }

    let uvfits = UvfitsReader::new(
        "test_files/1090008640/1090008640.uvfits",
        Some(metafits.path()),
    )
    .unwrap();
    let obs_context = uvfits.get_obs_context();
    let delays = match obs_context.dipole_delays.as_ref() {
        Some(Delays::Full(d)) => d,
        _ => unreachable!(),
    };
    // All delays should be 0.
    assert_eq!(delays, Array2::from_elem(delays.dim(), 0));
    let mut perturbed_gains = match obs_context.dipole_gains.as_ref() {
        Some(g) => g.clone(),
        _ => unreachable!(),
    };

    // If the gains are mapped correctly, the before and after gains are
    // different.
    assert_abs_diff_ne!(gains, &perturbed_gains);

    // The first tile's gains of the perturbed metafits
    // (corresponding to Tile017) will have a dead dipole.
    assert_eq!(
        perturbed_gains.slice(s![0, ..]).as_slice().unwrap(),
        [
            1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0,
            1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0
        ]
    );
    // Tile011 is all 1s.
    assert_eq!(
        perturbed_gains.slice(s![6, ..]).as_slice().unwrap(),
        &[1.0; 32]
    );

    // Confirm that the gains are equal when the gain rows are swapped.
    let row1 = perturbed_gains.slice(s![0, ..]).into_owned();
    let row2 = perturbed_gains.slice(s![6, ..]).into_owned();
    perturbed_gains.slice_mut(s![0, ..]).assign(&row2);
    perturbed_gains.slice_mut(s![6, ..]).assign(&row1);
    assert_abs_diff_eq!(gains, &perturbed_gains);

    // Test that the dipole delays/gains aren't mapped when an unknown tile name
    // is encountered.
    let metafits = tempfile::NamedTempFile::new().expect("couldn't make a temp file");
    std::fs::copy(metafits_path, metafits.path()).unwrap();
    unsafe {
        let metafits = CString::new(metafits.path().display().to_string())
            .unwrap()
            .into_raw();
        let mut fptr = std::ptr::null_mut();
        let mut status = 0;

        // ffopen = fits_open_file
        fitsio_sys::ffopen(
            &mut fptr,   /* O - FITS file pointer                   */
            metafits,    /* I - full name of file to open           */
            1,           /* I - 0 = open readonly; 1 = read/write   */
            &mut status, /* IO - error status                       */
        );
        fits_check_status(status).unwrap();
        drop(CString::from_raw(metafits));
        // ffmahd = fits_movabs_hdu
        fitsio_sys::ffmahd(
            fptr,                 /* I - FITS file pointer             */
            2,                    /* I - number of the HDU to move to  */
            std::ptr::null_mut(), /* O - type of extension, 0, 1, or 2 */
            &mut status,          /* IO - error status                 */
        );
        fits_check_status(status).unwrap();

        let mut tile_name = CString::new("darkness").unwrap().into_raw();
        // ffpcls = fits_write_col_str
        fitsio_sys::ffpcls(
            fptr,           /* I - FITS file pointer                       */
            4,              /* I - number of column to write (1 = 1st col) */
            87,             /* I - first row to write (1 = 1st row)        */
            1,              /* I - first vector element to write (1 = 1st) */
            1,              /* I - number of strings to write              */
            &mut tile_name, /* I - array of pointers to strings            */
            &mut status,    /* IO - error status                           */
        );
        fits_check_status(status).unwrap();
        fitsio_sys::ffpcls(
            fptr,           /* I - FITS file pointer                       */
            4,              /* I - number of column to write (1 = 1st col) */
            88,             /* I - first row to write (1 = 1st row)        */
            1,              /* I - first vector element to write (1 = 1st) */
            1,              /* I - number of strings to write              */
            &mut tile_name, /* I - array of pointers to strings            */
            &mut status,    /* IO - error status                           */
        );
        fits_check_status(status).unwrap();
        drop(CString::from_raw(tile_name));

        // ffclos = fits_close_file
        fitsio_sys::ffclos(fptr, &mut status);
        fits_check_status(status).unwrap();
    }

    let uvfits = UvfitsReader::new(
        "test_files/1090008640/1090008640.uvfits",
        Some(metafits.path()),
    )
    .unwrap();
    let obs_context = uvfits.get_obs_context();
    let delays = match obs_context.dipole_delays.as_ref() {
        Some(Delays::Full(d)) => d,
        _ => unreachable!(),
    };
    // All delays should be 0.
    assert_eq!(delays, Array2::from_elem(delays.dim(), 0));
    let perturbed_gains = match obs_context.dipole_gains.as_ref() {
        Some(g) => g,
        _ => unreachable!(),
    };
    // The gains should be the same as the unaltered-metafits case.
    assert_abs_diff_eq!(gains, perturbed_gains);
}
