// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

use std::{collections::HashSet, path::PathBuf};

use approx::assert_abs_diff_eq;
use hifitime::{Duration, Unit};
use itertools::izip;
use marlu::{c32, LatLngHeight, RADec, UvfitsWriter, VisContext, VisWritable, XyzGeodetic};
use ndarray::prelude::*;
use tempfile::{tempdir, NamedTempFile};

use super::*;
use crate::{
    calibrate::di::code::{get_cal_vis, tests::test_1090008640_quality},
    jones_test::TestJones,
    math::TileBaselineMaps,
    tests::reduced_obsids::get_reduced_1090008640_uvfits,
};
use mwa_hyperdrive_common::{hifitime, itertools, marlu, ndarray};

// TODO(dev): move these to Marlu
fn write_then_read_uvfits(autos: bool) {
    let output = NamedTempFile::new().expect("Couldn't create temporary file");
    let phase_centre = RADec::new_degrees(0.0, -27.0);
    let timesteps = [Epoch::from_gpst_seconds(1065880128.0)];
    let num_timesteps = timesteps.len();
    let num_tiles = 128;
    let autocorrelations_present = autos;
    let fine_chan_width_hz = 80000.0;
    let num_chans = 16;
    let fine_chan_freqs_hz: Vec<f64> = (0..num_chans)
        .into_iter()
        .map(|i| 150e6 + fine_chan_width_hz * i as f64)
        .collect();

    let (tile_names, xyzs): (Vec<String>, Vec<XyzGeodetic>) = (0..num_tiles)
        .into_iter()
        .map(|i| {
            (
                format!("Tile{}", i),
                XyzGeodetic {
                    x: 1.0 * i as f64,
                    y: 2.0 * i as f64,
                    z: 3.0 * i as f64,
                },
            )
        })
        .unzip();

    let flagged_tiles = vec![];

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

    let cross_ant_pairs: Vec<_> = all_ant_pairs
        .clone()
        .into_iter()
        .filter(|&(tile1, tile2)| tile1 != tile2)
        .collect();

    let num_cross_baselines = cross_ant_pairs.len();

    let sel_ant_pairs = if autocorrelations_present {
        all_ant_pairs
    } else {
        cross_ant_pairs
    };

    let flagged_fine_chans = HashSet::new();
    let maps = TileBaselineMaps::new(num_tiles, &flagged_tiles);

    let array_pos = LatLngHeight::new_mwa();

    // Just in case this gets accidentally changed.
    assert_eq!(
        num_timesteps, 1,
        "num_timesteps should always be 1 for this test"
    );

    let vis_ctx = VisContext {
        num_sel_timesteps: num_timesteps,
        start_timestamp: *timesteps.first().unwrap(),
        int_time: Duration::from_f64(1., Unit::Second),
        num_sel_chans: num_chans,
        start_freq_hz: fine_chan_freqs_hz[0],
        freq_resolution_hz: fine_chan_width_hz,
        sel_baselines: sel_ant_pairs.clone(),
        avg_time: 1,
        avg_freq: 1,
        num_vis_pols: 4,
    };

    let sel_vis = Array3::from_shape_fn(
        vis_ctx.avg_dims(),
        |(timestep_idx, ant_pair_idx, chan_idx)| {
            let (tile1, tile2) = sel_ant_pairs[ant_pair_idx];
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

    let result =
        UvfitsWriter::from_marlu(output.path(), &vis_ctx, Some(array_pos), phase_centre, None);
    assert!(result.is_ok(), "Failed to create new uvfits file");
    let mut output_writer = result.unwrap();

    let result =
        output_writer.write_vis_marlu(sel_vis.view(), sel_weights.view(), &vis_ctx, &xyzs, false);
    assert!(
        result.is_ok(),
        "Failed to write visibilities to uvfits file: {:?}",
        result.unwrap_err()
    );
    result.unwrap();

    let result = output_writer.write_uvfits_antenna_table(&tile_names, &xyzs);
    assert!(
        result.is_ok(),
        "Failed to finish writing uvfits file: {:?}",
        result.unwrap_err()
    );
    result.unwrap();

    // Inspect the file for sanity's sake!
    let metafits: Option<&str> = None;
    let result = UvfitsReader::new(&output.path(), metafits);
    assert!(
        result.is_ok(),
        "Failed to read the just-created uvfits file"
    );
    let uvfits = result.unwrap();

    let mut cross_vis_read = Array2::zeros((num_cross_baselines, num_chans));
    let mut cross_weights_read = Array2::zeros((num_cross_baselines, num_chans));
    let mut auto_vis_read = Array2::zeros((num_tiles, num_chans));
    let mut auto_weights_read = Array2::zeros((num_tiles, num_chans));

    for (timestep, vis_written, weights_written) in izip!(
        0..num_timesteps,
        sel_vis.outer_iter(),
        sel_weights.outer_iter()
    ) {
        let result = uvfits.read_crosses(
            cross_vis_read.view_mut(),
            cross_weights_read.view_mut(),
            timestep,
            &maps.tile_to_unflagged_cross_baseline_map,
            &flagged_fine_chans,
        );
        assert!(
            result.is_ok(),
            "Failed to read crosses from the just-created uvfits file: {:?}",
            result.unwrap_err()
        );
        result.unwrap();

        for ((vis_written, weights_written, _), vis_read, weights_read) in izip!(
            izip!(
                vis_written.axis_iter(Axis(1)),
                weights_written.axis_iter(Axis(1)),
                sel_ant_pairs.clone().into_iter()
            )
            .filter(|(_, _, (tile1, tile2))| { tile1 != tile2 }),
            cross_vis_read.outer_iter(),
            cross_weights_read.outer_iter()
        ) {
            assert_abs_diff_eq!(
                vis_read.mapv(TestJones::from),
                vis_written.mapv(TestJones::from)
            );
            assert_abs_diff_eq!(weights_read, weights_written);
        }

        if autocorrelations_present {
            let result = uvfits.read_autos(
                auto_vis_read.view_mut(),
                auto_weights_read.view_mut(),
                timestep,
                &flagged_tiles,
                &flagged_fine_chans,
            );
            assert!(
                result.is_ok(),
                "Failed to read autos from the just-created uvfits file: {:?}",
                result.unwrap_err()
            );
            result.unwrap();

            for ((vis_written, weights_written, _), vis_read, weights_read) in izip!(
                izip!(
                    vis_written.axis_iter(Axis(1)),
                    weights_written.axis_iter(Axis(1)),
                    sel_ant_pairs.clone().into_iter()
                )
                .filter(|(_, _, (tile1, tile2))| { tile1 == tile2 }),
                auto_vis_read.outer_iter(),
                auto_weights_read.outer_iter()
            ) {
                assert_abs_diff_eq!(
                    vis_read.mapv(TestJones::from),
                    vis_written.mapv(TestJones::from)
                );
                assert_abs_diff_eq!(weights_read, weights_written);
            }
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
    let num_chans = obs_context.num_fine_chans_per_coarse_chan;
    let TileBaselineMaps {
        tile_to_unflagged_cross_baseline_map,
        ..
    } = TileBaselineMaps::new(total_num_tiles, &[]);

    assert_abs_diff_eq!(
        obs_context.timestamps.first().as_gpst_seconds(),
        1090008658.0
    );

    let mut vis = Array2::zeros((num_baselines, num_chans));
    let mut vis_weights = Array2::zeros((num_baselines, num_chans));
    let result = uvfits_reader.read_crosses(
        vis.view_mut(),
        vis_weights.view_mut(),
        *obs_context.all_timesteps.first(),
        &tile_to_unflagged_cross_baseline_map,
        &HashSet::new(),
    );
    assert!(result.is_ok(), "{}", result.unwrap_err());

    // These values are exactly the same as the raw data when all corrections
    // (except the PFB gains) are turned on. See the
    // read_1090008640_cross_vis_with_corrections test.
    assert_abs_diff_eq!(
        TestJones::from(vis[(0, 0)]),
        TestJones::from([
            c32::new(-1.2564129e2, -1.497961e1),
            c32::new(8.207059e1, -1.4936417e2),
            c32::new(-7.306871e1, 2.36177e2),
            c32::new(-5.5305626e1, -2.3209404e1)
        ])
    );
    assert_abs_diff_eq!(
        TestJones::from(vis[(10, 16)]),
        TestJones::from([
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
    let num_chans = obs_context.num_fine_chans_per_coarse_chan;

    assert_abs_diff_eq!(
        obs_context.timestamps.first().as_gpst_seconds(),
        1090008658.0
    );

    let mut vis = Array2::zeros((total_num_tiles, num_chans));
    let mut vis_weights = Array2::zeros((total_num_tiles, num_chans));
    let result = uvfits_reader.read_autos(
        vis.view_mut(),
        vis_weights.view_mut(),
        *obs_context.all_timesteps.first(),
        &[],
        &HashSet::new(),
    );
    assert!(result.is_ok(), "{}", result.unwrap_err());

    assert_abs_diff_eq!(
        TestJones::from(vis[(0, 0)]),
        TestJones::from([
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
        TestJones::from(vis[(0, 2)]),
        TestJones::from([
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
        TestJones::from(vis[(0, 16)]),
        TestJones::from([
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
        TestJones::from(vis[(10, 16)]),
        TestJones::from([
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
    let num_chans = obs_context.num_fine_chans_per_coarse_chan;
    let tile_flags = [1, 9];
    let num_unflagged_tiles = total_num_tiles - tile_flags.len();
    let chan_flags = HashSet::from([1]);
    let num_unflagged_chans = num_chans - chan_flags.len();

    assert_abs_diff_eq!(
        obs_context.timestamps.first().as_gpst_seconds(),
        1090008658.0
    );

    let mut vis = Array2::zeros((num_unflagged_tiles, num_unflagged_chans));
    let mut vis_weights = Array2::zeros((num_unflagged_tiles, num_unflagged_chans));
    let result = uvfits_reader.read_autos(
        vis.view_mut(),
        vis_weights.view_mut(),
        *obs_context.all_timesteps.first(),
        &tile_flags,
        &chan_flags,
    );
    assert!(result.is_ok(), "{}", result.unwrap_err());

    // Use the same values as the test above, adjusting only the indices.
    assert_abs_diff_eq!(
        TestJones::from(vis[(0, 0)]),
        TestJones::from([
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
        TestJones::from(vis[(0, 1)]),
        TestJones::from([
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
        TestJones::from(vis[(0, 15)]),
        TestJones::from([
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
        TestJones::from(vis[(8, 15)]),
        TestJones::from([
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
    let num_chans = obs_context.num_fine_chans_per_coarse_chan;
    let TileBaselineMaps {
        tile_to_unflagged_cross_baseline_map,
        ..
    } = TileBaselineMaps::new(total_num_tiles, &[]);

    assert_abs_diff_eq!(
        obs_context.timestamps.first().as_gpst_seconds(),
        1090008658.0
    );

    let mut cross_vis = Array2::zeros((num_baselines, num_chans));
    let mut cross_vis_weights = Array2::zeros((num_baselines, num_chans));
    let mut auto_vis = Array2::zeros((total_num_tiles, num_chans));
    let mut auto_vis_weights = Array2::zeros((total_num_tiles, num_chans));
    let result = uvfits_reader.read_crosses_and_autos(
        cross_vis.view_mut(),
        cross_vis_weights.view_mut(),
        auto_vis.view_mut(),
        auto_vis_weights.view_mut(),
        *obs_context.all_timesteps.first(),
        &tile_to_unflagged_cross_baseline_map,
        &[],
        &HashSet::new(),
    );
    assert!(result.is_ok(), "{}", result.unwrap_err());

    assert_abs_diff_eq!(
        TestJones::from(cross_vis[(0, 0)]),
        TestJones::from([
            c32::new(-1.2564129e2, -1.497961e1),
            c32::new(8.207059e1, -1.4936417e2),
            c32::new(-7.306871e1, 2.36177e2),
            c32::new(-5.5305626e1, -2.3209404e1)
        ])
    );
    assert_abs_diff_eq!(
        TestJones::from(cross_vis[(10, 16)]),
        TestJones::from([
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
        TestJones::from(auto_vis[(0, 0)]),
        TestJones::from([
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
        TestJones::from(auto_vis[(0, 2)]),
        TestJones::from([
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
        TestJones::from(auto_vis[(0, 16)]),
        TestJones::from([
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
        TestJones::from(auto_vis[(10, 16)]),
        TestJones::from([
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
    let temp_dir = tempdir().expect("Couldn't make temp dir").into_path();
    let vis_path = temp_dir.join("vis.uvfits");
    // uncomment this to write to tmp instead
    // let vis_path = PathBuf::from("/tmp/vis.uvfits");

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

    let mut writer = UvfitsWriter::from_marlu(
        &vis_path,
        &vis_ctx,
        Some(array_pos),
        phase_centre,
        Some(format!("synthesized test data {}", obsid)),
    )
    .unwrap();

    writer
        .write_vis_marlu(
            vis_data.view(),
            weight_data.view(),
            &vis_ctx,
            &tile_xyzs,
            false,
        )
        .unwrap();

    writer
        .write_uvfits_antenna_table(&tile_names, &tile_xyzs)
        .unwrap();

    let uvfits_reader = UvfitsReader::new::<&PathBuf, &PathBuf>(&vis_path, None).unwrap();
    let uvfits_ctx = uvfits_reader.get_obs_context();

    let expected_timestamps = (0..num_timesteps)
        .map(|t| Epoch::from_gpst_seconds((obsid + t) as f64 + 0.5))
        .collect::<Vec<_>>();
    assert_eq!(
        uvfits_ctx
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
