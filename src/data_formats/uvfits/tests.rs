// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

use std::collections::HashSet;

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
use mwa_hyperdrive_beam::Delays;
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
    let result = UvfitsReader::new(&output.path(), metafits, &mut Delays::NotNecessary);
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
fn test_1090008640_uvfits_reads_correctly() {
    let args = get_reduced_1090008640_uvfits();
    let uvfits_reader = if let [metafits, uvfits] = &args.data.unwrap()[..] {
        match UvfitsReader::new(uvfits, Some(metafits), &mut Delays::None) {
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
