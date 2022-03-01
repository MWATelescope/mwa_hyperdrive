// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

use std::collections::HashSet;

use approx::assert_abs_diff_eq;
use hifitime::Epoch;
use marlu::{c32, pos::xyz::xyzs_to_cross_uvws_parallel, Jones, RADec, XyzGeodetic, UVW};
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
use mwa_hyperdrive_common::{hifitime, marlu, ndarray};

#[test]
fn test_get_truncated_date_str() {
    let mjd = 56580.575370370374;
    let mjd_seconds = mjd * 24.0 * 3600.0;
    // The number of seconds between 1858-11-17T00:00:00 (MJD epoch) and
    // 1900-01-01T00:00:00 (TAI epoch) is 1297728000.
    let epoch_diff = 1297728000.0;
    let epoch = Epoch::from_tai_seconds(mjd_seconds - epoch_diff);
    assert_eq!(get_truncated_date_string(epoch), "2013-10-15T00:00:00.0");
}

#[test]
fn test_get_truncated_date_str_leading_zeros() {
    let mjd = 56541.575370370374;
    let mjd_seconds = mjd * 24.0 * 3600.0;
    let epoch_diff = 1297728000.0;
    let epoch = Epoch::from_tai_seconds(mjd_seconds - epoch_diff);
    assert_eq!(get_truncated_date_string(epoch), "2013-09-06T00:00:00.0");
}

#[test]
fn test_encode_uvfits_baseline() {
    assert_eq!(encode_uvfits_baseline(1, 1), 257);
    // TODO: Test the other part of the if statement.
}

#[test]
fn test_decode_uvfits_baseline() {
    assert_eq!(decode_uvfits_baseline(257), (1, 1));
    // TODO: Test the other part of the if statement.
}

#[test]
// Make a tiny uvfits file. The result has been verified by CASA's
// "importuvfits" function.
fn test_new_uvfits_is_sensible() {
    let tmp_uvfits_file = NamedTempFile::new().unwrap();
    let num_timesteps = 1;
    let num_tiles = 3;
    let num_baselines = (num_tiles * (num_tiles - 1)) / 2;
    let num_chans = 2;
    let obsid = 1065880128.0;
    let start_epoch = Epoch::from_gpst_seconds(obsid);
    let maps = TileBaselineMaps::new(num_tiles, &[]);
    let chan_flags = HashSet::new();

    let mut u = UvfitsWriter::new(
        tmp_uvfits_file.path(),
        num_timesteps,
        num_baselines,
        num_chans,
        false,
        start_epoch,
        Some(40e3),
        170e6,
        RADec::new_degrees(0.0, 60.0),
        Some("test"),
        &maps.unflagged_cross_baseline_to_tile_map,
        &chan_flags,
    )
    .unwrap();

    let mut f = u.open().unwrap();
    let mut row = vec![0.0; 5];
    row.append(&mut (0..num_chans).into_iter().map(|i| i as f32).collect());
    for _timestep_index in 0..num_timesteps {
        for baseline_index in 0..num_baselines {
            let (tile1, tile2) = maps.unflagged_cross_baseline_to_tile_map[&baseline_index];
            u.write_vis(&mut f, UVW::default(), tile1, tile2, start_epoch, &mut row)
                .unwrap();
        }
    }

    let names = ["Tile1", "Tile2", "Tile3"];
    let positions: Vec<XyzGeodetic> = (0..names.len())
        .into_iter()
        .map(|i| XyzGeodetic {
            x: i as f64,
            y: i as f64 * 2.0,
            z: i as f64 * 3.0,
        })
        .collect();
    u.write_uvfits_antenna_table(&names, &positions).unwrap();
}

fn write_then_read_uvfits(autos: bool) {
    let output = NamedTempFile::new().expect("Couldn't create temporary file");
    let phase_centre = RADec::new_degrees(0.0, -27.0);
    let lst_rad = 0.0;
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
    let num_cross_baselines = (num_tiles * (num_tiles - 1)) / 2;
    let uvws = xyzs_to_cross_uvws_parallel(&xyzs, phase_centre.to_hadec(lst_rad));
    let num_baselines = if autocorrelations_present {
        (num_tiles * (num_tiles + 1)) / 2
    } else {
        num_cross_baselines
    };

    let flagged_tiles = vec![];
    let flagged_fine_chans = HashSet::new();
    let maps = TileBaselineMaps::new(num_tiles, &flagged_tiles);

    // Just in case this gets accidentally changed.
    assert_eq!(
        num_timesteps, 1,
        "num_timesteps should always be 1 for this test"
    );

    let result = UvfitsWriter::new(
        output.path(),
        num_timesteps,
        num_baselines,
        num_chans,
        autocorrelations_present,
        *timesteps.first().unwrap(),
        Some(fine_chan_width_hz),
        fine_chan_freqs_hz[num_chans / 2],
        phase_centre,
        None,
        &maps.unflagged_cross_baseline_to_tile_map,
        &flagged_fine_chans,
    );
    assert!(result.is_ok(), "Failed to create new uvfits file");
    let mut output_writer = result.unwrap();

    let mut cross_vis = Array2::from_elem(
        (num_cross_baselines, num_chans),
        Jones::from([
            c32::new(1.0, 1.0),
            c32::new(1.0, 1.0),
            c32::new(1.0, 1.0),
            c32::new(1.0, 1.0),
        ]),
    );
    cross_vis
        .iter_mut()
        .enumerate()
        .for_each(|(i, v)| *v *= i as f32);
    let cross_weights = Array2::ones(cross_vis.dim());

    let mut auto_vis = Array2::from_elem(
        (num_tiles, num_chans),
        Jones::from([
            c32::new(1.0, 1.0),
            c32::new(1.0, 1.0),
            c32::new(1.0, 1.0),
            c32::new(1.0, 1.0),
        ]),
    );
    auto_vis
        .iter_mut()
        .enumerate()
        .for_each(|(i, v)| *v *= i as f32);
    let auto_weights = Array2::ones(auto_vis.dim());

    for timestep in timesteps {
        let result = if autocorrelations_present {
            output_writer.write_cross_and_auto_timestep_vis(
                cross_vis.view(),
                cross_weights.view(),
                auto_vis.view(),
                auto_weights.view(),
                &uvws,
                timestep,
            )
        } else {
            output_writer.write_cross_timestep_vis(
                cross_vis.view(),
                Array2::ones(cross_vis.dim()).view(),
                &uvws,
                timestep,
            )
        };
        assert!(
            result.is_ok(),
            "Failed to write visibilities to uvfits file: {:?}",
            result.unwrap_err()
        );
        result.unwrap();
    }

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
    for (timestep, _) in timesteps.iter().enumerate() {
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

        let cross_vis = cross_vis.mapv(TestJones::from);
        let cross_vis_read = cross_vis_read.mapv(TestJones::from);
        assert_abs_diff_eq!(cross_vis_read, cross_vis);
        assert_abs_diff_eq!(cross_weights_read, cross_weights);

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

            let auto_vis = auto_vis.mapv(TestJones::from);
            let auto_vis_read = auto_vis_read.mapv(TestJones::from);
            assert_abs_diff_eq!(auto_vis_read, auto_vis);
            assert_abs_diff_eq!(auto_weights_read, auto_weights);
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

// TODO: Test visibility averaging.

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
