// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

use std::collections::HashSet;

use approx::assert_abs_diff_eq;
use serial_test::serial; // Need to test serially because casacore is a steaming pile.
use tempfile::tempdir;

use super::*;
use crate::{
    calibrate::di::code::{get_cal_vis, tests::test_1090008640_quality},
    jones_test::TestJones,
    math::TileBaselineMaps,
    tests::reduced_obsids::get_reduced_1090008640_ms,
};

#[test]
#[serial]
fn test_1090008640_ms_reads_correctly() {
    let args = get_reduced_1090008640_ms();
    let ms_reader = if let [metafits, ms] = &args.data.unwrap()[..] {
        match MS::new(ms, Some(metafits), &mut Delays::None) {
            Ok(m) => m,
            Err(e) => panic!("{}", e),
        }
    } else {
        panic!("There weren't 2 elements in args.data");
    };

    let obs_context = &ms_reader.obs_context;
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
    let result = ms_reader.read_crosses(
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
