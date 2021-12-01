// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Tests against calibration parameters and converting arguments to parameters.

use super::*;
use crate::tests::{full_obsids::*, reduced_obsids::*, *};

#[test]
fn test_get_flagged_baselines_set() {
    let total_num_tiles = 128;
    let mut tile_flags = HashSet::new();
    let flagged_baselines = get_flagged_baselines_set(total_num_tiles, &tile_flags);
    assert!(flagged_baselines.is_empty());

    tile_flags.insert(127);
    let flagged_baselines = get_flagged_baselines_set(total_num_tiles, &tile_flags);
    assert_eq!(flagged_baselines.len(), 127);
    assert!(flagged_baselines.contains(&126));
    assert!(flagged_baselines.contains(&252));
    assert!(flagged_baselines.contains(&8127));
}

// #[test]
// fn test_new_params() {
//     let args = get_1090008640_smallest();
//     let params = match args.into_params() {
//         Ok(p) => p,
//         Err(e) => panic!("{}", e),
//     };
//     // The default time resolution should be 2.0s, as per the metafits.
//     assert_abs_diff_eq!(params.time_res.unwrap(), 2.0);
//     // The default freq resolution should be 40kHz, as per the metafits.
//     assert_abs_diff_eq!(params.freq.res.unwrap(), 40e3, epsilon = 1e-10);
// }

// #[test]
// fn test_new_params_time_averaging() {
//     // The native time resolution is 2.0s.
//     let mut args = get_1090008640_smallest();
//     // 4.0 should be a multiple of 2.0s
//     args.time_res = Some(4.0);
//     let params = match args.into_params() {
//         Ok(p) => p,
//         Err(e) => panic!("{}", e),
//     };
//     assert_abs_diff_eq!(params.time_res.unwrap(), 4.0);

//     let mut args = get_1090008640();
//     // 8.0 should be a multiple of 2.0s
//     args.time_res = Some(8.0);
//     let params = match args.into_params() {
//         Ok(p) => p,
//         Err(e) => panic!("{}", e),
//     };
//     assert_abs_diff_eq!(params.time_res.unwrap(), 8.0);
// }

// #[test]
// fn test_new_params_time_averaging_fail() {
//     // The native time resolution is 2.0s.
//     let mut args = get_1090008640_smallest();
//     // 2.01 is not a multiple of 2.0s
//     args.time_res = Some(2.01);
//     let result = args.into_params();
//     assert!(
//         result.is_err(),
//         "Expected CalibrateParams to have not been successfully created"
//     );

//     let mut args = get_1090008640_smallest();
//     // 3.0 is not a multiple of 2.0s
//     args.time_res = Some(3.0);
//     let result = args.into_params();
//     assert!(
//         result.is_err(),
//         "Expected CalibrateParams to have not been successfully created"
//     );
// }

// #[test]
// fn test_new_params_freq_averaging() {
//     // The native freq. resolution is 40kHz.
//     let mut args = get_1090008640_smallest();
//     // 80e3 should be a multiple of 40kHz
//     args.freq_res = Some(80e3);
//     let params = match args.into_params() {
//         Ok(p) => p,
//         Err(e) => panic!("{}", e),
//     };
//     assert_abs_diff_eq!(params.freq.res, 80e3, epsilon = 1e-10);

//     let mut args = get_1090008640_smallest();
//     // 200e3 should be a multiple of 40kHz
//     args.freq_res = Some(200e3);
//     let params = match args.into_params() {
//         Ok(p) => p,
//         Err(e) => panic!("{}", e),
//     };
//     assert_abs_diff_eq!(params.freq.res, 200e3, epsilon = 1e-10);
// }

// #[test]
// fn test_new_params_freq_averaging_fail() {
//     // The native freq. resolution is 40kHz.
//     let mut args = get_1090008640_smallest();
//     // 10e3 is not a multiple of 40kHz
//     args.freq_res = Some(10e3);
//     let result = args.into_params();
//     assert!(
//         result.is_err(),
//         "Expected CalibrateParams to have not been successfully created"
//     );

//     let mut args = get_1090008640_smallest();

//     // 79e3 is not a multiple of 40kHz
//     args.freq_res = Some(79e3);
//     let result = args.into_params();
//     assert!(
//         result.is_err(),
//         "Expected CalibrateParams to have not been successfully created"
//     );
// }

#[test]
fn test_new_params_tile_flags() {
    // 1090008640 has no flagged tiles in its metafits.
    let mut args = get_1090008640();
    // Manually flag antennas 1, 2 and 3.
    args.tile_flags = Some(vec!["1".to_string(), "2".to_string(), "3".to_string()]);
    let params = match args.into_params() {
        Ok(p) => p,
        Err(e) => panic!("{}", e),
    };
    assert_eq!(params.tile_flags.len(), 3);
    assert!(params.tile_flags.contains(&1));
    assert!(params.tile_flags.contains(&2));
    assert!(params.tile_flags.contains(&3));
    assert_eq!(params.unflagged_cross_baseline_to_tile_map.len(), 7750);
    assert_eq!(params.tile_to_unflagged_cross_baseline_map.len(), 7750);

    assert_eq!(params.unflagged_cross_baseline_to_tile_map[&0], (0, 4));
    assert_eq!(params.unflagged_cross_baseline_to_tile_map[&1], (0, 5));
    assert_eq!(params.unflagged_cross_baseline_to_tile_map[&2], (0, 6));
    assert_eq!(params.unflagged_cross_baseline_to_tile_map[&3], (0, 7));

    assert_eq!(params.tile_to_unflagged_cross_baseline_map[&(0, 4)], 0);
    assert_eq!(params.tile_to_unflagged_cross_baseline_map[&(0, 5)], 1);
    assert_eq!(params.tile_to_unflagged_cross_baseline_map[&(0, 6)], 2);
    assert_eq!(params.tile_to_unflagged_cross_baseline_map[&(0, 7)], 3);
}

// The following tests use full MWA data.

#[test]
#[serial]
#[ignore]
fn test_new_params_real_data() {
    let args = get_1065880128();
    let result = args.into_params();
    assert!(
        result.is_ok(),
        "Expected CalibrateParams to have been successfully created"
    );
}

// #[test]
// #[serial]
// #[ignore]
// fn test_lst_from_timestep_native_real() {
//     let args = get_1065880128();
//     let context = match CorrelatorContext::new(&args.metafits.unwrap(), &args.gpuboxes.unwrap())
//     {
//         Ok(c) => c,
//         Err(e) => panic!("{}", e),
//     };
//     let time_res = context.metafits_context.corr_int_time_ms as f64 / 1e3;
//     let new_lst = lst_from_timestep(0, &context, time_res);
//     // gpstime 1065880126.25
//     assert_abs_diff_eq!(new_lst, 6.074695614533638, epsilon = 1e-10);

//     let new_lst = lst_from_timestep(1, &context, time_res);
//     // gpstime 1065880126.75
//     assert_abs_diff_eq!(new_lst, 6.074732075112903, epsilon = 1e-10);
// }

// #[test]
// #[serial]
// #[ignore]
// fn test_lst_from_timestep_averaged_real() {
//     let args = get_1065880128();
//     let context = match CorrelatorContext::new(&args.metafits.unwrap(), &args.gpuboxes.unwrap())
//     {
//         Ok(c) => c,
//         Err(e) => panic!("{}", e),
//     };
//     // The native time res. is 0.5s, let's make our target 2s here.
//     let time_res = 2.0;
//     let new_lst = lst_from_timestep(0, &context, time_res);
//     // gpstime 1065880127
//     assert_abs_diff_eq!(new_lst, 6.074750305402534, epsilon = 1e-10);

//     let new_lst = lst_from_timestep(1, &context, time_res);
//     // gpstime 1065880129
//     assert_abs_diff_eq!(new_lst, 6.074896147719591, epsilon = 1e-10);
// }
