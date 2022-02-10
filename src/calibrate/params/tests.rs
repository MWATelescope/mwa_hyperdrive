// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Tests against calibration parameters and converting arguments to parameters.

use approx::assert_abs_diff_eq;
use serial_test::serial;

use super::InvalidArgsError;
use crate::tests::{full_obsids::*, reduced_obsids::*};

#[test]
fn test_new_params_defaults() {
    let args = get_reduced_1090008640(true);
    let params = match args.into_params() {
        Ok(p) => p,
        Err(e) => panic!("{}", e),
    };
    let obs_context = params.get_obs_context();
    // The default time resolution should be 2.0s, as per the metafits.
    assert_abs_diff_eq!(obs_context.time_res.unwrap(), 2.0);
    // The default freq resolution should be 40kHz, as per the metafits.
    assert_abs_diff_eq!(obs_context.freq_res.unwrap(), 40e3);
    // No tiles are flagged in the input data, and no additional flags were
    // supplied.
    assert_eq!(obs_context.flagged_tiles.len(), 0);
    assert_eq!(params.flagged_tiles.len(), 0);

    // By default there are 5 flagged channels per coarse channel. We only have
    // one coarse channel here so we expect 27/32 channels. Also no picket fence
    // shenanigans.
    assert_eq!(params.fences.len(), 1);
    assert_eq!(params.fences[0].chanblocks.len(), 27);
}

#[test]
fn test_new_params_no_input_flags() {
    let mut args = get_reduced_1090008640(true);
    args.ignore_input_data_tile_flags = true;
    args.ignore_input_data_fine_channel_flags = true;
    let params = match args.into_params() {
        Ok(p) => p,
        Err(e) => panic!("{}", e),
    };
    let obs_context = params.get_obs_context();
    assert_abs_diff_eq!(obs_context.time_res.unwrap(), 2.0);
    assert_abs_diff_eq!(obs_context.freq_res.unwrap(), 40e3);
    assert_eq!(obs_context.flagged_tiles.len(), 0);
    assert_eq!(params.flagged_tiles.len(), 0);

    assert_eq!(params.fences.len(), 1);
    assert_eq!(params.fences[0].chanblocks.len(), 32);
}

#[test]
fn test_new_params_time_averaging() {
    // The native time resolution is 2.0s.
    let mut args = get_reduced_1090008640(true);
    // 1 is a valid time average factor.
    args.time_average_factor = Some("1".to_string());
    let result = args.into_params();
    assert!(result.is_ok());

    let mut args = get_reduced_1090008640(true);
    // 2 is a valid time average factor.
    args.time_average_factor = Some("2".to_string());
    let result = args.into_params();
    assert!(result.is_ok());

    let mut args = get_reduced_1090008640(true);
    // 4.0s should be a multiple of 2.0s
    args.time_average_factor = Some("4.0s".to_string());
    let result = args.into_params();
    assert!(result.is_ok());

    let mut args = get_reduced_1090008640(true);
    // 8.0s should be a multiple of 2.0s
    args.time_average_factor = Some("8.0s".to_string());
    let result = args.into_params();
    assert!(result.is_ok());
}

#[test]
fn test_new_params_time_averaging_fail() {
    // The native time resolution is 2.0s.
    let mut args = get_reduced_1090008640(true);
    // 1.5 is an invalid time average factor.
    args.time_average_factor = Some("1.5".to_string());
    let result = args.into_params();
    assert!(result.is_err());
    let err = match result {
        Ok(_) => unreachable!(),
        Err(err) => err,
    };
    assert!(matches!(err, InvalidArgsError::CalTimeFactorNotInteger));

    let mut args = get_reduced_1090008640(true);
    // 2.01s is not a multiple of 2.0s
    args.time_average_factor = Some("2.01s".to_string());
    let result = args.into_params();
    assert!(result.is_err());
    let err = match result {
        Ok(_) => unreachable!(),
        Err(err) => err,
    };
    assert!(matches!(
        err,
        InvalidArgsError::CalTimeResNotMulitple { .. }
    ));

    let mut args = get_reduced_1090008640(true);
    // 3.0s is not a multiple of 2.0s
    args.time_average_factor = Some("3.0s".to_string());
    let result = args.into_params();
    assert!(result.is_err());
    let err = match result {
        Ok(_) => unreachable!(),
        Err(err) => err,
    };
    assert!(matches!(
        err,
        InvalidArgsError::CalTimeResNotMulitple { .. }
    ));
}

#[test]
fn test_new_params_freq_averaging() {
    // The native freq. resolution is 40kHz.
    let mut args = get_reduced_1090008640(true);
    // 3 is a valid freq average factor.
    args.freq_average_factor = Some("3".to_string());
    let result = args.into_params();
    assert!(result.is_ok());

    let mut args = get_reduced_1090008640(true);
    // 80kHz should be a multiple of 40kHz
    args.freq_average_factor = Some("80kHz".to_string());
    let result = args.into_params();
    assert!(result.is_ok());

    let mut args = get_reduced_1090008640(true);
    // 200kHz should be a multiple of 40kHz
    args.freq_average_factor = Some("200kHz".to_string());
    let result = args.into_params();
    assert!(result.is_ok());
}

#[test]
fn test_new_params_freq_averaging_fail() {
    // The native freq. resolution is 40kHz.
    let mut args = get_reduced_1090008640(true);
    // 1.5 is an invalid freq average factor.
    args.freq_average_factor = Some("1.5".to_string());
    let result = args.into_params();
    assert!(result.is_err());
    let err = match result {
        Ok(_) => unreachable!(),
        Err(err) => err,
    };
    assert!(matches!(err, InvalidArgsError::CalFreqFactorNotInteger));

    let mut args = get_reduced_1090008640(true);
    // 10kHz is not a multiple of 40kHz
    args.freq_average_factor = Some("10kHz".to_string());
    let result = args.into_params();
    assert!(result.is_err());
    let err = match result {
        Ok(_) => unreachable!(),
        Err(err) => err,
    };
    assert!(matches!(
        err,
        InvalidArgsError::CalFreqResNotMulitple { .. }
    ));

    let mut args = get_reduced_1090008640(true);
    // 79kHz is not a multiple of 40kHz
    args.freq_average_factor = Some("79kHz".to_string());
    let result = args.into_params();
    assert!(result.is_err());
    let err = match result {
        Ok(_) => unreachable!(),
        Err(err) => err,
    };
    assert!(matches!(
        err,
        InvalidArgsError::CalFreqResNotMulitple { .. }
    ));
}

#[test]
fn test_new_params_tile_flags() {
    // 1090008640 has no flagged tiles in its metafits.
    let mut args = get_reduced_1090008640(true);
    // Manually flag antennas 1, 2 and 3.
    args.tile_flags = Some(vec!["1".to_string(), "2".to_string(), "3".to_string()]);
    let params = match args.into_params() {
        Ok(p) => p,
        Err(e) => panic!("{}", e),
    };
    assert_eq!(params.flagged_tiles.len(), 3);
    assert!(params.flagged_tiles.contains(&1));
    assert!(params.flagged_tiles.contains(&2));
    assert!(params.flagged_tiles.contains(&3));
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
    assert!(result.is_ok());
}
