// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Tests against calibration parameters and converting arguments to parameters.

use approx::assert_abs_diff_eq;
use marlu::{
    constants::{MWA_HEIGHT_M, MWA_LAT_DEG, MWA_LONG_DEG},
    LatLngHeight,
};

use super::DiCalArgsError::{
    BadArrayPosition, BadDelays, CalFreqFactorNotInteger, CalFreqResNotMultiple,
    CalTimeFactorNotInteger, CalTimeResNotMultiple, CalibrationOutputFile, InvalidDataInput,
    MultipleMeasurementSets, MultipleMetafits, MultipleUvfits, NoInputData,
};
use crate::{
    beam::BeamType,
    tests::reduced_obsids::{
        get_reduced_1090008640, get_reduced_1090008640_ms, get_reduced_1090008640_uvfits,
    },
};

#[test]
fn test_new_params_defaults() {
    let args = get_reduced_1090008640(false, true);
    let params = args.into_params().unwrap();
    let obs_context = params.get_obs_context();
    // The default time resolution should be 2.0s, as per the metafits.
    assert_abs_diff_eq!(obs_context.time_res.unwrap().to_seconds(), 2.0);
    // The default freq resolution should be 40kHz, as per the metafits.
    assert_abs_diff_eq!(obs_context.freq_res.unwrap(), 40e3);
    // No tiles are flagged in the input data, and no additional flags were
    // supplied.
    assert_eq!(
        obs_context.get_total_num_tiles(),
        obs_context.get_num_unflagged_tiles()
    );
    assert_eq!(params.tile_baseline_flags.flagged_tiles.len(), 0);

    // By default there are 5 flagged channels per coarse channel. We only have
    // one coarse channel here so we expect 27/32 channels. Also no picket fence
    // shenanigans.
    assert_eq!(params.fences.len(), 1);
    assert_eq!(params.fences[0].chanblocks.len(), 27);
}

#[test]
fn test_new_params_no_input_flags() {
    let mut args = get_reduced_1090008640(false, true);
    args.ignore_input_data_tile_flags = true;
    args.ignore_input_data_fine_channel_flags = true;
    let params = args.into_params().unwrap();
    let obs_context = params.get_obs_context();
    assert_abs_diff_eq!(obs_context.time_res.unwrap().to_seconds(), 2.0);
    assert_abs_diff_eq!(obs_context.freq_res.unwrap(), 40e3);
    assert_eq!(
        obs_context.get_total_num_tiles(),
        obs_context.get_num_unflagged_tiles(),
    );
    assert_eq!(params.tile_baseline_flags.flagged_tiles.len(), 0);

    assert_eq!(params.fences.len(), 1);
    assert_eq!(params.fences[0].chanblocks.len(), 32);
}

#[test]
fn test_new_params_time_averaging() {
    // The native time resolution is 2.0s.
    let mut args = get_reduced_1090008640(false, true);
    // 1 is a valid time average factor.
    args.timesteps_per_timeblock = Some("1".to_string());
    let result = args.into_params();
    assert!(result.is_ok());

    let mut args = get_reduced_1090008640(false, true);
    // 2 is a valid time average factor.
    args.timesteps_per_timeblock = Some("2".to_string());
    let result = args.into_params();
    assert!(result.is_ok());

    let mut args = get_reduced_1090008640(false, true);
    // 4.0s should be a multiple of 2.0s
    args.timesteps_per_timeblock = Some("4.0s".to_string());
    let result = args.into_params();
    assert!(result.is_ok());

    let mut args = get_reduced_1090008640(false, true);
    // 8.0s should be a multiple of 2.0s
    args.timesteps_per_timeblock = Some("8.0s".to_string());
    let result = args.into_params();
    assert!(result.is_ok());
}

#[test]
fn test_new_params_time_averaging_fail() {
    // The native time resolution is 2.0s.
    let mut args = get_reduced_1090008640(false, true);
    // 1.5 is an invalid time average factor.
    args.timesteps_per_timeblock = Some("1.5".to_string());
    let result = args.into_params();
    assert!(result.is_err());
    assert!(matches!(result, Err(CalTimeFactorNotInteger)));

    let mut args = get_reduced_1090008640(false, true);
    // 2.01s is not a multiple of 2.0s
    args.timesteps_per_timeblock = Some("2.01s".to_string());
    let result = args.into_params();
    assert!(result.is_err());
    assert!(matches!(result, Err(CalTimeResNotMultiple { .. })));

    let mut args = get_reduced_1090008640(false, true);
    // 3.0s is not a multiple of 2.0s
    args.timesteps_per_timeblock = Some("3.0s".to_string());
    let result = args.into_params();
    assert!(result.is_err());
    assert!(matches!(result, Err(CalTimeResNotMultiple { .. })));
}

#[test]
fn test_new_params_freq_averaging() {
    // The native freq. resolution is 40kHz.
    let mut args = get_reduced_1090008640(false, true);
    // 3 is a valid freq average factor.
    args.freq_average_factor = Some("3".to_string());
    let result = args.into_params();
    assert!(result.is_ok());

    let mut args = get_reduced_1090008640(false, true);
    // 80kHz should be a multiple of 40kHz
    args.freq_average_factor = Some("80kHz".to_string());
    let result = args.into_params();
    assert!(result.is_ok());

    let mut args = get_reduced_1090008640(false, true);
    // 200kHz should be a multiple of 40kHz
    args.freq_average_factor = Some("200kHz".to_string());
    let result = args.into_params();
    assert!(result.is_ok());
}

#[test]
fn test_new_params_freq_averaging_fail() {
    // The native freq. resolution is 40kHz.
    let mut args = get_reduced_1090008640(false, true);
    // 1.5 is an invalid freq average factor.
    args.freq_average_factor = Some("1.5".to_string());
    let result = args.into_params();
    assert!(result.is_err());
    assert!(matches!(result, Err(CalFreqFactorNotInteger)));

    let mut args = get_reduced_1090008640(false, true);
    // 10kHz is not a multiple of 40kHz
    args.freq_average_factor = Some("10kHz".to_string());
    let result = args.into_params();
    assert!(result.is_err());
    assert!(matches!(result, Err(CalFreqResNotMultiple { .. })));

    let mut args = get_reduced_1090008640(false, true);
    // 79kHz is not a multiple of 40kHz
    args.freq_average_factor = Some("79kHz".to_string());
    let result = args.into_params();
    assert!(result.is_err());
    assert!(matches!(result, Err(CalFreqResNotMultiple { .. })));
}

#[test]
fn test_new_params_tile_flags() {
    // 1090008640 has no flagged tiles in its metafits.
    let mut args = get_reduced_1090008640(false, true);
    // Manually flag antennas 1, 2 and 3.
    args.tile_flags = Some(vec!["1".to_string(), "2".to_string(), "3".to_string()]);
    let params = match args.into_params() {
        Ok(p) => p,
        Err(e) => panic!("{}", e),
    };
    assert_eq!(params.tile_baseline_flags.flagged_tiles.len(), 3);
    assert!(params.tile_baseline_flags.flagged_tiles.contains(&1));
    assert!(params.tile_baseline_flags.flagged_tiles.contains(&2));
    assert!(params.tile_baseline_flags.flagged_tiles.contains(&3));
    assert_eq!(
        params
            .tile_baseline_flags
            .tile_to_unflagged_cross_baseline_map
            .len(),
        7750
    );

    assert_eq!(
        params
            .tile_baseline_flags
            .tile_to_unflagged_cross_baseline_map[&(0, 4)],
        0
    );
    assert_eq!(
        params
            .tile_baseline_flags
            .tile_to_unflagged_cross_baseline_map[&(0, 5)],
        1
    );
    assert_eq!(
        params
            .tile_baseline_flags
            .tile_to_unflagged_cross_baseline_map[&(0, 6)],
        2
    );
    assert_eq!(
        params
            .tile_baseline_flags
            .tile_to_unflagged_cross_baseline_map[&(0, 7)],
        3
    );
}

#[test]
fn test_handle_delays() {
    let mut args = get_reduced_1090008640(false, true);
    args.no_beam = false;
    // only 3 delays instead of 16 expected
    args.delays = Some((0..3).collect::<Vec<u32>>());
    let result = args.clone().into_params();

    assert!(result.is_err());
    assert!(matches!(result, Err(BadDelays)));

    // delays > 32
    args.delays = Some((20..36).collect::<Vec<u32>>());
    let result = args.clone().into_params();

    assert!(result.is_err());
    assert!(matches!(result, Err(BadDelays)));

    let delays = (0..16).collect::<Vec<u32>>();
    args.delays = Some(delays.clone());
    let result = args.into_params();

    assert!(result.is_ok(), "result={:?} not Ok", result.err().unwrap());

    let fee_beam = result.unwrap().beam;
    assert_eq!(fee_beam.get_beam_type(), BeamType::FEE);
    let beam_delays = fee_beam
        .get_dipole_delays()
        .expect("expected some delays to be provided from the FEE beam!");
    // Each row of the delays should be the same as the 16 input values.
    for row in beam_delays.outer_iter() {
        assert_eq!(row.as_slice().unwrap(), delays);
    }
}

#[test]
fn test_unity_dipole_gains() {
    let mut args = get_reduced_1090008640(false, true);
    args.no_beam = false;
    let params = args.clone().into_params().unwrap();

    let fee_beam = params.beam;
    assert_eq!(fee_beam.get_beam_type(), BeamType::FEE);
    let beam_gains = fee_beam.get_dipole_gains();

    // Because there are dead dipoles in the metafits, we expect some of the
    // gains to not be 1.
    assert!(!beam_gains.unwrap().iter().all(|g| (*g - 1.0).abs() < f64::EPSILON));

    // Now ignore dead dipoles.
    args.unity_dipole_gains = true;
    let params = args.into_params().unwrap();

    let fee_beam = params.beam;
    assert_eq!(fee_beam.get_beam_type(), BeamType::FEE);
    let beam_gains = fee_beam.get_dipole_gains();

    // Now we expect all gains to be 1s, as we're ignoring dead dipoles.
    assert!(beam_gains.unwrap().iter().all(|g| (*g - 1.0).abs() < f64::EPSILON));
    // Verify that there are no dead dipoles in the delays.
    assert!(fee_beam
        .get_dipole_delays()
        .unwrap()
        .iter()
        .all(|d| *d != 32));
}

#[test]
fn test_handle_no_input() {
    let mut args = get_reduced_1090008640(false, true);
    args.data = None;
    let result = args.into_params();

    assert!(result.is_err());
    assert!(matches!(result, Err(NoInputData)));
}

#[test]
fn test_handle_multiple_metafits() {
    // when reading raw
    let mut args = get_reduced_1090008640(false, true);
    args.data
        .as_mut()
        .unwrap()
        .push("test_files/1090008640_WODEN/1090008640.metafits".into());
    let result = args.into_params();

    assert!(result.is_err());
    assert!(matches!(result, Err(MultipleMetafits(_))));

    // when reading ms
    let mut args = get_reduced_1090008640_ms();
    args.data
        .as_mut()
        .unwrap()
        .push("test_files/1090008640_WODEN/1090008640.metafits".into());
    let result = args.into_params();

    assert!(result.is_err());
    assert!(matches!(result, Err(MultipleMetafits(_))));

    // when reading uvfits
    let mut args = get_reduced_1090008640_uvfits();
    args.data
        .as_mut()
        .unwrap()
        .push("test_files/1090008640_WODEN/1090008640.metafits".into());
    let result = args.into_params();

    assert!(result.is_err());
    assert!(matches!(result, Err(MultipleMetafits(_))));
}

#[test]
fn test_handle_multiple_ms() {
    let mut args = get_reduced_1090008640_ms();
    args.data
        .as_mut()
        .unwrap()
        .push("test_files/1090008640/1090008640.ms".into());
    let result = args.into_params();

    assert!(result.is_err());
    assert!(matches!(result, Err(MultipleMeasurementSets(_))));
}

#[test]
fn test_handle_multiple_uvfits() {
    let mut args = get_reduced_1090008640_uvfits();
    args.data
        .as_mut()
        .unwrap()
        .push("test_files/1090008640/1090008640.uvfits".into());
    let result = args.into_params();

    assert!(result.is_err());
    assert!(matches!(result, Err(MultipleUvfits(_))));
}

#[test]
fn test_handle_only_metafits() {
    let mut args = get_reduced_1090008640(false, true);
    args.data = Some(vec!["test_files/1090008640/1090008640.metafits".into()]);
    let result = args.into_params();

    assert!(result.is_err());
    assert!(matches!(result, Err(InvalidDataInput(_))));
}

#[test]
fn test_handle_invalid_output() {
    let mut args = get_reduced_1090008640(false, true);
    args.outputs = Some(vec!["invalid.out".into()]);
    let result = args.into_params();

    assert!(result.is_err());
    assert!(matches!(result, Err(CalibrationOutputFile { .. })));
}

#[test]
fn test_handle_array_pos() {
    let mut args = get_reduced_1090008640(false, true);
    let expected = vec![MWA_LONG_DEG + 1.0, MWA_LAT_DEG + 1.0, MWA_HEIGHT_M + 1.0];
    args.array_position = Some(expected.clone());
    let result = args.into_params().unwrap();

    assert_abs_diff_eq!(
        result.array_position,
        LatLngHeight {
            longitude_rad: expected[0].to_radians(),
            latitude_rad: expected[1].to_radians(),
            height_metres: expected[2]
        }
    );
}

#[test]
fn test_handle_bad_array_pos() {
    let mut args = get_reduced_1090008640(false, true);
    let expected = vec![MWA_LONG_DEG + 1.0, MWA_LAT_DEG + 1.0];
    args.array_position = Some(expected);
    let result = args.into_params();
    assert!(result.is_err());
    assert!(matches!(result.err().unwrap(), BadArrayPosition { .. }))
}
