// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Tests against calibration parameters and converting arguments to parameters.

use approx::{assert_abs_diff_eq, AbsDiffEq};
use marlu::{
    constants::{MWA_HEIGHT_M, MWA_LAT_DEG, MWA_LONG_DEG},
    LatLngHeight,
};
use serial_test::serial;

use super::InvalidArgsError::{
    BadArrayPosition, BadDelays, CalFreqFactorNotInteger, CalFreqResNotMulitple,
    CalTimeFactorNotInteger, CalTimeResNotMulitple, CalibrationOutputFile, InvalidDataInput,
    MultipleMeasurementSets, MultipleMetafits, MultipleUvfits, NoInputData, ParsePfbFlavour,
};
use crate::{
    data_formats::RawDataReader,
    tests::{full_obsids::*, reduced_obsids::*},
};
use mwa_hyperdrive_common::marlu;

#[test]
fn test_new_params_defaults() {
    let args = get_reduced_1090008640(true);
    let params = args.into_params().unwrap();
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
    let params = args.into_params().unwrap();
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
    assert!(matches!(result, Err(CalTimeFactorNotInteger)));

    let mut args = get_reduced_1090008640(true);
    // 2.01s is not a multiple of 2.0s
    args.time_average_factor = Some("2.01s".to_string());
    let result = args.into_params();
    assert!(result.is_err());
    assert!(matches!(result, Err(CalTimeResNotMulitple { .. })));

    let mut args = get_reduced_1090008640(true);
    // 3.0s is not a multiple of 2.0s
    args.time_average_factor = Some("3.0s".to_string());
    let result = args.into_params();
    assert!(result.is_err());
    assert!(matches!(result, Err(CalTimeResNotMulitple { .. })));
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
    assert!(matches!(result, Err(CalFreqFactorNotInteger)));

    let mut args = get_reduced_1090008640(true);
    // 10kHz is not a multiple of 40kHz
    args.freq_average_factor = Some("10kHz".to_string());
    let result = args.into_params();
    assert!(result.is_err());
    assert!(matches!(result, Err(CalFreqResNotMulitple { .. })));

    let mut args = get_reduced_1090008640(true);
    // 79kHz is not a multiple of 40kHz
    args.freq_average_factor = Some("79kHz".to_string());
    let result = args.into_params();
    assert!(result.is_err());
    assert!(matches!(result, Err(CalFreqResNotMulitple { .. })));
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
    assert_eq!(params.tile_to_unflagged_cross_baseline_map.len(), 7750);

    assert_eq!(params.tile_to_unflagged_cross_baseline_map[&(0, 4)], 0);
    assert_eq!(params.tile_to_unflagged_cross_baseline_map[&(0, 5)], 1);
    assert_eq!(params.tile_to_unflagged_cross_baseline_map[&(0, 6)], 2);
    assert_eq!(params.tile_to_unflagged_cross_baseline_map[&(0, 7)], 3);
}

// The following tests use full MWA data.

#[test]
#[serial]
#[ignore = "files not in repo"]
fn test_new_params_real_data() {
    let args = get_1065880128();
    let result = args.into_params();
    assert!(result.is_ok());
}

#[test]
fn test_handle_delays() {
    let mut args = get_reduced_1090008640(true);
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
    args.delays = Some(delays);
    let result = args.into_params();

    assert!(result.is_ok(), "result={:?} not Ok", result.err().unwrap());

    // XXX(dev): not testable yet.
    // let fee_beam = result.unwrap().beam.downcast::<FeeBeam>().unwrap();
    // assert_eq!(fee_beam.delays[(0, 0)], delays[0]);
}

#[test]
fn test_handle_no_input() {
    let mut args = get_reduced_1090008640(true);
    args.data = None;
    let result = args.into_params();

    assert!(result.is_err());
    assert!(matches!(result, Err(NoInputData)));
}

#[test]
fn test_handle_multiple_metafits() {
    // when reading raw
    let mut args = get_reduced_1090008640(true);
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
    let mut args = get_reduced_1090008640(true);
    args.data = Some(vec!["test_files/1090008640/1090008640.metafits".into()]);
    let result = args.into_params();

    assert!(result.is_err());
    assert!(matches!(result, Err(InvalidDataInput)));
}

#[test]
fn test_handle_pfb() {
    let mut args = get_reduced_1090008640(true);
    args.pfb_flavour = Some("invalid_pfb".into());
    let result = args.clone().into_params();

    assert!(result.is_err());
    assert!(matches!(result, Err(ParsePfbFlavour(_))));

    args.pfb_flavour = Some("Cotter2014".into());
    let result = args.into_params();

    assert!(result.is_ok());
    let input_data_ptr = Box::into_raw(result.unwrap().input_data);
    let raw_input_data = unsafe { Box::from_raw(input_data_ptr as *mut RawDataReader) };
    let gains = raw_input_data.pfb_gains.unwrap();
    assert_eq!(&gains[..2], &[0.5002092286_f64, 0.5025463233]);
}

#[test]
fn test_handle_invalid_output() {
    let mut args = get_reduced_1090008640(true);
    args.outputs = Some(vec!["invalid.out".into()]);
    let result = args.into_params();

    assert!(result.is_err());
    assert!(matches!(result, Err(CalibrationOutputFile { .. })));
}

#[derive(PartialEq, Debug)]
pub(crate) struct TestLatLngHeight(LatLngHeight);

impl From<LatLngHeight> for TestLatLngHeight {
    fn from(other: LatLngHeight) -> Self {
        Self(other)
    }
}

#[cfg(test)]
impl AbsDiffEq for TestLatLngHeight {
    type Epsilon = f64;

    fn default_epsilon() -> f64 {
        f64::EPSILON
    }

    fn abs_diff_eq(&self, other: &Self, epsilon: f64) -> bool {
        f64::abs_diff_eq(&self.0.longitude_rad, &other.0.longitude_rad, epsilon)
            && f64::abs_diff_eq(&self.0.latitude_rad, &other.0.latitude_rad, epsilon)
            && f64::abs_diff_eq(&self.0.height_metres, &other.0.height_metres, epsilon)
    }
}

#[test]
fn test_handle_array_pos() {
    let mut args = get_reduced_1090008640(true);
    let expected = vec![MWA_LONG_DEG + 1.0, MWA_LAT_DEG + 1.0, MWA_HEIGHT_M + 1.0];
    args.array_position = Some(expected.clone());
    let result = args.into_params().unwrap();

    assert_abs_diff_eq!(
        TestLatLngHeight::from(result.array_position),
        TestLatLngHeight::from(LatLngHeight {
            longitude_rad: expected[0].to_radians(),
            latitude_rad: expected[1].to_radians(),
            height_metres: expected[2]
        })
    );
}

#[test]
fn test_handle_bad_array_pos() {
    let mut args = get_reduced_1090008640(true);
    let expected = vec![MWA_LONG_DEG + 1.0, MWA_LAT_DEG + 1.0];
    args.array_position = Some(expected);
    let result = args.into_params();
    assert!(result.is_err());
    assert!(matches!(result.err().unwrap(), BadArrayPosition { .. }))
}

#[test]
/// test that it correctly stacks averaging factor.
///
/// TODO(dev): use test data with multiple timesteps to test time factor stacking
fn test_handle_both_vis_avg() {
    let mut args = get_reduced_1090008640(true);
    args.freq_average_factor = Some("80kHz".into());
    args.output_vis_freq_average = Some("160kHz".into());
    args.outputs = Some(vec!["test.uvfits".into()]);
    let result = args.into_params().unwrap();

    // output averaging factor should be relative to cal averaging factor.
    assert_eq!(result.output_vis_freq_average_factor, 2);
}
