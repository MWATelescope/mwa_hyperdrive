// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Tests against the command-line interface for calibration.

use approx::assert_abs_diff_eq;
use mwa_hyperdrive_common::{
    clap::{ErrorKind::WrongNumberOfValues, Parser},
    marlu::constants::{MWA_HEIGHT_M, MWA_LAT_DEG, MWA_LONG_DEG},
};

use crate::*;

#[test]
fn test_calibrate_help_is_correct() {
    // First with --help
    let cmd = hyperdrive().arg("--help").ok();
    assert!(cmd.is_ok());
    let (stdout, stderr) = get_cmd_output(cmd);
    assert!(stderr.is_empty());
    assert!(stdout.contains("di-calibrate"));
    assert!(stdout.contains("Perform direction-independent calibration on the input MWA data"));

    // Second with -h
    let cmd = hyperdrive().arg("-h").ok();
    assert!(cmd.is_ok());
    let (stdout, stderr) = get_cmd_output(cmd);
    assert!(stderr.is_empty());
    assert!(stdout.contains("di-calibrate"));
    assert!(stdout.contains("Perform direction-independent calibration on the input MWA data"));
}

#[test]
fn test_1090008640_di_calibrate_uses_array_position() {
    let tmp_dir = TempDir::new().expect("couldn't make tmp dir").into_path();
    let args = get_reduced_1090008640(true, false);
    let data = args.data.unwrap();
    let metafits = &data[0];
    let gpufits = &data[1];
    let sols = tmp_dir.join("sols.fits");
    let cal_model = tmp_dir.join("hyp_model.uvfits");

    // with non-default array position
    let exp_lat_deg = MWA_LAT_DEG - 1.;
    let exp_long_deg = MWA_LONG_DEG - 1.;
    let exp_height_m = MWA_HEIGHT_M - 1.;

    #[rustfmt::skip]
    let cal_args = CalibrateUserArgs::parse_from(&[
        "di-calibrate",
        "--data", metafits, gpufits,
        "--source-list", args.source_list.as_ref().unwrap(),
        "--outputs", &format!("{}", sols.display()),
        "--model-filename", &format!("{}", cal_model.display()),
        "--array-position",
            &format!("{}", exp_long_deg),
            &format!("{}", exp_lat_deg),
            &format!("{}", exp_height_m),
    ]);

    let pos = cal_args.array_position.unwrap();

    assert_abs_diff_eq!(pos[0], exp_long_deg);
    assert_abs_diff_eq!(pos[1], exp_lat_deg);
    assert_abs_diff_eq!(pos[2], exp_height_m);
}

#[test]
fn test_1090008640_di_calibrate_array_pos_requires_3_args() {
    let tmp_dir = TempDir::new().expect("couldn't make tmp dir").into_path();
    let args = get_reduced_1090008640(true, false);
    let data = args.data.unwrap();
    let metafits = &data[0];
    let gpufits = &data[1];
    let sols = tmp_dir.join("sols.fits");
    let cal_model = tmp_dir.join("hyp_model.uvfits");

    // no height specified
    let exp_lat_deg = MWA_LAT_DEG - 1.;
    let exp_long_deg = MWA_LONG_DEG - 1.;

    #[rustfmt::skip]
    let result = CalibrateUserArgs::try_parse_from(&[
        "di-calibrate",
        "--data", metafits, gpufits,
        "--source-list", args.source_list.as_ref().unwrap(),
        "--outputs", &format!("{}", sols.display()),
        "--model-filename", &format!("{}", cal_model.display()),
        "--array-position",
            &format!("{}", exp_long_deg),
            &format!("{}", exp_lat_deg),
    ]);

    assert!(result.is_err());
    assert!(matches!(result.err().unwrap().kind(), WrongNumberOfValues));
}
