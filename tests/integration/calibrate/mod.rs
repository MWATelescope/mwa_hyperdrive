// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Integration tests for calibration.

mod cli_args;

use approx::assert_abs_diff_eq;
use clap::Parser;
use mwalib::*;
use serial_test::serial;

use crate::*;
use mwa_hyperdrive::calibrate::{di_calibrate, solutions::CalibrationSolutions, CalibrateError};
use mwa_hyperdrive_common::{clap, mwalib};

/// If di-calibrate is working, it should not write anything to stderr.
#[test]
fn test_no_stderr() {
    let tmp_dir = TempDir::new().expect("couldn't make tmp dir").into_path();
    let sols = tmp_dir.join("sols.fits");
    let args = get_reduced_1090008640(true, false);
    let data = args.data.unwrap();

    #[rustfmt::skip]
    let cmd = hyperdrive()
        .args(&[
            "di-calibrate",
            "--data", &data[0], &data[1],
            "--source-list", &args.source_list.unwrap(),
            "--outputs", &format!("{}", sols.display()),
        ])
        .ok();
    assert!(cmd.is_ok(), "di-calibrate failed on simple test data!");
    let (_, stderr) = get_cmd_output(cmd);
    assert!(stderr.is_empty(), "stderr wasn't empty: {stderr}");
}

#[test]
#[serial]
fn test_1090008640_di_calibrate_writes_solutions() {
    let tmp_dir = TempDir::new().expect("couldn't make tmp dir").into_path();
    let args = get_reduced_1090008640(true, false);
    let data = args.data.unwrap();
    let metafits = &data[0];
    let gpufits = &data[1];
    let sols = tmp_dir.join("sols.fits");
    let cal_model = tmp_dir.join("hyp_model.uvfits");

    #[rustfmt::skip]
    let cal_args = CalibrateUserArgs::parse_from(&[
        "di-calibrate",
        "--data", metafits, gpufits,
        "--source-list", &args.source_list.unwrap(),
        "--outputs", &format!("{}", sols.display()),
        "--model-filename", &format!("{}", cal_model.display()),
    ]);

    // Run di-cal and check that it succeeds
    let result = di_calibrate::<PathBuf>(Box::new(cal_args), None, false);
    assert!(result.is_ok(), "result={:?} not ok", result.err().unwrap());

    // check solutions file has been created, is readable
    assert!(sols.exists(), "sols file not written");
    let sol_data = CalibrationSolutions::read_solutions_from_ext(sols, metafits.into()).unwrap();
    assert_eq!(sol_data.obsid, Some(1090008640));
}

#[test]
fn test_1090008640_woden() {
    let tmp_dir = TempDir::new().expect("couldn't make tmp dir").into_path();
    let solutions_path = tmp_dir.join("sols.bin");

    // Reading from a uvfits file without a metafits file should fail because
    // there's no beam information.
    #[rustfmt::skip]
    let cal_args = CalibrateUserArgs::parse_from(&[
        "di-calibrate",
        "--data", "test_files/1090008640_WODEN/output_band01.uvfits",
        "--source-list", "test_files/1090008640_WODEN/srclist_3x3_grid.txt",
        "--outputs", &format!("{}", solutions_path.display()),
    ]);

    // Run di-cal and check that it fails
    let result = di_calibrate::<PathBuf>(Box::new(cal_args), None, false);
    assert!(
        matches!(result, Err(CalibrateError::InvalidArgs(_))),
        "result={:?} is not InvalidArgs",
        result.err().unwrap()
    );

    // This time give the metafits file.
    let cmd = hyperdrive()
        .args(&[
            "di-calibrate",
            "--data",
            "test_files/1090008640_WODEN/output_band01.uvfits",
            "test_files/1090008640_WODEN/1090008640.metafits",
            "--source-list",
            "test_files/1090008640_WODEN/srclist_3x3_grid.txt",
            #[cfg(feature = "cuda")]
            "--cpu",
            "--outputs",
            &format!("{}", solutions_path.display()),
        ])
        .ok();
    assert!(cmd.is_ok(), "{:?}", get_cmd_output(cmd));
    let (stdout, _) = get_cmd_output(cmd);

    // Verify that none of the calibration solutions are failures (i.e. not set
    // to NaN).
    let mut found_a_chanblock_line = false;
    for line in stdout.lines() {
        if line.starts_with("Chanblock") {
            found_a_chanblock_line = true;
            assert!(
                !line.contains("failed"),
                "Expected no lines with 'failed': {}",
                line
            );
        }
    }
    assert!(
        found_a_chanblock_line,
        "No 'Chanblock' lines found. Has the code changed?"
    );

    let metafits: Option<PathBuf> = None;
    let bin_sols =
        CalibrationSolutions::read_solutions_from_ext(&solutions_path, metafits.as_ref()).unwrap();
    assert_eq!(bin_sols.di_jones.dim(), (1, 128, 32));
    assert_eq!(bin_sols.start_timestamps.len(), 1);
    assert_eq!(bin_sols.end_timestamps.len(), 1);
    assert_eq!(bin_sols.average_timestamps.len(), 1);
    assert_abs_diff_eq!(
        bin_sols.start_timestamps[0].as_gpst_seconds(),
        // output_band01 lists the start time as 1090008640, but it should
        // probably be 1090008642.
        1090008640.0
    );
    assert_abs_diff_eq!(bin_sols.end_timestamps[0].as_gpst_seconds(), 1090008640.0);
    assert_abs_diff_eq!(
        bin_sols.average_timestamps[0].as_gpst_seconds(),
        1090008640.0,
    );
    assert!(!bin_sols.di_jones.iter().any(|jones| jones.any_nan()));

    // Re-do calibration, but this time into the hyperdrive fits format.
    let solutions_path = tmp_dir.join("sols.fits");

    let cal_args = CalibrateUserArgs::parse_from(&[
        "di-calibrate",
        "--data",
        "test_files/1090008640_WODEN/output_band01.uvfits",
        "test_files/1090008640_WODEN/1090008640.metafits",
        "--source-list",
        "test_files/1090008640_WODEN/srclist_3x3_grid.txt",
        #[cfg(feature = "cuda")]
        "--cpu",
        "--outputs",
        &format!("{}", solutions_path.display()),
    ]);

    // Run di-cal and check that it fails
    let result = di_calibrate::<PathBuf>(Box::new(cal_args), None, false);
    assert!(result.is_ok(), "result={:?} not ok", result.err().unwrap());

    let hyp_sols = result.unwrap().unwrap();
    assert_eq!(hyp_sols.di_jones.dim(), bin_sols.di_jones.dim());
    assert_eq!(hyp_sols.start_timestamps.len(), 1);
    assert_eq!(hyp_sols.end_timestamps.len(), 1);
    assert_eq!(hyp_sols.average_timestamps.len(), 1);
    assert_abs_diff_eq!(hyp_sols.start_timestamps[0].as_gpst_seconds(), 1090008640.0);
    assert_abs_diff_eq!(hyp_sols.end_timestamps[0].as_gpst_seconds(), 1090008640.0);
    assert_abs_diff_eq!(
        hyp_sols.average_timestamps[0].as_gpst_seconds(),
        1090008640.0
    );
    assert!(!hyp_sols.di_jones.iter().any(|jones| jones.any_nan()));

    let bin_sols_di_jones = bin_sols.di_jones.mapv(TestJones::from);
    let hyp_sols_di_jones = hyp_sols.di_jones.mapv(TestJones::from);
    assert_abs_diff_eq!(bin_sols_di_jones, hyp_sols_di_jones);
}

#[test]
#[serial]
fn test_1090008640_di_calibrate_writes_vis_uvfits() {
    let tmp_dir = TempDir::new().expect("couldn't make tmp dir").into_path();
    let args = get_reduced_1090008640(true, false);
    let data = args.data.unwrap();
    let metafits = &data[0];
    let gpufits = &data[1];
    let out_vis_path = tmp_dir.join("vis.uvfits");
    let cal_model = tmp_dir.join("hyp_model.uvfits");

    #[rustfmt::skip]
    let cal_args = CalibrateUserArgs::parse_from(&[
        "di-calibrate",
        "--data", metafits, gpufits,
        "--source-list", &args.source_list.unwrap(),
        "--outputs", &format!("{}", out_vis_path.display()),
        "--model-filename", &format!("{}", cal_model.display()),
    ]);

    // Run di-cal and check that it succeeds
    let result = di_calibrate::<PathBuf>(Box::new(cal_args), None, false);
    assert!(result.is_ok(), "result={:?} not ok", result.err().unwrap());

    // check vis file has been created, is readable
    assert!(out_vis_path.exists(), "out vis file not written");
    let exp_timesteps = 1;
    let exp_baselines = 8128;
    let exp_channels = 32;

    let mut out_vis = fits_open!(&out_vis_path).unwrap();
    let hdu0 = fits_open_hdu!(&mut out_vis, 0).unwrap();
    let gcount: String = get_required_fits_key!(&mut out_vis, &hdu0, "GCOUNT").unwrap();
    assert_eq!(
        gcount.parse::<usize>().unwrap(),
        exp_timesteps * exp_baselines
    );
    let num_fine_freq_chans: String =
        get_required_fits_key!(&mut out_vis, &hdu0, "NAXIS4").unwrap();
    assert_eq!(num_fine_freq_chans.parse::<usize>().unwrap(), exp_channels);
}

#[test]
#[serial]
fn test_1090008640_di_calibrate_writes_vis_uvfits_avg_freq() {
    let tmp_dir = TempDir::new().expect("couldn't make tmp dir").into_path();
    let args = get_reduced_1090008640(true, false);
    let data = args.data.unwrap();
    let metafits = &data[0];
    let gpufits = &data[1];
    let out_vis_path = tmp_dir.join("vis.uvfits");
    let cal_model = tmp_dir.join("hyp_model.uvfits");

    let freq_avg_factor = 2;

    #[rustfmt::skip]
    let cal_args = CalibrateUserArgs::parse_from(&[
        "di-calibrate",
        "--data", metafits, gpufits,
        "--source-list", &args.source_list.unwrap(),
        "--outputs", &format!("{}", out_vis_path.display()),
        "--model-filename", &format!("{}", cal_model.display()),
        "--output-vis-freq-average", &format!("{}", freq_avg_factor)
    ]);

    // Run di-cal and check that it succeeds
    let result = di_calibrate::<PathBuf>(Box::new(cal_args), None, false);
    assert!(result.is_ok(), "result={:?} not ok", result.err().unwrap());

    // check vis file has been created, is readable
    assert!(out_vis_path.exists(), "out vis file not written");
    let exp_timesteps = 1;
    let exp_baselines = 8128;
    let exp_channels = 32 / freq_avg_factor;

    let mut out_vis = fits_open!(&out_vis_path).unwrap();
    let hdu0 = fits_open_hdu!(&mut out_vis, 0).unwrap();
    let gcount: String = get_required_fits_key!(&mut out_vis, &hdu0, "GCOUNT").unwrap();
    assert_eq!(
        gcount.parse::<usize>().unwrap(),
        exp_timesteps * exp_baselines
    );
    let num_fine_freq_chans: String =
        get_required_fits_key!(&mut out_vis, &hdu0, "NAXIS4").unwrap();
    assert_eq!(num_fine_freq_chans.parse::<usize>().unwrap(), exp_channels);
}
