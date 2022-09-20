// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Integration tests for calibration.

mod cli_args;

use approx::assert_abs_diff_eq;
use clap::Parser;
use serial_test::serial;

use crate::*;
use mwa_hyperdrive::{calibrate::di_calibrate, solutions::CalibrationSolutions};
use mwa_hyperdrive_common::{clap, setup_logging};

#[test]
#[serial]
fn test_1090008640_di_calibrate_writes_solutions() {
    let tmp_dir = TempDir::new().expect("couldn't make tmp dir");
    let args = get_reduced_1090008640(true, false);
    let data = args.data.unwrap();
    let metafits = &data[0];
    let gpufits = &data[1];
    let sols = tmp_dir.path().join("sols.fits");
    let cal_model = tmp_dir.path().join("hyp_model.uvfits");

    #[rustfmt::skip]
    let cal_args = CalibrateUserArgs::parse_from(&[
        "di-calibrate",
        "--data", metafits, gpufits,
        "--source-list", &args.source_list.unwrap(),
        "--outputs", &format!("{}", sols.display()),
        "--model-filenames", &format!("{}", cal_model.display()),
        "--no-progress-bars",
    ]);

    // Run di-cal and check that it succeeds
    let result = di_calibrate(Box::new(cal_args), None, false);
    assert!(result.is_ok(), "result={:?} not ok", result.err().unwrap());

    // check solutions file has been created, is readable
    assert!(sols.exists(), "sols file not written");
    let sol_data = CalibrationSolutions::read_solutions_from_ext(sols, metafits.into()).unwrap();
    assert_eq!(sol_data.obsid, Some(1090008640));
}

#[test]
fn test_1090008640_woden() {
    let tmp_dir = TempDir::new().expect("couldn't make tmp dir");
    let solutions_path = tmp_dir.path().join("sols.bin");

    // Reading from a uvfits file without a metafits file should fail because
    // there's no beam information.
    #[rustfmt::skip]
    let cal_args = CalibrateUserArgs::parse_from(&[
        "di-calibrate",
        "--data", "test_files/1090008640_WODEN/output_band01.uvfits",
        "--source-list", "test_files/1090008640_WODEN/srclist_3x3_grid.txt",
        "--outputs", &format!("{}", solutions_path.display()),
        "--no-progress-bars",
    ]);

    // Run di-cal and check that it fails
    let result = di_calibrate(Box::new(cal_args), None, false);
    assert!(result.is_err());
    assert!(result.err().unwrap().to_string().contains(
        "Tried to create a beam object, but MWA dipole delay information isn't available!"
    ));

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
    assert_eq!(bin_sols.start_timestamps.as_ref().map(|v| v.len()), Some(1));
    assert_eq!(bin_sols.end_timestamps.as_ref().map(|v| v.len()), Some(1));
    assert_eq!(
        bin_sols.average_timestamps.as_ref().map(|v| v.len()),
        Some(1)
    );
    assert_abs_diff_eq!(
        bin_sols.start_timestamps.unwrap()[0].as_gpst_seconds(),
        // output_band01 lists the start time as 1090008640, but it should
        // probably be 1090008642.
        1090008640.0
    );
    assert_abs_diff_eq!(
        bin_sols.end_timestamps.unwrap()[0].as_gpst_seconds(),
        1090008640.0
    );
    assert_abs_diff_eq!(
        bin_sols.average_timestamps.unwrap()[0].as_gpst_seconds(),
        1090008640.0,
    );
    assert!(!bin_sols.di_jones.iter().any(|jones| jones.any_nan()));

    // Re-do calibration, but this time into the hyperdrive fits format.
    let solutions_path = tmp_dir.path().join("sols.fits");

    #[rustfmt::skip]
    let cal_args = CalibrateUserArgs::parse_from(&[
        "di-calibrate",
        "--data", "test_files/1090008640_WODEN/output_band01.uvfits", "test_files/1090008640_WODEN/1090008640.metafits",
        "--source-list", "test_files/1090008640_WODEN/srclist_3x3_grid.txt",
        "--outputs", &format!("{}", solutions_path.display()),
        "--no-progress-bars",
        #[cfg(feature = "cuda")]
        "--cpu",
    ]);

    // Run di-cal and check that it fails
    let result = di_calibrate(Box::new(cal_args), None, false);
    assert!(result.is_ok(), "result={:?} not ok", result.err().unwrap());

    let hyp_sols = result.unwrap().unwrap();
    assert_eq!(hyp_sols.di_jones.dim(), bin_sols.di_jones.dim());
    assert_eq!(hyp_sols.start_timestamps.as_ref().map(|v| v.len()), Some(1));
    assert_eq!(hyp_sols.end_timestamps.as_ref().map(|v| v.len()), Some(1));
    assert_eq!(
        hyp_sols.average_timestamps.as_ref().map(|v| v.len()),
        Some(1)
    );
    assert_abs_diff_eq!(
        hyp_sols.start_timestamps.unwrap()[0].as_gpst_seconds(),
        1090008640.0
    );
    assert_abs_diff_eq!(
        hyp_sols.end_timestamps.unwrap()[0].as_gpst_seconds(),
        1090008640.0
    );
    assert_abs_diff_eq!(
        hyp_sols.average_timestamps.unwrap()[0].as_gpst_seconds(),
        1090008640.0
    );
    assert!(!hyp_sols.di_jones.iter().any(|jones| jones.any_nan()));

    assert_abs_diff_eq!(bin_sols.di_jones, hyp_sols.di_jones);
}

/// Get the calibration arguments associated with the obsid 1233282520. This
/// observational data is inside the hyperdrive git repo, but has been reduced;
/// there is only 1 coarse channel and 1 timestep.
///
/// data generated with <https://github.com/MWATelescope/Birli/blob/main/tests/data/adjust_gpufits.py> `--timestep-limit=1`
fn get_reduced_1233282520() -> CalibrateUserArgs {
    // Ensure that the required files are there.
    let data = vec![
        "test_files/1233282520/1233282520_noquack.metafits".to_string(),
        "test_files/1233282520/1233282520_20190204022824_gpubox12_00.fits".to_string(),
        "test_files/1233282520/1233282520_20190204022824_gpubox13_00.fits".to_string(),
    ];
    for file in &data {
        let pb = PathBuf::from(file);
        assert!(
            pb.exists(),
            "Could not find {}, which is required for this test",
            pb.display()
        );
    }

    let srclist = "test_files/1233282520/1233282520_121-132_model.txt".to_string();
    assert!(
        PathBuf::from(&srclist).exists(),
        "Could not find {}, which is required for this test",
        srclist
    );

    CalibrateUserArgs {
        data: Some(data),
        source_list: Some(srclist),
        ..Default::default()
    }
}

#[test]
fn test_1233282520_picket() {
    let tmp_dir = TempDir::new().expect("couldn't make tmp dir");

    setup_logging(3).unwrap();

    let mut args = get_reduced_1233282520();
    let data = args.data.clone().unwrap();
    let metafits = &data[0];
    let sols = tmp_dir.path().join("sols.fits");
    args.outputs = Some(vec![sols.clone()]);
    args.no_progress_bars = true;

    // Run di-cal and check that it succeeds
    let result = di_calibrate(Box::new(args), None, false);
    assert!(result.is_ok(), "result={:?} not ok", result.err().unwrap());

    // check solutions file has been created, is readable
    assert!(sols.exists(), "sols file not written");
    let sol_data = CalibrationSolutions::read_solutions_from_ext(sols, metafits.into()).unwrap();
    assert_eq!(sol_data.obsid, Some(1233282520));
}
