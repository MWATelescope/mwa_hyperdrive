// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Code for calibration testing.

mod cli_args;
mod missing_files;
mod multi_timeblock;

use std::path::PathBuf;

use approx::assert_abs_diff_eq;
use tempfile::TempDir;

use crate::{get_cmd_output, hyperdrive};
use mwa_hyperdrive::CalibrationSolutions;

#[test]
fn test_1090008640_woden() {
    let tmp_dir = TempDir::new().expect("couldn't make tmp dir");
    let solutions_path = tmp_dir.path().join("sols.bin");

    // Reading from a uvfits file without a metafits file should fail because
    // there's no beam information.
    let cmd = hyperdrive()
        .arg("di-calibrate")
        .arg("--data")
        .arg("test_files/1090008640_WODEN/output_band01.uvfits")
        .arg("--source-list")
        .arg("test_files/1090008640_WODEN/srclist_3x3_grid.txt")
        .arg("--outputs")
        .arg(format!("{}", solutions_path.display()))
        .arg("--no-progress-bars")
        .ok();
    assert!(cmd.is_err());
    let (_, stderr) = get_cmd_output(cmd);
    assert!(stderr.contains(
        "Tried to set up a 'FEE' beam, which requires MWA dipole delays, but none are available"
    ));

    // This time give the metafits file.
    let cmd = hyperdrive()
        .arg("di-calibrate")
        .arg("--data")
        .arg("test_files/1090008640_WODEN/output_band01.uvfits")
        .arg("test_files/1090008640_WODEN/1090008640.metafits")
        .arg("--source-list")
        .arg("test_files/1090008640_WODEN/srclist_3x3_grid.txt")
        .arg("--outputs")
        .arg(format!("{}", solutions_path.display()))
        .arg("--no-progress-bars")
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
                "Expected no lines with 'failed': {line}"
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
        bin_sols.start_timestamps.unwrap()[0].to_gpst_seconds(),
        // output_band01 lists the start time as 1090008640, but it should
        // probably be 1090008642.
        1090008640.0
    );
    assert_abs_diff_eq!(
        bin_sols.end_timestamps.unwrap()[0].to_gpst_seconds(),
        1090008640.0
    );
    assert_abs_diff_eq!(
        bin_sols.average_timestamps.unwrap()[0].to_gpst_seconds(),
        1090008640.0,
    );
    assert!(!bin_sols.di_jones.iter().any(|jones| jones.any_nan()));

    // Re-do calibration, but this time into the hyperdrive fits format.
    let solutions_path = tmp_dir.path().join("sols.fits");

    let cmd = hyperdrive()
        .arg("di-calibrate")
        .arg("--data")
        .arg("test_files/1090008640_WODEN/output_band01.uvfits")
        .arg("test_files/1090008640_WODEN/1090008640.metafits")
        .arg("--source-list")
        .arg("test_files/1090008640_WODEN/srclist_3x3_grid.txt")
        .arg("--outputs")
        .arg(format!("{}", solutions_path.display()))
        .arg("--no-progress-bars")
        .ok();

    // Run di-cal and check that it succeeds
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
                "Expected no lines with 'failed': {line}"
            );
        }
    }
    assert!(
        found_a_chanblock_line,
        "No 'Chanblock' lines found. Has the code changed?"
    );

    let hyp_sols =
        CalibrationSolutions::read_solutions_from_ext(solutions_path, metafits.as_ref()).unwrap();
    assert_eq!(hyp_sols.di_jones.dim(), bin_sols.di_jones.dim());
    assert_eq!(hyp_sols.start_timestamps.as_ref().map(|v| v.len()), Some(1));
    assert_eq!(hyp_sols.end_timestamps.as_ref().map(|v| v.len()), Some(1));
    assert_eq!(
        hyp_sols.average_timestamps.as_ref().map(|v| v.len()),
        Some(1)
    );
    assert_abs_diff_eq!(
        hyp_sols.start_timestamps.unwrap()[0].to_gpst_seconds(),
        1090008640.0
    );
    assert_abs_diff_eq!(
        hyp_sols.end_timestamps.unwrap()[0].to_gpst_seconds(),
        1090008640.0
    );
    assert_abs_diff_eq!(
        hyp_sols.average_timestamps.unwrap()[0].to_gpst_seconds(),
        1090008640.0
    );
    assert!(!hyp_sols.di_jones.iter().any(|jones| jones.any_nan()));

    assert_abs_diff_eq!(bin_sols.di_jones, hyp_sols.di_jones);
}
