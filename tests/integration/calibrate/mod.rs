// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Integration tests for calibration.

mod cli_args;

use approx::assert_abs_diff_eq;
use marlu::time::epoch_as_gps_seconds;

use crate::*;
use mwa_hyperdrive::calibrate::solutions::CalibrationSolutions;
use mwa_hyperdrive_common::marlu;

#[test]
fn test_1090008640_woden() {
    let tmp_dir = TempDir::new().expect("couldn't make tmp dir").into_path();
    let mut solutions_path = tmp_dir.clone();
    solutions_path.push("sols.bin");

    // Reading from a uvfits file without a metafits file should fail.
    let cmd = hyperdrive()
        .args(&[
            "di-calibrate",
            "--data",
            "test_files/1090008640_WODEN/output_band01.uvfits",
            "--source-list",
            "test_files/1090008640_WODEN/srclist_3x3_grid.txt",
            "--outputs",
            &format!("{}", solutions_path.display()),
        ])
        .ok();
    assert!(cmd.is_err(), "{:?}", get_cmd_output(cmd));

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
    let (stdout, stderr) = get_cmd_output(cmd);
    assert!(stderr.is_empty(), "{}", stderr);

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
    assert_abs_diff_eq!(
        epoch_as_gps_seconds(*bin_sols.start_timestamps.first().unwrap()),
        // 1090008642 is the obsid + 2s, which is the centroid of the first and
        // only timestep.
        1090008642.0,
        epsilon = 1e-3
    );
    assert_abs_diff_eq!(
        epoch_as_gps_seconds(*bin_sols.start_timestamps.last().unwrap()),
        1090008642.0,
        epsilon = 1e-3
    );
    assert!(!bin_sols.di_jones.iter().any(|jones| jones.any_nan()));

    // Re-do calibration, but this time into the hyperdrive fits format.
    let mut solutions_path = tmp_dir;
    solutions_path.push("sols.fits");

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
    let (_, stderr) = get_cmd_output(cmd);
    assert!(stderr.is_empty());

    let hyp_sols =
        CalibrationSolutions::read_solutions_from_ext(&solutions_path, metafits).unwrap();
    assert_eq!(hyp_sols.di_jones.dim(), bin_sols.di_jones.dim());
    assert_abs_diff_eq!(
        epoch_as_gps_seconds(*hyp_sols.start_timestamps.first().unwrap()),
        1090008642.0,
        epsilon = 1e-3
    );
    assert_abs_diff_eq!(
        epoch_as_gps_seconds(*hyp_sols.start_timestamps.last().unwrap()),
        1090008642.0,
        epsilon = 1e-3
    );
    assert!(!hyp_sols.di_jones.iter().any(|jones| jones.any_nan()));

    let bin_sols_di_jones = bin_sols.di_jones.mapv(TestJones::from);
    let hyp_sols_di_jones = hyp_sols.di_jones.mapv(TestJones::from);
    assert_abs_diff_eq!(bin_sols_di_jones, hyp_sols_di_jones, epsilon = 1e-7);
}
