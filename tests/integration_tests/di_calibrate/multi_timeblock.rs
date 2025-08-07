// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Test multi-timeblock calibration to reproduce and verify the fix for issue #59.

use tempfile::TempDir;

use crate::{get_cmd_output, hyperdrive};

/// Test that reproduces the multi-timeblock calibration bug and verifies the fix.
/// This test reproduces the issue where individual timeblocks would fail to converge
/// (0/N chanblocks) even when the all-timesteps calibration succeeded, due to
/// inappropriate initial guesses being copied from the all-timesteps solution.
///
/// The fix was to use identity matrices as initial guesses for individual timeblocks
/// instead of copying the all-timesteps solution.
#[test]
fn test_multi_timeblock_calibration_1061316544() {
    let tmp_dir = TempDir::new().expect("couldn't make tmp dir");
    let solutions_path = tmp_dir.path().join("dical.fits");

    // Run calibration with small timeblocks to trigger the multi-timeblock path
    let cmd = hyperdrive()
        .arg("di-calibrate")
        .arg("--data")
        .arg("test_files/1061316544/1061316544.metafits")
        .arg("test_files/1061316544/1061316544.uvfits")
        .arg("--source-list")
        .arg("test_files/1061316544/srclist_1061316544.yaml")
        .arg("--outputs")
        .arg(format!("{}", solutions_path.display()))
        .arg("--timesteps-per-timeblock")
        .arg("2") // Small timeblocks to trigger the multi-timeblock path
        .arg("--max-iterations")
        .arg("20") // Fewer iterations for speed
        .arg("--freq-average")
        .arg("320kHz") // Larger averaging for speed
        .arg("--no-progress-bars")
        .ok();

    // The command should succeed with the fix
    assert!(
        cmd.is_ok(),
        "Multi-timeblock calibration failed: {:?}",
        get_cmd_output(cmd)
    );
    let (stdout, stderr) = get_cmd_output(cmd);

    // Verify there are no error messages in stderr
    assert!(
        !stderr.contains("error"),
        "Unexpected error in stderr: {}",
        stderr
    );
    assert!(
        !stderr.contains("failed"),
        "Unexpected failure in stderr: {}",
        stderr
    );

    // Parse the stdout to verify the calibration results
    let mut found_timeblock_results = false;
    let mut timeblock_convergence_failures = 0;
    let mut total_timeblocks = 0;

    for line in stdout.lines() {
        // Check individual timeblock results
        if line.contains("Timeblock")
            && line.contains("chanblocks converged")
            && !line.contains("All timesteps")
        {
            found_timeblock_results = true;
            total_timeblocks += 1;

            // Extract convergence percentage - should be > 0% with the fix
            // Look for patterns like "(0/4)" or "0%" which indicate complete failure
            if line.contains(" 0/") || (line.contains("0%") && !line.contains("100%")) {
                timeblock_convergence_failures += 1;
            }
        }
    }

    // Verify the expected calibration pattern

    assert!(
        found_timeblock_results,
        "Should find individual timeblock results. Stdout: {}",
        stdout
    );

    assert!(
        total_timeblocks > 1,
        "Should have multiple timeblocks (found {}). Stdout: {}",
        total_timeblocks,
        stdout
    );

    // The critical test: with the fix, most individual timeblocks should succeed.
    // Some timeblocks may fail if they contain primarily flagged data, but not ALL should fail.
    // Before the fix, ALL timeblocks would show 0% convergence due to bad initial guesses.
    assert!(
        timeblock_convergence_failures < total_timeblocks,
        "Found {} out of {} timeblocks with 0% convergence. This indicates the multi-timeblock bug \
         if ALL timeblocks fail. Some timeblocks may legitimately fail if they contain bad data, \
         but the majority should succeed with the fix. Stdout: {}",
        timeblock_convergence_failures, total_timeblocks, stdout
    );

    // Verify the solutions file was created and is valid
    assert!(
        solutions_path.exists(),
        "Calibration solutions file should be created"
    );

    // Try to read the solutions to ensure they're valid
    let sols = mwa_hyperdrive::CalibrationSolutions::read_solutions_from_ext(
        &solutions_path,
        None::<&std::path::PathBuf>,
    );
    assert!(
        sols.is_ok(),
        "Should be able to read the calibration solutions: {:?}",
        sols.err()
    );

    let sols = sols.unwrap();
    // Verify we have multiple timeblocks in the solutions
    assert!(
        sols.di_jones.dim().0 > 1,
        "Solutions should contain multiple timeblocks, got shape: {:?}",
        sols.di_jones.dim()
    );

    // Verify solutions are not all NaN (which would indicate calibration failure)
    let all_nan = sols.di_jones.iter().all(|jones| jones.any_nan());
    assert!(!all_nan, "Calibration solutions should not be all NaN");
}
