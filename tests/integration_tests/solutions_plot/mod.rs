// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Integration tests for solutions-plot.

use std::path::Path;

use marlu::Jones;
use mwa_hyperdrive::CalibrationSolutions;
use ndarray::prelude::*;
use tempfile::TempDir;

use crate::{get_cmd_output, get_reduced_1090008640, hyperdrive, Files};

/// Solutions with `num_tiles` tiles, all identity, so that they aren't
/// flagged. `leakage` is written into the off-diagonal terms of every tile.
fn write_solutions(tmp_dir: &Path, num_tiles: usize, leakage: f64) -> String {
    let mut jones = Jones::identity();
    jones[1] = num_complex::Complex::new(leakage, 0.0);
    jones[2] = num_complex::Complex::new(leakage, 0.0);
    let sols = CalibrationSolutions {
        di_jones: Array3::from_elem((1, num_tiles, 32), jones),
        ..Default::default()
    };
    let file = tmp_dir.join("sols.fits");
    sols.write_solutions_from_ext::<&Path>(&file).unwrap();
    file.display().to_string()
}

/// A metafits with fewer tiles than the solutions used to panic with an
/// index-out-of-bounds; it should be a proper error.
#[test]
#[cfg(feature = "plotting")]
fn test_metafits_with_too_few_tiles_is_an_error() {
    let tmp_dir = TempDir::new().expect("couldn't make tmp dir");
    let sols = write_solutions(tmp_dir.path(), 256, 0.0);
    let Files { data, .. } = get_reduced_1090008640(false);
    let metafits = data[0].clone();

    #[rustfmt::skip]
    let cmd = hyperdrive()
        .args([
            "solutions-plot",
            &sols,
            "--metafits", &metafits,
            "--output-directory", &format!("{}", tmp_dir.path().display()),
        ])
        .ok();
    assert!(
        cmd.is_err(),
        "solutions-plot should fail with a mismatched metafits"
    );
    let (_, stderr) = get_cmd_output(cmd);
    assert!(
        !stderr.contains("panicked"),
        "solutions-plot panicked instead of returning an error: {stderr}"
    );
    assert!(
        stderr.contains("There are 128 tile names but the solutions have 256 tiles"),
        "unexpected stderr: {stderr}"
    );
}

/// Tiles with valid gains but NaN leakage terms should still be plotted when
/// the cross pols are ignored.
#[test]
#[cfg(feature = "plotting")]
fn test_nan_leakages_with_ignore_cross_pols_plots_ok() {
    let tmp_dir = TempDir::new().expect("couldn't make tmp dir");
    let sols = write_solutions(tmp_dir.path(), 128, f64::NAN);

    #[rustfmt::skip]
    let cmd = hyperdrive()
        .args([
            "solutions-plot",
            &sols,
            "--ignore-cross-pols",
            "--output-directory", &format!("{}", tmp_dir.path().display()),
        ])
        .ok();
    assert!(cmd.is_ok(), "solutions-plot failed: {}", cmd.err().unwrap());
    assert!(tmp_dir.path().join("sols_amps.png").exists());
    assert!(tmp_dir.path().join("sols_phases.png").exists());
}
