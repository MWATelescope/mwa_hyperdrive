// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Tests against trying to calibrate but missing required files. Mostly useful
//! for checking that the error messages are of acceptable quality.

use std::io::Write;

use tempfile::Builder;

use crate::{get_cmd_output, get_reduced_1090008640, hyperdrive};

/// Try to calibrate raw MWA data without a metafits file.
#[test]
fn arg_file_missing_metafits() {
    let args = get_reduced_1090008640(false, true);
    let source_list = args.source_list.unwrap();
    let data = args.data.unwrap();
    let metafits = data
        .iter()
        .find(|d| d.contains("metafits"))
        .unwrap()
        .clone();
    let data: Vec<String> = data
        .into_iter()
        .filter(|d| !d.contains("metafits"))
        .collect();

    let cmd = hyperdrive()
        .arg("di-calibrate")
        .arg("--source-list")
        .arg(&source_list)
        .arg("--data")
        .args(&data)
        .arg("--dry-run")
        .ok();
    assert!(cmd.is_err());
    let (_, stderr) = get_cmd_output(cmd);
    assert!(
        stderr.contains("Received gpuboxes without a metafits file"),
        "{}",
        stderr
    );

    // Try again, but this time specifying the metafits file from the CLI.
    let cmd = hyperdrive()
        .arg("di-calibrate")
        .arg("--source-list")
        .arg(&source_list)
        .arg("--data")
        .arg(metafits)
        .args(&data)
        .arg("--dry-run")
        .ok();
    assert!(cmd.is_ok(), "{:?}", cmd.unwrap_err());
}

/// Try to calibrate raw MWA data without gpubox files.
#[test]
fn arg_file_missing_gpuboxes() {
    let args = get_reduced_1090008640(false, true);
    let source_list = args.source_list.unwrap();
    let data = args.data.unwrap();
    let gpuboxes: Vec<String> = data
        .iter()
        .filter(|d| d.contains("gpubox"))
        .cloned()
        .collect();
    let data: Vec<String> = data.into_iter().filter(|d| !d.contains("gpubox")).collect();

    let cmd = hyperdrive()
        .arg("di-calibrate")
        .arg("--source-list")
        .arg(&source_list)
        .arg("--data")
        .args(&data)
        .arg("--dry-run")
        .ok();
    assert!(cmd.is_err());
    let (_, stderr) = get_cmd_output(cmd);
    assert!(
        stderr.contains("Received only a metafits file and mwaf files"),
        "{}",
        stderr
    );

    // Try again, but this time specifying the gpubox files from the CLI.
    let cmd = hyperdrive()
        .arg("di-calibrate")
        .arg("--source-list")
        .arg(&source_list)
        .arg("--data")
        .args(&gpuboxes)
        .args(&data)
        .arg("--dry-run")
        .ok();
    assert!(cmd.is_ok(), "{:?}", cmd.unwrap_err());
}

/// Ensure that di-calibrate issues a warning when no mwaf files are supplied.
#[test]
fn missing_mwafs() {
    let args = get_reduced_1090008640(false, true);
    let source_list = args.source_list.unwrap();
    let data = args.data.unwrap();
    let metafits = &data[0];
    let gpubox = &data[1];
    let mwaf = &data[2];

    // Don't include an mwaf file; we expect a warning to be logged for this
    // reason.
    let cmd = hyperdrive()
        .arg("di-calibrate")
        .arg("--source-list")
        .arg(&source_list)
        .arg("--data")
        .arg(metafits)
        .arg(gpubox)
        .arg("--dry-run")
        .ok();
    assert!(cmd.is_ok(), "{:?}", cmd.unwrap_err());
    let (stdout, _) = get_cmd_output(cmd);
    assert!(stdout.contains("No mwaf files supplied"), "{}", stdout);

    // Include an mwaf file; we don't expect a warning this time.
    let cmd = hyperdrive()
        .arg("di-calibrate")
        .arg("--source-list")
        .arg(&source_list)
        .arg("--data")
        .arg(metafits)
        .arg(gpubox)
        .arg(mwaf)
        .arg("--dry-run")
        .ok();
    assert!(cmd.is_ok(), "{:?}", cmd.unwrap_err());
    let (stdout, _) = get_cmd_output(cmd);
    assert!(!stdout.contains("No mwaf files supplied"), "{}", stdout);
}

/// Ensure that di-calibrate issues a warning when no mwaf files are supplied
/// via argument files.
#[test]
fn arg_file_missing_mwafs() {
    let args = get_reduced_1090008640(false, false);
    let args_str = toml::to_vec(&args).unwrap();
    let mut args_file = Builder::new().suffix(".toml").tempfile().unwrap();
    args_file.write_all(&args_str).unwrap();

    // Don't include an mwaf file; we expect a warning to be logged for this
    // reason.
    let cmd = hyperdrive()
        .arg("di-calibrate")
        .arg(args_file.path().to_str().unwrap())
        .arg("--dry-run")
        .ok();
    assert!(cmd.is_ok(), "{:?}", cmd.unwrap_err());
    let (stdout, _) = get_cmd_output(cmd);
    assert!(stdout.contains("No mwaf files supplied"), "{}", stdout);

    // Include an mwaf file; we don't expect a warning this time.
    let args = get_reduced_1090008640(false, true);
    let args_str = toml::to_vec(&args).unwrap();
    let mut args_file = Builder::new().suffix(".toml").tempfile().unwrap();
    args_file.write_all(&args_str).unwrap();

    let cmd = hyperdrive()
        .arg("di-calibrate")
        .arg(args_file.path().to_str().unwrap())
        .arg("--dry-run")
        .ok();
    assert!(cmd.is_ok(), "{:?}", cmd.unwrap_err());
    let (stdout, _) = get_cmd_output(cmd);
    assert!(!stdout.contains("No mwaf files supplied"), "{}", stdout);
}
