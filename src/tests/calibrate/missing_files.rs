// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Tests against trying to calibrate but missing required files. Mostly useful
//! for checking that the error messages are of acceptable quality.

use crate::{
    calibrate::params::InvalidArgsError,
    tests::{reduced_obsids::get_reduced_1090008640, *},
};

/// Try to calibrate raw MWA data without a metafits file.
#[test]
fn param_file_missing_metafits() {
    let args = get_reduced_1090008640(true);
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
        .ok();
    assert!(cmd.is_err());
    let (_, stderr) = get_cmd_output(cmd);
    assert!(
        stderr.contains(&InvalidArgsError::InvalidDataInput.to_string()),
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
        .ok();
    assert!(cmd.is_ok(), "{:?}", cmd.unwrap_err());
}

/// Try to calibrate raw MWA data without gpubox files.
#[test]
fn param_file_missing_gpuboxes() {
    let args = get_reduced_1090008640(true);
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
        .ok();
    assert!(cmd.is_err());
    let (_, stderr) = get_cmd_output(cmd);
    assert!(
        stderr.contains(&InvalidArgsError::InvalidDataInput.to_string()),
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
        .ok();
    assert!(cmd.is_ok(), "{:?}", cmd.unwrap_err());
}

/// Make a toml argument file without mwaf files.
#[test]
fn param_file_missing_mwafs() {
    let args = get_reduced_1090008640(true);
    let source_list = args.source_list.unwrap();
    let data = args.data.unwrap();
    let mwafs: Vec<String> = data
        .iter()
        .filter(|d| d.contains("mwaf"))
        .cloned()
        .collect();
    let data: Vec<String> = data.into_iter().filter(|d| !d.contains("mwaf")).collect();

    // Because the mwaf files are optional, the command should succeed but issue
    // a warning.
    let cmd = hyperdrive()
        .arg("di-calibrate")
        .arg("--source-list")
        .arg(&source_list)
        .arg("--data")
        .args(&data)
        .ok();
    assert!(cmd.is_ok(), "{:?}", cmd.unwrap_err());
    let (stdout, _) = get_cmd_output(cmd);
    assert!(
        &stdout.contains("No mwaf flag files supplied"),
        "{}",
        stdout
    );

    // Check that the warning isn't present when the mwafs are actually given.
    let cmd = hyperdrive()
        .arg("di-calibrate")
        .arg("--source-list")
        .arg(&source_list)
        .arg("--data")
        .args(&data)
        .args(&mwafs)
        .ok();
    assert!(cmd.is_ok(), "{:?}", cmd.unwrap_err());
    let (stdout, _) = get_cmd_output(cmd);
    assert!(
        !&stdout.contains("No mwaf flag files supplied"),
        "{}",
        stdout
    );
}
