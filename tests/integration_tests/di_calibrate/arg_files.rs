// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! This module tests the "calibrate" command-line interface in hyperdrive with
//! toml and json argument files.

use tempfile::tempdir;

use crate::{
    get_cmd_output, get_reduced_1090008640, hyperdrive, make_file_in_dir, serialise_cal_args_json,
    serialise_cal_args_toml,
};

#[test]
fn arg_file_absolute_paths() {
    let args = get_reduced_1090008640(false, true);
    let temp_dir = tempdir().expect("Couldn't make tempdir");

    let (toml, mut toml_file) = make_file_in_dir("calibrate.toml", temp_dir.path());
    serialise_cal_args_toml(&args, &mut toml_file);
    let cmd = hyperdrive()
        .arg("di-calibrate")
        .arg(toml.display().to_string())
        .arg("--dry-run")
        .ok();
    assert!(cmd.is_ok(), "{}", get_cmd_output(cmd).1);

    let (json, mut json_file) = make_file_in_dir("calibrate.json", temp_dir.path());
    serialise_cal_args_json(&args, &mut json_file);
    let cmd = hyperdrive()
        .arg("di-calibrate")
        .arg(json.display().to_string())
        .arg("--dry-run")
        .ok();
    assert!(cmd.is_ok(), "{}", get_cmd_output(cmd).1);
}

#[test]
fn arg_file_absolute_globs() {
    let args = get_reduced_1090008640(false, true);
    let temp_dir = tempdir().expect("Couldn't make tempdir");

    let (toml_pb, mut toml) = make_file_in_dir("calibrate.toml", temp_dir.path());
    serialise_cal_args_toml(&args, &mut toml);
    let cmd = hyperdrive()
        .arg("di-calibrate")
        .arg(toml_pb.display().to_string())
        .arg("--dry-run")
        .ok();
    assert!(cmd.is_ok(), "{}", get_cmd_output(cmd).1);

    let (json_pb, mut json) = make_file_in_dir("calibrate.json", temp_dir.path());
    serialise_cal_args_json(&args, &mut json);
    let cmd = hyperdrive()
        .arg("di-calibrate")
        .arg(json_pb.display().to_string())
        .arg("--dry-run")
        .arg("--verb")
        .ok();
    assert!(cmd.is_ok(), "{}", get_cmd_output(cmd).1);
}

#[test]
fn arg_file_relative_globs() {
    let mut args = get_reduced_1090008640(false, true);
    args.data = Some(vec![
        "test_files/1090008640/*.metafits".to_string(),
        "test_files/1090008640/*gpubox*".to_string(),
        "test_files/1090008640/*.mwaf".to_string(),
    ]);
    args.source_list = Some("test_files/1090008640/*srclist*_100.yaml".to_string());

    let temp_dir = tempdir().expect("Couldn't make tempdir");

    let (toml_pb, mut toml) = make_file_in_dir("calibrate.toml", temp_dir.path());
    serialise_cal_args_toml(&args, &mut toml);
    let cmd = hyperdrive()
        .arg("di-calibrate")
        .arg(toml_pb.display().to_string())
        .arg("--dry-run")
        .ok();
    assert!(cmd.is_ok(), "{}", get_cmd_output(cmd).1);

    let (json_pb, mut json) = make_file_in_dir("calibrate.json", temp_dir.path());
    serialise_cal_args_json(&args, &mut json);
    let cmd = hyperdrive()
        .arg("di-calibrate")
        .arg(json_pb.display().to_string())
        .arg("--dry-run")
        .ok();
    assert!(cmd.is_ok(), "{}", get_cmd_output(cmd).1);
}
