// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! This module tests the "calibrate" command-line interface in hyperdrive with
//! toml and json argument files.

use std::path::PathBuf;

use tempfile::tempdir;

use crate::tests::{
    get_cmd_output, hyperdrive_di_calibrate, make_file_in_dir,
    reduced_obsids::get_reduced_1090008640, serialise_args_json, serialise_args_toml,
};

#[test]
#[ignore = "assert_cmd_not_working_in_coverage"]
fn arg_file_absolute_paths() {
    let args = get_reduced_1090008640(true);
    let temp_dir = tempdir().expect("Couldn't make tempdir");

    let (toml, mut toml_file) = make_file_in_dir("calibrate.toml", temp_dir.path());
    serialise_args_toml(&args, &mut toml_file);
    let cmd = hyperdrive_di_calibrate(None)
        .arg(toml.display().to_string())
        .arg("--dry-run")
        .ok();
    assert!(cmd.is_ok(), "{}", get_cmd_output(cmd).1);

    let (json, mut json_file) = make_file_in_dir("calibrate.json", temp_dir.path());
    serialise_args_json(&args, &mut json_file);
    let cmd = hyperdrive_di_calibrate(None)
        .arg(json.display().to_string())
        .arg("--dry-run")
        .ok();
    assert!(cmd.is_ok(), "{}", get_cmd_output(cmd).1);
}

#[test]
#[ignore = "assert_cmd_not_working_in_coverage"]
fn arg_file_absolute_globs() {
    let mut args = get_reduced_1090008640(true);
    let abs_path = PathBuf::from(&args.data.unwrap()[0])
        .canonicalize()
        .unwrap()
        .parent()
        .unwrap()
        .display()
        .to_string();
    args.data = Some(vec![
        format!("{}/*.metafits", abs_path),
        format!("{}/*gpubox*", abs_path),
        format!("{}/*.mwaf", abs_path),
    ]);
    args.source_list = Some(format!("{}/*srclist*_100.yaml", abs_path));

    let temp_dir = tempdir().expect("Couldn't make tempdir");

    let (toml_pb, mut toml) = make_file_in_dir(&"calibrate.toml", temp_dir.path());
    serialise_args_toml(&args, &mut toml);
    let cmd = hyperdrive_di_calibrate(None)
        .arg(toml_pb.display().to_string())
        .arg("--dry-run")
        .ok();
    assert!(cmd.is_ok(), "{}", get_cmd_output(cmd).1);

    let (json_pb, mut json) = make_file_in_dir(&"calibrate.json", temp_dir.path());
    serialise_args_json(&args, &mut json);
    let cmd = hyperdrive_di_calibrate(None)
        .arg(json_pb.display().to_string())
        .arg("--dry-run")
        .ok();
    assert!(cmd.is_ok(), "{}", get_cmd_output(cmd).1);
}

#[test]
#[ignore = "assert_cmd_not_working_in_coverage"]
fn arg_file_relative_globs() {
    let mut args = get_reduced_1090008640(true);
    let rel_path = PathBuf::from(&args.data.unwrap()[0])
        .parent()
        .unwrap()
        .display()
        .to_string();
    args.data = Some(vec![
        format!("{}/*.metafits", rel_path),
        format!("{}/*gpubox*", rel_path),
        format!("{}/*.mwaf", rel_path),
    ]);
    args.source_list = Some(format!("{}/*srclist*_100.yaml", rel_path));

    let temp_dir = tempdir().expect("Couldn't make tempdir");

    let (toml_pb, mut toml) = make_file_in_dir(&"calibrate.toml", temp_dir.path());
    serialise_args_toml(&args, &mut toml);
    let cmd = hyperdrive_di_calibrate(None)
        .arg(toml_pb.display().to_string())
        .arg("--dry-run")
        .ok();
    assert!(cmd.is_ok(), "{}", get_cmd_output(cmd).1);

    let (json_pb, mut json) = make_file_in_dir(&"calibrate.json", temp_dir.path());
    serialise_args_json(&args, &mut json);
    let cmd = hyperdrive_di_calibrate(None)
        .arg(json_pb.display().to_string())
        .arg("--dry-run")
        .ok();
    assert!(cmd.is_ok(), "{}", get_cmd_output(cmd).1);
}
