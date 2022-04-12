// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Integration tests and helpful functions.

mod calibrate;
pub(crate) mod reduced_obsids;

use std::fs::File;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::Output;
use std::str::from_utf8;

use assert_cmd::{output::OutputError, Command};
use tempfile::{NamedTempFile, TempPath};

pub(crate) use crate::calibrate::args::CalibrateUserArgs;
use mwa_hyperdrive_common::{flate2, serde_json, toml};

/// Use `assert_cmd` to run the hyperdrive binary with "di-calibrate". If
/// `current_dir` is supplied, then it is used as the working directory for the
/// child process.
pub(crate) fn hyperdrive_di_calibrate(current_dir: Option<&Path>) -> Command {
    let mut cmd = Command::cargo_bin("hyperdrive").unwrap();
    cmd.arg("di-calibrate");
    if let Some(d) = current_dir {
        cmd.current_dir(d);
    }
    cmd
}

pub(crate) fn get_cmd_output(result: Result<Output, OutputError>) -> (String, String) {
    let output = match result {
        Ok(o) => o,
        Err(o) => o.as_output().unwrap().clone(),
    };
    (
        from_utf8(&output.stdout).unwrap().to_string(),
        from_utf8(&output.stderr).unwrap().to_string(),
    )
}

pub(crate) fn make_file_in_dir<T: AsRef<Path>, U: AsRef<Path>>(
    filename: T,
    dir: U,
) -> (PathBuf, File) {
    let path = dir.as_ref().join(filename);
    let f = File::create(&path).expect("couldn't make file");
    (path, f)
}

pub(crate) fn deflate_gz_into_file<T: AsRef<Path>>(gz_file: T, out_file: &mut File) {
    let mut gz = flate2::read::GzDecoder::new(File::open(gz_file).unwrap());
    std::io::copy(&mut gz, out_file).unwrap();
}

pub(crate) fn deflate_gz_into_tempfile<T: AsRef<Path>>(file: T) -> TempPath {
    let (mut temp_file, temp_path) = NamedTempFile::new().unwrap().into_parts();
    deflate_gz_into_file(file, &mut temp_file);
    temp_path
}

pub(crate) fn serialise_args_toml(args: &CalibrateUserArgs, file: &mut File) {
    let ser = toml::to_string_pretty(&args).expect("couldn't serialise CalibrateUserArgs as toml");
    write!(file, "{}", ser).unwrap();
}

pub(crate) fn serialise_args_json(args: &CalibrateUserArgs, file: &mut File) {
    let ser =
        serde_json::to_string_pretty(&args).expect("couldn't serialise CalibrateUserArgs as json");
    write!(file, "{}", ser).unwrap();
}
