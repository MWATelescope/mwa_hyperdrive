// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

mod args;
mod real_data;

use std::fs::File;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::Output;
use std::str::from_utf8;

use assert_cmd::{output::OutputError, Command};
use tempfile::TempDir;

use mwa_hyperdrive::calibrate::args::CalibrateUserArgs;

fn make_file_in_dir<T: AsRef<Path> + ?Sized>(filename: &T, dir: &Path) -> (PathBuf, File) {
    let mut path = dir.to_path_buf();
    path.push(filename);
    let f = File::create(&path).expect("couldn't make file");
    (path, f)
}

fn path_to_string(p: &Path) -> String {
    p.to_str().map(|s| s.to_string()).unwrap()
}

fn deflate_gz<T: AsRef<Path>>(gz_file: &T, out_file: &mut File) {
    let mut gz = flate2::read::GzDecoder::new(File::open(gz_file).unwrap());
    std::io::copy(&mut gz, out_file).unwrap();
}

fn serialise_args_toml(args: &CalibrateUserArgs, file: &mut File) {
    let ser = toml::to_string_pretty(&args).expect("couldn't serialise CalibrateUserArgs as toml");
    write!(file, "{}", ser).unwrap();
}

fn serialise_args_json(args: &CalibrateUserArgs, file: &mut File) {
    let ser =
        serde_json::to_string_pretty(&args).expect("couldn't serialise CalibrateUserArgs as json");
    write!(file, "{}", ser).unwrap();
}

fn get_cmd_output(result: Result<Output, OutputError>) -> (String, String) {
    let output = match result {
        Ok(o) => o,
        Err(o) => o.as_output().unwrap().clone(),
    };
    (
        from_utf8(&output.stdout).unwrap().to_string(),
        from_utf8(&output.stderr).unwrap().to_string(),
    )
}

fn hyperdrive() -> Command {
    Command::cargo_bin("hyperdrive").unwrap()
}
