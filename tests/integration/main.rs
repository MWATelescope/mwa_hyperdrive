// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Integration tests.
//!
//! Some help for laying out these tests was taken from:
//! https://matklad.github.io/2021/02/27/delete-cargo-integration-tests.html

mod calibrate;
mod jones_test;

use std::fs::File;
use std::path::{Path, PathBuf};
use std::process::Output;
use std::str::from_utf8;

use assert_cmd::{output::OutputError, Command};
use jones_test::*;
use tempfile::TempDir;

fn hyperdrive() -> Command {
    Command::cargo_bin("hyperdrive").unwrap()
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

fn make_file_in_dir<T: AsRef<Path>, U: AsRef<Path>>(filename: T, dir: U) -> (PathBuf, File) {
    let path = dir.as_ref().join(filename);
    let f = File::create(&path).expect("couldn't make file");
    (path, f)
}
