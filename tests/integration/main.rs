// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Integration tests.
//!
//! Some help for laying out these tests was taken from:
//! https://matklad.github.io/2021/02/27/delete-cargo-integration-tests.html

mod calibrate;
mod jones_test;
mod modelling;

use std::fs::File;
use std::path::{Path, PathBuf};
use std::process::Output;
use std::str::from_utf8;

use assert_cmd::{output::OutputError, Command};
use jones_test::*;
use tempfile::TempDir;

use mwa_hyperdrive::calibrate::args::CalibrateUserArgs;

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

/// Get the calibration arguments associated with the obsid 1090008640. This
/// observational data is inside the hyperdrive git repo, but has been reduced;
/// there is only 1 coarse channel and 1 timestep.
fn get_reduced_1090008640(use_fee_beam: bool, include_mwaf: bool) -> CalibrateUserArgs {
    // Ensure that the required files are there.
    let mut data = vec![
        "test_files/1090008640/1090008640.metafits".to_string(),
        "test_files/1090008640/1090008640_20140721201027_gpubox01_00.fits".to_string(),
    ];
    if include_mwaf {
        data.push("test_files/1090008640/1090008640_01.mwaf".to_string());
    }
    for file in &data {
        let pb = PathBuf::from(file);
        assert!(
            pb.exists(),
            "Could not find {}, which is required for this test",
            pb.display()
        );
    }

    let srclist =
        "test_files/1090008640/srclist_pumav3_EoR0aegean_EoR1pietro+ForA_1090008640_100.yaml"
            .to_string();
    assert!(
        PathBuf::from(&srclist).exists(),
        "Could not find {}, which is required for this test",
        srclist
    );

    CalibrateUserArgs {
        data: Some(data),
        source_list: Some(srclist),
        no_beam: !use_fee_beam,
        ..Default::default()
    }
}
