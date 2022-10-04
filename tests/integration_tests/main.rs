// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Integration tests.
//!
//! Some help for laying out these tests was taken from:
//! https://matklad.github.io/2021/02/27/delete-cargo-integration-tests.html

mod di_calibrate;
mod no_stderr;
mod solutions_apply;

use std::{
    fs::File,
    io::Write,
    path::{Path, PathBuf},
    process::Output,
    str::from_utf8,
};

use assert_cmd::{output::OutputError, Command};
use marlu::Jones;
use ndarray::prelude::*;

use mwa_hyperdrive::{CalibrationSolutions, DiCalArgs};

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

fn get_1090008640_identity_solutions_file(tmp_dir: &Path) -> PathBuf {
    let sols = CalibrationSolutions {
        di_jones: Array3::from_elem((1, 128, 32), Jones::identity()),
        ..Default::default()
    };
    let file = tmp_dir.join("sols.fits");
    sols.write_solutions_from_ext::<&Path>(&file).unwrap();
    file
}

fn make_file_in_dir<T: AsRef<Path>, U: AsRef<Path>>(filename: T, dir: U) -> (PathBuf, File) {
    let path = dir.as_ref().join(filename);
    let f = File::create(&path).expect("couldn't make file");
    (path, f)
}

fn serialise_cal_args_toml(args: &DiCalArgs, file: &mut File) {
    let ser = toml::to_string_pretty(&args).expect("couldn't serialise DiCalArgs as toml");
    write!(file, "{}", ser).unwrap();
}

fn serialise_cal_args_json(args: &DiCalArgs, file: &mut File) {
    let ser = serde_json::to_string_pretty(&args).expect("couldn't serialise DiCalArgs as json");
    write!(file, "{}", ser).unwrap();
}

/// Get the calibration arguments associated with the obsid 1090008640 (raw MWA
/// data). This observational data is inside the hyperdrive git repo, but has
/// been reduced; there is only 1 coarse channel and 1 timestep.
fn get_reduced_1090008640(use_fee_beam: bool, include_mwaf: bool) -> DiCalArgs {
    // Use absolute paths.
    let test_files = PathBuf::from("test_files/1090008640")
        .canonicalize()
        .unwrap();

    // Ensure that the required files are there.
    let mut data = vec![
        format!("{}/1090008640.metafits", test_files.display()),
        format!(
            "{}/1090008640_20140721201027_gpubox01_00.fits",
            test_files.display()
        ),
    ];
    if include_mwaf {
        data.push(format!("{}/1090008640_01.mwaf", test_files.display()));
    }
    for file in &data {
        let pb = PathBuf::from(file);
        assert!(
            pb.exists(),
            "Could not find {}, which is required for this test",
            pb.display()
        );
    }

    let srclist = format!(
        "{}/srclist_pumav3_EoR0aegean_EoR1pietro+ForA_1090008640_100.yaml",
        test_files.display()
    );
    assert!(
        PathBuf::from(&srclist).exists(),
        "Could not find {}, which is required for this test",
        srclist
    );

    DiCalArgs {
        data: Some(data),
        source_list: Some(srclist),
        no_beam: !use_fee_beam,
        ..Default::default()
    }
}
