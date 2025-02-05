// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Integration tests.
//!
//! Some help for laying out these tests was taken from:
//! https://matklad.github.io/2021/02/27/delete-cargo-integration-tests.html

mod di_calibrate;
mod no_stderr;
mod peel;
mod solutions_apply;

use std::{
    path::{Path, PathBuf},
    process::Output,
    str::from_utf8,
};

use assert_cmd::{output::OutputError, Command};
use marlu::Jones;
use ndarray::prelude::*;

use mwa_hyperdrive::CalibrationSolutions;

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

fn get_identity_solutions_file(tmp_dir: &Path) -> PathBuf {
    let sols = CalibrationSolutions {
        di_jones: Array3::from_elem((1, 128, 32), Jones::identity()),
        ..Default::default()
    };
    let file = tmp_dir.join("sols.fits");
    sols.write_solutions_from_ext::<&Path>(&file).unwrap();
    file
}

struct Files {
    data: Vec<String>,
    srclist: String,
}

/// Get the calibration arguments associated with the obsid 1090008640 (raw MWA
/// data). This observational data is inside the hyperdrive git repo, but has
/// been reduced; there is only 1 coarse channel and 1 timestep.
fn get_reduced_1090008640(include_mwaf: bool) -> Files {
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
        "Could not find {srclist}, which is required for this test"
    );

    Files { data, srclist }
}
