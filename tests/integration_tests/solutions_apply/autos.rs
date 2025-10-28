// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Test auto-correlation passthrough/omission in solutions-apply.

use tempfile::TempDir;

use crate::{get_cmd_output, hyperdrive};
use fitsio::{hdu::FitsHdu, FitsFile};

fn read_gcount(path: &std::path::Path) -> usize {
    let mut f = FitsFile::open(path).expect("open fits");
    let hdu0: FitsHdu = f.hdu(0).expect("open primary hdu");
    let gcount: String = hdu0
        .read_key(&mut f, "GCOUNT")
        .expect("read GCOUNT from fits");
    gcount.parse::<usize>().expect("parse GCOUNT")
}

fn read_num_tiles(path: &std::path::Path) -> usize {
    let mut f = FitsFile::open(path).expect("open fits");
    let an_hdu: FitsHdu = f.hdu("AIPS AN").expect("open AIPS AN");
    let names: Vec<String> = an_hdu.read_col(&mut f, "ANNAME").expect("read ANNAME col");
    names.len()
}

/// Calibrate and apply on a uvfits that contains autos; verify output autos behavior.
///
/// - We first run `di-calibrate` on 1061316544 uvfits to produce solutions.
/// - Then run three `solutions-apply` variants:
///   1) default (should include autos)
///   2) `--output-no-autos` (should exclude autos)
///   3) `--no-autos` (input ignores autos; output should also not contain autos)
#[test]
fn test_apply_autos_and_flags_on_1061316544() {
    let tmp = TempDir::new().expect("couldn't make tmp dir");
    let sols_path = tmp.path().join("dical.fits");
    let out_default = tmp.path().join("apply_default.uvfits");
    let out_output_no_autos = tmp.path().join("apply_output_no_autos.uvfits");
    let out_no_autos = tmp.path().join("apply_no_autos.uvfits");

    // 1) Calibrate to get solutions
    let cal_cmd = hyperdrive()
        .arg("di-calibrate")
        .arg("--data")
        .arg("test_files/1061316544/1061316544.metafits")
        .arg("test_files/1061316544/1061316544.uvfits")
        .arg("--source-list")
        .arg("test_files/1061316544/srclist_1061316544.yaml")
        .arg("--outputs")
        .arg(format!("{}", sols_path.display()))
        .arg("--no-progress-bars")
        .ok();
    assert!(
        cal_cmd.is_ok(),
        "di-calibrate invocation failed: {:?}",
        get_cmd_output(cal_cmd)
    );
    assert!(sols_path.exists(), "solutions not written");

    // 2a) Apply (default): should include autos
    let apply_default = hyperdrive()
        .arg("solutions-apply")
        .arg("--data")
        .arg("test_files/1061316544/1061316544.metafits")
        .arg("test_files/1061316544/1061316544.uvfits")
        .arg("--solutions")
        .arg(format!("{}", sols_path.display()))
        .arg("--outputs")
        .arg(format!("{}", out_default.display()))
        .arg("--no-progress-bars")
        .ok();
    assert!(
        apply_default.is_ok(),
        "apply (default) failed: {:?}",
        get_cmd_output(apply_default)
    );
    assert!(out_default.exists(), "apply default output not written");

    // 2b) Apply with --output-no-autos: should exclude autos
    let apply_output_no_autos = hyperdrive()
        .arg("solutions-apply")
        .arg("--data")
        .arg("test_files/1061316544/1061316544.metafits")
        .arg("test_files/1061316544/1061316544.uvfits")
        .arg("--solutions")
        .arg(format!("{}", sols_path.display()))
        .arg("--outputs")
        .arg(format!("{}", out_output_no_autos.display()))
        .arg("--output-no-autos")
        .arg("--no-progress-bars")
        .ok();
    assert!(
        apply_output_no_autos.is_ok(),
        "apply (--output-no-autos) failed: {:?}",
        get_cmd_output(apply_output_no_autos)
    );
    assert!(
        out_output_no_autos.exists(),
        "apply --output-no-autos output not written"
    );

    // 2c) Apply with --no-autos (input ignores autos): should also not contain autos
    let apply_no_autos = hyperdrive()
        .arg("solutions-apply")
        .arg("--data")
        .arg("test_files/1061316544/1061316544.metafits")
        .arg("test_files/1061316544/1061316544.uvfits")
        .arg("--solutions")
        .arg(format!("{}", sols_path.display()))
        .arg("--outputs")
        .arg(format!("{}", out_no_autos.display()))
        .arg("--no-autos")
        .arg("--no-progress-bars")
        .ok();
    assert!(
        apply_no_autos.is_ok(),
        "apply (--no-autos) failed: {:?}",
        get_cmd_output(apply_no_autos)
    );
    assert!(out_no_autos.exists(), "apply --no-autos output not written");

    // Determine geometry from the output itself
    let ntiles = read_num_tiles(&out_default);
    let crosses_only = (ntiles * (ntiles - 1)) / 2;
    let crosses_plus_autos = crosses_only + ntiles;

    // Read GCOUNT for each output and compare with inferred timesteps
    let gcount_default = read_gcount(&out_default);
    assert_eq!(
        gcount_default % crosses_plus_autos,
        0,
        "GCOUNT divisible by baselines+autos"
    );
    let timesteps = gcount_default / crosses_plus_autos;

    let gcount_out_no_autos = read_gcount(&out_output_no_autos);
    assert_eq!(
        gcount_out_no_autos,
        timesteps * crosses_only,
        "--output-no-autos should exclude autos"
    );

    let gcount_no_autos = read_gcount(&out_no_autos);
    assert_eq!(
        gcount_no_autos,
        timesteps * crosses_only,
        "--no-autos input should yield crosses-only output"
    );
}
