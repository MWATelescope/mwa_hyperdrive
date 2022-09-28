// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Tests to ensure there is no stderr output for successful commands.

use std::{collections::HashMap, io::Write};

use tempfile::TempDir;

use crate::{
    get_1090008640_identity_solutions_file, get_cmd_output, get_reduced_1090008640, hyperdrive,
};

#[test]
fn test_di_no_stderr() {
    let tmp_dir = TempDir::new().expect("couldn't make tmp dir");
    let sols = tmp_dir.path().join("sols.fits");
    let args = get_reduced_1090008640(true, false);
    let data = args.data.unwrap();

    #[rustfmt::skip]
    let cmd = hyperdrive()
        .args([
            "di-calibrate",
            "--data", &data[0], &data[1],
            "--source-list", &args.source_list.unwrap(),
            "--outputs", &format!("{}", sols.display()),
        ])
        .ok();
    assert!(
        cmd.is_ok(),
        "di-calibrate failed on simple test data: {}",
        cmd.err().unwrap()
    );
    let (_, stderr) = get_cmd_output(cmd);
    assert!(stderr.is_empty(), "stderr wasn't empty: {stderr}");
}

#[test]
fn test_solutions_apply_no_stderr() {
    let tmp_dir = TempDir::new().expect("couldn't make tmp dir");
    let output = tmp_dir.path().join("out.uvfits");
    let sols = get_1090008640_identity_solutions_file(tmp_dir.path());
    let args = get_reduced_1090008640(true, false);
    let data = args.data.unwrap();

    #[rustfmt::skip]
    let cmd = hyperdrive()
        .args([
            "solutions-apply",
            "--data", &data[0], &data[1],
            "--solutions", &format!("{}", sols.display()),
            "--outputs", &format!("{}", output.display()),
        ])
        .ok();
    assert!(
        cmd.is_ok(),
        "solutions-apply failed on simple test data: {}",
        cmd.err().unwrap()
    );
    let (_, stderr) = get_cmd_output(cmd);
    assert!(stderr.is_empty(), "stderr wasn't empty: {stderr}");
}

#[test]
fn test_vis_simulate_and_vis_subtract_no_stderr() {
    // First test vis-simulate.
    let num_timesteps = 2;
    let num_chans = 2;

    let temp_dir = TempDir::new().expect("couldn't make tmp dir");
    let model_path = temp_dir.path().join("model.uvfits");
    let args = get_reduced_1090008640(true, false);
    let metafits = args.data.as_ref().unwrap()[0].clone();

    #[rustfmt::skip]
    let cmd = hyperdrive()
        .args([
            "vis-simulate",
            "--metafits", &metafits,
            "--source-list", args.source_list.as_ref().unwrap(),
            "--output-model-files", &format!("{}", model_path.display()),
            "--num-timesteps", &format!("{}", num_timesteps),
            "--num-fine-channels", &format!("{}", num_chans),
            "--no-progress-bars"
        ])
        .ok();
    assert!(
        cmd.is_ok(),
        "vis-simulate failed on simple test data: {}",
        cmd.err().unwrap()
    );
    let (_, stderr) = get_cmd_output(cmd);
    assert!(stderr.is_empty(), "stderr wasn't empty: {stderr}");

    // Now vis-subtract. Use the vis-simulate result as the input.
    let sub_path = temp_dir.path().join("subtracted.uvfits");

    #[rustfmt::skip]
    let cmd = hyperdrive()
        .args([
            "vis-subtract",
            "--data", &metafits, &format!("{}", model_path.display()),
            "--source-list", &args.source_list.unwrap(),
            "--invert",
            "--output", &format!("{}", sub_path.display()),
            "--no-progress-bars"
        ])
        .ok();
    assert!(
        cmd.is_ok(),
        "vis-subtract failed on simple test data: {}",
        cmd.err().unwrap()
    );
    let (_, stderr) = get_cmd_output(cmd);
    assert!(stderr.is_empty(), "stderr wasn't empty: {stderr}");
}

#[test]
fn test_solutions_convert_no_stderr() {
    let tmp_dir = TempDir::new().expect("couldn't make tmp dir");
    let output = tmp_dir.path().join("sols.bin");
    let sols = get_1090008640_identity_solutions_file(tmp_dir.path());

    #[rustfmt::skip]
    let cmd = hyperdrive()
        .args([
            "solutions-convert",
            &format!("{}", sols.display()),
            &format!("{}", output.display()),
        ])
        .ok();
    assert!(
        cmd.is_ok(),
        "solutions-convert failed on simple test data: {}",
        cmd.err().unwrap()
    );
    let (_, stderr) = get_cmd_output(cmd);
    assert!(stderr.is_empty(), "stderr wasn't empty: {stderr}");
}

#[test]
#[cfg(feature = "plotting")]
fn test_solutions_plot_no_stderr() {
    let tmp_dir = TempDir::new().expect("couldn't make tmp dir");
    let sols = get_1090008640_identity_solutions_file(tmp_dir.path());
    let args = get_reduced_1090008640(true, false);
    let metafits = args.data.as_ref().unwrap()[0].clone();

    #[rustfmt::skip]
    let cmd = hyperdrive()
        .args([
            "solutions-plot",
            &format!("{}", sols.display()),
            "--metafits", &metafits,
        ])
        .ok();
    assert!(
        cmd.is_ok(),
        "solutions-plot failed on simple test data: {}",
        cmd.err().unwrap()
    );
    let (_, stderr) = get_cmd_output(cmd);
    assert!(stderr.is_empty(), "stderr wasn't empty: {stderr}");
}

#[test]
fn test_srclist_by_beam_no_stderr() {
    let tmp_dir = TempDir::new().expect("couldn't make tmp dir");
    let args = get_reduced_1090008640(true, false);
    let metafits = args.data.as_ref().unwrap()[0].clone();
    let srclist = args.source_list.unwrap();
    let output = tmp_dir.path().join("srclist.txt");

    #[rustfmt::skip]
    let cmd = hyperdrive()
        .args([
            "srclist-by-beam",
            &srclist,
            &format!("{}", output.display()),
            "--number", "2",
            "--output-type", "rts",
            "--metafits", &metafits,
        ])
        .ok();
    assert!(
        cmd.is_ok(),
        "srclist-by-beam failed on simple test data: {}",
        cmd.err().unwrap()
    );
    let (_, stderr) = get_cmd_output(cmd);
    assert!(stderr.is_empty(), "stderr wasn't empty: {stderr}");
}

#[test]
fn test_srclist_convert_no_stderr() {
    let tmp_dir = TempDir::new().expect("couldn't make tmp dir");
    let args = get_reduced_1090008640(true, false);
    let srclist = args.source_list.unwrap();
    let output = tmp_dir.path().join("srclist.txt");

    #[rustfmt::skip]
    let cmd = hyperdrive()
        .args([
            "srclist-convert",
            &srclist,
            &format!("{}", output.display()),
            "--output-type", "rts",
        ])
        .ok();
    assert!(
        cmd.is_ok(),
        "srclist-convert failed on simple test data: {}",
        cmd.err().unwrap()
    );
    let (_, stderr) = get_cmd_output(cmd);
    assert!(stderr.is_empty(), "stderr wasn't empty: {stderr}");
}

#[test]
fn test_srclist_shift_no_stderr() {
    let tmp_dir = TempDir::new().expect("couldn't make tmp dir");
    let args = get_reduced_1090008640(true, false);
    let srclist = args.source_list.unwrap();
    let output = tmp_dir.path().join("shifted.txt");
    let shifts = tmp_dir.path().join("shifts.json");

    // Populate the shifts with one source.
    let mut f = std::fs::File::create(&shifts).unwrap();

    let shift = HashMap::from([(
        "J000045-272248",
        HashMap::from([("ra", 0.1), ("dec", -0.1)]),
    )]);
    serde_json::to_writer(&mut f, &shift).unwrap();
    f.flush().unwrap();
    drop(f);

    #[rustfmt::skip]
    let cmd = hyperdrive()
        .args([
            "srclist-shift",
            &srclist,
            &format!("{}", shifts.display()),
            &format!("{}", output.display()),
            "--output-type", "rts",
        ])
        .ok();
    assert!(
        cmd.is_ok(),
        "srclist-shift failed on simple test data: {}",
        cmd.err().unwrap()
    );
    let (_, stderr) = get_cmd_output(cmd);
    assert!(stderr.is_empty(), "stderr wasn't empty: {stderr}");
}

#[test]
fn test_srclist_verify_no_stderr() {
    let args = get_reduced_1090008640(true, false);
    let srclist = args.source_list.unwrap();

    #[rustfmt::skip]
    let cmd = hyperdrive()
        .args([
            "srclist-verify",
            &srclist,
        ])
        .ok();
    assert!(
        cmd.is_ok(),
        "srclist-verify failed on simple test data: {}",
        cmd.err().unwrap()
    );
    let (_, stderr) = get_cmd_output(cmd);
    assert!(stderr.is_empty(), "stderr wasn't empty: {stderr}");
}

#[test]
fn test_dipole_gains_no_stderr() {
    let args = get_reduced_1090008640(true, false);
    let metafits = args.data.as_ref().unwrap()[0].clone();

    #[rustfmt::skip]
    let cmd = hyperdrive()
        .args([
            "dipole-gains",
            &metafits,
        ])
        .ok();
    assert!(
        cmd.is_ok(),
        "dipole-gains failed on simple test data: {}",
        cmd.err().unwrap()
    );
    let (_, stderr) = get_cmd_output(cmd);
    assert!(stderr.is_empty(), "stderr wasn't empty: {stderr}");
}
