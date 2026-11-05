// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Integration tests for crosstalk-plot.

use std::fmt::Write;
use std::path::Path;

use tempfile::TempDir;

use crate::{get_cmd_output, get_reduced_1090008640, hyperdrive, Files};

const HEADER: &str = "pol,ant1,ant2,length_m,horizon_ns,tau_peak_ns,t,z,same_rx,rx1,rx2,flag";

/// Write a crosstalk-qa-style CSV for one observation, scoring every pair of
/// the first `num_tiles` tiles in the metafits. Every baseline touching
/// `hub` is flagged in XX, and one in five rows has no peak.
fn write_scores(dir: &Path, obsid: u32, metafits: &str, num_tiles: usize, hub: usize) -> String {
    let context = mwalib::MetafitsContext::new(metafits, None).unwrap();
    let names: Vec<&str> = context
        .antennas
        .iter()
        .take(num_tiles)
        .map(|a| a.tile_name.as_str())
        .collect();
    let mut csv = format!("{HEADER}\r\n");
    let mut row = 0;
    for pol in ["XX", "YY"] {
        for i in 0..names.len() {
            for j in i + 1..names.len() {
                row += 1;
                let flag = pol == "XX" && (i == hub || j == hub);
                let z = if flag { 9.0 } else { (row % 7) as f64 - 3.0 };
                let (tau, t) = if row % 5 == 0 {
                    ("nan".to_string(), 0.0)
                } else {
                    (format!("{}", (row % 40) as f64 * 50.0 - 1000.0), 30.0)
                };
                writeln!(
                    csv,
                    "{pol},{},{},{},{},{tau},{t},{z},True,1,1,{}\r",
                    names[i],
                    names[j],
                    10.0 + (i + j) as f64,
                    40.0,
                    if flag { "True" } else { "False" }
                )
                .unwrap();
            }
        }
    }
    let file = dir.join(format!("{obsid}.csv"));
    std::fs::write(&file, csv).unwrap();
    file.display().to_string()
}

#[test]
fn test_crosstalk_plot_writes_every_figure() {
    let tmp_dir = TempDir::new().expect("couldn't make tmp dir");
    let Files { data, .. } = get_reduced_1090008640(false);
    let metafits = data[0].clone();
    let csvs: Vec<String> = (0..3)
        .map(|i| write_scores(tmp_dir.path(), 1090008640 + i, &metafits, 24, 3))
        .collect();

    let out = tmp_dir.path().join("plots");
    let mut args = vec!["crosstalk-plot".to_string()];
    args.extend(csvs);
    args.extend([
        "--metafits".to_string(),
        metafits,
        "--output-directory".to_string(),
        out.display().to_string(),
    ]);
    let cmd = hyperdrive().args(&args).ok();
    assert!(cmd.is_ok(), "crosstalk-plot failed: {}", cmd.err().unwrap());
    let (stdout, _) = get_cmd_output(cmd);
    for name in [
        "tile_obs",
        "tile_spread",
        "baseline_matrix",
        "z_vs_length",
        "peak_delay",
    ] {
        let file = out.join(format!("crosstalk_{name}.png"));
        assert!(file.exists(), "{} wasn't written", file.display());
    }
    // The hub tile is the top XX outlier, flagged in every observation.
    let hub_line = stdout
        .lines()
        .find(|l| l.contains("XX tile"))
        .expect("an XX outlier tile is reported");
    assert!(
        hub_line.contains("median 23"),
        "unexpected report: {hub_line}"
    );
}

#[test]
fn test_crosstalk_plot_rejects_a_csv_without_scores() {
    let tmp_dir = TempDir::new().expect("couldn't make tmp dir");
    let Files { data, .. } = get_reduced_1090008640(false);
    let csv = tmp_dir.path().join("1090008640.csv");
    std::fs::write(&csv, "pol,ant1,ant2\nXX,Tile011,Tile012\n").unwrap();

    #[rustfmt::skip]
    let cmd = hyperdrive()
        .args([
            "crosstalk-plot",
            &csv.display().to_string(),
            "--metafits", &data[0],
            "--output-directory", &tmp_dir.path().display().to_string(),
        ])
        .ok();
    assert!(
        cmd.is_err(),
        "crosstalk-plot should fail without score columns"
    );
    let (_, stderr) = get_cmd_output(cmd);
    assert!(
        !stderr.contains("panicked"),
        "crosstalk-plot panicked: {stderr}"
    );
    assert!(
        stderr.contains("no 'length_m' column"),
        "unexpected stderr: {stderr}"
    );
}
