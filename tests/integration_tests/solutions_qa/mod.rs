// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Integration tests for solutions-qa.

use std::f64::consts::TAU;
use std::path::Path;

use marlu::{c64, Jones};
use ndarray::prelude::*;
use tempfile::TempDir;
use vec1::Vec1;

use crate::{get_cmd_output, get_reduced_1090008640, hyperdrive, Files};
use mwa_hyperdrive::CalibrationSolutions;

const NUM_CHANBLOCKS: usize = 384;
const FREQ_RES_HZ: f64 = 80e3;
const FIRST_FREQ_HZ: f64 = 167.04e6;

/// Write solutions for one observation of the tiles in `metafits`. Every
/// gain has the same smooth bandpass; RG6_90 tiles also have a 2% ripple from
/// a 706 ns cable reflection, every gain has 0.1% noise, and the tile at `noisy` has 15% noise on its
/// amplitudes. The two chanblocks at each coarse-channel edge have no
/// solution.
fn write_solutions(dir: &Path, obsid: u32, metafits: &str, noisy: usize) -> String {
    let context = mwalib::MetafitsContext::new(metafits, None).unwrap();
    let num_tiles = context.antennas.len();
    let freqs: Vec<f64> = (0..NUM_CHANBLOCKS)
        .map(|i| FIRST_FREQ_HZ + i as f64 * FREQ_RES_HZ)
        .collect();
    let edge = |chan: usize| matches!(chan % 16, 0 | 15);
    let mut di_jones = Array3::from_elem((1, num_tiles, NUM_CHANBLOCKS), Jones::default());
    for (i_tile, antenna) in context.antennas.iter().enumerate() {
        let reflection = antenna.rfinput_x.flavour.trim() == "RG6_90";
        for (i_chan, &f) in freqs.iter().enumerate() {
            let g = if edge(i_chan) {
                c64::new(f64::NAN, f64::NAN)
            } else {
                let x = i_chan as f64 / NUM_CHANBLOCKS as f64;
                let mut g = c64::from_polar(
                    (1.0 + 0.3 * x) * (1.0 + 0.01 * i_tile as f64),
                    0.2 * i_tile as f64 + 0.5 * x,
                );
                if reflection {
                    g *= 1.0 + 0.02 * c64::from_polar(1.0, -TAU * f * 706e-9);
                }
                // A repeatable pseudo-random 0.1% noise floor.
                let r = ((i_chan * 31 + i_tile * 977 + obsid as usize) % 97) as f64 / 48.0 - 1.0;
                g *= 1.0 + 1e-3 * c64::from_polar(1.0, TAU * r);
                if i_tile == noisy {
                    // A repeatable pseudo-random +/- 15%.
                    let r = ((i_chan * 7919 + obsid as usize) % 101) as f64 / 50.0 - 1.0;
                    g *= 1.0 + 0.15 * r;
                }
                g
            };
            let zero = c64::new(0.0, 0.0);
            di_jones[(0, i_tile, i_chan)] = Jones::from([g, zero, zero, g]);
        }
    }
    let results = Array2::from_shape_fn((1, NUM_CHANBLOCKS), |(_, c)| {
        if edge(c) {
            f64::NAN
        } else {
            1e-9 * (1.0 + (c % 5) as f64)
        }
    });
    let sols = CalibrationSolutions {
        di_jones,
        flagged_chanblocks: (0..NUM_CHANBLOCKS as u16)
            .filter(|&c| edge(c as usize))
            .collect(),
        chanblock_freqs: Some(Vec1::try_from_vec(freqs).unwrap()),
        obsid: Some(obsid),
        tile_names: Some(
            Vec1::try_from_vec(
                context
                    .antennas
                    .iter()
                    .map(|a| a.tile_name.clone())
                    .collect(),
            )
            .unwrap(),
        ),
        max_iterations: Some(50),
        stop_threshold: Some(1e-8),
        min_threshold: Some(1e-4),
        calibration_results: Some(results),
        ..Default::default()
    };
    let file = dir.join(format!("hyp_soln_{obsid}.fits"));
    sols.write_solutions_from_ext(&file).unwrap();
    file.display().to_string()
}

#[test]
fn test_solutions_qa_writes_every_figure() {
    let tmp_dir = TempDir::new().expect("couldn't make tmp dir");
    let Files { data, .. } = get_reduced_1090008640(false);
    let metafits = data[0].clone();
    let noisy = 5;
    let obsids = [1090008640, 1090008760, 1090008880];
    let files: Vec<String> = obsids
        .iter()
        .map(|&o| write_solutions(tmp_dir.path(), o, &metafits, noisy))
        .collect();

    let out = tmp_dir.path().join("plots");
    let mut args = vec!["solutions-qa".to_string()];
    args.extend(files);
    args.extend([
        "--metafits".to_string(),
        metafits.clone(),
        "--output-directory".to_string(),
        out.display().to_string(),
    ]);
    let cmd = hyperdrive().args(&args).ok();
    assert!(cmd.is_ok(), "solutions-qa failed: {}", cmd.err().unwrap());
    let (stdout, _) = get_cmd_output(cmd);

    let mut names = vec!["amp_rms".to_string(), "convergence".to_string()];
    for o in obsids {
        names.push(format!("amp_{o}"));
        names.push(format!("delay_{o}"));
    }
    for name in names {
        let file = out.join(format!("solutions_qa_{name}.png"));
        assert!(file.exists(), "{} wasn't written", file.display());
    }

    // The noisy tile is an amplitude outlier in every observation.
    let context = mwalib::MetafitsContext::new(&metafits, None).unwrap();
    let noisy_name = context.antennas[noisy].tile_name.trim().to_string();
    let line = stdout
        .lines()
        .find(|l| l.contains("XX tile") && l.contains("amplitude outlier"))
        .expect("an XX amplitude outlier is reported");
    assert!(
        line.contains(&noisy_name) && line.contains("3/3 obs"),
        "unexpected report: {line}"
    );
}

#[test]
fn test_solutions_qa_rejects_unmatched_tiles() {
    let tmp_dir = TempDir::new().expect("couldn't make tmp dir");
    let Files { data, .. } = get_reduced_1090008640(false);
    // 3 tiles without names can't be matched to the metafits' 128.
    let sols = CalibrationSolutions {
        di_jones: Array3::from_elem((1, 3, 8), Jones::identity()),
        ..Default::default()
    };
    let file = tmp_dir.path().join("sols.fits");
    sols.write_solutions_from_ext(&file).unwrap();

    #[rustfmt::skip]
    let cmd = hyperdrive()
        .args([
            "solutions-qa",
            &file.display().to_string(),
            "--metafits", &data[0],
            "--output-directory", &tmp_dir.path().display().to_string(),
        ])
        .ok();
    assert!(cmd.is_err(), "solutions-qa should fail on unmatched tiles");
    let (_, stderr) = get_cmd_output(cmd);
    assert!(
        !stderr.contains("panicked"),
        "solutions-qa panicked: {stderr}"
    );
    assert!(
        stderr.contains("can't be matched"),
        "unexpected stderr: {stderr}"
    );
}
