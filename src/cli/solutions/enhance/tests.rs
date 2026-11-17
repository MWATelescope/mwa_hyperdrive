// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Tests for solutions-enhance.

use std::{collections::HashSet, path::Path};

use marlu::Jones;
use ndarray::prelude::*;
use serial_test::serial;
use tempfile::TempDir;
use vec1::Vec1;

use super::*;
use crate::{
    cli::solutions::apply::tests::get_1090008640_identity_solutions_file,
    solutions::enhance::polyfit_eval,
    tests::{get_reduced_1090008640_raw_pbs, DataAsPathBufs},
    CalibrationSolutions, HyperdriveError,
};

fn get_args(solutions: &Path, output: &Path, extra: &[&str]) -> SolutionsEnhanceArgs {
    let DataAsPathBufs { metafits, vis, .. } = get_reduced_1090008640_raw_pbs();
    let mut args = vec![
        "solutions-enhance".to_string(),
        "--data".to_string(),
        format!("{}", metafits.display()),
        format!("{}", vis[0].display()),
        "--solutions".to_string(),
        format!("{}", solutions.display()),
        "--outputs".to_string(),
        format!("{}", output.display()),
    ];
    args.extend(extra.iter().map(|s| s.to_string()));
    SolutionsEnhanceArgs::parse_from(args)
}

/// The RMS of what a cubic leaves of the values (which are at the given
/// channels).
fn rms_about_cubic(chans: &[usize], values: &[f64]) -> f64 {
    let x = chans
        .iter()
        .map(|&c| c as f64 / 16.0 - 1.0)
        .collect::<Vec<_>>();
    let cubic = polyfit_eval(&x, values, &vec![true; values.len()], 3).unwrap();
    (values
        .iter()
        .zip(cubic)
        .map(|(v, c)| (v - c).powi(2))
        .sum::<f64>()
        / values.len() as f64)
        .sqrt()
}

#[test]
fn test_solutions_enhance_takes_structure_from_autos() {
    let tmp_dir = TempDir::new().unwrap();
    let output = tmp_dir.path().join("enhanced.fits");

    // Make solutions that are flat in frequency, with one flagged tile.
    let flagged_tile = 3;
    let num_chans = 32;
    let mut sols = CalibrationSolutions {
        di_jones: Array3::from_elem((1, 128, num_chans), Jones::identity()),
        chanblock_freqs: Some(
            Vec1::try_from(Array1::linspace(196495000.0, 197735000.0, num_chans).to_vec()).unwrap(),
        ),
        ..Default::default()
    };
    sols.di_jones
        .slice_mut(s![0, flagged_tile, ..])
        .fill(Jones::nan());
    let sols_file = tmp_dir.path().join("sols.fits");
    sols.write_solutions_from_ext::<&Path>(&sols_file).unwrap();

    // Don't correct the bandpass of the coarse channel; the autos then have
    // plenty of frequency structure.
    let params = get_args(&sols_file, &output, &["--pfb-flavour", "none"])
        .parse()
        .unwrap();

    // Read the autos.
    let input_vis_params = &params.input_vis_params;
    let obs_context = input_vis_params.get_obs_context();
    assert!(obs_context.autocorrelations_present);
    assert_eq!(obs_context.fine_chan_freqs.len(), num_chans);
    let mut autos_fb = Array2::from_elem((num_chans, 128), Jones::<f32>::default());
    let mut auto_weights_fb = Array2::zeros((num_chans, 128));
    input_vis_params
        .vis_reader
        .read_autos(
            autos_fb.view_mut(),
            auto_weights_fb.view_mut(),
            *input_vis_params.timeblocks.first().timesteps.first(),
            &input_vis_params.tile_baseline_flags,
            &HashSet::new(),
        )
        .unwrap();

    let flagged_chans = input_vis_params.spw.flagged_chan_indices.clone();

    let enhanced = params.run().unwrap();
    assert_eq!(enhanced.di_jones.dim(), (1, 128, num_chans));

    // The solutions are written, and what's written is what was returned.
    let written = CalibrationSolutions::read_solutions_from_ext(&output, None::<&Path>).unwrap();
    assert_eq!(written.di_jones.dim(), enhanced.di_jones.dim());
    assert_eq!(written.chanblock_freqs, sols.chanblock_freqs);
    for (w, e) in written.di_jones.iter().zip(enhanced.di_jones.iter()) {
        if e.any_nan() {
            assert!(w.any_nan());
        } else {
            for i in 0..4 {
                assert!((w[i] - e[i]).norm() < 1e-12);
            }
        }
    }

    let mut num_tiles_checked = 0;
    for (i_tile, enhanced_f) in enhanced
        .di_jones
        .slice(s![0, .., ..])
        .outer_iter()
        .enumerate()
    {
        // The flagged tile is still flagged.
        if i_tile == flagged_tile {
            assert!(enhanced_f.iter().all(|j| j.any_nan()));
            continue;
        }
        // Everything else has a value in every channel, even where the autos
        // are flagged (these channels are bridged).
        assert!(enhanced_f.iter().all(|j| !j.any_nan()));

        for i_elem in [0, 3] {
            // Applying the enhanced solutions to the autos (where they aren't
            // flagged) leaves no fine frequency structure. With so little
            // bandwidth, what isn't fine is a cubic.
            let mut chans = vec![];
            let mut before = vec![];
            let mut after = vec![];
            for (i_chan, j) in enhanced_f.iter().enumerate() {
                if auto_weights_fb[(i_chan, i_tile)] > 0.0
                    && !flagged_chans.contains(&(i_chan as u16))
                {
                    let auto = f64::from(autos_fb[(i_chan, i_tile)][i_elem].re);
                    chans.push(i_chan);
                    before.push(auto.ln());
                    after.push((auto * j[i_elem].norm_sqr()).ln());
                }
            }
            assert!(chans.len() > 20);
            let before = rms_about_cubic(&chans, &before);
            let after = rms_about_cubic(&chans, &after);
            assert!(before > 0.01, "tile {i_tile}: {before}");
            assert!(after < before / 20.0, "tile {i_tile}: {before} {after}");
        }
        // The cross terms are untouched.
        for j in enhanced_f {
            assert_eq!(j[1].norm(), 0.0);
            assert_eq!(j[2].norm(), 0.0);
        }
        num_tiles_checked += 1;
    }
    assert_eq!(num_tiles_checked, 127);
}

#[test]
fn test_solutions_enhance_amplitude_only() {
    let tmp_dir = TempDir::new().unwrap();
    let output = tmp_dir.path().join("enhanced.fits");
    let sols_file = get_1090008640_identity_solutions_file(tmp_dir.path());

    let enhanced = get_args(
        &sols_file,
        &output,
        &["--pfb-flavour", "none", "--amplitude-only"],
    )
    .parse()
    .unwrap()
    .run()
    .unwrap();
    // The input solutions have no phase, so neither do the outputs.
    let mut max_amp_change: f64 = 0.0;
    for j in &enhanced.di_jones {
        for i_elem in [0, 3] {
            assert!(j[i_elem].arg().abs() < 1e-9);
            max_amp_change = max_amp_change.max((j[i_elem].norm() - 1.0).abs());
        }
    }
    // The amplitudes take the passband ripple of the autos (the flagged edge
    // channels, where the ripple is largest, are bridged).
    assert!(max_amp_change > 0.05, "{max_amp_change}");

    // With phases, the outputs aren't real.
    let enhanced = get_args(&sols_file, &output, &["--pfb-flavour", "none"])
        .parse()
        .unwrap()
        .run()
        .unwrap();
    assert!(enhanced.di_jones.iter().any(|j| j[0].arg().abs() > 1e-3));
}

#[test]
fn test_solutions_enhance_dry_run_and_default_output() {
    let tmp_dir = TempDir::new().unwrap();
    let output = tmp_dir.path().join("enhanced.fits");
    let sols_file = get_1090008640_identity_solutions_file(tmp_dir.path());

    get_args(&sols_file, &output, &[]).run(true).unwrap();
    assert!(!output.exists());

    get_args(&sols_file, &output, &[]).run(false).unwrap();
    assert!(output.exists());

    let mut args = get_args(&sols_file, &output, &[]);
    args.outputs = None;
    let params = args.parse().unwrap();
    assert_eq!(
        params.output_solution_files.as_slice(),
        [PathBuf::from(DEFAULT_OUTPUT_SOLUTIONS_FILENAME)]
    );
}

#[test]
fn test_solutions_enhance_bad_args() {
    let tmp_dir = TempDir::new().unwrap();
    let output = tmp_dir.path().join("enhanced.fits");
    let sols_file = get_1090008640_identity_solutions_file(tmp_dir.path());

    // No solutions.
    let mut args = get_args(&sols_file, &output, &[]);
    args.solutions = None;
    assert!(matches!(
        args.parse(),
        Err(HyperdriveError::SolutionsEnhance(_))
    ));

    // An output that isn't a solutions file.
    let result = get_args(&sols_file, &tmp_dir.path().join("enhanced.uvfits"), &[]).parse();
    assert!(matches!(result, Err(HyperdriveError::SolutionsEnhance(_))));

    // Bad numbers.
    for extra in ["--delay-cut=0", "--max-mismatch=-1"] {
        let result = get_args(&sols_file, &output, &[extra]).parse();
        assert!(matches!(result, Err(HyperdriveError::SolutionsEnhance(_))));
    }
}

#[test]
fn test_solutions_enhance_ignores_flagged_chans() {
    let tmp_dir = TempDir::new().unwrap();
    let output = tmp_dir.path().join("enhanced.fits");
    let sols_file = get_1090008640_identity_solutions_file(tmp_dir.path());

    // Channels flagged in the data (here 8 and 9) don't contribute their autos,
    // so the enhanced solutions there differ from those made without the
    // flags.
    let get = |extra: &[&str]| {
        let mut args = vec!["--pfb-flavour", "none"];
        args.extend_from_slice(extra);
        get_args(&sols_file, &output, &args)
            .parse()
            .unwrap()
            .run()
            .unwrap()
    };
    let unflagged = get(&[]);
    let flagged = get(&["--fine-chan-flags", "8", "9"]);
    let mut num_differ = 0;
    for (u, f) in unflagged
        .di_jones
        .slice(s![0, .., 8..10])
        .iter()
        .zip(flagged.di_jones.slice(s![0, .., 8..10]).iter())
    {
        if (u[0] - f[0]).norm() > 1e-6 {
            num_differ += 1;
        }
    }
    assert!(num_differ > 0, "flagged channels made no difference");
}

#[test]
fn test_solutions_enhance_irregular_freqs() {
    let tmp_dir = TempDir::new().unwrap();
    let output = tmp_dir.path().join("enhanced.fits");

    // A gap in the middle of the band.
    let mut freqs = Array1::linspace(196495000.0, 197735000.0, 32).to_vec();
    freqs[16..].iter_mut().for_each(|f| *f += 1.28e6);
    let sols = CalibrationSolutions {
        di_jones: Array3::from_elem((1, 128, 32), Jones::identity()),
        chanblock_freqs: Some(Vec1::try_from(freqs).unwrap()),
        ..Default::default()
    };
    let sols_file = tmp_dir.path().join("sols.fits");
    sols.write_solutions_from_ext::<&Path>(&sols_file).unwrap();

    let result = get_args(&sols_file, &output, &[]).run(false);
    assert!(
        matches!(result, Err(HyperdriveError::SolutionsEnhance(_))),
        "{result:?}"
    );
    assert!(!output.exists());
}

#[test]
#[serial]
fn test_solutions_enhance_needs_autos() {
    let tmp_dir = TempDir::new().unwrap();
    let output = tmp_dir.path().join("enhanced.fits");
    let sols_file = get_1090008640_identity_solutions_file(tmp_dir.path());
    let ms = Path::new("test_files/1090008640/1090008640_cotter_trunc_noautos.ms")
        .canonicalize()
        .unwrap();

    let mut args = get_args(&sols_file, &output, &[]);
    args.data_args.files = Some(vec![format!("{}", ms.display())]);
    args.solutions = Some(format!("{}", sols_file.display()));
    let result = args.run(false);
    assert!(
        matches!(result, Err(HyperdriveError::SolutionsEnhance(_))),
        "{result:?}"
    );
    assert!(!output.exists());
}
