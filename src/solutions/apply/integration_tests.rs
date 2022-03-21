// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Integration tests for solutions-apply.

use std::path::{Path, PathBuf};

use approx::assert_abs_diff_eq;
use clap::Parser;
use marlu::Jones;
use mwalib::*;
use ndarray::prelude::*;
use serial_test::serial;
use tempfile::TempDir;

use crate::{
    solutions::{CalibrationSolutions, SolutionsApplyArgs},
    tests::reduced_obsids::get_reduced_1090008640,
    vis_io::read::{MsReader, UvfitsReader, VisRead},
};
use mwa_hyperdrive_common::{clap, marlu, mwalib};

fn get_1090008640_identity_solutions() -> CalibrationSolutions {
    CalibrationSolutions {
        di_jones: Array3::from_elem((1, 128, 32), Jones::identity()),
        obsid: Some(1090008640),
        flagged_tiles: vec![],
        flagged_chanblocks: vec![],
        ..Default::default()
    }
}

fn get_1090008640_identity_solutions_file(tmp_dir: &Path) -> PathBuf {
    let sols = get_1090008640_identity_solutions();
    let sols_file = tmp_dir.join("sols.fits");
    sols.write_solutions_from_ext::<&Path, &Path>(&sols_file, None)
        .unwrap();
    sols_file
}

#[test]
fn test_1090008640_solutions_apply_writes_vis_uvfits() {
    let tmp_dir = TempDir::new().expect("couldn't make tmp dir");
    let args = get_reduced_1090008640(false);
    let data = args.data.unwrap();
    let metafits = &data[0];
    let gpubox = &data[1];
    let solutions = get_1090008640_identity_solutions_file(tmp_dir.path());
    let out_vis_path = tmp_dir.path().join("vis.uvfits");

    #[rustfmt::skip]
    let args = SolutionsApplyArgs::parse_from(&[
        "solutions-apply",
        "--data", metafits, gpubox,
        "--solutions", &format!("{}", solutions.display()),
        "--outputs", &format!("{}", out_vis_path.display()),
        "--no-progress-bars",
    ]);

    // Run solutions-apply and check that it succeeds
    let result = args.run(false);
    assert!(result.is_ok(), "result={:?} not ok", result.err().unwrap());

    // check vis file has been created, is readable
    assert!(out_vis_path.exists(), "out vis file not written");
    let exp_timesteps = 1;
    let exp_baselines = 8256;
    let exp_channels = 32;

    let mut out_vis = fits_open!(&out_vis_path).unwrap();
    let hdu0 = fits_open_hdu!(&mut out_vis, 0).unwrap();
    let gcount: String = get_required_fits_key!(&mut out_vis, &hdu0, "GCOUNT").unwrap();
    assert_eq!(
        gcount.parse::<usize>().unwrap(),
        exp_timesteps * exp_baselines
    );
    let num_fine_freq_chans: String =
        get_required_fits_key!(&mut out_vis, &hdu0, "NAXIS4").unwrap();
    assert_eq!(num_fine_freq_chans.parse::<usize>().unwrap(), exp_channels);
}

#[test]
fn test_1090008640_solutions_apply_writes_vis_uvfits_no_autos() {
    let tmp_dir = TempDir::new().expect("couldn't make tmp dir");
    let args = get_reduced_1090008640(false);
    let data = args.data.unwrap();
    let metafits = &data[0];
    let gpubox = &data[1];
    let solutions = get_1090008640_identity_solutions_file(tmp_dir.path());
    let out_vis_path = tmp_dir.path().join("vis.uvfits");

    #[rustfmt::skip]
    let args = SolutionsApplyArgs::parse_from(&[
        "solutions-apply",
        "--data", metafits, gpubox,
        "--solutions", &format!("{}", solutions.display()),
        "--outputs", &format!("{}", out_vis_path.display()),
        "--no-autos",
        "--no-progress-bars",
    ]);

    // Run solutions-apply and check that it succeeds
    let result = args.run(false);
    assert!(result.is_ok(), "result={:?} not ok", result.err().unwrap());

    // check vis file has been created, is readable
    assert!(out_vis_path.exists(), "out vis file not written");
    let exp_timesteps = 1;
    let exp_baselines = 8128;
    let exp_channels = 32;

    let mut out_vis = fits_open!(&out_vis_path).unwrap();
    let hdu0 = fits_open_hdu!(&mut out_vis, 0).unwrap();
    let gcount: String = get_required_fits_key!(&mut out_vis, &hdu0, "GCOUNT").unwrap();
    assert_eq!(
        gcount.parse::<usize>().unwrap(),
        exp_timesteps * exp_baselines
    );
    let num_fine_freq_chans: String =
        get_required_fits_key!(&mut out_vis, &hdu0, "NAXIS4").unwrap();
    assert_eq!(num_fine_freq_chans.parse::<usize>().unwrap(), exp_channels);
}

#[test]
fn test_1090008640_solutions_apply_writes_vis_uvfits_avg_freq() {
    let tmp_dir = TempDir::new().expect("couldn't make tmp dir");
    let args = get_reduced_1090008640(false);
    let data = args.data.unwrap();
    let metafits = &data[0];
    let gpubox = &data[1];
    let solutions = get_1090008640_identity_solutions_file(tmp_dir.path());
    let out_vis_path = tmp_dir.path().join("vis.uvfits");

    let freq_avg_factor = 2;

    #[rustfmt::skip]
    let args = SolutionsApplyArgs::parse_from(&[
        "solutions-apply",
        "--data", metafits, gpubox,
        "--solutions", &format!("{}", solutions.display()),
        "--outputs", &format!("{}", out_vis_path.display()),
        "--freq-average", &format!("{}", freq_avg_factor),
        "--no-progress-bars",
    ]);

    // Run solutions-apply and check that it succeeds
    let result = args.run(false);
    assert!(result.is_ok(), "result={:?} not ok", result.err().unwrap());

    // check vis file has been created, is readable
    assert!(out_vis_path.exists(), "out vis file not written");
    let exp_timesteps = 1;
    let exp_baselines = 8256;
    let exp_channels = 32 / freq_avg_factor;

    let mut out_vis = fits_open!(&out_vis_path).unwrap();
    let hdu0 = fits_open_hdu!(&mut out_vis, 0).unwrap();
    let gcount: String = get_required_fits_key!(&mut out_vis, &hdu0, "GCOUNT").unwrap();
    assert_eq!(
        gcount.parse::<usize>().unwrap(),
        exp_timesteps * exp_baselines
    );
    let num_fine_freq_chans: String =
        get_required_fits_key!(&mut out_vis, &hdu0, "NAXIS4").unwrap();
    assert_eq!(num_fine_freq_chans.parse::<usize>().unwrap(), exp_channels);
}

#[test]
#[serial]
fn test_1090008640_solutions_apply_writes_vis_uvfits_and_ms() {
    let tmp_dir = TempDir::new().expect("couldn't make tmp dir");
    let args = get_reduced_1090008640(false);
    let data = args.data.unwrap();
    let metafits = &data[0];
    let gpubox = &data[1];
    let solutions = get_1090008640_identity_solutions_file(tmp_dir.path());
    let out_uvfits_path = tmp_dir.path().join("vis.uvfits");
    let out_ms_path = tmp_dir.path().join("vis.ms");

    #[rustfmt::skip]
    let args = SolutionsApplyArgs::parse_from(&[
        "solutions-apply",
        "--data", metafits, gpubox,
        "--solutions", &format!("{}", solutions.display()),
        "--outputs",
            &format!("{}", out_uvfits_path.display()),
            &format!("{}", out_ms_path.display()),
        "--no-progress-bars",
    ]);

    // Run solutions-apply and check that it succeeds
    let result = args.run(false);
    assert!(result.is_ok(), "result={:?} not ok", result.err().unwrap());

    // check vis file has been created, is readable
    assert!(out_uvfits_path.exists(), "out vis file not written");
    let exp_timesteps = 1;
    let exp_channels = 32;

    let uvfits_data = UvfitsReader::new(&out_uvfits_path, Some(metafits)).unwrap();

    let uvfits_ctx = uvfits_data.get_obs_context();

    // check ms file has been created, is readable
    assert!(out_ms_path.exists(), "out vis file not written");

    let ms_data = MsReader::new(&out_ms_path, Some(metafits)).unwrap();

    let ms_ctx = ms_data.get_obs_context();

    // XXX(dev): Can't write obsid to ms file without MwaObsContext "MS obsid not available (no MWA_GPS_TIME in OBSERVATION table)"
    // assert_eq!(uvfits_ctx.obsid, ms_ctx.obsid);
    assert_eq!(uvfits_ctx.obsid, Some(1090008640));
    assert_eq!(uvfits_ctx.timestamps, ms_ctx.timestamps);
    assert_eq!(uvfits_ctx.timestamps.len(), 1);
    assert_abs_diff_eq!(uvfits_ctx.timestamps[0].as_gpst_seconds(), 1090008658.);
    assert_eq!(uvfits_ctx.all_timesteps, ms_ctx.all_timesteps);
    assert_eq!(uvfits_ctx.all_timesteps.len(), exp_timesteps);
    assert_eq!(uvfits_ctx.fine_chan_freqs, ms_ctx.fine_chan_freqs);
    assert_eq!(uvfits_ctx.fine_chan_freqs.len(), exp_channels);
}
