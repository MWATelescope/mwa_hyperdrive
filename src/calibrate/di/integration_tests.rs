// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Integration tests for sky-model visibilities generated during calibration.

use std::collections::HashSet;

use approx::{assert_abs_diff_eq, assert_abs_diff_ne};
use clap::Parser;
use marlu::Jones;
use mwalib::{fitsio_sys, *};
use ndarray::prelude::*;
use serial_test::serial;
use tempfile::TempDir;

use crate::{
    calibrate::{args::CalibrateUserArgs, di_calibrate},
    jones_test::TestJones,
    math::TileBaselineMaps,
    solutions::CalibrationSolutions,
    tests::reduced_obsids::get_reduced_1090008640,
    vis_io::read::{MsReader, VisRead},
    vis_utils::simulate::VisSimulateArgs,
};
use mwa_hyperdrive_common::{clap, marlu, mwalib, ndarray};

#[test]
/// Generate a model with "vis-simulate" (in uvfits), then feed it to
/// "di-calibrate" and write out the model used for calibration (as uvfits). The
/// visibilities should be exactly the same.
fn test_1090008640_calibrate_model_uvfits() {
    let num_timesteps = 2;
    let num_chans = 10;

    let temp_dir = TempDir::new().expect("couldn't make tmp dir");
    let model = temp_dir.path().join("model.uvfits");
    let args = get_reduced_1090008640(false);
    let metafits = &args.data.as_ref().unwrap()[0];
    let srclist = args.source_list.unwrap();
    #[rustfmt::skip]
    let sim_args = VisSimulateArgs::parse_from(&[
        "vis-simulate",
        "--metafits", metafits,
        "--source-list", &srclist,
        "--output-model-files", &format!("{}", model.display()),
        "--num-timesteps", &format!("{}", num_timesteps),
        "--num-fine-channels", &format!("{}", num_chans),
        "--veto-threshold", "0.0", // Don't complicate things with vetoing
        "--no-progress-bars",
    ]);

    // Run vis-simulate and check that it succeeds
    let result = sim_args.run(false);
    assert!(result.is_ok(), "result={:?} not ok", result.err().unwrap());

    let sols = temp_dir.path().join("sols.fits");
    let cal_model = temp_dir.path().join("cal_model.uvfits");

    #[rustfmt::skip]
    let cal_args = CalibrateUserArgs::parse_from(&[
        "di-calibrate",
        "--data", &format!("{}", model.display()), metafits,
        "--source-list", &srclist,
        "--outputs", &format!("{}", sols.display()),
        "--model-filenames", &format!("{}", cal_model.display()),
        // If we don't give the array position, there's a bit flip in the results. grrr
        "--array-position", "116.67081523611111", "-26.703319405555554", "377.827",
        "--veto-threshold", "0.0", // Don't complicate things with vetoing
        "--no-progress-bars",
    ]);

    // Run di-cal and check that it succeeds
    let result = di_calibrate(Box::new(cal_args), None, false);
    assert!(result.is_ok(), "result={:?} not ok", result.err().unwrap());

    let mut uvfits_m = fits_open!(&model).unwrap();
    let hdu_m = fits_open_hdu!(&mut uvfits_m, 0).unwrap();
    let gcount_m: String = get_required_fits_key!(&mut uvfits_m, &hdu_m, "GCOUNT").unwrap();
    let pcount_m: String = get_required_fits_key!(&mut uvfits_m, &hdu_m, "PCOUNT").unwrap();
    let floats_per_pol_m: String = get_required_fits_key!(&mut uvfits_m, &hdu_m, "NAXIS2").unwrap();
    let num_pols_m: String = get_required_fits_key!(&mut uvfits_m, &hdu_m, "NAXIS3").unwrap();
    let num_fine_freq_chans_m: String =
        get_required_fits_key!(&mut uvfits_m, &hdu_m, "NAXIS4").unwrap();
    let jd_zero_m: String = get_required_fits_key!(&mut uvfits_m, &hdu_m, "PZERO5").unwrap();
    let ptype4_m: String = get_required_fits_key!(&mut uvfits_m, &hdu_m, "PTYPE4").unwrap();

    let mut uvfits_c = fits_open!(&cal_model).unwrap();
    let hdu_c = fits_open_hdu!(&mut uvfits_c, 0).unwrap();
    let gcount_c: String = get_required_fits_key!(&mut uvfits_c, &hdu_c, "GCOUNT").unwrap();
    let pcount_c: String = get_required_fits_key!(&mut uvfits_c, &hdu_c, "PCOUNT").unwrap();
    let floats_per_pol_c: String = get_required_fits_key!(&mut uvfits_c, &hdu_c, "NAXIS2").unwrap();
    let num_pols_c: String = get_required_fits_key!(&mut uvfits_c, &hdu_c, "NAXIS3").unwrap();
    let num_fine_freq_chans_c: String =
        get_required_fits_key!(&mut uvfits_c, &hdu_c, "NAXIS4").unwrap();
    let jd_zero_c: String = get_required_fits_key!(&mut uvfits_c, &hdu_c, "PZERO5").unwrap();
    let ptype4_c: String = get_required_fits_key!(&mut uvfits_c, &hdu_c, "PTYPE4").unwrap();

    assert_eq!(gcount_m, gcount_c);
    assert_eq!(pcount_m, pcount_c);
    assert_eq!(floats_per_pol_m, floats_per_pol_c);
    assert_eq!(num_pols_m, num_pols_c);
    assert_eq!(num_fine_freq_chans_m, num_fine_freq_chans_c);
    assert_eq!(jd_zero_m, jd_zero_c);
    assert_eq!(ptype4_m, ptype4_c);

    let hdu_m = fits_open_hdu!(&mut uvfits_m, 1).unwrap();
    let tile_names_m: Vec<String> = get_fits_col!(&mut uvfits_m, &hdu_m, "ANNAME").unwrap();
    let hdu_c = fits_open_hdu!(&mut uvfits_c, 1).unwrap();
    let tile_names_c: Vec<String> = get_fits_col!(&mut uvfits_c, &hdu_c, "ANNAME").unwrap();
    for (tile_m, tile_c) in tile_names_m.into_iter().zip(tile_names_c.into_iter()) {
        assert_eq!(tile_m, tile_c);
    }

    // Test visibility values.
    fits_open_hdu!(&mut uvfits_m, 0).unwrap();
    let mut group_params_m = Array1::zeros(5);
    let mut vis_m = Array1::zeros(10 * 4 * 3);
    fits_open_hdu!(&mut uvfits_c, 0).unwrap();
    let mut group_params_c = group_params_m.clone();
    let mut vis_c = vis_m.clone();

    let mut status = 0;
    for i_row in 0..gcount_m.parse::<i64>().unwrap() {
        unsafe {
            // ffggpe = fits_read_grppar_flt
            fitsio_sys::ffggpe(
                uvfits_m.as_raw(),           /* I - FITS file pointer                       */
                1 + i_row,                   /* I - group to read (1 = 1st group)           */
                1,                           /* I - first vector element to read (1 = 1st)  */
                group_params_m.len() as i64, /* I - number of values to read                */
                group_params_m.as_mut_ptr(), /* O - array of values that are returned       */
                &mut status,                 /* IO - error status                           */
            );
            assert_eq!(status, 0, "Status wasn't 0");
            assert_abs_diff_ne!(group_params_m, group_params_c);
            // ffggpe = fits_read_grppar_flt
            fitsio_sys::ffggpe(
                uvfits_c.as_raw(),           /* I - FITS file pointer                       */
                1 + i_row,                   /* I - group to read (1 = 1st group)           */
                1,                           /* I - first vector element to read (1 = 1st)  */
                group_params_c.len() as i64, /* I - number of values to read                */
                group_params_c.as_mut_ptr(), /* O - array of values that are returned       */
                &mut status,                 /* IO - error status                           */
            );
            assert_eq!(status, 0, "Status wasn't 0");
            assert_abs_diff_eq!(group_params_m, group_params_c);

            // ffgpve = fits_read_img_flt
            fitsio_sys::ffgpve(
                uvfits_m.as_raw(),  /* I - FITS file pointer                       */
                1 + i_row,          /* I - group to read (1 = 1st group)           */
                1,                  /* I - first vector element to read (1 = 1st)  */
                vis_m.len() as i64, /* I - number of values to read                */
                0.0,                /* I - value for undefined pixels              */
                vis_m.as_mut_ptr(), /* O - array of values that are returned       */
                &mut 0,             /* O - set to 1 if any values are null; else 0 */
                &mut status,        /* IO - error status                           */
            );
            assert_abs_diff_ne!(vis_m, vis_c);
            // ffgpve = fits_read_img_flt
            fitsio_sys::ffgpve(
                uvfits_c.as_raw(),  /* I - FITS file pointer                       */
                1 + i_row,          /* I - group to read (1 = 1st group)           */
                1,                  /* I - first vector element to read (1 = 1st)  */
                vis_c.len() as i64, /* I - number of values to read                */
                0.0,                /* I - value for undefined pixels              */
                vis_c.as_mut_ptr(), /* O - array of values that are returned       */
                &mut 0,             /* O - set to 1 if any values are null; else 0 */
                &mut status,        /* IO - error status                           */
            );
            assert_eq!(status, 0, "Status wasn't 0");
            assert_abs_diff_eq!(vis_m, vis_c);
        };
    }

    // Inspect the solutions; they should all be close to identity.
    let result =
        CalibrationSolutions::read_solutions_from_ext(&sols, Some(&args.data.as_ref().unwrap()[0]));
    assert!(result.is_ok());
    let sols = result.unwrap();
    assert_abs_diff_eq!(
        sols.di_jones.mapv(TestJones::from),
        Array3::from_elem(sols.di_jones.dim(), TestJones::from(Jones::identity())),
        epsilon = 1e-15
    );
}

#[test]
#[serial]
/// Generate a model with "vis-simulate" (in a measurement set), then feed it to
/// "di-calibrate" and write out the model used for calibration (into a
/// measurement set). The visibilities should be exactly the same.
fn test_1090008640_calibrate_model_ms() {
    let num_timesteps = 2;
    let num_chans = 10;

    let temp_dir = TempDir::new().expect("couldn't make tmp dir");
    let model = temp_dir.path().join("model.ms");
    let args = get_reduced_1090008640(false);
    let metafits = &args.data.as_ref().unwrap()[0];
    let srclist = args.source_list.unwrap();
    #[rustfmt::skip]
    let sim_args = VisSimulateArgs::parse_from(&[
        "vis-simulate",
        "--metafits", metafits,
        "--source-list", &srclist,
        "--output-model-files", &format!("{}", model.display()),
        "--num-timesteps", &format!("{}", num_timesteps),
        "--num-fine-channels", &format!("{}", num_chans),
        "--no-progress-bars"
    ]);

    // Run vis-simulate and check that it succeeds
    let result = sim_args.run(false);
    assert!(result.is_ok(), "result={:?} not ok", result.err().unwrap());

    let sols = temp_dir.path().join("sols.fits");
    let cal_model = temp_dir.path().join("cal_model.ms");
    #[rustfmt::skip]
    let cal_args = CalibrateUserArgs::parse_from(&[
        "di-calibrate",
        "--data", &format!("{}", model.display()), metafits,
        "--source-list", &srclist,
        "--outputs", &format!("{}", sols.display()),
        "--model-filenames", &format!("{}", cal_model.display()),
        "--no-progress-bars"
    ]);

    // Run di-cal and check that it succeeds
    let result = di_calibrate(Box::new(cal_args), None, false);
    assert!(result.is_ok(), "result={:?} not ok", result.err().unwrap());

    let ms_m = MsReader::new(&model, Some(metafits)).unwrap();
    let ctx_m = ms_m.get_obs_context();
    let ms_c = MsReader::new(&cal_model, Some(metafits)).unwrap();
    let ctx_c = ms_c.get_obs_context();
    assert_eq!(ctx_m.all_timesteps, ctx_c.all_timesteps);
    assert_eq!(ctx_m.all_timesteps.len(), num_timesteps);
    assert_eq!(ctx_m.timestamps, ctx_c.timestamps);
    assert_eq!(ctx_m.fine_chan_freqs, ctx_c.fine_chan_freqs);
    assert_eq!(ctx_m.flagged_tiles, ctx_c.flagged_tiles);
    assert_eq!(ctx_m.tile_xyzs, ctx_c.tile_xyzs);
    assert_eq!(ctx_m.flagged_fine_chans, ctx_c.flagged_fine_chans);

    let flagged_fine_chans_set: HashSet<usize> = ctx_m.flagged_fine_chans.iter().cloned().collect();
    let tile_to_baseline_map = TileBaselineMaps::new(ctx_m.tile_xyzs.len(), &ctx_m.flagged_tiles)
        .tile_to_unflagged_cross_baseline_map;
    let max_baseline_idx = tile_to_baseline_map.values().max().unwrap();
    let data_shape = (
        max_baseline_idx + 1,
        ctx_m.fine_chan_freqs.len() - ctx_m.flagged_fine_chans.len(),
    );
    let mut vis_m = Array2::<Jones<f32>>::zeros(data_shape);
    let mut vis_c = Array2::<Jones<f32>>::zeros(data_shape);
    let mut weight_m = Array2::<f32>::zeros(data_shape);
    let mut weight_c = Array2::<f32>::zeros(data_shape);

    for timestep in ctx_m.all_timesteps.clone() {
        ms_m.read_crosses(
            vis_m.view_mut(),
            weight_m.view_mut(),
            timestep,
            &tile_to_baseline_map,
            &flagged_fine_chans_set,
        )
        .unwrap();
        ms_c.read_crosses(
            vis_c.view_mut(),
            weight_c.view_mut(),
            timestep,
            &tile_to_baseline_map,
            &flagged_fine_chans_set,
        )
        .unwrap();

        assert_abs_diff_eq!(vis_m.mapv(TestJones::from), vis_c.mapv(TestJones::from));
        assert_abs_diff_eq!(weight_m, weight_c);
    }

    // Inspect the solutions; they should all be close to identity.
    let result =
        CalibrationSolutions::read_solutions_from_ext(&sols, Some(&args.data.as_ref().unwrap()[0]));
    assert!(result.is_ok());
    let sols = result.unwrap();
    assert_abs_diff_eq!(
        sols.di_jones.mapv(TestJones::from),
        Array3::from_elem(sols.di_jones.dim(), TestJones::from(Jones::identity())),
        epsilon = 1e-15
    );
}
