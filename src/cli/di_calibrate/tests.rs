// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Tests against calibration parameters and converting arguments to parameters.

use std::{collections::HashSet, fs::File, io::Write, path::PathBuf};

use approx::{assert_abs_diff_eq, assert_abs_diff_ne};
use clap::Parser;
use marlu::{
    constants::{MWA_HEIGHT_M, MWA_LAT_DEG, MWA_LONG_DEG},
    Jones, LatLngHeight,
};
use ndarray::prelude::*;
use serial_test::serial;
use tempfile::{tempdir, TempDir};

use super::{DiCalArgs, DiCalCliArgs};
use crate::{
    cli::{
        common::{BeamArgs, InputVisArgs, SkyModelWithVetoArgs},
        vis_simulate::VisSimulateArgs,
    },
    io::read::{
        fits::{fits_get_col, fits_get_required_key, fits_open, fits_open_hdu},
        MsReader, VisRead,
    },
    math::TileBaselineFlags,
    params::CalVis,
    tests::{
        get_reduced_1090008640_ms, get_reduced_1090008640_raw, get_reduced_1090008640_uvfits,
        DataAsStrings,
    },
    CalibrationSolutions, HyperdriveError,
};

fn get_reduced_1090008640(use_fee_beam: bool, include_mwaf: bool) -> DiCalArgs {
    let DataAsStrings {
        metafits,
        mut vis,
        mut mwafs,
        srclist,
    } = get_reduced_1090008640_raw();
    let mut files = vec![metafits];
    files.append(&mut vis);
    if include_mwaf {
        files.append(&mut mwafs);
    }

    DiCalArgs {
        args_file: None,
        data_args: InputVisArgs {
            files: Some(files),
            ..Default::default()
        },
        srclist_args: SkyModelWithVetoArgs {
            source_list: Some(srclist),
            ..Default::default()
        },
        beam_args: BeamArgs {
            beam_type: Some(
                {
                    if use_fee_beam {
                        "fee"
                    } else {
                        "none"
                    }
                }
                .to_string(),
            ),
            ..Default::default()
        },
        ..Default::default()
    }
}

#[test]
fn test_new_params_defaults() {
    let args = get_reduced_1090008640(false, true);
    let params = args.parse().unwrap();
    let input_vis_params = &params.input_vis_params;
    let obs_context = input_vis_params.get_obs_context();
    let total_num_tiles = obs_context.get_total_num_tiles();
    let num_unflagged_tiles =
        total_num_tiles - input_vis_params.tile_baseline_flags.flagged_tiles.len();
    // The default time resolution should be 2.0s, as per the metafits.
    assert_abs_diff_eq!(obs_context.time_res.unwrap().to_seconds(), 2.0);
    // The default freq resolution should be 40kHz, as per the metafits.
    assert_abs_diff_eq!(obs_context.freq_res.unwrap(), 40e3);
    // No tiles are flagged in the input data, and no additional flags were
    // supplied.
    assert_eq!(total_num_tiles, num_unflagged_tiles);
    assert_eq!(input_vis_params.tile_baseline_flags.flagged_tiles.len(), 0);

    // By default there are 5 flagged channels per coarse channel. We only have
    // one coarse channel here so we expect 27/32 channels.
    assert_eq!(input_vis_params.spw.chanblocks.len(), 27);
}

#[test]
fn test_new_params_no_input_flags() {
    let mut args = get_reduced_1090008640(false, true);
    args.data_args.ignore_input_data_tile_flags = true;
    args.data_args.ignore_input_data_fine_channel_flags = true;
    let params = args.parse().unwrap();
    let input_vis_params = &params.input_vis_params;
    let obs_context = input_vis_params.get_obs_context();
    let total_num_tiles = obs_context.get_total_num_tiles();
    let num_unflagged_tiles =
        total_num_tiles - input_vis_params.tile_baseline_flags.flagged_tiles.len();
    assert_abs_diff_eq!(obs_context.time_res.unwrap().to_seconds(), 2.0);
    assert_abs_diff_eq!(obs_context.freq_res.unwrap(), 40e3);
    assert_eq!(total_num_tiles, num_unflagged_tiles);
    assert_eq!(input_vis_params.tile_baseline_flags.flagged_tiles.len(), 0);

    assert_eq!(input_vis_params.spw.chanblocks.len(), 32);
}

#[test]
fn test_new_params_time_averaging() {
    // The native time resolution is 2.0s.
    let mut args = get_reduced_1090008640(false, true);
    // 1 is a valid time average factor.
    args.calibration_args.timesteps_per_timeblock = Some("1".to_string());
    let result = args.parse();
    assert!(result.is_ok());

    let mut args = get_reduced_1090008640(false, true);
    // 2 is a valid time average factor.
    args.calibration_args.timesteps_per_timeblock = Some("2".to_string());
    let result = args.parse();
    assert!(result.is_ok());

    let mut args = get_reduced_1090008640(false, true);
    // 4.0s should be a multiple of 2.0s
    args.calibration_args.timesteps_per_timeblock = Some("4.0s".to_string());
    let result = args.parse();
    assert!(result.is_ok());

    let mut args = get_reduced_1090008640(false, true);
    // 8.0s should be a multiple of 2.0s
    args.calibration_args.timesteps_per_timeblock = Some("8.0s".to_string());
    let result = args.parse();
    assert!(result.is_ok());
}

#[test]
fn test_new_params_time_averaging_fail() {
    // The native time resolution is 2.0s.
    let mut args = get_reduced_1090008640(false, true);
    // 1.5 is an invalid time average factor.
    args.calibration_args.timesteps_per_timeblock = Some("1.5".to_string());
    let result = args.parse();
    assert!(result.is_err());
    assert!(result
        .err()
        .unwrap()
        .to_string()
        .contains("Calibration time average factor isn't an integer"));

    let mut args = get_reduced_1090008640(false, true);
    // 2.01s is not a multiple of 2.0s
    args.calibration_args.timesteps_per_timeblock = Some("2.01s".to_string());
    let result = args.parse();
    assert!(result.is_err());
    assert!(result.err().unwrap().to_string().contains(
        "Calibration time resolution isn't a multiple of input data's: 2.01 seconds vs 2 seconds"
    ));

    let mut args = get_reduced_1090008640(false, true);
    // 3.0s is not a multiple of 2.0s
    args.calibration_args.timesteps_per_timeblock = Some("3.0s".to_string());
    let result = args.parse();
    assert!(result.is_err());
    assert!(result.err().unwrap().to_string().contains(
        "Calibration time resolution isn't a multiple of input data's: 3 seconds vs 2 seconds"
    ));
}

#[test]
fn test_new_params_freq_averaging() {
    // The native freq. resolution is 40kHz.
    let mut args = get_reduced_1090008640(false, true);
    // 3 is a valid freq average factor.
    args.data_args.freq_average = Some("3".to_string());
    let result = args.parse();
    assert!(result.is_ok());

    let mut args = get_reduced_1090008640(false, true);
    // 80kHz should be a multiple of 40kHz
    args.data_args.freq_average = Some("80kHz".to_string());
    let result = args.parse();
    assert!(result.is_ok());

    let mut args = get_reduced_1090008640(false, true);
    // 200kHz should be a multiple of 40kHz
    args.data_args.freq_average = Some("200kHz".to_string());
    let result = args.parse();
    assert!(result.is_ok());
}

#[test]
fn test_new_params_tile_flags() {
    // 1090008640 has no flagged tiles in its metafits.
    let mut args = get_reduced_1090008640(false, true);
    // Manually flag antennas 1, 2 and 3.
    args.data_args.tile_flags = Some(vec!["1".to_string(), "2".to_string(), "3".to_string()]);
    let params = args.parse().unwrap();
    let input_vis_params = &params.input_vis_params;
    let tile_baseline_flags = &input_vis_params.tile_baseline_flags;
    assert_eq!(tile_baseline_flags.flagged_tiles.len(), 3);
    assert!(tile_baseline_flags.flagged_tiles.contains(&1));
    assert!(tile_baseline_flags.flagged_tiles.contains(&2));
    assert!(tile_baseline_flags.flagged_tiles.contains(&3));
    assert_eq!(
        tile_baseline_flags
            .tile_to_unflagged_cross_baseline_map
            .len(),
        7750
    );

    assert_eq!(
        tile_baseline_flags.tile_to_unflagged_cross_baseline_map[&(0, 4)],
        0
    );
    assert_eq!(
        tile_baseline_flags.tile_to_unflagged_cross_baseline_map[&(0, 5)],
        1
    );
    assert_eq!(
        tile_baseline_flags.tile_to_unflagged_cross_baseline_map[&(0, 6)],
        2
    );
    assert_eq!(
        input_vis_params
            .tile_baseline_flags
            .tile_to_unflagged_cross_baseline_map[&(0, 7)],
        3
    );
}

#[test]
fn test_handle_invalid_output() {
    let mut args = get_reduced_1090008640(false, true);
    args.calibration_args.solutions = Some(vec!["invalid.out".into()]);
    let result = args.parse();

    assert!(result.is_err());
    assert!(result
        .err()
        .unwrap()
        .to_string()
        .contains("Cannot write calibration solutions to a file type 'out'"));
}

#[track_caller]
fn test_args_with_arg_file(args: &DiCalArgs) {
    let temp_dir = tempdir().expect("Couldn't make tempdir");
    for filename in ["calibrate.toml", "calibrate.json"] {
        let arg_file = temp_dir.path().join(filename);
        let mut f = File::create(&arg_file).expect("couldn't make file");
        let ser = match filename.split('.').last() {
            Some("toml") => {
                toml::to_string_pretty(&args).expect("couldn't serialise DiCalArgs as toml")
            }
            Some("json") => {
                serde_json::to_string_pretty(&args).expect("couldn't serialise DiCalArgs as json")
            }
            _ => unreachable!(),
        };
        eprintln!("{ser}");
        write!(&mut f, "{ser}").unwrap();
        // I don't know why, but the first argument ("di-calibrate" here) is
        // necessary, and the result is the same for any string!
        let parsed_args = DiCalArgs::parse_from(["di-calibrate", &arg_file.display().to_string()])
            .merge()
            .unwrap();
        assert!(parsed_args.data_args.files.is_some());
        let result = parsed_args.run(true).expect("args happily ingested");
        // No solutions returned because we did a dry run.
        assert!(result.is_none());
    }
}

#[test]
fn arg_file_absolute_paths() {
    let args = get_reduced_1090008640(false, true);
    test_args_with_arg_file(&args);
}

#[test]
fn arg_file_absolute_globs() {
    let mut args = get_reduced_1090008640(false, true);
    let first = PathBuf::from(&args.data_args.files.unwrap()[0]);
    let parent = first.parent().unwrap();
    args.data_args.files = Some(vec![
        format!("{}/*.metafits", parent.display()),
        format!("{}/*gpubox*", parent.display()),
        format!("{}/*.mwaf", parent.display()),
    ]);
    test_args_with_arg_file(&args);
}

#[test]
fn arg_file_relative_globs() {
    let mut args = get_reduced_1090008640(false, true);
    args.data_args.files = Some(vec![
        "test_files/1090008640/*.metafits".to_string(),
        "test_files/1090008640/*gpubox*".to_string(),
        "test_files/1090008640/*.mwaf".to_string(),
    ]);
    args.srclist_args.source_list = Some("test_files/1090008640/*srclist*_100.yaml".to_string());
    test_args_with_arg_file(&args);
}

#[test]
#[serial]
fn test_1090008640_di_calibrate_writes_solutions() {
    let tmp_dir = TempDir::new().expect("couldn't make tmp dir");

    let n_dirs = std::env::var("N_DIRS")
        .unwrap_or_else(|_| "1025".to_string()) // 192 passes, 193 fails.
        .parse::<usize>()
        .unwrap();

    let DataAsStrings {
        metafits,
        vis,
        srclist,
        ..
    } = get_reduced_1090008640_raw();
    let gpufits = &vis[0];
    let sols = tmp_dir.path().join("sols.fits");
    let cal_model = tmp_dir.path().join("hyp_model.uvfits");

    #[rustfmt::skip]
    let cal_args = DiCalArgs::parse_from([
        "di-calibrate",
        "--data", &metafits, gpufits,
        "--source-list", &srclist,
        "--outputs", &format!("{}", sols.display()),
        "--model-filenames", &format!("{}", cal_model.display()),
        "--num-sources", &format!("{n_dirs}"),
    ]);

    // Run di-cal and check that it succeeds
    let result = cal_args.run(false);
    assert!(result.is_ok(), "result={:?} not ok", result.err().unwrap());

    // check solutions file has been created, is readable
    assert!(sols.exists(), "sols file not written");
    let sol_data = CalibrationSolutions::read_solutions_from_ext(sols, metafits.into()).unwrap();
    assert_eq!(sol_data.obsid, Some(1090008640));
}

#[test]
fn test_1090008640_di_calibrate_uses_array_position() {
    let tmp_dir = TempDir::new().expect("couldn't make tmp dir");
    let DataAsStrings {
        metafits,
        vis,
        srclist,
        ..
    } = get_reduced_1090008640_raw();
    let gpufits = &vis[0];
    let sols = tmp_dir.path().join("sols.fits");
    let cal_model = tmp_dir.path().join("hyp_model.uvfits");

    // with non-default array position
    let exp_lat_deg = MWA_LAT_DEG - 1.;
    let exp_long_deg = MWA_LONG_DEG - 1.;
    let exp_height_m = MWA_HEIGHT_M - 1.;

    #[rustfmt::skip]
    let cal_args = DiCalArgs::parse_from([
        "di-calibrate",
        "--data", &metafits, gpufits,
        "--source-list", &srclist,
        "--outputs", &format!("{}", sols.display()),
        "--model-filenames", &format!("{}", cal_model.display()),
        "--array-position",
            &format!("{exp_long_deg}"),
            &format!("{exp_lat_deg}"),
            &format!("{exp_height_m}"),
    ]);

    let pos = cal_args.data_args.array_position.unwrap();

    assert_abs_diff_eq!(pos[0], exp_long_deg);
    assert_abs_diff_eq!(pos[1], exp_lat_deg);
    assert_abs_diff_eq!(pos[2], exp_height_m);
}

#[test]
fn test_1090008640_di_calibrate_array_pos_requires_3_args() {
    let tmp_dir = TempDir::new().expect("couldn't make tmp dir");
    let DataAsStrings {
        metafits,
        vis,
        srclist,
        ..
    } = get_reduced_1090008640_raw();
    let gpufits = &vis[0];
    let sols = tmp_dir.path().join("sols.fits");
    let cal_model = tmp_dir.path().join("hyp_model.uvfits");

    // no height specified
    let exp_lat_deg = MWA_LAT_DEG - 1.;
    let exp_long_deg = MWA_LONG_DEG - 1.;

    #[rustfmt::skip]
    let result = DiCalArgs::try_parse_from([
        "di-calibrate",
        "--data", &metafits, gpufits,
        "--source-list", &srclist,
        "--outputs", &format!("{}", sols.display()),
        "--model-filenames", &format!("{}", cal_model.display()),
        "--array-position",
            &format!("{exp_long_deg}"),
            &format!("{exp_lat_deg}"),
    ]);

    assert!(result.is_err());
    assert!(matches!(
        result.err().unwrap().kind(),
        clap::ErrorKind::WrongNumberOfValues
    ));
}

#[test]
/// Generate a model with "vis-simulate" (in uvfits), then feed it to
/// "di-calibrate" and write out the model used for calibration (as uvfits). The
/// visibilities should be exactly the same.
fn test_1090008640_calibrate_model_uvfits() {
    let num_timesteps = 2;
    let num_chans = 10;

    let n_dirs = std::env::var("N_DIRS")
        .unwrap_or_else(|_| "1025".to_string()) // 192 passes, 193 fails.
        .parse::<usize>()
        .unwrap();

    let temp_dir = TempDir::new().expect("couldn't make tmp dir");
    let model = temp_dir.path().join("model.uvfits");
    let DataAsStrings {
        metafits, srclist, ..
    } = get_reduced_1090008640_raw();
    #[rustfmt::skip]
    let sim_args = VisSimulateArgs::parse_from([
        "vis-simulate",
        "--metafits", &metafits,
        "--source-list", &srclist,
        "--output-model-files", &format!("{}", model.display()),
        "--num-timesteps", &format!("{num_timesteps}"),
        "--num-fine-channels", &format!("{num_chans}"),
        "--num-sources", &format!("{n_dirs}"),
        "--veto-threshold", "0.0", // Don't complicate things with vetoing
        // The array position is needed because, if not specified, it's read
        // slightly different out of the uvfits.
        "--array-position", "116.67081523611111", "-26.703319405555554", "377.827",
    ]);

    // Run vis-simulate and check that it succeeds
    let result = sim_args.run(false);
    assert!(result.is_ok(), "result={:?} not ok", result.err().unwrap());

    let sols = temp_dir.path().join("sols.fits");
    let cal_model = temp_dir.path().join("cal_model.uvfits");

    #[rustfmt::skip]
    let cal_args = DiCalArgs::parse_from([
        "di-calibrate",
        "--data", &format!("{}", model.display()), &metafits,
        "--source-list", &srclist,
        "--outputs", &format!("{}", sols.display()),
        "--model-filenames", &format!("{}", cal_model.display()),
        "--num-sources", &format!("{n_dirs}"),
        "--veto-threshold", "0.0", // Don't complicate things with vetoing
        "--array-position", "116.67081523611111", "-26.703319405555554", "377.827",
    ]);

    // Run di-cal and check that it succeeds
    let result = cal_args.parse().unwrap().run();
    assert!(result.is_ok(), "result={:?} not ok", result.err().unwrap());
    let sols = result.unwrap();

    let mut uvfits_m = fits_open(&model).unwrap();
    let hdu_m = fits_open_hdu(&mut uvfits_m, 0).unwrap();
    let gcount_m: String = fits_get_required_key(&mut uvfits_m, &hdu_m, "GCOUNT").unwrap();
    let pcount_m: String = fits_get_required_key(&mut uvfits_m, &hdu_m, "PCOUNT").unwrap();
    let floats_per_pol_m: String = fits_get_required_key(&mut uvfits_m, &hdu_m, "NAXIS2").unwrap();
    let num_pols_m: String = fits_get_required_key(&mut uvfits_m, &hdu_m, "NAXIS3").unwrap();
    let num_fine_freq_chans_m: String =
        fits_get_required_key(&mut uvfits_m, &hdu_m, "NAXIS4").unwrap();
    let jd_zero_m: String = fits_get_required_key(&mut uvfits_m, &hdu_m, "PZERO5").unwrap();
    let ptype4_m: String = fits_get_required_key(&mut uvfits_m, &hdu_m, "PTYPE4").unwrap();

    let mut uvfits_c = fits_open(&cal_model).unwrap();
    let hdu_c = fits_open_hdu(&mut uvfits_c, 0).unwrap();
    let gcount_c: String = fits_get_required_key(&mut uvfits_c, &hdu_c, "GCOUNT").unwrap();
    let pcount_c: String = fits_get_required_key(&mut uvfits_c, &hdu_c, "PCOUNT").unwrap();
    let floats_per_pol_c: String = fits_get_required_key(&mut uvfits_c, &hdu_c, "NAXIS2").unwrap();
    let num_pols_c: String = fits_get_required_key(&mut uvfits_c, &hdu_c, "NAXIS3").unwrap();
    let num_fine_freq_chans_c: String =
        fits_get_required_key(&mut uvfits_c, &hdu_c, "NAXIS4").unwrap();
    let jd_zero_c: String = fits_get_required_key(&mut uvfits_c, &hdu_c, "PZERO5").unwrap();
    let ptype4_c: String = fits_get_required_key(&mut uvfits_c, &hdu_c, "PTYPE4").unwrap();

    assert_eq!(gcount_m, gcount_c);
    assert_eq!(pcount_m, pcount_c);
    assert_eq!(floats_per_pol_m, floats_per_pol_c);
    assert_eq!(num_pols_m, num_pols_c);
    assert_eq!(num_fine_freq_chans_m, num_fine_freq_chans_c);
    assert_eq!(jd_zero_m, jd_zero_c);
    assert_eq!(ptype4_m, ptype4_c);

    let hdu_m = fits_open_hdu(&mut uvfits_m, 1).unwrap();
    let tile_names_m: Vec<String> = fits_get_col(&mut uvfits_m, &hdu_m, "ANNAME").unwrap();
    let hdu_c = fits_open_hdu(&mut uvfits_c, 1).unwrap();
    let tile_names_c: Vec<String> = fits_get_col(&mut uvfits_c, &hdu_c, "ANNAME").unwrap();
    for (tile_m, tile_c) in tile_names_m.into_iter().zip(tile_names_c.into_iter()) {
        assert_eq!(tile_m, tile_c);
    }

    // Test visibility values.
    fits_open_hdu(&mut uvfits_m, 0).unwrap();
    let mut group_params_m = Array1::zeros(5);
    let mut vis_m = Array1::zeros(10 * 4 * 3);
    fits_open_hdu(&mut uvfits_c, 0).unwrap();
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
    assert_abs_diff_eq!(
        sols.di_jones,
        Array3::from_elem(sols.di_jones.dim(), Jones::identity()),
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
    let DataAsStrings {
        metafits,
        vis: _,
        mwafs: _,
        srclist,
    } = get_reduced_1090008640_raw();

    // Non-default array position
    let lat_deg = MWA_LAT_DEG - 1.;
    let long_deg = MWA_LONG_DEG - 1.;
    let height_m = MWA_HEIGHT_M - 1.;

    #[rustfmt::skip]
    let sim_args = VisSimulateArgs::parse_from([
        "vis-simulate",
        "--metafits", &metafits,
        "--source-list", &srclist,
        "--output-model-files", &format!("{}", model.display()),
        "--num-timesteps", &format!("{num_timesteps}"),
        "--num-fine-channels", &format!("{num_chans}"),
        "--array-position",
            &format!("{long_deg}"),
            &format!("{lat_deg}"),
            &format!("{height_m}"),
    ]);

    // Run vis-simulate and check that it succeeds
    let result = sim_args.run(false);
    assert!(result.is_ok(), "result={:?} not ok", result.err().unwrap());

    let sols = temp_dir.path().join("sols.fits");
    let cal_model = temp_dir.path().join("cal_model.ms");
    #[rustfmt::skip]
    let cal_args = DiCalArgs::parse_from([
        "di-calibrate",
        "--data", &format!("{}", model.display()), &metafits,
        "--source-list", &srclist,
        "--outputs", &format!("{}", sols.display()),
        "--model-filenames", &format!("{}", cal_model.display()),
        "--array-position",
            &format!("{long_deg}"),
            &format!("{lat_deg}"),
            &format!("{height_m}"),
    ]);

    // Run di-cal and check that it succeeds
    let result = cal_args.parse().unwrap().run();
    assert!(result.is_ok(), "result={:?} not ok", result.err().unwrap());
    let sols = result.unwrap();

    let array_pos = LatLngHeight::mwa();
    let ms_m = MsReader::new(
        model,
        None,
        Some(&PathBuf::from(&metafits)),
        Some(array_pos),
    )
    .unwrap();
    let ctx_m = ms_m.get_obs_context();
    let ms_c = MsReader::new(
        cal_model,
        None,
        Some(&PathBuf::from(metafits)),
        Some(array_pos),
    )
    .unwrap();
    let ctx_c = ms_c.get_obs_context();
    assert_eq!(ctx_m.all_timesteps, ctx_c.all_timesteps);
    assert_eq!(ctx_m.all_timesteps.len(), num_timesteps);
    assert_eq!(ctx_m.timestamps, ctx_c.timestamps);
    assert_eq!(ctx_m.fine_chan_freqs, ctx_c.fine_chan_freqs);
    let m_flags = ctx_m.flagged_tiles.iter().copied().collect();
    let c_flags = &ctx_c.flagged_tiles;
    for m in &m_flags {
        assert!(c_flags.contains(m));
    }
    assert_eq!(ctx_m.tile_xyzs, ctx_c.tile_xyzs);
    assert_eq!(ctx_m.flagged_fine_chans, ctx_c.flagged_fine_chans);

    let flagged_fine_chans_set: HashSet<u16> = ctx_m.flagged_fine_chans.iter().copied().collect();
    let tile_baseline_flags = TileBaselineFlags::new(ctx_m.tile_xyzs.len(), m_flags);
    let max_baseline_idx = tile_baseline_flags
        .tile_to_unflagged_cross_baseline_map
        .values()
        .max()
        .unwrap();
    let data_shape = (
        ctx_m.fine_chan_freqs.len() - ctx_m.flagged_fine_chans.len(),
        max_baseline_idx + 1,
    );
    let mut vis_m = Array2::<Jones<f32>>::zeros(data_shape);
    let mut vis_c = Array2::<Jones<f32>>::zeros(data_shape);
    let mut weight_m = Array2::<f32>::zeros(data_shape);
    let mut weight_c = Array2::<f32>::zeros(data_shape);

    for &timestep in &ctx_m.all_timesteps {
        ms_m.read_crosses(
            vis_m.view_mut(),
            weight_m.view_mut(),
            timestep,
            &tile_baseline_flags,
            &flagged_fine_chans_set,
        )
        .unwrap();
        ms_c.read_crosses(
            vis_c.view_mut(),
            weight_c.view_mut(),
            timestep,
            &tile_baseline_flags,
            &flagged_fine_chans_set,
        )
        .unwrap();

        // Unlike the equivalent uvfits test, we have to use an epsilon here.
        // This is due to the MS antenna positions being in geocentric
        // coordinates and not geodetic like uvfits; in the process of
        // converting from geocentric to geodetic, small float errors are
        // introduced. If a metafits' positions are used instead, the results
        // are *exactly* the same, but we should trust the MS's positions, so
        // these errors must remain.
        #[cfg(feature = "gpu-single")]
        assert_abs_diff_eq!(vis_m, vis_c, epsilon = 2e-4);
        #[cfg(not(feature = "gpu-single"))]
        assert_abs_diff_eq!(vis_m, vis_c, epsilon = 4e-6);
        assert_abs_diff_eq!(weight_m, weight_c);
    }

    // Inspect the solutions; they should all be close to identity.
    #[cfg(feature = "gpu-single")]
    let epsilon = 6e-8;
    #[cfg(not(feature = "gpu-single"))]
    let epsilon = 2e-9;
    assert_abs_diff_eq!(
        sols.di_jones,
        Array3::from_elem(sols.di_jones.dim(), Jones::identity()),
        epsilon = epsilon
    );
}

#[test]
/// Generate a model with "vis-simulate" (in uvfits), then feed it to
/// "di-calibrate", testing the solution timeblocks that come out.
fn test_cal_timeblocks() {
    let num_timesteps = 3;
    let num_chans = 5;
    let n_dirs = std::env::var("N_DIRS")
        .unwrap_or_else(|_| "1025".to_string()) // 192 passes, 193 fails.
        .parse::<usize>()
        .unwrap();

    let temp_dir = TempDir::new().expect("couldn't make tmp dir");
    let model = temp_dir.path().join("model.uvfits");
    let DataAsStrings {
        metafits, srclist, ..
    } = get_reduced_1090008640_raw();
    #[rustfmt::skip]
    let sim_args = VisSimulateArgs::parse_from([
        "vis-simulate",
        "--metafits", &metafits,
        "--source-list", &srclist,
        "--output-model-files", &format!("{}", model.display()),
        "--num-timesteps", &format!("{num_timesteps}"),
        "--num-fine-channels", &format!("{num_chans}"),
        "--num-sources", &format!("{n_dirs}"),
        "--veto-threshold", "0.0", // Don't complicate things with vetoing
        // The array position is needed because, if not specified, it's read
        // slightly different out of the uvfits.
        "--array-position", "116.67081523611111", "-26.703319405555554", "377.827",
    ]);
    sim_args.run(false).unwrap();

    let sols_file = temp_dir.path().join("sols.fits");

    #[rustfmt::skip]
    let cal_args = DiCalArgs::parse_from([
        "di-calibrate",
        "--data", &format!("{}", model.display()), &metafits,
        "--source-list", &srclist,
        "--outputs", &format!("{}", sols_file.display()),
        "--num-sources", &format!("{n_dirs}"),
        "--veto-threshold", "0.0", // Don't complicate things with vetoing
        "--array-position", "116.67081523611111", "-26.703319405555554", "377.827",
    ]);
    let sols = cal_args.run(false).unwrap().unwrap();
    let num_cal_timeblocks = sols.di_jones.len_of(Axis(0));
    // We didn't specify anything with calibration timeblocks, so this should be
    // 1 (all input data timesteps are used at once in calibration).
    assert_eq!(num_cal_timeblocks, 1);
    #[cfg(not(feature = "gpu-single"))]
    let eps = 0.0; // I am amazed
    #[cfg(feature = "gpu-single")]
    let eps = 2e-8;
    assert_abs_diff_eq!(
        sols.di_jones,
        Array3::from_elem(sols.di_jones.dim(), Jones::identity()),
        epsilon = eps
    );

    #[rustfmt::skip]
    let cal_args = DiCalArgs::parse_from([
        "di-calibrate",
        "--data", &format!("{}", model.display()), &metafits,
        "--source-list", &srclist,
        "--outputs", &format!("{}", sols_file.display()),
        "--timesteps-per-timeblock", "2",
        "--num-sources", &format!("{n_dirs}"),
        "--veto-threshold", "0.0", // Don't complicate things with vetoing
        "--array-position", "116.67081523611111", "-26.703319405555554", "377.827",
    ]);
    let sols = cal_args.run(false).unwrap().unwrap();
    let num_cal_timeblocks = sols.di_jones.len_of(Axis(0));
    // 3 / 2 = 1.5 = 2 rounded up
    assert_eq!(num_cal_timeblocks, 2);
    #[cfg(not(feature = "gpu-single"))]
    let eps = 0.0;
    #[cfg(feature = "gpu-single")]
    let eps = 4e-8;
    assert_abs_diff_eq!(
        sols.di_jones,
        Array3::from_elem(sols.di_jones.dim(), Jones::identity()),
        epsilon = eps
    );
}

#[test]
fn test_flagging_all_uvw_lengths_causes_error() {
    let mut args = get_reduced_1090008640(false, false);
    args.calibration_args.uvw_min = Some("3000L".to_string());
    let error = args.parse().err().unwrap();
    assert!(matches!(error, HyperdriveError::DiCalibrate(_)));
    match &error {
        HyperdriveError::DiCalibrate(s) => {
            let s = s.as_str();
            assert!(s == "All baselines were flagged due to UVW cutoffs. Try adjusting the UVW min and/or max.");
        }
        _ => unreachable!(),
    }

    let mut args = get_reduced_1090008640(false, false);
    args.calibration_args.uvw_min = Some("0L".to_string());
    args.calibration_args.uvw_max = Some("1L".to_string());
    let error = args.parse().err().unwrap();
    assert!(matches!(error, HyperdriveError::DiCalibrate(_)));
    match &error {
        HyperdriveError::DiCalibrate(s) => {
            let s = s.as_str();
            assert!(s == "All baselines were flagged due to UVW cutoffs. Try adjusting the UVW min and/or max.");
        }
        _ => unreachable!(),
    }
}

/// Given calibration parameters and visibilities, this function tests that
/// everything matches an expected quality. The values may change over time but
/// they should be consistent with whatever tests use this test code.
fn test_1090008640_quality(
    params: crate::params::DiCalParams,
    vis_data: ArrayView3<Jones<f32>>,
    vis_model: ArrayView3<Jones<f32>>,
) {
    let (_, cal_results) = crate::di_calibrate::calibrate_timeblocks(
        vis_data,
        vis_model,
        &params.input_vis_params.timeblocks,
        &params.input_vis_params.spw.chanblocks,
        50,
        1e-8,
        1e-4,
        crate::context::Polarisations::default(),
        false,
    );

    // Only one timeblock.
    assert_eq!(cal_results.dim().0, 1);

    let mut count_50 = 0;
    let mut count_42 = 0;
    let mut chanblocks_42 = vec![];
    let mut fewest_iterations = u32::MAX;
    for cal_result in cal_results {
        match cal_result.num_iterations {
            50 => {
                count_50 += 1;
                fewest_iterations = fewest_iterations.min(cal_result.num_iterations);
            }
            42 => {
                count_42 += 1;
                chanblocks_42.push(cal_result.chanblock.unwrap());
                fewest_iterations = fewest_iterations.min(cal_result.num_iterations);
            }
            0 => panic!("0 iterations? Something is wrong."),
            _ => {
                if cal_result.num_iterations % 2 == 1 {
                    panic!("An odd number of iterations shouldn't be possible; at the time of writing, only even numbers are allowed.");
                }
                fewest_iterations = fewest_iterations.min(cal_result.num_iterations);
            }
        }

        assert!(
            cal_result.converged,
            "Chanblock {} did not converge",
            cal_result.chanblock.unwrap()
        );
        assert_eq!(cal_result.num_failed, 0);
        assert!(cal_result.max_precision < 1e8);
    }

    let expected_count_50 = 14;
    let expected_count_42 = 1;
    let expected_chanblocks_42 = vec![13];
    let expected_fewest_iterations = 40;
    if count_50 != expected_count_50
        || count_42 != expected_count_42
        || chanblocks_42 != expected_chanblocks_42
        || fewest_iterations != expected_fewest_iterations
    {
        panic!(
            r#"
Calibration quality has changed. This test expects:
  {expected_count_50} chanblocks with 50 iterations (got {count_50}),
  {expected_count_42} chanblocks with 42 iterations (got {count_42}),
  chanblocks {expected_chanblocks_42:?} to need 42 iterations (got {chanblocks_42:?}), and
  no chanblocks to finish in less than {expected_fewest_iterations} iterations (got {fewest_iterations}).
"#
        );
    }
}

#[test]
fn test_1090008640_calibration_quality_raw() {
    let temp_dir = tempdir().expect("Couldn't make temp dir");

    let DataAsStrings {
        metafits,
        mut vis,
        mwafs: _,
        srclist,
    } = get_reduced_1090008640_raw();
    let args = DiCalArgs {
        data_args: InputVisArgs {
            files: Some(vec![metafits, vis.swap_remove(0)]),
            // To be consistent with other data quality tests, add these flags.
            fine_chan_flags: Some(vec![0, 1, 2, 16, 30, 31]),
            pfb_flavour: Some("none".to_string()),
            ..Default::default()
        },
        srclist_args: SkyModelWithVetoArgs {
            source_list: Some(srclist),
            ..Default::default()
        },
        beam_args: BeamArgs {
            beam_type: Some("none".to_string()),
            ..Default::default()
        },
        calibration_args: DiCalCliArgs {
            solutions: Some(vec![temp_dir.path().join("hyp_sols.fits")]),
            ..Default::default()
        },
        ..Default::default()
    };

    let params = args.parse().unwrap();
    let CalVis {
        vis_data,
        vis_model,
        ..
    } = params
        .get_cal_vis()
        .expect("Couldn't read data and generate a model");
    test_1090008640_quality(params, vis_data.view(), vis_model.view());
}

#[test]
#[serial]
fn test_1090008640_calibration_quality_ms() {
    let temp_dir = tempdir().expect("Couldn't make temp dir");

    let DataAsStrings {
        metafits,
        mut vis,
        mwafs: _,
        srclist,
    } = get_reduced_1090008640_ms();
    let args = DiCalArgs {
        data_args: InputVisArgs {
            files: Some(vec![metafits, vis.swap_remove(0)]),
            // To be consistent with other data quality tests, add these flags.
            fine_chan_flags: Some(vec![0, 1, 2, 16, 30, 31]),
            ..Default::default()
        },
        srclist_args: SkyModelWithVetoArgs {
            source_list: Some(srclist),
            ..Default::default()
        },
        beam_args: BeamArgs {
            beam_type: Some("none".to_string()),
            ..Default::default()
        },
        calibration_args: DiCalCliArgs {
            solutions: Some(vec![temp_dir.path().join("hyp_sols.fits")]),
            ..Default::default()
        },
        ..Default::default()
    };

    let params = args.parse().unwrap();
    let CalVis {
        vis_data,
        vis_model,
        ..
    } = params
        .get_cal_vis()
        .expect("Couldn't read data and generate a model");
    test_1090008640_quality(params, vis_data.view(), vis_model.view());
}

#[test]
fn test_1090008640_calibration_quality_uvfits() {
    let temp_dir = tempdir().expect("Couldn't make temp dir");

    let DataAsStrings {
        metafits,
        mut vis,
        mwafs: _,
        srclist,
    } = get_reduced_1090008640_uvfits();
    let args = DiCalArgs {
        data_args: InputVisArgs {
            files: Some(vec![metafits, vis.swap_remove(0)]),
            // To be consistent with other data quality tests, add these flags.
            fine_chan_flags: Some(vec![0, 1, 2, 16, 30, 31]),
            ..Default::default()
        },
        srclist_args: SkyModelWithVetoArgs {
            source_list: Some(srclist),
            ..Default::default()
        },
        beam_args: BeamArgs {
            beam_type: Some("none".to_string()),
            ..Default::default()
        },
        calibration_args: DiCalCliArgs {
            solutions: Some(vec![temp_dir.path().join("hyp_sols.fits")]),
            ..Default::default()
        },
        ..Default::default()
    };

    let params = args.parse().unwrap();
    let CalVis {
        vis_data,
        vis_model,
        ..
    } = params
        .get_cal_vis()
        .expect("Couldn't read data and generate a model");
    test_1090008640_quality(params, vis_data.view(), vis_model.view());
}
