// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Integration tests for calibration.

mod cli_args;

use std::{
    collections::{HashMap, HashSet},
    fs::File,
    io::{BufWriter, Write},
};

use approx::assert_abs_diff_eq;
use birli::{
    marlu::{LatLngHeight, RADec, UvfitsWriter, VisContext, VisWritable, XyzGeodetic},
    Jones,
};
use clap::Parser;
use mwa_hyperdrive_beam::Delays;
use mwa_hyperdrive_srclist::{
    hyperdrive::source_list_to_yaml, ComponentType, FluxDensity, FluxDensityType, Source,
    SourceComponent, SourceList,
};
use mwalib::*;
use serial_test::serial;

use crate::*;
use mwa_hyperdrive::{
    calibrate::{di_calibrate, solutions::CalibrationSolutions, CalibrateError},
    data_formats::{InputData, UvfitsReader, MS},
};
use mwa_hyperdrive_common::{
    clap,
    hifitime::{Duration, Epoch, Unit},
    itertools::izip,
    mwalib,
    num_traits::Zero,
    vec1::vec1,
};
use ndarray::prelude::*;

/// If di-calibrate is working, it should not write anything to stderr.
#[test]
fn test_no_stderr() {
    let tmp_dir = TempDir::new().expect("couldn't make tmp dir").into_path();
    let sols = tmp_dir.join("sols.fits");
    let args = get_reduced_1090008640(true, false);
    let data = args.data.unwrap();

    #[rustfmt::skip]
    let cmd = hyperdrive()
        .args(&[
            "di-calibrate",
            "--data", &data[0], &data[1],
            "--source-list", &args.source_list.unwrap(),
            "--outputs", &format!("{}", sols.display()),
        ])
        .ok();
    assert!(cmd.is_ok(), "di-calibrate failed on simple test data!");
    let (_, stderr) = get_cmd_output(cmd);
    assert!(stderr.is_empty(), "stderr wasn't empty: {stderr}");
}

#[test]
#[serial]
fn test_1090008640_di_calibrate_writes_solutions() {
    let tmp_dir = TempDir::new().expect("couldn't make tmp dir").into_path();
    let args = get_reduced_1090008640(true, false);
    let data = args.data.unwrap();
    let metafits = &data[0];
    let gpufits = &data[1];
    let sols = tmp_dir.join("sols.fits");
    let cal_model = tmp_dir.join("hyp_model.uvfits");

    #[rustfmt::skip]
    let cal_args = CalibrateUserArgs::parse_from(&[
        "di-calibrate",
        "--data", metafits, gpufits,
        "--source-list", &args.source_list.unwrap(),
        "--outputs", &format!("{}", sols.display()),
        "--model-filename", &format!("{}", cal_model.display()),
        "--no-progress-bars",
    ]);

    // Run di-cal and check that it succeeds
    let result = di_calibrate::<PathBuf>(Box::new(cal_args), None, false);
    assert!(result.is_ok(), "result={:?} not ok", result.err().unwrap());

    // check solutions file has been created, is readable
    assert!(sols.exists(), "sols file not written");
    let sol_data = CalibrationSolutions::read_solutions_from_ext(sols, metafits.into()).unwrap();
    assert_eq!(sol_data.obsid, Some(1090008640));
}

#[test]
fn test_1090008640_woden() {
    let tmp_dir = TempDir::new().expect("couldn't make tmp dir").into_path();
    let solutions_path = tmp_dir.join("sols.bin");

    // Reading from a uvfits file without a metafits file should fail because
    // there's no beam information.
    #[rustfmt::skip]
    let cal_args = CalibrateUserArgs::parse_from(&[
        "di-calibrate",
        "--data", "test_files/1090008640_WODEN/output_band01.uvfits",
        "--source-list", "test_files/1090008640_WODEN/srclist_3x3_grid.txt",
        "--outputs", &format!("{}", solutions_path.display()),
        "--no-progress-bars",
    ]);

    // Run di-cal and check that it fails
    let result = di_calibrate::<PathBuf>(Box::new(cal_args), None, false);
    assert!(
        matches!(result, Err(CalibrateError::InvalidArgs(_))),
        "result={:?} is not InvalidArgs",
        result.err().unwrap()
    );

    // This time give the metafits file.
    let cmd = hyperdrive()
        .args(&[
            "di-calibrate",
            "--data",
            "test_files/1090008640_WODEN/output_band01.uvfits",
            "test_files/1090008640_WODEN/1090008640.metafits",
            "--source-list",
            "test_files/1090008640_WODEN/srclist_3x3_grid.txt",
            #[cfg(feature = "cuda")]
            "--cpu",
            "--outputs",
            &format!("{}", solutions_path.display()),
        ])
        .ok();
    assert!(cmd.is_ok(), "{:?}", get_cmd_output(cmd));
    let (stdout, _) = get_cmd_output(cmd);

    // Verify that none of the calibration solutions are failures (i.e. not set
    // to NaN).
    let mut found_a_chanblock_line = false;
    for line in stdout.lines() {
        if line.starts_with("Chanblock") {
            found_a_chanblock_line = true;
            assert!(
                !line.contains("failed"),
                "Expected no lines with 'failed': {}",
                line
            );
        }
    }
    assert!(
        found_a_chanblock_line,
        "No 'Chanblock' lines found. Has the code changed?"
    );

    let metafits: Option<PathBuf> = None;
    let bin_sols =
        CalibrationSolutions::read_solutions_from_ext(&solutions_path, metafits.as_ref()).unwrap();
    assert_eq!(bin_sols.di_jones.dim(), (1, 128, 32));
    assert_eq!(bin_sols.start_timestamps.len(), 1);
    assert_eq!(bin_sols.end_timestamps.len(), 1);
    assert_eq!(bin_sols.average_timestamps.len(), 1);
    assert_abs_diff_eq!(
        bin_sols.start_timestamps[0].as_gpst_seconds(),
        // output_band01 lists the start time as 1090008640, but it should
        // probably be 1090008642.
        1090008640.0
    );
    assert_abs_diff_eq!(bin_sols.end_timestamps[0].as_gpst_seconds(), 1090008640.0);
    assert_abs_diff_eq!(
        bin_sols.average_timestamps[0].as_gpst_seconds(),
        1090008640.0,
    );
    assert!(!bin_sols.di_jones.iter().any(|jones| jones.any_nan()));

    // Re-do calibration, but this time into the hyperdrive fits format.
    let solutions_path = tmp_dir.join("sols.fits");

    let cal_args = CalibrateUserArgs::parse_from(&[
        "di-calibrate",
        "--data",
        "test_files/1090008640_WODEN/output_band01.uvfits",
        "test_files/1090008640_WODEN/1090008640.metafits",
        "--source-list",
        "test_files/1090008640_WODEN/srclist_3x3_grid.txt",
        #[cfg(feature = "cuda")]
        "--cpu",
        "--outputs",
        &format!("{}", solutions_path.display()),
        "--no-progress-bars",
    ]);

    // Run di-cal and check that it fails
    let result = di_calibrate::<PathBuf>(Box::new(cal_args), None, false);
    assert!(result.is_ok(), "result={:?} not ok", result.err().unwrap());

    let hyp_sols = result.unwrap().unwrap();
    assert_eq!(hyp_sols.di_jones.dim(), bin_sols.di_jones.dim());
    assert_eq!(hyp_sols.start_timestamps.len(), 1);
    assert_eq!(hyp_sols.end_timestamps.len(), 1);
    assert_eq!(hyp_sols.average_timestamps.len(), 1);
    assert_abs_diff_eq!(hyp_sols.start_timestamps[0].as_gpst_seconds(), 1090008640.0);
    assert_abs_diff_eq!(hyp_sols.end_timestamps[0].as_gpst_seconds(), 1090008640.0);
    assert_abs_diff_eq!(
        hyp_sols.average_timestamps[0].as_gpst_seconds(),
        1090008640.0
    );
    assert!(!hyp_sols.di_jones.iter().any(|jones| jones.any_nan()));

    let bin_sols_di_jones = bin_sols.di_jones.mapv(TestJones::from);
    let hyp_sols_di_jones = hyp_sols.di_jones.mapv(TestJones::from);
    assert_abs_diff_eq!(bin_sols_di_jones, hyp_sols_di_jones);
}

#[test]
#[serial]
fn test_1090008640_di_calibrate_writes_vis_uvfits() {
    let tmp_dir = TempDir::new().expect("couldn't make tmp dir").into_path();
    let args = get_reduced_1090008640(true, false);
    let data = args.data.unwrap();
    let metafits = &data[0];
    let gpufits = &data[1];
    let out_vis_path = tmp_dir.join("vis.uvfits");
    let cal_model = tmp_dir.join("hyp_model.uvfits");

    #[rustfmt::skip]
    let cal_args = CalibrateUserArgs::parse_from(&[
        "di-calibrate",
        "--data", metafits, gpufits,
        "--source-list", &args.source_list.unwrap(),
        "--outputs", &format!("{}", out_vis_path.display()),
        "--model-filename", &format!("{}", cal_model.display()),
        "--no-progress-bars",
    ]);

    // Run di-cal and check that it succeeds
    let result = di_calibrate::<PathBuf>(Box::new(cal_args), None, false);
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
#[serial]
fn test_1090008640_di_calibrate_writes_vis_uvfits_avg_freq() {
    let tmp_dir = TempDir::new().expect("couldn't make tmp dir").into_path();
    let args = get_reduced_1090008640(true, false);
    let data = args.data.unwrap();
    let metafits = &data[0];
    let gpufits = &data[1];
    let out_vis_path = tmp_dir.join("vis.uvfits");
    let cal_model = tmp_dir.join("hyp_model.uvfits");

    let freq_avg_factor = 2;

    #[rustfmt::skip]
    let cal_args = CalibrateUserArgs::parse_from(&[
        "di-calibrate",
        "--data", metafits, gpufits,
        "--source-list", &args.source_list.unwrap(),
        "--outputs", &format!("{}", out_vis_path.display()),
        "--model-filename", &format!("{}", cal_model.display()),
        "--output-vis-freq-average", &format!("{}", freq_avg_factor),
        "--no-progress-bars",
    ]);

    // Run di-cal and check that it succeeds
    let result = di_calibrate::<PathBuf>(Box::new(cal_args), None, false);
    assert!(result.is_ok(), "result={:?} not ok", result.err().unwrap());

    // check vis file has been created, is readable
    assert!(out_vis_path.exists(), "out vis file not written");
    let exp_timesteps = 1;
    let exp_baselines = 8128;
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
fn test_1090008640_di_calibrate_writes_vis_uvfits_ms() {
    let tmp_dir = TempDir::new().expect("couldn't make tmp dir").into_path();
    let args = get_reduced_1090008640(true, false);
    let data = args.data.unwrap();
    let metafits = &data[0];
    let gpufits = &data[1];
    let out_uvfits_path = tmp_dir.join("vis.uvfits");
    let out_ms_path = tmp_dir.join("vis.ms");
    let cal_model = tmp_dir.join("hyp_model.uvfits");

    #[rustfmt::skip]
    let cal_args = CalibrateUserArgs::parse_from(&[
        "di-calibrate",
        "--data", metafits, gpufits,
        "--source-list", &args.source_list.unwrap(),
        "--outputs",
            &format!("{}", out_uvfits_path.display()),
            &format!("{}", out_ms_path.display()),
        "--model-filename", &format!("{}", cal_model.display()),
        "--no-progress-bars",
    ]);

    // Run di-cal and check that it succeeds
    let result = di_calibrate::<PathBuf>(Box::new(cal_args), None, false);
    assert!(result.is_ok(), "result={:?} not ok", result.err().unwrap());

    let exp_timesteps = 1;
    let exp_channels = 32;

    // check uvfits file has been created, is readable
    assert!(out_uvfits_path.exists(), "out vis file not written");

    let uvfits_data =
        UvfitsReader::new(&out_uvfits_path, Some(metafits), &mut Delays::None).unwrap();

    let uvfits_ctx = uvfits_data.get_obs_context();

    // check ms file has been created, is readable
    assert!(out_ms_path.exists(), "out vis file not written");

    let ms_data = MS::new(&out_ms_path, Some(metafits), &mut Delays::None).unwrap();

    let ms_ctx = ms_data.get_obs_context();

    // XXX(dev): Can't write obsid to ms file without MwaObsContext "MS obsid not available (no MWA_GPS_TIME in OBSERVATION table)"
    // assert_eq!(uvfits_ctx.obsid, ms_ctx.obsid);
    assert_eq!(uvfits_ctx.obsid, Some(1090008640));
    assert_eq!(uvfits_ctx.timestamps, ms_ctx.timestamps);
    assert_eq!(
        uvfits_ctx.timestamps,
        vec![Epoch::from_gpst_seconds(1090008659.)]
    );
    assert_eq!(uvfits_ctx.all_timesteps, ms_ctx.all_timesteps);
    assert_eq!(uvfits_ctx.all_timesteps.len(), exp_timesteps);
    assert_eq!(uvfits_ctx.fine_chan_freqs, ms_ctx.fine_chan_freqs);
    assert_eq!(uvfits_ctx.fine_chan_freqs.len(), exp_channels);
}

pub fn synthesize_test_data(vis_ctx: &VisContext) -> (Array3<Jones<f32>>, Array3<f32>) {
    let shape = vis_ctx.sel_dims();

    let vis_data = Array3::<Jones<f32>>::from_shape_fn(shape, |(t, c, b)| {
        let (ant1, ant2) = vis_ctx.sel_baselines[b];
        Jones::from([t as f32, c as f32, ant1 as f32, ant2 as f32, 1., 0., 1., 0.])
    });

    let weight_data = Array3::<f32>::from_elem(shape, 1.);

    (vis_data, weight_data)
}

#[test]
pub fn test_cal_vis_output_avg_time() {
    let num_timesteps = 10;
    let num_channels = 10;
    let ant_pairs = vec![(0, 1), (0, 2), (1, 2)];

    let obsid = 1090000000;

    let vis_ctx = VisContext {
        num_sel_timesteps: num_timesteps,
        start_timestamp: Epoch::from_gpst_seconds(obsid as f64),
        int_time: Duration::from_f64(1., Unit::Second),
        num_sel_chans: num_channels,
        start_freq_hz: 128_000_000.,
        freq_resolution_hz: 10_000.,
        sel_baselines: ant_pairs,
        avg_time: 1,
        avg_freq: 1,
        num_vis_pols: 4,
    };

    let (vis_data, weight_data) = synthesize_test_data(&vis_ctx);

    let tmp_dir = TempDir::new().expect("couldn't make tmp dir").into_path();

    // XXX
    // let in_vis_path = tmp_dir.join("vis.uvfits");
    let in_vis_path = PathBuf::from("/tmp/vis.uvfits");

    let phase_centre = RADec::new_degrees(0., -27.);
    let array_pos = LatLngHeight::new_mwa();
    #[rustfmt::skip]
    let tile_xyzs = vec![
        XyzGeodetic { x: 0., y: 0., z: 0., },
        XyzGeodetic { x: 1., y: 0., z: 0., },
        XyzGeodetic { x: 0., y: 1., z: 0., },
    ];
    let tile_names = vec!["tile_0_0", "tile_1_0", "tile_0_1"];

    let mut writer = UvfitsWriter::from_marlu(
        &in_vis_path,
        &vis_ctx,
        Some(array_pos),
        phase_centre,
        Some(format!("synthesized test data {}", obsid)),
    )
    .unwrap();

    writer
        .write_vis_marlu(
            vis_data.view(),
            weight_data.view(),
            &vis_ctx,
            &tile_xyzs,
            false,
        )
        .unwrap();

    writer
        .write_uvfits_antenna_table(&tile_names, &tile_xyzs)
        .unwrap();

    let mut source_list = SourceList::new();
    source_list.insert(
        "source".into(),
        Source {
            components: vec1![SourceComponent {
                radec: phase_centre,
                comp_type: ComponentType::Point,
                flux_type: FluxDensityType::PowerLaw {
                    si: -0.7,
                    fd: FluxDensity {
                        freq: vis_ctx.start_freq_hz,
                        i: 1.0,
                        q: 0.0,
                        u: 0.0,
                        v: 0.0,
                    },
                },
            }],
        },
    );
    let srclist_path = tmp_dir.join("srclist.yaml");
    let mut srclist_buf = BufWriter::new(File::create(&srclist_path).unwrap());
    source_list_to_yaml(&mut srclist_buf, &source_list, None).unwrap();
    srclist_buf.flush().unwrap();

    let out_vis_path = tmp_dir.join("cal-vis.uvfits");

    let cal_args = CalibrateUserArgs {
        data: Some(vec![format!("{}", in_vis_path.display())]),
        outputs: Some(vec![out_vis_path.clone()]),
        source_list: Some(format!("{}", srclist_path.display())),
        no_beam: true,
        timesteps: Some(vec![1, 3, 9]),
        output_vis_time_average: Some("3s".into()),
        output_vis_freq_average: Some("20kHz".into()),
        ..Default::default()
    };

    let result = di_calibrate::<PathBuf>(Box::new(cal_args), None, false);
    assert!(result.is_ok(), "result={:?} not ok", result.err().unwrap());

    // we start at timestep index 1, with averaging 4. Averaged timesteps look like this:
    // [[1, _, 3], [_, _, _], [_, _, 9]]

    let uvfits_reader =
        UvfitsReader::new::<&PathBuf, &PathBuf>(&out_vis_path, None, &mut Delays::None).unwrap();

    let uvfits_ctx = uvfits_reader.get_obs_context();

    assert_eq!(
        uvfits_ctx.timestamps,
        vec![
            Epoch::from_gpst_seconds((obsid + 3) as f64),
            Epoch::from_gpst_seconds((obsid + 6) as f64),
            Epoch::from_gpst_seconds((obsid + 9) as f64)
        ]
    );

    assert_eq!(uvfits_ctx.guess_freq_res(), 20_000.);
    assert_eq!(uvfits_ctx.guess_time_res().in_unit(Unit::Second), 3.);

    let avg_shape = (
        vis_ctx.sel_baselines.len(),
        uvfits_ctx.fine_chan_freqs.len(),
    );
    let mut avg_data = Array2::from_elem(avg_shape, Jones::<f32>::default());
    let mut avg_weights = Array2::from_elem(avg_shape, 0_f32);

    let bl_map: HashMap<(usize, usize), usize> = vis_ctx
        .sel_baselines
        .iter()
        .cloned()
        .enumerate()
        .map(|(a, b)| (b, a))
        .collect();

    let flagged_fine_chans: HashSet<usize> =
        uvfits_ctx.flagged_fine_chans.iter().cloned().collect();

    uvfits_reader
        .read_crosses(
            avg_data.view_mut(),
            avg_weights.view_mut(),
            0,
            &bl_map,
            &flagged_fine_chans,
        )
        .unwrap();

    // weight should be the number of input visibilities that went in to the averaged visibility
    for &weight in avg_weights.iter() {
        // 2 timesteps, 2 channels
        assert_eq!(weight, 4_f32);
    }

    uvfits_reader
        .read_crosses(
            avg_data.view_mut(),
            avg_weights.view_mut(),
            1,
            &bl_map,
            &flagged_fine_chans,
        )
        .unwrap();

    // no selected timesteps went into timestep 1.
    for (&vis, &weight) in izip!(avg_data.iter(), avg_weights.iter()) {
        assert_eq!(vis, Jones::zero());
        assert_eq!(weight, 0.);
    }

    uvfits_reader
        .read_crosses(
            avg_data.view_mut(),
            avg_weights.view_mut(),
            2,
            &bl_map,
            &flagged_fine_chans,
        )
        .unwrap();

    // weight should be the number of input visibilities that went in to the averaged visibility
    for &weight in avg_weights.iter() {
        // 1 timesteps, 2 channels
        assert_eq!(weight, 2_f32);
    }
}
