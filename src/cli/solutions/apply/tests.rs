// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Tests for solutions-apply.

use std::{collections::HashSet, path::Path};

use approx::{assert_abs_diff_eq, assert_relative_eq};
use crossbeam_utils::atomic::AtomicCell;
use itertools::{izip, Itertools};
use marlu::Jones;
use ndarray::prelude::*;
use serial_test::serial;
use tempfile::TempDir;
use vec1::Vec1;

use super::*;
use crate::{
    cli::{common::InputVisArgs, vis_convert::VisConvertArgs},
    io::read::{
        fits::{fits_get_required_key, fits_open, fits_open_hdu},
        MsReader, UvfitsReader, VisRead,
    },
    math::TileBaselineFlags,
    tests::{
        get_reduced_1090008640_ms, get_reduced_1090008640_raw, get_reduced_1090008640_raw_pbs,
        get_reduced_1090008640_uvfits, DataAsPathBufs, DataAsStrings,
    },
    CalibrationSolutions,
};

fn test_solutions_apply_trivial(mut args: SolutionsApplyArgs) {
    let tmp_dir = TempDir::new().unwrap();
    let output = tmp_dir.path().join("test.uvfits");
    let error = AtomicCell::new(false);
    let metafits = PathBuf::from(args.data_args.files.as_ref().unwrap().last().unwrap());

    // Make some solutions that are all identity; the output visibilities should
    // be the same as the input.
    let sols_file = get_1090008640_identity_solutions_file(tmp_dir.path());
    args.solutions = Some(sols_file.display().to_string());
    args.outputs = Some(vec![output.clone()]);
    args.output_vis_time_average = None;
    args.output_vis_freq_average = None;

    let mut params = args.parse().unwrap();

    // Get the reference visibilities.
    let obs_context = params.input_vis_params.vis_reader.get_obs_context();
    let total_num_tiles = obs_context.get_total_num_tiles();
    let total_num_baselines = (total_num_tiles * (total_num_tiles - 1)) / 2;
    let total_num_channels = obs_context.fine_chan_freqs.len();
    let tile_baseline_flags = &params.input_vis_params.tile_baseline_flags;
    let flagged_tiles = &tile_baseline_flags.flagged_tiles;
    assert!(flagged_tiles.is_empty());
    let flagged_fine_chans = &params.input_vis_params.spw.flagged_chan_indices;

    let mut ref_crosses = Array2::zeros((total_num_channels, total_num_baselines));
    let mut ref_cross_weights = Array2::zeros((total_num_channels, total_num_baselines));
    let mut ref_autos = Array2::zeros((total_num_channels, total_num_tiles));
    let mut ref_auto_weights = Array2::zeros((total_num_channels, total_num_tiles));
    params
        .input_vis_params
        .read_timeblock(
            params.input_vis_params.timeblocks.first(),
            ref_crosses.view_mut(),
            ref_cross_weights.view_mut(),
            Some((ref_autos.view_mut(), ref_auto_weights.view_mut())),
            &error,
        )
        .unwrap();
    params.run().unwrap();

    // Read the output visibilities.
    let output_data = UvfitsReader::new(output.clone(), Some(&metafits), None).unwrap();
    let mut crosses = Array2::zeros((total_num_channels, total_num_baselines));
    let mut cross_weights = Array2::zeros((total_num_channels, total_num_baselines));
    let mut autos = Array2::zeros((total_num_channels, total_num_tiles));
    let mut auto_weights = Array2::zeros((total_num_channels, total_num_tiles));
    output_data
        .read_crosses_and_autos(
            crosses.view_mut(),
            cross_weights.view_mut(),
            autos.view_mut(),
            auto_weights.view_mut(),
            0,
            tile_baseline_flags,
            flagged_fine_chans,
        )
        .unwrap();

    assert_abs_diff_eq!(crosses, ref_crosses);
    assert_abs_diff_eq!(cross_weights, ref_cross_weights);
    assert_abs_diff_eq!(autos, ref_autos);
    assert_abs_diff_eq!(auto_weights, ref_auto_weights);

    // Now make the solutions all "2"; the output visibilities should be 4x the
    // input.
    params
        .input_vis_params
        .solutions
        .as_mut()
        .unwrap()
        .di_jones
        .mapv_inplace(|j| j * 2.0);
    params.run().unwrap();

    // Read the output visibilities.
    let output_data = UvfitsReader::new(output.clone(), Some(&metafits), None).unwrap();
    crosses.fill(Jones::default());
    cross_weights.fill(0.0);
    autos.fill(Jones::default());
    auto_weights.fill(0.0);
    output_data
        .read_crosses_and_autos(
            crosses.view_mut(),
            cross_weights.view_mut(),
            autos.view_mut(),
            auto_weights.view_mut(),
            0,
            tile_baseline_flags,
            flagged_fine_chans,
        )
        .unwrap();

    assert_abs_diff_eq!(crosses, &ref_crosses * 4.0);
    assert_abs_diff_eq!(cross_weights, ref_cross_weights);
    assert_abs_diff_eq!(autos, &ref_autos * 4.0);
    assert_abs_diff_eq!(auto_weights, ref_auto_weights);

    // Now make the solutions equal to the tile index.
    params
        .input_vis_params
        .solutions
        .as_mut()
        .unwrap()
        .di_jones
        .slice_mut(s![0, .., ..])
        .outer_iter_mut()
        .enumerate()
        .for_each(|(i_tile, mut sols)| {
            sols.fill(Jones::identity() * (i_tile + 1) as f64);
        });
    params.run().unwrap();

    // Read the output visibilities.
    let output_data = UvfitsReader::new(output.clone(), Some(&metafits), None).unwrap();
    crosses.fill(Jones::default());
    cross_weights.fill(0.0);
    autos.fill(Jones::default());
    auto_weights.fill(0.0);
    output_data
        .read_crosses_and_autos(
            crosses.view_mut(),
            cross_weights.view_mut(),
            autos.view_mut(),
            auto_weights.view_mut(),
            0,
            tile_baseline_flags,
            flagged_fine_chans,
        )
        .unwrap();

    for (i_baseline, (baseline, ref_baseline)) in crosses
        .axis_iter(Axis(1))
        .zip_eq(ref_crosses.axis_iter(Axis(1)))
        .enumerate()
    {
        let (tile1, tile2) = tile_baseline_flags.unflagged_cross_baseline_to_tile_map[&i_baseline];
        let ref_baseline =
            ref_baseline.mapv(Jones::<f64>::from) * (tile1 + 1) as f64 * (tile2 + 1) as f64;
        // Need to use a relative test because the numbers get quite big and
        // absolute epsilons get scarily big.
        assert_relative_eq!(
            baseline.mapv(Jones::<f64>::from),
            ref_baseline,
            max_relative = 1e-7
        );
    }
    assert_abs_diff_eq!(cross_weights, ref_cross_weights);

    for (i_tile, (baseline, ref_baseline)) in autos
        .axis_iter(Axis(1))
        .zip_eq(ref_autos.axis_iter(Axis(1)))
        .enumerate()
    {
        let i_tile = tile_baseline_flags.unflagged_auto_index_to_tile_map[&i_tile];
        let ref_baseline =
            ref_baseline.mapv(Jones::<f64>::from) * (i_tile + 1) as f64 * (i_tile + 1) as f64;
        assert_relative_eq!(
            baseline.mapv(Jones::<f64>::from),
            ref_baseline,
            max_relative = 1e-7
        );
    }
    assert_abs_diff_eq!(auto_weights, ref_auto_weights);

    // Use tile indices for solutions again, but now flag some tiles.
    let flags = [10, 78];
    params.input_vis_params.tile_baseline_flags =
        TileBaselineFlags::new(total_num_tiles, HashSet::from(flags));
    let tile_baseline_flags = &params.input_vis_params.tile_baseline_flags;
    // Re-generate the reference data.
    let num_unflagged_tiles = total_num_tiles - tile_baseline_flags.flagged_tiles.len();
    let num_unflagged_cross_baselines = (num_unflagged_tiles * (num_unflagged_tiles - 1)) / 2;
    let mut ref_crosses_fb = Array2::zeros((total_num_channels, num_unflagged_cross_baselines));
    let mut ref_cross_weights_fb =
        Array2::zeros((total_num_channels, num_unflagged_cross_baselines));
    let mut ref_autos_fb = Array2::zeros((total_num_channels, num_unflagged_tiles));
    let mut ref_auto_weights_fb = Array2::zeros((total_num_channels, num_unflagged_tiles));
    params.run().unwrap();

    // Read the output visibilities.
    let output_data = UvfitsReader::new(output.clone(), Some(&metafits), None).unwrap();
    let mut crosses_fb = Array2::zeros((total_num_channels, num_unflagged_cross_baselines));
    let mut cross_weights_fb = Array2::zeros((total_num_channels, num_unflagged_cross_baselines));
    let mut autos_fb = Array2::zeros((total_num_channels, num_unflagged_tiles));
    let mut auto_weights_fb = Array2::zeros((total_num_channels, num_unflagged_tiles));
    output_data
        .read_crosses_and_autos(
            crosses_fb.view_mut(),
            cross_weights_fb.view_mut(),
            autos_fb.view_mut(),
            auto_weights_fb.view_mut(),
            0,
            tile_baseline_flags,
            flagged_fine_chans,
        )
        .unwrap();

    // Read in the newly-flagged data without any solutions being applied.
    params
        .input_vis_params
        .vis_reader
        .read_crosses_and_autos(
            ref_crosses_fb.view_mut(),
            ref_cross_weights_fb.view_mut(),
            ref_autos_fb.view_mut(),
            ref_auto_weights_fb.view_mut(),
            obs_context.unflagged_timesteps[0],
            tile_baseline_flags,
            flagged_fine_chans,
        )
        .unwrap();

    for (i_baseline, (baseline, ref_baseline)) in crosses_fb
        .axis_iter(Axis(1))
        .zip_eq(ref_crosses_fb.axis_iter(Axis(1)))
        .enumerate()
    {
        let (tile1, tile2) = tile_baseline_flags.unflagged_cross_baseline_to_tile_map[&i_baseline];
        let ref_baseline =
            ref_baseline.mapv(Jones::<f64>::from) * (tile1 + 1) as f64 * (tile2 + 1) as f64;
        assert_relative_eq!(
            baseline.mapv(Jones::<f64>::from),
            ref_baseline,
            max_relative = 1e-7
        );
    }
    for (i_tile, (tile, ref_tile)) in autos_fb
        .axis_iter(Axis(1))
        .zip_eq(ref_autos_fb.axis_iter(Axis(1)))
        .enumerate()
    {
        let tile_factor = tile_baseline_flags.unflagged_auto_index_to_tile_map[&i_tile];
        let ref_tile =
            ref_tile.mapv(Jones::<f64>::from) * (tile_factor + 1) as f64 * (tile_factor + 1) as f64;
        assert_relative_eq!(tile.mapv(Jones::<f64>::from), ref_tile, max_relative = 1e-7);
    }
    assert_abs_diff_eq!(cross_weights_fb, ref_cross_weights_fb);
    assert_abs_diff_eq!(auto_weights_fb, ref_auto_weights_fb);

    // Finally, flag some channels.
    let flags = [3, 18];
    params
        .input_vis_params
        .spw
        .flagged_chan_indices
        .extend(flags);
    params
        .input_vis_params
        .spw
        .flagged_chanblock_indices
        .extend(flags);
    let flagged_fine_chans = &params.input_vis_params.spw.flagged_chan_indices;
    // Remove the newly-flagged chanblocks. (This is awkward because SPWs
    // weren't designed to be modified.)
    params.input_vis_params.spw.chanblocks = (0..)
        .zip(params.input_vis_params.spw.chanblocks)
        .filter(|(i, _)| !flagged_fine_chans.contains(i))
        .map(|(_, c)| c)
        .collect();
    params.run().unwrap();

    // Read the output visibilities.
    let output_data = UvfitsReader::new(output, Some(&metafits), None).unwrap();
    crosses_fb.fill(Jones::default());
    cross_weights_fb.fill(0.0);
    autos_fb.fill(Jones::default());
    auto_weights_fb.fill(0.0);
    output_data
        .read_crosses_and_autos(
            crosses_fb.view_mut(),
            cross_weights_fb.view_mut(),
            autos_fb.view_mut(),
            auto_weights_fb.view_mut(),
            0,
            tile_baseline_flags,
            // We want to read all channels, even the flagged ones.
            &HashSet::new(),
        )
        .unwrap();

    // Read in the raw data without any solutions or flags being applied.
    params
        .input_vis_params
        .vis_reader
        .read_crosses_and_autos(
            ref_crosses_fb.view_mut(),
            ref_cross_weights_fb.view_mut(),
            ref_autos_fb.view_mut(),
            ref_auto_weights_fb.view_mut(),
            obs_context.unflagged_timesteps[0],
            tile_baseline_flags,
            &HashSet::new(),
        )
        .unwrap();

    // Manually flag the flagged channels.
    for &f in flagged_fine_chans {
        let f = usize::from(f);
        ref_crosses_fb.slice_mut(s![f, ..]).fill(Jones::default());
        ref_cross_weights_fb.slice_mut(s![f, ..]).fill(-0.0);
        ref_autos_fb.slice_mut(s![f, ..]).fill(Jones::default());
        ref_auto_weights_fb.slice_mut(s![f, ..]).fill(-0.0);
    }
    for (i_baseline, (baseline_f, ref_baseline_f)) in crosses_fb
        .axis_iter(Axis(1))
        .zip_eq(ref_crosses_fb.axis_iter(Axis(1)))
        .enumerate()
    {
        let (tile1, tile2) = tile_baseline_flags.unflagged_cross_baseline_to_tile_map[&i_baseline];
        let ref_baseline_f =
            ref_baseline_f.mapv(Jones::<f64>::from) * (tile1 + 1) as f64 * (tile2 + 1) as f64;
        assert_relative_eq!(
            baseline_f.mapv(Jones::<f64>::from),
            ref_baseline_f,
            max_relative = 1e-7
        );
    }
    for (i_tile, (tile, ref_tile)) in autos_fb
        .axis_iter(Axis(1))
        .zip_eq(ref_autos_fb.axis_iter(Axis(1)))
        .enumerate()
    {
        let tile_factor = tile_baseline_flags.unflagged_auto_index_to_tile_map[&i_tile];
        let ref_tile =
            ref_tile.mapv(Jones::<f64>::from) * (tile_factor + 1) as f64 * (tile_factor + 1) as f64;
        assert_relative_eq!(tile.mapv(Jones::<f64>::from), ref_tile, max_relative = 1e-7);
    }
    assert_abs_diff_eq!(cross_weights_fb, ref_cross_weights_fb);
    assert_abs_diff_eq!(auto_weights_fb, ref_auto_weights_fb);
}

#[test]
fn test_solutions_apply_trivial_raw() {
    let DataAsStrings {
        metafits,
        vis: mut files,
        mwafs: _,
        srclist: _,
    } = get_reduced_1090008640_raw();
    files.push(metafits);

    let args = SolutionsApplyArgs {
        data_args: InputVisArgs {
            files: Some(files),
            pfb_flavour: Some("none".to_string()),
            no_digital_gains: false,
            no_cable_length_correction: false,
            no_geometric_correction: false,
            ignore_input_data_fine_channel_flags: true,
            ..Default::default()
        },
        ..Default::default()
    };

    test_solutions_apply_trivial(args)
}

// If all data-reading routines are working correctly, these extra tests are
// redundant. But, it gives us more confidence that things are working.

#[test]
#[serial]
fn test_solutions_apply_trivial_ms() {
    let DataAsStrings {
        metafits,
        vis: mut files,
        mwafs: _,
        srclist: _,
    } = get_reduced_1090008640_ms();
    files.push(metafits);

    let args = SolutionsApplyArgs {
        data_args: InputVisArgs {
            files: Some(files),
            ignore_input_data_fine_channel_flags: true,
            ..Default::default()
        },
        ..Default::default()
    };

    test_solutions_apply_trivial(args)
}

#[test]
fn test_solutions_apply_trivial_uvfits() {
    let DataAsStrings {
        metafits,
        vis: mut files,
        mwafs: _,
        srclist: _,
    } = get_reduced_1090008640_uvfits();
    files.push(metafits);

    let args = SolutionsApplyArgs {
        data_args: InputVisArgs {
            files: Some(files),
            ignore_input_data_fine_channel_flags: true,
            ..Default::default()
        },
        ..Default::default()
    };

    test_solutions_apply_trivial(args)
}

pub(crate) fn get_1090008640_identity_solutions_file(tmp_dir: &Path) -> PathBuf {
    let sols = CalibrationSolutions {
        di_jones: Array3::from_elem((1, 128, 32), Jones::identity()),
        chanblock_freqs: Some(
            Vec1::try_from(Array1::linspace(196495000.0, 197735000.0, 32).to_vec()).unwrap(),
        ),
        ..Default::default()
    };
    let file = tmp_dir.join("sols.fits");
    sols.write_solutions_from_ext::<&Path>(&file).unwrap();
    file
}

#[test]
fn test_1090008640_solutions_apply_writes_vis_uvfits() {
    let tmp_dir = TempDir::new().expect("couldn't make tmp dir");
    let DataAsPathBufs { metafits, vis, .. } = get_reduced_1090008640_raw_pbs();
    let solutions = get_1090008640_identity_solutions_file(tmp_dir.path());
    let out_vis_path = tmp_dir.path().join("vis.uvfits");

    #[rustfmt::skip]
    let args = SolutionsApplyArgs::parse_from([
        "solutions-apply",
        "--data", &format!("{}", metafits.display()), &format!("{}", vis[0].display()),
        "--solutions", &format!("{}", solutions.display()),
        "--outputs", &format!("{}", out_vis_path.display()),
    ]);

    // Run solutions-apply and check that it succeeds
    let result = args.run(false);
    assert!(result.is_ok(), "result={:?} not ok", result.err().unwrap());

    // check vis file has been created, is readable
    assert!(out_vis_path.exists(), "out vis file not written");
    let exp_timesteps = 1;
    let exp_baselines = 8256;
    let exp_channels = 32;

    let mut out_vis = fits_open(&out_vis_path).unwrap();
    let hdu0 = fits_open_hdu(&mut out_vis, 0).unwrap();
    let gcount: String = fits_get_required_key(&mut out_vis, &hdu0, "GCOUNT").unwrap();
    assert_eq!(
        gcount.parse::<usize>().unwrap(),
        exp_timesteps * exp_baselines
    );
    let num_fine_freq_chans: String = fits_get_required_key(&mut out_vis, &hdu0, "NAXIS4").unwrap();
    assert_eq!(num_fine_freq_chans.parse::<usize>().unwrap(), exp_channels);
}

#[test]
fn test_1090008640_solutions_apply_writes_vis_uvfits_no_autos() {
    let tmp_dir = TempDir::new().expect("couldn't make tmp dir");
    let DataAsPathBufs { metafits, vis, .. } = get_reduced_1090008640_raw_pbs();
    let solutions = get_1090008640_identity_solutions_file(tmp_dir.path());
    let out_vis_path = tmp_dir.path().join("vis.uvfits");

    #[rustfmt::skip]
    let args = SolutionsApplyArgs::parse_from([
        "solutions-apply",
        "--data", &format!("{}", metafits.display()), &format!("{}", vis[0].display()),
        "--solutions", &format!("{}", solutions.display()),
        "--outputs", &format!("{}", out_vis_path.display()),
        "--no-autos",
    ]);

    // Run solutions-apply and check that it succeeds
    let result = args.run(false);
    assert!(result.is_ok(), "result={:?} not ok", result.err().unwrap());

    // check vis file has been created, is readable
    assert!(out_vis_path.exists(), "out vis file not written");
    let exp_timesteps = 1;
    let exp_baselines = 8128;
    let exp_channels = 32;

    let mut out_vis = fits_open(&out_vis_path).unwrap();
    let hdu0 = fits_open_hdu(&mut out_vis, 0).unwrap();
    let gcount: String = fits_get_required_key(&mut out_vis, &hdu0, "GCOUNT").unwrap();
    assert_eq!(
        gcount.parse::<usize>().unwrap(),
        exp_timesteps * exp_baselines
    );
    let num_fine_freq_chans: String = fits_get_required_key(&mut out_vis, &hdu0, "NAXIS4").unwrap();
    assert_eq!(num_fine_freq_chans.parse::<usize>().unwrap(), exp_channels);
}

#[test]
fn test_1090008640_solutions_apply_writes_vis_uvfits_avg_freq() {
    let tmp_dir = TempDir::new().expect("couldn't make tmp dir");
    let DataAsStrings {
        metafits, mut vis, ..
    } = get_reduced_1090008640_raw();
    let solutions = get_1090008640_identity_solutions_file(tmp_dir.path());
    let out_vis_path = tmp_dir.path().join("vis.uvfits");

    let freq_avg_factor = 2;

    #[rustfmt::skip]
    let args = SolutionsApplyArgs::parse_from([
        "solutions-apply",
        "--data", &metafits, &vis.swap_remove(0),
        "--solutions", &format!("{}", solutions.display()),
        "--outputs", &format!("{}", out_vis_path.display()),
        "--freq-average", &format!("{freq_avg_factor}"),
    ]);

    // Run solutions-apply and check that it succeeds
    let result = args.run(false);
    assert!(result.is_ok(), "result={:?} not ok", result.err().unwrap());

    // check vis file has been created, is readable
    assert!(out_vis_path.exists(), "out vis file not written");
    let exp_timesteps = 1;
    let exp_baselines = 8256;
    let exp_channels = 16;

    let mut out_vis = fits_open(&out_vis_path).unwrap();
    let hdu0 = fits_open_hdu(&mut out_vis, 0).unwrap();
    let gcount: String = fits_get_required_key(&mut out_vis, &hdu0, "GCOUNT").unwrap();
    assert_eq!(
        gcount.parse::<usize>().unwrap(),
        exp_timesteps * exp_baselines
    );
    let num_fine_freq_chans: String = fits_get_required_key(&mut out_vis, &hdu0, "NAXIS4").unwrap();
    std::fs::copy(out_vis_path, PathBuf::from("/tmp/hyp_test.uvfits")).unwrap();
    assert_eq!(num_fine_freq_chans.parse::<usize>().unwrap(), exp_channels);
}

#[test]
#[serial]
fn test_1090008640_solutions_apply_writes_vis_uvfits_and_ms() {
    let tmp_dir = TempDir::new().expect("couldn't make tmp dir");
    let DataAsPathBufs { metafits, vis, .. } = get_reduced_1090008640_raw_pbs();
    let solutions = get_1090008640_identity_solutions_file(tmp_dir.path());
    let out_uvfits_path = tmp_dir.path().join("vis.uvfits");
    let out_ms_path = tmp_dir.path().join("vis.ms");

    #[rustfmt::skip]
    let args = SolutionsApplyArgs::parse_from([
        "solutions-apply",
        "--data", &format!("{}", metafits.display()), &format!("{}", vis[0].display()),
        "--solutions", &format!("{}", solutions.display()),
        "--outputs",
            &format!("{}", out_uvfits_path.display()),
            &format!("{}", out_ms_path.display()),
    ]);

    // Run solutions-apply and check that it succeeds
    let result = args.run(false);
    assert!(result.is_ok(), "result={:?} not ok", result.err().unwrap());

    // check vis file has been created, is readable
    assert!(out_uvfits_path.exists(), "out vis file not written");
    let exp_timesteps = 1;
    let exp_channels = 32;

    let uvfits_data = UvfitsReader::new(out_uvfits_path, Some(&metafits), None).unwrap();

    let uvfits_ctx = uvfits_data.get_obs_context();

    // check ms file has been created, is readable
    assert!(out_ms_path.exists(), "out vis file not written");

    let ms_data = MsReader::new(out_ms_path, None, Some(&metafits), None).unwrap();

    let ms_ctx = ms_data.get_obs_context();

    assert_eq!(uvfits_ctx.obsid, ms_ctx.obsid);
    assert_eq!(uvfits_ctx.timestamps, ms_ctx.timestamps);
    assert_eq!(uvfits_ctx.timestamps.len(), 1);
    assert_abs_diff_eq!(uvfits_ctx.timestamps[0].to_gpst_seconds(), 1090008658.);
    assert_eq!(uvfits_ctx.all_timesteps, ms_ctx.all_timesteps);
    assert_eq!(uvfits_ctx.all_timesteps.len(), exp_timesteps);
    assert_eq!(uvfits_ctx.fine_chan_freqs, ms_ctx.fine_chan_freqs);
    assert_eq!(uvfits_ctx.fine_chan_freqs.len(), exp_channels);
}

#[test]
/// This test is probably only re-testing things that get tested above, but...
/// why not.
fn test_1090008640_solutions_apply_correct_vis() {
    let tmp_dir = TempDir::new().expect("couldn't make tmp dir");

    let mut sols = CalibrationSolutions {
        di_jones: Array3::from_shape_fn((1, 128, 32), |(_, i, _)| {
            Jones::identity() * (i + 1) as f64
        }),
        ..Default::default()
    };
    let sols_file = tmp_dir.path().join("sols.fits");
    sols.write_solutions_from_ext::<&Path>(&sols_file).unwrap();
    let sols_file_string = sols_file.display().to_string();

    let flagged_tiles = HashSet::from([1, 3, 5]);

    let DataAsStrings {
        metafits, mut vis, ..
    } = get_reduced_1090008640_uvfits();
    let metafits_pb = PathBuf::from(&metafits);
    let uvfits = vis.swap_remove(0);
    let vis_out = tmp_dir.path().join("vis.uvfits");
    let vis_out_string = vis_out.display().to_string();

    #[rustfmt::skip]
    let mut args = vec![
        "solutions-apply",
        "--data", &metafits, &uvfits,
        "--solutions", &sols_file_string,
        "--outputs", &vis_out_string,
    ];
    let flag_strings = flagged_tiles
        .iter()
        .map(|f| format!("{f}"))
        .collect::<Vec<_>>();
    if !flag_strings.is_empty() {
        args.push("--tile-flags");
        for f in &flag_strings {
            args.push(f);
        }
    }
    let args = SolutionsApplyArgs::parse_from(args);

    // Run solutions-apply and check that it succeeds
    let result = args.run(false);
    assert!(result.is_ok(), "result={:?} not ok", result.err().unwrap());
    assert!(vis_out.exists(), "out vis file not written");

    let uncal_reader =
        UvfitsReader::new(PathBuf::from(uvfits.clone()), Some(&metafits_pb), None).unwrap();
    let cal_reader = UvfitsReader::new(vis_out, Some(&metafits_pb), None).unwrap();
    let obs_context = cal_reader.get_obs_context();

    let total_num_tiles = obs_context.get_total_num_tiles();
    let total_num_cross_baselines = {
        let n = total_num_tiles;
        (n * (n - 1)) / 2
    };
    let num_unflagged_tiles = total_num_tiles - flagged_tiles.len();
    assert_eq!(num_unflagged_tiles, 125);
    let num_unflagged_cross_baselines = {
        let n = num_unflagged_tiles;
        (n * (n - 1)) / 2
    };

    // Check cross correlations.
    let unflagged_maps = TileBaselineFlags::new(total_num_tiles, HashSet::new());
    let flagged_maps = TileBaselineFlags::new(total_num_tiles, flagged_tiles);

    let mut uncal_cross_vis_data =
        Array2::zeros((obs_context.fine_chan_freqs.len(), total_num_cross_baselines));
    let mut cal_cross_vis_data = Array2::zeros((
        obs_context.fine_chan_freqs.len(),
        num_unflagged_cross_baselines,
    ));
    let mut uncal_cross_vis_weights = Array2::zeros(uncal_cross_vis_data.dim());
    let mut cal_cross_vis_weights = Array2::zeros(cal_cross_vis_data.dim());
    uncal_reader
        .read_crosses(
            uncal_cross_vis_data.view_mut(),
            uncal_cross_vis_weights.view_mut(),
            0,
            &unflagged_maps,
            &HashSet::new(),
        )
        .unwrap();
    cal_reader
        .read_crosses(
            cal_cross_vis_data.view_mut(),
            cal_cross_vis_weights.view_mut(),
            0,
            &flagged_maps,
            &HashSet::new(),
        )
        .unwrap();
    for tile1 in 0..total_num_tiles - 2 {
        for tile2 in tile1 + 1..total_num_tiles - 1 {
            if let Some(i_flagged_baseline) = flagged_maps
                .tile_to_unflagged_cross_baseline_map
                .get(&(tile1, tile2))
                .copied()
            {
                let i_unflagged_baseline =
                    unflagged_maps.tile_to_unflagged_cross_baseline_map[&(tile1, tile2)];
                let expected_multiplier = ((tile1 + 1) * (tile2 + 1)) as f32;
                assert_abs_diff_eq!(
                    uncal_cross_vis_data
                        .slice(s![.., i_unflagged_baseline])
                        .to_owned()
                        * expected_multiplier,
                    cal_cross_vis_data.slice(s![.., i_flagged_baseline]),
                    epsilon = 1e-10
                );
            }
        }
    }

    // Check auto correlations.
    let mut uncal_auto_vis_data =
        Array2::zeros((obs_context.fine_chan_freqs.len(), total_num_tiles));
    let mut cal_auto_vis_data =
        Array2::zeros((obs_context.fine_chan_freqs.len(), num_unflagged_tiles));
    let mut uncal_auto_vis_weights = Array2::zeros(uncal_auto_vis_data.dim());
    let mut cal_auto_vis_weights = Array2::zeros(cal_auto_vis_data.dim());
    uncal_reader
        .read_autos(
            uncal_auto_vis_data.view_mut(),
            uncal_auto_vis_weights.view_mut(),
            0,
            &unflagged_maps,
            &HashSet::new(),
        )
        .unwrap();
    cal_reader
        .read_autos(
            cal_auto_vis_data.view_mut(),
            cal_auto_vis_weights.view_mut(),
            0,
            &flagged_maps,
            &HashSet::new(),
        )
        .unwrap();
    for tile in 0..total_num_tiles - 1 {
        if let Some(i_flagged_tile) = flagged_maps
            .tile_to_unflagged_cross_baseline_map
            .get(&(tile, tile))
            .copied()
        {
            let expected_multiplier = ((tile + 1) * (tile + 1)) as f32;
            assert_abs_diff_eq!(
                uncal_auto_vis_data.slice(s![.., tile]).to_owned() * expected_multiplier,
                cal_auto_vis_data.slice(s![.., i_flagged_tile]),
                epsilon = 1e-10
            );
        }
    }

    // Do all this again, but this time make some tile solutions all NaN. The
    // output uvfits should have *exactly* the same data for tiles we haven't
    // affected.
    let bad_sols = [10, 100];
    for i_bad_sol in bad_sols {
        sols.di_jones
            .slice_mut(s![0, i_bad_sol, ..])
            .fill(Jones::nan());
    }
    let sols_file = tmp_dir.path().join("sols.fits");
    sols.write_solutions_from_ext::<&Path>(&sols_file).unwrap();
    let vis_out = tmp_dir.path().join("vis2.uvfits");
    let vis_out_string = vis_out.display().to_string();

    #[rustfmt::skip]
    let mut args = vec![
        "solutions-apply",
        "--data", &metafits, &uvfits,
        "--solutions", &sols_file_string,
        "--outputs", &vis_out_string,
    ];
    if !flag_strings.is_empty() {
        args.push("--tile-flags");
        for f in &flag_strings {
            args.push(f);
        }
    }
    let args = SolutionsApplyArgs::parse_from(args);

    // Run solutions-apply and check that it succeeds
    let result = args.run(false);
    assert!(result.is_ok(), "result={:?} not ok", result.err().unwrap());
    assert!(vis_out.exists(), "out vis file not written");

    let cal2_reader = UvfitsReader::new(vis_out, Some(&metafits_pb), None).unwrap();
    let mut cal2_cross_vis_data = Array2::zeros((
        obs_context.fine_chan_freqs.len(),
        num_unflagged_cross_baselines,
    ));
    let mut cal2_cross_vis_weights = Array2::zeros(cal2_cross_vis_data.dim());
    let mut cal2_auto_vis_data =
        Array2::zeros((obs_context.fine_chan_freqs.len(), num_unflagged_tiles));
    let mut cal2_auto_vis_weights = Array2::zeros(cal2_auto_vis_data.dim());
    cal2_reader
        .read_crosses_and_autos(
            cal2_cross_vis_data.view_mut(),
            cal2_cross_vis_weights.view_mut(),
            cal2_auto_vis_data.view_mut(),
            cal2_auto_vis_weights.view_mut(),
            0,
            &flagged_maps,
            &HashSet::new(),
        )
        .unwrap();

    for i_bl in 0..num_unflagged_cross_baselines {
        let (tile1, tile2) = flagged_maps.unflagged_cross_baseline_to_tile_map[&i_bl];
        if bad_sols.contains(&tile1) || bad_sols.contains(&tile2) {
            assert!(cal2_cross_vis_data
                .slice(s![.., i_bl])
                .iter()
                .flat_map(|j| j.to_float_array())
                .all(|f| f.abs() < f32::EPSILON));
            assert_abs_diff_eq!(
                cal_cross_vis_weights.slice(s![.., i_bl]).map(|w| -w.abs()),
                cal2_cross_vis_weights.slice(s![.., i_bl])
            );
            assert!(cal2_cross_vis_weights
                .slice(s![.., i_bl])
                .iter()
                .all(|w| *w < 0.0));
        } else {
            assert_abs_diff_eq!(
                cal_cross_vis_data.slice(s![.., i_bl]),
                cal2_cross_vis_data.slice(s![.., i_bl])
            );
            assert_abs_diff_eq!(
                cal_cross_vis_weights.slice(s![.., i_bl]),
                cal2_cross_vis_weights.slice(s![.., i_bl])
            );
        }
    }

    for i_tile in 0..num_unflagged_tiles {
        let tile = flagged_maps.unflagged_auto_index_to_tile_map[&i_tile];
        if bad_sols.contains(&tile) {
            assert!(cal2_auto_vis_data
                .slice(s![.., i_tile])
                .iter()
                .flat_map(|j| j.to_float_array())
                .all(|f| f.abs() < f32::EPSILON));
            assert_abs_diff_eq!(
                cal_auto_vis_weights.slice(s![.., i_tile]).map(|w| -w.abs()),
                cal2_auto_vis_weights.slice(s![.., i_tile])
            );
            assert!(cal2_auto_vis_weights
                .slice(s![.., i_tile])
                .iter()
                .all(|w| *w < 0.0));
        } else {
            assert_abs_diff_eq!(
                cal_auto_vis_data.slice(s![.., i_tile]),
                cal2_auto_vis_data.slice(s![.., i_tile])
            );
            assert_abs_diff_eq!(
                cal_auto_vis_weights.slice(s![.., i_tile]),
                cal2_auto_vis_weights.slice(s![.., i_tile])
            );
        }
    }
}

#[test]
fn test_solutions_apply_works_with_implicit_or_explicit_sols() {
    let DataAsStrings {
        metafits,
        vis: mut files,
        mwafs: _,
        srclist: _,
    } = get_reduced_1090008640_raw();
    files.push(metafits.clone());

    let mut args = SolutionsApplyArgs {
        data_args: InputVisArgs {
            files: Some(files),
            pfb_flavour: Some("none".to_string()),
            no_digital_gains: false,
            no_cable_length_correction: false,
            no_geometric_correction: false,
            ignore_input_data_fine_channel_flags: true,
            ..Default::default()
        },
        ..Default::default()
    };

    let tmp_dir = TempDir::new().unwrap();
    let explicit_output = tmp_dir.path().join("explicit.uvfits");
    let sols_file = get_1090008640_identity_solutions_file(tmp_dir.path());
    args.solutions = Some(sols_file.display().to_string());
    args.outputs = Some(vec![explicit_output.clone()]);
    args.clone().parse().unwrap().run().unwrap();

    let implicit_output = tmp_dir.path().join("implicit.uvfits");
    args.data_args
        .files
        .as_mut()
        .unwrap()
        .push(sols_file.display().to_string());
    args.solutions = None;
    args.outputs = Some(vec![implicit_output.clone()]);
    args.parse().unwrap().run().unwrap();

    // The visibilities should be exactly the same.
    let error = AtomicCell::new(false);
    let explicit_vis_params = InputVisArgs {
        files: Some(vec![
            metafits.clone(),
            explicit_output.display().to_string(),
        ]),
        ..Default::default()
    }
    .parse("")
    .unwrap();
    let obs_context = explicit_vis_params.get_obs_context();
    let total_num_tiles = obs_context.get_total_num_tiles();
    let total_num_baselines = (total_num_tiles * (total_num_tiles - 1)) / 2;
    let total_num_channels = obs_context.fine_chan_freqs.len();
    let mut explicit_crosses = Array3::zeros((
        explicit_vis_params.timeblocks.len(),
        total_num_channels,
        total_num_baselines,
    ));
    let mut explicit_cross_weights = Array3::zeros((
        explicit_vis_params.timeblocks.len(),
        total_num_channels,
        total_num_baselines,
    ));
    let mut explicit_autos = Array3::zeros((
        explicit_vis_params.timeblocks.len(),
        total_num_channels,
        total_num_tiles,
    ));
    let mut explicit_auto_weights = Array3::zeros((
        explicit_vis_params.timeblocks.len(),
        total_num_channels,
        total_num_tiles,
    ));
    for (timeblock, crosses, cross_weights, autos, auto_weights) in izip!(
        explicit_vis_params.timeblocks.iter(),
        explicit_crosses.outer_iter_mut(),
        explicit_cross_weights.outer_iter_mut(),
        explicit_autos.outer_iter_mut(),
        explicit_auto_weights.outer_iter_mut()
    ) {
        explicit_vis_params
            .read_timeblock(
                timeblock,
                crosses,
                cross_weights,
                Some((autos, auto_weights)),
                &error,
            )
            .unwrap();
    }

    let implicit_vis_params = InputVisArgs {
        files: Some(vec![metafits, implicit_output.display().to_string()]),
        ..Default::default()
    }
    .parse("")
    .unwrap();
    let obs_context = implicit_vis_params.get_obs_context();
    assert_eq!(total_num_tiles, obs_context.get_total_num_tiles());
    assert_eq!(
        total_num_baselines,
        (total_num_tiles * (total_num_tiles - 1)) / 2
    );
    assert_eq!(total_num_channels, obs_context.fine_chan_freqs.len());
    let mut implicit_crosses = Array3::zeros((
        implicit_vis_params.timeblocks.len(),
        total_num_channels,
        total_num_baselines,
    ));
    let mut implicit_cross_weights = Array3::zeros((
        implicit_vis_params.timeblocks.len(),
        total_num_channels,
        total_num_baselines,
    ));
    let mut implicit_autos = Array3::zeros((
        implicit_vis_params.timeblocks.len(),
        total_num_channels,
        total_num_tiles,
    ));
    let mut implicit_auto_weights = Array3::zeros((
        implicit_vis_params.timeblocks.len(),
        total_num_channels,
        total_num_tiles,
    ));
    for (timeblock, crosses, cross_weights, autos, auto_weights) in izip!(
        implicit_vis_params.timeblocks.iter(),
        implicit_crosses.outer_iter_mut(),
        implicit_cross_weights.outer_iter_mut(),
        implicit_autos.outer_iter_mut(),
        implicit_auto_weights.outer_iter_mut()
    ) {
        implicit_vis_params
            .read_timeblock(
                timeblock,
                crosses,
                cross_weights,
                Some((autos, auto_weights)),
                &error,
            )
            .unwrap();
    }

    assert_abs_diff_eq!(explicit_crosses, implicit_crosses);
    assert_abs_diff_eq!(explicit_cross_weights, implicit_cross_weights);
    assert_abs_diff_eq!(explicit_autos, implicit_autos);
    assert_abs_diff_eq!(explicit_auto_weights, implicit_auto_weights);
}

#[test]
fn test_solutions_apply_needs_sols_but_is_otherwise_vis_convert() {
    let DataAsStrings {
        metafits,
        vis: mut files,
        mwafs: _,
        srclist: _,
    } = get_reduced_1090008640_raw();
    files.push(metafits.clone());

    let mut args = SolutionsApplyArgs {
        data_args: InputVisArgs {
            files: Some(files),
            pfb_flavour: Some("none".to_string()),
            no_digital_gains: false,
            no_cable_length_correction: false,
            no_geometric_correction: false,
            ignore_input_data_fine_channel_flags: true,
            ..Default::default()
        },
        ..Default::default()
    };

    let tmp_dir = TempDir::new().unwrap();
    let apply_output = tmp_dir.path().join("apply.uvfits");
    let sols_file = get_1090008640_identity_solutions_file(tmp_dir.path());
    args.solutions = Some(sols_file.display().to_string());
    args.outputs = Some(vec![apply_output.clone()]);
    args.clone().parse().unwrap().run().unwrap();

    let convert_output = tmp_dir.path().join("convert.uvfits");
    let args = VisConvertArgs {
        data_args: args.data_args,
        outputs: Some(vec![convert_output.clone()]),
        ..Default::default()
    };
    args.parse().unwrap().run().unwrap();

    // Seeing as the solutions are identities, the visibilities should be
    // exactly the same.
    let error = AtomicCell::new(false);
    let apply_vis_params = InputVisArgs {
        files: Some(vec![metafits.clone(), apply_output.display().to_string()]),
        ..Default::default()
    }
    .parse("")
    .unwrap();
    let obs_context = apply_vis_params.get_obs_context();
    let total_num_tiles = obs_context.get_total_num_tiles();
    let total_num_baselines = (total_num_tiles * (total_num_tiles - 1)) / 2;
    let total_num_channels = obs_context.fine_chan_freqs.len();
    let mut apply_crosses = Array3::zeros((
        apply_vis_params.timeblocks.len(),
        total_num_channels,
        total_num_baselines,
    ));
    let mut apply_cross_weights = Array3::zeros((
        apply_vis_params.timeblocks.len(),
        total_num_channels,
        total_num_baselines,
    ));
    let mut apply_autos = Array3::zeros((
        apply_vis_params.timeblocks.len(),
        total_num_channels,
        total_num_tiles,
    ));
    let mut apply_auto_weights = Array3::zeros((
        apply_vis_params.timeblocks.len(),
        total_num_channels,
        total_num_tiles,
    ));
    for (timeblock, crosses, cross_weights, autos, auto_weights) in izip!(
        apply_vis_params.timeblocks.iter(),
        apply_crosses.outer_iter_mut(),
        apply_cross_weights.outer_iter_mut(),
        apply_autos.outer_iter_mut(),
        apply_auto_weights.outer_iter_mut()
    ) {
        apply_vis_params
            .read_timeblock(
                timeblock,
                crosses,
                cross_weights,
                Some((autos, auto_weights)),
                &error,
            )
            .unwrap();
    }

    let convert_vis_params = InputVisArgs {
        files: Some(vec![metafits, convert_output.display().to_string()]),
        ..Default::default()
    }
    .parse("")
    .unwrap();
    let obs_context = convert_vis_params.get_obs_context();
    assert_eq!(total_num_tiles, obs_context.get_total_num_tiles());
    assert_eq!(
        total_num_baselines,
        (total_num_tiles * (total_num_tiles - 1)) / 2
    );
    assert_eq!(total_num_channels, obs_context.fine_chan_freqs.len());
    let mut convert_crosses = Array3::zeros((
        convert_vis_params.timeblocks.len(),
        total_num_channels,
        total_num_baselines,
    ));
    let mut convert_cross_weights = Array3::zeros((
        convert_vis_params.timeblocks.len(),
        total_num_channels,
        total_num_baselines,
    ));
    let mut convert_autos = Array3::zeros((
        convert_vis_params.timeblocks.len(),
        total_num_channels,
        total_num_tiles,
    ));
    let mut convert_auto_weights = Array3::zeros((
        convert_vis_params.timeblocks.len(),
        total_num_channels,
        total_num_tiles,
    ));
    for (timeblock, crosses, cross_weights, autos, auto_weights) in izip!(
        convert_vis_params.timeblocks.iter(),
        convert_crosses.outer_iter_mut(),
        convert_cross_weights.outer_iter_mut(),
        convert_autos.outer_iter_mut(),
        convert_auto_weights.outer_iter_mut()
    ) {
        convert_vis_params
            .read_timeblock(
                timeblock,
                crosses,
                cross_weights,
                Some((autos, auto_weights)),
                &error,
            )
            .unwrap();
    }

    assert_abs_diff_eq!(apply_crosses, convert_crosses);
    assert_abs_diff_eq!(apply_cross_weights, convert_cross_weights);
    assert_abs_diff_eq!(apply_autos, convert_autos);
    assert_abs_diff_eq!(apply_auto_weights, convert_auto_weights);
}
