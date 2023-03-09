// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Tests for solutions-apply.

use std::{collections::HashSet, path::Path};

use approx::{assert_abs_diff_eq, assert_relative_eq};
use marlu::Jones;
use mwalib::{
    _get_required_fits_key, _open_fits, _open_hdu, fits_open, fits_open_hdu, get_required_fits_key,
};
use ndarray::prelude::*;
use serial_test::serial;
use tempfile::TempDir;
use vec1::vec1;

use super::*;
use crate::{
    io::read::{pfb_gains::PfbFlavour, RawDataCorrections},
    tests::reduced_obsids::{
        get_reduced_1090008640, get_reduced_1090008640_ms, get_reduced_1090008640_uvfits,
    },
};

fn test_solutions_apply_trivial(input_data: &dyn VisRead, metafits: &str) {
    // Make some solutions that are all identity; the output visibilities should
    // be the same as the input.
    // Get the reference visibilities.
    let obs_context = input_data.get_obs_context();
    let flagged_tiles = obs_context.get_tile_flags(false, None).unwrap();
    assert!(flagged_tiles.is_empty());
    let total_num_tiles = obs_context.get_total_num_tiles();
    let total_num_baselines = (total_num_tiles * (total_num_tiles - 1)) / 2;
    let total_num_channels = obs_context.fine_chan_freqs.len();
    let tile_baseline_flags = TileBaselineFlags::new(total_num_tiles, flagged_tiles);
    let mut flagged_fine_chans = HashSet::new();
    let mut ref_crosses = Array2::zeros((total_num_channels, total_num_baselines));
    let mut ref_cross_weights = Array2::zeros((total_num_channels, total_num_baselines));
    let mut ref_autos = Array2::zeros((total_num_channels, total_num_tiles));
    let mut ref_auto_weights = Array2::zeros((total_num_channels, total_num_tiles));
    input_data
        .read_crosses_and_autos(
            ref_crosses.view_mut(),
            ref_cross_weights.view_mut(),
            ref_autos.view_mut(),
            ref_auto_weights.view_mut(),
            obs_context.unflagged_timesteps[0],
            &tile_baseline_flags,
            &flagged_fine_chans,
        )
        .unwrap();

    let mut sols = CalibrationSolutions {
        di_jones: Array3::from_elem((1, total_num_tiles, total_num_channels), Jones::identity()),
        flagged_tiles: vec![],
        flagged_chanblocks: vec![],
        ..Default::default()
    };
    let timesteps = Vec1::try_from_vec(obs_context.unflagged_timesteps.clone()).unwrap();
    let tmp_dir = TempDir::new().unwrap();
    let output = tmp_dir.path().join("test.uvfits");
    let outputs = vec1![(output.clone(), VisOutputType::Uvfits)];

    apply_solutions_inner(
        input_data,
        &sols,
        &timesteps,
        LatLngHeight::mwa(),
        Duration::default(),
        false,
        &tile_baseline_flags,
        &flagged_fine_chans,
        true,
        &outputs,
        1,
        1,
        true,
    )
    .unwrap();

    // Read the output visibilities.
    let output_data = UvfitsReader::new(&output, Some(&metafits)).unwrap();
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
            &tile_baseline_flags,
            &flagged_fine_chans,
        )
        .unwrap();

    assert_abs_diff_eq!(crosses, ref_crosses);
    assert_abs_diff_eq!(cross_weights, ref_cross_weights);
    assert_abs_diff_eq!(autos, ref_autos);
    assert_abs_diff_eq!(auto_weights, ref_auto_weights);

    // Now make the solutions all "2"; the output visibilities should be 4x the
    // input.
    sols.di_jones.mapv_inplace(|j| j * 2.0);
    apply_solutions_inner(
        input_data,
        &sols,
        &timesteps,
        LatLngHeight::mwa(),
        Duration::default(),
        false,
        &tile_baseline_flags,
        &flagged_fine_chans,
        true,
        &outputs,
        1,
        1,
        true,
    )
    .unwrap();

    // Read the output visibilities.
    let output_data = UvfitsReader::new(&output, Some(&metafits)).unwrap();
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
            &tile_baseline_flags,
            &flagged_fine_chans,
        )
        .unwrap();

    assert_abs_diff_eq!(crosses, &ref_crosses * 4.0);
    assert_abs_diff_eq!(cross_weights, ref_cross_weights);
    assert_abs_diff_eq!(autos, &ref_autos * 4.0);
    assert_abs_diff_eq!(auto_weights, ref_auto_weights);

    // Now make the solutions equal to the tile index.
    sols.di_jones
        .slice_mut(s![0, .., ..])
        .outer_iter_mut()
        .enumerate()
        .for_each(|(i_tile, mut sols)| {
            sols.fill(Jones::identity() * (i_tile + 1) as f64);
        });
    apply_solutions_inner(
        input_data,
        &sols,
        &timesteps,
        LatLngHeight::mwa(),
        Duration::default(),
        false,
        &tile_baseline_flags,
        &flagged_fine_chans,
        true,
        &outputs,
        1,
        1,
        true,
    )
    .unwrap();

    // Read the output visibilities.
    let output_data = UvfitsReader::new(&output, Some(&metafits)).unwrap();
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
            &tile_baseline_flags,
            &flagged_fine_chans,
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
    let mut flagged_tiles = tile_baseline_flags.flagged_tiles;
    flagged_tiles.insert(10);
    flagged_tiles.insert(78);
    let tile_baseline_flags = TileBaselineFlags::new(total_num_tiles, flagged_tiles);
    for f in &tile_baseline_flags.flagged_tiles {
        sols.flagged_tiles.push(*f);
    }
    // Re-generate the reference data.
    let num_unflagged_tiles = total_num_tiles - tile_baseline_flags.flagged_tiles.len();
    let num_unflagged_cross_baselines = (num_unflagged_tiles * (num_unflagged_tiles - 1)) / 2;
    let mut ref_crosses = Array2::zeros((total_num_channels, num_unflagged_cross_baselines));
    let mut ref_cross_weights = Array2::zeros((total_num_channels, num_unflagged_cross_baselines));
    let mut ref_autos = Array2::zeros((total_num_channels, num_unflagged_tiles));
    let mut ref_auto_weights = Array2::zeros((total_num_channels, num_unflagged_tiles));
    input_data
        .read_crosses_and_autos(
            ref_crosses.view_mut(),
            ref_cross_weights.view_mut(),
            ref_autos.view_mut(),
            ref_auto_weights.view_mut(),
            obs_context.unflagged_timesteps[0],
            &tile_baseline_flags,
            &flagged_fine_chans,
        )
        .unwrap();

    apply_solutions_inner(
        input_data,
        &sols,
        &timesteps,
        LatLngHeight::mwa(),
        Duration::default(),
        false,
        &tile_baseline_flags,
        &flagged_fine_chans,
        true,
        &outputs,
        1,
        1,
        true,
    )
    .unwrap();

    // Read the output visibilities.
    let output_data = UvfitsReader::new(&output, Some(&metafits)).unwrap();
    let mut crosses = Array2::zeros((total_num_channels, num_unflagged_cross_baselines));
    let mut cross_weights = Array2::zeros((total_num_channels, num_unflagged_cross_baselines));
    let mut autos = Array2::zeros((total_num_channels, num_unflagged_tiles));
    let mut auto_weights = Array2::zeros((total_num_channels, num_unflagged_tiles));
    output_data
        .read_crosses_and_autos(
            crosses.view_mut(),
            cross_weights.view_mut(),
            autos.view_mut(),
            auto_weights.view_mut(),
            0,
            &tile_baseline_flags,
            &flagged_fine_chans,
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
        assert_relative_eq!(
            baseline.mapv(Jones::<f64>::from),
            ref_baseline,
            max_relative = 1e-7
        );
    }
    for (i_tile, (tile, ref_tile)) in autos
        .axis_iter(Axis(1))
        .zip_eq(ref_autos.axis_iter(Axis(1)))
        .enumerate()
    {
        let tile_factor = tile_baseline_flags.unflagged_auto_index_to_tile_map[&i_tile];
        let ref_tile =
            ref_tile.mapv(Jones::<f64>::from) * (tile_factor + 1) as f64 * (tile_factor + 1) as f64;
        assert_relative_eq!(tile.mapv(Jones::<f64>::from), ref_tile, max_relative = 1e-7);
    }
    assert_abs_diff_eq!(cross_weights, ref_cross_weights);
    assert_abs_diff_eq!(auto_weights, ref_auto_weights);

    // Finally, flag some channels.
    flagged_fine_chans.insert(3);
    flagged_fine_chans.insert(18);
    input_data
        .read_crosses_and_autos(
            ref_crosses.view_mut(),
            ref_cross_weights.view_mut(),
            ref_autos.view_mut(),
            ref_auto_weights.view_mut(),
            obs_context.unflagged_timesteps[0],
            &tile_baseline_flags,
            // We want to read all channels, even the flagged ones.
            &HashSet::new(),
        )
        .unwrap();

    apply_solutions_inner(
        input_data,
        &sols,
        &timesteps,
        LatLngHeight::mwa(),
        Duration::default(),
        false,
        &tile_baseline_flags,
        &flagged_fine_chans,
        true,
        &outputs,
        1,
        1,
        true,
    )
    .unwrap();

    // Read the output visibilities.
    let output_data = UvfitsReader::new(&output, Some(&metafits)).unwrap();
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
            &tile_baseline_flags,
            &HashSet::new(),
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
        assert_relative_eq!(
            baseline.mapv(Jones::<f64>::from),
            ref_baseline,
            max_relative = 1e-7
        );
    }
    for (i_tile, (tile, ref_tile)) in autos
        .axis_iter(Axis(1))
        .zip_eq(ref_autos.axis_iter(Axis(1)))
        .enumerate()
    {
        let tile_factor = tile_baseline_flags.unflagged_auto_index_to_tile_map[&i_tile];
        let ref_tile =
            ref_tile.mapv(Jones::<f64>::from) * (tile_factor + 1) as f64 * (tile_factor + 1) as f64;
        assert_relative_eq!(tile.mapv(Jones::<f64>::from), ref_tile, max_relative = 1e-7);
    }
    // Manually negate the weights corresponding to our flagged channels.
    for c in flagged_fine_chans {
        ref_cross_weights
            .slice_mut(s![c, ..])
            .map_inplace(|w| *w = -w.abs());
        ref_auto_weights
            .slice_mut(s![c, ..])
            .map_inplace(|w| *w = -w.abs());
    }
    assert_abs_diff_eq!(cross_weights, ref_cross_weights);
    assert_abs_diff_eq!(auto_weights, ref_auto_weights);
}

#[test]
fn test_solutions_apply_trivial_raw() {
    let cal_args = get_reduced_1090008640(false, false);
    let mut data = cal_args.data.unwrap().into_iter();
    let metafits = data.next().unwrap();
    let gpubox = data.next().unwrap();
    let input_data = RawDataReader::new(
        &metafits,
        &[gpubox],
        None,
        RawDataCorrections {
            pfb_flavour: PfbFlavour::None,
            digital_gains: false,
            cable_length: false,
            geometric: false,
        },
    )
    .unwrap();

    test_solutions_apply_trivial(&input_data, &metafits)
}

// If all data-reading routines are working correctly, these extra tests are
// redundant. But, it gives us more confidence that things are working.

#[test]
#[serial]
fn test_solutions_apply_trivial_ms() {
    let cal_args = get_reduced_1090008640_ms();
    let mut data = cal_args.data.unwrap().into_iter();
    let metafits = data.next().unwrap();
    let ms = data.next().unwrap();
    let input_data = MsReader::new(ms, Some(&metafits), None).unwrap();

    test_solutions_apply_trivial(&input_data, &metafits)
}

#[test]
fn test_solutions_apply_trivial_uvfits() {
    let cal_args = get_reduced_1090008640_uvfits();
    let mut data = cal_args.data.unwrap().into_iter();
    let metafits = data.next().unwrap();
    let uvfits = data.next().unwrap();
    let input_data = UvfitsReader::new(uvfits, Some(&metafits)).unwrap();

    test_solutions_apply_trivial(&input_data, &metafits)
}

pub(crate) fn get_1090008640_identity_solutions_file(tmp_dir: &Path) -> PathBuf {
    let sols = CalibrationSolutions {
        di_jones: Array3::from_elem((1, 128, 32), Jones::identity()),
        ..Default::default()
    };
    let file = tmp_dir.join("sols.fits");
    sols.write_solutions_from_ext::<&Path>(&file).unwrap();
    file
}

#[test]
fn test_1090008640_solutions_apply_writes_vis_uvfits() {
    let tmp_dir = TempDir::new().expect("couldn't make tmp dir");
    let args = get_reduced_1090008640(false, false);
    let data = args.data.unwrap();
    let metafits = &data[0];
    let gpubox = &data[1];
    let solutions = get_1090008640_identity_solutions_file(tmp_dir.path());
    let out_vis_path = tmp_dir.path().join("vis.uvfits");

    #[rustfmt::skip]
    let args = SolutionsApplyArgs::parse_from([
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
    let args = get_reduced_1090008640(false, false);
    let data = args.data.unwrap();
    let metafits = &data[0];
    let gpubox = &data[1];
    let solutions = get_1090008640_identity_solutions_file(tmp_dir.path());
    let out_vis_path = tmp_dir.path().join("vis.uvfits");

    #[rustfmt::skip]
    let args = SolutionsApplyArgs::parse_from([
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
    let args = get_reduced_1090008640(false, false);
    let data = args.data.unwrap();
    let metafits = &data[0];
    let gpubox = &data[1];
    let solutions = get_1090008640_identity_solutions_file(tmp_dir.path());
    let out_vis_path = tmp_dir.path().join("vis.uvfits");

    let freq_avg_factor = 2;

    #[rustfmt::skip]
    let args = SolutionsApplyArgs::parse_from([
        "solutions-apply",
        "--data", metafits, gpubox,
        "--solutions", &format!("{}", solutions.display()),
        "--outputs", &format!("{}", out_vis_path.display()),
        "--freq-average", &format!("{freq_avg_factor}"),
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
    let args = get_reduced_1090008640(false, false);
    let data = args.data.unwrap();
    let metafits = &data[0];
    let gpubox = &data[1];
    let solutions = get_1090008640_identity_solutions_file(tmp_dir.path());
    let out_uvfits_path = tmp_dir.path().join("vis.uvfits");
    let out_ms_path = tmp_dir.path().join("vis.ms");

    #[rustfmt::skip]
    let args = SolutionsApplyArgs::parse_from([
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

    let ms_data = MsReader::new(&out_ms_path, Some(metafits), None).unwrap();

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

    let args = get_reduced_1090008640_uvfits();
    let data = args.data.unwrap();
    let metafits = &data[0];
    let uvfits = &data[1];
    let vis_out = tmp_dir.path().join("vis.uvfits");
    let vis_out_string = vis_out.display().to_string();

    #[rustfmt::skip]
    let mut args = vec![
        "solutions-apply",
        "--data", metafits, uvfits,
        "--solutions", &sols_file_string,
        "--outputs", &vis_out_string,
        "--no-progress-bars"
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

    let uncal_reader = UvfitsReader::new(uvfits, Some(metafits)).unwrap();
    let cal_reader = UvfitsReader::new(vis_out, Some(metafits)).unwrap();
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
        "--data", metafits, uvfits,
        "--solutions", &sols_file_string,
        "--outputs", &vis_out_string,
        "--no-progress-bars",
        // Deliberately ignore solution tile flags, otherwise the code will
        // write out a different number of baselines.
        "--ignore-input-solutions-tile-flags"
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

    let cal2_reader = UvfitsReader::new(vis_out, Some(metafits)).unwrap();
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
