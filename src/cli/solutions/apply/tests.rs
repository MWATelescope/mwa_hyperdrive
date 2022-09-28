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
    pfb_gains::PfbFlavour,
    tests::reduced_obsids::{
        get_reduced_1090008640, get_reduced_1090008640_ms, get_reduced_1090008640_uvfits,
    },
    vis_io::read::RawDataCorrections,
};

fn test_solutions_apply_trivial(input_data: &dyn VisRead, metafits: &str) {
    // Make some solutions that are all identity; the output visibilities should
    // be the same as the input.
    // Get the reference visibilities.
    let obs_context = input_data.get_obs_context();
    let mut tile_flags = obs_context.get_tile_flags(false, None).unwrap();
    assert!(tile_flags.is_empty());
    let total_num_tiles = obs_context.get_total_num_tiles();
    let total_num_baselines = (total_num_tiles * (total_num_tiles - 1)) / 2;
    let total_num_channels = obs_context.fine_chan_freqs.len();
    let maps = TileBaselineMaps::new(total_num_tiles, &tile_flags);
    let mut flagged_fine_chans = HashSet::new();
    let mut ref_crosses = Array2::zeros((total_num_baselines, total_num_channels));
    let mut ref_cross_weights = Array2::zeros((total_num_baselines, total_num_channels));
    let mut ref_autos = Array2::zeros((total_num_tiles, total_num_channels));
    let mut ref_auto_weights = Array2::zeros((total_num_tiles, total_num_channels));
    input_data
        .read_crosses_and_autos(
            ref_crosses.view_mut(),
            ref_cross_weights.view_mut(),
            ref_autos.view_mut(),
            ref_auto_weights.view_mut(),
            obs_context.unflagged_timesteps[0],
            &maps.tile_to_unflagged_cross_baseline_map,
            &tile_flags,
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
        LatLngHeight::new_mwa(),
        Duration::from_total_nanoseconds(0),
        false,
        &tile_flags,
        &maps.tile_to_unflagged_cross_baseline_map,
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
    let mut crosses = Array2::zeros((total_num_baselines, total_num_channels));
    let mut cross_weights = Array2::zeros((total_num_baselines, total_num_channels));
    let mut autos = Array2::zeros((total_num_tiles, total_num_channels));
    let mut auto_weights = Array2::zeros((total_num_tiles, total_num_channels));
    output_data
        .read_crosses_and_autos(
            crosses.view_mut(),
            cross_weights.view_mut(),
            autos.view_mut(),
            auto_weights.view_mut(),
            0,
            &maps.tile_to_unflagged_cross_baseline_map,
            &tile_flags,
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
        LatLngHeight::new_mwa(),
        Duration::from_total_nanoseconds(0),
        false,
        &tile_flags,
        &maps.tile_to_unflagged_cross_baseline_map,
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
            &maps.tile_to_unflagged_cross_baseline_map,
            &tile_flags,
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
        LatLngHeight::new_mwa(),
        Duration::from_total_nanoseconds(0),
        false,
        &tile_flags,
        &maps.tile_to_unflagged_cross_baseline_map,
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
            &maps.tile_to_unflagged_cross_baseline_map,
            &tile_flags,
            &flagged_fine_chans,
        )
        .unwrap();

    for (i_baseline, (baseline, ref_baseline)) in crosses
        .outer_iter()
        .zip_eq(ref_crosses.outer_iter())
        .enumerate()
    {
        let (tile1, tile2) = cross_correlation_baseline_to_tiles(total_num_tiles, i_baseline);
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
    assert_abs_diff_eq!(autos, ref_autos);
    assert_abs_diff_eq!(auto_weights, ref_auto_weights);

    // Use tile indices for solutions again, but now flag some tiles.
    tile_flags.extend_from_slice(&[10, 78]);
    let maps = TileBaselineMaps::new(total_num_tiles, &tile_flags);
    sols.flagged_tiles.extend_from_slice(&tile_flags);
    // Re-generate the reference data.
    let num_unflagged_tiles = total_num_tiles - tile_flags.len();
    let num_unflagged_cross_baselines = (num_unflagged_tiles * (num_unflagged_tiles - 1)) / 2;
    let mut ref_crosses = Array2::zeros((num_unflagged_cross_baselines, total_num_channels));
    let mut ref_cross_weights = Array2::zeros((num_unflagged_cross_baselines, total_num_channels));
    let mut ref_autos = Array2::zeros((num_unflagged_tiles, total_num_channels));
    let mut ref_auto_weights = Array2::zeros((num_unflagged_tiles, total_num_channels));
    input_data
        .read_crosses_and_autos(
            ref_crosses.view_mut(),
            ref_cross_weights.view_mut(),
            ref_autos.view_mut(),
            ref_auto_weights.view_mut(),
            obs_context.unflagged_timesteps[0],
            &maps.tile_to_unflagged_cross_baseline_map,
            &tile_flags,
            &flagged_fine_chans,
        )
        .unwrap();

    apply_solutions_inner(
        input_data,
        &sols,
        &timesteps,
        LatLngHeight::new_mwa(),
        Duration::from_total_nanoseconds(0),
        false,
        &tile_flags,
        &maps.tile_to_unflagged_cross_baseline_map,
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
    let mut crosses = Array2::zeros((num_unflagged_cross_baselines, total_num_channels));
    let mut cross_weights = Array2::zeros((num_unflagged_cross_baselines, total_num_channels));
    let mut autos = Array2::zeros((num_unflagged_tiles, total_num_channels));
    let mut auto_weights = Array2::zeros((num_unflagged_tiles, total_num_channels));
    output_data
        .read_crosses_and_autos(
            crosses.view_mut(),
            cross_weights.view_mut(),
            autos.view_mut(),
            auto_weights.view_mut(),
            0,
            &maps.tile_to_unflagged_cross_baseline_map,
            &tile_flags,
            &flagged_fine_chans,
        )
        .unwrap();

    for (i_baseline, (baseline, ref_baseline)) in crosses
        .outer_iter()
        .zip_eq(ref_crosses.outer_iter())
        .enumerate()
    {
        let (mut tile1, mut tile2) =
            cross_correlation_baseline_to_tiles(num_unflagged_tiles, i_baseline);
        // `tile1` and `tile2` are based on the number of unflagged tiles, not
        // the total number of tiles. This means they need to be adjusted by how
        // many tiles are flagged before them.
        tile1 += tile_flags.iter().take_while(|&&flag| flag <= tile1).count();
        tile2 += tile_flags.iter().take_while(|&&flag| flag <= tile2).count();

        let ref_baseline =
            ref_baseline.mapv(Jones::<f64>::from) * (tile1 + 1) as f64 * (tile2 + 1) as f64;
        assert_relative_eq!(
            baseline.mapv(Jones::<f64>::from),
            ref_baseline,
            max_relative = 1e-7
        );
    }
    for (i_tile, (tile, ref_tile)) in autos
        .outer_iter()
        .zip_eq(ref_autos.outer_iter())
        .enumerate()
    {
        // `i_tile` is based on the number of unflagged tiles, not the total
        // number of tiles. It needs to be adjusted by how many tiles are
        // flagged before it.
        let tile_factor = tile_flags
            .iter()
            .take_while(|&&flag| flag <= i_tile)
            .count();

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
            &maps.tile_to_unflagged_cross_baseline_map,
            &tile_flags,
            // We want to read all channels, even the flagged ones.
            &HashSet::new(),
        )
        .unwrap();

    apply_solutions_inner(
        input_data,
        &sols,
        &timesteps,
        LatLngHeight::new_mwa(),
        Duration::from_total_nanoseconds(0),
        false,
        &tile_flags,
        &maps.tile_to_unflagged_cross_baseline_map,
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
            &maps.tile_to_unflagged_cross_baseline_map,
            &tile_flags,
            &HashSet::new(),
        )
        .unwrap();

    for (i_baseline, (baseline, ref_baseline)) in crosses
        .outer_iter()
        .zip_eq(ref_crosses.outer_iter())
        .enumerate()
    {
        let (mut tile1, mut tile2) =
            cross_correlation_baseline_to_tiles(num_unflagged_tiles, i_baseline);
        tile1 += tile_flags.iter().take_while(|&&flag| flag <= tile1).count();
        tile2 += tile_flags.iter().take_while(|&&flag| flag <= tile2).count();

        let ref_baseline =
            ref_baseline.mapv(Jones::<f64>::from) * (tile1 + 1) as f64 * (tile2 + 1) as f64;
        assert_relative_eq!(
            baseline.mapv(Jones::<f64>::from),
            ref_baseline,
            max_relative = 1e-7
        );
    }
    for (i_tile, (tile, ref_tile)) in autos
        .outer_iter()
        .zip_eq(ref_autos.outer_iter())
        .enumerate()
    {
        let tile_factor = tile_flags
            .iter()
            .take_while(|&&flag| flag <= i_tile)
            .count();

        let ref_tile =
            ref_tile.mapv(Jones::<f64>::from) * (tile_factor + 1) as f64 * (tile_factor + 1) as f64;
        assert_relative_eq!(tile.mapv(Jones::<f64>::from), ref_tile, max_relative = 1e-7);
    }
    // Manually negate the weights corresponding to our flagged channels.
    for c in flagged_fine_chans {
        ref_cross_weights
            .slice_mut(s![.., c])
            .map_inplace(|w| *w = -w.abs());
        ref_auto_weights
            .slice_mut(s![.., c])
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

    test_solutions_apply_trivial(Box::new(input_data).deref(), &metafits)
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
    let input_data = MsReader::new(&ms, Some(&metafits)).unwrap();

    test_solutions_apply_trivial(Box::new(input_data).deref(), &metafits)
}

#[test]
fn test_solutions_apply_trivial_uvfits() {
    let cal_args = get_reduced_1090008640_uvfits();
    let mut data = cal_args.data.unwrap().into_iter();
    let metafits = data.next().unwrap();
    let uvfits = data.next().unwrap();
    let input_data = UvfitsReader::new(&uvfits, Some(&metafits)).unwrap();

    test_solutions_apply_trivial(Box::new(input_data).deref(), &metafits)
}

pub(crate) fn get_1090008640_identity_solutions_file(tmp_dir: &Path) -> PathBuf {
    let sols = CalibrationSolutions {
        di_jones: Array3::from_elem((1, 128, 32), Jones::identity()),
        ..Default::default()
    };
    let file = tmp_dir.join("sols.fits");
    sols.write_solutions_from_ext::<&Path, &Path>(&file, None)
        .unwrap();
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
