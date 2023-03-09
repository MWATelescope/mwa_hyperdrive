// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Tests for reading from raw MWA files.

use approx::{abs_diff_eq, assert_abs_diff_eq, assert_abs_diff_ne};
use itertools::Itertools;
use marlu::c32;
use ndarray::prelude::*;
use tempfile::{tempdir, TempDir};

use super::*;
use crate::{
    cli::di_calibrate::DiCalArgs,
    di_calibrate::{get_cal_vis, tests::test_1090008640_quality},
    tests::{deflate_gz_into_file, reduced_obsids::get_reduced_1090008640},
};

struct CrossData {
    data_array: Array2<Jones<f32>>,
    weights_array: Array2<f32>,
}

struct AutoData {
    data_array: Array2<Jones<f32>>,
    weights_array: Array2<f32>,
}

fn get_cross_vis(args: DiCalArgs) -> CrossData {
    let result = args.into_params();
    let params = match result {
        Ok(p) => p,
        Err(e) => panic!("{}", e),
    };

    let num_unflagged_cross_baselines = params
        .tile_baseline_flags
        .tile_to_unflagged_cross_baseline_map
        .len();
    let num_unflagged_fine_chans = params.unflagged_fine_chan_freqs.len();
    let vis_shape = (num_unflagged_fine_chans, num_unflagged_cross_baselines);
    let mut data_array = Array2::zeros(vis_shape);
    let mut weights_array = Array2::zeros(vis_shape);

    let result = params.input_data.read_crosses(
        data_array.view_mut(),
        weights_array.view_mut(),
        *params.timesteps.first(),
        &params.tile_baseline_flags,
        &params.flagged_fine_chans,
    );
    assert!(result.is_ok(), "{}", result.unwrap_err());
    result.unwrap();

    CrossData {
        data_array,
        weights_array,
    }
}

fn get_auto_vis(args: DiCalArgs) -> AutoData {
    let result = args.into_params();
    let params = match result {
        Ok(p) => p,
        Err(e) => panic!("{}", e),
    };

    let num_unflagged_tiles = params.unflagged_tile_xyzs.len();
    let num_unflagged_fine_chans = params.unflagged_fine_chan_freqs.len();
    let vis_shape = (num_unflagged_fine_chans, num_unflagged_tiles);
    let mut data_array = Array2::zeros(vis_shape);
    let mut weights_array = Array2::zeros(vis_shape);

    let result = params.input_data.read_autos(
        data_array.view_mut(),
        weights_array.view_mut(),
        *params.timesteps.first(),
        &params.tile_baseline_flags,
        &params.flagged_fine_chans,
    );
    assert!(result.is_ok(), "{}", result.unwrap_err());
    result.unwrap();

    AutoData {
        data_array,
        weights_array,
    }
}

fn get_cross_and_auto_vis(args: DiCalArgs) -> (CrossData, AutoData) {
    let result = args.into_params();
    let params = match result {
        Ok(p) => p,
        Err(e) => panic!("{}", e),
    };

    let num_unflagged_cross_baselines = params
        .tile_baseline_flags
        .tile_to_unflagged_cross_baseline_map
        .len();
    let num_unflagged_fine_chans = params.unflagged_fine_chan_freqs.len();
    let vis_shape = (num_unflagged_fine_chans, num_unflagged_cross_baselines);
    let mut cross_data = CrossData {
        data_array: Array2::zeros(vis_shape),
        weights_array: Array2::zeros(vis_shape),
    };

    let num_unflagged_tiles = params.unflagged_tile_xyzs.len();
    let vis_shape = (num_unflagged_fine_chans, num_unflagged_tiles);
    let mut auto_data = AutoData {
        data_array: Array2::zeros(vis_shape),
        weights_array: Array2::zeros(vis_shape),
    };

    let result = params.input_data.read_crosses_and_autos(
        cross_data.data_array.view_mut(),
        cross_data.weights_array.view_mut(),
        auto_data.data_array.view_mut(),
        auto_data.weights_array.view_mut(),
        *params.timesteps.first(),
        &params.tile_baseline_flags,
        &params.flagged_fine_chans,
    );
    assert!(result.is_ok(), "{}", result.unwrap_err());
    result.unwrap();

    (cross_data, auto_data)
}

#[test]
fn read_1090008640_cross_vis() {
    // Other tests check that PFB gains and digital gains are correctly applied.
    // These simple _vis tests just check that the values are right.
    let mut args = get_reduced_1090008640(false, false);
    args.pfb_flavour = Some("none".to_string());
    args.no_cable_length_correction = true;
    args.no_geometric_correction = true;
    args.no_digital_gains = true;
    args.ignore_input_data_fine_channel_flags = true;
    let CrossData {
        data_array: vis,
        weights_array: weights,
    } = get_cross_vis(args);

    assert_abs_diff_eq!(
        vis[(0, 0)],
        Jones::from([
            c32::new(1.6775006e2, -8.475e1),
            c32::new(2.1249968e1, 2.5224997e2),
            c32::new(-1.03750015e2, -3.5224997e2),
            c32::new(8.7499985e1, -1.674997e1)
        ])
    );
    assert_abs_diff_eq!(
        vis[(16, 10)],
        Jones::from([
            c32::new(4.0899994e2, -1.2324997e2),
            c32::new(5.270001e2, 7.7025006e2),
            c32::new(4.1725012e2, 7.262501e2),
            c32::new(7.0849994e2, -7.175003e1),
        ])
    );

    assert_abs_diff_eq!(weights, Array2::from_elem(weights.dim(), 8.0));
}

// Test the visibility values with corrections applied (except PFB gains).
#[test]
fn read_1090008640_cross_vis_with_corrections() {
    let mut args = get_reduced_1090008640(false, false);
    args.pfb_flavour = Some("none".to_string());
    args.ignore_input_data_fine_channel_flags = true;
    let CrossData {
        data_array: vis,
        weights_array: weights,
    } = get_cross_vis(args);

    assert_abs_diff_eq!(
        vis[(0, 0)],
        Jones::from([
            c32::new(-1.2564129e2, -1.4979609e1),
            c32::new(8.207058e1, -1.4936417e2),
            c32::new(-7.30687e1, 2.3617699e2),
            c32::new(-5.5305626e1, -2.3209404e1)
        ])
    );
    assert_abs_diff_eq!(
        vis[(16, 10)],
        Jones::from([
            c32::new(-4.1381275e1, -2.6381876e2),
            c32::new(5.220332e2, -2.6055228e2),
            c32::new(4.8540738e2, -1.9634505e2),
            c32::new(1.6101786e1, -4.4489474e2),
        ])
    );

    assert_abs_diff_eq!(weights, Array2::from_elem(weights.dim(), 8.0));
}

#[test]
fn read_1090008640_auto_vis() {
    let mut args = get_reduced_1090008640(false, false);
    args.pfb_flavour = Some("none".to_string());
    args.no_cable_length_correction = true;
    args.no_geometric_correction = true;
    args.no_digital_gains = true;
    args.ignore_input_data_fine_channel_flags = true;
    let AutoData {
        data_array: vis,
        weights_array: weights,
    } = get_auto_vis(args);

    assert_abs_diff_eq!(
        vis[(0, 0)],
        Jones::from([
            c32::new(7.955224e4, 6.400678e-7),
            c32::new(-1.10225e3, 1.9750005e2),
            c32::new(-1.10225e3, -1.9750005e2),
            c32::new(7.552825e4, -9.822998e-7)
        ])
    );
    assert_abs_diff_eq!(
        vis[(2, 0)],
        Jones::from([
            c32::new(1.0605874e5, -2.0732023e-6),
            c32::new(-1.5845e3, 1.5025009e2),
            c32::new(-1.5845e3, -1.5025009e2),
            c32::new(1.0007399e5, -1.0403619e-6)
        ])
    );
    assert_abs_diff_eq!(
        vis[(16, 0)],
        Jones::from([
            c32::new(1.593375e5, 2.8569048e-8),
            c32::new(-1.5977499e3, -6.5500046e1),
            c32::new(-1.5977499e3, 6.5500046e1),
            c32::new(1.5064273e5, -1.413011e-6)
        ])
    );
    assert_abs_diff_eq!(
        vis[(16, 10)],
        Jones::from([
            c32::new(1.5991898e5, 2.289782e-6),
            c32::new(-1.9817502e3, -2.81125e3),
            c32::new(-1.9817502e3, 2.81125e3),
            c32::new(1.8102623e5, 3.0765423e-6),
        ])
    );

    assert_abs_diff_eq!(weights, Array2::from_elem(weights.dim(), 8.0));
}

#[test]
fn read_1090008640_auto_vis_with_flags() {
    let mut args = get_reduced_1090008640(false, false);
    args.pfb_flavour = Some("none".to_string());
    args.no_cable_length_correction = true;
    args.no_geometric_correction = true;
    args.no_digital_gains = true;
    args.ignore_input_data_fine_channel_flags = true;
    args.tile_flags = Some(vec!["1".to_string(), "9".to_string()]);
    args.fine_chan_flags = Some(vec![1]);
    let AutoData {
        data_array: vis,
        weights_array: weights,
    } = get_auto_vis(args);

    // Use the same values as the test above, adjusting only the indices.
    assert_abs_diff_eq!(
        vis[(0, 0)],
        Jones::from([
            c32::new(7.955224e4, 6.400678e-7),
            c32::new(-1.10225e3, 1.9750005e2),
            c32::new(-1.10225e3, -1.9750005e2),
            c32::new(7.552825e4, -9.822998e-7)
        ])
    );
    assert_abs_diff_eq!(
        // Channel 2 -> 1
        vis[(1, 0)],
        Jones::from([
            c32::new(1.0605874e5, -2.0732023e-6),
            c32::new(-1.5845e3, 1.5025009e2),
            c32::new(-1.5845e3, -1.5025009e2),
            c32::new(1.0007399e5, -1.0403619e-6)
        ])
    );
    assert_abs_diff_eq!(
        // Channel 16 -> 15
        vis[(15, 0)],
        Jones::from([
            c32::new(1.593375e5, 2.8569048e-8),
            c32::new(-1.5977499e3, -6.5500046e1),
            c32::new(-1.5977499e3, 6.5500046e1),
            c32::new(1.5064273e5, -1.413011e-6)
        ])
    );
    assert_abs_diff_eq!(
        // Two flagged tiles before tile 10; use index 8. Channel 16 -> 15.
        vis[(15, 8)],
        Jones::from([
            c32::new(1.5991898e5, 2.289782e-6),
            c32::new(-1.9817502e3, -2.81125e3),
            c32::new(-1.9817502e3, 2.81125e3),
            c32::new(1.8102623e5, 3.0765423e-6),
        ])
    );

    assert_abs_diff_eq!(weights, Array2::from_elem(weights.dim(), 8.0));
}

#[test]
fn read_1090008640_cross_and_auto_vis() {
    let mut args = get_reduced_1090008640(false, false);
    args.pfb_flavour = Some("none".to_string());
    args.no_cable_length_correction = true;
    args.no_geometric_correction = true;
    args.no_digital_gains = true;
    args.ignore_input_data_fine_channel_flags = true;
    let (cross_data, auto_data) = get_cross_and_auto_vis(args);

    // Test values should match those used in "cross_vis" and "auto_vis" tests;
    assert_abs_diff_eq!(
        cross_data.data_array[(0, 0)],
        Jones::from([
            c32::new(1.6775006e2, -8.475e1),
            c32::new(2.1249968e1, 2.5224997e2),
            c32::new(-1.03750015e2, -3.5224997e2),
            c32::new(8.7499985e1, -1.674997e1)
        ])
    );
    assert_abs_diff_eq!(
        cross_data.data_array[(16, 10)],
        Jones::from([
            c32::new(4.0899994e2, -1.2324997e2),
            c32::new(5.270001e2, 7.7025006e2),
            c32::new(4.1725012e2, 7.262501e2),
            c32::new(7.0849994e2, -7.175003e1),
        ])
    );

    assert_abs_diff_eq!(
        auto_data.data_array[(0, 0)],
        Jones::from([
            c32::new(7.955224e4, 6.400678e-7),
            c32::new(-1.10225e3, 1.9750005e2),
            c32::new(-1.10225e3, -1.9750005e2),
            c32::new(7.552825e4, -9.822998e-7)
        ])
    );
    assert_abs_diff_eq!(
        auto_data.data_array[(16, 10)],
        Jones::from([
            c32::new(1.5991898e5, 2.289782e-6),
            c32::new(-1.9817502e3, -2.81125e3),
            c32::new(-1.9817502e3, 2.81125e3),
            c32::new(1.8102623e5, 3.0765423e-6),
        ])
    );

    assert_abs_diff_eq!(
        cross_data.weights_array,
        Array2::from_elem(cross_data.weights_array.dim(), 8.0)
    );
    assert_abs_diff_eq!(
        auto_data.weights_array,
        Array2::from_elem(auto_data.weights_array.dim(), 8.0)
    );
}

#[test]
fn pfb_empirical_gains() {
    let mut args = get_reduced_1090008640(false, false);
    args.pfb_flavour = Some("empirical".to_string());
    args.ignore_input_data_fine_channel_flags = true;
    let CrossData {
        data_array: vis_pfb,
        weights_array: weights_pfb,
    } = get_cross_vis(args);

    let mut args = get_reduced_1090008640(false, false);
    args.pfb_flavour = Some("none".to_string());
    args.ignore_input_data_fine_channel_flags = true;
    let CrossData {
        data_array: vis_no_pfb,
        weights_array: weights_no_pfb,
    } = get_cross_vis(args);

    // Test each visibility individually.
    vis_pfb
        .outer_iter()
        .zip_eq(vis_no_pfb.outer_iter())
        .zip_eq(pfb_gains::EMPIRICAL_40KHZ.iter())
        .for_each(|((vis_pfb, vis_no_pfb), &gain)| {
            vis_pfb
                .iter()
                .zip(vis_no_pfb.iter())
                .for_each(|(vis_pfb, vis_no_pfb)| {
                    // Promote the Jones matrices for better accuracy.
                    assert_abs_diff_eq!(
                        Jones::<f64>::from(vis_pfb) / Jones::<f64>::from(vis_no_pfb),
                        Jones::identity() / gain,
                        epsilon = 8e-5
                    );
                });
        });

    // Weights are definitely not the same.
    assert_abs_diff_ne!(weights_pfb, weights_no_pfb);
}

#[test]
fn pfb_levine_gains() {
    let mut args = get_reduced_1090008640(false, false);
    args.pfb_flavour = Some("levine".to_string());
    args.no_digital_gains = true;
    args.ignore_input_data_fine_channel_flags = true;
    let CrossData {
        data_array: vis_pfb,
        weights_array: weights_pfb,
    } = get_cross_vis(args);

    let mut args = get_reduced_1090008640(false, false);
    args.pfb_flavour = Some("none".to_string());
    args.no_digital_gains = true;
    args.ignore_input_data_fine_channel_flags = true;
    let CrossData {
        data_array: vis_no_pfb,
        weights_array: weights_no_pfb,
    } = get_cross_vis(args);

    // Test each visibility individually.
    vis_pfb
        .outer_iter()
        .zip_eq(vis_no_pfb.outer_iter())
        .zip(pfb_gains::LEVINE_40KHZ.iter())
        .for_each(|((vis_pfb, vis_no_pfb), &gain)| {
            vis_pfb
                .iter()
                .zip(vis_no_pfb.iter())
                .for_each(|(vis_pfb, vis_no_pfb)| {
                    // Promote the Jones matrices for better accuracy.
                    assert_abs_diff_eq!(
                        Jones::<f64>::from(vis_pfb) / Jones::<f64>::from(vis_no_pfb),
                        Jones::identity() / gain,
                        epsilon = 2e-4
                    );
                });
        });

    // Weights are definitely not the same.
    assert_abs_diff_ne!(weights_pfb, weights_no_pfb);
}

#[test]
fn test_digital_gains() {
    let mut args = get_reduced_1090008640(false, false);
    // Some("none") turns off the PFB correction, whereas None would be the
    // default behaviour (apply PFB correction).
    args.pfb_flavour = Some("none".to_string());
    args.no_digital_gains = false;
    let CrossData {
        data_array: vis_dg,
        weights_array: weights_dg,
    } = get_cross_vis(args);

    let mut args = get_reduced_1090008640(false, false);
    args.pfb_flavour = Some("none".to_string());
    args.no_digital_gains = true;
    let CrossData {
        data_array: vis_no_dg,
        weights_array: weights_no_dg,
    } = get_cross_vis(args);

    let i_bl = 0;
    // Promote the Jones matrices for better accuracy.
    let mut result: Array1<Jones<f64>> = vis_dg.slice(s![.., i_bl]).mapv(Jones::from);
    result /= &vis_no_dg.slice(s![.., i_bl]).mapv(Jones::from);
    // Baseline 0 is made from antennas 0 and 1. Both have a digital gain of 78.
    let dg: f64 = 78.0 / 64.0;
    let expected = Array1::from_elem(result.dim(), Jones::identity()) / dg / dg;
    assert_abs_diff_eq!(result, expected, epsilon = 1e-6);

    let i_bl = 103;
    let mut result: Array1<Jones<f64>> = vis_dg.slice(s![.., i_bl]).mapv(Jones::from);
    result /= &vis_no_dg.slice(s![.., i_bl]).mapv(Jones::from);
    // Baseline 103 is made from antennas 0 and 104.
    let dg1: f64 = 78.0 / 64.0;
    let dg2: f64 = 97.0 / 64.0;
    let expected = Array1::from_elem(result.dim(), Jones::identity()) / dg1 / dg2;
    assert_abs_diff_eq!(result, expected, epsilon = 1e-6);

    // Weights are definitely the same.
    assert_abs_diff_eq!(weights_dg, weights_no_dg);
}

#[test]
fn test_mwaf_flags() {
    // First test without any mwaf flags.
    let mut args = get_reduced_1090008640(false, false);
    args.ignore_input_data_fine_channel_flags = true;
    args.ignore_input_data_tile_flags = true;
    args.pfb_flavour = Some("none".to_string());
    args.no_digital_gains = false;

    let result = args.into_params();
    let params = match result {
        Ok(p) => p,
        Err(e) => panic!("{}", e),
    };
    let timesteps = params.timesteps;

    // Set up our arrays for reading.
    let num_unflagged_cross_baselines = params
        .tile_baseline_flags
        .tile_to_unflagged_cross_baseline_map
        .len();
    let num_unflagged_tiles = params.unflagged_tile_xyzs.len();
    let num_unflagged_fine_chans = params.unflagged_fine_chan_freqs.len();
    let cross_vis_shape = (num_unflagged_fine_chans, num_unflagged_cross_baselines);
    let mut cross_data_array = Array2::from_elem(cross_vis_shape, Jones::identity());
    let mut cross_weights_array = Array2::ones(cross_vis_shape);
    let auto_vis_shape = (num_unflagged_fine_chans, num_unflagged_tiles);
    let mut auto_data_array = Array2::from_elem(auto_vis_shape, Jones::identity());
    let mut auto_weights_array = Array2::ones(auto_vis_shape);

    let result = params.input_data.read_crosses_and_autos(
        cross_data_array.view_mut(),
        cross_weights_array.view_mut(),
        auto_data_array.view_mut(),
        auto_weights_array.view_mut(),
        *timesteps.first(),
        &params.tile_baseline_flags,
        &params.flagged_fine_chans,
    );
    assert!(result.is_ok(), "{}", result.unwrap_err());
    result.unwrap();

    // Now use the flags from our doctored mwaf file.
    let mut args = get_reduced_1090008640(false, true);
    args.ignore_input_data_fine_channel_flags = true;
    args.ignore_input_data_tile_flags = true;
    args.pfb_flavour = Some("none".to_string());
    args.no_digital_gains = false;

    let result = args.into_params();
    let params = match result {
        Ok(p) => p,
        Err(e) => panic!("{}", e),
    };

    let mut flagged_cross_data_array = Array2::from_elem(cross_vis_shape, Jones::identity());
    let mut flagged_cross_weights_array = Array2::ones(cross_vis_shape);
    let mut flagged_auto_data_array = Array2::from_elem(auto_vis_shape, Jones::identity());
    let mut flagged_auto_weights_array = Array2::ones(auto_vis_shape);

    let result = params.input_data.read_crosses_and_autos(
        flagged_cross_data_array.view_mut(),
        flagged_cross_weights_array.view_mut(),
        flagged_auto_data_array.view_mut(),
        flagged_auto_weights_array.view_mut(),
        *timesteps.first(),
        &params.tile_baseline_flags,
        &params.flagged_fine_chans,
    );
    assert!(result.is_ok(), "{}", result.unwrap_err());
    result.unwrap();

    // Cross-correlation weights are different because one of the visibilities
    // is flagged.
    assert_ne!(cross_weights_array, flagged_cross_weights_array);
    // No autos are flagged, though.
    assert_eq!(auto_weights_array, flagged_auto_weights_array);

    // Iterate over the weight arrays, checking for flags.
    let num_tiles = params.get_total_num_tiles();
    let num_bls = (num_tiles * (num_tiles + 1)) / 2;
    let num_freqs = params.get_obs_context().fine_chan_freqs.len();
    // Unfortunately we have to conditionally select either the auto or cross
    // visibilities.
    for i_chan in 0..num_freqs {
        let mut i_auto = 0;
        let mut i_cross = 0;
        for i_bl in 0..num_bls {
            let (tile1, tile2) =
                marlu::math::baseline_to_tiles(params.unflagged_tile_xyzs.len(), i_bl);

            let weight = if tile1 == tile2 {
                i_auto += 1;
                flagged_auto_weights_array[(i_chan, i_auto - 1)]
            } else {
                i_cross += 1;
                flagged_cross_weights_array[(i_chan, i_cross - 1)]
            };
            let expected = if tile1 == 0 && tile2 == 12 && i_chan == 2 {
                -8.0
            } else {
                8.0
            };
            assert!(
                abs_diff_eq!(weight, expected), "weight = {weight}, expected = {expected}, i_chan = {i_chan}, i_bl = {i_bl}, tile1 = {tile1}, tile2 = {tile2}"
            );
        }
    }
}

#[test]
fn test_mwaf_flags_primes() {
    // First test without any mwaf flags.
    let mut args = get_reduced_1090008640(false, false);
    args.ignore_input_data_fine_channel_flags = true;
    args.ignore_input_data_tile_flags = true;
    args.pfb_flavour = Some("none".to_string());
    args.no_digital_gains = false;

    let result = args.into_params();
    let params = match result {
        Ok(p) => p,
        Err(e) => panic!("{}", e),
    };
    let timesteps = params.timesteps;

    // Set up our arrays for reading.
    let num_unflagged_cross_baselines = params
        .tile_baseline_flags
        .tile_to_unflagged_cross_baseline_map
        .len();
    let num_unflagged_tiles = params.unflagged_tile_xyzs.len();
    let num_unflagged_fine_chans = params.unflagged_fine_chan_freqs.len();
    let cross_vis_shape = (num_unflagged_fine_chans, num_unflagged_cross_baselines);
    let mut cross_data_array = Array2::from_elem(cross_vis_shape, Jones::identity());
    let mut cross_weights_array = Array2::ones(cross_vis_shape);
    let auto_vis_shape = (num_unflagged_fine_chans, num_unflagged_tiles);
    let mut auto_data_array = Array2::from_elem(auto_vis_shape, Jones::identity());
    let mut auto_weights_array = Array2::ones(auto_vis_shape);

    let result = params.input_data.read_crosses_and_autos(
        cross_data_array.view_mut(),
        cross_weights_array.view_mut(),
        auto_data_array.view_mut(),
        auto_weights_array.view_mut(),
        *timesteps.first(),
        &params.tile_baseline_flags,
        &params.flagged_fine_chans,
    );
    assert!(result.is_ok(), "{}", result.unwrap_err());
    result.unwrap();

    // Now use the flags from our "primes" mwaf file.
    let mut args = get_reduced_1090008640(false, false);
    args.ignore_input_data_fine_channel_flags = true;
    args.ignore_input_data_tile_flags = true;
    args.pfb_flavour = Some("none".to_string());
    args.no_digital_gains = false;
    let temp_dir = TempDir::new().unwrap();
    let mwaf_pb = temp_dir.path().join("primes.mwaf");
    let mut mwaf_file = std::fs::File::create(&mwaf_pb).unwrap();
    deflate_gz_into_file("test_files/1090008640/primes_01.mwaf.gz", &mut mwaf_file);
    match &mut args.data {
        Some(d) => d.push(mwaf_pb.display().to_string()),
        None => unreachable!(),
    }

    let result = args.into_params();
    let params = match result {
        Ok(p) => p,
        Err(e) => panic!("{}", e),
    };

    let mut flagged_cross_data_array = Array2::from_elem(cross_vis_shape, Jones::identity());
    let mut flagged_cross_weights_array = Array2::ones(cross_vis_shape);
    let mut flagged_auto_data_array = Array2::from_elem(auto_vis_shape, Jones::identity());
    let mut flagged_auto_weights_array = Array2::ones(auto_vis_shape);

    let result = params.input_data.read_crosses_and_autos(
        flagged_cross_data_array.view_mut(),
        flagged_cross_weights_array.view_mut(),
        flagged_auto_data_array.view_mut(),
        flagged_auto_weights_array.view_mut(),
        *timesteps.first(),
        &params.tile_baseline_flags,
        &params.flagged_fine_chans,
    );
    assert!(result.is_ok(), "{}", result.unwrap_err());
    result.unwrap();

    // There's a difference -- mwaf flags applied.
    assert_ne!(cross_weights_array, flagged_cross_weights_array);
    assert_ne!(auto_weights_array, flagged_auto_weights_array);

    // Iterate over the arrays, where are the differences? They should be
    // primes.
    let num_tiles = params.get_total_num_tiles();
    let num_bls = (num_tiles * (num_tiles + 1)) / 2;
    let num_freqs = params.get_obs_context().fine_chan_freqs.len();
    // Unfortunately we have to conditionally select either the auto or cross
    // visibilities.
    for i_chan in 0..num_freqs {
        let mut i_auto = 0;
        let mut i_cross = 0;
        for i_bl in 0..num_bls {
            // This mwaf file was created with baselines moving slower than
            // frequencies.
            let is_prime = crate::math::is_prime(i_bl * num_freqs + i_chan);
            let (tile1, tile2) =
                marlu::math::baseline_to_tiles(params.unflagged_tile_xyzs.len(), i_bl);

            let weight = if tile1 == tile2 {
                i_auto += 1;
                flagged_auto_weights_array[(i_chan, i_auto - 1)]
            } else {
                i_cross += 1;
                flagged_cross_weights_array[(i_chan, i_cross - 1)]
            };
            let expected = if is_prime { -8.0 } else { 8.0 };
            assert!(
                abs_diff_eq!(weight, expected), "weight = {weight}, expected = {expected}, i_chan = {i_chan}, i_bl = {i_bl}, tile1 = {tile1}, tile2 = {tile2}"
            );
        }
    }
}

/// Test that cotter flags are correctly (as possible) ingested.
#[test]
fn test_mwaf_flags_cotter() {
    let mut args = get_reduced_1090008640(false, false);
    args.ignore_input_data_fine_channel_flags = true;
    args.ignore_input_data_tile_flags = true;
    args.pfb_flavour = Some("none".to_string());
    args.no_digital_gains = false;

    let temp_dir = TempDir::new().unwrap();
    let mwaf_pb = temp_dir.path().join("cotter.mwaf");
    let mut mwaf_file = std::fs::File::create(&mwaf_pb).unwrap();
    deflate_gz_into_file(
        "test_files/1090008640/1090008640_01_cotter.mwaf.gz",
        &mut mwaf_file,
    );
    match &mut args.data {
        Some(d) => d.push(mwaf_pb.display().to_string()),
        None => unreachable!(),
    }

    let result = args.clone().into_params();
    let params = match result {
        Ok(p) => p,
        Err(e) => panic!("{}", e),
    };
    let timesteps = &params.timesteps;

    // Set up our arrays for reading.
    let num_unflagged_cross_baselines = params
        .tile_baseline_flags
        .tile_to_unflagged_cross_baseline_map
        .len();
    let num_unflagged_tiles = params.unflagged_tile_xyzs.len();
    let num_unflagged_fine_chans = params.unflagged_fine_chan_freqs.len();
    let cross_vis_shape = (num_unflagged_fine_chans, num_unflagged_cross_baselines);
    let mut cross_data_array = Array2::from_elem(cross_vis_shape, Jones::identity());
    let mut cross_weights_array = Array2::ones(cross_vis_shape);
    let auto_vis_shape = (num_unflagged_fine_chans, num_unflagged_tiles);
    let mut auto_data_array = Array2::from_elem(auto_vis_shape, Jones::identity());
    let mut auto_weights_array = Array2::ones(auto_vis_shape);

    let result = params.input_data.read_crosses_and_autos(
        cross_data_array.view_mut(),
        cross_weights_array.view_mut(),
        auto_data_array.view_mut(),
        auto_weights_array.view_mut(),
        *timesteps.first(),
        &params.tile_baseline_flags,
        &params.flagged_fine_chans,
    );
    assert!(result.is_ok(), "{}", result.unwrap_err());
    result.unwrap();

    // Iterate over the weight arrays.
    let num_tiles = params.get_total_num_tiles();
    let num_bls = (num_tiles * (num_tiles + 1)) / 2;
    let num_freqs = params.get_obs_context().fine_chan_freqs.len();
    // Unfortunately we have to conditionally select either the auto or cross
    // visibilities.
    for i_chan in 0..num_freqs {
        let mut i_auto = 0;
        let mut i_cross = 0;
        for i_bl in 0..num_bls {
            let (tile1, tile2) =
                marlu::math::baseline_to_tiles(params.unflagged_tile_xyzs.len(), i_bl);

            let weight = if tile1 == tile2 {
                i_auto += 1;
                auto_weights_array[(i_chan, i_auto - 1)]
            } else {
                i_cross += 1;
                cross_weights_array[(i_chan, i_cross - 1)]
            };
            let expected = if tile1 == 0 && tile2 == 12 && i_chan == 2 {
                -8.0
            } else {
                8.0
            };
            assert!(
                abs_diff_eq!(weight, expected), "weight = {weight}, expected = {expected}, i_chan = {i_chan}, i_bl = {i_bl}, tile1 = {tile1}, tile2 = {tile2}"
            );
        }
    }

    // Do it all again, but this time with the forward offset flags.
    let mut mwaf_file = std::fs::File::create(&mwaf_pb).unwrap();
    deflate_gz_into_file(
        "test_files/1090008640/1090008640_01_cotter_offset_forwards.mwaf.gz",
        &mut mwaf_file,
    );
    match &mut args.data {
        Some(d) => {
            d.pop();
            d.push(mwaf_pb.display().to_string())
        }
        None => unreachable!(),
    }

    let result = args.clone().into_params();
    let params = match result {
        Ok(p) => p,
        Err(e) => panic!("{}", e),
    };
    let timesteps = &params.timesteps;

    let result = params.input_data.read_crosses_and_autos(
        cross_data_array.view_mut(),
        cross_weights_array.view_mut(),
        auto_data_array.view_mut(),
        auto_weights_array.view_mut(),
        *timesteps.first(),
        &params.tile_baseline_flags,
        &params.flagged_fine_chans,
    );
    assert!(result.is_ok(), "{}", result.unwrap_err());
    result.unwrap();

    for i_chan in 0..num_freqs {
        let mut i_auto = 0;
        let mut i_cross = 0;
        for i_bl in 0..num_bls {
            let (tile1, tile2) =
                marlu::math::baseline_to_tiles(params.unflagged_tile_xyzs.len(), i_bl);

            let weight = if tile1 == tile2 {
                i_auto += 1;
                auto_weights_array[(i_chan, i_auto - 1)]
            } else {
                i_cross += 1;
                cross_weights_array[(i_chan, i_cross - 1)]
            };
            let expected = if tile1 == 0 && tile2 == 12 && i_chan == 2 {
                -8.0
            } else {
                8.0
            };
            assert!(
                abs_diff_eq!(weight, expected), "weight = {weight}, expected = {expected}, i_chan = {i_chan}, i_bl = {i_bl}, tile1 = {tile1}, tile2 = {tile2}"
            );
        }
    }

    // Finally the backward offset flags.
    let mut mwaf_file = std::fs::File::create(&mwaf_pb).unwrap();
    deflate_gz_into_file(
        "test_files/1090008640/1090008640_01_cotter_offset_backwards.mwaf.gz",
        &mut mwaf_file,
    );
    match &mut args.data {
        Some(d) => {
            d.pop();
            d.push(mwaf_pb.display().to_string())
        }
        None => unreachable!(),
    }

    let result = args.clone().into_params();
    let params = match result {
        Ok(p) => p,
        Err(e) => panic!("{}", e),
    };
    let timesteps = &params.timesteps;

    let result = params.input_data.read_crosses_and_autos(
        cross_data_array.view_mut(),
        cross_weights_array.view_mut(),
        auto_data_array.view_mut(),
        auto_weights_array.view_mut(),
        *timesteps.first(),
        &params.tile_baseline_flags,
        &params.flagged_fine_chans,
    );
    assert!(result.is_ok(), "{}", result.unwrap_err());
    result.unwrap();

    for i_chan in 0..num_freqs {
        let mut i_auto = 0;
        let mut i_cross = 0;
        for i_bl in 0..num_bls {
            let (tile1, tile2) =
                marlu::math::baseline_to_tiles(params.unflagged_tile_xyzs.len(), i_bl);

            let weight = if tile1 == tile2 {
                i_auto += 1;
                auto_weights_array[(i_chan, i_auto - 1)]
            } else {
                i_cross += 1;
                cross_weights_array[(i_chan, i_cross - 1)]
            };
            let expected = if tile1 == 0 && tile2 == 12 && i_chan == 2 {
                -8.0
            } else {
                8.0
            };
            assert!(
                abs_diff_eq!(weight, expected), "weight = {weight}, expected = {expected}, i_chan = {i_chan}, i_bl = {i_bl}, tile1 = {tile1}, tile2 = {tile2}"
            );
        }
    }
}

#[test]
fn test_default_flags_per_coarse_chan() {
    assert_eq!(
        get_80khz_fine_chan_flags_per_coarse_chan(10000, 128, true),
        &[0, 1, 2, 3, 4, 5, 6, 7, 120, 121, 122, 123, 124, 125, 126, 127]
    );
    assert_eq!(
        get_80khz_fine_chan_flags_per_coarse_chan(10000, 128, false),
        &[0, 1, 2, 3, 4, 5, 6, 7, 64, 120, 121, 122, 123, 124, 125, 126, 127]
    );

    assert_eq!(
        get_80khz_fine_chan_flags_per_coarse_chan(20000, 64, true),
        &[0, 1, 2, 3, 60, 61, 62, 63]
    );
    assert_eq!(
        get_80khz_fine_chan_flags_per_coarse_chan(20000, 64, false),
        &[0, 1, 2, 3, 32, 60, 61, 62, 63]
    );

    assert_eq!(
        get_80khz_fine_chan_flags_per_coarse_chan(40000, 32, true),
        &[0, 1, 30, 31]
    );
    assert_eq!(
        get_80khz_fine_chan_flags_per_coarse_chan(40000, 32, false),
        &[0, 1, 16, 30, 31]
    );

    // Future proofing?
    assert_eq!(
        get_80khz_fine_chan_flags_per_coarse_chan(7200, 100, true),
        &[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99]
    );
    assert_eq!(
        get_80khz_fine_chan_flags_per_coarse_chan(7200, 100, false),
        &[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 50, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99]
    );
}

#[test]
fn test_1090008640_calibration_quality() {
    let mut args = get_reduced_1090008640(false, false);
    let temp_dir = tempdir().expect("Couldn't make temp dir");
    args.outputs = Some(vec![temp_dir.path().join("hyp_sols.fits")]);
    args.pfb_flavour = Some("none".to_string());
    // To be consistent with other data quality tests, add these flags.
    args.fine_chan_flags = Some(vec![0, 1, 2, 16, 30, 31]);

    let result = args.into_params();
    let params = match result {
        Ok(r) => r,
        Err(e) => panic!("{}", e),
    };

    let cal_vis = get_cal_vis(&params, false).expect("Couldn't read data and generate a model");
    test_1090008640_quality(params, cal_vis);
}
