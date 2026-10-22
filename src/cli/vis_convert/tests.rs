// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

use std::{
    num::NonZeroU16,
    path::{Path, PathBuf},
};

use approx::assert_abs_diff_eq;
use clap::Parser;
use ndarray::prelude::*;
use serial_test::serial;
use tempfile::TempDir;

use super::VisConvertArgs;
use crate::{
    cli::{common::InputVisArgs, vis_simulate::VisSimulateArgs},
    io::read::VisRead,
    params::VisConvertParams,
    tests::{get_reduced_1061316544_uvfits, get_reduced_1090008640_raw, DataAsStrings},
    MsReader, UvfitsReader,
};

#[test]
fn test_dry_run_returns_early() {
    use crate::io::read::UvfitsReader;
    // Verify dry-run doesn't write, and real run produces data matching direct read
    let temp_dir = TempDir::new().expect("couldn't make tmp dir");
    let uvfits_converted = temp_dir.path().join("converted.uvfits");
    let DataAsStrings { vis, .. } = get_reduced_1061316544_uvfits();

    let uvfits_converted_string = uvfits_converted.display().to_string();
    #[rustfmt::skip]
    let args = vec![
        "vis-convert",
        "--data", &vis[0],
        "--outputs", &uvfits_converted_string,
        "--autos",
    ];
    let vis_convert_args = VisConvertArgs::parse_from(args);
    // Dry-run should not create the file
    assert!(vis_convert_args.clone().run(true).is_ok());
    assert!(!uvfits_converted.exists());

    // Run for real and ensure output equals a direct read of input
    assert!(vis_convert_args.run(false).is_ok());
    assert!(uvfits_converted.exists());

    let input_reader = UvfitsReader::new(PathBuf::from(&vis[0]), None, None).unwrap();
    let output_reader = UvfitsReader::new(uvfits_converted.clone(), None, None).unwrap();

    // Compare a single timestep crosses/autos and weights exactly
    let obs_in = input_reader.get_obs_context();
    let ntiles = obs_in.get_total_num_tiles();
    let nbl = (ntiles * (ntiles - 1)) / 2;
    let nch = obs_in.fine_chan_freqs.len();
    let mut in_cross = Array2::zeros((nch, nbl));
    let mut in_cross_w = Array2::zeros((nch, nbl));
    let mut in_auto = Array2::zeros((nch, ntiles));
    let mut in_auto_w = Array2::zeros((nch, ntiles));
    input_reader
        .read_crosses_and_autos(
            in_cross.view_mut(),
            in_cross_w.view_mut(),
            in_auto.view_mut(),
            in_auto_w.view_mut(),
            0,
            &crate::math::TileBaselineFlags::new(ntiles, std::collections::HashSet::new()),
            &std::collections::HashSet::new(),
        )
        .unwrap();

    let mut out_cross = Array2::zeros((nch, nbl));
    let mut out_cross_w = Array2::zeros((nch, nbl));
    let mut out_auto = Array2::zeros((nch, ntiles));
    let mut out_auto_w = Array2::zeros((nch, ntiles));
    output_reader
        .read_crosses_and_autos(
            out_cross.view_mut(),
            out_cross_w.view_mut(),
            out_auto.view_mut(),
            out_auto_w.view_mut(),
            0,
            &crate::math::TileBaselineFlags::new(ntiles, std::collections::HashSet::new()),
            &std::collections::HashSet::new(),
        )
        .unwrap();

    approx::assert_abs_diff_eq!(in_cross, out_cross);
    approx::assert_abs_diff_eq!(in_cross_w, out_cross_w);
    approx::assert_abs_diff_eq!(in_auto, out_auto);
    approx::assert_abs_diff_eq!(in_auto_w, out_auto_w);
}

#[test]
fn test_per_coarse_chan_flags_and_smallest_contiguous_band_writing() {
    let temp_dir = TempDir::new().expect("couldn't make tmp dir");
    let uvfits_converted = temp_dir.path().join("converted.uvfits");
    let ms_converted = temp_dir.path().join("converted.ms");

    fn get_data_object(
        output: &Path,
        per_coarse_chan_flags: Option<Vec<String>>,
        output_smallest_contiguous_band: bool,
    ) -> Box<dyn VisRead> {
        let DataAsStrings {
            metafits,
            vis,
            mwafs: _,
            srclist: _,
        } = get_reduced_1090008640_raw();
        let metafits_pb = PathBuf::from(&metafits);

        let output_string = output.display().to_string();
        #[rustfmt::skip]
        let mut args = vec![
            "vis-convert",
            "--data", &vis[0], &metafits,
            "--outputs", &output_string
        ];
        if output_smallest_contiguous_band {
            args.push("--output-smallest-contiguous-band");
        }
        if let Some(per_coarse_chan_flags) = per_coarse_chan_flags.as_ref() {
            args.push("--fine-chan-flags-per-coarse-chan");
            for f in per_coarse_chan_flags {
                args.push(f.as_str());
            }
        }
        let vis_convert_args = VisConvertArgs::parse_from(args);
        vis_convert_args.run(false).unwrap();

        match output.extension().and_then(|os_str| os_str.to_str()) {
            Some("uvfits") => {
                Box::new(UvfitsReader::new(output.to_path_buf(), Some(&metafits_pb), None).unwrap())
            }
            Some("ms") => Box::new(
                MsReader::new(output.to_path_buf(), None, Some(&metafits_pb), None).unwrap(),
            ),
            _ => unreachable!(),
        }
    }

    for output_smallest_contiguous_band in [false, true] {
        for output in [&uvfits_converted, &ms_converted] {
            let data = get_data_object(output, None, output_smallest_contiguous_band);
            let obs_context = data.get_obs_context();
            if output_smallest_contiguous_band {
                assert_eq!(obs_context.fine_chan_freqs.len(), 28);
            } else {
                assert_eq!(obs_context.fine_chan_freqs.len(), 32);
            };
            assert_eq!(
                obs_context.num_fine_chans_per_coarse_chan,
                Some(NonZeroU16::new(32).unwrap())
            );
            assert_eq!(
                obs_context.mwa_coarse_chan_nums.as_deref(),
                Some([154].as_slice())
            );
            match output.extension().and_then(|os_str| os_str.to_str()) {
                Some("uvfits") => {
                    // uvfits currently doesn't try to determine this.
                    assert_eq!(obs_context.flagged_fine_chans_per_coarse_chan, None);
                }
                Some("ms") => {
                    assert_eq!(
                        obs_context.flagged_fine_chans_per_coarse_chan.as_deref(),
                        if output_smallest_contiguous_band {
                            // The MS reader can't tell if the edge channels are
                            // flagged or not; they're not available.
                            Some([16].as_slice())
                        } else {
                            Some([0, 1, 16, 30, 31].as_slice())
                        }
                    );
                }
                _ => unreachable!(),
            }
        }

        // Now test with additional per-coarse-chan flags.
        for output in [&uvfits_converted, &ms_converted] {
            let data = get_data_object(
                output,
                Some(vec!["2".to_string(), "5".to_string()]),
                output_smallest_contiguous_band,
            );
            let obs_context = data.get_obs_context();
            if output_smallest_contiguous_band {
                assert_eq!(obs_context.fine_chan_freqs.len(), 27);
            } else {
                assert_eq!(obs_context.fine_chan_freqs.len(), 32);
            };
            assert_eq!(
                obs_context.num_fine_chans_per_coarse_chan,
                Some(NonZeroU16::new(32).unwrap())
            );
            assert_eq!(
                obs_context.mwa_coarse_chan_nums.as_deref(),
                Some([154].as_slice())
            );
            match output.extension().and_then(|os_str| os_str.to_str()) {
                Some("uvfits") => {
                    // uvfits currently doesn't try to determine this.
                    assert_eq!(
                        obs_context.flagged_fine_chans_per_coarse_chan.as_deref(),
                        None
                    );
                }
                Some("ms") => {
                    assert_eq!(
                        obs_context.flagged_fine_chans_per_coarse_chan.as_deref(),
                        if output_smallest_contiguous_band {
                            // The MS reader can't tell if the edge channels are
                            // flagged or not; they're not available.
                            Some([5, 16].as_slice())
                        } else {
                            Some([0, 1, 2, 5, 16, 30, 31].as_slice())
                        }
                    );
                }
                _ => unreachable!(),
            }
        }
    }
}

#[test]
fn test_averaging_flags() {
    let temp_dir = TempDir::new().expect("couldn't make tmp dir");
    let uvfits_converted = temp_dir.path().join("converted.uvfits");
    let DataAsStrings { vis, .. } = get_reduced_1061316544_uvfits();
    // let vis = PathBuf::from(&vis[0]);
    let uvfits_converted_string = uvfits_converted.display().to_string();
    #[rustfmt::skip]
    let args = vec![
        "vis-convert",
        "--data", &vis[0],
        "--outputs", &uvfits_converted_string,
        "--freq-average", "80kHz",
        "--ignore-input-data-fine-channel-flags",
        "--autos",
    ];

    let vis_convert_args = VisConvertArgs::parse_from(args);

    let vis_convert_params = vis_convert_args.parse().unwrap();
    vis_convert_params.run().unwrap();
    let VisConvertParams {
        input_vis_params, ..
    } = vis_convert_params;

    let uvreader = UvfitsReader::new(uvfits_converted, None, None).unwrap();
    // let obs_context = uvreader.get_obs_context();
    let num_unflagged_tiles = input_vis_params.get_num_unflagged_tiles();
    let num_unflagged_cross_baselines = (num_unflagged_tiles * (num_unflagged_tiles - 1)) / 2;
    let num_fine_channels = input_vis_params.spw.chanblocks.len();
    let flagged_channels = input_vis_params.spw.flagged_chan_indices;
    let cross_vis_shape = (num_fine_channels, num_unflagged_cross_baselines);
    let auto_vis_shape = (num_fine_channels, num_unflagged_tiles);

    let mut cross_data_fb = Array2::zeros(cross_vis_shape);
    let mut cross_weights_fb = Array2::zeros(cross_vis_shape);
    let mut auto_data_fb = Array2::zeros(auto_vis_shape);
    let mut auto_weights_fb = Array2::zeros(auto_vis_shape);
    let timestep = 0;
    uvreader
        .read_crosses_and_autos(
            cross_data_fb.view_mut(),
            cross_weights_fb.view_mut(),
            auto_data_fb.view_mut(),
            auto_weights_fb.view_mut(),
            timestep,
            &input_vis_params.tile_baseline_flags,
            &flagged_channels,
        )
        .unwrap();
    auto_data_fb.indexed_iter().for_each(|((chan, tile), val)| {
        let j = val;
        if tile == 76 {
            return;
        }
        assert!(
            j[0].re > 0.0,
            "auto_data_fb[{}, {}][0].re = {}",
            chan,
            tile,
            j[0].re
        );
    });
}

/// Channels with a non-integer width (in Hz) are written with the right
/// frequencies, including when some channels are flagged. Previously, the
/// channel width was rounded to an integer, so written frequencies drifted
/// from the real ones, and writing could panic.
///
/// Adapted from Alec Thomson's hyperdrive fork
/// (<https://github.com/AlecThomson/mwa_hyperdrive>).
#[test]
#[serial] // casacore's table cache is not thread-safe
fn test_non_integer_channel_width() {
    let temp_dir = TempDir::new().expect("couldn't make tmp dir");
    let DataAsStrings {
        metafits, srclist, ..
    } = get_reduced_1090008640_raw();
    let model = temp_dir.path().join("model.ms");
    let model_string = model.display().to_string();
    let num_chans = 300;
    let freq_res_khz = 5.42535;
    #[rustfmt::skip]
    VisSimulateArgs::parse_from([
        "vis-simulate",
        "--metafits", &metafits,
        "--source-list", &srclist,
        "--output-model-files", &model_string,
        "--num-timesteps", "1",
        "--num-fine-channels", &format!("{num_chans}"),
        "--freq-res", &format!("{freq_res_khz}"),
        "--no-beam",
    ])
    .run(false)
    .unwrap();
    let in_freqs = MsReader::new(model.clone(), None, Some(Path::new(&metafits)), None)
        .unwrap()
        .get_obs_context()
        .fine_chan_freqs
        .clone();

    for (flags, freq_average) in [(vec!["0", "1", "100", "101"], "1"), (vec!["7"], "4")] {
        let out = temp_dir.path().join(format!("out_{freq_average}.ms"));
        let out_string = out.display().to_string();
        let mut args = vec![
            "vis-convert",
            "--data",
            &model_string,
            &metafits,
            "--freq-average",
            freq_average,
            "--outputs",
            &out_string,
            "--fine-chan-flags",
        ];
        args.extend(flags.iter().copied());
        VisConvertArgs::parse_from(args).run(false).unwrap();

        let out_ctx = MsReader::new(out, None, Some(Path::new(&metafits)), None).unwrap();
        let out_ctx = out_ctx.get_obs_context();
        let freq_average: usize = freq_average.parse().unwrap();
        let freq_res = freq_res_khz * 1e3 * freq_average as f64;
        assert_abs_diff_eq!(out_ctx.freq_res.unwrap(), freq_res, epsilon = 1e-6);
        // Flagged channels are written (as flagged), so the whole band is
        // there.
        assert_eq!(out_ctx.fine_chan_freqs.len(), num_chans / freq_average);
        for (i, &out_freq) in out_ctx.fine_chan_freqs.iter().enumerate() {
            let expected = in_freqs[0] as f64
                + i as f64 * freq_res
                + (freq_average - 1) as f64 / 2.0 * freq_res_khz * 1e3;
            // Frequencies are stored in integer Hz.
            assert_abs_diff_eq!(out_freq as f64, expected, epsilon = 1.0);
        }
    }
}

/// Convert a non-MWA measurement set with a 5425.35 Hz channel width
/// (`test_files/mini_ms/mini.ms`) to uvfits and to a measurement set, with
/// flags, with frequency averaging, and writing only the smallest contiguous
/// band of unflagged chanblocks. The written channel grid must match the
/// input one.
#[test]
#[serial] // casacore's table cache is not thread-safe
fn test_vis_convert_fractional_chan_width() {
    let temp_dir = TempDir::new().expect("couldn't make tmp dir");
    let ms = "test_files/mini_ms/mini.ms";
    let width = 5425.35;
    let num_chans = 32;

    // (averaging factor, flagged channels, output type, smallest band only,
    // expected first chanblock index and number of chanblocks)
    #[rustfmt::skip]
    let cases = [
        (1, vec![], "uvfits", false, 0, 32),
        (1, vec!["0", "1", "10"], "ms", false, 0, 32),
        (4, vec!["7"], "ms", false, 0, 8),
        (8, vec!["0", "1", "2", "3", "4", "5", "6", "7"], "uvfits", false, 0, 4),
        // Flagged channels at the band edges are dropped, those within it are
        // written as flagged.
        (1, vec!["0", "1", "10", "30", "31"], "ms", true, 2, 28),
        (1, vec!["0", "1", "10", "30", "31"], "uvfits", true, 2, 28),
        // Chanblock 0 is entirely flagged, chanblock 7 is not.
        (4, vec!["0", "1", "2", "3", "28", "29"], "ms", true, 1, 7),
    ];
    for (i_case, (freq_average, flags, ext, smallest_band, first_chanblock, num_chanblocks)) in
        cases.into_iter().enumerate()
    {
        let out = temp_dir.path().join(format!("out_{i_case}.{ext}"));
        let out_string = out.display().to_string();
        let freq_average_string = freq_average.to_string();
        #[rustfmt::skip]
        let mut args = vec![
            "vis-convert",
            "--data", ms,
            "--freq-average", &freq_average_string,
            "--outputs", &out_string,
        ];
        if smallest_band {
            args.push("--output-smallest-contiguous-band");
        }
        if !flags.is_empty() {
            args.push("--fine-chan-flags");
            args.extend(flags.iter().copied());
        }
        VisConvertArgs::parse_from(args).run(false).unwrap();

        let (out_freqs, out_freq_res) = if ext == "ms" {
            let reader = MsReader::new(out, None, None, None).unwrap();
            let obs_context = reader.get_obs_context();
            (
                obs_context.fine_chan_freqs.clone(),
                obs_context.freq_res.unwrap(),
            )
        } else {
            let reader = UvfitsReader::new(out, None, None).unwrap();
            let obs_context = reader.get_obs_context();
            (
                obs_context.fine_chan_freqs.clone(),
                obs_context.freq_res.unwrap(),
            )
        };

        let res = width * freq_average as f64;
        assert!(
            (out_freq_res - res).abs() < 1e-6,
            "case {i_case}: written channel width {out_freq_res} != {res}"
        );
        if !smallest_band {
            assert_eq!(num_chanblocks, num_chans / freq_average);
        }
        assert_eq!(out_freqs.len(), num_chanblocks, "case {i_case}");
        for (i, &out_freq) in out_freqs.iter().enumerate() {
            let expected = 100e6
                + width * (freq_average - 1) as f64 / 2.0
                + (first_chanblock + i) as f64 * res;
            // Frequencies are stored in integer Hz.
            assert!(
                (out_freq as f64 - expected).abs() <= 1.0,
                "case {i_case}: written channel {i} is {out_freq} Hz, expected {expected} Hz"
            );
        }
    }
}

/// A measurement set with missing timesteps (gaps in time) averaged with a time
/// factor of 0 ("all timesteps") yields one timeblock.
#[test]
#[serial] // casacore's table cache is not thread-safe
fn test_time_average_all_with_missing_timesteps() {
    let temp_dir = TempDir::new().expect("couldn't make tmp dir");
    let ms = temp_dir.path().join("gaps.ms");
    let ms_string = ms.display().to_string();
    let DataAsStrings {
        metafits, srclist, ..
    } = get_reduced_1090008640_raw();
    let model = temp_dir.path().join("model.uvfits");
    let model_string = model.display().to_string();
    #[rustfmt::skip]
    VisSimulateArgs::parse_from([
        "vis-simulate",
        "--metafits", &metafits,
        "--source-list", &srclist,
        "--output-model-files", &model_string,
        "--num-timesteps", "4",
        "--num-fine-channels", "2",
        "--no-beam",
    ])
    .run(false)
    .unwrap();

    // Write timesteps 0, 1 and 3; timestep 2 is missing from the output.
    #[rustfmt::skip]
    let args = vec![
        "vis-convert",
        "--data", &model_string, &metafits,
        "--timesteps", "0", "1", "3",
        "--outputs", &ms_string,
    ];
    VisConvertArgs::parse_from(args).run(false).unwrap();

    let args = InputVisArgs {
        files: Some(vec![ms_string, metafits]),
        time_average: Some("0".to_string()),
        ..Default::default()
    };
    let params = args.parse("").unwrap();
    assert_eq!(params.get_obs_context().timestamps.len(), 3);
    assert_eq!(params.timeblocks.len(), 1);
    assert_eq!(params.timeblocks.first().timesteps.len(), 3);
}
