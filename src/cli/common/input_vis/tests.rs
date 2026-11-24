// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

use std::{collections::HashSet, path::PathBuf};

use approx::{assert_abs_diff_eq, assert_relative_eq};
use crossbeam_utils::atomic::AtomicCell;
use hifitime::{Duration, Epoch};
use marlu::{
    constants::{MWA_HEIGHT_M, MWA_LAT_DEG, MWA_LONG_DEG},
    Jones, LatLngHeight, RADec, UvfitsWriter, VisContext, VisWrite, XyzGeodetic,
};
use ndarray::prelude::*;
use serial_test::serial;
use tempfile::TempDir;

use super::{
    InputVisArgs,
    InputVisArgsError::{
        FreqFactorNotInteger, FreqResNotMultiple, NoInputData, TimeFactorNotInteger,
        TimeResNotMultiple,
    },
};
use crate::{
    cli::{
        common::{BeamArgs, ModellingArgs, OutputVisArgs, SkyModelWithVetoArgs},
        vis_simulate::{VisSimulateArgs, VisSimulateCliArgs},
    },
    context::ObsContext,
    flagging::MwafFlags,
    io::read::{AutoData, CrossData, RawDataCorrections, VisInputType, VisRead, VisReadError},
    params::InputVisParams,
    tests::{
        get_reduced_1090008640_ms, get_reduced_1090008640_raw, get_reduced_1090008640_uvfits,
        DataAsStrings,
    },
};

#[test]
fn test_handle_no_input() {
    let args = InputVisArgs::default();
    let result = args.parse("");

    assert!(result.is_err());
    assert!(matches!(result, Err(NoInputData)));
}

#[test]
fn test_handle_multiple_metafits() {
    let DataAsStrings {
        metafits, mut vis, ..
    } = get_reduced_1090008640_raw();
    let mut files = vec![metafits.clone()];
    files.append(&mut vis);
    // Add the metafits again.
    files.push(metafits);

    let args = InputVisArgs {
        files: Some(files),
        ..Default::default()
    };
    let result = args.parse("");

    assert!(result.is_err());
    assert!(result
        .err()
        .unwrap()
        .to_string()
        .contains("Multiple metafits files were specified"));
}

#[test]
fn test_handle_multiple_ms() {
    let DataAsStrings {
        metafits, mut vis, ..
    } = get_reduced_1090008640_ms();
    let mut files = vec![metafits];
    let ms = vis.swap_remove(0);
    files.push(ms.clone());
    files.push(ms);

    let args = InputVisArgs {
        files: Some(files),
        ..Default::default()
    };
    let result = args.parse("");

    assert!(result.is_err());
    assert!(result
        .err()
        .unwrap()
        .to_string()
        .contains("Multiple measurement sets were specified"));
}

#[test]
fn test_handle_multiple_uvfits() {
    let DataAsStrings {
        metafits, mut vis, ..
    } = get_reduced_1090008640_uvfits();
    let mut files = vec![metafits];
    let uvfits = vis.swap_remove(0);
    files.push(uvfits.clone());
    files.push(uvfits);

    let args = InputVisArgs {
        files: Some(files),
        ..Default::default()
    };
    let result = args.parse("");

    assert!(result.is_err());
    assert!(result
        .err()
        .unwrap()
        .to_string()
        .contains("Multiple uvfits files were specified"));
}

#[test]
fn test_handle_only_metafits() {
    let DataAsStrings { metafits, .. } = get_reduced_1090008640_raw();
    let files = vec![metafits];

    let args = InputVisArgs {
        files: Some(files),
        ..Default::default()
    };
    let result = args.parse("");

    assert!(result.is_err());
    assert!(result
        .err()
        .unwrap()
        .to_string()
        .contains("Received only a metafits file;"));
}

#[test]
fn test_handle_array_pos() {
    let DataAsStrings {
        metafits, mut vis, ..
    } = get_reduced_1090008640_raw();
    let mut files = vec![metafits];
    files.append(&mut vis);

    let expected = [MWA_LONG_DEG + 1.0, MWA_LAT_DEG + 1.0, MWA_HEIGHT_M + 1.0];
    let args = InputVisArgs {
        files: Some(files),
        array_position: Some(expected.to_vec()),
        ..Default::default()
    };
    let params = args.parse("").unwrap();

    assert_abs_diff_eq!(
        params.vis_reader.get_obs_context().array_position,
        LatLngHeight {
            longitude_rad: expected[0].to_radians(),
            latitude_rad: expected[1].to_radians(),
            height_metres: expected[2]
        }
    );
}

#[test]
fn test_handle_bad_array_pos() {
    let DataAsStrings {
        metafits, mut vis, ..
    } = get_reduced_1090008640_raw();
    let mut files = vec![metafits];
    files.append(&mut vis);

    let two_elems_when_it_should_be_three = [MWA_LONG_DEG + 1.0, MWA_LAT_DEG + 1.0];
    let args = InputVisArgs {
        files: Some(files),
        array_position: Some(two_elems_when_it_should_be_three.to_vec()),
        ..Default::default()
    };
    let result = args.parse("");

    assert!(result.is_err());
    assert!(result
        .err()
        .unwrap()
        .to_string()
        .contains("Array position specified as"));
}

#[test]
fn test_parse_time_average() {
    let DataAsStrings {
        metafits, mut vis, ..
    } = get_reduced_1090008640_raw();
    let mut files = vec![metafits];
    files.append(&mut vis);

    let mut args = InputVisArgs {
        files: Some(files),
        time_average: Some("1".to_string()),
        ..Default::default()
    };
    args.clone().parse("").unwrap();

    args.time_average = Some("2".to_string());
    args.clone().parse("").unwrap();

    args.time_average = Some("20".to_string());
    args.clone().parse("").unwrap();

    // The native time resolution is 2s.
    args.time_average = Some("2s".to_string());
    args.clone().parse("").unwrap();

    args.time_average = Some("4s".to_string());
    args.clone().parse("").unwrap();

    args.time_average = Some("20s".to_string());
    args.clone().parse("").unwrap();

    args.time_average = Some("1.5".to_string());
    let result = args.clone().parse("");
    assert!(matches!(result.err(), Some(TimeFactorNotInteger)));

    args.time_average = Some("3s".to_string());
    let result = args.clone().parse("");
    assert!(matches!(result.err(), Some(TimeResNotMultiple { .. })));

    args.time_average = Some("7s".to_string());
    let result = args.parse("");
    assert!(matches!(result.err(), Some(TimeResNotMultiple { .. })));
}

#[test]
fn test_parse_freq_average() {
    let DataAsStrings {
        metafits, mut vis, ..
    } = get_reduced_1090008640_raw();
    let mut files = vec![metafits];
    files.append(&mut vis);

    let mut args = InputVisArgs {
        files: Some(files),
        freq_average: Some("1".to_string()),
        ..Default::default()
    };
    args.clone().parse("").unwrap();

    args.freq_average = Some("2".to_string());
    args.clone().parse("").unwrap();

    args.freq_average = Some("20".to_string());
    args.clone().parse("").unwrap();

    // The native freq. resolution is 40kHz.
    args.freq_average = Some("40kHz".to_string());
    args.clone().parse("").unwrap();

    args.freq_average = Some("80kHz".to_string());
    args.clone().parse("").unwrap();

    args.freq_average = Some("960kHz".to_string());
    args.clone().parse("").unwrap();

    args.freq_average = Some("1.5".to_string());
    let result = args.clone().parse("");
    assert!(matches!(result.err(), Some(FreqFactorNotInteger)));

    args.freq_average = Some("10kHz".to_string());
    let result = args.clone().parse("");
    assert!(matches!(result.err(), Some(FreqResNotMultiple { .. })));

    args.freq_average = Some("79kHz".to_string());
    let result = args.parse("");
    assert!(matches!(result.err(), Some(FreqResNotMultiple { .. })));
}

#[test]
fn test_freq_averaging_works() {
    let DataAsStrings {
        metafits, mut vis, ..
    } = get_reduced_1090008640_raw();
    let mut files = vec![metafits];
    files.append(&mut vis);

    let mut args = InputVisArgs {
        files: Some(files),
        freq_average: Some("1".to_string()),
        ..Default::default()
    };
    let default_params = args.clone().parse("").unwrap();
    let num_unflagged_tiles = default_params.get_num_unflagged_tiles();
    let num_unflagged_cross_baselines = (num_unflagged_tiles * (num_unflagged_tiles - 1)) / 2;
    let cross_vis_shape = (
        default_params.spw.chanblocks.len(),
        num_unflagged_cross_baselines,
    );
    let auto_vis_shape = (default_params.spw.chanblocks.len(), num_unflagged_tiles);

    let mut default_crosses_fb = Array2::zeros(cross_vis_shape);
    let mut default_cross_weights_fb = Array2::zeros(cross_vis_shape);
    let mut default_autos_fb = Array2::zeros(auto_vis_shape);
    let mut default_auto_weights_fb = Array2::zeros(auto_vis_shape);
    let error = AtomicCell::new(false);
    default_params
        .read_timeblock(
            default_params.timeblocks.first(),
            default_crosses_fb.view_mut(),
            default_cross_weights_fb.view_mut(),
            Some((
                default_autos_fb.view_mut(),
                default_auto_weights_fb.view_mut(),
            )),
            &error,
        )
        .unwrap();

    // 27 unflagged channels, 8128 unflagged baselines.
    assert_eq!(default_crosses_fb.dim(), (27, 8128));
    assert_eq!(default_cross_weights_fb.dim(), (27, 8128));
    // 27 unflagged channels, 128 unflagged tiles.
    assert_eq!(default_autos_fb.dim(), (27, 128));
    assert_eq!(default_auto_weights_fb.dim(), (27, 128));

    args.ignore_input_data_fine_channel_flags = true;
    let ref_params = args.clone().parse("").unwrap();
    let num_unflagged_tiles = ref_params.get_num_unflagged_tiles();
    let num_unflagged_cross_baselines = (num_unflagged_tiles * (num_unflagged_tiles - 1)) / 2;
    let cross_vis_shape = (
        ref_params.spw.chanblocks.len(),
        num_unflagged_cross_baselines,
    );
    let auto_vis_shape = (ref_params.spw.chanblocks.len(), num_unflagged_tiles);

    let mut ref_crosses_fb = Array2::zeros(cross_vis_shape);
    let mut ref_cross_weights_fb = Array2::zeros(cross_vis_shape);
    let mut ref_autos_fb = Array2::zeros(auto_vis_shape);
    let mut ref_auto_weights_fb = Array2::zeros(auto_vis_shape);
    let error = AtomicCell::new(false);
    ref_params
        .read_timeblock(
            ref_params.timeblocks.first(),
            ref_crosses_fb.view_mut(),
            ref_cross_weights_fb.view_mut(),
            Some((ref_autos_fb.view_mut(), ref_auto_weights_fb.view_mut())),
            &error,
        )
        .unwrap();

    // 32 unflagged channels, 8128 unflagged baselines.
    assert_eq!(ref_crosses_fb.dim(), (32, 8128));
    assert_eq!(ref_cross_weights_fb.dim(), (32, 8128));
    // 32 unflagged channels, 128 unflagged tiles.
    assert_eq!(ref_autos_fb.dim(), (32, 128));
    assert_eq!(ref_auto_weights_fb.dim(), (32, 128));

    args.ignore_input_data_fine_channel_flags = false;
    args.freq_average = Some("2".to_string());
    let av_params = args.clone().parse("").unwrap();
    let num_unflagged_tiles = av_params.get_num_unflagged_tiles();
    let num_unflagged_cross_baselines = (num_unflagged_tiles * (num_unflagged_tiles - 1)) / 2;
    let cross_vis_shape = (
        av_params.spw.chanblocks.len(),
        num_unflagged_cross_baselines,
    );
    let auto_vis_shape = (av_params.spw.chanblocks.len(), num_unflagged_tiles);

    let mut av_crosses_fb = Array2::zeros(cross_vis_shape);
    let mut av_cross_weights_fb = Array2::zeros(cross_vis_shape);
    let mut av_autos_fb = Array2::zeros(auto_vis_shape);
    let mut av_auto_weights_fb = Array2::zeros(auto_vis_shape);
    let error = AtomicCell::new(false);
    av_params
        .read_timeblock(
            av_params.timeblocks.first(),
            av_crosses_fb.view_mut(),
            av_cross_weights_fb.view_mut(),
            Some((av_autos_fb.view_mut(), av_auto_weights_fb.view_mut())),
            &error,
        )
        .unwrap();

    // 14 unflagged chanblocks, 8128 unflagged baselines.
    assert_eq!(av_crosses_fb.dim(), (14, 8128));
    assert_eq!(av_cross_weights_fb.dim(), (14, 8128));
    // 14 unflagged chanblocks, 128 unflagged tiles.
    assert_eq!(av_autos_fb.dim(), (14, 128));
    assert_eq!(av_auto_weights_fb.dim(), (14, 128));

    // Channels 0 1 16 30 31 are flagged by default, so manually average the
    // unflagged channels and compare with the averaged arrays.
    let flags = [0, 1, 16, 30, 31];
    let mut av_vis = Jones::default();
    let mut weight_sum: f64 = 0.0;

    for i_bl in 0..8128 {
        let av_crosses_f = av_crosses_fb.slice(s![.., i_bl]);
        let mut av_crosses_iter = av_crosses_f.iter();
        let av_cross_weights_f = av_cross_weights_fb.slice(s![.., i_bl]);
        let mut av_cross_weights_iter = av_cross_weights_f.iter();
        let mut i_unflagged_chan = 0;
        for i_chan in 0..32 {
            if i_chan % 2 == 0 && weight_sum.abs() > 0.0 {
                // Compare.
                av_vis /= weight_sum;
                let j = Jones::from(av_crosses_iter.next().unwrap());
                assert_relative_eq!(av_vis, j, max_relative = 1e-7);
                assert_relative_eq!(
                    weight_sum,
                    *av_cross_weights_iter.next().unwrap() as f64,
                    max_relative = 1e-7
                );

                av_vis = Jones::default();
                weight_sum = 0.0;
            }

            if flags.contains(&i_chan) {
                continue;
            }
            // Compare unaveraged vis with one another.
            assert_abs_diff_eq!(
                ref_crosses_fb[(i_chan, i_bl)],
                default_crosses_fb[(i_unflagged_chan, i_bl)]
            );
            assert_abs_diff_eq!(
                ref_cross_weights_fb[(i_chan, i_bl)],
                default_cross_weights_fb[(i_unflagged_chan, i_bl)]
            );
            i_unflagged_chan += 1;

            let weight = ref_cross_weights_fb[(i_chan, i_bl)] as f64;
            // Promote Jones before dividing to keep precision high.
            av_vis += Jones::from(ref_crosses_fb[(i_chan, i_bl)]) * weight;
            weight_sum += weight;
        }
    }

    for i_tile in 0..128 {
        let av_autos_f = av_autos_fb.slice(s![.., i_tile]);
        let mut av_autos_iter = av_autos_f.iter();
        let av_auto_weights_f = av_auto_weights_fb.slice(s![.., i_tile]);
        let mut av_auto_weights_iter = av_auto_weights_f.iter();
        let mut i_unflagged_chan = 0;
        for i_chan in 0..32 {
            if i_chan % 2 == 0 && weight_sum.abs() > 0.0 {
                // Compare.
                av_vis /= weight_sum;
                let j = Jones::from(av_autos_iter.next().unwrap());
                assert_relative_eq!(av_vis, j, max_relative = 1e-7);
                assert_relative_eq!(
                    weight_sum,
                    *av_auto_weights_iter.next().unwrap() as f64,
                    max_relative = 1e-7
                );

                av_vis = Jones::default();
                weight_sum = 0.0;
            }

            if flags.contains(&i_chan) {
                continue;
            }
            // Compare unaveraged vis with one another.
            assert_abs_diff_eq!(
                ref_autos_fb[(i_chan, i_tile)],
                default_autos_fb[(i_unflagged_chan, i_tile)]
            );
            assert_abs_diff_eq!(
                ref_auto_weights_fb[(i_chan, i_tile)],
                default_auto_weights_fb[(i_unflagged_chan, i_tile)]
            );
            i_unflagged_chan += 1;

            let weight = ref_auto_weights_fb[(i_chan, i_tile)] as f64;
            // Promote Jones before dividing to keep precision high.
            av_vis += Jones::from(ref_autos_fb[(i_chan, i_tile)]) * weight;
            weight_sum += weight;
        }
    }

    // Do it all again with 3 channel averaging.
    args.freq_average = Some("3".to_string());
    let av_params = args.parse("").unwrap();
    let num_unflagged_tiles = av_params.get_num_unflagged_tiles();
    let num_unflagged_cross_baselines = (num_unflagged_tiles * (num_unflagged_tiles - 1)) / 2;
    let cross_vis_shape = (
        av_params.spw.chanblocks.len(),
        num_unflagged_cross_baselines,
    );
    let auto_vis_shape = (av_params.spw.chanblocks.len(), num_unflagged_tiles);

    let mut av_crosses_fb = Array2::zeros(cross_vis_shape);
    let mut av_cross_weights_fb = Array2::zeros(cross_vis_shape);
    let mut av_autos_fb = Array2::zeros(auto_vis_shape);
    let mut av_auto_weights_fb = Array2::zeros(auto_vis_shape);
    let error = AtomicCell::new(false);
    av_params
        .read_timeblock(
            av_params.timeblocks.first(),
            av_crosses_fb.view_mut(),
            av_cross_weights_fb.view_mut(),
            Some((av_autos_fb.view_mut(), av_auto_weights_fb.view_mut())),
            &error,
        )
        .unwrap();

    // 10 unflagged chanblocks, 8128 unflagged baselines.
    assert_eq!(av_crosses_fb.dim(), (10, 8128));
    assert_eq!(av_cross_weights_fb.dim(), (10, 8128));
    // 10 unflagged chanblocks, 128 unflagged tiles.
    assert_eq!(av_autos_fb.dim(), (10, 128));
    assert_eq!(av_auto_weights_fb.dim(), (10, 128));

    let flags = [0, 1, 16, 30, 31];
    let mut av_vis = Jones::default();
    let mut weight_sum: f64 = 0.0;

    for i_bl in 0..8128 {
        let av_crosses_f = av_crosses_fb.slice(s![.., i_bl]);
        let mut av_crosses_iter = av_crosses_f.iter();
        let av_cross_weights_f = av_cross_weights_fb.slice(s![.., i_bl]);
        let mut av_cross_weights_iter = av_cross_weights_f.iter();
        let mut i_unflagged_chan = 0;
        for i_chan in 0..32 {
            if i_chan % 3 == 0 && weight_sum.abs() > 0.0 {
                // Compare.
                av_vis /= weight_sum;
                let j = Jones::from(av_crosses_iter.next().unwrap());
                assert_relative_eq!(av_vis, j, max_relative = 1e-7);
                assert_relative_eq!(
                    weight_sum,
                    *av_cross_weights_iter.next().unwrap() as f64,
                    max_relative = 1e-7
                );

                av_vis = Jones::default();
                weight_sum = 0.0;
            }

            if flags.contains(&i_chan) {
                continue;
            }
            // Compare unaveraged vis with one another.
            assert_abs_diff_eq!(
                ref_crosses_fb[(i_chan, i_bl)],
                default_crosses_fb[(i_unflagged_chan, i_bl)]
            );
            assert_abs_diff_eq!(
                ref_cross_weights_fb[(i_chan, i_bl)],
                default_cross_weights_fb[(i_unflagged_chan, i_bl)]
            );
            i_unflagged_chan += 1;

            let weight = ref_cross_weights_fb[(i_chan, i_bl)] as f64;
            // Promote Jones before dividing to keep precision high.
            av_vis += Jones::from(ref_crosses_fb[(i_chan, i_bl)]) * weight;
            weight_sum += weight;
        }
    }

    for i_tile in 0..128 {
        let av_autos_f = av_autos_fb.slice(s![.., i_tile]);
        let mut av_autos_iter = av_autos_f.iter();
        let av_auto_weights_f = av_auto_weights_fb.slice(s![.., i_tile]);
        let mut av_auto_weights_iter = av_auto_weights_f.iter();
        let mut i_unflagged_chan = 0;
        for i_chan in 0..32 {
            if i_chan % 3 == 0 && weight_sum.abs() > 0.0 {
                // Compare.
                av_vis /= weight_sum;
                let j = Jones::from(av_autos_iter.next().unwrap());
                assert_relative_eq!(av_vis, j, max_relative = 1e-7);
                assert_relative_eq!(
                    weight_sum,
                    *av_auto_weights_iter.next().unwrap() as f64,
                    max_relative = 1e-7
                );

                av_vis = Jones::default();
                weight_sum = 0.0;
            }

            if flags.contains(&i_chan) {
                continue;
            }
            // Compare unaveraged vis with one another.
            assert_abs_diff_eq!(
                ref_autos_fb[(i_chan, i_tile)],
                default_autos_fb[(i_unflagged_chan, i_tile)]
            );
            assert_abs_diff_eq!(
                ref_auto_weights_fb[(i_chan, i_tile)],
                default_auto_weights_fb[(i_unflagged_chan, i_tile)]
            );
            i_unflagged_chan += 1;

            let weight = ref_auto_weights_fb[(i_chan, i_tile)] as f64;
            // Promote Jones before dividing to keep precision high.
            av_vis += Jones::from(ref_autos_fb[(i_chan, i_tile)]) * weight;
            weight_sum += weight;
        }
    }
}

#[test]
/// `timesteps_to_timeblocks` now needs the time resolution, because not
/// supplying it was giving incorrect timeblocks when averaging. This test
/// helps to ensure that the expected behaviour works.
fn sparse_timeblocks_with_averaging() {
    // First, make some data with enough timesteps.
    let tmp_dir = TempDir::new().unwrap();
    let DataAsStrings {
        metafits,
        vis: _,
        mwafs: _,
        srclist,
    } = get_reduced_1090008640_raw();

    let output = tmp_dir.path().join("20ts.uvfits");
    let args = VisSimulateArgs {
        args_file: None,
        beam_args: BeamArgs {
            beam_type: None,
            no_beam: true,
            delays: None,
            unity_dipole_gains: true,
            beam_file: None,
        },
        modelling_args: ModellingArgs {
            ..Default::default()
        },
        srclist_args: SkyModelWithVetoArgs {
            source_list: Some(srclist),
            num_sources: Some(1),
            ..Default::default()
        },
        simulate_args: VisSimulateCliArgs {
            metafits: Some(PathBuf::from(&metafits)),
            num_timesteps: Some(20),
            time_res: Some(2.0),
            num_fine_channels: Some(1),
            output_model_files: Some(vec![output.clone()]),
            output_autos: false,
            ..Default::default()
        },
    };
    args.run(false).unwrap();

    // Now try to read it with sparse timesteps and averaging.
    let args = InputVisArgs {
        files: Some(vec![metafits, output.display().to_string()]),
        timesteps: Some(vec![6, 12, 18]),
        time_average: Some("8s".to_string()),
        ..Default::default()
    };
    let params = args.parse("").unwrap();
    assert_eq!(params.timeblocks.len(), 3);

    let output_params = OutputVisArgs {
        outputs: Some(vec![tmp_dir.path().join("output.uvfits")]),
        output_vis_time_average: None,
        output_vis_freq_average: None,
        output_autos: false,
    }
    .parse(
        params.time_res,
        params.spw.freq_res,
        &params.timeblocks.mapped_ref(|tb| tb.median),
        false,
        "adsf.uvfits",
        None,
    )
    .unwrap();
    assert_eq!(output_params.output_timeblocks.len(), 3);
}

#[test]
fn test_non_mwa_data_ignores_per_coarse_chan_flags() {
    // 1 MHz channels don't divide an MWA coarse channel, so this isn't treated
    // as MWA data.
    let tmp_dir = TempDir::new().unwrap();
    let uvfits = tmp_dir.path().join("non_mwa.uvfits");
    let vis_ctx = VisContext {
        num_sel_timesteps: 1,
        start_timestamp: Epoch::from_gpst_seconds(1_400_000_000.0),
        int_time: Duration::from_seconds(10.0),
        num_sel_chans: 4,
        start_freq_hz: 744e6,
        freq_resolution_hz: 1e6,
        sel_baselines: vec![(0, 1), (0, 2), (1, 2)],
        avg_time: 1,
        avg_freq: 1,
        num_vis_pols: 4,
    };
    let vis = Array3::from_elem(vis_ctx.avg_dims(), Jones::identity());
    let weights = Array3::from_elem(vis_ctx.avg_dims(), 1.0);
    let mut writer = UvfitsWriter::from_marlu(
        &uvfits,
        &vis_ctx,
        LatLngHeight::mwa(),
        RADec::from_degrees(0.0, -27.0),
        Duration::from_seconds(0.0),
        None,
        vec!["ak01".into(), "ak02".into(), "ak03".into()],
        vec![
            XyzGeodetic::default(),
            XyzGeodetic {
                x: 100.0,
                y: 20.0,
                z: 0.0,
            },
            XyzGeodetic {
                x: 30.0,
                y: 150.0,
                z: 10.0,
            },
        ],
        true,
        None,
    )
    .unwrap();
    writer
        .write_vis(vis.view(), weights.view(), &vis_ctx)
        .unwrap();
    writer.finalise().unwrap();

    // The per-coarse-channel flags are ignored rather than rejected.
    let params = InputVisArgs {
        files: Some(vec![uvfits.display().to_string()]),
        fine_chan_flags_per_coarse_chan: Some(vec![0, 1]),
        ..Default::default()
    }
    .parse("")
    .unwrap();
    assert!(params.spw.flagged_chan_indices.is_empty());
    assert_eq!(params.spw.chanblocks.len(), 4);
}

/// A non-MWA measurement set with a 5425.35 Hz channel width
/// (`test_files/mini_ms/mini.ms`). The channel centres are stored to the
/// nearest hertz, so they are alternately 5425 and 5426 Hz apart; they must
/// still form one spectral window, with and without averaging and flags, and
/// the chanblock frequencies must not drift from the real ones. (Before
/// frequencies were handled as floating-point hertz, this data was taken to
/// be a "picket fence", which exits the program.)
#[test]
#[serial] // casacore's table cache is not thread-safe
fn test_fractional_chan_width_chanblocks() {
    let ms = "test_files/mini_ms/mini.ms".to_string();
    let width = 5425.35;
    let num_chans = 32;

    for (freq_average, flags) in [
        (1, vec![]),
        (1, vec![0, 1, 31]),
        (4, vec![0, 1, 9]),
        (4, vec![4, 5, 6, 7, 9]),
        (8, (0..8).collect()),
    ] {
        let args = InputVisArgs {
            files: Some(vec![ms.clone()]),
            freq_average: Some(freq_average.to_string()),
            fine_chan_flags: (!flags.is_empty()).then(|| flags.clone()),
            ..Default::default()
        };
        let params = args.parse("").unwrap();
        let spw = &params.spw;

        let num_chanblocks = num_chans / freq_average;
        // A chanblock is flagged only when all of its channels are.
        let flagged_chanblocks: HashSet<u16> = (0..num_chanblocks as u16)
            .filter(|&i| {
                (i * freq_average as u16..(i + 1) * freq_average as u16).all(|c| flags.contains(&c))
            })
            .collect();
        assert_eq!(
            spw.chanblocks.len() + spw.flagged_chanblock_indices.len(),
            num_chanblocks,
            "freq_average {freq_average}, flags {flags:?}"
        );
        assert_eq!(spw.flagged_chanblock_indices, flagged_chanblocks);
        assert_eq!(spw.flagged_chan_indices, flags.iter().copied().collect());
        let res = width * freq_average as f64;
        assert_abs_diff_eq!(spw.freq_res, res, epsilon = 1e-9);

        // Each chanblock is centred on its channels. The input channel centres
        // are integer hertz, so allow a hertz.
        let expected = |i_chanblock: usize| {
            100e6 + width * (freq_average - 1) as f64 / 2.0 + i_chanblock as f64 * res
        };
        assert_abs_diff_eq!(spw.first_freq, expected(0), epsilon = 1.0);
        for chanblock in &spw.chanblocks {
            assert_abs_diff_eq!(
                chanblock.freq,
                expected(usize::from(chanblock.chanblock_index)),
                epsilon = 1.0
            );
        }
        let all_freqs = spw.get_all_freqs();
        assert_eq!(all_freqs.len(), num_chanblocks);
        for (i, &freq) in all_freqs.iter().enumerate() {
            assert_abs_diff_eq!(freq, expected(i), epsilon = 1.0);
        }
    }
}

/// As above, but with a real measurement set (`test_files/mini_ms/mini.ms`,
/// 5425.35 Hz channels at 100 MHz). Previously it was taken for MWA data with
/// 236 fine channels per coarse channel, and per-coarse-channel flags were
/// applied on that made-up grid (flagging channel 0 here).
#[test]
#[serial] // casacore's table cache is not thread-safe
fn test_mini_ms_ignores_per_coarse_chan_flags() {
    let params = InputVisArgs {
        files: Some(vec!["test_files/mini_ms/mini.ms".to_string()]),
        fine_chan_flags_per_coarse_chan: Some(vec![0, 1]),
        ..Default::default()
    }
    .parse("")
    .unwrap();
    let obs_context = params.get_obs_context();
    assert!(obs_context.mwa_coarse_chan_nums.is_none());
    assert!(obs_context.num_fine_chans_per_coarse_chan.is_none());
    assert!(params.spw.flagged_chan_indices.is_empty());
    assert_eq!(params.spw.chanblocks.len(), 32);
}

/// Wraps another reader and sets one visibility of every timestep to NaN
/// without touching its weight, like data that had NaN calibration
/// solutions applied to it without being flagged.
struct NanInjector {
    inner: Box<dyn VisRead>,
    /// The (channel, baseline) index into the array being read into.
    index: (usize, usize),
}

impl VisRead for NanInjector {
    fn get_obs_context(&self) -> &ObsContext {
        self.inner.get_obs_context()
    }

    fn get_input_data_type(&self) -> VisInputType {
        self.inner.get_input_data_type()
    }

    fn get_metafits_context(&self) -> Option<&mwalib::MetafitsContext> {
        self.inner.get_metafits_context()
    }

    fn get_flags(&self) -> Option<&MwafFlags> {
        self.inner.get_flags()
    }

    fn get_raw_data_corrections(&self) -> Option<RawDataCorrections> {
        self.inner.get_raw_data_corrections()
    }

    fn set_raw_data_corrections(&mut self, corrections: RawDataCorrections) {
        self.inner.set_raw_data_corrections(corrections)
    }

    fn get_marlu_mwa_info(&self) -> Option<marlu::MwaObsContext> {
        self.inner.get_marlu_mwa_info()
    }

    fn read_inner_dispatch(
        &self,
        mut cross_data: Option<CrossData>,
        mut auto_data: Option<AutoData>,
        timestep: usize,
        flagged_fine_chans: &HashSet<u16>,
    ) -> Result<(), VisReadError> {
        self.inner.read_inner_dispatch(
            cross_data.as_mut().map(|c| CrossData {
                vis_fb: c.vis_fb.view_mut(),
                weights_fb: c.weights_fb.view_mut(),
                tile_baseline_flags: c.tile_baseline_flags,
            }),
            auto_data.as_mut().map(|a| AutoData {
                vis_fb: a.vis_fb.view_mut(),
                weights_fb: a.weights_fb.view_mut(),
                tile_baseline_flags: a.tile_baseline_flags,
            }),
            timestep,
            flagged_fine_chans,
        )?;
        if let Some(c) = cross_data.as_mut() {
            c.vis_fb[self.index] = Jones::nan();
        }
        Ok(())
    }
}

/// Read the first timeblock of the reduced 1090008640 raw data, with or
/// without a NaN injected into (unaveraged) channel 5 of baseline 10.
fn read_1090008640(
    freq_average: Option<&str>,
    inject_nan: bool,
) -> (Array2<Jones<f32>>, Array2<f32>) {
    let DataAsStrings {
        metafits, mut vis, ..
    } = get_reduced_1090008640_raw();
    let mut files = vec![metafits];
    files.append(&mut vis);
    let mut params = InputVisArgs {
        files: Some(files),
        freq_average: freq_average.map(|s| s.to_string()),
        // Channel 5 means the same thing whether or not the flagged
        // channels are left out of the arrays being read into.
        ignore_input_data_fine_channel_flags: true,
        ..Default::default()
    }
    .parse("")
    .unwrap();
    if inject_nan {
        params = InputVisParams {
            vis_reader: Box::new(NanInjector {
                inner: params.vis_reader,
                index: (5, 10),
            }),
            ..params
        };
    }

    let num_tiles = params.get_num_unflagged_tiles();
    let shape = (params.spw.chanblocks.len(), num_tiles * (num_tiles - 1) / 2);
    let mut data_fb = Array2::zeros(shape);
    let mut weights_fb = Array2::zeros(shape);
    params
        .read_timeblock(
            params.timeblocks.first(),
            data_fb.view_mut(),
            weights_fb.view_mut(),
            None,
            &AtomicCell::new(false),
        )
        .unwrap();
    (data_fb, weights_fb)
}

/// An unflagged NaN visibility comes out of the reader flagged and zeroed.
/// <https://github.com/MWATelescope/mwa_hyperdrive/issues/78>
#[test]
fn test_unflagged_nan_is_flagged_when_not_averaging() {
    let (ref_data, ref_weights) = read_1090008640(None, false);
    assert!(ref_weights[(5, 10)] > 0.0);
    assert_ne!(ref_data[(5, 10)], Jones::default());

    let (data, weights) = read_1090008640(None, true);
    assert_eq!(data[(5, 10)], Jones::default());
    assert_eq!(weights[(5, 10)], -ref_weights[(5, 10)]);

    // Nothing else changed.
    let (mut data, mut weights) = (data, weights);
    data[(5, 10)] = ref_data[(5, 10)];
    weights[(5, 10)] = ref_weights[(5, 10)];
    assert_eq!(data, ref_data);
    assert_eq!(weights, ref_weights);
}

/// An unflagged NaN visibility doesn't make the average it belongs to NaN;
/// the average is of the other visibilities.
#[test]
fn test_unflagged_nan_is_left_out_of_averages() {
    // The reduced data has 40 kHz channels; average pairs of them.
    // Channel 5 is in the third chanblock, with channel 4.
    let (unaveraged_data, unaveraged_weights) = read_1090008640(None, false);
    let (ref_data, ref_weights) = read_1090008640(Some("80kHz"), false);
    assert_eq!(
        ref_data.len_of(Axis(0)) * 2,
        unaveraged_data.len_of(Axis(0))
    );
    assert!(unaveraged_weights[(4, 10)] > 0.0);
    assert!(unaveraged_weights[(5, 10)] > 0.0);

    let (data, weights) = read_1090008640(Some("80kHz"), true);
    assert!(data.iter().all(|j| !j.any_nan()));
    // The average is channel 4 alone.
    assert_eq!(data[(2, 10)], unaveraged_data[(4, 10)]);
    assert_eq!(weights[(2, 10)], unaveraged_weights[(4, 10)]);
    assert_ne!(data[(2, 10)], ref_data[(2, 10)]);

    // Nothing else changed.
    let (mut data, mut weights) = (data, weights);
    data[(2, 10)] = ref_data[(2, 10)];
    weights[(2, 10)] = ref_weights[(2, 10)];
    assert_eq!(data, ref_data);
    assert_eq!(weights, ref_weights);
}
