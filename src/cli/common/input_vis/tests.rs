// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

use std::path::PathBuf;

use approx::{assert_abs_diff_eq, assert_relative_eq};
use crossbeam_utils::atomic::AtomicCell;
use marlu::{
    constants::{MWA_HEIGHT_M, MWA_LAT_DEG, MWA_LONG_DEG},
    Jones, LatLngHeight,
};
use ndarray::prelude::*;
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
            output_no_autos: true,
            ..Default::default()
        },
    };
    args.run(false).unwrap();

    // Now try to read it with sparse timesteps and averaging.
    let args = InputVisArgs {
        files: Some(vec![metafits, output.display().to_string()]),
        timesteps: Some(vec![6, 12, 18]),
        time_average: Some("8s".to_string()),
        no_autos: true,
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
