use std::{fs::File, io::Write as _};

use approx::assert_abs_diff_eq;
use clap::Parser as _;
use tempfile::tempdir;

use crate::{
    cli::common::{InputVisArgs, SkyModelWithVetoArgs},
    params::{InputVisParams, OutputVisParams, PeelParams},
    tests::{get_reduced_1090008640_raw, DataAsStrings},
};

use super::PeelArgs;

#[track_caller]
fn get_reduced_1090008640() -> PeelArgs {
    let DataAsStrings {
        metafits,
        mut vis,
        srclist,
        ..
    } = get_reduced_1090008640_raw();
    let mut files = vec![metafits];
    files.append(&mut vis);
    PeelArgs {
        data_args: InputVisArgs {
            files: Some(files),
            ..Default::default()
        },
        srclist_args: SkyModelWithVetoArgs {
            source_list: Some(srclist),
            ..Default::default()
        },
        ..Default::default()
    }
}

#[track_caller]
// should be 40kHz, 2s raw
fn get_merged_1090008640(extra_argv: Vec<String>) -> PeelArgs {
    let args = get_reduced_1090008640();
    let temp_dir = tempdir().expect("Couldn't make tempdir");
    let arg_file = temp_dir.path().join("peel.toml");
    let mut f = File::create(&arg_file).expect("couldn't make file");
    let ser = toml::to_string_pretty(&args).expect("couldn't serialise PeelArgs as toml");
    eprintln!("{ser}");
    write!(&mut f, "{ser}").unwrap();
    let mut argv = vec!["peel".to_string(), arg_file.display().to_string()];
    argv.extend(extra_argv);
    PeelArgs::parse_from(argv).merge().unwrap()
}

// testing the frequency averaging args all works together:
//   --freq-average input averaging settings
//   --iono-freq-average - mid-peeling averaging settings
//   --output-vis-freq-average - output averaging settings

#[test]
fn frequency_averaging_defaults() {
    let args = get_merged_1090008640(vec![]);
    let PeelParams {
        input_vis_params: InputVisParams { spw: input_spw, .. },
        output_vis_params,
        low_res_spw,
        ..
    } = args.parse().unwrap();
    let output_freq_average_factor = match output_vis_params {
        Some(OutputVisParams {
            output_freq_average_factor,
            ..
        }) => output_freq_average_factor,
        _ => panic!("Expected OutputVisParams::Single"),
    };
    assert_eq!(input_spw.freq_res, 40e3);
    assert_eq!(low_res_spw.freq_res, 1.28e6);
    assert_eq!(output_freq_average_factor.get(), 1);
}

#[test]
fn frequency_averaging_explicit_output() {
    let args = get_merged_1090008640(vec!["--output-vis-freq-average=80kHz".to_string()]);
    let PeelParams {
        input_vis_params: InputVisParams { spw: input_spw, .. },
        output_vis_params,
        low_res_spw,
        ..
    } = args.parse().unwrap();
    let output_freq_average_factor = match output_vis_params {
        Some(OutputVisParams {
            output_freq_average_factor,
            ..
        }) => output_freq_average_factor,
        _ => panic!("Expected OutputVisParams::Single"),
    };
    assert_eq!(input_spw.freq_res, 40e3);
    assert_eq!(low_res_spw.freq_res, 1.28e6);
    assert_eq!(output_freq_average_factor.get(), 2);
}

#[test]
fn frequency_averaging_explicit_output_iono() {
    let args = get_merged_1090008640(vec![
        "--output-vis-freq-average=80kHz".to_string(),
        "--iono-freq-average=320kHz".to_string(),
    ]);
    let PeelParams {
        input_vis_params: InputVisParams { spw: input_spw, .. },
        output_vis_params,
        low_res_spw,
        ..
    } = args.parse().unwrap();
    let output_freq_average_factor = match output_vis_params {
        Some(OutputVisParams {
            output_freq_average_factor,
            ..
        }) => output_freq_average_factor,
        _ => panic!("Expected OutputVisParams::Single"),
    };
    assert_eq!(input_spw.freq_res, 40e3);
    assert_eq!(low_res_spw.freq_res, 320e3);
    assert_eq!(output_freq_average_factor.get(), 2);
}

#[test]
fn frequency_averaging_explicit_in_out() {
    let args = get_merged_1090008640(vec![
        "--freq-average=80kHz".to_string(),
        "--output-vis-freq-average=160kHz".to_string(),
        "--iono-freq-average=320kHz".to_string(),
    ]);
    let PeelParams {
        input_vis_params: InputVisParams { spw: input_spw, .. },
        output_vis_params,
        low_res_spw,
        ..
    } = args.parse().unwrap();
    let output_freq_average_factor = match output_vis_params {
        Some(OutputVisParams {
            output_freq_average_factor,
            ..
        }) => output_freq_average_factor,
        _ => panic!("Expected OutputVisParams::Single"),
    };
    assert_eq!(input_spw.freq_res, 80e3);
    assert_eq!(low_res_spw.freq_res, 320e3);
    assert_eq!(output_freq_average_factor.get(), 2);
}

#[test]
fn frequency_averaging_explicit() {
    let args = get_merged_1090008640(vec![
        "--freq-average=80kHz".to_string(),
        "--output-vis-freq-average=160kHz".to_string(),
        "--iono-freq-average=320kHz".to_string(),
    ]);
    let PeelParams {
        input_vis_params: InputVisParams { spw: input_spw, .. },
        output_vis_params,
        low_res_spw,
        ..
    } = args.parse().unwrap();
    let output_freq_average_factor = match output_vis_params {
        Some(OutputVisParams {
            output_freq_average_factor,
            ..
        }) => output_freq_average_factor,
        _ => panic!("Expected OutputVisParams::Single"),
    };
    assert_eq!(input_spw.freq_res, 80e3);
    assert_eq!(low_res_spw.freq_res, 320e3);
    assert_eq!(output_freq_average_factor.get(), 2);
}

// in this case the input data is 2s but there's only one timestep, so the
// time res will be clipped to 2s
#[test]
fn time_averaging_explicit_output_clip() {
    let args = get_merged_1090008640(vec!["--output-vis-time-average=4s".to_string()]);
    let PeelParams {
        input_vis_params: InputVisParams { time_res, .. },
        output_vis_params,
        iono_time_average_factor,
        ..
    } = args.parse().unwrap();
    let output_time_average_factor = match output_vis_params {
        Some(OutputVisParams {
            output_time_average_factor,
            ..
        }) => output_time_average_factor,
        _ => panic!("Expected OutputVisParams::Single"),
    };
    assert_abs_diff_eq!(time_res.to_seconds(), 2.0);
    assert_eq!(output_time_average_factor.get(), 1);
    assert_eq!(iono_time_average_factor.get(), 1);
}

// TODO: testing the time averaging args all works together:
//   --time-average input averaging settings
//   --iono-time-average - mid-peeling averaging settings
//   --output-vis-time-average - output averaging settings
//
// this requires test data with more than one timestep
//
// #[test]
// fn time_averaging_defaults() {
//     let args = get_merged_1090008640(vec![]);
//     let PeelParams {
//         input_vis_params: InputVisParams { time_res, .. },
//         output_vis_params,
//         iono_time_average_factor,
//         ..
//     } = args.parse().unwrap();
//     let output_time_average_factor = match output_vis_params {
//         Some(OutputVisParams {
//             output_time_average_factor,
//             ..
//         }) => output_time_average_factor,
//         _ => panic!("Expected OutputVisParams::Single"),
//     };
//     assert_abs_diff_eq!(time_res.to_seconds(), 2.0);
//     assert_eq!(iono_time_average_factor.get(), 4);
//     assert_eq!(output_time_average_factor.get(), 1);
// }

// #[test]
// fn time_averaging_explicit_output() {
//     let args = get_merged_1090008640(vec!["--output-vis-time-average=4s".to_string()]);
//     let PeelParams {
//         input_vis_params: InputVisParams { time_res, .. },
//         output_vis_params,
//         iono_time_average_factor,
//         ..
//     } = args.parse().unwrap();
//     let output_time_average_factor = match output_vis_params {
//         Some(OutputVisParams {
//             output_time_average_factor,
//             ..
//         }) => output_time_average_factor,
//         _ => panic!("Expected OutputVisParams::Single"),
//     };
//     assert_abs_diff_eq!(time_res.to_seconds(), 2.0);
//     assert_eq!(iono_time_average_factor.get(), 4);
//     assert_eq!(output_time_average_factor.get(), 2);
// }

// #[test]
// fn time_averaging_explicit_output_iono() {
//     let args = get_merged_1090008640(vec![
//         "--output-vis-time-average=4s".to_string(),
//         "--iono-time-average=16s".to_string(),
//     ]);
//     let PeelParams {
//         input_vis_params: InputVisParams { time_res, .. },
//         output_vis_params,
//         iono_time_average_factor,
//         ..
//     } = args.parse().unwrap();
//     let output_time_average_factor = match output_vis_params {
//         Some(OutputVisParams {
//             output_time_average_factor,
//             ..
//         }) => output_time_average_factor,
//         _ => panic!("Expected OutputVisParams::Single"),
//     };
//     assert_abs_diff_eq!(time_res.to_seconds(), 2.0);
//     assert_eq!(iono_time_average_factor.get(), 8);
//     assert_eq!(output_time_average_factor.get(), 2);
// }

// #[test]
// fn time_averaging_explicit_in_out() {
//     // enable logging
//     use env_logger::Env;
//     env_logger::Builder::from_env(Env::default().default_filter_or("debug")).init();
//     let args = get_merged_1090008640(vec![
//         "--time-average=4s".to_string(),
//         "--output-vis-time-average=16s".to_string(),
//     ]);
//     let PeelParams {
//         input_vis_params: InputVisParams { time_res, .. },
//         output_vis_params,
//         iono_time_average_factor,
//         ..
//     } = args.parse().unwrap();
//     let output_time_average_factor = match output_vis_params {
//         Some(OutputVisParams {
//             output_time_average_factor,
//             ..
//         }) => output_time_average_factor,
//         _ => panic!("Expected OutputVisParams::Single"),
//     };
//     assert_abs_diff_eq!(time_res.to_seconds(), 4.0);
//     assert_eq!(iono_time_average_factor.get(), 2);
//     assert_eq!(output_time_average_factor.get(), 8);
// }

// #[test]
// fn time_averaging_explicit() {
//     let args = get_merged_1090008640(vec![
//         "--time-average=4s".to_string(),
//         "--output-vis-time-average=16s".to_string(),
//         "--iono-time-average=32s".to_string(),
//     ]);
//     let PeelParams {
//         input_vis_params: InputVisParams { time_res, .. },
//         output_vis_params,
//         iono_time_average_factor,
//         ..
//     } = args.parse().unwrap();
//     let output_time_average_factor = match output_vis_params {
//         Some(OutputVisParams {
//             output_time_average_factor,
//             ..
//         }) => output_time_average_factor,
//         _ => panic!("Expected OutputVisParams::Single"),
//     };
//     assert_abs_diff_eq!(time_res.to_seconds(), 4.0);
//     assert_eq!(iono_time_average_factor.get(), 4);
//     assert_eq!(output_time_average_factor.get(), 8);
// }
