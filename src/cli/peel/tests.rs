use std::{fs::File, io::Write as _};

use approx::assert_abs_diff_eq;
use clap::Parser as _;
use serial_test::serial;
use tempfile::tempdir;

use crate::{
    cli::common::{InputVisArgs, SkyModelWithVetoArgs},
    params::{InputVisParams, OutputVisParams, PeelLoopParams, PeelParams, PeelWeightParams},
    tests::{get_reduced_1090008640_raw, DataAsStrings},
    HyperdriveError,
};

use super::{PeelArgs, PeelCliArgs};

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

// should be 40kHz, 2s raw
fn get_merged_1090008640(extra_argv: Vec<String>) -> PeelParams {
    let args = get_reduced_1090008640();
    let temp_dir = tempdir().expect("Couldn't make tempdir");
    let arg_file = temp_dir.path().join("peel.toml");
    let mut f = File::create(&arg_file).expect("couldn't make file");
    let ser = toml::to_string_pretty(&args).expect("couldn't serialise PeelArgs as toml");
    eprintln!("{ser}");
    write!(&mut f, "{ser}").unwrap();
    let mut argv = vec!["peel".to_string(), arg_file.display().to_string()];
    argv.extend(extra_argv);
    let params = PeelArgs::parse_from(argv).merge().unwrap().parse().unwrap();
    drop(f);
    drop(temp_dir);
    params
}

// testing the frequency averaging args all works together:
//   --freq-average input averaging settings
//   --iono-freq-average - mid-peeling averaging settings
//   --output-vis-freq-average - output averaging settings

#[test]
#[serial]
fn frequency_averaging_defaults() {
    let PeelParams {
        input_vis_params: InputVisParams { spw: input_spw, .. },
        output_vis_params,
        low_res_spw,
        ..
    } = get_merged_1090008640(vec![]);
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
#[serial]
fn frequency_averaging_explicit_output() {
    let PeelParams {
        input_vis_params: InputVisParams { spw: input_spw, .. },
        output_vis_params,
        low_res_spw,
        ..
    } = get_merged_1090008640(vec!["--output-vis-freq-average=80kHz".to_string()]);
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
#[serial]
fn frequency_averaging_explicit_output_iono() {
    let PeelParams {
        input_vis_params: InputVisParams { spw: input_spw, .. },
        output_vis_params,
        low_res_spw,
        ..
    } = get_merged_1090008640(vec![
        "--output-vis-freq-average=80kHz".to_string(),
        "--iono-freq-average=320kHz".to_string(),
    ]);
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
#[serial]
fn frequency_averaging_explicit_in_out() {
    let PeelParams {
        input_vis_params: InputVisParams { spw: input_spw, .. },
        output_vis_params,
        low_res_spw,
        ..
    } = get_merged_1090008640(vec![
        "--freq-average=80kHz".to_string(),
        "--output-vis-freq-average=160kHz".to_string(),
        "--iono-freq-average=320kHz".to_string(),
    ]);
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
#[serial]
fn frequency_averaging_explicit() {
    let PeelParams {
        input_vis_params: InputVisParams { spw: input_spw, .. },
        output_vis_params,
        low_res_spw,
        ..
    } = get_merged_1090008640(vec![
        "--freq-average=80kHz".to_string(),
        "--output-vis-freq-average=160kHz".to_string(),
        "--iono-freq-average=320kHz".to_string(),
    ]);
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
#[serial]
fn time_averaging_explicit_output_clip() {
    let PeelParams {
        input_vis_params: InputVisParams { time_res, .. },
        output_vis_params,
        iono_time_average_factor,
        ..
    } = get_merged_1090008640(vec!["--output-vis-time-average=4s".to_string()]);
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

// testing that parse() catches invalid number of iono sources
#[test]
fn handle_iono_greater_than_total() {
    let mut args = get_reduced_1090008640();
    args.peel_args = PeelCliArgs {
        num_sources_to_iono_subtract: Some(2), // --iono-sub=2
        ..Default::default()
    };
    args.srclist_args.num_sources = Some(1); // --num-sources=1
    match args.parse() {
        Err(HyperdriveError::Peel(s)) if s.contains("The number of sources to subtract (1) is less than the number of sources to iono subtract (2)") => {} // expected
        Err(e) => panic!("Expected TooManyIonoSub, got {e}"),
        _ => panic!("Expected an error, got Ok"),
    };
}

// integration test for the peel command
#[test]
fn test_peel_writes_files() {
    let temp_dir = tempdir().expect("Couldn't make tempdir");
    let json_file = temp_dir.path().join("peel.json");
    let uvfits_file = temp_dir.path().join("peel.uvfits");

    let params = get_merged_1090008640(vec![
        "--iono-sub=1".to_string(),
        "--num-passes=2".to_string(),
        "--num-loops=1".to_string(),
        "--iono-time-average=8s".to_string(),
        "--iono-freq-average=1280kHz".to_string(),
        "--uvw-min=50m".to_string(),
        "--uvw-max=300m".to_string(),
        "--short-baseline-sigma=40".to_string(),
        "--convergence=0.9".to_string(),
        "--outputs".to_string(),
        uvfits_file.display().to_string(),
        json_file.display().to_string(),
    ]);
    let PeelParams {
        input_vis_params: InputVisParams { spw: input_spw, .. },
        // output_vis_params,
        // iono_timeblocks,
        // iono_time_average_factor,
        low_res_spw,
        peel_weight_params:
            PeelWeightParams {
                uvw_min_metres,
                uvw_max_metres,
                short_baseline_sigma,
            },
        peel_loop_params:
            PeelLoopParams {
                num_passes,
                num_loops,
                convergence,
            },
        num_sources_to_iono_subtract,
        ..
    } = &params;
    assert_abs_diff_eq!(*num_sources_to_iono_subtract, 1);
    assert_abs_diff_eq!(*uvw_min_metres, 50.0);
    assert_abs_diff_eq!(*uvw_max_metres, 300.0);
    assert_abs_diff_eq!(*short_baseline_sigma, 40.0);
    assert_abs_diff_eq!(*convergence, 0.9);
    assert_eq!(num_passes.get(), 2);
    assert_eq!(num_loops.get(), 1);
    assert_eq!(input_spw.freq_res, 40e3);
    assert_eq!(low_res_spw.freq_res, 1280e3);
    params.run().unwrap();
    assert!(uvfits_file.exists());
    assert!(json_file.exists());
}

#[test]
#[serial]
fn iono_sub_selection_defaults_to_off() {
    let PeelParams {
        iono_sub_selection,
        iono_sub_apparent_fluxes,
        ..
    } = get_merged_1090008640(vec![]);
    assert!(!iono_sub_selection.is_enabled());
    assert_eq!(iono_sub_selection.max_scatter_ratio, None);
    assert_eq!(iono_sub_selection.max_predicted_scatter_ratio, None);
    assert_eq!(
        iono_sub_selection.min_timeblocks,
        crate::params::DEFAULT_MIN_TIMEBLOCKS
    );
    // No fluxes are computed unless the predicted criterion asks for them.
    assert!(iono_sub_apparent_fluxes.is_none());
}

#[test]
#[serial]
fn iono_sub_selection_args_are_parsed() {
    let PeelParams {
        iono_sub_selection,
        iono_sub_apparent_fluxes,
        num_sources_to_iono_subtract,
        source_list,
        ..
    } = get_merged_1090008640(vec![
        "--iono-sub-max-scatter=4.5".to_string(),
        "--iono-sub-min-timeblocks=2".to_string(),
        "--iono-sub-max-predicted-scatter=3".to_string(),
    ]);
    assert!(iono_sub_selection.is_enabled());
    assert_eq!(iono_sub_selection.max_scatter_ratio, Some(4.5));
    assert_eq!(iono_sub_selection.min_timeblocks, 2);
    assert_eq!(iono_sub_selection.max_predicted_scatter_ratio, Some(3.0));

    // One apparent flux per iono-subtract candidate, positive, and in the
    // same (descending) order that vetoing sorted the source list by.
    let fluxes = iono_sub_apparent_fluxes.expect("fluxes computed for the predicted criterion");
    assert_eq!(fluxes.len(), num_sources_to_iono_subtract);
    assert_eq!(num_sources_to_iono_subtract, source_list.len());
    assert!(fluxes.iter().all(|f| f.is_finite() && *f > 0.0));
    assert!(
        fluxes.windows(2).all(|w| w[0] >= w[1]),
        "apparent fluxes should be descending: {fluxes:?}"
    );
}

#[test]
#[serial]
fn iono_sub_selection_rejects_bad_values() {
    let args = get_reduced_1090008640();
    let temp_dir = tempdir().expect("Couldn't make tempdir");
    let arg_file = temp_dir.path().join("peel.toml");
    let mut f = File::create(&arg_file).expect("couldn't make file");
    let ser = toml::to_string_pretty(&args).expect("couldn't serialise PeelArgs as toml");
    write!(&mut f, "{ser}").unwrap();

    for (extra, needle) in [
        (
            "--iono-sub-max-scatter=0",
            "--iono-sub-max-scatter must be a positive",
        ),
        (
            "--iono-sub-max-scatter=-2",
            "--iono-sub-max-scatter must be a positive",
        ),
        (
            "--iono-sub-max-predicted-scatter=0",
            "--iono-sub-max-predicted-scatter must be a positive",
        ),
        (
            "--iono-sub-min-timeblocks=1",
            "--iono-sub-min-timeblocks must be at least 2",
        ),
    ] {
        let argv = vec![
            "peel".to_string(),
            arg_file.display().to_string(),
            extra.to_string(),
        ];
        let result = PeelArgs::parse_from(argv).merge().unwrap().parse();
        match result {
            Err(e) => {
                let msg = e.to_string();
                assert!(msg.contains(needle), "{extra}: unexpected error {msg}");
            }
            Ok(_) => panic!("{extra} should have been rejected"),
        }
    }
}

#[test]
#[serial]
fn iono_sub_selection_args_merge_from_file() {
    // Values in the arguments file are used unless overridden on the CLI.
    let mut args = get_reduced_1090008640();
    args.peel_args = PeelCliArgs {
        iono_sub_max_scatter: Some(2.5),
        iono_sub_max_predicted_scatter: Some(6.0),
        ..Default::default()
    };
    let temp_dir = tempdir().expect("Couldn't make tempdir");
    let arg_file = temp_dir.path().join("peel.toml");
    let mut f = File::create(&arg_file).expect("couldn't make file");
    let ser = toml::to_string_pretty(&args).expect("couldn't serialise PeelArgs as toml");
    write!(&mut f, "{ser}").unwrap();
    let argv = vec![
        "peel".to_string(),
        arg_file.display().to_string(),
        "--iono-sub-max-scatter=9".to_string(),
    ];
    let PeelParams {
        iono_sub_selection, ..
    } = PeelArgs::parse_from(argv).merge().unwrap().parse().unwrap();
    assert_eq!(iono_sub_selection.max_scatter_ratio, Some(9.0));
    assert_eq!(iono_sub_selection.max_predicted_scatter_ratio, Some(6.0));
}

// End to end: peel runs with both selection criteria on and still writes one
// JSON entry per iono-subtract source. The test data has a single bright
// source, which is its own "brightest decile", so it must survive both
// criteria and end up fitted (non-default constants).
#[test]
#[serial]
fn test_peel_runs_with_iono_sub_selection() {
    let temp_dir = tempdir().expect("Couldn't make tempdir");
    let json_file = temp_dir.path().join("peel.json");
    let uvfits_file = temp_dir.path().join("peel.uvfits");

    let params = get_merged_1090008640(vec![
        "--iono-sub=1".to_string(),
        "--num-passes=1".to_string(),
        "--num-loops=1".to_string(),
        "--iono-time-average=8s".to_string(),
        "--iono-freq-average=1280kHz".to_string(),
        "--iono-sub-max-scatter=4".to_string(),
        "--iono-sub-min-timeblocks=2".to_string(),
        "--iono-sub-max-predicted-scatter=4".to_string(),
        "--outputs".to_string(),
        uvfits_file.display().to_string(),
        json_file.display().to_string(),
    ]);
    assert!(params.iono_sub_selection.is_enabled());
    params.run().unwrap();
    assert!(uvfits_file.exists());
    assert!(json_file.exists());

    let json: serde_json::Value = serde_json::from_reader(File::open(&json_file).unwrap()).unwrap();
    let entries = json.as_object().expect("JSON object keyed by source name");
    assert_eq!(entries.len(), 1);
    let (_, consts) = entries.iter().next().unwrap();
    let alphas = consts["alphas"].as_array().unwrap();
    let gains = consts["gains"].as_array().unwrap();
    assert!(!alphas.is_empty());
    assert_eq!(alphas.len(), gains.len());
    // The only source is the reference for both criteria, so it is fitted in
    // every timeblock: at least one timeblock must have moved off the
    // defaults.
    assert!(
        alphas.iter().any(|a| a.as_f64().unwrap() != 0.0)
            || gains.iter().any(|g| g.as_f64().unwrap() != 1.0),
        "source should have been fitted: {consts}"
    );
}
