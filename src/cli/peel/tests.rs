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
fn peel_cannot_exceed_iono() {
    let mut args = get_reduced_1090008640();
    args.peel_args = PeelCliArgs {
        num_sources_to_iono_subtract: Some(1),
        num_sources_to_peel: Some(2),
        ..Default::default()
    };
    match args.parse() {
        Err(HyperdriveError::Peel(s)) if s.contains("cannot exceed the number of sources to iono subtract") => {}
        Err(e) => panic!("Expected TooManyPeel error, got {e}"),
        Ok(_) => panic!("Expected an error, got Ok"),
    }
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
fn test_peel_writes_di_per_source() {
    use crate::CalibrationSolutions;
    use ndarray::Axis;

    let temp_dir = tempdir().expect("Couldn't make tempdir");
    let di_dir = temp_dir.path().join("di_solutions");

    // Request DI solutions per source for 2 sources
    let params = get_merged_1090008640(vec![
        "--iono-sub=2".to_string(),
        format!("--di-per-source-dir={}", di_dir.display()),
    ]);

    // Run peel
    params.run().unwrap();

    // Verify directory and files exist
    assert!(di_dir.exists(), "DI per-source directory not created");
    let mut fits_files: Vec<_> = std::fs::read_dir(&di_dir)
        .unwrap()
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().and_then(|s| s.to_str()) == Some("fits"))
        .collect();
    fits_files.sort();
    assert_eq!(fits_files.len(), 2, "Expected 2 per-source DI files");

    // Quick sanity check: can read one and it has one timeblock
    let sols = CalibrationSolutions::read_solutions_from_ext(&fits_files[0], Option::<&std::path::Path>::None)
        .expect("able to read per-source DI solutions");
    assert_eq!(sols.di_jones.len_of(Axis(0)), 1, "Per-source DI should have 1 timeblock");
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
