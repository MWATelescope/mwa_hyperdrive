// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

use approx::assert_abs_diff_eq;
use hifitime::Epoch;
use marlu::{c64, Jones};
use ndarray::prelude::*;
use vec1::{vec1, Vec1};

use super::*;
use crate::{pfb_gains::PfbFlavour, vis_io::read::RawDataCorrections};

fn make_solutions() -> CalibrationSolutions {
    let num_timeblocks = 2;
    let num_tiles = 128;
    let num_baselines = (num_tiles * (num_tiles - 1)) / 2;
    let num_chanblocks = 768;
    let di_jones = Array1::range(
        1.0,
        (num_timeblocks * num_tiles * num_chanblocks + 1) as _,
        1.0,
    );
    let mut di_jones = di_jones
        .into_shape((num_timeblocks, num_tiles, num_chanblocks))
        .unwrap()
        .mapv(|v| {
            Jones::from([
                c64::new(1.01, 1.02),
                c64::new(1.03, 1.04),
                c64::new(1.05, 1.06),
                c64::new(1.07, 1.08),
            ]) * v
        });
    // Sprinkle some flags.
    let flagged_tiles = 3..5;
    let flagged_chanblocks = 5..8;
    di_jones
        .slice_mut(s![.., flagged_tiles.clone(), ..])
        .fill(Jones::nan());
    di_jones
        .slice_mut(s![.., .., flagged_chanblocks.clone()])
        .fill(Jones::nan());

    CalibrationSolutions {
        di_jones,
        flagged_tiles: flagged_tiles.into_iter().collect(),
        flagged_chanblocks: flagged_chanblocks.into_iter().map(|i| i as u16).collect(),
        chanblock_freqs: Vec1::try_from_vec(
            (0..num_chanblocks)
                .map(|i| ((i + 10) * 10) as f64)
                .collect(),
        )
        .ok(),
        obsid: Some(1090008640),
        start_timestamps: Some(vec1![
            Epoch::from_gpst_seconds(1090008640.0),
            Epoch::from_gpst_seconds(1090008650.0),
        ]),
        end_timestamps: Some(vec1![
            Epoch::from_gpst_seconds(1090008650.0),
            Epoch::from_gpst_seconds(1090008660.0),
        ]),
        average_timestamps: Some(vec1![
            Epoch::from_gpst_seconds(1090008645.0),
            Epoch::from_gpst_seconds(1090008655.0),
        ]),
        max_iterations: Some(30),
        stop_threshold: Some(1e-10),
        min_threshold: Some(1e-5),
        raw_data_corrections: Some(RawDataCorrections {
            pfb_flavour: PfbFlavour::Cotter2014,
            digital_gains: true,
            cable_length: false,
            geometric: false,
        }),
        tile_names: Vec1::try_from_vec((0..num_tiles).map(|i| format!("tile{i:03}")).collect())
            .ok(),
        dipole_gains: Some(Array2::ones((num_tiles, 32)).into_shared()),
        dipole_delays: Some(Array2::zeros((num_tiles, 16)).into_shared()),
        beam_file: None,
        calibration_results: Some(Array2::from_elem((num_timeblocks, num_chanblocks), 1e-6)),
        baseline_weights: Vec1::try_from_vec((0..num_baselines).map(|i| i as _).collect()).ok(),
        uvw_min: Some(82.0),
        uvw_max: Some(f64::INFINITY),
        freq_centroid: Some(182e6),
        modeller: Some("CPU".to_string()),
    }
}

#[test]
fn test_write_and_read_hyperdrive_solutions() {
    let sols = make_solutions();

    let tmp_file = tempfile::NamedTempFile::new().expect("Couldn't make tmp file");
    let result = hyperdrive::write(&sols, tmp_file.path());
    assert!(result.is_ok(), "{:?}", result.err());
    result.unwrap();

    let result = hyperdrive::read(tmp_file.path());
    assert!(result.is_ok(), "{:?}", result.err());
    let sols_from_disk = result.unwrap();

    assert_eq!(sols.di_jones.dim(), sols_from_disk.di_jones.dim());
    // Can't use assert_abs_diff_eq on the whole array, because it rejects NaN
    // equality.
    sols.di_jones
        .into_iter()
        .zip(sols_from_disk.di_jones.into_iter())
        .for_each(|(expected, result)| {
            if expected.any_nan() {
                assert!(result.any_nan());
            } else {
                assert_abs_diff_eq!(expected, result);
            }
        });

    assert_eq!(
        sols_from_disk.raw_data_corrections.unwrap().pfb_flavour,
        PfbFlavour::Cotter2014
    );
    assert!(sols_from_disk.raw_data_corrections.unwrap().digital_gains);
    assert!(!sols_from_disk.raw_data_corrections.unwrap().cable_length);
    assert!(!sols_from_disk.raw_data_corrections.unwrap().geometric);

    assert_eq!(sols_from_disk.max_iterations, sols.max_iterations);
    assert!(sols_from_disk.stop_threshold.is_some());
    let disk_stop_threshold = sols_from_disk.stop_threshold.unwrap();
    assert_abs_diff_eq!(disk_stop_threshold, sols.stop_threshold.unwrap());
    assert!(sols_from_disk.min_threshold.is_some());
    let disk_min_threshold = sols_from_disk.min_threshold.unwrap();
    assert_abs_diff_eq!(disk_min_threshold, sols.min_threshold.unwrap());

    assert_eq!(sols_from_disk.flagged_tiles[..], [3, 4]);
    assert_eq!(sols_from_disk.flagged_chanblocks[..], [5, 6, 7]);

    assert!(sols_from_disk.start_timestamps.is_some());
    let disk_start_timestamps = sols_from_disk.start_timestamps.unwrap();
    assert_eq!(disk_start_timestamps.len(), 2);
    assert_abs_diff_eq!(disk_start_timestamps[0].to_gpst_seconds(), 1090008640.0);
    assert_abs_diff_eq!(disk_start_timestamps[1].to_gpst_seconds(), 1090008650.0);

    assert!(sols_from_disk.end_timestamps.is_some());
    let disk_end_timestamps = sols_from_disk.end_timestamps.unwrap();
    assert_eq!(disk_end_timestamps.len(), 2);
    assert_abs_diff_eq!(disk_end_timestamps[0].to_gpst_seconds(), 1090008650.0);
    assert_abs_diff_eq!(disk_end_timestamps[1].to_gpst_seconds(), 1090008660.0);

    assert!(sols_from_disk.average_timestamps.is_some());
    let disk_average_timestamps = sols_from_disk.average_timestamps.unwrap();
    assert_eq!(disk_average_timestamps.len(), 2);
    assert_abs_diff_eq!(disk_average_timestamps[0].to_gpst_seconds(), 1090008645.0);
    assert_abs_diff_eq!(disk_average_timestamps[1].to_gpst_seconds(), 1090008655.0);

    assert!(sols_from_disk.chanblock_freqs.is_some());
    let disk_freqs = sols_from_disk.chanblock_freqs.unwrap();
    assert_abs_diff_eq!(disk_freqs[..], sols.chanblock_freqs.unwrap());

    assert!(sols_from_disk.baseline_weights.is_some());
    let disk_baseline_weights = sols_from_disk.baseline_weights.unwrap();
    assert_abs_diff_eq!(disk_baseline_weights[..], sols.baseline_weights.unwrap());
}

#[test]
fn test_write_and_read_hyperdrive_solutions_no_metadata() {
    let CalibrationSolutions {
        di_jones,
        flagged_tiles,
        flagged_chanblocks,
        chanblock_freqs: _,
        obsid: _,
        start_timestamps: _,
        end_timestamps: _,
        average_timestamps: _,
        max_iterations: _,
        stop_threshold: _,
        min_threshold: _,
        raw_data_corrections: _,
        tile_names: _,
        dipole_gains: _,
        dipole_delays: _,
        beam_file: _,
        calibration_results: _,
        baseline_weights: _,
        uvw_min: _,
        uvw_max: _,
        freq_centroid: _,
        modeller: _,
    } = make_solutions();
    // Only di_jones, flagged_tiles and flagged_chanblocks are required.
    let sols = CalibrationSolutions {
        di_jones,
        flagged_tiles,
        flagged_chanblocks,
        ..Default::default()
    };

    let tmp_file = tempfile::NamedTempFile::new().expect("Couldn't make tmp file");
    let result = hyperdrive::write(&sols, tmp_file.path());
    assert!(result.is_ok(), "{:?}", result.err());
    result.unwrap();

    let result = hyperdrive::read(tmp_file.path());
    assert!(result.is_ok(), "{:?}", result.err());
    let sols_from_disk = result.unwrap();

    assert_eq!(sols.di_jones.dim(), sols_from_disk.di_jones.dim());
    // Can't use assert_abs_diff_eq on the whole array, because it rejects NaN
    // equality.
    sols.di_jones
        .into_iter()
        .zip(sols_from_disk.di_jones.into_iter())
        .for_each(|(expected, result)| {
            if expected.any_nan() {
                assert!(result.any_nan());
            } else {
                assert_abs_diff_eq!(expected, result);
            }
        });

    assert_eq!(
        sols_from_disk.raw_data_corrections,
        sols.raw_data_corrections
    );
    assert_eq!(sols_from_disk.max_iterations, sols.max_iterations);
    assert_eq!(sols_from_disk.stop_threshold, sols.stop_threshold);
    assert_eq!(sols_from_disk.min_threshold, sols_from_disk.min_threshold);

    assert_eq!(sols_from_disk.flagged_tiles, sols.flagged_tiles);
    assert_eq!(sols_from_disk.flagged_chanblocks, sols.flagged_chanblocks);

    assert_eq!(sols_from_disk.start_timestamps, sols.start_timestamps);
    assert_eq!(sols_from_disk.end_timestamps, sols.end_timestamps);
    assert_eq!(sols_from_disk.average_timestamps, sols.average_timestamps);

    assert_eq!(sols_from_disk.chanblock_freqs, sols.chanblock_freqs);
    assert_eq!(sols_from_disk.baseline_weights, sols.baseline_weights);
}

#[test]
fn test_read_bad_hyperdrive_solutions() {
    // TODO: This test is not exhaustive.

    let mut sols = make_solutions();
    // There's currently nothing stopping someone from writing nonsensical
    // solutions, but hyperdrive will complain upon reading them.
    sols.baseline_weights = Some(vec1![10.0, 1.0]);
    let tmp_file = tempfile::NamedTempFile::new().expect("Couldn't make tmp file");
    let result = hyperdrive::write(&sols, tmp_file.path());
    assert!(result.is_ok(), "{:?}", result.err());
    result.unwrap();

    let result = hyperdrive::read(tmp_file.path());
    assert!(result.is_err());
    let e = result.err().unwrap();
    assert!(matches!(e, SolutionsReadError::BadShape { .. }));
    match e {
        SolutionsReadError::BadShape {
            thing,
            expected,
            actual,
        } => {
            assert!(thing.contains("BASELINES"));
            assert_eq!(expected, 8128);
            assert_eq!(actual, 2);
        }
        _ => unreachable!(),
    }
}

#[test]
fn test_write_and_read_ao_solutions() {
    let sols = make_solutions();
    let tmp_file = tempfile::NamedTempFile::new().expect("Couldn't make tmp file");
    let result = ao::write(&sols, tmp_file.path());
    assert!(result.is_ok());
    result.unwrap();

    let result = ao::read(tmp_file.path());
    assert!(result.is_ok());
    let sols_from_disk = result.unwrap();

    assert_eq!(sols.di_jones.dim(), sols_from_disk.di_jones.dim());
    // Can't use assert_abs_diff_eq on the whole array, because it rejects NaN
    // equality.
    sols.di_jones
        .into_iter()
        .zip(sols_from_disk.di_jones.into_iter())
        .for_each(|(expected, result)| {
            if expected.any_nan() {
                assert!(result.any_nan());
            } else {
                assert_abs_diff_eq!(expected, result);
            }
        });

    assert!(sols_from_disk.start_timestamps.is_some());
    let disk_start_timestamps = sols_from_disk.start_timestamps.unwrap();
    // Only one start and end; limited by the format.
    assert_eq!(disk_start_timestamps.len(), 1);
    assert_abs_diff_eq!(disk_start_timestamps[0].to_gpst_seconds(), 1090008640.0);

    assert!(sols_from_disk.end_timestamps.is_some());
    let disk_end_timestamps = sols_from_disk.end_timestamps.unwrap();
    assert_eq!(disk_end_timestamps.len(), 1);
    assert_abs_diff_eq!(disk_end_timestamps[0].to_gpst_seconds(), 1090008660.0);

    // Not technically part of the format, but hyperdrive populates it.
    assert!(sols_from_disk.average_timestamps.is_some());
    let disk_average_timestamps = sols_from_disk.average_timestamps.unwrap();
    assert_eq!(disk_average_timestamps.len(), 1);
    assert_abs_diff_eq!(disk_average_timestamps[0].to_gpst_seconds(), 1090008650.0);
}
