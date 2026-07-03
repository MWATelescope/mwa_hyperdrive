// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

use approx::assert_abs_diff_eq;
use hdf5_metno::File as Hdf5File;
use ndarray::array;
use tempfile::tempdir;

use super::BeamArgs;
use crate::beam::{
    BeamError::{BadDelays, NoBeamFile},
    BeamType,
};

fn make_test_cma21_feko_cube() -> std::path::PathBuf {
    let dir = tempdir().unwrap();
    let path = dir.path().join("cma21_feko_test.h5");
    let file = Hdf5File::create(&path).unwrap();
    file.new_dataset_builder()
        .with_data(&[120e6_f64])
        .create("freq_hz")
        .unwrap();
    file.new_dataset_builder()
        .with_data(&[46.0_f64, 47.0, 48.0])
        .create("theta_deg")
        .unwrap();
    file.new_dataset_builder()
        .with_data(&[89.0_f64, 90.0, 91.0])
        .create("phi_deg")
        .unwrap();
    let beam = vec![1.0, 1.0, 1.0, 1.0, 4.0, 1.0, 1.0, 9.0, 1.0];
    file.new_dataset::<f64>()
        .shape((1, 3, 3))
        .create("beam_xx")
        .unwrap()
        .write_raw(&beam)
        .unwrap();
    let persisted = dir.keep();
    persisted.join("cma21_feko_test.h5")
}

#[test]
fn test_handle_delays() {
    let args = BeamArgs {
        // only 3 delays instead of 16 expected
        delays: Some((0..3).collect::<Vec<u32>>()),
        beam_type: Some("fee".to_string()),
        ..Default::default()
    };

    let result = args.parse(1, None, None, None);
    assert!(result.is_err());
    assert!(matches!(result, Err(BadDelays)));

    let args = BeamArgs {
        // delays > 32
        delays: Some((20..36).collect::<Vec<u32>>()),
        beam_type: Some("fee".to_string()),
        ..Default::default()
    };
    let result = args.parse(1, None, None, None);

    assert!(result.is_err());
    assert!(matches!(result, Err(BadDelays)));

    let delays = (0..16).collect::<Vec<u32>>();
    let args = BeamArgs {
        // delays > 32
        delays: Some(delays.clone()),
        beam_type: Some("fee".to_string()),
        ..Default::default()
    };
    let result = args.parse(1, None, None, None);

    assert!(result.is_ok(), "result={:?} not Ok", result.err().unwrap());

    let fee_beam = result.unwrap();
    assert_eq!(fee_beam.get_beam_type(), BeamType::FEE);
    let beam_delays = fee_beam
        .get_dipole_delays()
        .expect("expected some delays to be provided from the FEE beam!");
    // Each row of the delays should be the same as the 16 input values.
    for row in beam_delays.outer_iter() {
        assert_eq!(row.as_slice().unwrap(), delays);
    }
}

#[test]
fn test_unity_dipole_gains() {
    let args = BeamArgs {
        delays: Some(vec![0; 16]),
        beam_type: Some("fee".to_string()),
        ..Default::default()
    };

    // Let one of the dipoles be dead.
    let dipole_gains = array![
        [1.0; 16],
        [1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    ];
    let beam = args.parse(2, None, Some(dipole_gains), None).unwrap();
    assert_eq!(beam.get_beam_type(), BeamType::FEE);
    let beam_gains = beam.get_dipole_gains().unwrap();

    // We should find that not all dipole gains are 1.
    assert!(!beam_gains.iter().all(|g| (*g - 1.0).abs() < f64::EPSILON));

    // Now ignore dead dipoles.
    let args = BeamArgs {
        delays: Some(vec![0; 16]),
        beam_type: Some("fee".to_string()),
        unity_dipole_gains: true,
        ..Default::default()
    };

    let dipole_gains = array![[1.0; 16], [1.0; 16]];
    let beam = args.parse(2, None, Some(dipole_gains), None).unwrap();
    assert_eq!(beam.get_beam_type(), BeamType::FEE);
    let beam_gains = beam.get_dipole_gains().unwrap();

    // We expect all gains to be 1s, as we're ignoring dead dipoles.
    assert!(beam_gains.iter().all(|g| (*g - 1.0).abs() < f64::EPSILON));
    // Verify that there are no dead dipoles in the delays.
    assert!(beam.get_dipole_delays().unwrap().iter().all(|d| *d != 32));
}

#[test]
fn test_explicit_cma21_stub_beam_type() {
    let args = BeamArgs {
        beam_type: Some("cma21-stub".to_string()),
        ..Default::default()
    };
    let beam = args.parse(40, None, None, None).unwrap();
    assert_eq!(beam.get_beam_type(), BeamType::Cma21Stub);
}

#[test]
fn test_explicit_cma21_gaussian_beam_type() {
    let args = BeamArgs {
        beam_type: Some("cma21-gaussian".to_string()),
        ..Default::default()
    };
    let beam = args.parse(40, None, None, None).unwrap();
    assert_eq!(beam.get_beam_type(), BeamType::Cma21Gaussian);
}

#[test]
fn test_explicit_cma21_feko_cube_requires_file() {
    let args = BeamArgs {
        beam_type: Some("cma21-feko-cube".to_string()),
        ..Default::default()
    };
    let result = args.parse(40, None, None, None);
    assert!(matches!(result, Err(NoBeamFile("cma21-feko-cube"))));
}

#[test]
fn test_explicit_cma21_feko_cube_beam_type() {
    let args = BeamArgs {
        beam_type: Some("cma21-feko-cube".to_string()),
        beam_file: Some(make_test_cma21_feko_cube()),
        ..Default::default()
    };
    let beam = args.parse(40, None, None, None).unwrap();
    assert_eq!(beam.get_beam_type(), BeamType::Cma21FekoCube);
}

#[test]
fn test_aman_dipole_gains() {
    let f = |metafits| {
        let metafits = mwalib::MetafitsContext::new(metafits, None).unwrap();
        let delays = crate::metafits::get_dipole_delays(&metafits);
        let gains = crate::metafits::get_dipole_gains(&metafits);
        (delays, gains)
    };

    let (vanilla_delays, vanilla_gains) = f("test_files/1120082744/1120082744.metafits");
    let (dipamps_delays, dipamps_gains) = f("test_files/1120082744/1120082744_DipAmps.metafits");
    assert_eq!(vanilla_delays, dipamps_delays);
    assert_ne!(vanilla_gains, dipamps_gains);

    // First X dipole for Tile011
    assert_abs_diff_eq!(dipamps_gains[(0, 0)] as f32, 0.89985347);
    // First Y dipole for Tile011
    assert_abs_diff_eq!(dipamps_gains[(0, 16)] as f32, 0.8930142);
}
