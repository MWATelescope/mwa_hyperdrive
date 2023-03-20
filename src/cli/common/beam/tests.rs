// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

use ndarray::array;

use super::BeamArgs;
use crate::beam::{BeamError::BadDelays, BeamType};

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
