// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

use approx::assert_abs_diff_eq;
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

#[test]
fn test_parse_analytic_beams() {
    for beam_type in ["analytic-mwa_pb", "mwa_pb", "analytic-rts", "rts", "RTS"] {
        let expected = if beam_type.contains("mwa_pb") {
            BeamType::AnalyticMwaPb
        } else {
            BeamType::AnalyticRts
        };

        let beam = BeamArgs {
            delays: Some(vec![0; 16]),
            beam_type: Some(beam_type.to_string()),
            ..Default::default()
        }
        .parse(1, None, None, None)
        .unwrap();
        assert_eq!(beam.get_beam_type(), expected);
        assert!(beam.get_beam_file().is_none());
    }

    let no_delays = BeamArgs {
        beam_type: Some("analytic-mwa_pb".to_string()),
        ..Default::default()
    }
    .parse(1, None, None, None);
    assert!(matches!(
        no_delays,
        Err(crate::beam::BeamError::NoDelays(_))
    ));

    let bad = BeamArgs {
        delays: Some(vec![0; 3]),
        beam_type: Some("analytic-rts".to_string()),
        ..Default::default()
    }
    .parse(1, None, None, None);
    assert!(matches!(bad, Err(BadDelays)));

    let unrecognised = BeamArgs {
        beam_type: Some("banana".to_string()),
        delays: Some(vec![0; 16]),
        ..Default::default()
    }
    .parse(1, None, None, None);
    assert!(matches!(
        unrecognised,
        Err(crate::beam::BeamError::Unrecognised(_))
    ));
}

#[test]
fn test_analytic_gains_and_warnings() {
    use crate::io::read::VisInputType;
    use ndarray::Array2;

    let dead = array![
        [1.0; 16],
        [1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    ];
    let beam = BeamArgs {
        delays: Some(vec![0; 16]),
        beam_type: Some("analytic-mwa_pb".to_string()),
        ..Default::default()
    }
    .parse(2, None, Some(dead.clone()), None)
    .unwrap();
    assert_eq!(beam.get_beam_type(), BeamType::AnalyticMwaPb);
    assert!(!beam
        .get_dipole_gains()
        .unwrap()
        .iter()
        .all(|g| (*g - 1.0).abs() < f64::EPSILON));

    let unity = BeamArgs {
        delays: Some(vec![0; 16]),
        beam_type: Some("analytic-rts".to_string()),
        unity_dipole_gains: true,
        ..Default::default()
    }
    .parse(2, None, Some(dead), None)
    .unwrap();
    assert!(unity
        .get_dipole_gains()
        .unwrap()
        .iter()
        .all(|g| (*g - 1.0).abs() < f64::EPSILON));

    let dipamps = Array2::from_elem((1, 32), 0.9);
    BeamArgs {
        delays: Some(vec![0; 16]),
        beam_type: Some("analytic-mwa_pb".to_string()),
        ..Default::default()
    }
    .parse(1, None, Some(dipamps), None)
    .unwrap();

    let delays_with_32 = {
        let mut d = vec![0; 16];
        d[15] = 32;
        d
    };
    BeamArgs {
        delays: Some(delays_with_32),
        beam_type: Some("analytic-rts".to_string()),
        unity_dipole_gains: true,
        ..Default::default()
    }
    .parse(1, None, None, None)
    .unwrap();

    let full = crate::beam::Delays::Full(array![
        [0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3],
        [0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3]
    ]);
    for input in [
        None,
        Some(VisInputType::MeasurementSet),
        Some(VisInputType::Uvfits),
    ] {
        BeamArgs {
            beam_type: Some("analytic-mwa_pb".to_string()),
            ..Default::default()
        }
        .parse(2, Some(full.clone()), None, input)
        .unwrap();
    }
}
