// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Tests for EveryBeam beam responses. These use a small synthetic
//! phased-array measurement set (see `test_files/everybeam/make_mini_ms.py`) with 8
//! stations. The first three stations are rigidly rotated by 0, 30 and 75
//! degrees, and the third station has some flagged X elements.

use std::collections::HashSet;

use approx::assert_abs_diff_eq;
use hifitime::Epoch;
use marlu::{c64, AzEl, Jones, RADec};
use ndarray::prelude::*;

use super::{reference::*, *};

const MS: &str = "test_files/everybeam/mini.ms";

fn get_beam(normalisation: &str) -> EveryBeam {
    EveryBeam::new(
        Path::new(MS),
        Some(8),
        EveryBeamOptions {
            beam_normalisation_mode: Some(normalisation.to_string()),
            ..Default::default()
        },
    )
    .unwrap()
}

fn ref_epoch() -> Epoch {
    Epoch::from_mjd_utc(REF_TIME_MJD_S / 86400.0)
}

fn ref_radecs() -> Vec<RADec> {
    REF_RADECS
        .iter()
        .map(|&(ra, dec)| RADec::from_radians(ra, dec))
        .collect()
}

/// Convert a reference Jones matrix (EveryBeam's convention) to hyperdrive's
/// convention (swap the columns).
fn ref_to_hyperdrive(r: &[[f64; 2]; 4]) -> Jones<f64> {
    let c = |i: usize| c64::new(r[i][0], r[i][1]);
    Jones::from([c(1), c(0), c(3), c(2)])
}

fn check_against_reference(normalisation: &str, reference: &[[[[[f64; 2]; 4]; 3]; 2]; 3]) {
    let beam = get_beam(normalisation);
    let radecs = ref_radecs();
    let stations = [0, 1, 2];
    let mut results = Array3::default((stations.len(), REF_FREQS.len(), radecs.len()));
    beam.inner
        .calc_jones_radec(
            &radecs,
            &REF_FREQS,
            &stations,
            BeamTime {
                epoch: ref_epoch(),
                lst_rad: 0.0,
            },
            results.view_mut(),
        )
        .unwrap();

    for (i_station, station_ref) in reference.iter().enumerate() {
        for (i_freq, freq_ref) in station_ref.iter().enumerate() {
            for (i_dir, dir_ref) in freq_ref.iter().enumerate() {
                let expected = ref_to_hyperdrive(dir_ref);
                let got = results[(i_station, i_freq, i_dir)];
                // EveryBeam calculates in single precision.
                assert_abs_diff_eq!(got, expected, epsilon = 1e-6);
            }
        }
    }
}

#[test]
fn test_load_and_station_count() {
    let beam = get_beam("amplitude");
    assert_eq!(beam.get_num_tiles(), 8);
    assert!(matches!(beam.get_beam_type(), BeamType::EveryBeam));
    assert_eq!(beam.get_beam_file(), Some(Path::new(MS)));
    assert!(beam.get_dipole_delays().is_none());
    assert!(beam.get_dipole_gains().is_none());
    assert_eq!(beam.find_closest_freq(123.456e6), 123.456e6);
}

#[test]
fn test_station_count_mismatch() {
    let result = EveryBeam::new(Path::new(MS), Some(4), EveryBeamOptions::default());
    assert!(matches!(
        result,
        Err(BeamError::StationCountMismatch { ms: 8, tiles: 4 })
    ));
}

#[test]
fn test_bad_ms() {
    let result = EveryBeam::new(
        Path::new("test_files/everybeam/does_not_exist.ms"),
        None,
        EveryBeamOptions::default(),
    );
    assert!(matches!(result, Err(BeamError::EveryBeam(_))));
}

#[test]
fn test_bad_element_model() {
    let result = EveryBeam::new(
        Path::new(MS),
        None,
        EveryBeamOptions {
            element_response_model: Some("not_a_model".to_string()),
            ..Default::default()
        },
    );
    assert!(matches!(result, Err(BeamError::EveryBeam(_))));
}

#[test]
fn test_against_everybeam_python_no_normalisation() {
    check_against_reference("none", &REF_NONE);
}

#[test]
fn test_against_everybeam_python_amplitude_normalisation() {
    check_against_reference("amplitude", &REF_AMPLITUDE);
}

#[test]
fn test_rotated_stations() {
    // At the beam centre (zenith), a station's X dipole responds to the sky's
    // East and North components with cos(rotation) and sin(rotation), because
    // the stations are rigidly rotated. The rows are in each station's own
    // frame, so they aren't de-rotated.
    let beam = get_beam("amplitude");
    let radecs = ref_radecs();
    let mut results = Array3::default((3, 1, 1));
    beam.inner
        .calc_jones_radec(
            &radecs[..1],
            &[REF_FREQS[0]],
            &[0, 1, 2],
            BeamTime {
                epoch: ref_epoch(),
                lst_rad: 0.0,
            },
            results.view_mut(),
        )
        .unwrap();
    for (i_station, rotation_deg) in [0.0_f64, 30.0, 75.0].into_iter().enumerate() {
        let j = results[(i_station, 0, 0)];
        let (s, c) = rotation_deg.to_radians().sin_cos();
        // hyperdrive's columns are (East, North). Allow for the zenith not
        // being exactly at the beam centre after 10 seconds, and for the sign
        // convention of the X dipole.
        assert_abs_diff_eq!(j[0].norm(), c, epsilon = 5e-3);
        assert_abs_diff_eq!(j[1].norm(), s, epsilon = 5e-3);
        assert_abs_diff_eq!(j[2].norm(), s, epsilon = 5e-3);
        assert_abs_diff_eq!(j[3].norm(), c, epsilon = 5e-3);
    }
}

#[test]
fn test_azel_matches_radec() {
    let beam = get_beam("amplitude");
    let radecs = ref_radecs();
    let epoch = ref_epoch();
    // Any LST and latitude will do, as long as the same values are used to
    // generate the AzEls and supplied as the beam time.
    let lst_rad = 1.2;
    let latitude_rad = -0.4682;
    let azels: Vec<AzEl> = radecs
        .iter()
        .map(|r| r.to_hadec(lst_rad).to_azel(latitude_rad))
        .collect();
    let time = BeamTime { epoch, lst_rad };

    let mut expected = Array3::default((1, 1, radecs.len()));
    beam.inner
        .calc_jones_radec(&radecs, &[110e6], &[1], time, expected.view_mut())
        .unwrap();
    let got = beam
        .calc_jones_array(&azels, 110e6, Some(1), latitude_rad, Some(time))
        .unwrap();
    for (g, e) in got.iter().zip(expected.iter()) {
        assert_abs_diff_eq!(*g, *e, epsilon = 1e-6);
    }

    let single = beam
        .calc_jones(azels[1], 110e6, Some(1), latitude_rad, Some(time))
        .unwrap();
    assert_abs_diff_eq!(single, expected[(0, 0, 1)], epsilon = 1e-6);

    // No tile index means the first station.
    let first = beam
        .calc_jones(azels[1], 110e6, None, latitude_rad, Some(time))
        .unwrap();
    let first_explicit = beam
        .calc_jones(azels[1], 110e6, Some(0), latitude_rad, Some(time))
        .unwrap();
    assert_abs_diff_eq!(first, first_explicit);
}

#[test]
fn test_needs_time() {
    let beam = get_beam("amplitude");
    let result = beam.calc_jones(AzEl::from_degrees(0.0, 80.0), 110e6, Some(0), -0.4682, None);
    assert!(matches!(result, Err(BeamError::NeedsTime)));
}

#[test]
fn test_bad_station_index() {
    let beam = get_beam("amplitude");
    let time = BeamTime {
        epoch: ref_epoch(),
        lst_rad: 0.0,
    };
    let result = beam.calc_jones(
        AzEl::from_degrees(0.0, 80.0),
        110e6,
        Some(8),
        -0.4682,
        Some(time),
    );
    assert!(matches!(
        result,
        Err(BeamError::BadTileIndex {
            got: 8,
            num_tiles: 8
        })
    ));
}

#[test]
fn test_many_directions_tiles_freqs() {
    // Use more directions than there are blocks, so that the multi-threaded
    // code is exercised, and check against direction-by-direction results.
    let beam = get_beam("amplitude");
    let epoch = ref_epoch();
    let lst_rad = 1.2;
    let latitude_rad = -0.4682;
    let time = BeamTime { epoch, lst_rad };
    let num_blocks = BLOCKS_PER_THREAD * rayon::current_num_threads();
    let azels: Vec<AzEl> = (0..3 * num_blocks + 7)
        .map(|i| AzEl::from_degrees(i as f64 * 1.7 % 360.0, 30.0 + (i % 60) as f64))
        .collect();
    // Several frequencies (which EveryBeam is given all at once) and stations
    // (out of order, and not all of them).
    let freqs: Vec<f64> = (0..11).map(|i| 100e6 + 1.7e6 * i as f64).collect();
    let tiles = [2, 0, 5];

    let mut results = Array3::default((tiles.len(), freqs.len(), azels.len()));
    beam.calc_jones_tiles_freqs(
        &azels,
        &freqs,
        &tiles,
        latitude_rad,
        Some(time),
        results.view_mut(),
    )
    .unwrap();

    for (i_tile, &tile) in tiles.iter().enumerate() {
        for (i_freq, &freq) in freqs.iter().enumerate() {
            for (i_dir, &azel) in azels.iter().enumerate().step_by(13) {
                let expected = beam
                    .calc_jones(azel, freq, Some(tile), latitude_rad, Some(time))
                    .unwrap();
                // EveryBeam is given all of the frequencies at once here, but
                // one at a time for the expected values.
                assert_abs_diff_eq!(results[(i_tile, i_freq, i_dir)], expected, epsilon = 1e-12);
            }
        }
    }
}

#[test]
fn test_unique_tiles() {
    let beam = get_beam("amplitude");
    // Every station is unique; flagged tiles aren't included.
    let flagged = HashSet::from([1]);
    let (unique_tiles, map) = beam.get_unique_tiles(3, &flagged);
    assert_eq!(unique_tiles, vec![0, 2]);
    assert_eq!(map, vec![0, 0, 1]);

    let (unique_tiles, map) = beam.get_unique_tiles(3, &HashSet::new());
    assert_eq!(unique_tiles, vec![0, 1, 2]);
    assert_eq!(map, vec![0, 1, 2]);
}

/// Unnormalised responses have an arbitrary scale (e.g. skala40_wave's isn't
/// near 1), so source vetoing must use an amplitude-normalised beam.
#[test]
fn test_veto_beam_is_normalised() {
    let options = |norm: Option<&str>| EveryBeamOptions {
        element_response_model: Some("skala40_wave".to_string()),
        beam_normalisation_mode: norm.map(|s| s.to_string()),
        ..Default::default()
    };
    let rms = |beam: &EveryBeam| {
        let mut results = Array3::default((1, 1, 1));
        beam.inner
            .calc_jones_radec(
                &ref_radecs()[..1],
                &REF_FREQS[..1],
                &[0],
                BeamTime {
                    epoch: ref_epoch(),
                    lst_rad: 0.0,
                },
                results.view_mut(),
            )
            .unwrap();
        let j: Jones<f64> = results[(0, 0, 0)];
        (0.5 * j.iter().map(|c| c.norm_sqr()).sum::<f64>()).sqrt()
    };

    for norm in [None, Some("none")] {
        let beam = EveryBeam::new(Path::new(MS), Some(8), options(norm)).unwrap();
        let veto = beam
            .veto_beam()
            .expect("unnormalised beams have a veto beam");
        let veto = veto.get_beam_file();
        assert_eq!(veto, Some(Path::new(MS)));
        let veto_beam = beam.inner.veto.as_ref().unwrap();
        // The first reference direction is the delay (pointing) centre.
        assert_abs_diff_eq!(rms(veto_beam), 1.0, epsilon = 1e-3);
        eprintln!("skala40_wave unnormalised RMS response: {}", rms(&beam));
    }

    // Normalised beams are used for vetoing as they are.
    let beam = EveryBeam::new(Path::new(MS), Some(8), options(Some("amplitude"))).unwrap();
    assert!(beam.veto_beam().is_none());
}

#[test]
fn test_beam_knots() {
    let freqs: Vec<f64> = (0..100).map(|i| 150e6 + 40e3 * i as f64).collect();
    // By default, knots every 1.28 MHz (spanning the channels, plus guard
    // knots).
    let beam = get_beam("amplitude");
    let knots = beam.get_beam_knots(&freqs);
    assert!(knots.windows(2).all(|w| (w[1] - w[0] - 1.28e6).abs() < 1.0));
    assert!(knots[0] < freqs[0] - 1.28e6 && *knots.last().unwrap() > freqs[99] + 1.28e6);
    // A spacing of 0 evaluates every channel.
    let every = |spacing: f64| {
        EveryBeam::new(
            Path::new(MS),
            Some(8),
            EveryBeamOptions {
                beam_normalisation_mode: Some("amplitude".to_string()),
                knot_spacing_hz: Some(spacing),
                ..Default::default()
            },
        )
        .unwrap()
    };
    assert_eq!(every(0.0).get_beam_knots(&freqs), freqs);
    let knots = every(500e3).get_beam_knots(&freqs);
    assert!(knots.windows(2).all(|w| (w[1] - w[0] - 500e3).abs() < 1.0));
}

/// Splining EveryBeam's responses from the knots reproduces the responses at
/// every fine channel. (These test stations are small.)
#[test]
fn test_spline_from_knots_matches_every_channel() {
    use crate::beam::BeamFreqSpline;

    let beam = get_beam("amplitude");
    let time = BeamTime {
        epoch: ref_epoch(),
        lst_rad: 1.2,
    };
    let latitude_rad = -0.4682;
    // From the beam centre (the zenith of REF_RADECS[0]) outwards.
    let centre = ref_radecs()[0];
    let azels: Vec<AzEl> = [0.0f64, 5.0, 20.0, 45.0, 70.0]
        .iter()
        .map(|d| {
            RADec::from_radians(centre.ra, centre.dec + d.to_radians())
                .to_hadec(time.lst_rad)
                .to_azel(latitude_rad)
        })
        .collect();
    let freqs: Vec<f64> = (0..150).map(|i| 100e6 + 40e3 * i as f64).collect();
    let knots = beam.get_beam_knots(&freqs);
    let spline = BeamFreqSpline::new(knots.clone(), &freqs);
    let s = spline.second_derivs.as_ref().unwrap();
    let tiles = [0, 2];
    let mut at_knots = Array3::default((tiles.len(), knots.len(), azels.len()));
    beam.calc_jones_tiles_freqs(
        &azels,
        &knots,
        &tiles,
        latitude_rad,
        Some(time),
        at_knots.view_mut(),
    )
    .unwrap();
    let mut exact = Array3::default((tiles.len(), freqs.len(), azels.len()));
    beam.calc_jones_tiles_freqs(
        &azels,
        &freqs,
        &tiles,
        latitude_rad,
        Some(time),
        exact.view_mut(),
    )
    .unwrap();
    let mut max_err: f64 = 0.0;
    for i_tile in 0..tiles.len() {
        for i_dir in 0..azels.len() {
            let y = at_knots.slice(s![i_tile, .., i_dir]);
            let m: Vec<Jones<f64>> = s
                .outer_iter()
                .map(|row| {
                    row.iter()
                        .zip(y.iter())
                        .fold(Jones::default(), |acc, (&w, &j)| acc + j * w)
                })
                .collect();
            for (i_freq, &(lo, hi, [a, b, c, d])) in spline.interp.iter().enumerate() {
                let interp = y[lo] * a + y[hi] * b + m[lo] * c + m[hi] * d;
                for k in 0..4 {
                    max_err = max_err.max((interp[k] - exact[(i_tile, i_freq, i_dir)][k]).norm());
                }
            }
        }
    }
    // EveryBeam calculates in single precision, so allow for that.
    assert!(max_err < 1e-5, "max error {max_err:e}");
}

/// Blocks of work cover every (station, direction) exactly once, and there
/// aren't many more of them than requested.
#[test]
fn test_blocks_cover_all_work() {
    for (num_stations, num_directions, target) in [
        (1, 1, 1),
        (1, 1, 64),
        (512, 1, 64),
        (1, 8000, 64),
        (3, 8000, 64),
        (512, 8000, 64),
        (100, 7, 64),
        (5, 3, 0),
    ] {
        let blocks = Blocks::new(num_stations, num_directions, target);
        let mut seen = Array2::<u8>::zeros((num_stations, num_directions));
        for i_block in 0..blocks.len() {
            let (s, d) = blocks.ranges(i_block, num_stations, num_directions);
            assert!(!s.is_empty() && !d.is_empty());
            seen.slice_mut(s![s, d]).mapv_inplace(|n| n + 1);
        }
        assert!(
            seen.iter().all(|&n| n == 1),
            "{num_stations} stations, {num_directions} directions, {target} blocks"
        );
        assert!(blocks.len() <= 2 * target.max(1), "{blocks:?}");
    }
}

#[test]
fn test_cluster_directions() {
    let arcsec = |a: f64| (a / 3600.0).to_radians();
    // Two close pairs and a loner, not in declination order.
    let radecs = [
        RADec::from_radians(1.0, -0.5),
        RADec::from_radians(2.0, 0.3),
        RADec::from_radians(1.0 + arcsec(20.0), -0.5 + arcsec(10.0)),
        RADec::from_radians(2.0, 0.3 + arcsec(50.0)),
        RADec::from_radians(0.5, 1.0),
    ];
    let limit = arcsec(60.0);
    let (centres, map) = cluster_directions(&radecs, limit);
    assert_eq!(centres.len(), 3);
    assert_eq!(map[0], map[2]);
    assert_eq!(map[1], map[3]);
    let groups: HashSet<usize> = map.iter().copied().collect();
    assert_eq!(groups.len(), 3);
    for (radec, &i_centre) in radecs.iter().zip(&map) {
        assert!(radec.separation(centres[i_centre]) <= limit);
    }
    // A pair's centre is midway between its members.
    assert_abs_diff_eq!(
        centres[map[1]].dec,
        0.3 + arcsec(25.0),
        epsilon = arcsec(1e-6)
    );

    // Nothing is grouped when the limit is smaller than every separation.
    let (centres, map) = cluster_directions(&radecs, arcsec(1.0));
    assert_eq!(centres.len(), radecs.len());
    for (radec, &i_centre) in radecs.iter().zip(&map) {
        assert!(radec.separation(centres[i_centre]) < arcsec(1e-6));
    }

    // Directions either side of RA = 0 are close together.
    let radecs = [
        RADec::from_radians(arcsec(10.0), 0.1),
        RADec::from_radians(std::f64::consts::TAU - arcsec(10.0), 0.1),
    ];
    let (centres, map) = cluster_directions(&radecs, limit);
    assert_eq!(centres.len(), 1);
    assert_eq!(map, [0, 0]);
    assert!(radecs[0].separation(centres[0]) < arcsec(11.0));
}

#[test]
fn test_proximity_limit_shares_beam_evaluations() {
    let arcsec = |a: f64| (a / 3600.0).to_radians();
    let exact = get_beam("amplitude");
    let shared = EveryBeam::new(
        Path::new(MS),
        Some(8),
        EveryBeamOptions {
            beam_normalisation_mode: Some("amplitude".to_string()),
            proximity_limit_rad: Some(arcsec(60.0)),
            ..Default::default()
        },
    )
    .unwrap();

    // Two pairs of directions 20 and 30 arcsec apart, and one on its own.
    let centre = ref_radecs()[0];
    let radecs = [
        centre,
        RADec::from_radians(centre.ra + 0.1, centre.dec - 0.1),
        RADec::from_radians(centre.ra, centre.dec + arcsec(20.0)),
        RADec::from_radians(centre.ra - 0.2, centre.dec + 0.05),
        RADec::from_radians(centre.ra + 0.1, centre.dec - 0.1 + arcsec(30.0)),
    ];
    let freqs = [100e6, 110e6, 123e6];
    let stations = [0, 3];
    let time = BeamTime {
        epoch: ref_epoch(),
        lst_rad: 0.0,
    };
    let calc = |beam: &EveryBeam| {
        let mut results = Array3::default((stations.len(), freqs.len(), radecs.len()));
        beam.inner
            .calc_jones_radec(&radecs, &freqs, &stations, time, results.view_mut())
            .unwrap();
        results
    };
    let (shared, exact): (Array3<Jones<f64>>, Array3<Jones<f64>>) = (calc(&shared), calc(&exact));

    // The members of a pair get the same response, which isn't either
    // member's own response but is close to both.
    assert_eq!(shared.slice(s![.., .., 0]), shared.slice(s![.., .., 2]));
    assert_eq!(shared.slice(s![.., .., 1]), shared.slice(s![.., .., 4]));
    assert_ne!(shared.slice(s![.., .., 0]), exact.slice(s![.., .., 0]));
    assert_abs_diff_eq!(shared, exact, epsilon = 1e-3);
    // The lone direction is unaffected.
    assert_abs_diff_eq!(
        shared.slice(s![.., .., 3]),
        exact.slice(s![.., .., 3]),
        epsilon = 1e-12
    );
}
