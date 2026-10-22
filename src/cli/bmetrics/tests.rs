// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

use approx::assert_abs_diff_eq;
use birli::metrics::{MetricsContext, SSINS};
use clap::Parser;
use hdf5::types::VarLenAscii;
use marlu::{
    c32,
    hifitime::{Duration, Epoch},
    ndarray::prelude::*,
    Jones, LatLngHeight, XyzGeodetic,
};
use tempfile::TempDir;

use super::{
    bl_index,
    ins::{IncoherentNoiseSpectrum, InsTelescope, INS_POLS},
    BmetricsArgs,
};
use crate::tests::{get_reduced_1061316544_uvfits, get_reduced_1090008640_raw, DataAsStrings};

/// Tile pairs, autos included, in upper-triangle order.
fn ant_pairs(num_tiles: usize) -> Vec<(usize, usize)> {
    (0..num_tiles)
        .flat_map(|i| (i..num_tiles).map(move |j| (i, j)))
        .collect()
}

fn timestamps(n: usize) -> Vec<Epoch> {
    (0..n)
        .map(|i| Epoch::from_gpst_seconds(1090008640.0 + 2.0 * i as f64 + 1.0))
        .collect()
}

/// A different, deterministic value for every visibility.
fn synthetic_vis(num_times: usize, num_freqs: usize, num_baselines: usize) -> Array3<Jones<f32>> {
    Array3::from_shape_fn((num_times, num_freqs, num_baselines), |(t, f, b)| {
        let x = (t * t + 1) as f32 * (f + 1) as f32 + b as f32;
        Jones::from([
            c32::new(x, 0.5 * x),
            c32::new(2.0 * x, 0.0),
            c32::new(0.0, 3.0 * x),
            c32::new(-4.0 * x, x),
        ])
    })
}

fn read_str(group: &hdf5::Group, name: &str) -> String {
    group
        .dataset(name)
        .unwrap()
        .read_scalar::<VarLenAscii>()
        .unwrap()
        .as_str()
        .to_string()
}

fn read_f64(file: &hdf5::File, name: &str) -> Array3<f64> {
    let ds = file.dataset(name).unwrap();
    let shape = ds.shape();
    Array3::from_shape_vec(
        (shape[0], shape[1], shape[2]),
        ds.read_raw::<f64>().unwrap(),
    )
    .unwrap()
}

#[test]
fn test_bl_index() {
    for (b, (i, j)) in ant_pairs(5).into_iter().enumerate() {
        assert_eq!(bl_index(i, j, 5), b);
    }
}

#[test]
fn test_ins_averages_differenced_cross_amplitudes() {
    let num_tiles = 3;
    let pairs = ant_pairs(num_tiles);
    let vis = synthetic_vis(3, 2, pairs.len());
    let mut flags = Array3::from_elem(vis.dim(), false);
    // Flag baseline (0, 2) in timestep 1, which is in both differenced pairs.
    let flagged_bl = bl_index(0, 2, num_tiles);
    flags[(1, 0, flagged_bl)] = true;
    // Flag every baseline in the last timestep of channel 1.
    flags.slice_mut(s![2, 1, ..]).fill(true);

    let ins = IncoherentNoiseSpectrum::new(vis.view(), flags.view(), &pairs, &timestamps(3));
    assert_eq!(ins.metric.dim(), (2, 2, 4));
    assert_eq!(ins.times.len(), 2);
    assert_abs_diff_eq!(
        ins.times[0].to_gpst_seconds(),
        1090008640.0 + 2.0,
        epsilon = 1e-6
    );

    let crosses: Vec<usize> = pairs
        .iter()
        .enumerate()
        .filter(|(_, (i, j))| i != j)
        .map(|(b, _)| b)
        .collect();
    for t in 0..2 {
        for f in 0..2 {
            let used: Vec<usize> = crosses
                .iter()
                .copied()
                .filter(|&b| !flags[(t, f, b)] && !flags[(t + 1, f, b)])
                .collect();
            assert_abs_diff_eq!(ins.weights[(t, f, 0)], used.len() as f64);
            // INS polarisations are XX, YY, XY, YX.
            for (p, j) in [0, 3, 1, 2].into_iter().enumerate() {
                if used.is_empty() {
                    assert!(ins.metric[(t, f, p)].is_infinite());
                    continue;
                }
                let expected = used
                    .iter()
                    .map(|&b| f64::from((vis[(t + 1, f, b)][j] - vis[(t, f, b)][j]).norm()))
                    .sum::<f64>()
                    / used.len() as f64;
                assert_abs_diff_eq!(ins.metric[(t, f, p)], expected, epsilon = 1e-9);
            }
        }
    }
    // The flagged baseline is missing from both pairs in channel 0, and
    // channel 1 has nothing in the second pair.
    assert_abs_diff_eq!(ins.weights[(0, 0, 0)], 2.0);
    assert_abs_diff_eq!(ins.weights[(1, 0, 0)], 2.0);
    assert_abs_diff_eq!(ins.weights[(0, 1, 0)], 3.0);
    assert_abs_diff_eq!(ins.weights[(1, 1, 0)], 0.0);
}

/// With nothing flagged, the INS z-scores are those of Birli's SSINS metric
/// when it's given only the cross-correlations.
#[test]
fn test_ins_mean_subtracted_matches_birli_ssins() {
    let num_tiles = 4;
    let pairs = ant_pairs(num_tiles);
    let num_times = 5;
    let mut vis = synthetic_vis(num_times, 3, pairs.len());
    // Something noise-like so that the z-scores aren't degenerate.
    vis.indexed_iter_mut().for_each(|((t, f, b), j)| {
        let r = ((t * 31 + f * 17 + b * 7) % 11) as f32;
        *j += Jones::from([c32::new(r, -r); 4]);
    });
    let flags = Array3::from_elem(vis.dim(), false);
    let times = timestamps(num_times);
    let ins = IncoherentNoiseSpectrum::new(vis.view(), flags.view(), &pairs, &times);
    let ms = ins.mean_subtracted();

    let cross_pairs: Vec<(usize, usize)> = pairs.iter().copied().filter(|(i, j)| i != j).collect();
    let cross_idx: Vec<usize> = pairs
        .iter()
        .enumerate()
        .filter(|(_, (i, j))| i != j)
        .map(|(b, _)| b)
        .collect();
    let cross_vis = vis.select(Axis(2), &cross_idx);
    let ctx = MetricsContext {
        antennas: vec![],
        fine_chan_freqs_hz: vec![150e6, 150.04e6, 150.08e6],
        timestamps_s: times.iter().map(|e| e.to_gpst_seconds()).collect(),
        antenna_pairs: cross_pairs,
    };
    let ssins =
        SSINS::new_from_metadata(cross_vis.view(), &ctx, &vec![false; num_times], &[false; 3]);
    for (p, birli_p) in [0, 3, 1, 2].into_iter().enumerate() {
        for t in 0..num_times - 1 {
            for f in 0..3 {
                assert_abs_diff_eq!(
                    ms[(t, f, p)],
                    f64::from(ssins.zscore[(t, f, birli_p)]),
                    epsilon = 1e-3
                );
            }
        }
    }
}

#[test]
fn test_ins_uvflag_h5_layout() {
    let num_tiles = 3;
    let pairs = ant_pairs(num_tiles);
    let vis = synthetic_vis(4, 2, pairs.len());
    let flags = Array3::from_elem(vis.dim(), false);
    let ins = IncoherentNoiseSpectrum::new(vis.view(), flags.view(), &pairs, &timestamps(4));

    let telescope = InsTelescope {
        name: "MWA".to_string(),
        array_position: LatLngHeight::mwa(),
        dut1: Duration::from_seconds(-0.1),
        antenna_names: vec!["Tile011".into(), "Tile012".into(), "Tile013".into()],
        antenna_numbers: vec![11, 12, 13],
        antenna_xyzs: vec![
            XyzGeodetic::default(),
            XyzGeodetic {
                x: 1.0,
                y: 0.0,
                z: 0.0,
            },
            XyzGeodetic {
                x: 0.0,
                y: 100.0,
                z: 10.0,
            },
        ],
    };
    let tmp = TempDir::new().unwrap();
    let path = tmp.path().join("1090008640_SSINS_data.h5");
    ins.write_uvflag_h5(&path, &[150e6, 150.04e6], 40e3, &telescope)
        .unwrap();

    let file = hdf5::File::open(&path).unwrap();
    let header = file.group("Header").unwrap();
    assert_eq!(read_str(&header, "type"), "waterfall");
    assert_eq!(read_str(&header, "mode"), "metric");
    assert_eq!(read_str(&header, "telescope_name"), "MWA");
    assert!(read_str(&header, "history").contains("spectrum_type:cross"));
    let read_i64 = |name: &str| header.dataset(name).unwrap().read_scalar::<i64>().unwrap();
    assert_eq!(read_i64("Ntimes"), 3);
    assert_eq!(read_i64("Nfreqs"), 2);
    assert_eq!(read_i64("Npols"), 4);
    assert_eq!(
        header
            .dataset("polarization_array")
            .unwrap()
            .read_raw::<i32>()
            .unwrap(),
        INS_POLS
    );
    let times_jd = header
        .dataset("time_array")
        .unwrap()
        .read_raw::<f64>()
        .unwrap();
    for (jd, t) in times_jd.iter().zip(ins.times.iter()) {
        assert_abs_diff_eq!(*jd, t.to_jde_utc_days());
    }
    let lsts = header
        .dataset("lst_array")
        .unwrap()
        .read_raw::<f64>()
        .unwrap();
    assert!(lsts
        .iter()
        .all(|l| (0.0..std::f64::consts::TAU).contains(l)));
    // 2 s apart is ~2.0055 s of sidereal time.
    let dlst = (lsts[1] - lsts[0]) * 86400.0 / std::f64::consts::TAU;
    assert_abs_diff_eq!(dlst, 2.0 * 1.0027379, epsilon = 1e-4);

    // Rotating geodetic XYZ into ITRF preserves lengths.
    let ds = header.dataset("antenna_positions").unwrap();
    assert_eq!(ds.shape(), vec![3, 3]);
    let pos = ds.read_raw::<f64>().unwrap();
    assert_abs_diff_eq!(pos[0..3], [0.0; 3], epsilon = 1e-6);
    let len = |p: &[f64]| p.iter().map(|x| x * x).sum::<f64>().sqrt();
    assert_abs_diff_eq!(len(&pos[3..6]), 1.0, epsilon = 1e-6);
    assert_abs_diff_eq!(len(&pos[6..9]), 10100f64.sqrt(), epsilon = 1e-6);
    assert_abs_diff_eq!(pos[8], 10.0, epsilon = 1e-9);

    let data = file.group("Data").unwrap();
    assert!(data.dataset("metric_array").is_ok());
    assert_abs_diff_eq!(read_f64(&file, "Data/metric_array"), ins.metric);
    assert_abs_diff_eq!(read_f64(&file, "Data/weights_array"), ins.weights);
    assert_abs_diff_eq!(read_f64(&file, "Data/weights_square_array"), ins.weights);
}

#[test]
fn test_bmetrics_uvfits_ins() {
    let tmp = TempDir::new().unwrap();
    let ins_path = tmp.path().join("1061316544_SSINS_data.h5");
    let metrics_path = tmp.path().join("metrics.fits");
    let vis = get_reduced_1061316544_uvfits().vis;
    #[rustfmt::skip]
    let args = BmetricsArgs::parse_from([
        "bmetrics",
        "--data", &vis[0],
        "--ins", ins_path.to_str().unwrap(),
        "--output", metrics_path.to_str().unwrap(),
    ]);
    args.run(false).unwrap();
    assert!(metrics_path.exists());

    let file = hdf5::File::open(&ins_path).unwrap();
    // 3 timesteps make 2 differences; 32 channels; 4 pols.
    let metric = read_f64(&file, "Data/metric_array");
    assert_eq!(metric.dim(), (2, 32, 4));
    let weights = read_f64(&file, "Data/weights_array");
    let max_weight = weights.iter().copied().fold(0.0, f64::max);
    // At most every cross-correlation of the 128 tiles.
    assert!(max_weight > 0.0 && max_weight <= 8128.0);
    assert!(metric.iter().zip(weights.iter()).all(|(m, &w)| if w > 0.0 {
        m.is_finite() && *m >= 0.0
    } else {
        m.is_infinite()
    }));
}

#[test]
fn test_bmetrics_needs_an_output() {
    let vis = get_reduced_1061316544_uvfits().vis;
    assert!(BmetricsArgs::try_parse_from(["bmetrics", "--data", &vis[0]]).is_err());
}

/// The raw-data path works with one timestep, but an INS needs two.
#[test]
fn test_bmetrics_raw_one_timestep() {
    let tmp = TempDir::new().unwrap();
    let ins_path = tmp.path().join("ins.h5");
    let metrics_path = tmp.path().join("metrics.fits");
    let DataAsStrings { metafits, vis, .. } = get_reduced_1090008640_raw();
    let data = [metafits.as_str(), vis[0].as_str()];

    let mut args = vec!["bmetrics", "--data"];
    args.extend(data);
    args.extend(["--output", metrics_path.to_str().unwrap()]);
    BmetricsArgs::parse_from(&args).run(false).unwrap();
    assert!(metrics_path.exists());

    args.extend(["--ins", ins_path.to_str().unwrap(), "--ins-corrections"]);
    let err = BmetricsArgs::parse_from(&args).run(false).unwrap_err();
    assert!(err.to_string().contains("at least 2 timesteps"), "{err}");
    assert!(!ins_path.exists());
}

#[test]
fn test_apparent_lst_matches_astropy() {
    // astropy 7.1: t = Time(1090008642, format="gps", scale="utc", location=MWA);
    // t.delta_ut1_utc = -0.31295757; t.sidereal_time("apparent")
    let lst = super::ins::apparent_lst(
        Epoch::from_gpst_seconds(1090008642.0),
        Duration::from_seconds(-0.31295757),
        LatLngHeight::mwa().longitude_rad,
    );
    assert_abs_diff_eq!(lst, 6.262100753245431, epsilon = 1e-8);
}
