// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! QA metrics of calibration solutions, after mwa_qa's `calqa`, but compared
//! within cable flavours so that a tile is judged against tiles with the same
//! signal path.
//!
//! All per-tile arrays are indexed by the metafits tile index of
//! [`Tiles`](crate::cli::crosstalk_plot::tiles::Tiles), with NaN for tiles
//! that are flagged or not in the solutions.

use std::collections::HashMap;
use std::f64::consts::TAU;
use std::path::Path;

use log::warn;
use marlu::{c64, constants::VEL_C};

use super::SolutionsQaError;
use crate::cli::crosstalk_plot::tiles::Tiles;
use crate::solutions::CalibrationSolutions;

/// The polarisations the metrics are made for: the diagonals of the Jones
/// matrices.
pub(super) const POLS: [&str; 2] = ["XX", "YY"];
const POL_INDICES: [usize; 2] = [0, 3];

/// Modified z-scores beyond this are outliers (Iglewicz and Hoaglin).
pub(super) const Z_CUT: f64 = 3.5;

/// The degree of the polynomials fitted to each tile's gain amplitude and
/// phase before taking the delay spectrum, so that only structure on scales
/// finer than the band is left.
const DETREND_DEGREE: usize = 4;

/// The velocity factor assumed for coax cable flavours when estimating the
/// delay of a cable reflection.
const COAX_VELOCITY_FACTOR: f64 = 0.85;

/// The metrics of one solutions file.
pub(super) struct ObsQa {
    /// The obsid, or the file name if there isn't one.
    pub(super) label: String,
    /// Chanblock centre frequencies (Hz), or NaN if the file doesn't say.
    pub(super) freqs_hz: Vec<f64>,
    /// |gain| relative to same-flavour tiles in the same chanblock, divided by
    /// the tile's own median: `[pol][tile][chanblock]`.
    pub(super) norm_amp: [Vec<Vec<f64>>; 2],
    /// The RMS of `norm_amp - 1` over chanblocks: `[pol][tile]`.
    pub(super) amp_rms: [Vec<f64>; 2],
    /// `amp_rms` as a modified z-score within each flavour: `[pol][tile]`.
    pub(super) amp_z: [Vec<f64>; 2],
    /// The non-negative delays of the delay spectra (ns).
    pub(super) delays_ns: Vec<f64>,
    /// The power of the detrended gains' delay spectrum (dB): `[pol][tile][delay]`.
    pub(super) delay_db: [Vec<Vec<f64>>; 2],
    /// The final precision of each chanblock's solution (first timeblock),
    /// NaN where there is no solution.
    pub(super) precision: Option<Vec<f64>>,
    pub(super) stop_threshold: Option<f64>,
    pub(super) min_threshold: Option<f64>,
}

impl ObsQa {
    pub(super) fn num_chanblocks(&self) -> usize {
        self.freqs_hz.len()
    }
}

/// Read a solutions file and compute its metrics. Only the first timeblock is
/// used.
pub(super) fn read(file: &Path, tiles: &Tiles) -> Result<ObsQa, SolutionsQaError> {
    let sols = CalibrationSolutions::read_solutions_from_ext_inner(file, None)?;
    let label = sols.obsid.map(|o| o.to_string()).unwrap_or_else(|| {
        file.file_stem()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_default()
    });
    let (num_timeblocks, num_sol_tiles, num_chanblocks) = sols.di_jones.dim();
    if num_timeblocks > 1 {
        warn!(
            "{}: only the first of {num_timeblocks} timeblocks is used",
            file.display()
        );
    }

    // Which metafits tile each solutions tile is.
    let sol_to_tile: Vec<Option<usize>> = match &sols.tile_names {
        Some(names) => names.iter().map(|n| tiles.index_of(n.trim())).collect(),
        None if num_sol_tiles == tiles.len() => (0..num_sol_tiles).map(Some).collect(),
        None => {
            return Err(SolutionsQaError::TileCountMismatch {
                file: file.to_path_buf(),
                num_sol_tiles,
                num_metafits_tiles: tiles.len(),
            })
        }
    };
    let num_missing = sol_to_tile.iter().filter(|t| t.is_none()).count();
    if num_missing > 0 {
        warn!(
            "{}: {num_missing} solution tiles aren't in the metafits and are left out",
            file.display()
        );
    }

    // Gains per pol, in metafits tile order; NaN where there is no solution.
    let mut gains = [
        vec![vec![c64::new(f64::NAN, f64::NAN); num_chanblocks]; tiles.len()],
        vec![vec![c64::new(f64::NAN, f64::NAN); num_chanblocks]; tiles.len()],
    ];
    for (i_sol, tile) in sol_to_tile.iter().enumerate() {
        let Some(tile) = *tile else { continue };
        if sols.flagged_tiles.contains(&i_sol) {
            continue;
        }
        for (i_chan, j) in sols
            .di_jones
            .slice(ndarray::s![0, i_sol, ..])
            .iter()
            .enumerate()
        {
            for (pol, &i_pol) in POL_INDICES.iter().enumerate() {
                gains[pol][tile][i_chan] = j[i_pol];
            }
        }
    }

    let freqs_hz = match &sols.chanblock_freqs {
        Some(f) if f.len() == num_chanblocks => f.to_vec(),
        _ => vec![f64::NAN; num_chanblocks],
    };
    let flavours: Vec<&str> = tiles.tiles.iter().map(|t| t.flavour.as_str()).collect();

    let norm_amp = [0, 1].map(|pol| {
        let amps: Vec<Vec<f64>> = gains[pol]
            .iter()
            .map(|g| g.iter().map(|g| g.norm()).collect())
            .collect();
        normalised_amps(&amps, &flavours)
    });
    let amp_rms = [0, 1].map(|pol| {
        norm_amp[pol]
            .iter()
            .map(|a| rms_about_one(a))
            .collect::<Vec<f64>>()
    });
    let amp_z = [0, 1].map(|pol: usize| modified_z(&amp_rms[pol], &flavours));

    let (delays_ns, delay_db) = match chanblock_positions(&freqs_hz) {
        Some((positions, df_hz)) => {
            let n = positions.last().map(|&p| p + 1).unwrap_or(0);
            let delays_ns = (0..=n / 2)
                .map(|k| k as f64 / (n as f64 * df_hz) * 1e9)
                .collect();
            let db = [0, 1].map(|pol| {
                gains[pol]
                    .iter()
                    .map(|g| delay_spectrum_db(g, &positions, n))
                    .collect()
            });
            (delays_ns, db)
        }
        None => {
            warn!(
                "{}: no chanblock frequencies, so no delay spectra",
                file.display()
            );
            (vec![], [vec![], vec![]])
        }
    };

    let precision = sols.calibration_results.as_ref().map(|r| r.row(0).to_vec());

    Ok(ObsQa {
        label,
        freqs_hz,
        norm_amp,
        amp_rms,
        amp_z,
        delays_ns,
        delay_db,
        precision,
        stop_threshold: sols.stop_threshold,
        min_threshold: sols.min_threshold,
    })
}

/// The median of the finite values, or NaN if there are none.
pub(super) fn median(values: impl IntoIterator<Item = f64>) -> f64 {
    let mut v: Vec<f64> = values.into_iter().filter(|v| v.is_finite()).collect();
    if v.is_empty() {
        return f64::NAN;
    }
    v.sort_unstable_by(f64::total_cmp);
    let mid = v.len() / 2;
    if v.len() % 2 == 1 {
        v[mid]
    } else {
        0.5 * (v[mid - 1] + v[mid])
    }
}

/// Amplitudes (`[tile][chanblock]`) divided by the median of the tiles of the
/// same flavour in each chanblock, then by each tile's own median, so that
/// the bandpass shared by a flavour cancels and only differences between
/// tiles with the same signal path remain. Zero amplitudes count as missing.
pub(super) fn normalised_amps(amps: &[Vec<f64>], flavours: &[&str]) -> Vec<Vec<f64>> {
    let num_chans = amps.first().map(|a| a.len()).unwrap_or(0);
    let mut by_flavour: HashMap<&str, Vec<usize>> = HashMap::new();
    for (tile, f) in flavours.iter().enumerate() {
        by_flavour.entry(f).or_default().push(tile);
    }
    let valid = |a: f64| a.is_finite() && a > 0.0;
    let mut out = vec![vec![f64::NAN; num_chans]; amps.len()];
    for members in by_flavour.values() {
        for chan in 0..num_chans {
            let m = median(members.iter().map(|&t| amps[t][chan]).filter(|&a| valid(a)));
            for &t in members {
                let a = amps[t][chan];
                if valid(a) && m > 0.0 {
                    out[t][chan] = a / m;
                }
            }
        }
    }
    for row in &mut out {
        let m = median(row.iter().copied());
        for v in row.iter_mut() {
            *v /= m;
        }
    }
    out
}

/// The RMS of `values - 1` over the finite values, or NaN if there are none.
pub(super) fn rms_about_one(values: &[f64]) -> f64 {
    let (sum, n) = values
        .iter()
        .filter(|v| v.is_finite())
        .fold((0.0, 0usize), |(s, n), v| (s + (v - 1.0).powi(2), n + 1));
    if n == 0 {
        f64::NAN
    } else {
        (sum / n as f64).sqrt()
    }
}

/// Modified z-scores, 0.6745 (x - median) / MAD, computed within each group.
/// A group with fewer than 3 finite values, or no spread, gets NaN.
pub(super) fn modified_z(values: &[f64], groups: &[&str]) -> Vec<f64> {
    let mut by_group: HashMap<&str, Vec<usize>> = HashMap::new();
    for (i, g) in groups.iter().enumerate() {
        by_group.entry(g).or_default().push(i);
    }
    let mut z = vec![f64::NAN; values.len()];
    for members in by_group.values() {
        let finite: Vec<f64> = members
            .iter()
            .map(|&i| values[i])
            .filter(|v| v.is_finite())
            .collect();
        if finite.len() < 3 {
            continue;
        }
        let m = median(finite.iter().copied());
        let mad = median(finite.iter().map(|v| (v - m).abs()));
        if mad <= 0.0 || !mad.is_finite() {
            continue;
        }
        for &i in members {
            z[i] = 0.6745 * (values[i] - m) / mad;
        }
    }
    z
}

/// Where each chanblock sits on a uniform frequency grid, and the grid's
/// spacing (Hz). The spacing is the smallest gap between chanblocks, so gaps
/// from missing chanblocks are kept as gaps. None if the frequencies are
/// unknown or not increasing.
pub(super) fn chanblock_positions(freqs_hz: &[f64]) -> Option<(Vec<usize>, f64)> {
    if freqs_hz.len() < 2 || freqs_hz.iter().any(|f| !f.is_finite()) {
        return None;
    }
    let df = freqs_hz
        .windows(2)
        .map(|w| w[1] - w[0])
        .fold(f64::INFINITY, f64::min);
    if df <= 0.0 {
        return None;
    }
    let positions = freqs_hz
        .iter()
        .map(|f| ((f - freqs_hz[0]) / df).round() as usize)
        .collect();
    Some((positions, df))
}

/// The delay power spectrum (dB) of one tile's gains, at delays k / (n df)
/// for k in 0..=n/2, where `positions` places each chanblock on an `n`-long
/// uniform grid. The gains' amplitude and phase are first divided by smooth
/// polynomial fits, so the spectrum shows ripples (such as cable reflections)
/// rather than the bandpass. A Blackman-Harris window keeps the bandpass edges
/// from leaking. Power at +tau and -tau is averaged, because which sign a
/// reflection lands on depends on conventions. All NaN if there are too few
/// valid chanblocks.
pub(super) fn delay_spectrum_db(gains: &[c64], positions: &[usize], n: usize) -> Vec<f64> {
    let num_delays = n / 2 + 1;
    let valid: Vec<usize> = (0..gains.len())
        .filter(|&i| gains[i].is_finite() && gains[i].norm() > 0.0)
        .collect();
    if valid.len() <= 2 * (DETREND_DEGREE + 1) || n < 2 {
        return vec![f64::NAN; num_delays];
    }
    // x in [-1, 1] keeps the polynomial fit well conditioned.
    let x: Vec<f64> = valid
        .iter()
        .map(|&i| 2.0 * positions[i] as f64 / (n - 1) as f64 - 1.0)
        .collect();
    let amp: Vec<f64> = valid.iter().map(|&i| gains[i].norm()).collect();
    let phase = unwrap(&valid.iter().map(|&i| gains[i].arg()).collect::<Vec<_>>());
    let (Some(amp_fit), Some(phase_fit)) = (
        polyfit(&x, &amp, DETREND_DEGREE),
        polyfit(&x, &phase, DETREND_DEGREE),
    ) else {
        return vec![f64::NAN; num_delays];
    };

    let mut residual = vec![c64::new(0.0, 0.0); n];
    let mut weight_sum = 0.0;
    for (k, &i) in valid.iter().enumerate() {
        let model = c64::from_polar(polyval(&amp_fit, x[k]), polyval(&phase_fit, x[k]));
        if model.norm() <= 0.0 {
            continue;
        }
        let w = blackman_harris(positions[i], n);
        residual[positions[i]] = (gains[i] / model - 1.0) * w;
        weight_sum += w * w;
    }

    (0..num_delays)
        .map(|k| {
            let (mut pos, mut neg) = (c64::new(0.0, 0.0), c64::new(0.0, 0.0));
            for (m, r) in residual.iter().enumerate() {
                if r.norm_sqr() == 0.0 {
                    continue;
                }
                let phi = TAU * (k * m) as f64 / n as f64;
                let e = c64::from_polar(1.0, -phi);
                pos += r * e;
                neg += r * e.conj();
            }
            let power = 0.5 * (pos.norm_sqr() + neg.norm_sqr()) / weight_sum;
            10.0 * power.max(1e-30).log10()
        })
        .collect()
}

/// The 4-term Blackman-Harris window at sample `m` of `n`.
fn blackman_harris(m: usize, n: usize) -> f64 {
    let x = TAU * m as f64 / (n - 1) as f64;
    0.35875 - 0.48829 * x.cos() + 0.14128 * (2.0 * x).cos() - 0.01168 * (3.0 * x).cos()
}

/// Remove 2 pi jumps between consecutive phases.
fn unwrap(phases: &[f64]) -> Vec<f64> {
    let mut out = Vec::with_capacity(phases.len());
    let mut offset = 0.0;
    let mut prev = phases.first().copied().unwrap_or(0.0);
    for &p in phases {
        offset -= TAU * ((p - prev) / TAU).round();
        prev = p;
        out.push(p + offset);
    }
    out
}

/// Least-squares polynomial coefficients (lowest order first), from the
/// normal equations. None if they are singular.
fn polyfit(x: &[f64], y: &[f64], degree: usize) -> Option<Vec<f64>> {
    let m = degree + 1;
    let mut a = vec![vec![0.0; m + 1]; m];
    for (&xi, &yi) in x.iter().zip(y) {
        let powers: Vec<f64> = (0..m).map(|p| xi.powi(p as i32)).collect();
        for r in 0..m {
            for c in 0..m {
                a[r][c] += powers[r] * powers[c];
            }
            a[r][m] += powers[r] * yi;
        }
    }
    // Gaussian elimination with partial pivoting.
    for col in 0..m {
        let pivot = (col..m).max_by(|&i, &j| a[i][col].abs().total_cmp(&a[j][col].abs()))?;
        if a[pivot][col].abs() < 1e-12 {
            return None;
        }
        a.swap(col, pivot);
        let pivot_row = a[col].clone();
        for (row, r) in a.iter_mut().enumerate() {
            if row != col {
                let f = r[col] / pivot_row[col];
                for (v, p) in r.iter_mut().zip(&pivot_row).skip(col) {
                    *v -= f * p;
                }
            }
        }
    }
    Some((0..m).map(|r| a[r][m] / a[r][r]).collect())
}

fn polyval(coeffs: &[f64], x: f64) -> f64 {
    coeffs.iter().rev().fold(0.0, |acc, c| acc * x + c)
}

/// The delay (ns) of a reflection off the far end of a tile's cable, from the
/// cable length in the flavour name (e.g. "RG6_90" is 90 m of RG6), assuming
/// a velocity factor of 0.85 for coax. None for flavours with no length, such
/// as fibre.
pub(super) fn reflection_delay_ns(flavour: &str) -> Option<f64> {
    let (kind, length) = flavour.rsplit_once('_')?;
    let kind = kind.to_ascii_uppercase();
    if !(kind.starts_with("RG") || kind.starts_with("LMR")) {
        return None;
    }
    let length: f64 = length.parse().ok()?;
    Some(2.0 * length / (COAX_VELOCITY_FACTOR * VEL_C) * 1e9)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn median_ignores_nan() {
        assert_eq!(median([3.0, f64::NAN, 1.0, 2.0]), 2.0);
        assert_eq!(median([4.0, 1.0, 2.0, 3.0]), 2.5);
        assert!(median([f64::NAN]).is_nan());
    }

    #[test]
    fn normalised_amps_cancel_a_shared_bandpass() {
        // Three tiles of one flavour with the same bandpass at different
        // levels, and one tile of another flavour.
        let bandpass = [1.0, 2.0, 3.0, 4.0];
        let amps: Vec<Vec<f64>> = [1.0, 2.0, 4.0, 10.0]
            .iter()
            .map(|level| bandpass.iter().map(|b| b * level).collect())
            .collect();
        let out = normalised_amps(&amps, &["A", "A", "A", "B"]);
        for row in out {
            for v in row {
                assert_abs_diff_eq!(v, 1.0, epsilon = 1e-12);
            }
        }
    }

    #[test]
    fn normalised_amps_show_a_bad_channel() {
        let mut amps = vec![vec![1.0; 5]; 4];
        amps[2][3] = 1.5;
        amps[1][0] = 0.0;
        let out = normalised_amps(&amps, &["A"; 4]);
        assert_abs_diff_eq!(out[2][3], 1.5, epsilon = 1e-12);
        assert_abs_diff_eq!(out[0][3], 1.0, epsilon = 1e-12);
        assert!(out[1][0].is_nan());
        assert_abs_diff_eq!(rms_about_one(&out[0]), 0.0, epsilon = 1e-12);
        assert_abs_diff_eq!(
            rms_about_one(&out[2]),
            (0.25f64 / 5.0).sqrt(),
            epsilon = 1e-12
        );
    }

    #[test]
    fn modified_z_is_per_group() {
        let values = [1.0, 2.0, 3.0, 100.0, 10.0, 20.0, 30.0, f64::NAN];
        let groups = ["A", "A", "A", "A", "B", "B", "B", "B"];
        let z = modified_z(&values, &groups);
        // Group A: median 2.5, MAD 1.0.
        assert_abs_diff_eq!(z[3], 0.6745 * 97.5, epsilon = 1e-9);
        // Group B: median 20, MAD 10.
        assert_abs_diff_eq!(z[4], -0.6745, epsilon = 1e-9);
        assert!(z[7].is_nan());
        // Too few members.
        assert!(modified_z(&[1.0, 2.0], &["A", "A"])[0].is_nan());
    }

    #[test]
    fn chanblock_positions_keep_gaps() {
        let (positions, df) = chanblock_positions(&[100.0, 140.0, 220.0, 260.0]).unwrap();
        assert_eq!(positions, vec![0, 1, 3, 4]);
        assert_abs_diff_eq!(df, 40.0);
        assert!(chanblock_positions(&[1.0, f64::NAN]).is_none());
    }

    #[test]
    fn unwrap_removes_jumps() {
        let p = unwrap(&[3.0, -3.0, -2.5]);
        assert_abs_diff_eq!(p[1], -3.0 + TAU, epsilon = 1e-12);
        assert_abs_diff_eq!(p[2], -2.5 + TAU, epsilon = 1e-12);
    }

    #[test]
    fn polyfit_recovers_a_polynomial() {
        let x: Vec<f64> = (0..20).map(|i| -1.0 + i as f64 / 10.0).collect();
        let y: Vec<f64> = x.iter().map(|x| 1.0 - 2.0 * x + 0.5 * x * x * x).collect();
        let c = polyfit(&x, &y, 4).unwrap();
        for (got, want) in c.iter().zip([1.0, -2.0, 0.0, 0.5, 0.0]) {
            assert_abs_diff_eq!(*got, want, epsilon = 1e-9);
        }
    }

    #[test]
    fn delay_spectrum_finds_a_reflection() {
        // A smooth bandpass with a 1% ripple from a reflection at 1 us.
        let n = 384;
        let df = 80e3;
        let tau = 1e-6;
        let gains: Vec<c64> = (0..n)
            .map(|i| {
                let f = 167e6 + i as f64 * df;
                let smooth = c64::from_polar(1.0 + 0.2 * (i as f64 / n as f64), 0.3);
                smooth * (1.0 + 0.01 * c64::from_polar(1.0, -TAU * f * tau))
            })
            .collect();
        let positions: Vec<usize> = (0..n).collect();
        let db = delay_spectrum_db(&gains, &positions, n);
        let delays: Vec<f64> = (0..=n / 2)
            .map(|k| k as f64 / (n as f64 * df) * 1e9)
            .collect();
        let peak = (10..db.len())
            .max_by(|&a, &b| db[a].total_cmp(&db[b]))
            .unwrap();
        assert!(
            (delays[peak] - 1000.0).abs() < 2.0 * delays[1],
            "{}",
            delays[peak]
        );
        // The peak stands well clear of the floor.
        assert!(db[peak] - median(db[10..].iter().copied()) > 20.0);
    }

    #[test]
    fn delay_spectrum_needs_data() {
        let gains = vec![c64::new(f64::NAN, 0.0); 32];
        let positions: Vec<usize> = (0..32).collect();
        assert!(delay_spectrum_db(&gains, &positions, 32)
            .iter()
            .all(|v| v.is_nan()));
    }

    #[test]
    fn reflection_delays_follow_the_cable_length() {
        assert_abs_diff_eq!(reflection_delay_ns("RG6_90").unwrap(), 706.4, epsilon = 0.5);
        assert_abs_diff_eq!(
            reflection_delay_ns("LMR400_320").unwrap(),
            2511.6,
            epsilon = 0.5
        );
        assert!(reflection_delay_ns("RFOF").is_none());
        assert!(reflection_delay_ns("FIBRE_100").is_none());
    }
}
