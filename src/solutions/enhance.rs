// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Take the fine frequency structure of calibration solutions from
//! auto-correlations.
//!
//! A sky calibration that solves every channel on its own puts the errors of an
//! incomplete sky model into the solutions as frequency structure. The
//! auto-correlation of a tile measures the same analogue chain (|gain|^2 times
//! the total power the tile receives) with much more signal to noise than a
//! per-channel solution, and with no sky model. The total power is smooth in
//! frequency, so everything in an auto-correlation spectrum beyond a short
//! delay is the gain of the tile (after Li et al. 2019, ApJ 887:141, section
//! 5.2).
//!
//! Per tile and polarisation (the XX and YY terms of the Jones matrices; the
//! cross terms are left alone), with `ln g = ln|g| + i phase`:
//!
//! ```text
//! ln g  =  lowpass[ ln g_cal ]  +  highpass[ u + i v ]
//! ```
//!
//! - `u = -1/2 ln(auto)` is the log amplitude from the auto-correlation
//!   (hyperdrive solutions are the inverses of the gains) and `v` is the phase
//!   that goes with it for a causal response: an analogue chain (filters, cable
//!   reflections) only delays the signal, so the delay transform of `ln g` is
//!   one-sided and the phase ripple follows from the amplitude ripple (a
//!   Hilbert transform). No reflection modes are fitted and no phase is
//!   unwrapped across a fit.
//! - `g_cal` is the input solution. Its part below the delay cut is kept: the
//!   overall level and slope, and the slow structure in which the total power
//!   and the response to the sky differ (sky spectrum, beam, receiver noise,
//!   mutual coupling). Above the cut, the solution's own structure is
//!   replaced.
//!
//! The low pass is a Gaussian in frequency (half power at the delay cut) on
//! what a low-order polynomial leaves, with outlying channels left out.
//!
//! A tile and polarisation keeps its input solution when its auto-correlation
//! does not describe its gain (e.g. tiles with broken dipoles see a total power
//! with other structure than their response towards the field).

use std::f64::consts::{PI, TAU};

use marlu::{c64, Jones};
use ndarray::prelude::*;

/// The order of the polynomial taken out before the low pass.
const POLY_ORDER: usize = 3;

/// Channels of the input solution further than this many sigma from the low
/// pass are left out of it.
const CLIP_SIGMA: f64 = 5.0;

/// The number of channels averaged before comparing the fine structure of a
/// solution with that of its auto-correlation.
const MISMATCH_CHANS: usize = 8;

/// The fewest channels a tile needs to be fitted.
pub(crate) const MIN_CHANS: usize = 16;

/// Settings for [`enhance_timeblock`].
#[derive(Debug, Clone, Copy)]
pub(crate) struct EnhanceOptions {
    /// The delay \[microseconds\] below which the input solution is kept and
    /// above which the auto-correlations are used (the half-power point of a
    /// Gaussian low pass).
    pub(crate) delay_cut_us: f64,

    /// A tile and polarisation keeps its input solution when the difference
    /// between the fine structure of its solution and of its auto-correlation
    /// exceeds this many times the median over tiles. 0 fits every tile.
    pub(crate) max_mismatch: f64,

    /// If true, only the amplitudes of the solutions are changed.
    pub(crate) amplitude_only: bool,
}

/// What happened to one polarisation of one timeblock of solutions.
#[derive(Debug, Clone)]
pub(crate) struct EnhanceSummary {
    /// The number of tiles that took their fine structure from the
    /// auto-correlations.
    pub(crate) num_fitted: usize,

    /// The indices of the tiles that kept their input solutions because their
    /// auto-correlations don't describe them.
    pub(crate) kept_tiles: Vec<usize>,

    /// The slope of the fine phase structure of the input solutions against
    /// the phase derived from the auto-correlations. This is about 1 when the
    /// solutions are noiseless and the derived phase is right, between 0 and 1
    /// when the solutions are noisy, and negative if the solutions have the
    /// opposite phase convention.
    pub(crate) phase_slope: f64,

    /// The RMS over channels (median over tiles) of the fine log-amplitude
    /// structure in the auto-correlations.
    pub(crate) auto_fine_amp: f64,

    /// The RMS over channels (median over tiles) of the fine log-amplitude
    /// structure of the input solutions that isn't in the auto-correlations.
    pub(crate) cal_minus_auto_amp: f64,

    /// As `cal_minus_auto_amp`, but for the phase \[radians\].
    pub(crate) cal_minus_auto_phase: f64,
}

/// Replace the fine frequency structure of one timeblock of solutions
/// (`di_jones_tf`, dimensions tile and chanblock) with that of the
/// auto-correlations. `autos_ptf` has dimensions polarisation (XX then YY),
/// tile and chanblock, and holds the auto-correlation power, or NaN where
/// there isn't any. `freqs_hz` are the chanblock frequencies, which must be
/// regularly spaced. The returned summaries are for XX and YY.
pub(crate) fn enhance_timeblock(
    mut di_jones_tf: ArrayViewMut2<Jones<f64>>,
    autos_ptf: ArrayView3<f64>,
    freqs_hz: &[f64],
    options: &EnhanceOptions,
) -> [EnhanceSummary; 2] {
    let (num_tiles, num_chans) = di_jones_tf.dim();
    assert_eq!(autos_ptf.dim(), (2, num_tiles, num_chans));
    assert_eq!(freqs_hz.len(), num_chans);

    // Frequencies scaled to [-1, 1] for the polynomials.
    let x = {
        let (min, max) = freqs_hz
            .iter()
            .fold((f64::INFINITY, f64::NEG_INFINITY), |(min, max), &f| {
                (min.min(f), max.max(f))
            });
        let mean = freqs_hz.iter().sum::<f64>() / num_chans as f64;
        let span = if max > min { max - min } else { 1.0 };
        freqs_hz
            .iter()
            .map(|f| 2.0 * (f - mean) / span)
            .collect::<Vec<_>>()
    };
    let chan_hz = {
        let mut diffs = freqs_hz
            .windows(2)
            .map(|w| (w[1] - w[0]).abs())
            .collect::<Vec<_>>();
        median(&mut diffs)
    };
    // A Gaussian in frequency with half power at the delay cut:
    // exp(-2 pi^2 sigma^2 tau^2) = 1/2
    let sigma_chan = (2.0_f64.ln() / 2.0).sqrt() / (PI * options.delay_cut_us * 1e-6) / chan_hz;
    let kernel = gauss_kernel(sigma_chan);
    let hilbert = CausalPhase::new(num_chans);

    // The phase all tiles share in a channel cancels in every visibility and is
    // left undetermined by a per-channel calibration: it is taken out of the
    // whole Jones matrix (from XX). What YY then still shares (YY against XX)
    // is taken out for the fit and put back.
    let common_x = common_phase(di_jones_tf.view(), 0, None, &x);
    for mut di_jones_f in di_jones_tf.outer_iter_mut() {
        for (j, c) in di_jones_f.iter_mut().zip(common_x.iter()) {
            *j *= c.conj();
        }
    }
    let common_y = common_phase(di_jones_tf.view(), 3, None, &x);

    let summary_x = enhance_pol(
        di_jones_tf.view_mut(),
        autos_ptf.index_axis(Axis(0), 0),
        0,
        None,
        &x,
        &kernel,
        &hilbert,
        options,
    );
    let summary_y = enhance_pol(
        di_jones_tf.view_mut(),
        autos_ptf.index_axis(Axis(0), 1),
        3,
        Some(&common_y),
        &x,
        &kernel,
        &hilbert,
        options,
    );
    [summary_x, summary_y]
}

/// Do the work of [`enhance_timeblock`] for one element (`i_elem`) of the
/// Jones matrices. `common` is a phasor per channel that is taken out of the
/// solutions before fitting and put back afterwards.
#[allow(clippy::too_many_arguments)]
fn enhance_pol(
    mut di_jones_tf: ArrayViewMut2<Jones<f64>>,
    autos_tf: ArrayView2<f64>,
    i_elem: usize,
    common: Option<&[c64]>,
    x: &[f64],
    kernel: &[f64],
    hilbert: &CausalPhase,
    options: &EnhanceOptions,
) -> EnhanceSummary {
    let (num_tiles, num_chans) = di_jones_tf.dim();
    let all_chans = vec![true; num_chans];

    struct TileFit {
        i_tile: usize,
        /// The new log solution (amplitude and phase).
        new: Vec<c64>,
        /// The fine structure of the solution that isn't in the autos; NaN
        /// where a channel isn't usable.
        fine_cal: Vec<c64>,
        /// The fine structure of the autos.
        fine_auto: Vec<c64>,
    }

    let mut fits: Vec<TileFit> = Vec::with_capacity(num_tiles);
    for (i_tile, (di_jones_f, autos_f)) in di_jones_tf
        .outer_iter()
        .zip(autos_tf.outer_iter())
        .enumerate()
    {
        let gain = di_jones_f
            .iter()
            .enumerate()
            .map(|(i_chan, j)| match common {
                Some(c) => j[i_elem] * c[i_chan].conj(),
                None => j[i_elem],
            })
            .collect::<Vec<_>>();
        let good = gain
            .iter()
            .zip(autos_f.iter())
            .map(|(g, &a)| g.re.is_finite() && g.im.is_finite() && g.norm() > 0.0 && a > 0.0)
            .collect::<Vec<_>>();
        if good.iter().filter(|&&g| g).count() < MIN_CHANS.max(num_chans / 4) {
            continue;
        }

        // The log amplitude from the autos, with unusable channels bridged.
        let u = interp_bad(
            &autos_f
                .iter()
                .zip(good.iter())
                .map(|(&a, &g)| if g { -0.5 * a.ln() } else { 0.0 })
                .collect::<Vec<_>>(),
            &good,
        );
        let Some((u_low, _)) = lowpass(x, &u, &all_chans, POLY_ORDER, kernel, 0.0) else {
            continue;
        };
        let u_fine = u
            .iter()
            .zip(u_low.iter())
            .map(|(u, l)| u - l)
            .collect::<Vec<_>>();
        let v = hilbert.phase(&u_fine);

        // The parts of the solution that the autos don't describe.
        let ln_amp = gain.iter().map(|g| g.norm().ln()).collect::<Vec<_>>();
        let phase = unwrap_phase(&gain.iter().map(|g| g.arg()).collect::<Vec<_>>());
        let amp_diff = ln_amp
            .iter()
            .zip(u.iter())
            .map(|(a, u)| a - u)
            .collect::<Vec<_>>();
        let phase_diff = phase
            .iter()
            .zip(v.iter())
            .map(|(p, v)| p - v)
            .collect::<Vec<_>>();
        let (Some((amp_low, _)), Some((phase_low, _))) = (
            lowpass(x, &amp_diff, &good, POLY_ORDER, kernel, CLIP_SIGMA),
            lowpass(x, &phase_diff, &good, POLY_ORDER, kernel, CLIP_SIGMA),
        ) else {
            continue;
        };

        let mut new = Vec::with_capacity(num_chans);
        let mut fine_cal = Vec::with_capacity(num_chans);
        let mut fine_auto = Vec::with_capacity(num_chans);
        for i_chan in 0..num_chans {
            let re = u[i_chan] + amp_low[i_chan];
            let im = if options.amplitude_only {
                phase[i_chan]
            } else {
                v[i_chan] + phase_low[i_chan]
            };
            new.push(c64::new(re, im));
            fine_cal.push(if good[i_chan] {
                c64::new(
                    amp_diff[i_chan] - amp_low[i_chan],
                    phase_diff[i_chan] - phase_low[i_chan],
                )
            } else {
                c64::new(f64::NAN, f64::NAN)
            });
            fine_auto.push(c64::new(u_fine[i_chan], v[i_chan]));
        }
        fits.push(TileFit {
            i_tile,
            new,
            fine_cal,
            fine_auto,
        });
    }

    // What all tiles share in the difference: the fine structure common to the
    // solutions that the autos don't have (or the reverse).
    let common_fine = (0..num_chans)
        .map(|i_chan| {
            let mut re = fits
                .iter()
                .map(|f| f.fine_cal[i_chan].re)
                .collect::<Vec<_>>();
            let m = nan_median(&mut re);
            if m.is_finite() {
                m
            } else {
                0.0
            }
        })
        .collect::<Vec<_>>();

    // Find the tiles whose autos don't describe their gains.
    let mismatches = fits
        .iter()
        .map(|f| {
            let mut sum = 0.0;
            let mut n = 0;
            for (fine_cal, common_fine) in f
                .fine_cal
                .chunks_exact(MISMATCH_CHANS)
                .zip(common_fine.chunks_exact(MISMATCH_CHANS))
            {
                let block = nan_mean(
                    fine_cal
                        .iter()
                        .zip(common_fine.iter())
                        .map(|(f, c)| f.re - c),
                );
                if block.is_finite() {
                    sum += block * block;
                    n += 1;
                }
            }
            if n > 0 {
                (sum / n as f64).sqrt()
            } else {
                f64::NAN
            }
        })
        .collect::<Vec<_>>();
    let typical_mismatch = nan_median(&mut mismatches.clone());
    let keep = mismatches
        .iter()
        .map(|&m| options.max_mismatch > 0.0 && m > options.max_mismatch * typical_mismatch)
        .collect::<Vec<_>>();

    // Statistics over the tiles that are fitted.
    let mut slope_num = 0.0;
    let mut slope_den = 0.0;
    let mut auto_fine_amps = vec![];
    let mut cal_minus_auto_amps = vec![];
    let mut cal_minus_auto_phases = vec![];
    for (fit, _) in fits.iter().zip(keep.iter()).filter(|(_, &k)| !k) {
        for (fine_cal, fine_auto) in fit.fine_cal.iter().zip(fit.fine_auto.iter()) {
            if fine_cal.im.is_finite() && fine_auto.im.is_finite() {
                slope_num += fine_auto.im * (fine_cal.im + fine_auto.im);
                slope_den += fine_auto.im * fine_auto.im;
            }
        }
        auto_fine_amps.push(nan_mean(fit.fine_auto.iter().map(|f| f.re * f.re)).sqrt());
        cal_minus_auto_amps.push(nan_mean(fit.fine_cal.iter().map(|f| f.re * f.re)).sqrt());
        cal_minus_auto_phases.push(nan_mean(fit.fine_cal.iter().map(|f| f.im * f.im)).sqrt());
    }

    // Put the new solutions in.
    let mut kept_tiles = vec![];
    let mut num_fitted = 0;
    for (fit, &keep) in fits.iter().zip(keep.iter()) {
        if keep {
            kept_tiles.push(fit.i_tile);
            continue;
        }
        num_fitted += 1;
        for (i_chan, (j, new)) in di_jones_tf
            .index_axis_mut(Axis(0), fit.i_tile)
            .iter_mut()
            .zip(fit.new.iter())
            .enumerate()
        {
            // Flagged values stay flagged, and values that can't be derived
            // (no usable channels within reach of the low pass) are left as
            // they are.
            let old = j[i_elem];
            if !(old.re.is_finite()
                && old.im.is_finite()
                && new.re.is_finite()
                && new.im.is_finite())
            {
                continue;
            }
            let mut g = c64::from_polar(new.re.exp(), new.im);
            if let Some(c) = common {
                g *= c[i_chan];
            }
            j[i_elem] = g;
        }
    }

    EnhanceSummary {
        num_fitted,
        kept_tiles,
        phase_slope: if slope_den > 0.0 {
            slope_num / slope_den
        } else {
            f64::NAN
        },
        auto_fine_amp: nan_median(&mut auto_fine_amps),
        cal_minus_auto_amp: nan_median(&mut cal_minus_auto_amps),
        cal_minus_auto_phase: nan_median(&mut cal_minus_auto_phases),
    }
}

/// Get the phasor (per channel) of the phase that the tiles share in element
/// `i_elem` of their Jones matrices. `remove` is a phasor per channel taken out
/// before anything is done.
fn common_phase(
    di_jones_tf: ArrayView2<Jones<f64>>,
    i_elem: usize,
    remove: Option<&[c64]>,
    x: &[f64],
) -> Vec<c64> {
    const ORDER: usize = 1;
    let (_, num_chans) = di_jones_tf.dim();
    let gain = |j: &Jones<f64>, i_chan: usize| match remove {
        Some(r) => j[i_elem] * r[i_chan].conj(),
        None => j[i_elem],
    };

    // The mean phasor over tiles.
    let reference = (0..num_chans)
        .map(|i_chan| {
            let mut sum = c64::new(0.0, 0.0);
            let mut n = 0;
            for j in di_jones_tf.column(i_chan) {
                let g = gain(j, i_chan);
                let g = g / g.norm();
                if g.re.is_finite() && g.im.is_finite() {
                    sum += g;
                    n += 1;
                }
            }
            if n > 0 {
                sum / n as f64
            } else {
                c64::new(f64::NAN, f64::NAN)
            }
        })
        .collect::<Vec<_>>();
    let coherent = nan_median(&mut reference.iter().map(|r| r.norm()).collect::<Vec<_>>()) > 0.5;
    let mut out = reference
        .iter()
        .map(|r| {
            let n = r.norm();
            if coherent && n.is_finite() && n > 0.0 {
                r / n
            } else {
                c64::new(1.0, 0.0)
            }
        })
        .collect::<Vec<_>>();

    // What's left of the phase of each tile after a line is taken out; the
    // median of this over tiles is also shared.
    let mut residuals = Array2::from_elem(di_jones_tf.dim(), f64::NAN);
    for (di_jones_f, mut residuals_f) in di_jones_tf.outer_iter().zip(residuals.outer_iter_mut()) {
        let phase = unwrap_phase(
            &di_jones_f
                .iter()
                .enumerate()
                .map(|(i_chan, j)| (gain(j, i_chan) * out[i_chan].conj()).arg())
                .collect::<Vec<_>>(),
        );
        let good = phase.iter().map(|p| p.is_finite()).collect::<Vec<_>>();
        if good.iter().filter(|&&g| g).count() < 4 * (ORDER + 1) {
            continue;
        }
        let Some(line) = polyfit_eval(x, &phase, &good, ORDER) else {
            continue;
        };
        for ((r, p), l) in residuals_f.iter_mut().zip(phase.iter()).zip(line.iter()) {
            *r = p - l;
        }
    }
    for (i_chan, out) in out.iter_mut().enumerate() {
        let mut column = residuals.column(i_chan).to_vec();
        let common = nan_median(&mut column);
        if common.is_finite() {
            *out *= c64::from_polar(1.0, common);
        }
    }
    out
}

/// A normalised Gaussian with a standard deviation of `sigma` samples, out to
/// 4 sigma.
fn gauss_kernel(sigma: f64) -> Vec<f64> {
    let half = (4.0 * sigma).ceil().max(0.0) as isize;
    let mut k = (-half..=half)
        .map(|i| (-0.5 * (i as f64 / sigma).powi(2)).exp())
        .collect::<Vec<_>>();
    let sum = k.iter().sum::<f64>();
    k.iter_mut().for_each(|k| *k /= sum);
    k
}

/// Convolve `a` with a symmetric kernel of odd length. The result has the same
/// length as `a`; values beyond its ends are taken as 0.
fn convolve_same(a: &[f64], kernel: &[f64]) -> Vec<f64> {
    let half = kernel.len() / 2;
    (0..a.len())
        .map(|i| {
            let lo = i.saturating_sub(half);
            let hi = (i + half + 1).min(a.len());
            a[lo..hi]
                .iter()
                .zip(kernel[lo + half - i..].iter())
                .map(|(a, k)| a * k)
                .sum()
        })
        .collect()
}

/// The low pass of `y` over the channels `good`: a polynomial of the given
/// order plus the Gaussian average of what it leaves. Channels that aren't good
/// (and, with `clip` > 0, outliers) carry no weight, so they are bridged.
/// Returns the low pass (NaN where no channel is within reach) and the
/// channels used, or `None` if there are too few channels.
fn lowpass(
    x: &[f64],
    y: &[f64],
    good: &[bool],
    order: usize,
    kernel: &[f64],
    clip: f64,
) -> Option<(Vec<f64>, Vec<bool>)> {
    let min_chans = 4 * (order + 1);
    let good = good
        .iter()
        .zip(y.iter())
        .map(|(&g, y)| g && y.is_finite())
        .collect::<Vec<_>>();
    if good.iter().filter(|&&g| g).count() < min_chans {
        return None;
    }

    let mut used = good.clone();
    let mut low = vec![];
    for _ in 0..3 {
        let poly = polyfit_eval(x, y, &used, order)?;
        let resid = y
            .iter()
            .zip(poly.iter())
            .zip(used.iter())
            .map(|((y, p), &u)| if u { y - p } else { 0.0 })
            .collect::<Vec<_>>();
        let norm = convolve_same(
            &used
                .iter()
                .map(|&u| if u { 1.0 } else { 0.0 })
                .collect::<Vec<_>>(),
            kernel,
        );
        low = convolve_same(&resid, kernel)
            .into_iter()
            .zip(norm)
            .zip(poly)
            .map(|((r, n), p)| if n < 1e-3 { f64::NAN } else { p + r / n })
            .collect();
        if clip <= 0.0 {
            break;
        }

        let dev = y
            .iter()
            .zip(low.iter())
            .map(|(y, l)| y - l)
            .collect::<Vec<_>>();
        let mut used_dev = dev
            .iter()
            .zip(used.iter())
            .filter(|(_, &u)| u)
            .map(|(&d, _)| d)
            .collect::<Vec<_>>();
        let centre = nan_median(&mut used_dev);
        used_dev.iter_mut().for_each(|d| *d = (*d - centre).abs());
        let sigma = 1.4826 * nan_median(&mut used_dev);
        let keep = if sigma > 0.0 {
            good.iter()
                .zip(dev.iter())
                .map(|(&g, d)| g && d.abs() <= clip * sigma)
                .collect::<Vec<_>>()
        } else {
            good.clone()
        };
        let num_keep = keep.iter().filter(|&&k| k).count();
        if num_keep == used.iter().filter(|&&u| u).count() || num_keep < min_chans {
            break;
        }
        used = keep;
    }
    Some((low, used))
}

/// Fit a polynomial of the given order to the `good` values of `y` and
/// evaluate it at all `x`. `x` should be within [-1, 1]. Returns `None` if the
/// fit can't be done.
pub(crate) fn polyfit_eval(x: &[f64], y: &[f64], good: &[bool], order: usize) -> Option<Vec<f64>> {
    let n = order + 1;
    // The normal equations, with the right-hand side as the last column.
    let mut a = vec![vec![0.0; n + 1]; n];
    let mut powers = vec![0.0; 2 * n - 1];
    for ((&x, &y), _) in x.iter().zip(y.iter()).zip(good.iter()).filter(|(_, &g)| g) {
        let mut p = 1.0;
        for power in powers.iter_mut() {
            *power = p;
            p *= x;
        }
        for (i, row) in a.iter_mut().enumerate() {
            for (j, elem) in row.iter_mut().take(n).enumerate() {
                *elem += powers[i + j];
            }
            row[n] += powers[i] * y;
        }
    }

    // Gaussian elimination with partial pivoting.
    for i in 0..n {
        let pivot = (i..n).max_by(|&p, &q| a[p][i].abs().total_cmp(&a[q][i].abs()))?;
        if a[pivot][i].abs() < 1e-300 || !a[pivot][i].is_finite() {
            return None;
        }
        a.swap(i, pivot);
        let (pivot_rows, rows) = a.split_at_mut(i + 1);
        let pivot_row = &pivot_rows[i];
        for row in rows {
            let factor = row[i] / pivot_row[i];
            for (elem, pivot_elem) in row.iter_mut().zip(pivot_row.iter()).skip(i) {
                *elem -= factor * pivot_elem;
            }
        }
    }
    let mut coeffs = vec![0.0; n];
    for i in (0..n).rev() {
        let sum = (i + 1..n).map(|j| a[i][j] * coeffs[j]).sum::<f64>();
        coeffs[i] = (a[i][n] - sum) / a[i][i];
    }

    Some(
        x.iter()
            .map(|&x| coeffs.iter().rev().fold(0.0, |acc, c| acc * x + c))
            .collect(),
    )
}

/// Fill in the values of `y` that aren't `good` linearly (the ends take the
/// nearest good value). At least one value must be good.
fn interp_bad(y: &[f64], good: &[bool]) -> Vec<f64> {
    let good_indices = good
        .iter()
        .enumerate()
        .filter(|(_, &g)| g)
        .map(|(i, _)| i)
        .collect::<Vec<_>>();
    let mut out = y.to_vec();
    let (Some(&first), Some(&last)) = (good_indices.first(), good_indices.last()) else {
        return out;
    };
    out[..first].fill(y[first]);
    out[last + 1..].fill(y[last]);
    for w in good_indices.windows(2) {
        let (i0, i1) = (w[0], w[1]);
        for (i, out) in out.iter_mut().enumerate().take(i1).skip(i0 + 1) {
            let frac = (i - i0) as f64 / (i1 - i0) as f64;
            *out = y[i0] + frac * (y[i1] - y[i0]);
        }
    }
    out
}

/// Unwrap the finite values of a phase \[radians\]; the others are NaN.
fn unwrap_phase(phase: &[f64]) -> Vec<f64> {
    let mut out = vec![f64::NAN; phase.len()];
    let mut last: Option<(f64, f64)> = None;
    for (out, &p) in out.iter_mut().zip(phase.iter()) {
        if !p.is_finite() {
            continue;
        }
        *out = match last {
            None => p,
            Some((last_in, last_out)) => {
                let diff = p - last_in;
                let mut wrapped = (diff + PI).rem_euclid(TAU) - PI;
                if wrapped == -PI && diff > 0.0 {
                    wrapped = PI;
                }
                last_out + if diff.abs() < PI { diff } else { wrapped }
            }
        };
        last = Some((p, *out));
    }
    out
}

/// Gets the phase that goes with a log amplitude for a causal response.
struct CausalPhase {
    /// The number of samples the spectrum is extended by (as a mirror image) at
    /// each end, to keep the ends of the band clean.
    pad: usize,

    /// The imaginary part of the kernel that the extended spectrum is
    /// convolved with (circularly).
    kernel: Vec<f64>,
}

impl CausalPhase {
    fn new(num_chans: usize) -> CausalPhase {
        let pad = (num_chans / 2).min(num_chans.saturating_sub(1));
        let m = num_chans + 2 * pad;
        // The kernel is the inverse Fourier transform of a filter that keeps
        // only one side of the delay transform. The side kept (negative
        // delays, in the sense of the usual discrete Fourier transform) is
        // what hyperdrive solutions have (the inverses of the gains, in the
        // uvfits convention).
        let sin = (0..m)
            .map(|i| (TAU * i as f64 / m as f64).sin())
            .collect::<Vec<_>>();
        let kernel = (0..m)
            .map(|d| {
                let mut sum = 0.0;
                if m % 2 == 0 {
                    sum += sin[(m / 2 * d) % m];
                }
                for f in m / 2 + 1..m {
                    sum += 2.0 * sin[(f * d) % m];
                }
                sum / m as f64
            })
            .collect();
        CausalPhase { pad, kernel }
    }

    /// The phase `v` for which `u + i v` has a one-sided delay transform.
    fn phase(&self, u: &[f64]) -> Vec<f64> {
        let n = u.len();
        let pad = self.pad;
        let m = self.kernel.len();
        assert_eq!(m, n + 2 * pad);
        // Extend `u` with its mirror image at both ends.
        let mut ext: Vec<f64> = Vec::with_capacity(m);
        ext.extend(u[1..=pad].iter().rev());
        ext.extend(u.iter());
        ext.extend(u[n - 1 - pad..n - 1].iter().rev());

        (pad..pad + n)
            .map(|i| {
                ext.iter()
                    .enumerate()
                    .map(|(k, e)| e * self.kernel[(i + m - k) % m])
                    .sum::<f64>()
            })
            .collect()
    }
}

/// The median of the values; they get sorted. NaN if there aren't any.
fn median(values: &mut [f64]) -> f64 {
    if values.is_empty() {
        return f64::NAN;
    }
    values.sort_unstable_by(|a, b| a.total_cmp(b));
    let mid = values.len() / 2;
    if values.len() % 2 == 0 {
        0.5 * (values[mid - 1] + values[mid])
    } else {
        values[mid]
    }
}

/// The median of the finite values. NaN if there aren't any.
pub(crate) fn nan_median(values: &mut [f64]) -> f64 {
    let mut finite = values
        .iter()
        .copied()
        .filter(|v| v.is_finite())
        .collect::<Vec<_>>();
    median(&mut finite)
}

/// The mean of the finite values. NaN if there aren't any.
fn nan_mean(values: impl Iterator<Item = f64>) -> f64 {
    let (sum, n) = values
        .filter(|v| v.is_finite())
        .fold((0.0, 0), |(sum, n), v| (sum + v, n + 1));
    if n > 0 {
        sum / n as f64
    } else {
        f64::NAN
    }
}

#[cfg(test)]
mod tests {
    use approx::{assert_abs_diff_eq, assert_relative_eq};

    use super::*;

    const NUM_CHANS: usize = 384;
    const CHAN_HZ: f64 = 80e3;

    fn freqs() -> Vec<f64> {
        (0..NUM_CHANS).map(|i| 167e6 + CHAN_HZ * i as f64).collect()
    }

    /// The gain of a tile with a cable reflection: 1 / (1 - r exp(-2 pi i f
    /// tau)), a causal (minimum phase) response.
    fn reflection_gain(freq_hz: f64, amp: f64, delay_s: f64) -> c64 {
        1.0 / (1.0 - amp * c64::from_polar(1.0, -TAU * freq_hz * delay_s))
    }

    /// A small deterministic pseudo-random number generator (uniform in [-1,
    /// 1)).
    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> f64 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((self.0 >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
        }
    }

    #[test]
    fn test_polyfit_eval() {
        let x = (0..50).map(|i| -1.0 + i as f64 / 24.5).collect::<Vec<_>>();
        let y = x
            .iter()
            .map(|x| 1.0 - 2.0 * x + 0.5 * x * x + 3.0 * x * x * x)
            .collect::<Vec<_>>();
        let mut good = vec![true; 50];
        let mut y_bad = y.clone();
        y_bad[10] = f64::NAN;
        good[10] = false;
        let fit = polyfit_eval(&x, &y_bad, &good, 3).unwrap();
        for (f, y) in fit.iter().zip(y.iter()) {
            assert_abs_diff_eq!(f, y, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_unwrap_phase() {
        let truth = (0..100).map(|i| 0.3 * i as f64).collect::<Vec<_>>();
        let mut wrapped = truth
            .iter()
            .map(|p| c64::from_polar(1.0, *p).arg())
            .collect::<Vec<_>>();
        wrapped[40] = f64::NAN;
        let unwrapped = unwrap_phase(&wrapped);
        for (i, (u, t)) in unwrapped.iter().zip(truth.iter()).enumerate() {
            if i == 40 {
                assert!(u.is_nan());
            } else {
                assert_abs_diff_eq!(u, t, epsilon = 1e-10);
            }
        }
    }

    #[test]
    fn test_interp_bad() {
        let y = [0.0, 1.0, 0.0, 0.0, 4.0, 0.0];
        let good = [false, true, false, false, true, false];
        assert_eq!(interp_bad(&y, &good), vec![1.0, 1.0, 2.0, 3.0, 4.0, 4.0]);
    }

    #[test]
    fn test_convolve_same() {
        let a = [1.0, 2.0, 3.0, 4.0];
        assert_eq!(
            convolve_same(&a, &[0.25, 0.5, 0.25]),
            vec![1.0, 2.0, 3.0, 2.75]
        );
        // A kernel longer than the data.
        assert_eq!(
            convolve_same(&[1.0, 2.0], &[1.0, 1.0, 1.0, 1.0, 1.0]),
            vec![3.0, 3.0]
        );
    }

    /// For an even number of samples the kernel is the textbook discrete
    /// Hilbert transform (with the sign for a response whose delay transform
    /// is on the negative side).
    #[test]
    fn test_causal_phase_kernel() {
        let hilbert = CausalPhase::new(64);
        let m = hilbert.kernel.len();
        assert_eq!(m, 128);
        for (d, k) in hilbert.kernel.iter().enumerate() {
            let expected = if d % 2 == 1 {
                -2.0 / m as f64 / (PI * d as f64 / m as f64).tan()
            } else {
                0.0
            };
            assert_abs_diff_eq!(*k, expected, epsilon = 1e-12);
        }
    }

    /// The phase of a reflection follows from its amplitude.
    #[test]
    fn test_causal_phase_of_reflection() {
        for num_chans in [NUM_CHANS, NUM_CHANS - 1] {
            let hilbert = CausalPhase::new(num_chans);
            let freqs = &freqs()[..num_chans];
            // hyperdrive solutions are the inverses of the gains.
            let sols = freqs
                .iter()
                .map(|&f| 1.0 / reflection_gain(f, 0.05, 0.6e-6))
                .collect::<Vec<_>>();
            let u = sols.iter().map(|s| s.norm().ln()).collect::<Vec<_>>();
            let v = hilbert.phase(&u);
            // Away from the ends of the band, the phase is recovered.
            let max_err = |skip: usize| {
                v.iter()
                    .zip(sols.iter())
                    .skip(skip)
                    .take(num_chans - 2 * skip)
                    .map(|(v, s)| (v - s.arg()).abs())
                    .fold(0.0, f64::max)
            };
            // The ripple is 0.05 rad. It's recovered to a few percent, except
            // near the ends of the band: the mirror image the spectrum is
            // extended with doesn't continue the ripple, so the error grows
            // towards the ends (to the size of the ripple in the last
            // channel).
            assert!(max_err(96) < 1.2e-3, "{}", max_err(96));
            assert!(max_err(32) < 4e-3, "{}", max_err(32));
            assert!(max_err(8) < 1.2e-2, "{}", max_err(8));
            assert!(max_err(0) < 0.05, "{}", max_err(0));
        }
    }

    /// Make solutions and autos for some tiles: every tile has its own smooth
    /// bandpass and reflection; the solutions also have per-channel noise.
    /// Returns the noisy solutions, the true solutions and the autos.
    #[allow(clippy::type_complexity)]
    fn make_data(
        num_tiles: usize,
        noise: f64,
    ) -> (Array2<Jones<f64>>, Array2<Jones<f64>>, Array3<f64>) {
        let freqs = freqs();
        let mut rng = Rng(1234);
        let mut noisy = Array2::from_elem((num_tiles, NUM_CHANS), Jones::identity());
        let mut truth = noisy.clone();
        let mut autos = Array3::zeros((2, num_tiles, NUM_CHANS));
        for i_tile in 0..num_tiles {
            for (i_pol, i_elem) in [(0, 0), (1, 3)] {
                let amp = 0.01 + 0.04 * rng.next().abs();
                let delay = (0.3 + 0.9 * rng.next().abs()) * 1e-6;
                let level = 1.0 + 0.2 * rng.next();
                let slope = 0.1 * rng.next();
                let phase0 = 3.0 * rng.next();
                let phase_slope = 2.0 * rng.next();
                for (i_chan, &f) in freqs.iter().enumerate() {
                    let x = (f - freqs[0]) / (freqs[NUM_CHANS - 1] - freqs[0]) - 0.5;
                    // The smooth part of the gain (including a phase that
                    // isn't causal, like a cable delay or what a sky
                    // calibration refers to another tile) and the reflection.
                    let gain = level
                        * (1.0 + slope * x + 0.05 * (3.0 * x).cos())
                        * c64::from_polar(1.0, phase0 + phase_slope * x)
                        * reflection_gain(f, amp, delay);
                    // The total power the tile sees is smooth.
                    let sky = 1e4 * (f / 180e6).powf(-2.5);
                    autos[(i_pol, i_tile, i_chan)] = gain.norm_sqr() * sky;
                    let sol = 1.0 / gain;
                    truth[(i_tile, i_chan)][i_elem] = sol;
                    noisy[(i_tile, i_chan)][i_elem] =
                        sol * (1.0 + noise * c64::new(rng.next(), rng.next()));
                }
            }
        }
        (noisy, truth, autos)
    }

    /// The RMS of the fractional difference between solutions, after taking
    /// out the phase that all tiles share in a channel.
    fn rms_error(a: ArrayView2<Jones<f64>>, b: ArrayView2<Jones<f64>>, i_elem: usize) -> f64 {
        let (num_tiles, num_chans) = a.dim();
        let mut sum = 0.0;
        let mut n = 0;
        // Leave out the ends of the band, where a low pass is less sure.
        for i_chan in 20..num_chans - 20 {
            let mut common = c64::new(0.0, 0.0);
            for i_tile in 0..num_tiles {
                let r = a[(i_tile, i_chan)][i_elem] / b[(i_tile, i_chan)][i_elem];
                common += r / r.norm();
            }
            common /= common.norm();
            for i_tile in 0..num_tiles {
                let r = a[(i_tile, i_chan)][i_elem] / b[(i_tile, i_chan)][i_elem] / common;
                if r.re.is_finite() {
                    sum += (r - 1.0).norm_sqr();
                    n += 1;
                }
            }
        }
        (sum / n as f64).sqrt()
    }

    const OPTIONS: EnhanceOptions = EnhanceOptions {
        delay_cut_us: 0.1,
        max_mismatch: 5.0,
        amplitude_only: false,
    };

    #[test]
    fn test_enhance_removes_noise() {
        let num_tiles = 16;
        let (noisy, truth, autos) = make_data(num_tiles, 0.03);
        let mut enhanced = noisy.clone();
        let summaries = enhance_timeblock(enhanced.view_mut(), autos.view(), &freqs(), &OPTIONS);

        for (summary, i_elem) in summaries.iter().zip([0, 3]) {
            assert_eq!(summary.num_fitted, num_tiles);
            assert!(summary.kept_tiles.is_empty());
            // The phase ripples of the solutions follow the autos.
            assert_relative_eq!(summary.phase_slope, 1.0, epsilon = 0.1);

            let before = rms_error(noisy.view(), truth.view(), i_elem);
            let after = rms_error(enhanced.view(), truth.view(), i_elem);
            assert!(before > 0.02, "{before}");
            // Only the noise below the delay cut is left.
            assert!(after < before / 4.0, "before {before}, after {after}");
        }
        // The cross terms are untouched (but for the common phase).
        for (e, n) in enhanced.iter().zip(noisy.iter()) {
            assert_abs_diff_eq!(e[1].norm(), n[1].norm(), epsilon = 1e-12);
            assert_abs_diff_eq!(e[2].norm(), n[2].norm(), epsilon = 1e-12);
        }
    }

    #[test]
    fn test_enhance_amplitude_only() {
        let (noisy, truth, autos) = make_data(16, 0.03);
        let mut enhanced = noisy.clone();
        let options = EnhanceOptions {
            amplitude_only: true,
            ..OPTIONS
        };
        enhance_timeblock(enhanced.view_mut(), autos.view(), &freqs(), &options);
        // Taking out the common phase is all that happens to the phases.
        let mut expected = noisy.clone();
        let x = {
            let f = freqs();
            let mean = f.iter().sum::<f64>() / f.len() as f64;
            let span = f[f.len() - 1] - f[0];
            f.iter()
                .map(|f| 2.0 * (f - mean) / span)
                .collect::<Vec<_>>()
        };
        let common = common_phase(noisy.view(), 0, None, &x);
        for mut row in expected.outer_iter_mut() {
            for (j, c) in row.iter_mut().zip(common.iter()) {
                *j *= c.conj();
            }
        }
        let mut amp_err_before = 0.0;
        let mut amp_err_after = 0.0;
        for ((e, x), t) in enhanced.iter().zip(expected.iter()).zip(truth.iter()) {
            for i_elem in [0, 3] {
                assert_abs_diff_eq!((e[i_elem] / x[i_elem]).arg(), 0.0, epsilon = 1e-9);
                amp_err_before += (x[i_elem].norm() / t[i_elem].norm() - 1.0).powi(2);
                amp_err_after += (e[i_elem].norm() / t[i_elem].norm() - 1.0).powi(2);
            }
        }
        assert!(amp_err_after < amp_err_before / 16.0);
    }

    #[test]
    fn test_enhance_keeps_flags_and_mismatched_tiles() {
        let num_tiles = 16;
        let (mut noisy, _, mut autos) = make_data(num_tiles, 0.01);
        // Tile 2 is flagged in the solutions, tile 3 has no autos.
        for i_chan in 0..NUM_CHANS {
            noisy[(2, i_chan)] = Jones::nan();
            autos[(0, 3, i_chan)] = f64::NAN;
            autos[(1, 3, i_chan)] = f64::NAN;
        }
        // A flagged channel, and one without autos.
        for i_tile in 0..num_tiles {
            noisy[(i_tile, 100)] = Jones::nan();
            autos[(0, i_tile, 200)] = f64::NAN;
            autos[(1, i_tile, 200)] = f64::NAN;
        }
        // The XX auto of tile 5 has structure that its gain doesn't.
        for (i_chan, f) in freqs().into_iter().enumerate() {
            autos[(0, 5, i_chan)] *= 1.0 + 0.3 * (TAU * f * 0.8e-6).sin();
        }

        let mut enhanced = noisy.clone();
        let [summary_x, summary_y] =
            enhance_timeblock(enhanced.view_mut(), autos.view(), &freqs(), &OPTIONS);
        assert_eq!(summary_x.kept_tiles, vec![5]);
        assert_eq!(summary_x.num_fitted, num_tiles - 3);
        assert!(summary_y.kept_tiles.is_empty());
        assert_eq!(summary_y.num_fitted, num_tiles - 2);

        for i_chan in 0..NUM_CHANS {
            assert!(enhanced[(2, i_chan)].any_nan());
            for i_tile in 0..num_tiles {
                let (e, n) = (enhanced[(i_tile, i_chan)], noisy[(i_tile, i_chan)]);
                if i_tile == 2 || i_chan == 100 {
                    assert!(e.any_nan());
                    continue;
                }
                assert!(!e.any_nan());
                // Kept solutions differ only by the common phase.
                if i_tile == 3 || i_tile == 5 {
                    assert_abs_diff_eq!(e[0].norm(), n[0].norm(), epsilon = 1e-12);
                }
                if i_tile == 3 {
                    assert_abs_diff_eq!(e[3].norm(), n[3].norm(), epsilon = 1e-12);
                }
            }
        }
        // Channels without autos are bridged, not left as they were.
        assert!((enhanced[(0, 200)][0].norm() / noisy[(0, 200)][0].norm() - 1.0).abs() > 1e-6);
    }

    /// Fewer channels than the low pass reaches over mustn't be a problem.
    #[test]
    fn test_enhance_few_channels() {
        let (noisy, _, autos) = make_data(8, 0.01);
        let n = 32;
        let mut enhanced = noisy.slice(s![.., ..n]).to_owned();
        let summaries = enhance_timeblock(
            enhanced.view_mut(),
            autos.slice(s![.., .., ..n]),
            &freqs()[..n],
            &OPTIONS,
        );
        assert_eq!(summaries[0].num_fitted + summaries[0].kept_tiles.len(), 8);
        assert!(enhanced.iter().all(|j| !j.any_nan()));
    }
}
