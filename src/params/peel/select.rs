// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Choosing which sources get ionospheric constants fitted.
//!
//! Fitting an ionospheric offset to a source whose fit noise exceeds the real
//! offset makes the subtraction *worse*: the model is shifted by a random
//! amount before it is subtracted. The fit noise scales inversely with a
//! source's apparent flux density, so for a long, brightness-ordered list of
//! "iono-subtract" sources only the first few hundred are usually worth
//! fitting. Two criteria pick them, both relative to the brightest decile of
//! the candidates so that no absolute number has to be tuned per field:
//!
//! - **Predicted** (up front, from the sky model): a source is fitted only if
//!   its predicted fit noise, `ref_flux / flux`, is at most some ratio. Needs
//!   the candidates' apparent flux densities.
//! - **Measured** (online, from the fits): after a few timeblocks have been
//!   fitted, a source whose fitted (α, β) scatter across those timeblocks is
//!   more than some ratio times the brightest decile's is dropped from fitting
//!   for the rest of the observation.
//!
//! A source that fails the selection is not thrown away: it still goes
//! through the fitter with its POSITION frozen, i.e. α and β stay at zero
//! while its gain is iterated like any other source's. (Dropping the gain as
//! well was tried first and made the foreground wedge slightly worse: the
//! faint sources' gains are the part of their fit that does useful work.) In
//! the output such a source has α = β = 0 with a fitted, non-unity gain,
//! whereas a source hyperdrive never touched has gain = 1 as well.

use log::info;

use super::IonoConsts;

/// User-facing settings. All criteria default to off.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct IonoSubSelection {
    /// Drop a source from fitting once the scatter of its fitted (α, β) over
    /// the timeblocks fitted so far exceeds this multiple of the median scatter
    /// of the brightest decile. `None` disables the measured criterion.
    pub(crate) max_scatter_ratio: Option<f64>,

    /// How many timeblocks must have been fitted before the measured
    /// criterion is applied (a scatter needs at least two samples; more makes
    /// it less noisy).
    pub(crate) min_timeblocks: usize,

    /// Only fit sources whose predicted fit noise (`ref_flux / flux`, i.e. the
    /// inverse of the apparent flux relative to the brightest decile's median)
    /// is at most this ratio. `None` disables the predicted criterion.
    pub(crate) max_predicted_scatter_ratio: Option<f64>,
}

impl Default for IonoSubSelection {
    fn default() -> Self {
        Self {
            max_scatter_ratio: None,
            min_timeblocks: DEFAULT_MIN_TIMEBLOCKS,
            max_predicted_scatter_ratio: None,
        }
    }
}

pub(crate) const DEFAULT_MIN_TIMEBLOCKS: usize = 3;

impl IonoSubSelection {
    pub(crate) fn is_enabled(&self) -> bool {
        self.max_scatter_ratio.is_some() || self.max_predicted_scatter_ratio.is_some()
    }
}

/// Runtime state: which candidates are still being fitted, and their fit
/// history. Candidate indices are positions in the brightness-ordered
/// iono-subtract list.
#[derive(Debug, Clone)]
pub(crate) struct IonoSubSelector {
    selection: IonoSubSelection,
    /// Whether each candidate is currently fitted.
    active: Vec<bool>,
    /// Fitted (α, β) per candidate, one entry per timeblock in which it was
    /// fitted.
    history: Vec<Vec<(f64, f64)>>,
    /// Timeblocks recorded so far.
    num_timeblocks: usize,
    /// Candidates dropped up front by the predicted criterion.
    num_dropped_predicted: usize,
    /// Candidates dropped by the measured criterion.
    num_dropped_measured: usize,
}

/// The indices making up the brightest decile of `n` candidates (at least
/// one).
fn brightest_decile(n: usize) -> std::ops::Range<usize> {
    0..(n / 10).max(1).min(n)
}

fn median(values: &mut [f64]) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    values.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    Some(values[values.len() / 2])
}

/// RMS deviation of a series of (α, β) from its mean: √(var(α) + var(β)).
/// `None` for fewer than two samples.
fn scatter(samples: &[(f64, f64)]) -> Option<f64> {
    if samples.len() < 2 {
        return None;
    }
    let n = samples.len() as f64;
    let (ma, mb) = samples
        .iter()
        .fold((0.0, 0.0), |(a, b), &(x, y)| (a + x / n, b + y / n));
    let var = samples
        .iter()
        .map(|&(x, y)| (x - ma).powi(2) + (y - mb).powi(2))
        .sum::<f64>()
        / n;
    Some(var.sqrt())
}

impl IonoSubSelector {
    /// Set up the selector for `num_candidates` sources. If the predicted
    /// criterion is enabled, `apparent_fluxes` (one per candidate, in
    /// brightness order) is used to drop candidates immediately.
    pub(crate) fn new(
        num_candidates: usize,
        selection: IonoSubSelection,
        apparent_fluxes: Option<&[f64]>,
    ) -> Self {
        let mut active = vec![true; num_candidates];
        let mut num_dropped_predicted = 0;

        if let (Some(max_ratio), Some(fluxes)) =
            (selection.max_predicted_scatter_ratio, apparent_fluxes)
        {
            assert_eq!(
                fluxes.len(),
                num_candidates,
                "one apparent flux density per iono-subtract candidate"
            );
            let mut decile: Vec<f64> = fluxes[brightest_decile(num_candidates)].to_vec();
            if let Some(ref_flux) = median(&mut decile) {
                for (is_active, &flux) in active.iter_mut().zip(fluxes) {
                    // Predicted fit noise relative to the reference, ∝ 1/flux.
                    let predicted = if flux > 0.0 {
                        ref_flux / flux
                    } else {
                        f64::INFINITY
                    };
                    if predicted > max_ratio {
                        *is_active = false;
                        num_dropped_predicted += 1;
                    }
                }
                info!(
                    "Ionospheric fitting restricted to {} of {} sources by predicted fit noise (apparent flux ≥ {:.3} Jy = {:.3} Jy / {})",
                    num_candidates - num_dropped_predicted,
                    num_candidates,
                    ref_flux / max_ratio,
                    ref_flux,
                    max_ratio
                );
            }
        }

        Self {
            selection,
            active,
            history: vec![Vec::new(); num_candidates],
            num_timeblocks: 0,
            num_dropped_predicted,
            num_dropped_measured: 0,
        }
    }

    pub(crate) fn num_candidates(&self) -> usize {
        self.active.len()
    }

    pub(crate) fn num_active(&self) -> usize {
        self.active.iter().filter(|&&a| a).count()
    }

    /// One flag per candidate: true if its position (α, β) is fitted in the
    /// next timeblock, false if frozen at zero (gain still fitted).
    pub(crate) fn fit_positions(&self) -> Vec<bool> {
        self.active.clone()
    }

    /// Indices of the candidates whose position is fitted in the next
    /// timeblock, in brightness order (used by the tests).
    #[cfg(test)]
    pub(crate) fn active_indices(&self) -> Vec<usize> {
        self.active
            .iter()
            .enumerate()
            .filter_map(|(i, &a)| a.then_some(i))
            .collect()
    }

    /// Record the constants fitted in a timeblock (`consts` is the full
    /// candidate-length slice; only active candidates are recorded), then
    /// apply the measured criterion. Returns the indices dropped by this call.
    pub(crate) fn record_timeblock(&mut self, consts: &[IonoConsts]) -> Vec<usize> {
        assert_eq!(consts.len(), self.active.len());
        self.num_timeblocks += 1;
        for (i, c) in consts.iter().enumerate() {
            if self.active[i] && c.alpha.is_finite() && c.beta.is_finite() {
                self.history[i].push((c.alpha, c.beta));
            }
        }

        let Some(max_ratio) = self.selection.max_scatter_ratio else {
            return vec![];
        };
        if self.num_timeblocks < self.selection.min_timeblocks.max(2) {
            return vec![];
        }

        let scatters: Vec<Option<f64>> = self
            .history
            .iter()
            .enumerate()
            .map(|(i, h)| if self.active[i] { scatter(h) } else { None })
            .collect();
        let mut decile: Vec<f64> = scatters[brightest_decile(self.active.len())]
            .iter()
            .flatten()
            .copied()
            .collect();
        let Some(ref_scatter) = median(&mut decile) else {
            return vec![];
        };

        let mut dropped = vec![];
        for (i, s) in scatters.iter().enumerate() {
            if let Some(s) = s {
                if *s > max_ratio * ref_scatter {
                    self.active[i] = false;
                    dropped.push(i);
                }
            }
        }
        self.num_dropped_measured += dropped.len();
        if !dropped.is_empty() {
            info!(
                "Timeblock {}: dropped {} sources from ionospheric fitting (scatter > {} × {:.3e}); {} of {} remain",
                self.num_timeblocks,
                dropped.len(),
                max_ratio,
                ref_scatter,
                self.num_active(),
                self.active.len()
            );
        }
        dropped
    }

    /// One-line account of what the selection did, for the end of a run.
    pub(crate) fn summary(&self) -> String {
        format!(
            "Ionospheric position fitting: {} of {} candidate sources fitted to the end ({} frozen up front by predicted fit noise, {} frozen by measured scatter over {} timeblocks; frozen sources keep a fitted gain)",
            self.num_active(),
            self.active.len(),
            self.num_dropped_predicted,
            self.num_dropped_measured,
            self.num_timeblocks
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn consts(pairs: &[(f64, f64)]) -> Vec<IonoConsts> {
        pairs
            .iter()
            .map(|&(alpha, beta)| IonoConsts {
                alpha,
                beta,
                gain: 1.0,
            })
            .collect()
    }

    #[test]
    fn decile_is_at_least_one_source() {
        assert_eq!(brightest_decile(0), 0..0);
        assert_eq!(brightest_decile(1), 0..1);
        assert_eq!(brightest_decile(9), 0..1);
        assert_eq!(brightest_decile(10), 0..1);
        assert_eq!(brightest_decile(20), 0..2);
        assert_eq!(brightest_decile(1000), 0..100);
    }

    #[test]
    fn scatter_is_rms_about_the_mean() {
        assert_eq!(scatter(&[]), None);
        assert_eq!(scatter(&[(1.0, 1.0)]), None);
        // Same point twice: no scatter.
        assert_eq!(scatter(&[(1.0, 2.0), (1.0, 2.0)]), Some(0.0));
        // ±1 in alpha only: variance 1.
        assert!((scatter(&[(-1.0, 0.0), (1.0, 0.0)]).unwrap() - 1.0).abs() < 1e-12);
        // ±1 in both: variance 2.
        assert!((scatter(&[(-1.0, -1.0), (1.0, 1.0)]).unwrap() - 2f64.sqrt()).abs() < 1e-12);
        // Translation invariant.
        assert!((scatter(&[(99.0, -99.0), (101.0, -97.0)]).unwrap() - 2f64.sqrt()).abs() < 1e-12);
    }

    #[test]
    fn disabled_selection_fits_everything() {
        let mut sel = IonoSubSelector::new(5, IonoSubSelection::default(), None);
        assert_eq!(sel.active_indices(), vec![0, 1, 2, 3, 4]);
        // Wildly scattered fits change nothing when no criterion is on.
        for t in 0..10 {
            let t = t as f64;
            let dropped = sel.record_timeblock(&consts(&[
                (0.0, 0.0),
                (0.0, 0.0),
                (t * 1e3, -t * 1e3),
                (0.0, 0.0),
                ((-1f64).powi(t as i32) * 1e6, 0.0),
            ]));
            assert!(dropped.is_empty());
        }
        assert_eq!(sel.num_active(), 5);
    }

    #[test]
    fn predicted_criterion_drops_faint_sources_up_front() {
        // 20 candidates; brightest decile = first 2, median flux = 15.
        let fluxes: Vec<f64> = (0..20).map(|i| 20.0 - i as f64).collect(); // 20, 19, ..., 1
        let selection = IonoSubSelection {
            max_predicted_scatter_ratio: Some(3.0),
            ..Default::default()
        };
        let sel = IonoSubSelector::new(20, selection, Some(&fluxes));
        // ref = median(20, 19) = 20 (upper median of two). Keep flux ≥ 20/3 ≈ 6.67,
        // i.e. fluxes 20..=7 → 14 sources.
        assert_eq!(sel.num_active(), 14);
        assert_eq!(sel.active_indices(), (0..14).collect::<Vec<_>>());
        assert_eq!(
            sel.fit_positions(),
            (0..20).map(|i| i < 14).collect::<Vec<_>>()
        );
        assert!(sel.summary().contains("6 frozen up front"));
    }

    #[test]
    fn predicted_criterion_with_zero_flux() {
        let fluxes = [10.0, 0.0, 10.0];
        let selection = IonoSubSelection {
            max_predicted_scatter_ratio: Some(100.0),
            ..Default::default()
        };
        let sel = IonoSubSelector::new(3, selection, Some(&fluxes));
        assert_eq!(sel.active_indices(), vec![0, 2]);
    }

    #[test]
    fn predicted_criterion_needs_fluxes() {
        // Enabled but no fluxes supplied: nothing is dropped.
        let selection = IonoSubSelection {
            max_predicted_scatter_ratio: Some(1.0),
            ..Default::default()
        };
        let sel = IonoSubSelector::new(4, selection, None);
        assert_eq!(sel.num_active(), 4);
    }

    #[test]
    fn measured_criterion_drops_noisy_sources_after_min_timeblocks() {
        // 10 candidates: index 0 is the brightest decile. Indices 0-4 are
        // stable (scatter ~1e-6), 5-9 are noisy (scatter ~1e-4), i.e. 100×.
        let selection = IonoSubSelection {
            max_scatter_ratio: Some(5.0),
            min_timeblocks: 3,
            max_predicted_scatter_ratio: None,
        };
        let mut sel = IonoSubSelector::new(10, selection, None);
        let make = |t: usize| {
            let sign = if t % 2 == 0 { 1.0 } else { -1.0 };
            consts(
                &(0..10)
                    .map(|i| {
                        let amp = if i < 5 { 1e-6 } else { 1e-4 };
                        (sign * amp, -sign * amp * 0.5)
                    })
                    .collect::<Vec<_>>(),
            )
        };
        // Timeblocks 1 and 2: below min_timeblocks, nothing dropped.
        assert!(sel.record_timeblock(&make(0)).is_empty());
        assert!(sel.record_timeblock(&make(1)).is_empty());
        assert_eq!(sel.num_active(), 10);
        // Timeblock 3: the noisy half goes.
        let dropped = sel.record_timeblock(&make(2));
        assert_eq!(dropped, vec![5, 6, 7, 8, 9]);
        assert_eq!(sel.active_indices(), vec![0, 1, 2, 3, 4]);
        // Dropped sources stay dropped even if later consts are quiet (they
        // would be defaults, which are not recorded).
        let quiet = consts(&[(0.0, 0.0); 10]);
        assert!(sel.record_timeblock(&quiet).is_empty());
        assert_eq!(sel.num_active(), 5);
        assert!(sel.summary().contains("5 frozen by measured scatter"));
    }

    #[test]
    fn measured_criterion_keeps_sources_within_ratio() {
        // Everyone scatters the same; nothing should be dropped.
        let selection = IonoSubSelection {
            max_scatter_ratio: Some(2.0),
            min_timeblocks: 2,
            max_predicted_scatter_ratio: None,
        };
        let mut sel = IonoSubSelector::new(10, selection, None);
        for t in 0..6 {
            let sign = if t % 2 == 0 { 1.0 } else { -1.0 };
            let dropped = sel.record_timeblock(&consts(&[(sign * 1e-5, sign * 1e-5); 10]));
            assert!(dropped.is_empty(), "timeblock {t} dropped {dropped:?}");
        }
        assert_eq!(sel.num_active(), 10);
    }

    #[test]
    fn measured_criterion_ignores_non_finite_fits() {
        let selection = IonoSubSelection {
            max_scatter_ratio: Some(2.0),
            min_timeblocks: 2,
            max_predicted_scatter_ratio: None,
        };
        let mut sel = IonoSubSelector::new(2, selection, None);
        // Source 1 is NaN every time: it has no history, so no scatter, so it
        // is neither the reference nor dropped.
        for _ in 0..4 {
            sel.record_timeblock(&consts(&[(1e-6, 0.0), (f64::NAN, f64::NAN)]));
        }
        assert_eq!(sel.num_active(), 2);
    }

    #[test]
    fn both_criteria_compose() {
        let fluxes: Vec<f64> = (0..10).map(|i| 100.0 / (i as f64 + 1.0)).collect();
        let selection = IonoSubSelection {
            max_scatter_ratio: Some(3.0),
            min_timeblocks: 2,
            max_predicted_scatter_ratio: Some(4.0), // keep flux ≥ 25: indices 0..=3
        };
        let mut sel = IonoSubSelector::new(10, selection, Some(&fluxes));
        assert_eq!(sel.active_indices(), vec![0, 1, 2, 3]);
        // Now index 3 turns out noisy.
        for t in 0..3 {
            let sign = if t % 2 == 0 { 1.0 } else { -1.0 };
            sel.record_timeblock(&consts(&[
                (sign * 1e-6, 0.0),
                (sign * 1e-6, 0.0),
                (sign * 1e-6, 0.0),
                (sign * 1e-3, 0.0),
                (0.0, 0.0),
                (0.0, 0.0),
                (0.0, 0.0),
                (0.0, 0.0),
                (0.0, 0.0),
                (0.0, 0.0),
            ]));
        }
        assert_eq!(sel.active_indices(), vec![0, 1, 2]);
        assert!(sel.summary().contains("6 frozen up front"));
        assert!(sel.summary().contains("1 frozen by measured"));
    }
}
