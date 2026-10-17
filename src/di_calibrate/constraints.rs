// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Solution constraints applied between calibration iterations, modelled on
//! DP3's DDECal constraints.
//!
//! DP3 solves all channel blocks of a solution interval together: after every
//! iteration the new solutions are projected onto the constraint (e.g. phase
//! only, a phase model that is a function of frequency, or a smoothness
//! kernel across frequency) and the next iteration starts from the projected
//! solutions. The types here provide the same projections for hyperdrive's
//! per-chanblock StefCal solutions. Parameter names and semantics follow
//! DP3's `mode`, `smoothnessconstraint`, `smoothnessreffrequency`,
//! `smoothnessrefdistance`, `smoothnessspectralexponent`,
//! `smoothness_kernel_truncation` and `max_tec_delay_wraps` so that the same
//! settings can be used with either tool.
//!
//! The constraints operate on hyperdrive's *internal* Jones matrices, i.e.
//! the `J` in `data ≈ J_p model J_q^H`, which is the same quantity as a DP3
//! gain `G`. The solutions hyperdrive writes out are the inverses of these.

use std::str::FromStr;

use marlu::{c64, Jones};
use ndarray::prelude::*;
use rayon::prelude::*;
use strum_macros::{Display, EnumIter, EnumString, IntoStaticStr};
use thiserror::Error;

/// `phase = -8.44797245e9 * TEC / freq`, as used by DP3.
pub const TEC_TO_PHASE: f64 = -8.44797245e9;

/// The solver "mode", named as DP3's DDECal `mode`. Each mode constrains the
/// form of the Jones matrices between iterations.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Default, Display, EnumIter, EnumString, IntoStaticStr,
)]
#[allow(non_camel_case_types)]
pub enum SolveMode {
    /// Unconstrained 2x2 Jones matrices (hyperdrive's default).
    #[default]
    #[strum(serialize = "fulljones")]
    FullJones,

    /// Diagonal Jones matrices (XX and YY complex gains).
    #[strum(serialize = "diagonal")]
    Diagonal,

    /// Diagonal, unit amplitude.
    #[strum(serialize = "diagonalphase")]
    DiagonalPhase,

    /// Diagonal, real and positive.
    #[strum(serialize = "diagonalamplitude")]
    DiagonalAmplitude,

    /// A single complex gain shared by XX and YY.
    #[strum(serialize = "scalar")]
    Scalar,

    /// A single unit-amplitude phase shared by XX and YY.
    #[strum(serialize = "scalarphase")]
    ScalarPhase,

    /// A single real, positive amplitude shared by XX and YY.
    #[strum(serialize = "scalaramplitude")]
    ScalarAmplitude,

    /// Scalar phase `φ(ν) = a/ν`.
    #[strum(serialize = "tec")]
    Tec,

    /// Scalar phase `φ(ν) = a/ν + b`.
    #[strum(serialize = "tec+phase", serialize = "tecandphase")]
    TecAndPhase,

    /// Scalar phase `φ(ν) = a/ν + c ν`.
    #[strum(serialize = "tec+delay")]
    TecAndDelay,

    /// Scalar phase `φ(ν) = a/ν + b + c ν`.
    #[strum(serialize = "tec+phase+delay")]
    TecPhaseAndDelay,

    /// Scalar phase linear in frequency, `φ(ν) = c ν` (not available in DP3).
    #[strum(serialize = "delay")]
    Delay,

    /// Scalar phase linear in frequency with an offset, `φ(ν) = b + c ν` (not
    /// available in DP3).
    #[strum(serialize = "phase+delay", serialize = "delay+phase")]
    PhaseAndDelay,
}

impl SolveMode {
    /// Which terms of `φ(ν) = a/ν + b + cν` are fitted, if this mode fits a
    /// phase model across frequency.
    pub(crate) fn phase_terms(self) -> Option<PhaseTerms> {
        match self {
            SolveMode::Tec => Some(PhaseTerms {
                tec: true,
                offset: false,
                delay: false,
            }),
            SolveMode::TecAndPhase => Some(PhaseTerms {
                tec: true,
                offset: true,
                delay: false,
            }),
            SolveMode::TecAndDelay => Some(PhaseTerms {
                tec: true,
                offset: false,
                delay: true,
            }),
            SolveMode::TecPhaseAndDelay => Some(PhaseTerms {
                tec: true,
                offset: true,
                delay: true,
            }),
            SolveMode::Delay => Some(PhaseTerms {
                tec: false,
                offset: false,
                delay: true,
            }),
            SolveMode::PhaseAndDelay => Some(PhaseTerms {
                tec: false,
                offset: true,
                delay: true,
            }),
            _ => None,
        }
    }

    fn is_scalar(self) -> bool {
        matches!(
            self,
            SolveMode::Scalar | SolveMode::ScalarPhase | SolveMode::ScalarAmplitude
        ) || self.phase_terms().is_some()
    }

    fn is_phase_only(self) -> bool {
        matches!(self, SolveMode::DiagonalPhase | SolveMode::ScalarPhase)
            || self.phase_terms().is_some()
    }

    fn is_amplitude_only(self) -> bool {
        matches!(
            self,
            SolveMode::DiagonalAmplitude | SolveMode::ScalarAmplitude
        )
    }

    /// Project a single Jones matrix onto this mode (everything that doesn't
    /// need other chanblocks).
    pub(crate) fn project(self, j: Jones<f64>) -> Jones<f64> {
        if self == SolveMode::FullJones {
            return j;
        }
        let (mut xx, mut yy) = (j[0], j[3]);
        if self.is_scalar() {
            let s = if self.is_amplitude_only() {
                c64::new((xx.norm() + yy.norm()) * 0.5, 0.0)
            } else {
                (xx + yy) * 0.5
            };
            xx = s;
            yy = s;
        }
        if self.is_phase_only() {
            xx = unit(xx);
            yy = unit(yy);
        } else if self.is_amplitude_only() {
            xx = c64::new(xx.norm(), 0.0);
            yy = c64::new(yy.norm(), 0.0);
        }
        Jones::from([xx, c64::default(), c64::default(), yy])
    }
}

fn unit(z: c64) -> c64 {
    let n = z.norm();
    if n > 0.0 && n.is_finite() {
        z / n
    } else {
        c64::new(1.0, 0.0)
    }
}

/// Which terms of `φ(x) = a/x + b + c x` (with `x = ν/ν_ref`) to fit.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PhaseTerms {
    pub(crate) tec: bool,
    pub(crate) offset: bool,
    pub(crate) delay: bool,
}

/// Fitted phase-model coefficients, in the relative frequency `x = ν/ν_ref`.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct PhaseFit {
    pub(crate) a: f64,
    pub(crate) b: f64,
    pub(crate) c: f64,
}

impl PhaseFit {
    fn eval(&self, x: f64) -> f64 {
        self.a / x + self.b + self.c * x
    }

    /// Convert to physical units given the reference frequency used for `x`:
    /// (TEC [TECU], phase offset [rad], delay [s]).
    pub(crate) fn to_physical(self, ref_freq_hz: f64) -> (f64, f64, f64) {
        (
            self.a * ref_freq_hz / TEC_TO_PHASE,
            self.b,
            self.c / (ref_freq_hz * std::f64::consts::TAU),
        )
    }
}

#[inline]
fn wrap(x: f64) -> f64 {
    use std::f64::consts::{PI, TAU};
    let mut w = x % TAU;
    if w >= PI {
        w -= TAU;
    } else if w < -PI {
        w += TAU;
    }
    w
}

/// The cost DP3 minimises: `Σ 1 - cos(residual) * weight`.
fn phase_cost(x: &[f64], y: &[f64], w: &[f64], fit: &PhaseFit) -> f64 {
    x.iter()
        .zip(y)
        .zip(w)
        .map(|((&x, &y), &w)| 1.0 - (y - fit.eval(x)).cos() * w)
        .sum()
}

/// Weighted linear least squares of `y ≈ Σ_k p_k basis_k(x)` for the enabled
/// terms. Returns `None` when the system is singular.
fn linear_phase_solve(x: &[f64], y: &[f64], w: &[f64], terms: PhaseTerms) -> Option<PhaseFit> {
    // Up to a 3x3 normal-equation system.
    let basis = |x: f64| -> [f64; 3] { [1.0 / x, 1.0, x] };
    let enabled = [terms.tec, terms.offset, terms.delay];
    let mut ata = [[0.0; 3]; 3];
    let mut atb = [0.0; 3];
    for ((&x, &y), &w) in x.iter().zip(y).zip(w) {
        if w <= 0.0 {
            continue;
        }
        let b = basis(x);
        for i in 0..3 {
            if !enabled[i] {
                continue;
            }
            atb[i] += w * b[i] * y;
            for j in 0..3 {
                if enabled[j] {
                    ata[i][j] += w * b[i] * b[j];
                }
            }
        }
    }
    // Gaussian elimination with partial pivoting over the enabled rows.
    let idx: Vec<usize> = (0..3).filter(|&i| enabled[i]).collect();
    let n = idx.len();
    let mut m = vec![vec![0.0; n + 1]; n];
    for (r, &i) in idx.iter().enumerate() {
        for (c, &j) in idx.iter().enumerate() {
            m[r][c] = ata[i][j];
        }
        m[r][n] = atb[i];
    }
    for col in 0..n {
        let piv = (col..n).max_by(|&a, &b| m[a][col].abs().total_cmp(&m[b][col].abs()))?;
        if m[piv][col].abs() < 1e-300 {
            return None;
        }
        m.swap(col, piv);
        for r in 0..n {
            if r != col {
                let f = m[r][col] / m[col][col];
                if f != 0.0 {
                    let (pivot_row, row) = if r < col {
                        let (a, b) = m.split_at_mut(col);
                        (&b[0], &mut a[r])
                    } else {
                        let (a, b) = m.split_at_mut(r);
                        (&a[col], &mut b[0])
                    };
                    for (x, &p) in row.iter_mut().zip(pivot_row.iter()).skip(col) {
                        *x -= f * p;
                    }
                }
            }
        }
    }
    let mut p = [0.0; 3];
    for (r, &i) in idx.iter().enumerate() {
        p[i] = m[r][n] / m[r][r];
        if !p[i].is_finite() {
            return None;
        }
    }
    Some(PhaseFit {
        a: p[0],
        b: p[1],
        c: p[2],
    })
}

/// Fit `φ(x) = a/x + b + c x` to wrapped phases, as DP3's
/// `TecOffsetDelayGridSearch` (least-squares method) does: a grid of starting
/// points for the wrapping terms `a` and `c` spanning `max_wraps` turns across
/// the band, a weighted linear fit of the wrapped residuals at each grid
/// point, then a few refinement passes on the best fit.
///
/// `x` are relative frequencies (`ν/ν_ref`, DP3 uses the first channel as
/// reference), `y` phases in radians, `w` weights (0 to ignore).
pub(crate) fn fit_phase_model(
    x: &[f64],
    y: &[f64],
    w: &[f64],
    terms: PhaseTerms,
    max_wraps: usize,
) -> PhaseFit {
    use std::f64::consts::TAU;
    assert_eq!(x.len(), y.len());
    assert_eq!(x.len(), w.len());
    let mut best = PhaseFit::default();
    if x.len() < 2 || w.iter().all(|&w| w <= 0.0) {
        return best;
    }
    let (first, last) = (x[0], x[x.len() - 1]);
    let max_y_range = TAU * max_wraps as f64;
    let max_a = if terms.tec && first != last {
        max_y_range / (1.0 / first - 1.0 / last)
    } else {
        0.0
    };
    let max_c = if terms.delay && first != last {
        max_y_range / (last - first)
    } else {
        0.0
    };
    let n_grid = max_wraps * 4 + 2;
    let a_grid: Vec<f64> = if terms.tec {
        (0..n_grid)
            .map(|i| (i as f64 / (max_wraps * 2) as f64 - 1.0) * max_a)
            .collect()
    } else {
        vec![0.0]
    };
    let c_grid: Vec<f64> = if terms.delay {
        (0..n_grid)
            .map(|i| (i as f64 / (max_wraps * 2) as f64 - 1.0) * max_c)
            .collect()
    } else {
        vec![0.0]
    };

    let mut best_cost = f64::MAX;
    let mut residual = vec![0.0; x.len()];
    let mut try_fit = |start: PhaseFit, best: &mut PhaseFit, best_cost: &mut f64| {
        for (r, (&x, &y)) in residual.iter_mut().zip(x.iter().zip(y)) {
            *r = wrap(y - start.eval(x));
        }
        // When an offset is fitted, centre the residuals on their circular
        // mean first, otherwise residuals straddling ±π (an offset near π)
        // are torn apart by the wrapping and the linear fit is meaningless.
        // (DP3 doesn't do this, and suffers from it.)
        let centre = if terms.offset {
            let (s, c) = residual.iter().zip(w).fold((0.0, 0.0), |(s, c), (&r, &w)| {
                (s + w * r.sin(), c + w * r.cos())
            });
            let centre = s.atan2(c);
            for r in residual.iter_mut() {
                *r = wrap(*r - centre);
            }
            centre
        } else {
            0.0
        };
        if let Some(delta) = linear_phase_solve(x, &residual, w, terms) {
            let fit = PhaseFit {
                a: start.a + delta.a,
                b: start.b + delta.b + centre,
                c: start.c + delta.c,
            };
            let cost = phase_cost(x, y, w, &fit);
            if cost < *best_cost {
                *best_cost = cost;
                *best = fit;
            }
        }
    };
    for &a in &a_grid {
        for &c in &c_grid {
            try_fit(PhaseFit { a, b: 0.0, c }, &mut best, &mut best_cost);
        }
    }
    for _ in 0..5 {
        try_fit(best, &mut best, &mut best_cost);
    }
    best
}

/// A port of DP3's `KernelSmoother` with the Gaussian kernel: the solutions
/// of one tile/polarisation are replaced by their weighted, kernel-smoothed
/// values across (possibly irregular) chanblock frequencies.
#[derive(Debug, Clone)]
pub struct KernelSmoother {
    freqs: Vec<f64>,
    /// Kernel size [Hz]; DP3's `smoothnessconstraint`.
    bandwidth: f64,
    /// DP3's `smoothnessreffrequency`; 0 disables the frequency scaling.
    ref_frequency: f64,
    /// DP3's `smoothnessspectralexponent`.
    spectral_exponent: f64,
    /// DP3's `smoothness_kernel_truncation`.
    truncate: bool,
}

impl KernelSmoother {
    pub(crate) fn new(
        freqs: Vec<f64>,
        bandwidth: f64,
        ref_frequency: f64,
        spectral_exponent: f64,
        truncate: bool,
    ) -> KernelSmoother {
        KernelSmoother {
            freqs,
            bandwidth,
            ref_frequency,
            spectral_exponent,
            truncate,
        }
    }

    /// Gaussian with `sigma = bandwidth / 3`, trimmed at the bandwidth when
    /// truncating (as DP3).
    fn kernel(&self, distance: f64) -> f64 {
        let x = distance / self.bandwidth;
        if self.truncate && !(-1.0..=1.0).contains(&x) {
            0.0
        } else {
            (-x * x * 9.0).exp()
        }
    }

    /// Smooth `data` in place. `kernel_size_factor` scales the frequency
    /// axis; larger factors give less smoothing (DP3's per-antenna and
    /// per-direction factors). Positions where no weight falls inside the
    /// kernel become NaN, as in DP3.
    pub(crate) fn smooth(&self, data: &mut [c64], weight: &[f64], kernel_size_factor: f64) {
        let n = self.freqs.len();
        assert_eq!(data.len(), n);
        assert_eq!(weight.len(), n);
        let mut out = vec![c64::default(); n];

        let mut band_left = 0;
        let lbound = self
            .freqs
            .partition_point(|&f| f < self.freqs[0] + self.bandwidth * 0.5);
        let mut band_right = (lbound + 1).min(n);
        for (i, out) in out.iter_mut().enumerate() {
            let mut start = 0;
            let mut end = n;
            let local_bandwidth = if self.ref_frequency == 0.0 {
                self.bandwidth
            } else {
                self.bandwidth * self.freqs[i] / self.ref_frequency
            };
            if self.truncate {
                while self.freqs[band_left] < self.freqs[i] - local_bandwidth * 0.5 {
                    band_left += 1;
                }
                while band_right != n
                    && self.freqs[band_right] < self.freqs[i] + local_bandwidth * 0.5
                {
                    band_right += 1;
                }
                start = band_left.saturating_sub(1);
                end = if band_right < n { band_right + 1 } else { n };
            }
            let frequency_correction =
                (local_bandwidth / self.bandwidth).powf(self.spectral_exponent);
            let kernel_correction = frequency_correction * kernel_size_factor;
            let mut sum = c64::default();
            let mut weight_sum = 0.0;
            for j in start..end {
                let distance = self.freqs[i] - self.freqs[j];
                let w = self.kernel(distance * kernel_correction) * weight[j];
                sum += data[j] * w;
                weight_sum += w;
            }
            *out = if weight_sum == 0.0 {
                c64::new(f64::NAN, f64::NAN)
            } else {
                sum / weight_sum
            };
        }
        data.copy_from_slice(&out);
    }
}

/// User-facing smoothness settings (DP3 `smoothness*` keys).
#[derive(Debug, Clone, PartialEq)]
pub struct SmoothnessSettings {
    /// Kernel size [Hz] (`smoothnessconstraint`).
    pub(crate) bandwidth_hz: f64,
    /// `smoothnessreffrequency` [Hz]; 0 disables.
    pub(crate) ref_frequency_hz: f64,
    /// `smoothnessspectralexponent`; DP3 default -1.
    pub(crate) spectral_exponent: f64,
    /// `smoothnessrefdistance` [m]; 0 disables.
    pub(crate) ref_distance_m: f64,
    /// `smoothnessrefantenna`; `None` means the first tile.
    pub(crate) ref_tile: Option<String>,
    /// `smoothness_kernel_truncation`.
    pub(crate) kernel_truncation: bool,
}

impl Default for SmoothnessSettings {
    fn default() -> Self {
        SmoothnessSettings {
            bandwidth_hz: 0.0,
            ref_frequency_hz: 0.0,
            spectral_exponent: -1.0,
            ref_distance_m: 0.0,
            ref_tile: None,
            kernel_truncation: true,
        }
    }
}

#[derive(Debug, Error)]
pub enum ConstraintError {
    #[error("Unknown smoothness reference tile '{0}'")]
    UnknownRefTile(String),

    #[error("The smoothness constraint kernel size must be positive, got {0}")]
    NonPositiveBandwidth(f64),
}

/// DP3's `CalculateAntennaSmoothnessFactors`: tiles closer to the reference
/// get more smoothing (factor `ref_distance / distance`), the reference tile
/// gets the largest factor. `positions` are per *unflagged* tile.
pub(crate) fn antenna_smoothness_factors(
    ref_distance: f64,
    ref_index: usize,
    positions: &[[f64; 3]],
) -> Vec<f64> {
    if ref_distance == 0.0 {
        return vec![1.0; positions.len()];
    }
    let mut factors = Vec::with_capacity(positions.len());
    let mut max_factor: f64 = 0.0;
    let r = positions[ref_index];
    for (i, p) in positions.iter().enumerate() {
        if i == ref_index {
            factors.push(0.0);
        } else {
            let d = ((r[0] - p[0]).powi(2) + (r[1] - p[1]).powi(2) + (r[2] - p[2]).powi(2)).sqrt();
            let f = ref_distance / d;
            max_factor = max_factor.max(f);
            factors.push(f);
        }
    }
    factors[ref_index] = max_factor;
    factors
}

/// Everything needed to constrain the solutions of one timeblock across all
/// of its unflagged chanblocks.
#[derive(Debug, Clone)]
pub struct Constraints {
    pub(crate) mode: SolveMode,
    smoother: Option<KernelSmoother>,
    /// Per unflagged tile; only used with a smoother.
    antenna_factors: Vec<f64>,
    /// DP3's `max_tec_delay_wraps`.
    pub(crate) max_tec_delay_wraps: usize,
    /// Unflagged chanblock frequencies [Hz], in solution order.
    freqs: Vec<f64>,
    /// Relative frequencies for the phase fits (`ν/ν_0`).
    rel_freqs: Vec<f64>,
    /// Weights per (unflagged tile, unflagged chanblock), e.g. summed data
    /// weights. Zero weight excludes a point from the fits and smoothing.
    tile_weights: Array2<f64>,
}

impl Constraints {
    /// Build the constraints. `tile_positions` and `tile_names` are per
    /// *unflagged* tile. `tile_weights` has shape (unflagged tiles, unflagged
    /// chanblocks); `None` means uniform weights.
    pub(crate) fn new(
        mode: SolveMode,
        smoothness: Option<&SmoothnessSettings>,
        max_tec_delay_wraps: usize,
        chanblock_freqs: Vec<f64>,
        tile_names: &[String],
        tile_positions: &[[f64; 3]],
        tile_weights: Option<Array2<f64>>,
    ) -> Result<Constraints, ConstraintError> {
        let num_tiles = tile_names.len();
        let num_chanblocks = chanblock_freqs.len();
        let (smoother, antenna_factors) = match smoothness {
            Some(s) if s.bandwidth_hz != 0.0 => {
                if s.bandwidth_hz < 0.0 || !s.bandwidth_hz.is_finite() {
                    return Err(ConstraintError::NonPositiveBandwidth(s.bandwidth_hz));
                }
                let ref_index = match &s.ref_tile {
                    None => 0,
                    Some(name) => tile_names
                        .iter()
                        .position(|n| n == name)
                        .ok_or_else(|| ConstraintError::UnknownRefTile(name.clone()))?,
                };
                let factors =
                    antenna_smoothness_factors(s.ref_distance_m, ref_index, tile_positions);
                let smoother = KernelSmoother::new(
                    chanblock_freqs.clone(),
                    s.bandwidth_hz,
                    s.ref_frequency_hz,
                    s.spectral_exponent,
                    s.kernel_truncation,
                );
                (Some(smoother), factors)
            }
            _ => (None, vec![1.0; num_tiles]),
        };
        let tile_weights =
            tile_weights.unwrap_or_else(|| Array2::ones((num_tiles, num_chanblocks)));
        assert_eq!(tile_weights.dim(), (num_tiles, num_chanblocks));
        let ref_freq = chanblock_freqs.first().copied().unwrap_or(1.0);
        let rel_freqs = chanblock_freqs.iter().map(|f| f / ref_freq).collect();
        Ok(Constraints {
            mode,
            smoother,
            antenna_factors,
            max_tec_delay_wraps,
            freqs: chanblock_freqs,
            rel_freqs,
            tile_weights,
        })
    }

    /// Is there anything to do?
    pub(crate) fn is_active(&self) -> bool {
        self.mode != SolveMode::FullJones || self.smoother.is_some()
    }

    /// The reference frequency of the relative frequencies used by the phase
    /// fits.
    pub(crate) fn ref_freq(&self) -> f64 {
        self.freqs.first().copied().unwrap_or(1.0)
    }

    /// Apply all constraints to the solutions of one timeblock, in DP3's
    /// order (smoothness, then the mode's phase model). `di_jones` has shape
    /// (unflagged tiles, unflagged chanblocks); `failed` has the same shape
    /// and marks solutions that must not be touched or used.
    ///
    /// Returns the per-tile phase-model fits when the mode fits one.
    pub(crate) fn apply(
        &self,
        mut di_jones: ArrayViewMut2<Jones<f64>>,
        failed: ArrayView2<bool>,
    ) -> Option<Vec<PhaseFit>> {
        let (num_tiles, num_chanblocks) = di_jones.dim();
        assert_eq!(failed.dim(), (num_tiles, num_chanblocks));

        // Per-element projection.
        if self.mode != SolveMode::FullJones {
            di_jones
                .iter_mut()
                .zip(failed.iter())
                .filter(|(_, &f)| !f)
                .for_each(|(j, _)| *j = self.mode.project(*j));
        }

        // Smoothness across chanblocks, per tile and polarisation.
        if let Some(smoother) = &self.smoother {
            di_jones
                .outer_iter_mut()
                .into_par_iter()
                .zip(failed.outer_iter())
                .zip(self.tile_weights.outer_iter())
                .zip(self.antenna_factors.par_iter())
                .for_each(|(((mut jones, failed), tile_weights), &antenna_factor)| {
                    let factor = if antenna_factor > 0.0 {
                        1.0 / antenna_factor
                    } else {
                        1.0
                    };
                    let mut data = vec![c64::default(); num_chanblocks];
                    let mut weights = vec![0.0; num_chanblocks];
                    for pol in 0..4 {
                        if self.mode != SolveMode::FullJones && (pol == 1 || pol == 2) {
                            continue;
                        }
                        for ((d, w), ((j, &f), &tw)) in data
                            .iter_mut()
                            .zip(weights.iter_mut())
                            .zip(jones.iter().zip(failed.iter()).zip(tile_weights.iter()))
                        {
                            let v = j[pol];
                            if !f && v.re.is_finite() && v.im.is_finite() && tw > 0.0 {
                                *d = v;
                                *w = tw;
                            } else {
                                *d = c64::default();
                                *w = 0.0;
                            }
                        }
                        smoother.smooth(&mut data, &weights, factor);
                        for ((j, &f), &d) in jones.iter_mut().zip(failed.iter()).zip(data.iter()) {
                            if !f && d.re.is_finite() && d.im.is_finite() {
                                j[pol] = d;
                            }
                        }
                    }
                    // Smoothing moves phase-only solutions off the unit
                    // circle; re-project.
                    if self.mode != SolveMode::FullJones {
                        for (j, &f) in jones.iter_mut().zip(failed.iter()) {
                            if !f {
                                *j = self.mode.project(*j);
                            }
                        }
                    }
                });
        }

        // Phase model across chanblocks.
        let terms = self.mode.phase_terms()?;
        self.apply_reference_tile(di_jones.view_mut(), failed);
        let fits: Vec<PhaseFit> = di_jones
            .outer_iter_mut()
            .into_par_iter()
            .zip(failed.outer_iter())
            .zip(self.tile_weights.outer_iter())
            .map(|((mut jones, failed), tile_weights)| {
                let mut phases = vec![0.0; num_chanblocks];
                let mut weights = vec![0.0; num_chanblocks];
                for ((p, w), ((j, &f), &tw)) in phases
                    .iter_mut()
                    .zip(weights.iter_mut())
                    .zip(jones.iter().zip(failed.iter()).zip(tile_weights.iter()))
                {
                    let v = j[0];
                    if !f && v.re.is_finite() && v.im.is_finite() {
                        *p = v.arg();
                        *w = tw;
                    }
                }
                let fit = fit_phase_model(
                    &self.rel_freqs,
                    &phases,
                    &weights,
                    terms,
                    self.max_tec_delay_wraps,
                );
                for ((j, &f), &x) in jones
                    .iter_mut()
                    .zip(failed.iter())
                    .zip(self.rel_freqs.iter())
                {
                    if !f {
                        let p = c64::from_polar(1.0, wrap(fit.eval(x)));
                        *j = Jones::from([p, c64::default(), c64::default(), p]);
                    }
                }
                fit
            })
            .collect();
        Some(fits)
    }

    /// DP3's `Constraint::ApplyReferenceAntenna`: divide all tiles' scalar
    /// solutions by those of the first tile with at least 20% unflagged
    /// chanblocks, per chanblock.
    fn apply_reference_tile(
        &self,
        mut di_jones: ArrayViewMut2<Jones<f64>>,
        failed: ArrayView2<bool>,
    ) {
        let (num_tiles, num_chanblocks) = di_jones.dim();
        let ref_tile = (0..num_tiles)
            .find(|&t| {
                let ok = failed.row(t).iter().filter(|&&f| !f).count();
                ok as f64 > 0.2 * num_chanblocks as f64
            })
            .unwrap_or(0);
        for c in 0..num_chanblocks {
            if failed[(ref_tile, c)] {
                continue;
            }
            let r = unit(di_jones[(ref_tile, c)][0]);
            for t in 0..num_tiles {
                if t == ref_tile || failed[(t, c)] {
                    continue;
                }
                let j = &mut di_jones[(t, c)];
                let s = j[0] / r;
                *j = Jones::from([s, c64::default(), c64::default(), s]);
            }
            let one = c64::new(1.0, 0.0);
            di_jones[(ref_tile, c)] = Jones::from([one, c64::default(), c64::default(), one]);
        }
    }
}

/// Parse a frequency with an optional unit ("2MHz", "400kHz", "2e6") to Hz.
pub(crate) fn parse_freq_hz(s: &str) -> Result<f64, crate::unit_parsing::UnitParseError> {
    use crate::unit_parsing::{parse_freq, FreqFormat};
    let (number, unit) = parse_freq(s)?;
    Ok(match unit {
        None | Some(FreqFormat::Hz) => number,
        Some(FreqFormat::kHz) => number * 1e3,
        Some(FreqFormat::MHz) => number * 1e6,
    })
}

impl FromStr for SmoothnessSettings {
    type Err = String;

    /// Only the kernel size, e.g. "2MHz" or "2e6"; the rest come from their
    /// own arguments.
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let bandwidth_hz = parse_freq_hz(s).map_err(|e| e.to_string())?;
        Ok(SmoothnessSettings {
            bandwidth_hz,
            ..Default::default()
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use std::f64::consts::TAU;

    #[test]
    fn solve_mode_names_match_dp3() {
        assert_eq!(
            SolveMode::from_str("fulljones").unwrap(),
            SolveMode::FullJones
        );
        assert_eq!(
            SolveMode::from_str("diagonal").unwrap(),
            SolveMode::Diagonal
        );
        assert_eq!(
            SolveMode::from_str("scalarphase").unwrap(),
            SolveMode::ScalarPhase
        );
        assert_eq!(
            SolveMode::from_str("tec+phase").unwrap(),
            SolveMode::TecAndPhase
        );
        assert_eq!(
            SolveMode::from_str("tecandphase").unwrap(),
            SolveMode::TecAndPhase
        );
        assert_eq!(
            SolveMode::from_str("tec+phase+delay").unwrap(),
            SolveMode::TecPhaseAndDelay
        );
        assert_eq!(
            SolveMode::from_str("phase+delay").unwrap(),
            SolveMode::PhaseAndDelay
        );
        assert!(SolveMode::from_str("rotation").is_err());
        assert_eq!(SolveMode::TecPhaseAndDelay.to_string(), "tec+phase+delay");
    }

    #[test]
    fn projections() {
        let j = Jones::from([
            c64::new(2.0, 2.0),
            c64::new(0.1, 0.0),
            c64::new(0.0, 0.1),
            c64::new(0.0, -3.0),
        ]);
        assert_eq!(SolveMode::FullJones.project(j), j);
        let d = SolveMode::Diagonal.project(j);
        assert_eq!(d[0], j[0]);
        assert_eq!(d[1], c64::default());
        assert_eq!(d[2], c64::default());
        assert_eq!(d[3], j[3]);
        let p = SolveMode::DiagonalPhase.project(j);
        assert_abs_diff_eq!(p[0].norm(), 1.0, epsilon = 1e-12);
        assert_abs_diff_eq!(p[0].arg(), j[0].arg(), epsilon = 1e-12);
        assert_abs_diff_eq!(p[3].arg(), j[3].arg(), epsilon = 1e-12);
        let a = SolveMode::DiagonalAmplitude.project(j);
        assert_abs_diff_eq!(a[0], c64::new(8f64.sqrt(), 0.0), epsilon = 1e-12);
        assert_abs_diff_eq!(a[3], c64::new(3.0, 0.0), epsilon = 1e-12);
        let s = SolveMode::Scalar.project(j);
        assert_abs_diff_eq!(s[0], c64::new(1.0, -0.5), epsilon = 1e-12);
        assert_eq!(s[0], s[3]);
        let sp = SolveMode::ScalarPhase.project(j);
        assert_abs_diff_eq!(sp[0].norm(), 1.0, epsilon = 1e-12);
        assert_abs_diff_eq!(sp[0].arg(), c64::new(1.0, -0.5).arg(), epsilon = 1e-12);
        let sa = SolveMode::ScalarAmplitude.project(j);
        assert_abs_diff_eq!(
            sa[0],
            c64::new((8f64.sqrt() + 3.0) / 2.0, 0.0),
            epsilon = 1e-12
        );
        // Phase modes project like scalarphase element-wise.
        assert_eq!(SolveMode::TecPhaseAndDelay.project(j), sp);
        // A zero gain in a phase-only mode becomes unity, not NaN.
        let z = SolveMode::ScalarPhase.project(Jones::default());
        assert_eq!(z[0], c64::new(1.0, 0.0));
    }

    #[test]
    fn wrap_phases() {
        assert_abs_diff_eq!(wrap(0.1), 0.1);
        assert_abs_diff_eq!(wrap(TAU + 0.1), 0.1, epsilon = 1e-12);
        assert_abs_diff_eq!(wrap(-TAU - 0.1), -0.1, epsilon = 1e-12);
        assert_abs_diff_eq!(wrap(std::f64::consts::PI), -std::f64::consts::PI);
    }

    fn band() -> Vec<f64> {
        // 768 x 40 kHz MWA-like band, relative to the first channel.
        let f0 = 167.0e6;
        (0..768).map(|i| (f0 + i as f64 * 40e3) / f0).collect()
    }

    #[test]
    fn linear_solve_recovers_unwrapped_model() {
        let x = band();
        let truth = PhaseFit {
            a: 0.3,
            b: -0.2,
            c: 0.5,
        };
        let y: Vec<f64> = x.iter().map(|&x| truth.eval(x)).collect();
        let w = vec![1.0; x.len()];
        let all = PhaseTerms {
            tec: true,
            offset: true,
            delay: true,
        };
        let fit = linear_phase_solve(&x, &y, &w, all).unwrap();
        assert_abs_diff_eq!(fit.a, truth.a, epsilon = 1e-9);
        assert_abs_diff_eq!(fit.b, truth.b, epsilon = 1e-9);
        assert_abs_diff_eq!(fit.c, truth.c, epsilon = 1e-9);
        // Disabled terms stay zero.
        let only_delay = PhaseTerms {
            tec: false,
            offset: false,
            delay: true,
        };
        let y2: Vec<f64> = x.iter().map(|&x| 0.7 * x).collect();
        let fit = linear_phase_solve(&x, &y2, &w, only_delay).unwrap();
        assert_abs_diff_eq!(fit.c, 0.7, epsilon = 1e-9);
        assert_eq!(fit.a, 0.0);
        assert_eq!(fit.b, 0.0);
        // All weights zero -> singular.
        assert!(linear_phase_solve(&x, &y, &vec![0.0; x.len()], all).is_none());
    }

    #[test]
    fn grid_search_recovers_wrapped_tec_offset_delay() {
        let x = band();
        // Several wraps across the band in both the TEC and delay terms.
        let truth = PhaseFit {
            a: 40.0,
            b: 1.0,
            c: -35.0,
        };
        let w = vec![1.0; x.len()];
        let y: Vec<f64> = x.iter().map(|&x| wrap(truth.eval(x))).collect();
        let fit = fit_phase_model(
            &x,
            &y,
            &w,
            PhaseTerms {
                tec: true,
                offset: true,
                delay: true,
            },
            5,
        );
        for (&x, &y) in x.iter().zip(&y) {
            assert_abs_diff_eq!(wrap(fit.eval(x) - y), 0.0, epsilon = 1e-6);
        }
        assert_abs_diff_eq!(phase_cost(&x, &y, &w, &fit), 0.0, epsilon = 1e-9);
    }

    #[test]
    fn grid_search_pure_delay_with_flags_and_noise() {
        let x = band();
        let truth = PhaseFit {
            a: 0.0,
            b: 0.4,
            c: 20.0,
        };
        let mut w = vec![1.0; x.len()];
        // Flag the edges and every 16th channel, as MWA data would be.
        for (i, w) in w.iter_mut().enumerate() {
            if i % 32 < 2 || i % 32 > 29 || i % 32 == 16 {
                *w = 0.0;
            }
        }
        let y: Vec<f64> = x
            .iter()
            .enumerate()
            .map(|(i, &x)| wrap(truth.eval(x) + 0.05 * ((i * 7919) % 13) as f64 / 13.0 - 0.025))
            .collect();
        let fit = fit_phase_model(
            &x,
            &y,
            &w,
            PhaseTerms {
                tec: false,
                offset: true,
                delay: true,
            },
            5,
        );
        assert_eq!(fit.a, 0.0);
        assert_abs_diff_eq!(fit.c, truth.c, epsilon = 2e-2);
        assert_abs_diff_eq!(wrap(fit.b - truth.b), 0.0, epsilon = 2e-2);
    }

    #[test]
    fn grid_search_offset_near_pi() {
        // Phases straddling ±π must still fit.
        let x = band();
        let truth = PhaseFit {
            a: 0.0,
            b: std::f64::consts::PI - 0.01,
            c: 0.02,
        };
        let w = vec![1.0; x.len()];
        let y: Vec<f64> = x
            .iter()
            .enumerate()
            .map(|(i, &x)| wrap(truth.eval(x) + 0.04 * ((i * 7919) % 13) as f64 / 13.0 - 0.02))
            .collect();
        let fit = fit_phase_model(
            &x,
            &y,
            &w,
            PhaseTerms {
                tec: false,
                offset: true,
                delay: true,
            },
            5,
        );
        assert_abs_diff_eq!(wrap(fit.b - truth.b), 0.0, epsilon = 1e-2);
        assert_abs_diff_eq!(fit.c, truth.c, epsilon = 1e-2);
    }

    #[test]
    fn phase_fit_physical_units() {
        let ref_freq = 150e6;
        // 1 TECU at 150 MHz.
        let a = TEC_TO_PHASE / ref_freq; // phase at x=1 is a
        let fit = PhaseFit {
            a,
            b: 0.5,
            c: TAU * ref_freq * 1e-9,
        };
        let (tec, phase, delay) = fit.to_physical(ref_freq);
        assert_abs_diff_eq!(tec, 1.0, epsilon = 1e-12);
        assert_abs_diff_eq!(phase, 0.5);
        assert_abs_diff_eq!(delay, 1e-9, epsilon = 1e-20);
    }

    #[test]
    fn kernel_matches_dp3() {
        let s = KernelSmoother::new(vec![0.0, 1.0], 2.0e6, 0.0, -1.0, true);
        assert_abs_diff_eq!(s.kernel(0.0), 1.0);
        // sigma = bandwidth / 3.
        assert_abs_diff_eq!(s.kernel(2.0e6 / 3.0), (-1.0f64).exp(), epsilon = 1e-12);
        assert_eq!(s.kernel(2.0e6 + 1.0), 0.0);
        let s = KernelSmoother::new(vec![0.0, 1.0], 2.0e6, 0.0, -1.0, false);
        assert!(s.kernel(2.0e6 + 1.0) > 0.0);
    }

    #[test]
    fn smoothing_preserves_constants_and_fills_from_neighbours() {
        let freqs: Vec<f64> = (0..64).map(|i| 150e6 + i as f64 * 40e3).collect();
        let s = KernelSmoother::new(freqs.clone(), 400e3, 0.0, -1.0, true);
        let mut data = vec![c64::new(1.5, -0.5); 64];
        let w = vec![1.0; 64];
        s.smooth(&mut data, &w, 1.0);
        for d in &data {
            assert_abs_diff_eq!(*d, c64::new(1.5, -0.5), epsilon = 1e-12);
        }
        // Zero-weight points take the smoothed value of their neighbours; a
        // point with no weighted neighbours within the kernel becomes NaN.
        let mut data = vec![c64::new(1.0, 0.0); 64];
        let mut w = vec![1.0; 64];
        data[10] = c64::new(100.0, 0.0);
        w[10] = 0.0;
        w[40..64].fill(0.0);
        s.smooth(&mut data, &w, 1.0);
        assert_abs_diff_eq!(data[10], c64::new(1.0, 0.0), epsilon = 1e-12);
        assert!(data[63].re.is_nan());
        assert!(data[39].re.is_finite());
    }

    #[test]
    fn smoothing_reduces_noise_and_respects_kernel_factor() {
        let freqs: Vec<f64> = (0..256).map(|i| 150e6 + i as f64 * 40e3).collect();
        let noise = |i: usize| ((i * 7919) % 17) as f64 / 17.0 - 0.5;
        let raw: Vec<c64> = (0..256).map(|i| c64::new(1.0 + noise(i), 0.0)).collect();
        let w = vec![1.0; 256];
        let rms = |d: &[c64]| {
            (d.iter().map(|z| (z.re - 1.0).powi(2)).sum::<f64>() / d.len() as f64).sqrt()
        };
        let s = KernelSmoother::new(freqs.clone(), 1e6, 0.0, -1.0, true);
        let mut a = raw.clone();
        s.smooth(&mut a, &w, 1.0);
        // A larger kernel-size factor means less smoothing (DP3 semantics).
        let mut b = raw.clone();
        s.smooth(&mut b, &w, 4.0);
        assert!(rms(&a) < 0.3 * rms(&raw));
        assert!(rms(&b) > rms(&a));
        assert!(rms(&b) < rms(&raw));
    }

    #[test]
    fn smoothing_ref_frequency_scales_kernel() {
        // With a reference frequency and exponent -1 the kernel size scales
        // as (freq / ref), so the top of the band is smoothed more (the
        // 1/freq behaviour of TEC phases).
        let freqs: Vec<f64> = (0..256).map(|i| 100e6 + i as f64 * 400e3).collect();
        let noise = |i: usize| ((i * 7919) % 17) as f64 / 17.0 - 0.5;
        let raw: Vec<c64> = (0..256).map(|i| c64::new(1.0 + noise(i), 0.0)).collect();
        let w = vec![1.0; 256];
        let s = KernelSmoother::new(freqs.clone(), 4e6, 100e6, -1.0, true);
        let mut a = raw.clone();
        s.smooth(&mut a, &w, 1.0);
        let rms = |d: &[c64]| {
            (d.iter().map(|z| (z.re - 1.0).powi(2)).sum::<f64>() / d.len() as f64).sqrt()
        };
        assert!(rms(&a[8..64]) > rms(&a[192..248]));
    }

    #[test]
    fn antenna_factors_match_dp3() {
        let pos = [[0.0, 0.0, 0.0], [100.0, 0.0, 0.0], [0.0, 400.0, 0.0]];
        assert_eq!(antenna_smoothness_factors(0.0, 0, &pos), vec![1.0; 3]);
        let f = antenna_smoothness_factors(200.0, 0, &pos);
        assert_abs_diff_eq!(f[1], 2.0);
        assert_abs_diff_eq!(f[2], 0.5);
        // The reference gets the maximum factor.
        assert_abs_diff_eq!(f[0], 2.0);
    }

    fn scalar(p: f64) -> Jones<f64> {
        let z = c64::from_polar(1.0, p);
        Jones::from([z, c64::default(), c64::default(), z])
    }

    #[test]
    fn constraints_fit_linear_phase_across_chanblocks() {
        let num_tiles = 4;
        let freqs: Vec<f64> = (0..96).map(|i| 150e6 + i as f64 * 80e3).collect();
        let names: Vec<String> = (0..num_tiles).map(|i| format!("Tile{i}")).collect();
        let pos = vec![[0.0; 3]; num_tiles];
        let c = Constraints::new(
            SolveMode::PhaseAndDelay,
            None,
            5,
            freqs.clone(),
            &names,
            &pos,
            None,
        )
        .unwrap();
        assert!(c.is_active());
        // Per-tile delays and offsets, plus a per-chanblock common phase that
        // the reference tile must absorb, plus noise.
        let delays = [0.0, 3e-9, -5e-9, 1e-9];
        let offsets = [0.0, 0.3, -1.0, 2.0];
        let noise = |t: usize, i: usize| ((t * 31 + i * 7919) % 11) as f64 / 11.0 * 0.1 - 0.05;
        let mut jones = Array2::from_elem((num_tiles, freqs.len()), Jones::identity());
        for t in 0..num_tiles {
            for (i, &f) in freqs.iter().enumerate() {
                let common = (i as f64 * 0.37).sin() * 2.0;
                jones[(t, i)] = scalar(TAU * delays[t] * f + offsets[t] + common + noise(t, i));
            }
        }
        let failed = Array2::from_elem(jones.dim(), false);
        let fits = c.apply(jones.view_mut(), failed.view()).unwrap();
        assert_eq!(fits.len(), num_tiles);
        for t in 0..num_tiles {
            let (tec, phase, delay) = fits[t].to_physical(c.ref_freq());
            assert_eq!(tec, 0.0);
            // Referenced to tile 0.
            assert_abs_diff_eq!(delay, delays[t] - delays[0], epsilon = 3e-10);
            // The offset is poorly constrained against the delay over a
            // narrow band.
            assert_abs_diff_eq!(wrap(phase - (offsets[t] - offsets[0])), 0.0, epsilon = 1e-1);
            for (i, &f) in freqs.iter().enumerate() {
                let j = jones[(t, i)];
                assert_abs_diff_eq!(j[0].norm(), 1.0, epsilon = 1e-12);
                assert_eq!(j[1], c64::default());
                assert_eq!(j[0], j[3]);
                let expected = TAU * (delays[t] - delays[0]) * f + offsets[t] - offsets[0];
                assert_abs_diff_eq!(wrap(j[0].arg() - expected), 0.0, epsilon = 1e-1);
            }
        }
    }

    #[test]
    fn constraints_smooth_diagonal_gains_and_skip_failed() {
        let num_tiles = 3;
        let freqs: Vec<f64> = (0..128).map(|i| 150e6 + i as f64 * 40e3).collect();
        let names: Vec<String> = (0..num_tiles).map(|i| format!("Tile{i}")).collect();
        let pos = vec![[0.0; 3]; num_tiles];
        let smooth = SmoothnessSettings {
            bandwidth_hz: 1e6,
            ..Default::default()
        };
        let c = Constraints::new(
            SolveMode::Diagonal,
            Some(&smooth),
            5,
            freqs.clone(),
            &names,
            &pos,
            None,
        )
        .unwrap();
        let noise = |t: usize, i: usize| ((t * 31 + i * 7919) % 11) as f64 / 11.0 * 0.4 - 0.2;
        let mut jones = Array2::from_elem((num_tiles, freqs.len()), Jones::identity());
        for t in 0..num_tiles {
            for i in 0..freqs.len() {
                jones[(t, i)] = Jones::from([
                    c64::new(2.0 + noise(t, i), 0.5),
                    c64::new(0.3, 0.3),
                    c64::new(0.3, 0.3),
                    c64::new(1.0, -0.5 + noise(t, i)),
                ]);
            }
        }
        let mut failed = Array2::from_elem(jones.dim(), false);
        failed[(1, 5)] = true;
        let before = jones.clone();
        assert!(c.apply(jones.view_mut(), failed.view()).is_none());
        let rms = |t: usize, pol: usize, target: c64, d: &Array2<Jones<f64>>| {
            (d.row(t)
                .iter()
                .map(|j| (j[pol] - target).norm_sqr())
                .sum::<f64>()
                / d.ncols() as f64)
                .sqrt()
        };
        for t in 0..num_tiles {
            assert!(
                rms(t, 0, c64::new(2.0, 0.5), &jones)
                    < 0.3 * rms(t, 0, c64::new(2.0, 0.5), &before)
            );
            assert!(
                rms(t, 3, c64::new(1.0, -0.5), &jones)
                    < 0.3 * rms(t, 3, c64::new(1.0, -0.5), &before)
            );
            for (i, j) in jones.row(t).iter().enumerate() {
                if failed[(t, i)] {
                    continue;
                }
                assert_eq!(j[1], c64::default());
                assert_eq!(j[2], c64::default());
            }
        }
        // The failed solution is untouched.
        assert_eq!(jones[(1, 5)], before[(1, 5)]);
    }

    #[test]
    fn constraints_errors() {
        let names = vec!["Tile0".to_string()];
        let pos = vec![[0.0; 3]];
        let smooth = SmoothnessSettings {
            bandwidth_hz: 1e6,
            ref_distance_m: 10.0,
            ref_tile: Some("nope".to_string()),
            ..Default::default()
        };
        assert!(matches!(
            Constraints::new(
                SolveMode::Diagonal,
                Some(&smooth),
                5,
                vec![1e8],
                &names,
                &pos,
                None
            ),
            Err(ConstraintError::UnknownRefTile(_))
        ));
        let smooth = SmoothnessSettings {
            bandwidth_hz: -1.0,
            ..Default::default()
        };
        assert!(matches!(
            Constraints::new(
                SolveMode::Diagonal,
                Some(&smooth),
                5,
                vec![1e8],
                &names,
                &pos,
                None
            ),
            Err(ConstraintError::NonPositiveBandwidth(_))
        ));
        let c =
            Constraints::new(SolveMode::FullJones, None, 5, vec![1e8], &names, &pos, None).unwrap();
        assert!(!c.is_active());
        assert_eq!(
            "2MHz".parse::<SmoothnessSettings>().unwrap().bandwidth_hz,
            2e6
        );
        assert_eq!(
            "2e6".parse::<SmoothnessSettings>().unwrap().bandwidth_hz,
            2e6
        );
        assert!("2 furlongs".parse::<SmoothnessSettings>().is_err());
    }
}
