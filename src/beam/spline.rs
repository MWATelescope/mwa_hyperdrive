// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Cubic-spline interpolation of beam responses in frequency.
//!
//! The FEE beam is only defined at the frequencies in its file (every 1.28
//! MHz). Between them, beam responses are interpolated with a not-a-knot cubic
//! spline through the responses at the file's frequencies ("knots") that span
//! the fine channels being modelled. The analytic beams are defined at any
//! frequency, but are splined the same way from knots on the 1.28 MHz grid so
//! that the beam-response arrays don't scale with the number of fine channels
//! (see [`crate::beam::BEAM_KNOT_SPACING_HZ`]).
//!
//! This is the same interpolation scheme used to make the 10 kHz FEE file
//! `MWA_embedded_element_pattern_rev2_interp_167_197MHz.h5` (a not-a-knot
//! `scipy.interpolate.CubicSpline` through the real and imaginary parts of
//! each spherical-wave coefficient, with knots spanning the band). The
//! un-normalised beam response is linear in the coefficients and a spline is
//! linear in the values it is fit to, so splining un-normalised responses
//! would equal splining the coefficients. hyperdrive's responses are
//! normalised to zenith (each element divided by the magnitude of a zenith
//! response at the same frequency), which is not linear in frequency, so
//! splining the normalised responses differs slightly from the dense file
//! (which normalises after splining): about 1e-6 at zenith to 1e-4 at 25
//! degrees elevation. The coarse-channel harmonics are suppressed identically.
//!
//! On each interval `[x_i, x_{i+1}]` (width `h`, `t = (x - x_i) / h`) the
//! spline is
//!
//! `y = (1 - t) y_i + t y_{i+1} + h^2 / 6 [((1 - t)^3 - (1 - t)) M_i + (t^3 - t) M_{i+1}]`
//!
//! where `M` are the spline's second derivatives at the knots, `M = S y` for a
//! matrix `S` that depends only on the knots. So the modellers store the beam
//! responses at the knots followed by their second derivatives (`S` applied
//! along the frequency axis), and every fine channel needs only the two
//! bracketing knots of each, with four weights.

use ndarray::prelude::*;

/// How to get the beam response at each fine channel from beam responses at
/// knot frequencies.
#[derive(Debug, Clone)]
pub(crate) struct BeamFreqSpline {
    /// The knot frequencies \[Hz\], ascending and unique.
    pub(crate) knots_hz: Vec<f64>,

    /// For each fine channel, the indices of the knots at or below and at or
    /// above it, and the weights `[a, b, c, d]` of the responses at those
    /// knots and of their second derivatives, i.e. `y = a y_lo + b y_hi + c
    /// M_lo + d M_hi`.
    pub(crate) interp: Vec<(usize, usize, [f64; 4])>,

    /// The `num_knots` x `num_knots` matrix `S` that gives the spline's second
    /// derivatives at the knots from the values at the knots (`M = S y`).
    /// `None` when every fine channel is at a knot (e.g. channels exactly at
    /// coarse-channel centres), in which case second derivatives are never
    /// needed.
    pub(crate) second_derivs: Option<Array2<f64>>,
}

impl BeamFreqSpline {
    /// Set up the spline. `knots_hz` must be ascending and unique and should
    /// span `freqs_hz`; fine channels outside the knots are clamped to the
    /// nearest knot.
    pub(crate) fn new(knots_hz: Vec<f64>, freqs_hz: &[f64]) -> BeamFreqSpline {
        assert!(
            knots_hz.windows(2).all(|w| w[0] < w[1]),
            "beam knot frequencies must be ascending and unique"
        );
        // Work in MHz to keep the second derivatives at a sensible scale (for
        // single-precision GPU code).
        let x: Vec<f64> = knots_hz.iter().map(|f| f / 1e6).collect();
        let mut on_knots = true;
        let interp: Vec<(usize, usize, [f64; 4])> = freqs_hz
            .iter()
            .map(|&freq| {
                let f = freq / 1e6;
                let n = x.len();
                assert!(n > 0, "no beam knot frequencies");
                if f <= x[0] {
                    on_knots &= f == x[0];
                    return (0, 0, [1.0, 0.0, 0.0, 0.0]);
                }
                if f >= x[n - 1] {
                    on_knots &= f == x[n - 1];
                    return (n - 1, n - 1, [1.0, 0.0, 0.0, 0.0]);
                }
                let i_hi = x.partition_point(|&k| k < f);
                if x[i_hi] == f {
                    return (i_hi, i_hi, [1.0, 0.0, 0.0, 0.0]);
                }
                on_knots = false;
                let i_lo = i_hi - 1;
                let h = x[i_hi] - x[i_lo];
                let t = (f - x[i_lo]) / h;
                let u = 1.0 - t;
                let h2_6 = h * h / 6.0;
                (
                    i_lo,
                    i_hi,
                    [u, t, h2_6 * (u * u * u - u), h2_6 * (t * t * t - t)],
                )
            })
            .collect();

        let second_derivs = if on_knots {
            None
        } else {
            Some(not_a_knot_second_deriv_matrix(&x))
        };

        BeamFreqSpline {
            knots_hz,
            interp,
            second_derivs,
        }
    }

    /// The number of knot rows of beam responses the modellers store per
    /// unique tile: the knots, and their second derivatives if needed.
    pub(crate) fn num_rows(&self) -> usize {
        if self.second_derivs.is_some() {
            2 * self.knots_hz.len()
        } else {
            self.knots_hz.len()
        }
    }
}

/// The matrix `S` such that the second derivatives of the not-a-knot cubic
/// spline through `(x, y)` at the knots `x` are `S y`. Matches
/// `scipy.interpolate.CubicSpline(x, y, bc_type="not-a-knot")`, including its
/// special cases: two knots give a straight line and three a parabola.
pub(crate) fn not_a_knot_second_deriv_matrix(x: &[f64]) -> Array2<f64> {
    let n = x.len();
    if n < 3 {
        return Array2::zeros((n, n));
    }
    let h: Vec<f64> = x.windows(2).map(|w| w[1] - w[0]).collect();
    // Solve A M = B y for M, i.e. S = A^-1 B.
    let mut a = Array2::<f64>::zeros((n, n));
    let mut b = Array2::<f64>::zeros((n, n));
    // Interior rows: continuity of the first derivative.
    for i in 1..n - 1 {
        a[(i, i - 1)] = h[i - 1];
        a[(i, i)] = 2.0 * (h[i - 1] + h[i]);
        a[(i, i + 1)] = h[i];
        b[(i, i - 1)] = 6.0 / h[i - 1];
        b[(i, i)] = -6.0 / h[i - 1] - 6.0 / h[i];
        b[(i, i + 1)] = 6.0 / h[i];
    }
    if n == 3 {
        // A parabola: constant second derivative.
        a[(0, 0)] = 1.0;
        a[(0, 1)] = -1.0;
        a[(2, 1)] = 1.0;
        a[(2, 2)] = -1.0;
    } else {
        // Not-a-knot: continuous third derivative at the second and
        // second-last knots.
        a[(0, 0)] = -1.0 / h[0];
        a[(0, 1)] = 1.0 / h[0] + 1.0 / h[1];
        a[(0, 2)] = -1.0 / h[1];
        a[(n - 1, n - 3)] = -1.0 / h[n - 3];
        a[(n - 1, n - 2)] = 1.0 / h[n - 3] + 1.0 / h[n - 2];
        a[(n - 1, n - 1)] = -1.0 / h[n - 2];
    }
    solve(a, b)
}

/// Solve `A X = B` with Gaussian elimination and partial pivoting.
fn solve(mut a: Array2<f64>, mut b: Array2<f64>) -> Array2<f64> {
    let n = a.nrows();
    for col in 0..n {
        let pivot = (col..n)
            .max_by(|&i, &j| a[(i, col)].abs().total_cmp(&a[(j, col)].abs()))
            .expect("not empty");
        assert!(a[(pivot, col)] != 0.0, "singular spline system");
        if pivot != col {
            for k in 0..n {
                a.swap((pivot, k), (col, k));
                b.swap((pivot, k), (col, k));
            }
        }
        for row in 0..n {
            if row == col {
                continue;
            }
            let factor = a[(row, col)] / a[(col, col)];
            if factor == 0.0 {
                continue;
            }
            for k in 0..n {
                a[(row, k)] -= factor * a[(col, k)];
                b[(row, k)] -= factor * b[(col, k)];
            }
        }
    }
    for row in 0..n {
        let d = a[(row, row)];
        b.row_mut(row).mapv_inplace(|v| v / d);
    }
    b
}

#[cfg(test)]
mod tests {
    use approx::assert_abs_diff_eq;

    use super::*;

    /// Evaluate the spline through `y` (one value per knot) at each fine
    /// channel.
    fn eval(spline: &BeamFreqSpline, y: &[f64]) -> Vec<f64> {
        let m: Vec<f64> = match &spline.second_derivs {
            Some(s) => s.dot(&ArrayView1::from(y)).to_vec(),
            None => vec![0.0; y.len()],
        };
        spline
            .interp
            .iter()
            .map(|&(lo, hi, [a, b, c, d])| a * y[lo] + b * y[hi] + c * m[lo] + d * m[hi])
            .collect()
    }

    #[test]
    fn reproduces_cubics_exactly() {
        // A not-a-knot spline reproduces any cubic.
        let knots: Vec<f64> = (0..8).map(|i| 166.4e6 + 1.28e6 * i as f64).collect();
        let freqs: Vec<f64> = (0..200).map(|i| 166.0e6 + 0.05e6 * i as f64).collect();
        let spline = BeamFreqSpline::new(knots.clone(), &freqs);
        let p = |f: f64| {
            let x = f / 1e6 - 170.0;
            0.3 - 0.2 * x + 0.05 * x * x - 0.01 * x * x * x
        };
        let y: Vec<f64> = knots.iter().map(|&f| p(f)).collect();
        for (f, v) in freqs.iter().zip(eval(&spline, &y)) {
            let f_c = f.clamp(knots[0], *knots.last().unwrap());
            assert_abs_diff_eq!(v, p(f_c), epsilon = 1e-12);
        }
    }

    #[test]
    fn matches_scipy_not_a_knot() {
        // scipy.interpolate.CubicSpline([0, 1, 2, 3, 4], [0, 1, 0, 2, 1])(x)
        // for x = 0.5, 1.5, 2.5, 3.5 (in MHz here).
        let knots: Vec<f64> = (0..5).map(|i| i as f64 * 1e6).collect();
        let freqs = [0.5e6, 1.5e6, 2.5e6, 3.5e6];
        let spline = BeamFreqSpline::new(knots, &freqs);
        let v = eval(&spline, &[0.0, 1.0, 0.0, 2.0, 1.0]);
        let expected = [1.234375, 0.265625, 0.828125, 2.421875];
        for (v, e) in v.into_iter().zip(expected) {
            assert_abs_diff_eq!(v, e, epsilon = 1e-12);
        }
    }

    #[test]
    fn small_knot_counts() {
        // Two knots: a line. Three: a parabola.
        let spline = BeamFreqSpline::new(vec![1e6, 2e6], &[1.25e6]);
        assert_abs_diff_eq!(eval(&spline, &[1.0, 3.0])[0], 1.5, epsilon = 1e-12);
        let spline = BeamFreqSpline::new(vec![0.0, 1e6, 2e6], &[0.5e6, 1.5e6]);
        // y = x^2
        let v = eval(&spline, &[0.0, 1.0, 4.0]);
        assert_abs_diff_eq!(v[0], 0.25, epsilon = 1e-12);
        assert_abs_diff_eq!(v[1], 2.25, epsilon = 1e-12);
    }

    #[test]
    fn channels_on_knots_need_no_second_derivatives() {
        let freqs = [150e6, 151e6, 152e6];
        let spline = BeamFreqSpline::new(freqs.to_vec(), &freqs);
        assert!(spline.second_derivs.is_none());
        assert_eq!(spline.num_rows(), 3);
        for (i, &(lo, hi, w)) in spline.interp.iter().enumerate() {
            assert_eq!((lo, hi, w), (i, i, [1.0, 0.0, 0.0, 0.0]));
        }
    }
}
