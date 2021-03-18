// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

/*!
Some helper mathematics.
 */

#![allow(dead_code)]

use crate::c64;

/// Invert a 2x2 matrix (represented here as a single-dimension 4-element
/// array). If the matrix is singular, then the results are all NaN.
///
/// To invert complex 2x2 matrices, see the code attached to the `Jones` struct.
pub(crate) fn invert_2x2(x: &[f64; 4]) -> [f64; 4] {
    // X = [[A B]
    //      [C D]]
    // det = AD - BC
    let inv_determinant = 1.0 / (x[0] * x[3] - x[1] * x[2]);
    // X^I = 1/det [[D  -B]
    //              [-C  A]]
    [
        inv_determinant * x[3],
        -inv_determinant * x[1],
        -inv_determinant * x[2],
        inv_determinant * x[0],
    ]
}

// Make traditional trigonometry possible.
/// Sine.
pub(crate) fn sin(x: f64) -> f64 {
    x.sin()
}
/// Cosine.
pub(crate) fn cos(x: f64) -> f64 {
    x.cos()
}
/// Inverse tangent. y comes before x, like the C function.
pub(crate) fn atan2(y: f64, x: f64) -> f64 {
    y.atan2(x)
}

/// Complex exponential. The argument is assumed to be purely imaginary.
///
/// This function doesn't actually use complex numbers; it just returns the real
/// and imag components from Euler's formula (i.e. e^{ix} = cos{x} + i sin{x}).
pub(crate) fn cexp(x: f64) -> c64 {
    let (s, c) = x.sin_cos();
    c64::new(c, s)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_inverse() {
        let x = [1.0, 2.0, 3.0, 4.0];
        let result = invert_2x2(&x);
        let expected = [-2.0, 1.0, 1.5, -0.5];
        for (r, e) in result.iter().zip(expected.iter()) {
            assert_abs_diff_eq!(r, e);
        }

        // Verify that multiplying a matrix by its inverse gives the identity.
        let xi_x = [
            result[0] * x[0] + result[1] * x[2],
            result[0] * x[1] + result[1] * x[3],
            result[2] * x[0] + result[3] * x[2],
            result[2] * x[1] + result[3] * x[3],
        ];
        let expected = [1.0, 0.0, 0.0, 1.0];
        for (r, e) in xi_x.iter().zip(expected.iter()) {
            assert_abs_diff_eq!(r, e);
        }
    }

    // I don't like Rust's atan2. This test helps me sleep at night knowing I'm
    // using it correctly.
    #[test]
    fn atan2_is_correct() {
        assert_abs_diff_eq!(atan2(-2.0, 1.0), -1.1071487177940904);
    }
}
