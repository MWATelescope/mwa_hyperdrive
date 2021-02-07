// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

/*!
Code for general Jones matrix math.

It's not ideal to use LAPACK for matrix multiplies or inverses, because it is
not possible to optimise only for 2x2 matrices. Here, we supply the math for
these special cases.

Derived from Torrance Hodgson's MWAjl:
// https://github.com/torrance/MWAjl/blob/master/src/matrix2x2.jl
 */

use crate::c64;

pub type Jones = [c64; 4];

pub const JONES_IDENTITY: Jones = [
    c64::new(1.0, 0.0),
    c64::new(0.0, 0.0),
    c64::new(0.0, 0.0),
    c64::new(1.0, 0.0),
];

/// A.B = C
///
/// Multiply two Jones matrices.
#[inline(always)]
pub fn a_x_b(a: &Jones, b: &Jones) -> Jones {
    let mut c = [c64::new(0.0, 0.0); 4];
    c[0] = a[0] * b[0] + a[1] * b[2];
    c[1] = a[0] * b[1] + a[1] * b[3];
    c[2] = a[2] * b[0] + a[3] * b[2];
    c[3] = a[2] * b[1] + a[3] * b[3];
    c
}

/// A.B^H = C
///
/// Multiply two Jones matrices, of which the second is Hermitian conjugated.
#[inline(always)]
pub fn a_x_bh(a: &Jones, b: &Jones) -> Jones {
    let mut c = [c64::new(0.0, 0.0); 4];
    c[0] = a[0] * b[0].conj() + a[1] * b[1].conj();
    c[1] = a[0] * b[2].conj() + a[1] * b[3].conj();
    c[2] = a[2] * b[0].conj() + a[3] * b[1].conj();
    c[3] = a[2] * b[2].conj() + a[3] * b[3].conj();
    c
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::*;

    #[test]
    fn test_a_x_b() {
        let i = c64::new(1.0, 2.0);
        let a = [i, i + 1.0, i + 2.0, i + 3.0];
        let b = [i * 2.0, i * 3.0, i * 4.0, i * 5.0];
        let c = a_x_b(&a, &b);
        let expected_c = [
            c64::new(-14.0, 32.0),
            c64::new(-19.0, 42.0),
            c64::new(-2.0, 56.0),
            c64::new(-3.0, 74.0),
        ];
        for (result, expected) in c.iter().zip(expected_c.iter()) {
            assert_abs_diff_eq!(result.re, expected.re, epsilon = 1e-10);
            assert_abs_diff_eq!(result.im, expected.im, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_a_x_bh() {
        let ident = [
            c64::new(1.0, 0.0),
            c64::new(0.0, 0.0),
            c64::new(0.0, 0.0),
            c64::new(1.0, 0.0),
        ];
        let a = [
            c64::new(1.0, 2.0),
            c64::new(3.0, 4.0),
            c64::new(5.0, 6.0),
            c64::new(7.0, 8.0),
        ];
        // A^H is the conjugate transpose.
        let c = a_x_bh(&ident, &a);
        let expected_c = [
            c64::new(1.0, -2.0),
            c64::new(5.0, -6.0),
            c64::new(3.0, -4.0),
            c64::new(7.0, -8.0),
        ];
        for (result, expected) in c.iter().zip(expected_c.iter()) {
            assert_abs_diff_eq!(result.re, expected.re, epsilon = 1e-10);
            assert_abs_diff_eq!(result.im, expected.im, epsilon = 1e-10);
        }
    }
}
