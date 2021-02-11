// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

/*!
Code to cache Jones matrices.

Because it's very likely that the Jones matrices derived from different tiles
are exactly the same (especially if we ignore the positions of the tiles), the
code here exists to cache Jones matrices given the parameters that would be
given to beam code.
 */

use crate::c64;

pub struct Jones(pub [c64; 4]);

const JONES_ZERO: Jones = Jones([c64::new(0.0, 0.0); 4]);

const JONES_IDENTITY: Jones = Jones([
    c64::new(1.0, 0.0),
    c64::new(0.0, 0.0),
    c64::new(0.0, 0.0),
    c64::new(1.0, 0.0),
]);

impl std::ops::Deref for Jones {
    type Target = [c64; 4];

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl std::ops::DerefMut for Jones {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl std::ops::Mul<Jones> for Jones {
    type Output = Self;

    #[inline(always)]
    fn mul(self, rhs: Jones) -> Self {
        let mut c = JONES_ZERO;
        let a = self.0;
        let b = rhs;
        c[0] = a[0] * b[0] + a[1] * b[2];
        c[1] = a[0] * b[1] + a[1] * b[3];
        c[2] = a[2] * b[0] + a[3] * b[2];
        c[3] = a[2] * b[1] + a[3] * b[3];
        c
    }
}

impl std::ops::MulAssign<Jones> for Jones {
    #[inline(always)]
    fn mul_assign(&mut self, rhs: Jones) {
        let mut c = JONES_ZERO;
        let a = self.0;
        let b = rhs;
        c[0] = a[0] * b[0] + a[1] * b[2];
        c[1] = a[0] * b[1] + a[1] * b[3];
        c[2] = a[2] * b[0] + a[3] * b[2];
        c[3] = a[2] * b[1] + a[3] * b[3];
        self.0 = *c;
    }
}

impl Jones {
    pub fn identity() -> Jones {
        JONES_IDENTITY
    }

    pub fn zero() -> Jones {
        JONES_ZERO
    }

    /// Multiply by a Jones matrix and store the result in a new Jones matrix.
    #[inline(always)]
    pub fn mul(&self, b: &Self) -> Self {
        let mut c = JONES_ZERO;
        let a = self;
        c[0] = a[0] * b[0] + a[1] * b[2];
        c[1] = a[0] * b[1] + a[1] * b[3];
        c[2] = a[2] * b[0] + a[3] * b[2];
        c[3] = a[2] * b[1] + a[3] * b[3];
        c
    }

    /// Multiply by a Jones matrix which gets Hermitian conjugated (J^H).
    #[inline(always)]
    pub fn mul_hermitian(&self, b: &Self) -> Self {
        let mut c = JONES_ZERO;
        let a = self;
        c[0] = a[0] * b[0].conj() + a[1] * b[1].conj();
        c[1] = a[0] * b[2].conj() + a[1] * b[3].conj();
        c[2] = a[2] * b[0].conj() + a[3] * b[1].conj();
        c[3] = a[2] * b[2].conj() + a[3] * b[3].conj();
        c
    }

    /// Get the inverse of the Jones matrix (J^I).
    ///
    /// Ideally, J^I . J = I. However it's possible that J is singular, in which
    /// case the contents of J^I are all NaN.
    pub fn inv(&self) -> Self {
        let mut inv = JONES_ZERO;
        let a = self;
        let inv_det = 1.0 / (a[0] * a[3] - a[1] * a[2]);
        inv[0] = inv_det * a[3];
        inv[1] = -inv_det * a[1];
        inv[2] = -inv_det * a[2];
        inv[3] = inv_det * a[0];
        inv
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::*;

    #[test]
    fn test_mul() {
        let i = c64::new(1.0, 2.0);
        let a = Jones([i, i + 1.0, i + 2.0, i + 3.0]);
        let b = Jones([i * 2.0, i * 3.0, i * 4.0, i * 5.0]);
        let c = a.mul(&b);
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
    fn test_ops_mul() {
        let i = c64::new(1.0, 2.0);
        let a = Jones([i, i + 1.0, i + 2.0, i + 3.0]);
        let b = Jones([i * 2.0, i * 3.0, i * 4.0, i * 5.0]);
        let c1 = a.mul(&b);
        let c2 = a * b;
        for (expected, result) in c1.iter().zip(c2.iter()) {
            assert_abs_diff_eq!(result.re, expected.re, epsilon = 1e-10);
            assert_abs_diff_eq!(result.im, expected.im, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_ops_mul_assign() {
        let i = c64::new(1.0, 2.0);
        let mut a = Jones([i, i + 1.0, i + 2.0, i + 3.0]);
        let b = Jones([i * 2.0, i * 3.0, i * 4.0, i * 5.0]);
        let c1 = a.mul(&b);
        a *= b;
        for (expected, result) in c1.iter().zip(a.iter()) {
            assert_abs_diff_eq!(result.re, expected.re, epsilon = 1e-10);
            assert_abs_diff_eq!(result.im, expected.im, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_mul_hermitian() {
        let ident = JONES_IDENTITY;
        let a = Jones([
            c64::new(1.0, 2.0),
            c64::new(3.0, 4.0),
            c64::new(5.0, 6.0),
            c64::new(7.0, 8.0),
        ]);
        // A^H is the conjugate transpose.
        let result = ident.mul_hermitian(&a);
        let expected_c = [
            c64::new(1.0, -2.0),
            c64::new(5.0, -6.0),
            c64::new(3.0, -4.0),
            c64::new(7.0, -8.0),
        ];
        for (res, exp) in result.iter().zip(expected_c.iter()) {
            assert_abs_diff_eq!(res.re, exp.re, epsilon = 1e-10);
            assert_abs_diff_eq!(res.im, exp.im, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_inv() {
        let a = Jones([
            c64::new(1.0, 2.0),
            c64::new(3.0, 4.0),
            c64::new(5.0, 6.0),
            c64::new(7.0, 8.0),
        ]);
        let result = a.inv() * a;
        let expected = JONES_IDENTITY;
        for (res, exp) in result.iter().zip(expected.iter()) {
            assert_abs_diff_eq!(res.re, exp.re, epsilon = 1e-10);
            assert_abs_diff_eq!(res.im, exp.im, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_inv_singular() {
        let a = Jones([
            c64::new(1.0, 0.0),
            c64::new(2.0, 0.0),
            c64::new(2.0, 0.0),
            c64::new(4.0, 0.0),
        ]);
        for j in a.inv().iter() {
            assert!(j.re.is_nan());
            assert!(j.im.is_nan());
        }
    }
}
