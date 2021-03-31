// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

/*!
Code for general Jones matrix math.

It's not ideal to use LAPACK for matrix multiplies or inverses, because it is
not possible to optimise only for 2x2 matrices. Here, we supply the math for
these special cases.

Parts of the code are derived from Torrance Hodgson's MWAjl:
https://github.com/torrance/MWAjl/blob/master/src/matrix2x2.jl
 */

#[cfg(feature = "beam")]
pub mod cache;

use num::Complex;

use crate::c64;

#[derive(Debug, PartialEq, Clone, Default)]
pub struct Jones([c64; 4]);

const JONES_ZERO: Jones = Jones([c64::new(0.0, 0.0); 4]);

const JONES_IDENTITY: Jones = Jones([
    c64::new(1.0, 0.0),
    c64::new(0.0, 0.0),
    c64::new(0.0, 0.0),
    c64::new(1.0, 0.0),
]);

impl Jones {
    pub fn identity() -> Self {
        JONES_IDENTITY
    }

    pub fn zero() -> Self {
        JONES_ZERO
    }

    /// From an input Jones matrix, get a copy that has been Hermitian
    /// conjugated (J^H).
    ///
    /// # Examples
    ///
    /// ```
    /// # use mwa_hyperdrive_core::c64;
    /// # use mwa_hyperdrive_core::jones::Jones;
    /// # use approx::assert_abs_diff_eq;
    /// let j = Jones::from([
    ///     c64::new(1.0, 2.0),
    ///     c64::new(3.0, 4.0),
    ///     c64::new(5.0, 6.0),
    ///     c64::new(7.0, 8.0),
    /// ]);
    /// let jh = j.h();
    /// let expected = Jones::from([
    ///     c64::new(1.0, -2.0),
    ///     c64::new(5.0, -6.0),
    ///     c64::new(3.0, -4.0),
    ///     c64::new(7.0, -8.0),
    /// ]);
    /// assert_abs_diff_eq!(jh, expected, epsilon = 1e-10);
    /// ```
    #[inline(always)]
    pub fn h(&self) -> Self {
        Self::from([
            self[0].conj(),
            self[2].conj(),
            self[1].conj(),
            self[3].conj(),
        ])
    }

    /// Multiply by a Jones matrix which gets Hermitian conjugated (J^H).
    ///
    /// # Examples
    ///
    /// ```
    /// # use mwa_hyperdrive_core::c64;
    /// # use mwa_hyperdrive_core::jones::Jones;
    /// # use approx::assert_abs_diff_eq;
    /// let i = Jones::identity();
    /// let a = Jones::from([
    ///     c64::new(1.0, 2.0),
    ///     c64::new(3.0, 4.0),
    ///     c64::new(5.0, 6.0),
    ///     c64::new(7.0, 8.0),
    /// ]);
    /// // A^H is the conjugate transpose.
    /// let result = i.mul_hermitian(&a);
    /// let expected = Jones::from([
    ///     c64::new(1.0, -2.0),
    ///     c64::new(5.0, -6.0),
    ///     c64::new(3.0, -4.0),
    ///     c64::new(7.0, -8.0),
    /// ]);
    /// assert_abs_diff_eq!(result, expected, epsilon = 1e-10);
    /// ```
    #[inline(always)]
    pub fn mul_hermitian(&self, b: &Self) -> Self {
        let mut a = self.clone();
        a *= b.h();
        a
    }

    /// Get the inverse of the Jones matrix (J^I).
    ///
    /// Ideally, J^I . J = I. However it's possible that J is singular, in which
    /// case the contents of J^I are all NaN.
    #[inline(always)]
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

    /// Calculate J1 . A . J2^H, but with J1 and J2 multiplied as an outer
    /// product, and A is a 4-element vector.
    // See Jack's thorough analysis here:
    // https://github.com/JLBLine/polarisation_tests_for_FEE
    #[inline(always)]
    #[rustfmt::skip]
    pub fn outer_mul(j1: Jones, a: [c64; 4], j2: Jones) -> [c64; 4] {
        let g1x = j1[0];
        let g1y = j1[3];
        let g2x = j2[0].conj();
        let g2y = j2[3].conj();
        let d1x = j1[1];
        let d1y = j1[2];
        let d2x = j2[1].conj();
        let d2y = j2[2].conj();
        [
            (g1x * g2x + d1x * d2x) * a[0]
          + (g1x * g2x - d1x * d2x) * a[1]
          + (g1x * d2x + d1x * g2x) * a[2]
          + (g1x * d2x - d1x * g2x) * a[3] * Complex::i(),

            (g1x * d2y + d1x * g2y) * a[0]
          + (g1x * d2y - d1x * g2y) * a[1]
          + (g1x * g2y + d1x * d2y) * a[2]
          + (g1x * g2y - d1x * d2y) * a[3] * Complex::i(),

            (d1y * g2x + g1y * d2x) * a[0]
          + (d1y * g2x - g1y * d2x) * a[1]
          + (d1y * d2x + g1y * g2x) * a[2]
          + (d1y * d2x - g1y * g2x) * a[3] * Complex::i(),

            (d1y * d2y + g1y * g2y) * a[0]
          + (d1y * d2y - g1y * g2y) * a[1]
          + (d1y * g2y + g1y * d2y) * a[2]
          + (d1y * g2y - g1y * d2y) * a[3] * Complex::i(),
        ]
    }
}

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

impl From<[c64; 4]> for Jones {
    fn from(arr: [c64; 4]) -> Self {
        Self(arr)
    }
}

impl std::ops::Add<Jones> for Jones {
    type Output = Self;

    #[inline(always)]
    fn add(self, rhs: Jones) -> Self {
        let mut c = JONES_ZERO;
        let a = self.0;
        let b = rhs;
        c[0] = a[0] + b[0];
        c[1] = a[1] + b[1];
        c[2] = a[2] + b[2];
        c[3] = a[3] + b[3];
        c
    }
}

impl std::ops::AddAssign<Jones> for Jones {
    #[inline(always)]
    fn add_assign(&mut self, rhs: Jones) {
        self[0] += rhs[0];
        self[1] += rhs[1];
        self[2] += rhs[2];
        self[3] += rhs[3];
    }
}

impl std::ops::Sub<Jones> for Jones {
    type Output = Self;

    #[inline(always)]
    fn sub(self, rhs: Jones) -> Self {
        let mut c = JONES_ZERO;
        let a = self.0;
        let b = rhs;
        c[0] = a[0] - b[0];
        c[1] = a[1] - b[1];
        c[2] = a[2] - b[2];
        c[3] = a[3] - b[3];
        c
    }
}

impl std::ops::SubAssign<Jones> for Jones {
    #[inline(always)]
    fn sub_assign(&mut self, rhs: Jones) {
        self[0] -= rhs[0];
        self[1] -= rhs[1];
        self[2] -= rhs[2];
        self[3] -= rhs[3];
    }
}

impl std::ops::Sub<&Jones> for Jones {
    type Output = Self;

    #[inline(always)]
    fn sub(self, rhs: &Jones) -> Self {
        let mut c = JONES_ZERO;
        let a = self.0;
        let b = rhs;
        c[0] = a[0] - b[0];
        c[1] = a[1] - b[1];
        c[2] = a[2] - b[2];
        c[3] = a[3] - b[3];
        c
    }
}

impl std::ops::SubAssign<&Jones> for Jones {
    #[inline(always)]
    fn sub_assign(&mut self, rhs: &Jones) {
        self[0] -= rhs[0];
        self[1] -= rhs[1];
        self[2] -= rhs[2];
        self[3] -= rhs[3];
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

impl std::ops::Mul<&Jones> for Jones {
    type Output = Self;

    #[inline(always)]
    fn mul(self, rhs: &Jones) -> Self {
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

impl std::ops::Mul<f64> for Jones {
    type Output = Self;

    #[inline(always)]
    fn mul(self, rhs: f64) -> Self {
        let mut a = self.0;
        a[0] *= rhs;
        a[1] *= rhs;
        a[2] *= rhs;
        a[3] *= rhs;
        Jones(a)
    }
}

impl std::ops::MulAssign<f64> for Jones {
    #[inline(always)]
    fn mul_assign(&mut self, rhs: f64) {
        self[0] *= rhs;
        self[1] *= rhs;
        self[2] *= rhs;
        self[3] *= rhs;
    }
}

impl std::ops::Mul<c64> for Jones {
    type Output = Self;

    #[inline(always)]
    fn mul(self, rhs: c64) -> Self {
        let mut a = self.0;
        a[0] *= rhs;
        a[1] *= rhs;
        a[2] *= rhs;
        a[3] *= rhs;
        Jones(a)
    }
}

impl std::ops::MulAssign<c64> for Jones {
    #[inline(always)]
    fn mul_assign(&mut self, rhs: c64) {
        self[0] *= rhs;
        self[1] *= rhs;
        self[2] *= rhs;
        self[3] *= rhs;
    }
}

impl num::traits::Zero for Jones {
    #[inline]
    fn zero() -> Self {
        Jones::zero()
    }

    #[inline]
    fn is_zero(&self) -> bool {
        *self == Jones::zero()
    }
}

impl approx::AbsDiffEq for Jones {
    type Epsilon = f64;

    fn default_epsilon() -> f64 {
        f64::EPSILON
    }

    #[inline]
    fn abs_diff_eq(&self, other: &Self, epsilon: f64) -> bool {
        (self[0] - other[0]).norm() <= epsilon
            && (self[1] - other[1]).norm() <= epsilon
            && (self[2] - other[2]).norm() <= epsilon
            && (self[3] - other[3]).norm() <= epsilon
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::*;

    fn one_through_eight() -> Jones {
        Jones([
            c64::new(1.0, 2.0),
            c64::new(3.0, 4.0),
            c64::new(5.0, 6.0),
            c64::new(7.0, 8.0),
        ])
    }

    #[test]
    fn test_add() {
        let a = one_through_eight();
        let b = one_through_eight();
        let c = a + b;
        let expected_c = Jones([
            c64::new(2.0, 4.0),
            c64::new(6.0, 8.0),
            c64::new(10.0, 12.0),
            c64::new(14.0, 16.0),
        ]);
        assert_abs_diff_eq!(c, expected_c, epsilon = 1e-10);
    }

    #[test]
    fn test_sub() {
        let a = one_through_eight();
        let b = one_through_eight();
        let c = a - b;
        let expected_c = Jones::zero();
        assert_abs_diff_eq!(c, expected_c, epsilon = 1e-10);
    }

    #[test]
    fn test_mul() {
        let i = c64::new(1.0, 2.0);
        let a = Jones([i, i + 1.0, i + 2.0, i + 3.0]);
        let b = Jones([i * 2.0, i * 3.0, i * 4.0, i * 5.0]);
        let c = a * &b;
        let expected_c = Jones([
            c64::new(-14.0, 32.0),
            c64::new(-19.0, 42.0),
            c64::new(-2.0, 56.0),
            c64::new(-3.0, 74.0),
        ]);
        assert_abs_diff_eq!(c, expected_c, epsilon = 1e-10);
    }

    #[test]
    fn test_mul_assign() {
        let i = c64::new(1.0, 2.0);
        let mut a = Jones([i, i + 1.0, i + 2.0, i + 3.0]);
        let b = Jones([i * 2.0, i * 3.0, i * 4.0, i * 5.0]);
        a *= b;
        let expected = Jones([
            c64::new(-14.0, 32.0),
            c64::new(-19.0, 42.0),
            c64::new(-2.0, 56.0),
            c64::new(-3.0, 74.0),
        ]);
        assert_abs_diff_eq!(a, expected, epsilon = 1e-10);
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
        let expected = Jones([
            c64::new(1.0, -2.0),
            c64::new(5.0, -6.0),
            c64::new(3.0, -4.0),
            c64::new(7.0, -8.0),
        ]);
        assert_abs_diff_eq!(result, expected, epsilon = 1e-10);
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
        assert_abs_diff_eq!(result, expected, epsilon = 1e-10);
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

    #[test]
    fn test_outer_mul() {
        let j1 = Jones([
            c64::new(1.0, 0.0),
            c64::new(2.0, 0.0),
            c64::new(3.0, 0.0),
            c64::new(4.0, 0.0),
        ]);
        let j2 = Jones([
            c64::new(1.0, 0.0),
            c64::new(2.0, 0.0),
            c64::new(3.0, 0.0),
            c64::new(4.0, 0.0),
        ]);
        let a = [c64::new(1.0, 0.0); 4];
        let result = Jones::outer_mul(j1, a, j2);
        let expected = [
            c64::new(5.0 - 3.0 + 4.0, 0.0),
            c64::new(11.0 - 5.0 + 10.0, -2.0),
            c64::new(11.0 - 5.0 + 10.0, 2.0),
            c64::new(25.0 - 7.0 + 24.0, 0.0),
        ];
        // Pretend that the result of the computation is `Jones` so we can use
        // assert_abs_diff_eq.
        assert_abs_diff_eq!(Jones::from(result), Jones::from(expected), epsilon = 1e-10);
    }
}
