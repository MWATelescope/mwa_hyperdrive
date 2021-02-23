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

#[cfg(feature = "beam")]
pub mod cache;

use crate::c64;

#[derive(Debug, PartialEq, Clone)]
pub struct Jones([c64; 4]);

const JONES_ZERO: Jones = Jones([c64::new(0.0, 0.0); 4]);

const JONES_IDENTITY: Jones = Jones([
    c64::new(1.0, 0.0),
    c64::new(0.0, 0.0),
    c64::new(0.0, 0.0),
    c64::new(1.0, 0.0),
]);

impl Jones {
    pub fn identity() -> Jones {
        JONES_IDENTITY
    }

    pub fn zero() -> Jones {
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

    #[inline]
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
}
