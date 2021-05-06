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

use std::ops::{Add, AddAssign, Deref, DerefMut, Mul, MulAssign, Sub, SubAssign};

use num::{traits::NumAssign, Complex, Float, Num};

use crate::FluxDensity;

#[derive(Debug, PartialEq, Clone, Default)]
pub struct Jones<F: Float + Num>([Complex<F>; 4]);

impl<F: Float> Jones<F> {
    pub fn identity() -> Self {
        Self::from([
            Complex::new(F::one(), F::zero()),
            Complex::new(F::zero(), F::zero()),
            Complex::new(F::zero(), F::zero()),
            Complex::new(F::one(), F::zero()),
        ])
    }

    /// From an input Jones matrix, get a copy that has been Hermitian
    /// conjugated (`J^H`).
    #[inline(always)]
    pub fn h(&self) -> Self {
        Self::from([
            self[0].conj(),
            self[2].conj(),
            self[1].conj(),
            self[3].conj(),
        ])
    }

    /// Multiply by a Jones matrix which gets Hermitian conjugated (`J^H`).
    #[inline(always)]
    pub fn mul_hermitian(&self, b: &Self) -> Self {
        self.clone() * b.h()
    }

    /// "Divide" two Jones matrices. Another way of looking at what this
    /// function does is solving `C` in `C B = A`, i.e.
    ///
    /// `C B = A`
    ///
    /// `C B B^-1 = A B^-1`
    ///
    /// `C = A / B`
    ///
    /// If B is singular, all the results' elements are NaN.
    #[inline(always)]
    pub fn div(&self, b: &Self) -> Self {
        let inv_det = Complex::new(F::one(), F::zero()) / (b[0] * b[3] - b[1] * b[2]);
        Self::from([
            (self[0] * b[3] - self[1] * b[2]) * inv_det,
            (self[1] * b[0] - self[0] * b[1]) * inv_det,
            (self[2] * b[3] - self[3] * b[2]) * inv_det,
            (self[3] * b[0] - self[2] * b[1]) * inv_det,
        ])
    }

    /// Get the inverse of the Jones matrix (`J^I`).
    ///
    /// Ideally, `J^I . J = I`. However it's possible that `J` is singular, in
    /// which case the contents of `J^I` are all NaN.
    #[inline(always)]
    pub fn inv(&self) -> Self {
        let inv_det = Complex::new(F::one(), F::zero()) / (self[0] * self[3] - self[1] * self[2]);
        Self::from([
            inv_det * self[3],
            -inv_det * self[1],
            -inv_det * self[2],
            inv_det * self[0],
        ])
    }

    /// Call [`Complex::norm_sqr()`] on each element of a Jones matrix.
    #[inline(always)]
    pub fn norm_sqr(&self) -> [F; 4] {
        [
            self[0].norm_sqr(),
            self[1].norm_sqr(),
            self[2].norm_sqr(),
            self[3].norm_sqr(),
        ]
    }

    pub fn axb(a: &Self, b: &Self) -> Self {
        a.clone() * b
    }

    pub fn axbh(a: &Self, b: &Self) -> Self {
        a.clone() * b.h()
    }
}

impl<F: Float + NumAssign> Jones<F> {
    pub fn plus_axb(c: &mut Self, a: &Self, b: &Self) {
        c[0] += a[0] * b[0] + a[1] * b[2];
        c[1] += a[0] * b[1] + a[1] * b[3];
        c[2] += a[2] * b[0] + a[3] * b[2];
        c[3] += a[2] * b[1] + a[3] * b[3];
    }

    pub fn plus_ahxb(c: &mut Self, a: &Self, b: &Self) {
        c[0] += a[0].conj() * b[0] + a[2].conj() * b[2];
        c[1] += a[0].conj() * b[1] + a[2].conj() * b[3];
        c[2] += a[1].conj() * b[0] + a[3].conj() * b[2];
        c[3] += a[1].conj() * b[1] + a[3].conj() * b[3];
    }
}

impl<F: Float> Deref for Jones<F> {
    type Target = [Complex<F>; 4];

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<F: Float> DerefMut for Jones<F> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl<F: Float> From<[Complex<F>; 4]> for Jones<F> {
    fn from(arr: [Complex<F>; 4]) -> Self {
        Self(arr)
    }
}

impl<F: Float> Add<Jones<F>> for Jones<F> {
    type Output = Self;

    #[inline(always)]
    fn add(self, rhs: Self) -> Self {
        Self::from([
            self[0] + rhs[0],
            self[1] + rhs[1],
            self[2] + rhs[2],
            self[3] + rhs[3],
        ])
    }
}

impl<F: Float + NumAssign> AddAssign<Jones<F>> for Jones<F> {
    #[inline(always)]
    fn add_assign(&mut self, rhs: Self) {
        self[0] += rhs[0];
        self[1] += rhs[1];
        self[2] += rhs[2];
        self[3] += rhs[3];
    }
}

impl<F: Float + NumAssign> AddAssign<&Jones<F>> for Jones<F> {
    #[inline(always)]
    fn add_assign(&mut self, rhs: &Self) {
        self[0] += rhs[0];
        self[1] += rhs[1];
        self[2] += rhs[2];
        self[3] += rhs[3];
    }
}

impl<F: Float> Sub<Jones<F>> for Jones<F> {
    type Output = Self;

    #[inline(always)]
    fn sub(self, rhs: Self) -> Self {
        Self::from([
            self[0] - rhs[0],
            self[1] - rhs[1],
            self[2] - rhs[2],
            self[3] - rhs[3],
        ])
    }
}

impl<F: Float> Sub<&Jones<F>> for Jones<F> {
    type Output = Self;

    #[inline(always)]
    fn sub(self, rhs: &Self) -> Self {
        Self::from([
            self[0] - rhs[0],
            self[1] - rhs[1],
            self[2] - rhs[2],
            self[3] - rhs[3],
        ])
    }
}

impl<F: Float> Sub<&mut Jones<F>> for Jones<F> {
    type Output = Self;

    #[inline(always)]
    fn sub(self, rhs: &mut Self) -> Self {
        Self::from([
            self[0] - rhs[0],
            self[1] - rhs[1],
            self[2] - rhs[2],
            self[3] - rhs[3],
        ])
    }
}

impl<F: Float + NumAssign> SubAssign<Jones<F>> for Jones<F> {
    #[inline(always)]
    fn sub_assign(&mut self, rhs: Self) {
        self[0] -= rhs[0];
        self[1] -= rhs[1];
        self[2] -= rhs[2];
        self[3] -= rhs[3];
    }
}

impl<F: Float + NumAssign> SubAssign<&Jones<F>> for Jones<F> {
    #[inline(always)]
    fn sub_assign(&mut self, rhs: &Self) {
        self[0] -= rhs[0];
        self[1] -= rhs[1];
        self[2] -= rhs[2];
        self[3] -= rhs[3];
    }
}

impl<F: Float> Mul<F> for Jones<F> {
    type Output = Self;

    #[inline(always)]
    fn mul(self, rhs: F) -> Self {
        Jones::from([self[0] * rhs, self[1] * rhs, self[2] * rhs, self[3] * rhs])
    }
}

impl<F: Float> Mul<Complex<F>> for Jones<F> {
    type Output = Self;

    #[inline(always)]
    fn mul(self, rhs: Complex<F>) -> Self {
        Jones::from([self[0] * rhs, self[1] * rhs, self[2] * rhs, self[3] * rhs])
    }
}

impl<F: Float> Mul<Jones<F>> for Jones<F> {
    type Output = Self;

    #[inline(always)]
    fn mul(self, rhs: Self) -> Self {
        Self::from([
            self[0] * rhs[0] + self[1] * rhs[2],
            self[0] * rhs[1] + self[1] * rhs[3],
            self[2] * rhs[0] + self[3] * rhs[2],
            self[2] * rhs[1] + self[3] * rhs[3],
        ])
    }
}

impl<F: Float> Mul<&Jones<F>> for Jones<F> {
    type Output = Self;

    #[inline(always)]
    fn mul(self, rhs: &Self) -> Self {
        Self::from([
            self[0] * rhs[0] + self[1] * rhs[2],
            self[0] * rhs[1] + self[1] * rhs[3],
            self[2] * rhs[0] + self[3] * rhs[2],
            self[2] * rhs[1] + self[3] * rhs[3],
        ])
    }
}

impl<F: Float + NumAssign> MulAssign<F> for Jones<F> {
    #[inline(always)]
    fn mul_assign(&mut self, rhs: F) {
        self[0] *= rhs;
        self[1] *= rhs;
        self[2] *= rhs;
        self[3] *= rhs;
    }
}

impl<F: Float + NumAssign> MulAssign<Complex<F>> for Jones<F> {
    #[inline(always)]
    fn mul_assign(&mut self, rhs: Complex<F>) {
        self[0] *= rhs;
        self[1] *= rhs;
        self[2] *= rhs;
        self[3] *= rhs;
    }
}

impl<F: Float + MulAssign> MulAssign<Jones<F>> for Jones<F> {
    #[inline(always)]
    fn mul_assign(&mut self, rhs: Self) {
        self.0 = [
            self[0] * rhs[0] + self[1] * rhs[2],
            self[0] * rhs[1] + self[1] * rhs[3],
            self[2] * rhs[0] + self[3] * rhs[2],
            self[2] * rhs[1] + self[3] * rhs[3],
        ];
    }
}

impl<F: Float + MulAssign> MulAssign<&Jones<F>> for Jones<F> {
    #[inline(always)]
    fn mul_assign(&mut self, rhs: &Self) {
        self.0 = [
            self[0] * rhs[0] + self[1] * rhs[2],
            self[0] * rhs[1] + self[1] * rhs[3],
            self[2] * rhs[0] + self[3] * rhs[2],
            self[2] * rhs[1] + self[3] * rhs[3],
        ];
    }
}

impl<F: Float> num::traits::Zero for Jones<F> {
    #[inline]
    fn zero() -> Self {
        Self::from([Complex::new(F::zero(), F::zero()); 4])
    }

    #[inline]
    fn is_zero(&self) -> bool {
        *self == Self::zero()
    }
}

impl From<Jones<f32>> for Jones<f64> {
    #[inline]
    fn from(j_c32: Jones<f32>) -> Self {
        Self::from([
            Complex::new(j_c32[0].re as _, j_c32[0].im as _),
            Complex::new(j_c32[1].re as _, j_c32[1].im as _),
            Complex::new(j_c32[2].re as _, j_c32[2].im as _),
            Complex::new(j_c32[3].re as _, j_c32[3].im as _),
        ])
    }
}

impl From<Jones<f64>> for Jones<f32> {
    #[inline]
    fn from(j_c64: Jones<f64>) -> Self {
        Self::from([
            Complex::new(j_c64[0].re as _, j_c64[0].im as _),
            Complex::new(j_c64[1].re as _, j_c64[1].im as _),
            Complex::new(j_c64[2].re as _, j_c64[2].im as _),
            Complex::new(j_c64[3].re as _, j_c64[3].im as _),
        ])
    }
}

impl From<FluxDensity> for Jones<f64> {
    #[inline]
    fn from(fd: FluxDensity) -> Self {
        Self::from([
            Complex::new(fd.i + fd.q, 0.0),
            Complex::new(fd.u, fd.v),
            Complex::new(fd.u, -fd.v),
            Complex::new(fd.i - fd.q, 0.0),
        ])
    }
}

impl From<&FluxDensity> for Jones<f64> {
    #[inline]
    fn from(fd: &FluxDensity) -> Self {
        Self::from([
            Complex::new(fd.i + fd.q, 0.0),
            Complex::new(fd.u, fd.v),
            Complex::new(fd.u, -fd.v),
            Complex::new(fd.i - fd.q, 0.0),
        ])
    }
}

#[cfg(test)]
impl<F: Float> approx::AbsDiffEq for Jones<F> {
    type Epsilon = F;

    fn default_epsilon() -> F {
        F::epsilon()
    }

    #[inline]
    fn abs_diff_eq(&self, other: &Self, epsilon: F) -> bool {
        (self[0] - other[0]).norm() <= epsilon
            && (self[1] - other[1]).norm() <= epsilon
            && (self[2] - other[2]).norm() <= epsilon
            && (self[3] - other[3]).norm() <= epsilon
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{c32, c64};
    use approx::*;

    fn one_through_eight() -> Jones<f64> {
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
        let expected_c = Jones::default();
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
        let ident = Jones::identity();
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
    fn test_div() {
        let a = Jones([
            c64::new(1.0, 0.0),
            c64::new(2.0, 0.0),
            c64::new(3.0, 0.0),
            c64::new(4.0, 0.0),
        ]);
        let b = Jones([
            c64::new(2.0, 0.0),
            c64::new(3.0, 0.0),
            c64::new(4.0, 0.0),
            c64::new(5.0, 0.0),
        ]);
        let expected = Jones([
            c64::new(1.5, 0.0),
            c64::new(-0.5, 0.0),
            c64::new(0.5, 0.0),
            c64::new(0.5, 0.0),
        ]);
        assert_abs_diff_eq!(a.div(&b), expected, epsilon = 1e-10);

        let a = Jones([
            c32::new(-1295920.9, -1150667.5),
            c32::new(-1116357.0, 1234393.3),
            c32::new(-4028358.8, -281923.3),
            c32::new(-325126.38, 3929351.3),
        ]);
        let b = Jones([
            c32::new(1377080.5, 0.0),
            c32::new(5765.743, -1371240.9),
            c32::new(5765.743, 1371240.9),
            c32::new(1365932.0, 0.0),
        ]);
        let expected = Jones([
            c32::new(-107.08988, -72.43006),
            c32::new(72.34611, -106.29652),
            c32::new(-169.56223, 57.398113),
            c32::new(-57.14347, -167.58751),
        ]);
        assert_abs_diff_eq!(a.div(&b), expected, epsilon = 1e-5);
    }

    #[test]
    fn test_div_singular() {
        let a = Jones([
            c64::new(1.0, 0.0),
            c64::new(2.0, 0.0),
            c64::new(2.0, 0.0),
            c64::new(4.0, 0.0),
        ]);
        for j in a.div(&a).iter() {
            assert!(j.re.is_nan());
            assert!(j.im.is_nan());
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
        let expected = Jones::identity();
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
    fn test_axb() {
        let i = c64::new(1.0, 2.0);
        let a = Jones([i, i + 1.0, i + 2.0, i + 3.0]);
        let b = Jones([i * 2.0, i * 3.0, i * 4.0, i * 5.0]);
        let c = Jones::axb(&a, &b);
        let expected_c = Jones([
            c64::new(-14.0, 32.0),
            c64::new(-19.0, 42.0),
            c64::new(-2.0, 56.0),
            c64::new(-3.0, 74.0),
        ]);
        assert_abs_diff_eq!(c, expected_c, epsilon = 1e-10);
    }

    #[test]
    fn test_axbh() {
        let i = c64::new(1.0, 2.0);
        let a = Jones([i, i + 1.0, i + 2.0, i + 3.0]);
        let b = Jones([i * 2.0, i * 3.0, i * 4.0, i * 5.0]);
        let c = Jones::axbh(&a, &b);
        let expected_c = Jones([
            c64::new(28.0, -6.0),
            c64::new(50.0, -10.0),
            c64::new(38.0, -26.0),
            c64::new(68.0, -46.0),
        ]);
        assert_abs_diff_eq!(c, expected_c, epsilon = 1e-10);
    }

    #[test]
    fn test_plus_axb() {
        let a = one_through_eight();
        let b = one_through_eight();
        let mut c = Jones::default();
        Jones::plus_axb(&mut c, &a, &b);
        let expected_c = Jones([
            c64::new(-12.0, 42.0),
            c64::new(-16.0, 62.0),
            c64::new(-20.0, 98.0),
            c64::new(-24.0, 150.0),
        ]);
        assert_abs_diff_eq!(c, expected_c, epsilon = 1e-10);
    }

    #[test]
    fn test_plus_ahxb() {
        let a = one_through_eight();
        let b = one_through_eight();
        let mut c = Jones::default();
        Jones::plus_ahxb(&mut c, &a, &b);
        let expected_c = Jones([
            c64::new(66.0, 0.0),
            c64::new(94.0, -4.0),
            c64::new(94.0, 4.0),
            c64::new(138.0, 0.0),
        ]);
        assert_abs_diff_eq!(c, expected_c, epsilon = 1e-10);
    }

    #[test]
    fn test_from_flux_density() {
        let fd = FluxDensity {
            freq: 170e6,
            i: 0.058438801501144624,
            q: -0.3929914018344019,
            u: -0.3899498110659575,
            v: -0.058562589895788,
        };
        let result = Jones::from(fd);
        assert_abs_diff_eq!(result[0], c64::new(fd.i + fd.q, 0.0));
        assert_abs_diff_eq!(result[1], c64::new(fd.u, fd.v));
        assert_abs_diff_eq!(result[2], c64::new(fd.u, -fd.v));
        assert_abs_diff_eq!(result[3], c64::new(fd.i - fd.q, 0.0));
    }

    #[test]
    fn test_from_flux_density_borrowed() {
        let fd = FluxDensity {
            freq: 170e6,
            i: 0.058438801501144624,
            q: -0.3929914018344019,
            u: -0.3899498110659575,
            v: -0.058562589895788,
        };
        let result = Jones::from(&fd);
        assert_abs_diff_eq!(result[0], c64::new(fd.i + fd.q, 0.0));
        assert_abs_diff_eq!(result[1], c64::new(fd.u, fd.v));
        assert_abs_diff_eq!(result[2], c64::new(fd.u, -fd.v));
        assert_abs_diff_eq!(result[3], c64::new(fd.i - fd.q, 0.0));
    }
}
