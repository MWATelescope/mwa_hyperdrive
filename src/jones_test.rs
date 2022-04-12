// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! A private Jones matrix type exclusively for testing.

use marlu::Jones;

use mwa_hyperdrive_common::{
    marlu,
    num_traits::{float::FloatCore, Float, Num},
    Complex,
};

#[derive(Clone, Copy, Default, PartialEq)]
pub(crate) struct TestJones<F: Float + Num>(Jones<F>);

impl<F: Float> From<Jones<F>> for TestJones<F> {
    #[inline]
    fn from(j: Jones<F>) -> Self {
        Self(j)
    }
}

impl<F: Float> From<[Complex<F>; 4]> for TestJones<F> {
    #[inline]
    fn from(j: [Complex<F>; 4]) -> Self {
        Self(Jones::from(j))
    }
}

impl<F: Float> From<[F; 8]> for TestJones<F> {
    #[inline]
    fn from(j: [F; 8]) -> Self {
        TestJones::from([
            Complex::<F>::new(j[0], j[1]),
            Complex::<F>::new(j[2], j[3]),
            Complex::<F>::new(j[4], j[5]),
            Complex::<F>::new(j[6], j[7]),
        ])
    }
}

impl<F: Float> std::ops::Deref for TestJones<F> {
    type Target = Jones<F>;

    #[inline]
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<F: Float> std::ops::DerefMut for TestJones<F> {
    #[inline]
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl std::fmt::Display for TestJones<f32> {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(
            f,
            "[[{:e}{:+e}j, {:e}{:+e}j] [{:e}{:+e}j, {:e}{:+e}j]]",
            self[0].re,
            self[0].im,
            self[1].re,
            self[1].im,
            self[2].re,
            self[2].im,
            self[3].re,
            self[3].im,
        )
    }
}

impl std::fmt::Display for TestJones<f64> {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(
            f,
            "[[{:e}{:+e}j, {:e}{:+e}j] [{:e}{:+e}j, {:e}{:+e}j]]",
            self[0].re,
            self[0].im,
            self[1].re,
            self[1].im,
            self[2].re,
            self[2].im,
            self[3].re,
            self[3].im,
        )
    }
}

impl std::fmt::Debug for TestJones<f32> {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(
            f,
            "[[{:e}{:+e}j, {:e}{:+e}j] [{:e}{:+e}j, {:e}{:+e}j]]",
            self[0].re,
            self[0].im,
            self[1].re,
            self[1].im,
            self[2].re,
            self[2].im,
            self[3].re,
            self[3].im,
        )
    }
}

impl std::fmt::Debug for TestJones<f64> {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(
            f,
            "[[{:e}{:+e}j, {:e}{:+e}j] [{:e}{:+e}j, {:e}{:+e}j]]",
            self[0].re,
            self[0].im,
            self[1].re,
            self[1].im,
            self[2].re,
            self[2].im,
            self[3].re,
            self[3].im,
        )
    }
}

impl<F: Float + approx::AbsDiffEq> approx::AbsDiffEq for TestJones<F>
where
    F::Epsilon: Clone,
{
    type Epsilon = F::Epsilon;

    #[inline]
    fn default_epsilon() -> F::Epsilon {
        F::default_epsilon()
    }

    #[inline]
    fn abs_diff_eq(&self, other: &Self, epsilon: F::Epsilon) -> bool {
        (0..4).all(|idx| Complex::<F>::abs_diff_eq(&self[idx], &other[idx], epsilon.clone()))
    }
}

impl<F: Float + FloatCore + approx::AbsDiffEq<Epsilon = F> + approx::RelativeEq> approx::RelativeEq
    for TestJones<F>
where
    F::Epsilon: Clone,
{
    #[inline]
    fn default_max_relative() -> F::Epsilon {
        F::default_epsilon()
    }

    #[inline]
    fn relative_eq(&self, other: &Self, epsilon: F::Epsilon, max_relative: F::Epsilon) -> bool {
        self.iter().zip(other.iter()).all(|(s, o)| {
            F::relative_eq(&s.re, &o.re, epsilon, max_relative)
                && F::relative_eq(&s.im, &o.im, epsilon, max_relative)
        })
    }

    #[inline]
    fn relative_ne(
        &self,
        other: &Self,
        epsilon: Self::Epsilon,
        max_relative: Self::Epsilon,
    ) -> bool {
        !Self::relative_eq(self, other, epsilon, max_relative)
    }
}
