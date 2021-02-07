// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

/*!
Some helper mathematics.
 */

#![allow(dead_code)]

use crate::c64;

// Make traditional trigonometry possible.
pub(crate) fn sin(x: f64) -> f64 {
    x.sin()
}
pub(crate) fn cos(x: f64) -> f64 {
    x.cos()
}

/// Complex exponential. The argument is assumed to be purely imaginary.
///
/// This function doesn't actually use complex numbers; it just returns the real
/// and imag components from Euler's formula (i.e. e^{ix} = cos{x} + i sin{x}).
pub(crate) fn cexp(x: f64) -> c64 {
    let (s, c) = x.sin_cos();
    c64::new(c, s)
}
