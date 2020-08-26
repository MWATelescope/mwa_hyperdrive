// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

/*!
Some helper mathematics.
*/

use num::complex::Complex64;

// Make traditional trigonometry possible.
pub fn sin(x: f64) -> f64 {
    x.sin()
}
pub fn cos(x: f64) -> f64 {
    x.cos()
}

/// Complex exponential. The argument is assumed to be purely imaginary.
///
/// This function doesn't actually use complex numbers; it just returns the real
/// and imag components from Euler's formula (i.e. e^{ix} = cos{x} + i sin{x}).
pub fn cexp(x: f64) -> Complex64 {
    let (s, c) = x.sin_cos();
    Complex64::new(c, s)
}
