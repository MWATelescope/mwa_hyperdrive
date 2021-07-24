// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Some helper mathematics.

#![allow(dead_code)]

use crate::c64;

// Make traditional trigonometry possible.
/// Sine.
///
/// # Examples
///
/// `assert_abs_diff_eq!(sin(FRAC_PI_6), 0.5);`
pub(crate) fn sin(x: f64) -> f64 {
    x.sin()
}

/// Cosine.
///
/// # Examples
///
/// `assert_abs_diff_eq!(cos(FRAC_PI_3), 0.5);`
pub(crate) fn cos(x: f64) -> f64 {
    x.cos()
}

/// Inverse tangent. y comes before x, like the C function.
///
/// # Examples
///
/// `assert_abs_diff_eq!(atan2(1, -1), 3.0 / 4.0 * PI);`
// I don't like Rust's atan2. This test helps me sleep at night knowing I'm
// using it correctly.
pub(crate) fn atan2(y: f64, x: f64) -> f64 {
    y.atan2(x)
}

/// Exponential.
///
/// # Examples
///
/// `assert_abs_diff_eq!(exp(1), 2.718281828);`
pub(crate) fn exp(x: f64) -> f64 {
    x.exp()
}

/// Complex exponential. The argument is assumed to be purely imaginary.
///
/// This function doesn't actually use complex numbers; it just returns the real
/// and imag components from Euler's formula (i.e. e^{ix} = cos{x} + i sin{x}).
///
/// # Examples
///
/// `assert_abs_diff_eq!(cexp(PI), c64::new(-1.0, 0.0));`
pub(crate) fn cexp(x: f64) -> c64 {
    let (im, re) = x.sin_cos();
    c64::new(re, im)
}

/// Convert a _cross-correlation_ baseline index into its constituent tile
/// indices. Baseline 0 _is not_ between tile 0 and tile 0; it is between tile 0
/// and tile 1.
// Courtesy Brian Crosse.
pub(crate) fn cross_correlation_baseline_to_tiles(
    total_num_tiles: usize,
    baseline: usize,
) -> (usize, usize) {
    // This works if we include auto-correlation "baselines".
    // let n = total_num_tiles as f64;
    // let bl = baseline as f64;
    // let tile1 = (-0.5 * (4.0 * n * (n + 1.0) - 8.0 * bl + 1.0).sqrt() + n + 0.5).floor();
    // let tile2 = bl - tile1 * (n - (tile1 + 1.0) / 2.0);

    let n = (total_num_tiles - 1) as f64;
    let bl = baseline as f64;
    let tile1 = (-0.5 * (4.0 * n * (n + 1.0) - 8.0 * bl + 1.0).sqrt() + n + 0.5).floor();
    let tile2 = bl - tile1 * (n - (tile1 + 1.0) / 2.0) + 1.0;
    (tile1 as usize, tile2 as usize)
}

/// From the number of baselines, get the number of tiles.
// From the definition of how many baselines there are in an array of N tiles,
// this is just the solved quadratic.
pub(crate) fn num_tiles_from_num_baselines(num_baselines: usize) -> usize {
    (1 + num::integer::Roots::sqrt(&(1 + 8 * num_baselines))) / 2
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use std::f64::consts::*;

    #[test]
    fn test_sin() {
        assert_abs_diff_eq!(sin(FRAC_PI_6), 0.5);
    }

    #[test]
    fn test_cos() {
        assert_abs_diff_eq!(cos(FRAC_PI_3), 0.5);
    }

    #[test]
    fn atan2_is_correct() {
        assert_abs_diff_eq!(atan2(-2.0, 1.0), -1.1071487177940904);
        assert_abs_diff_eq!(atan2(1.0, -1.0), 3.0 * FRAC_PI_4);
    }

    #[test]
    fn test_cexp() {
        assert_abs_diff_eq!(cexp(PI), c64::new(-1.0, 0.0));
    }

    #[test]
    fn test_cross_correlation_baseline_to_tiles() {
        // Let's pretend we have 128 tiles, therefore 8128 baselines. Check that
        // our function does the right thing.
        let n = 128;
        let mut bl_index = 0;
        for tile1 in 0..n {
            for tile2 in tile1 + 1..n {
                let (t1, t2) = cross_correlation_baseline_to_tiles(n, bl_index);
                assert_eq!(
                    tile1, t1,
                    "Expected tile1 = {}, got {}. bl = {}",
                    tile1, t1, bl_index
                );
                assert_eq!(
                    tile2, t2,
                    "Expected tile2 = {}, got {}. bl = {}",
                    tile2, t2, bl_index
                );
                bl_index += 1;
            }
        }

        // Try with a different number of tiles.
        let n = 126;
        let mut bl_index = 0;
        for tile1 in 0..n {
            for tile2 in tile1 + 1..n {
                let (t1, t2) = cross_correlation_baseline_to_tiles(n, bl_index);
                assert_eq!(
                    tile1, t1,
                    "Expected tile1 = {}, got {}. bl = {}",
                    tile1, t1, bl_index
                );
                assert_eq!(
                    tile2, t2,
                    "Expected tile2 = {}, got {}. bl = {}",
                    tile2, t2, bl_index
                );
                bl_index += 1;
            }
        }

        let n = 256;
        let mut bl_index = 0;
        for tile1 in 0..n {
            for tile2 in tile1 + 1..n {
                let (t1, t2) = cross_correlation_baseline_to_tiles(n, bl_index);
                assert_eq!(
                    tile1, t1,
                    "Expected tile1 = {}, got {}. bl = {}",
                    tile1, t1, bl_index
                );
                assert_eq!(
                    tile2, t2,
                    "Expected tile2 = {}, got {}. bl = {}",
                    tile2, t2, bl_index
                );
                bl_index += 1;
            }
        }
    }

    #[test]
    fn test_num_tiles_from_num_baselines() {
        assert_eq!(num_tiles_from_num_baselines(8128), 128);
        assert_eq!(num_tiles_from_num_baselines(8001), 127);
        assert_eq!(num_tiles_from_num_baselines(15), 6);
    }
}
