// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Some helper mathematics.

#![allow(dead_code)]

#[cfg(test)]
mod tests;

use hifitime::{Duration, Epoch, TimeUnit};
use std::collections::HashMap;

use mwa_hyperdrive_common::{c64, hifitime};

// Make traditional trigonometry possible.
/// Sine.
///
/// # Examples
///
/// `assert_abs_diff_eq!(sin(FRAC_PI_6), 0.5);`
#[inline]
pub(crate) fn sin(x: f64) -> f64 {
    x.sin()
}

/// Cosine.
///
/// # Examples
///
/// `assert_abs_diff_eq!(cos(FRAC_PI_3), 0.5);`
#[inline]
pub(crate) fn cos(x: f64) -> f64 {
    x.cos()
}

/// Inverse tangent. y comes before x, like the C function.
///
/// # Examples
///
/// `assert_abs_diff_eq!(atan2(1, -1), 3.0 / 4.0 * PI);`
// I don't like Rust's atan2. This fn helps me sleep at night knowing I'm using
// it correctly.
#[inline]
pub(crate) fn atan2(y: f64, x: f64) -> f64 {
    y.atan2(x)
}

/// Exponential.
///
/// # Examples
///
/// `assert_abs_diff_eq!(exp(1), 2.718281828);`
#[inline]
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
#[inline]
pub(crate) fn cexp(x: f64) -> c64 {
    let (im, re) = x.sin_cos();
    c64::new(re, im)
}

/// Is the supplied number prime? This isn't necessarily efficient code; it's
/// used just for testing. Stolen from
/// https://stackoverflow.com/questions/55790537/calculating-prime-numbers-in-rust
pub(crate) fn is_prime(n: usize) -> bool {
    let limit = (n as f64).sqrt() as usize;
    if n < 2 {
        return false;
    }

    for i in 2..=limit {
        if n % i == 0 {
            return false;
        }
    }

    true
}

/// Given a collection of [Epoch]s, return one that is the average of their
/// times.
pub(crate) fn average_epoch(es: &[Epoch]) -> Epoch {
    let duration_sum = es
        .iter()
        .fold(Duration::from_f64(0.0, TimeUnit::Second), |acc, t| {
            acc + t.as_et_duration()
        });
    Epoch::from_et_seconds(duration_sum.in_seconds() / es.len() as f64)
}

pub struct TileBaselineMaps {
    pub tile_to_unflagged_cross_baseline_map: HashMap<(usize, usize), usize>,
    pub unflagged_cross_baseline_to_tile_map: HashMap<usize, (usize, usize)>,
}

impl TileBaselineMaps {
    pub fn new(total_num_tiles: usize, flagged_tiles: &[usize]) -> TileBaselineMaps {
        let mut tile_to_unflagged_cross_baseline_map = HashMap::new();
        let mut unflagged_cross_baseline_to_tile_map = HashMap::new();
        let mut bl = 0;
        for tile1 in 0..total_num_tiles {
            if flagged_tiles.contains(&tile1) {
                continue;
            }
            for tile2 in tile1 + 1..total_num_tiles {
                if flagged_tiles.contains(&tile2) {
                    continue;
                }
                tile_to_unflagged_cross_baseline_map.insert((tile1, tile2), bl);
                unflagged_cross_baseline_to_tile_map.insert(bl, (tile1, tile2));
                bl += 1;
            }
        }

        TileBaselineMaps {
            tile_to_unflagged_cross_baseline_map,
            unflagged_cross_baseline_to_tile_map,
        }
    }
}
