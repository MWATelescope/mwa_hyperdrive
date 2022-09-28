// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Some helper mathematics.

#[cfg(test)]
mod tests;

use std::collections::HashMap;

use hifitime::Epoch;

use crate::misc::round_hundredths_of_a_second;

/// Is the supplied number prime? This isn't necessarily efficient code; it's
/// used just for testing. Stolen from
/// https://stackoverflow.com/questions/55790537/calculating-prime-numbers-in-rust
#[cfg(test)]
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
    let duration_sum = es.iter().fold(Epoch::from_gpst_seconds(0.0), |acc, t| {
        acc + t.as_gpst_seconds()
    });
    let average = duration_sum.as_gpst_seconds() / es.len() as f64;
    round_hundredths_of_a_second(Epoch::from_gpst_seconds(average))
}

pub(crate) struct TileBaselineMaps {
    /// Map between a pair of tile numbers and its unflagged *cross-correlation*
    /// baseline index. This is really useful for handling flagged tiles and
    /// baselines, e.g. if tiles 0 and 2 are flagged, (1, 3) maps to 0 (i.e. the
    /// first cross-correlation baseline is between tiles 1 and 3).
    pub(crate) tile_to_unflagged_cross_baseline_map: HashMap<(usize, usize), usize>,

    /// Map an unflagged *cross-correlation* baseline index to its constituent
    /// tile indices. e.g. If tile 0 is flagged, baseline 0 maps to tiles 1 and
    /// 2 (i.e. the first cross-correlation baseline is between tiles 1 and 2).
    pub(crate) unflagged_cross_baseline_to_tile_map: HashMap<usize, (usize, usize)>,
}

impl TileBaselineMaps {
    pub(crate) fn new(total_num_tiles: usize, flagged_tiles: &[usize]) -> TileBaselineMaps {
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
