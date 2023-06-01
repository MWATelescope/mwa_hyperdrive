// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Some helper mathematics.

#[cfg(test)]
mod tests;

use std::collections::{HashMap, HashSet};

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
        acc + t.to_gpst_seconds()
    });
    let average = duration_sum.to_gpst_seconds() / es.len() as f64;
    round_hundredths_of_a_second(Epoch::from_gpst_seconds(average))
}

/// Information on flagged tiles, baselines and maps to and from array indices.
pub struct TileBaselineFlags {
    /// Map between a pair of tile numbers and its unflagged *cross-correlation*
    /// baseline index. e.g. If tiles 0 and 2 are flagged, (1, 3) maps to 0
    /// (i.e. the first unflagged cross-correlation baseline is between tiles 1
    /// and 3). (1, 1) maps to 0 (i.e. the first unflagged auto-correlation
    /// corresponds to tile 1). The first tile index is always smaller or equal
    /// to the second.
    pub(crate) tile_to_unflagged_cross_baseline_map: HashMap<(usize, usize), usize>,

    /// Map between a tile index and its unflagged auto-correlation index. e.g.
    /// If tiles 0 and 2 are flagged, 1 maps to 0 (i.e. the first unflagged
    /// auto-correlation corresponds to tile 1) and 3 maps to 1.
    pub(crate) tile_to_unflagged_auto_index_map: HashMap<usize, usize>,

    /// Map an unflagged *cross-correlation* baseline index to its constituent
    /// tile indices. e.g. If tile 0 is flagged, baseline 0 maps to tiles 1 and
    /// 2 (i.e. the first unflagged cross-correlation baseline is between tiles
    /// 1 and 2). The first tile index is always smaller or equal to the second.
    pub(crate) unflagged_cross_baseline_to_tile_map: HashMap<usize, (usize, usize)>,

    /// Map an unflagged *auto-correlation* index to its tile index. e.g. If
    /// tile 0 is flagged, auto-correlation index 0 maps to tile 1 (i.e. the
    /// first unflagged auto-correlation is from tile 1).
    pub(crate) unflagged_auto_index_to_tile_map: HashMap<usize, usize>,

    /// Indices of flagged tiles. This is supplied by the user when this
    /// [`TileIndexMaps`] is created.
    pub(crate) flagged_tiles: HashSet<usize>,
}

impl TileBaselineFlags {
    /// Create a new set of maps and sets containing flags and indices. Will
    /// panic if `total_num_tiles` is 0.
    pub fn new(total_num_tiles: usize, mut flagged_tiles: HashSet<usize>) -> TileBaselineFlags {
        flagged_tiles.shrink_to_fit();

        let num_unflagged_tiles = total_num_tiles - flagged_tiles.len();
        let num_unflagged_cross_baselines = (num_unflagged_tiles * (num_unflagged_tiles - 1)) / 2;

        let mut tile_to_unflagged_cross_baseline_map =
            HashMap::with_capacity(num_unflagged_cross_baselines);
        let mut tile_to_unflagged_auto_index_map = HashMap::with_capacity(num_unflagged_tiles);
        let mut unflagged_cross_baseline_to_tile_map =
            HashMap::with_capacity(num_unflagged_cross_baselines);
        let mut unflagged_auto_index_to_tile_map = HashMap::with_capacity(num_unflagged_tiles);

        let mut unflagged_cross_bl = 0;
        let mut unflagged_auto = 0;
        for tile1 in 0..total_num_tiles {
            for tile2 in tile1..total_num_tiles {
                if flagged_tiles.contains(&tile1) || flagged_tiles.contains(&tile2) {
                } else if tile1 == tile2 {
                    tile_to_unflagged_auto_index_map.insert(tile1, unflagged_auto);
                    unflagged_auto_index_to_tile_map.insert(unflagged_auto, tile1);
                    unflagged_auto += 1;
                } else {
                    tile_to_unflagged_cross_baseline_map.insert((tile1, tile2), unflagged_cross_bl);
                    unflagged_cross_baseline_to_tile_map.insert(unflagged_cross_bl, (tile1, tile2));
                    unflagged_cross_bl += 1;
                }
            }
        }

        TileBaselineFlags {
            tile_to_unflagged_cross_baseline_map,
            tile_to_unflagged_auto_index_map,
            unflagged_cross_baseline_to_tile_map,
            unflagged_auto_index_to_tile_map,
            flagged_tiles,
        }
    }
}
