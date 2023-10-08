// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

use approx::assert_abs_diff_eq;
use hifitime::{Epoch, TimeUnits};

use super::*;

#[test]
fn test_average_epoch() {
    let epochs = [
        Epoch::from_gpst_seconds(1065880128.0),
        Epoch::from_gpst_seconds(1065880130.0),
        Epoch::from_gpst_seconds(1065880132.0),
    ];

    let average = average_epoch(epochs);
    assert_abs_diff_eq!(average.to_gpst_seconds(), 1065880130.0);
}

#[test]
fn test_average_epoch2() {
    let epochs = [
        Epoch::from_gpst_seconds(1065880128.0),
        Epoch::from_gpst_seconds(1090008640.0),
        Epoch::from_gpst_seconds(1118529192.0),
    ];

    let average = average_epoch(epochs);
    // This epsilon is huge, but the epochs span years. At least the first test
    // is accurate to precision.
    assert_abs_diff_eq!(average.to_gpst_seconds(), 1091472653.0, epsilon = 0.4);
}

#[test]
#[should_panic]
fn test_tile_baseline_flags_panics() {
    let maps = TileBaselineFlags::new(0, HashSet::new());
    assert!(!maps.flagged_tiles.is_empty(), "Shouldn't get here!");
}

#[test]
fn test_tile_baseline_flags_without_flags() {
    let total_num_tiles = 128;
    let tile_flags = HashSet::new();
    let maps = TileBaselineFlags::new(total_num_tiles, tile_flags);
    assert_eq!(maps.tile_to_unflagged_cross_baseline_map[&(0, 1)], 0);
    assert_eq!(maps.tile_to_unflagged_cross_baseline_map[&(0, 127)], 126);
    assert_eq!(maps.tile_to_unflagged_cross_baseline_map[&(126, 127)], 8127);
    assert!(maps
        .tile_to_unflagged_cross_baseline_map
        .get(&(0, 0))
        .is_none());
    assert_eq!(maps.unflagged_cross_baseline_to_tile_map[&0], (0, 1));
    assert_eq!(maps.unflagged_cross_baseline_to_tile_map[&8127], (126, 127));
    assert!(maps
        .unflagged_cross_baseline_to_tile_map
        .get(&8128)
        .is_none());

    assert_eq!(maps.tile_to_unflagged_auto_index_map[&0], 0);
    assert_eq!(maps.tile_to_unflagged_auto_index_map[&127], 127);
    assert!(maps.tile_to_unflagged_auto_index_map.get(&128).is_none());
    assert_eq!(maps.unflagged_auto_index_to_tile_map[&0], 0);
    assert_eq!(maps.unflagged_auto_index_to_tile_map[&127], 127);
    assert!(maps.unflagged_auto_index_to_tile_map.get(&128).is_none());

    assert!(maps.flagged_tiles.is_empty());

    // First index is always smaller or equal to second.
    for (i1, i2) in maps.tile_to_unflagged_cross_baseline_map.keys().copied() {
        assert!(i1 <= i2);
    }
    for (i1, i2) in maps.unflagged_cross_baseline_to_tile_map.values().copied() {
        assert!(i1 <= i2);
    }
}

#[test]
fn test_tile_baseline_flags() {
    let total_num_tiles = 128;
    let tile_flags = HashSet::from([1]);
    let maps = TileBaselineFlags::new(total_num_tiles, tile_flags);
    assert_eq!(maps.tile_to_unflagged_auto_index_map[&0], 0);
    assert_eq!(maps.tile_to_unflagged_auto_index_map[&2], 1);
    assert!(maps.tile_to_unflagged_auto_index_map.get(&1).is_none());
    assert!(maps.tile_to_unflagged_auto_index_map.get(&128).is_none());

    assert_eq!(maps.tile_to_unflagged_cross_baseline_map[&(0, 2)], 0);
    assert_eq!(maps.tile_to_unflagged_cross_baseline_map[&(0, 127)], 125);
    assert_eq!(maps.tile_to_unflagged_cross_baseline_map[&(2, 3)], 126);
    assert_eq!(maps.tile_to_unflagged_cross_baseline_map[&(126, 127)], 8000);
    assert!(maps
        .tile_to_unflagged_cross_baseline_map
        .get(&(0, 1))
        .is_none());
    assert_eq!(maps.unflagged_cross_baseline_to_tile_map[&0], (0, 2));
    assert_eq!(maps.unflagged_cross_baseline_to_tile_map[&126], (2, 3));
    assert_eq!(maps.unflagged_cross_baseline_to_tile_map[&8000], (126, 127));
    assert!(maps
        .unflagged_cross_baseline_to_tile_map
        .get(&8001)
        .is_none());

    assert_eq!(maps.unflagged_auto_index_to_tile_map[&0], 0);
    assert_eq!(maps.unflagged_auto_index_to_tile_map[&1], 2);
    assert_eq!(maps.unflagged_auto_index_to_tile_map[&126], 127);

    assert!(maps.flagged_tiles.contains(&1));

    // First index is always smaller or equal to second.
    for (i1, i2) in maps.tile_to_unflagged_cross_baseline_map.keys().copied() {
        assert!(i1 <= i2);
    }
    for (i1, i2) in maps.unflagged_cross_baseline_to_tile_map.values().copied() {
        assert!(i1 <= i2);
    }
}

#[test]
fn test_baseline_tile_pairs() {
    let total_num_tiles = 4;
    let tile_flags = HashSet::new();
    let maps = TileBaselineFlags::new(total_num_tiles, tile_flags);
    assert_eq!(
        maps.get_unflagged_baseline_tile_pairs()
            .collect::<Vec<_>>()
            .as_slice(),
        [
            (0, 0),
            (0, 1),
            (0, 2),
            (0, 3),
            (1, 1),
            (1, 2),
            (1, 3),
            (2, 2),
            (2, 3),
            (3, 3)
        ]
    );
    assert_eq!(
        maps.get_unflagged_cross_baseline_tile_pairs()
            .collect::<Vec<_>>()
            .as_slice(),
        [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    );

    let tile_flags = HashSet::from([1]);
    let maps = TileBaselineFlags::new(total_num_tiles, tile_flags);
    assert_eq!(
        maps.get_unflagged_baseline_tile_pairs()
            .collect::<Vec<_>>()
            .as_slice(),
        [(0, 0), (0, 2), (0, 3), (2, 2), (2, 3), (3, 3)]
    );
    assert_eq!(
        maps.get_unflagged_cross_baseline_tile_pairs()
            .collect::<Vec<_>>()
            .as_slice(),
        [(0, 2), (0, 3), (2, 3)]
    );
}

#[test]
fn test_is_prime() {
    assert!(!is_prime(0));
    assert!(!is_prime(1));
    assert!(is_prime(2));
    assert!(is_prime(3));
    assert!(!is_prime(4));
    assert!(is_prime(5));
    assert!(!is_prime(6));
    assert!(is_prime(7));
}

#[test]
fn test_hifitime_works_as_expected() {
    let e = Epoch::from_gpst_seconds(1090008639.999405);
    assert_abs_diff_eq!(e.round(10.milliseconds()).to_gpst_seconds(), 1090008640.0);

    let e = Epoch::from_gpst_seconds(1090008640.251);
    assert_abs_diff_eq!(e.round(10.milliseconds()).to_gpst_seconds(), 1090008640.25);

    let e = Epoch::from_gpst_seconds(1090008640.24999);
    assert_abs_diff_eq!(e.round(10.milliseconds()).to_gpst_seconds(), 1090008640.25);

    // No rounding.
    let e = Epoch::from_gpst_seconds(1090008640.26);
    assert_abs_diff_eq!(e.round(10.milliseconds()).to_gpst_seconds(), 1090008640.26);
}
