// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

use approx::assert_abs_diff_eq;

use super::*;

#[test]
fn test_average_epoch() {
    let epochs = [
        Epoch::from_gpst_seconds(1065880128.0),
        Epoch::from_gpst_seconds(1065880130.0),
        Epoch::from_gpst_seconds(1065880132.0),
    ];

    let average = average_epoch(&epochs);
    assert_abs_diff_eq!(average.as_gpst_seconds(), 1065880130.0);
}

#[test]
fn test_average_epoch2() {
    let epochs = [
        Epoch::from_gpst_seconds(1065880128.0),
        Epoch::from_gpst_seconds(1090008640.0),
        Epoch::from_gpst_seconds(1118529192.0),
    ];

    let average = average_epoch(&epochs);
    // This epsilon is huge, but the epochs span years. At least the first test
    // is accurate to precision.
    assert_abs_diff_eq!(average.as_gpst_seconds(), 1091472653.0, epsilon = 0.4);
}

#[test]
fn test_generate_tile_baseline_maps() {
    let total_num_tiles = 128;
    let mut tile_flags = vec![];
    let maps = TileBaselineMaps::new(total_num_tiles, &tile_flags);
    assert_eq!(maps.tile_to_unflagged_cross_baseline_map[&(0, 1)], 0);
    assert_eq!(maps.unflagged_cross_baseline_to_tile_map[&0], (0, 1));

    tile_flags.push(1);
    let maps = TileBaselineMaps::new(total_num_tiles, &tile_flags);
    assert_eq!(maps.tile_to_unflagged_cross_baseline_map[&(0, 2)], 0);
    assert_eq!(maps.tile_to_unflagged_cross_baseline_map[&(2, 3)], 126);
    assert_eq!(maps.unflagged_cross_baseline_to_tile_map[&0], (0, 2));
    assert_eq!(maps.unflagged_cross_baseline_to_tile_map[&126], (2, 3));
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
