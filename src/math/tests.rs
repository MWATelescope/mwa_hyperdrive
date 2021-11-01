// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

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
    fn test_generate_tile_baseline_maps() {
        let total_num_tiles = 128;
        let mut tile_flags = HashSet::new();
        let maps = generate_tile_baseline_maps(total_num_tiles, &tile_flags);
        assert_eq!(maps.tile_to_unflagged_baseline_map[&(0, 1)], 0);
        assert_eq!(maps.unflagged_baseline_to_tile_map[&0], (0, 1));

        tile_flags.insert(1);
        let maps = generate_tile_baseline_maps(total_num_tiles, &tile_flags);
        assert_eq!(maps.tile_to_unflagged_baseline_map[&(0, 2)], 0);
        assert_eq!(maps.tile_to_unflagged_baseline_map[&(2, 3)], 126);
        assert_eq!(maps.unflagged_baseline_to_tile_map[&0], (0, 2));
        assert_eq!(maps.unflagged_baseline_to_tile_map[&126], (2, 3));
    }
}
