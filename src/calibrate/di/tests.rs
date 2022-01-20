// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

use approx::assert_abs_diff_eq;
use marlu::Jones;
use ndarray::prelude::*;

use super::*;
use crate::jones_test::TestJones;

#[test]
/// Make the data be twice the model.
fn test_calibrate_trivial() {
    let num_timeblocks = 1;
    let num_tiles = 5;
    let num_chanblocks = 1;

    let vis_shape = (
        num_timeblocks,
        num_tiles * (num_tiles - 1) / 2,
        num_chanblocks,
    );
    let vis_data: Array3<Jones<f32>> = Array3::from_elem(vis_shape, Jones::identity() * 2.0);
    let vis_model: Array3<Jones<f32>> = Array3::from_elem(vis_shape, Jones::identity());
    let mut di_jones = Array3::from_elem(
        (num_timeblocks, num_tiles, num_chanblocks),
        Jones::identity(),
    );

    let timeblock_length = 1;
    for timeblock in 0..num_timeblocks {
        let time_range_start = timeblock * timeblock_length;
        let time_range_end = ((timeblock + 1) * timeblock_length).min(vis_data.dim().0);

        let mut di_jones_rev = di_jones.slice_mut(s![timeblock, .., ..]).reversed_axes();

        for (chanblock_index, mut di_jones_slice) in (0..num_chanblocks)
            .into_iter()
            .zip(di_jones_rev.outer_iter_mut())
        {
            let range = s![
                time_range_start..time_range_end,
                ..,
                chanblock_index..chanblock_index + 1
            ];
            let vis_data_slice = vis_data.slice(range);
            let vis_model_slice = vis_model.slice(range);
            let result = calibrate(
                vis_data_slice,
                vis_model_slice,
                di_jones_slice.view_mut(),
                &vec![1.0; vis_data.dim().1],
                20,
                1e-8,
                1e-5,
            );

            // The model is all identity and the actual data is 2 * identity.
            // Calibration actually gets the right answer on the first
            // iteration, but due to our converence checking on every second
            // iteration, the solutions change

            // solutions should also be 2 * identity.
            // assert_eq!(result.num_iterations, 8);
            assert_eq!(result.num_iterations, 2);
            assert_eq!(result.num_failed, 0);

            let expected = Array1::from_elem(di_jones_slice.len(), Jones::identity() * 2.0);

            let di_jones_slice = di_jones_slice.mapv(TestJones::from);
            let expected = expected.mapv(TestJones::from);
            assert_abs_diff_eq!(di_jones_slice, expected, epsilon = 1e-10);
        }
    }
}
