// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

use std::collections::HashSet;

use approx::assert_abs_diff_eq;
use vec1::vec1;

use super::*;
use crate::math::average_epoch;

#[test]
fn test_timesteps_to_timeblocks() {
    let all_timestamps: Vec<Epoch> = (0..20)
        .map(|i| Epoch::from_gpst_seconds(1065880128.0 + (2 * i) as f64))
        .collect();
    let all_timestamps = Vec1::try_from_vec(all_timestamps).unwrap();

    let time_res = Duration::from_seconds(2.0);
    let time_average_factor = NonZeroUsize::new(1).unwrap();
    let timesteps_to_use = vec1![2, 3, 4, 5];
    let timeblocks = timesteps_to_timeblocks(
        &all_timestamps,
        time_res,
        time_average_factor,
        Some(&timesteps_to_use),
    );
    // Time average factor is 1; 1 timestep per timeblock.
    assert_eq!(timeblocks.len(), 4);
    for ((timeblock, expected_indices), expected_timestamp) in timeblocks
        .into_iter()
        .zip([[0], [1], [2], [3]])
        .zip([1065880132.0, 1065880134.0, 1065880136.0, 1065880138.0])
    {
        assert_eq!(timeblock.range.len(), expected_indices.len());
        for (timestep, expected_index) in timeblock.range.zip(expected_indices) {
            assert_eq!(timestep, expected_index);
        }

        assert_eq!(
            average_epoch(timeblock.timestamps).to_gpst_seconds(),
            expected_timestamp
        );
        assert_eq!(timeblock.median.to_gpst_seconds(), expected_timestamp);
    }

    let time_average_factor = NonZeroUsize::new(2).unwrap();
    let timesteps_to_use = vec1![2, 3, 4, 5];
    let timeblocks = timesteps_to_timeblocks(
        &all_timestamps,
        time_res,
        time_average_factor,
        Some(&timesteps_to_use),
    );
    // 2 timesteps per timeblock.
    assert_eq!(timeblocks.len(), 2);
    for ((timeblock, expected_indices), expected_timestamp) in timeblocks
        .into_iter()
        .zip([[0, 1], [2, 3]])
        .zip([1065880133.0, 1065880137.0])
    {
        assert_eq!(timeblock.range.len(), expected_indices.len());
        for (timestep, expected_index) in timeblock.range.zip(expected_indices) {
            assert_eq!(timestep, expected_index);
        }

        assert_eq!(
            average_epoch(timeblock.timestamps).to_gpst_seconds(),
            expected_timestamp
        );
        assert_eq!(timeblock.median.to_gpst_seconds(), expected_timestamp);
    }

    let time_average_factor = NonZeroUsize::new(3).unwrap();
    let timesteps_to_use = vec1![2, 3, 4, 5];
    let timeblocks = timesteps_to_timeblocks(
        &all_timestamps,
        time_res,
        time_average_factor,
        Some(&timesteps_to_use),
    );
    // 3 timesteps per timeblock, but the last timeblock has only one timestep.
    assert_eq!(timeblocks.len(), 2);
    for ((timeblock, expected_indices), expected_timestamp) in timeblocks
        .into_iter()
        .zip([vec![0, 1, 2], vec![3]])
        .zip([1065880134.0, 1065880138.0])
    {
        assert_eq!(timeblock.range.len(), expected_indices.len());
        for (timestep, expected_index) in timeblock.range.clone().zip(expected_indices) {
            assert_eq!(timestep, expected_index);
        }

        assert_eq!(
            average_epoch(timeblock.timestamps).to_gpst_seconds(),
            expected_timestamp
        );
        // The median is different from the average for the second timeblock.
        if timeblock.range.len() == 3 {
            assert_eq!(timeblock.median.to_gpst_seconds(), expected_timestamp);
        } else {
            assert_eq!(timeblock.median.to_gpst_seconds(), expected_timestamp + 2.0);
        }
    }

    let timesteps_to_use = vec1![2, 15, 16];
    // Average all the timesteps together. This is what is used to calculate the
    // time average factor in this case.
    let time_average_factor =
        NonZeroUsize::new(*timesteps_to_use.last() - *timesteps_to_use.first() + 1).unwrap();
    assert_eq!(time_average_factor.get(), 15);
    let timeblocks = timesteps_to_timeblocks(
        &all_timestamps,
        time_res,
        time_average_factor,
        Some(&timesteps_to_use),
    );
    assert_eq!(timeblocks.len(), 1);
    for ((timeblock, expected_indices), expected_timestamp) in
        timeblocks.into_iter().zip([[0, 1, 2]]).zip([1065880150.0])
    {
        assert_eq!(timeblock.range.len(), expected_indices.len());
        for (timestep, expected_index) in timeblock.range.zip(expected_indices) {
            assert_eq!(timestep, expected_index);
        }

        assert_eq!(
            average_epoch(timeblock.timestamps).to_gpst_seconds(),
            expected_timestamp
        );
        // (2 + 16) / 2 = 9 is the median timestep
        // 1065880128.0 + (2 * 9) = 1065880146.0
        assert_eq!(timeblock.median.to_gpst_seconds(), 1065880146.0);
    }
}

#[test]
fn test_channels_to_chanblocks() {
    let all_channel_freqs = [12000];
    let freq_average_factor = NonZeroUsize::new(1).unwrap();
    let mut flagged_channels = HashSet::new();
    let freq_res = 1000;
    let spws = channels_to_chanblocks(
        &all_channel_freqs,
        freq_res,
        freq_average_factor,
        &flagged_channels,
    );
    assert_eq!(spws.len(), 1);
    assert_eq!(spws[0].chanblocks.len(), 1);
    assert!(spws[0].flagged_chanblock_indices.is_empty());
    assert_abs_diff_eq!(spws[0].chanblocks[0].freq, 12000.0);
    assert_abs_diff_eq!(spws[0].freq_res, freq_res as f64);
    assert_abs_diff_eq!(spws[0].first_freq, 12000.0);

    let all_channel_freqs = [10000, 11000, 12000, 13000, 14000];
    let spws = channels_to_chanblocks(
        &all_channel_freqs,
        freq_res,
        freq_average_factor,
        &flagged_channels,
    );
    assert_eq!(spws.len(), 1);
    assert_eq!(spws[0].chanblocks.len(), 5);
    assert!(spws[0].flagged_chanblock_indices.is_empty());
    assert_abs_diff_eq!(spws[0].chanblocks[0].freq, 10000.0);
    assert_abs_diff_eq!(spws[0].chanblocks[1].freq, 11000.0);
    assert_abs_diff_eq!(spws[0].chanblocks[2].freq, 12000.0);
    assert_abs_diff_eq!(spws[0].chanblocks[3].freq, 13000.0);
    assert_abs_diff_eq!(spws[0].chanblocks[4].freq, 14000.0);
    assert_abs_diff_eq!(spws[0].freq_res, 1000.0);
    assert_abs_diff_eq!(spws[0].first_freq, 10000.0);

    let all_channel_freqs = [10000, 11000, 12000, 13000, 14000, 20000];
    let spws = channels_to_chanblocks(
        &all_channel_freqs,
        freq_res,
        freq_average_factor,
        &flagged_channels,
    );
    assert_eq!(spws.len(), 2);
    assert_eq!(spws[0].chanblocks.len(), 5);
    assert_eq!(spws[1].chanblocks.len(), 1);
    assert!(spws[0].flagged_chanblock_indices.is_empty());
    assert!(spws[1].flagged_chanblock_indices.is_empty());
    assert_abs_diff_eq!(spws[0].chanblocks[0].freq, 10000.0);
    assert_abs_diff_eq!(spws[0].chanblocks[1].freq, 11000.0);
    assert_abs_diff_eq!(spws[0].chanblocks[2].freq, 12000.0);
    assert_abs_diff_eq!(spws[0].chanblocks[3].freq, 13000.0);
    assert_abs_diff_eq!(spws[0].chanblocks[4].freq, 14000.0);
    assert_abs_diff_eq!(spws[1].chanblocks[0].freq, 20000.0);
    assert_abs_diff_eq!(spws[0].freq_res, 1000.0);
    assert_abs_diff_eq!(spws[1].freq_res, 1000.0);
    assert_abs_diff_eq!(spws[0].first_freq, 10000.0);
    assert_abs_diff_eq!(spws[1].first_freq, 20000.0);

    flagged_channels.insert(3);
    let spws = channels_to_chanblocks(
        &all_channel_freqs,
        freq_res,
        freq_average_factor,
        &flagged_channels,
    );
    assert_eq!(spws.len(), 2);
    assert_eq!(spws[0].chanblocks.len(), 4);
    assert_eq!(spws[1].chanblocks.len(), 1);
    assert_eq!(spws[0].flagged_chanblock_indices.len(), 1);
    let mut sorted = spws[0]
        .flagged_chanblock_indices
        .iter()
        .copied()
        .collect::<Vec<_>>();
    sorted.sort_unstable();
    assert_eq!(sorted[0], 3);
    assert!(spws[1].flagged_chanblock_indices.is_empty());
    assert_abs_diff_eq!(spws[0].chanblocks[0].freq, 10000.0);
    assert_abs_diff_eq!(spws[0].chanblocks[1].freq, 11000.0);
    assert_abs_diff_eq!(spws[0].chanblocks[2].freq, 12000.0);
    assert_abs_diff_eq!(spws[0].chanblocks[3].freq, 14000.0);
    assert_abs_diff_eq!(spws[1].chanblocks[0].freq, 20000.0);
    assert_abs_diff_eq!(spws[0].freq_res, 1000.0);
    assert_abs_diff_eq!(spws[1].freq_res, 1000.0);
    assert_abs_diff_eq!(spws[0].first_freq, 10000.0);
    assert_abs_diff_eq!(spws[1].first_freq, 20000.0);

    let freq_average_factor = NonZeroUsize::new(2).unwrap();
    let spws = channels_to_chanblocks(
        &all_channel_freqs,
        freq_res,
        freq_average_factor,
        &flagged_channels,
    );
    assert_eq!(spws.len(), 2);
    assert_eq!(spws[0].chanblocks.len(), 3);
    assert_eq!(spws[1].chanblocks.len(), 1);
    assert!(spws[0].flagged_chanblock_indices.is_empty());
    assert!(spws[1].flagged_chanblock_indices.is_empty());
    assert_abs_diff_eq!(spws[0].chanblocks[0].freq, 10500.0);
    assert_abs_diff_eq!(spws[0].chanblocks[1].freq, 12500.0);
    assert_abs_diff_eq!(spws[0].chanblocks[2].freq, 14500.0);
    assert_abs_diff_eq!(spws[1].chanblocks[0].freq, 20500.0);
    assert_abs_diff_eq!(spws[0].freq_res, 2000.0);
    assert_abs_diff_eq!(spws[1].freq_res, 2000.0);
    assert_abs_diff_eq!(spws[0].first_freq, 10500.0);
    assert_abs_diff_eq!(spws[1].first_freq, 20500.0);

    let freq_average_factor = NonZeroUsize::new(3).unwrap();
    let spws = channels_to_chanblocks(
        &all_channel_freqs,
        freq_res,
        freq_average_factor,
        &flagged_channels,
    );
    assert_eq!(spws.len(), 2);
    assert_eq!(spws[0].chanblocks.len(), 2);
    assert_eq!(spws[1].chanblocks.len(), 1);
    assert!(spws[0].flagged_chanblock_indices.is_empty());
    assert!(spws[1].flagged_chanblock_indices.is_empty());
    assert_abs_diff_eq!(spws[0].chanblocks[0].freq, 11000.0);
    assert_abs_diff_eq!(spws[0].chanblocks[1].freq, 14000.0);
    assert_abs_diff_eq!(spws[1].chanblocks[0].freq, 21000.0);
    assert_abs_diff_eq!(spws[0].freq_res, 3000.0);
    assert_abs_diff_eq!(spws[1].freq_res, 3000.0);
    assert_abs_diff_eq!(spws[0].first_freq, 11000.0);
    assert_abs_diff_eq!(spws[1].first_freq, 21000.0);
}

// No frequencies, no spws.
#[test]
fn test_no_channels_to_chanblocks() {
    let all_channel_freqs = [];
    let freq_average_factor = NonZeroUsize::new(2).unwrap();
    let flagged_channels = HashSet::new();
    let spws = channels_to_chanblocks(
        &all_channel_freqs,
        10e3 as u64,
        freq_average_factor,
        &flagged_channels,
    );
    assert!(spws.is_empty());
}

fn test_time(
    time_resolution: Option<Duration>,
    user_input_time_factor: Option<&str>,
    default: NonZeroUsize,
    expected: Option<usize>,
) {
    let result = parse_time_average_factor(time_resolution, user_input_time_factor, default);

    match expected {
        // If expected is Some, then we expect the result to match.
        Some(expected) => {
            assert_eq!(result.unwrap().get(), expected, "res={time_resolution:?}, input={user_input_time_factor:?}, default={default}, expected={expected:?}");
        }
        // Otherwise, we expect failure.
        None => {
            assert!(result.is_err(), "res={time_resolution:?}, input={user_input_time_factor:?}, default={default}, expected={expected:?}, error={}", result.unwrap());
        }
    }
}

#[test]
fn test_parse_time_average_factor() {
    let time_resolution = Some(Duration::from_seconds(2.0));
    let default = NonZeroUsize::new(100).unwrap();

    let user_input_time_factor = Some("2");
    let expected = Some(2);
    test_time(time_resolution, user_input_time_factor, default, expected);

    let user_input_time_factor = Some("2s");
    let expected = Some(1);
    test_time(time_resolution, user_input_time_factor, default, expected);

    let user_input_time_factor = Some("4");
    let expected = Some(4);
    test_time(time_resolution, user_input_time_factor, default, expected);

    let user_input_time_factor = Some("4s");
    let expected = Some(2);
    test_time(time_resolution, user_input_time_factor, default, expected);

    let user_input_time_factor = Some("4.00000s");
    let expected = Some(2);
    test_time(time_resolution, user_input_time_factor, default, expected);

    let user_input_time_factor = Some(" 4.00000 s");
    let expected = Some(2);
    test_time(time_resolution, user_input_time_factor, default, expected);

    let user_input_time_factor = Some(" 4000.00 ms");
    let expected = Some(2);
    test_time(time_resolution, user_input_time_factor, default, expected);

    let user_input_time_factor = Some("1 s");
    let expected = None;
    test_time(time_resolution, user_input_time_factor, default, expected);

    let user_input_time_factor = Some(" 1000 ms");
    let expected = None;
    test_time(time_resolution, user_input_time_factor, default, expected);

    let user_input_time_factor = None;
    let expected = Some(default.get());
    test_time(time_resolution, user_input_time_factor, default, expected);
}

fn test_freq(
    freq_resolution: Option<f64>,
    user_input_freq_factor: Option<&str>,
    default: NonZeroUsize,
    expected: Option<usize>,
) {
    let result = parse_freq_average_factor(freq_resolution, user_input_freq_factor, default);

    match expected {
        // If expected is Some, then we expect the result to match.
        Some(expected) => {
            assert_eq!(result.unwrap().get(), expected, "res={freq_resolution:?}, input={user_input_freq_factor:?}, default={default}, expected={expected:?}");
        }
        // Otherwise, we expect failure.
        None => {
            assert!(result.is_err(), "res={freq_resolution:?}, input={user_input_freq_factor:?}, default={default}, expected={expected:?}, error={}", result.unwrap());
        }
    }
}

#[test]
fn test_parse_freq_average_factor() {
    let freq_resolution = Some(40000.0); // Hz
    let default = NonZeroUsize::new(1).unwrap();

    let user_input_freq_factor = Some("2");
    let expected = Some(2);
    test_freq(freq_resolution, user_input_freq_factor, default, expected);

    let user_input_freq_factor = Some("20kHz");
    let expected = None;
    test_freq(freq_resolution, user_input_freq_factor, default, expected);

    let user_input_freq_factor = Some("20000Hz");
    let expected = None;
    test_freq(freq_resolution, user_input_freq_factor, default, expected);

    let user_input_freq_factor = Some("20000 Hz");
    let expected = None;
    test_freq(freq_resolution, user_input_freq_factor, default, expected);

    let user_input_freq_factor = Some("40000 Hz");
    let expected = Some(1);
    test_freq(freq_resolution, user_input_freq_factor, default, expected);

    let user_input_freq_factor = Some(" 40000 hz ");
    let expected = Some(1);
    test_freq(freq_resolution, user_input_freq_factor, default, expected);

    let user_input_freq_factor = Some("40 khz");
    let expected = Some(1);
    test_freq(freq_resolution, user_input_freq_factor, default, expected);

    let user_input_freq_factor = None;
    let expected = Some(default.get());
    test_freq(freq_resolution, user_input_freq_factor, default, expected);
}

#[test]
fn test_vis_average_1d_time() {
    // 1 timestep, 4 channels, 1 baseline.
    let jones_from_tfb = array![[
        [Jones::identity()],
        [Jones::identity() * 2.0],
        [Jones::identity() * 3.0],
        [Jones::identity() * 4.0]
    ]];
    let weight_from_tfb = Array3::ones(jones_from_tfb.dim());
    let mut jones_to_fb = Array2::default((
        jones_from_tfb.len_of(Axis(1)),
        jones_from_tfb.len_of(Axis(2)),
    ));
    let mut weight_to_fb = Array2::default(jones_to_fb.dim());

    vis_average(
        jones_from_tfb.view(),
        jones_to_fb.view_mut(),
        weight_from_tfb.view(),
        weight_to_fb.view_mut(),
        &HashSet::new(),
    );

    for (jones_from, jones_to) in jones_from_tfb.iter().zip_eq(jones_to_fb.iter()) {
        assert_abs_diff_eq!(jones_from, jones_to);
    }
    for (weight_from, weight_to) in weight_from_tfb.iter().zip_eq(weight_to_fb.iter()) {
        assert_abs_diff_eq!(weight_from, weight_to);
    }
}

#[test]
fn test_vis_average() {
    // 2 timesteps, 4 channels, 1 baseline.
    let jones_from_tfb = array![
        [
            [Jones::identity()],
            [Jones::identity() * 2.0],
            [Jones::identity() * 3.0],
            [Jones::identity() * 4.0]
        ],
        [
            [Jones::identity() * 5.0],
            [Jones::identity() * 6.0],
            [Jones::identity() * 7.0],
            [Jones::identity() * 8.0]
        ]
    ];
    let mut weight_from_tfb = Array3::ones(jones_from_tfb.dim());
    let mut jones_to_fb = Array2::default((
        jones_from_tfb.len_of(Axis(1)),
        jones_from_tfb.len_of(Axis(2)),
    ));
    let mut weight_to_fb = Array2::default(jones_to_fb.dim());
    let num_chans = jones_from_tfb.len_of(Axis(1));

    vis_average(
        jones_from_tfb.view(),
        jones_to_fb.view_mut(),
        weight_from_tfb.view(),
        weight_to_fb.view_mut(),
        &HashSet::new(),
    );
    for (i, (jones, weight)) in jones_to_fb
        .iter()
        .copied()
        .zip(weight_to_fb.iter().copied())
        .enumerate()
    {
        // i -> ((i + 1) + (num_chans + i + 1)) / 2 -> (2*i + num_chans + 2) / 2
        let ii = (2 * i + num_chans + 2) as f32 / 2.0;
        assert_abs_diff_eq!(jones, Jones::identity() * ii);
        assert_abs_diff_eq!(weight, 2.0);
    }

    // Make the first channel's weights negative.
    weight_from_tfb.slice_mut(s![.., 0, ..]).fill(-1.0);

    vis_average(
        jones_from_tfb.view(),
        jones_to_fb.view_mut(),
        weight_from_tfb.view(),
        weight_to_fb.view_mut(),
        &HashSet::new(),
    );

    for (i, (jones, weight)) in jones_to_fb
        .iter()
        .copied()
        .zip(weight_to_fb.iter().copied())
        .enumerate()
    {
        let ii = (2 * i + num_chans + 2) as f32 / 2.0;
        assert_abs_diff_eq!(jones, Jones::identity() * ii);
        // The first channel's weight accumulates only negatives.
        if i == 0 {
            assert_abs_diff_eq!(weight, -2.0);
        } else {
            assert_abs_diff_eq!(weight, 2.0);
        }
    }

    // Make all weights positive, except for the very first one.
    weight_from_tfb.fill(1.0);
    weight_from_tfb[(0, 0, 0)] = -1.0;

    vis_average(
        jones_from_tfb.view(),
        jones_to_fb.view_mut(),
        weight_from_tfb.view(),
        weight_to_fb.view_mut(),
        &HashSet::new(),
    );

    for (i, (jones, weight)) in jones_to_fb
        .iter()
        .copied()
        .zip(weight_to_fb.iter().copied())
        .enumerate()
    {
        let ii = (2 * i + num_chans + 2) as f32 / 2.0;
        if i == 0 {
            // The first channel uses only data corresponding to the positive
            // weight.
            assert_abs_diff_eq!(jones, Jones::identity() * 5.0);
            assert_abs_diff_eq!(weight, 1.0);
        } else {
            assert_abs_diff_eq!(jones, Jones::identity() * ii);
            assert_abs_diff_eq!(weight, 2.0);
        }
    }

    // Now let's average in time and frequency.
    weight_from_tfb.fill(1.0);
    let mut jones_to_fb = Array2::default((num_chans / 2, jones_from_tfb.len_of(Axis(2))));
    let mut weight_to_fb = Array2::default(jones_to_fb.dim());

    vis_average(
        jones_from_tfb.view(),
        jones_to_fb.view_mut(),
        weight_from_tfb.view(),
        weight_to_fb.view_mut(),
        &HashSet::new(),
    );

    assert_abs_diff_eq!(jones_to_fb[(0, 0)], Jones::identity() * 14.0 / 4.0);
    assert_abs_diff_eq!(weight_to_fb[(0, 0)], 4.0);

    assert_abs_diff_eq!(jones_to_fb[(1, 0)], Jones::identity() * 22.0 / 4.0);
    assert_abs_diff_eq!(weight_to_fb[(1, 0)], 4.0);
}

#[test]
fn test_vis_average_non_uniform_weights() {
    // 2 timesteps, 4 channels, 1 baseline.
    let jones_from_tfb = array![
        [
            [Jones::identity()],
            [Jones::identity() * 2.0],
            [Jones::identity() * 3.0],
            [Jones::identity() * 4.0]
        ],
        [
            [Jones::identity() * 5.0],
            [Jones::identity() * 6.0],
            [Jones::identity() * 7.0],
            [Jones::identity() * 8.0]
        ]
    ];
    let mut weight_from_tfb = array![
        [[2.0], [3.0], [5.0], [7.0]],
        [[11.0], [13.0], [17.0], [19.0]]
    ];
    let mut jones_to_fb = Array2::default((
        jones_from_tfb.len_of(Axis(1)),
        jones_from_tfb.len_of(Axis(2)),
    ));
    let mut weight_to_fb = Array2::default(jones_to_fb.dim());
    let num_chans = jones_from_tfb.len_of(Axis(1));

    vis_average(
        jones_from_tfb.view(),
        jones_to_fb.view_mut(),
        weight_from_tfb.view(),
        weight_to_fb.view_mut(),
        &HashSet::new(),
    );

    assert_abs_diff_eq!(jones_to_fb[(0, 0)], Jones::identity() * 57.0 / 13.0);
    assert_abs_diff_eq!(weight_to_fb[(0, 0)], 13.0);

    assert_abs_diff_eq!(jones_to_fb[(1, 0)], Jones::identity() * 84.0 / 16.0);
    assert_abs_diff_eq!(weight_to_fb[(1, 0)], 16.0);

    assert_abs_diff_eq!(jones_to_fb[(2, 0)], Jones::identity() * 134.0 / 22.0);
    assert_abs_diff_eq!(weight_to_fb[(2, 0)], 22.0);

    assert_abs_diff_eq!(jones_to_fb[(3, 0)], Jones::identity() * 180.0 / 26.0);
    assert_abs_diff_eq!(weight_to_fb[(3, 0)], 26.0);

    // Make the first channel's weights negative.
    weight_from_tfb[(0, 0, 0)] *= -1.0;
    weight_from_tfb[(1, 0, 0)] *= -1.0;

    vis_average(
        jones_from_tfb.view(),
        jones_to_fb.view_mut(),
        weight_from_tfb.view(),
        weight_to_fb.view_mut(),
        &HashSet::new(),
    );

    // The first channel's weight accumulates only negatives.
    assert_abs_diff_eq!(jones_to_fb[(0, 0)], Jones::identity() * 57.0 / 13.0);
    assert_abs_diff_eq!(weight_to_fb[(0, 0)], -13.0);

    assert_abs_diff_eq!(jones_to_fb[(1, 0)], Jones::identity() * 84.0 / 16.0);
    assert_abs_diff_eq!(weight_to_fb[(1, 0)], 16.0);

    assert_abs_diff_eq!(jones_to_fb[(2, 0)], Jones::identity() * 134.0 / 22.0);
    assert_abs_diff_eq!(weight_to_fb[(2, 0)], 22.0);

    assert_abs_diff_eq!(jones_to_fb[(3, 0)], Jones::identity() * 180.0 / 26.0);
    assert_abs_diff_eq!(weight_to_fb[(3, 0)], 26.0);

    // Make all weights positive, except for the very first one.
    weight_from_tfb[(1, 0, 0)] *= -1.0;

    vis_average(
        jones_from_tfb.view(),
        jones_to_fb.view_mut(),
        weight_from_tfb.view(),
        weight_to_fb.view_mut(),
        &HashSet::new(),
    );

    // The first channel uses only data corresponding to the positive weight.
    assert_abs_diff_eq!(jones_to_fb[(0, 0)], Jones::identity() * 5.0);
    assert_abs_diff_eq!(weight_to_fb[(0, 0)], 11.0);

    assert_abs_diff_eq!(jones_to_fb[(1, 0)], Jones::identity() * 84.0 / 16.0);
    assert_abs_diff_eq!(weight_to_fb[(1, 0)], 16.0);

    assert_abs_diff_eq!(jones_to_fb[(2, 0)], Jones::identity() * 134.0 / 22.0);
    assert_abs_diff_eq!(weight_to_fb[(2, 0)], 22.0);

    assert_abs_diff_eq!(jones_to_fb[(3, 0)], Jones::identity() * 180.0 / 26.0);
    assert_abs_diff_eq!(weight_to_fb[(3, 0)], 26.0);

    // Now let's average in time and frequency.
    weight_from_tfb[(0, 0, 0)] *= -1.0;
    let mut jones_to_fb = Array2::default((num_chans / 2, jones_from_tfb.len_of(Axis(2))));
    let mut weight_to_fb = Array2::default(jones_to_fb.dim());

    vis_average(
        jones_from_tfb.view(),
        jones_to_fb.view_mut(),
        weight_from_tfb.view(),
        weight_to_fb.view_mut(),
        &HashSet::new(),
    );

    assert_abs_diff_eq!(jones_to_fb[(0, 0)], Jones::identity() * 141.0 / 29.0);
    assert_abs_diff_eq!(weight_to_fb[(0, 0)], 29.0);

    assert_abs_diff_eq!(jones_to_fb[(1, 0)], Jones::identity() * 314.0 / 48.0);
    assert_abs_diff_eq!(weight_to_fb[(1, 0)], 48.0);
}

#[test]
fn test_vis_average_non_uniform_weights_non_integral_array_shapes() {
    // 2 timesteps, 3 channels, 1 baseline.
    let jones_from_tfb = array![
        [
            [Jones::identity()],
            [Jones::identity() * 2.0],
            [Jones::identity() * 3.0]
        ],
        [
            [Jones::identity() * 4.0],
            [Jones::identity() * 5.0],
            [Jones::identity() * 6.0]
        ]
    ];
    let mut weight_from_tfb = array![[[2.0], [3.0], [5.0]], [[7.0], [11.0], [13.0]]];
    let mut jones_to_fb = Array2::default((2, jones_from_tfb.len_of(Axis(2))));
    let mut weight_to_fb = Array2::default(jones_to_fb.dim());

    vis_average(
        jones_from_tfb.view(),
        jones_to_fb.view_mut(),
        weight_from_tfb.view(),
        weight_to_fb.view_mut(),
        &HashSet::new(),
    );

    assert_abs_diff_eq!(jones_to_fb[(0, 0)], Jones::identity() * 91.0 / 23.0);
    assert_abs_diff_eq!(weight_to_fb[(0, 0)], 23.0);

    assert_abs_diff_eq!(jones_to_fb[(1, 0)], Jones::identity() * 93.0 / 18.0);
    assert_abs_diff_eq!(weight_to_fb[(1, 0)], 18.0);

    // Make the first channel's weights negative.
    weight_from_tfb[(0, 0, 0)] *= -1.0;
    weight_from_tfb[(1, 0, 0)] *= -1.0;

    vis_average(
        jones_from_tfb.view(),
        jones_to_fb.view_mut(),
        weight_from_tfb.view(),
        weight_to_fb.view_mut(),
        &HashSet::new(),
    );

    assert_abs_diff_eq!(jones_to_fb[(0, 0)], Jones::identity() * 61.0 / 14.0);
    assert_abs_diff_eq!(weight_to_fb[(0, 0)], 14.0);

    assert_abs_diff_eq!(jones_to_fb[(1, 0)], Jones::identity() * 93.0 / 18.0);
    assert_abs_diff_eq!(weight_to_fb[(1, 0)], 18.0);
}
