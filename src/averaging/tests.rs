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

    let time_average_factor = 1;
    let timesteps_to_use = vec1![2, 3, 4, 5];
    let timeblocks =
        timesteps_to_timeblocks(&all_timestamps, time_average_factor, &timesteps_to_use);
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
            average_epoch(&timeblock.timestamps).to_gpst_seconds(),
            expected_timestamp
        );
        assert_eq!(timeblock.median.to_gpst_seconds(), expected_timestamp);
    }

    let time_average_factor = 2;
    let timesteps_to_use = vec1![2, 3, 4, 5];
    let timeblocks =
        timesteps_to_timeblocks(&all_timestamps, time_average_factor, &timesteps_to_use);
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
            average_epoch(&timeblock.timestamps).to_gpst_seconds(),
            expected_timestamp
        );
        assert_eq!(timeblock.median.to_gpst_seconds(), expected_timestamp);
    }

    let time_average_factor = 3;
    let timesteps_to_use = vec1![2, 3, 4, 5];
    let timeblocks =
        timesteps_to_timeblocks(&all_timestamps, time_average_factor, &timesteps_to_use);
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
            average_epoch(&timeblock.timestamps).to_gpst_seconds(),
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
    let time_average_factor = *timesteps_to_use.last() - *timesteps_to_use.first() + 1;
    assert_eq!(time_average_factor, 15);
    let timeblocks =
        timesteps_to_timeblocks(&all_timestamps, time_average_factor, &timesteps_to_use);
    assert_eq!(timeblocks.len(), 1);
    for ((timeblock, expected_indices), expected_timestamp) in
        timeblocks.into_iter().zip([[0, 1, 2]]).zip([1065880150.0])
    {
        assert_eq!(timeblock.range.len(), expected_indices.len());
        for (timestep, expected_index) in timeblock.range.zip(expected_indices) {
            assert_eq!(timestep, expected_index);
        }

        assert_eq!(
            average_epoch(&timeblock.timestamps).to_gpst_seconds(),
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
    let freq_average_factor = 1;
    let mut flagged_channels = HashSet::new();
    let fences = channels_to_chanblocks(
        &all_channel_freqs,
        None,
        freq_average_factor,
        &flagged_channels,
    );
    assert_eq!(fences.len(), 1);
    assert_eq!(fences[0].chanblocks.len(), 1);
    assert!(fences[0].flagged_chanblock_indices.is_empty());
    assert_abs_diff_eq!(fences[0].chanblocks[0]._freq, 12000.0);
    assert_abs_diff_eq!(fences[0]._first_freq, 12000.0);
    assert!(fences[0]._freq_res.is_none());

    let all_channel_freqs = [10000, 11000, 12000, 13000, 14000];
    let fences = channels_to_chanblocks(
        &all_channel_freqs,
        None,
        freq_average_factor,
        &flagged_channels,
    );
    assert_eq!(fences.len(), 1);
    assert_eq!(fences[0].chanblocks.len(), 5);
    assert!(fences[0].flagged_chanblock_indices.is_empty());
    assert_abs_diff_eq!(fences[0].chanblocks[0]._freq, 10000.0);
    assert_abs_diff_eq!(fences[0].chanblocks[1]._freq, 11000.0);
    assert_abs_diff_eq!(fences[0].chanblocks[2]._freq, 12000.0);
    assert_abs_diff_eq!(fences[0].chanblocks[3]._freq, 13000.0);
    assert_abs_diff_eq!(fences[0].chanblocks[4]._freq, 14000.0);
    assert_abs_diff_eq!(fences[0]._first_freq, 10000.0);
    assert_abs_diff_eq!(fences[0]._freq_res.unwrap(), 1000.0);

    let all_channel_freqs = [10000, 11000, 12000, 13000, 14000, 20000];
    let fences = channels_to_chanblocks(
        &all_channel_freqs,
        None,
        freq_average_factor,
        &flagged_channels,
    );
    assert_eq!(fences.len(), 2);
    assert_eq!(fences[0].chanblocks.len(), 5);
    assert_eq!(fences[1].chanblocks.len(), 1);
    assert!(fences[0].flagged_chanblock_indices.is_empty());
    assert!(fences[1].flagged_chanblock_indices.is_empty());
    assert_abs_diff_eq!(fences[0].chanblocks[0]._freq, 10000.0);
    assert_abs_diff_eq!(fences[0].chanblocks[1]._freq, 11000.0);
    assert_abs_diff_eq!(fences[0].chanblocks[2]._freq, 12000.0);
    assert_abs_diff_eq!(fences[0].chanblocks[3]._freq, 13000.0);
    assert_abs_diff_eq!(fences[0].chanblocks[4]._freq, 14000.0);
    assert_abs_diff_eq!(fences[1].chanblocks[0]._freq, 20000.0);
    assert_abs_diff_eq!(fences[0]._first_freq, 10000.0);
    assert_abs_diff_eq!(fences[1]._first_freq, 20000.0);
    assert_abs_diff_eq!(fences[0]._freq_res.unwrap(), 1000.0);
    assert_abs_diff_eq!(fences[1]._freq_res.unwrap(), 1000.0);

    flagged_channels.insert(3);
    let fences = channels_to_chanblocks(
        &all_channel_freqs,
        None,
        freq_average_factor,
        &flagged_channels,
    );
    assert_eq!(fences.len(), 2);
    assert_eq!(fences[0].chanblocks.len(), 4);
    assert_eq!(fences[1].chanblocks.len(), 1);
    assert_eq!(fences[0].flagged_chanblock_indices.len(), 1);
    assert_eq!(fences[0].flagged_chanblock_indices[0], 3);
    assert!(fences[1].flagged_chanblock_indices.is_empty());
    assert_abs_diff_eq!(fences[0].chanblocks[0]._freq, 10000.0);
    assert_abs_diff_eq!(fences[0].chanblocks[1]._freq, 11000.0);
    assert_abs_diff_eq!(fences[0].chanblocks[2]._freq, 12000.0);
    assert_abs_diff_eq!(fences[0].chanblocks[3]._freq, 14000.0);
    assert_abs_diff_eq!(fences[1].chanblocks[0]._freq, 20000.0);
    assert_abs_diff_eq!(fences[0]._first_freq, 10000.0);
    assert_abs_diff_eq!(fences[1]._first_freq, 20000.0);
    assert_abs_diff_eq!(fences[0]._freq_res.unwrap(), 1000.0);
    assert_abs_diff_eq!(fences[1]._freq_res.unwrap(), 1000.0);

    let freq_average_factor = 2;
    let fences = channels_to_chanblocks(
        &all_channel_freqs,
        None,
        freq_average_factor,
        &flagged_channels,
    );
    assert_eq!(fences.len(), 2);
    assert_eq!(fences[0].chanblocks.len(), 3);
    assert_eq!(fences[1].chanblocks.len(), 1);
    assert!(fences[0].flagged_chanblock_indices.is_empty());
    assert!(fences[1].flagged_chanblock_indices.is_empty());
    assert_abs_diff_eq!(fences[0].chanblocks[0]._freq, 10500.0);
    assert_abs_diff_eq!(fences[0].chanblocks[1]._freq, 12500.0);
    assert_abs_diff_eq!(fences[0].chanblocks[2]._freq, 14500.0);
    assert_abs_diff_eq!(fences[1].chanblocks[0]._freq, 20500.0);
    assert_abs_diff_eq!(fences[0]._first_freq, 10000.0);
    assert_abs_diff_eq!(fences[1]._first_freq, 20000.0);
    assert_abs_diff_eq!(fences[0]._freq_res.unwrap(), 2000.0);
    assert_abs_diff_eq!(fences[1]._freq_res.unwrap(), 2000.0);

    let freq_average_factor = 3;
    let fences = channels_to_chanblocks(
        &all_channel_freqs,
        None,
        freq_average_factor,
        &flagged_channels,
    );
    assert_eq!(fences.len(), 2);
    assert_eq!(fences[0].chanblocks.len(), 2);
    assert_eq!(fences[1].chanblocks.len(), 1);
    assert!(fences[0].flagged_chanblock_indices.is_empty());
    assert!(fences[1].flagged_chanblock_indices.is_empty());
    assert_abs_diff_eq!(fences[0].chanblocks[0]._freq, 11000.0);
    assert_abs_diff_eq!(fences[0].chanblocks[1]._freq, 14000.0);
    assert_abs_diff_eq!(fences[1].chanblocks[0]._freq, 21000.0);
    assert_abs_diff_eq!(fences[0]._first_freq, 10000.0);
    assert_abs_diff_eq!(fences[1]._first_freq, 20000.0);
    assert_abs_diff_eq!(fences[0]._freq_res.unwrap(), 3000.0);
    assert_abs_diff_eq!(fences[1]._freq_res.unwrap(), 3000.0);
}

// No frequencies, no fences.
#[test]
fn test_no_channels_to_chanblocks() {
    let all_channel_freqs = [];
    let freq_average_factor = 2;
    let flagged_channels = HashSet::new();
    let fences = channels_to_chanblocks(
        &all_channel_freqs,
        None,
        freq_average_factor,
        &flagged_channels,
    );
    assert!(fences.is_empty());
}

fn test_time(
    time_resolution: Option<Duration>,
    user_input_time_factor: Option<&str>,
    default: usize,
    expected: Option<usize>,
) {
    let result = parse_time_average_factor(time_resolution, user_input_time_factor, default);

    match expected {
        // If expected is Some, then we expect the result to match.
        Some(expected) => {
            assert!(result.is_ok(), "res={time_resolution:?}, input={user_input_time_factor:?}, default={default}, expected={expected:?}, error={}", result.unwrap_err());
            assert_eq!(result.unwrap(), expected, "res={time_resolution:?}, input={user_input_time_factor:?}, default={default}, expected={expected:?}");
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
    let default = 100;

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
    let expected = Some(default);
    test_time(time_resolution, user_input_time_factor, default, expected);
}

fn test_freq(
    freq_resolution: Option<f64>,
    user_input_freq_factor: Option<&str>,
    default: usize,
    expected: Option<usize>,
) {
    let result = parse_freq_average_factor(freq_resolution, user_input_freq_factor, default);

    match expected {
        // If expected is Some, then we expect the result to match.
        Some(expected) => {
            assert!(result.is_ok(), "res={freq_resolution:?}, input={user_input_freq_factor:?}, default={default}, expected={expected:?}, error={}", result.unwrap_err());
            assert_eq!(result.unwrap(), expected, "res={freq_resolution:?}, input={user_input_freq_factor:?}, default={default}, expected={expected:?}");
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
    let default = 1;

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
    let expected = Some(default);
    test_freq(freq_resolution, user_input_freq_factor, default, expected);
}
