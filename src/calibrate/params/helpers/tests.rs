// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

use std::collections::HashSet;

use approx::assert_abs_diff_eq;
use hifitime::Epoch;

use super::*;
use mwa_hyperdrive_common::hifitime;

#[test]
fn test_timesteps_to_timeblocks() {
    let all_timestamps: Vec<Epoch> = (0..20)
        .into_iter()
        .map(|i| Epoch::from_gpst_seconds(1065880128.0 + (2 * i) as f64))
        .collect();

    let time_average_factor = 1;
    let timesteps_to_use = [2, 3, 4, 5];
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
        for (timestep, expected_index) in timeblock.range.into_iter().zip(expected_indices) {
            assert_eq!(timestep, expected_index);
        }

        assert_eq!(timeblock.average.as_gpst_seconds(), expected_timestamp);
    }

    let time_average_factor = 2;
    let timesteps_to_use = [2, 3, 4, 5];
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
        for (timestep, expected_index) in timeblock.range.into_iter().zip(expected_indices) {
            assert_eq!(timestep, expected_index);
        }

        assert_eq!(timeblock.average.as_gpst_seconds(), expected_timestamp);
    }

    let time_average_factor = 3;
    let timesteps_to_use = [2, 3, 4, 5];
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
        for (timestep, expected_index) in timeblock.range.into_iter().zip(expected_indices) {
            assert_eq!(timestep, expected_index);
        }

        assert_eq!(timeblock.average.as_gpst_seconds(), expected_timestamp);
    }

    let timesteps_to_use = [2, 15, 16];
    // Average all the timesteps together. This is what is used to calculate the
    // time average factor in this case.
    let time_average_factor =
        *timesteps_to_use.last().unwrap() - *timesteps_to_use.first().unwrap() + 1;
    assert_eq!(time_average_factor, 15);
    let timeblocks =
        timesteps_to_timeblocks(&all_timestamps, time_average_factor, &timesteps_to_use);
    assert_eq!(timeblocks.len(), 1);
    for ((timeblock, expected_indices), expected_timestamp) in
        timeblocks.into_iter().zip([[0, 1, 2]]).zip([1065880150.0])
    {
        assert_eq!(timeblock.range.len(), expected_indices.len());
        for (timestep, expected_index) in timeblock.range.into_iter().zip(expected_indices) {
            assert_eq!(timestep, expected_index);
        }

        assert_eq!(timeblock.average.as_gpst_seconds(), expected_timestamp);
    }

    // No timesteps, no timeblocks.
    let timesteps_to_use = [];
    let time_average_factor = 1;
    let timeblocks =
        timesteps_to_timeblocks(&all_timestamps, time_average_factor, &timesteps_to_use);
    assert_eq!(timeblocks.len(), 0);
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
    assert_abs_diff_eq!(fences[0].chanblocks[0].freq, 12000.0);
    assert_abs_diff_eq!(fences[0].first_freq, 12000.0);
    assert!(fences[0].freq_res.is_none());

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
    assert_abs_diff_eq!(fences[0].chanblocks[0].freq, 10000.0);
    assert_abs_diff_eq!(fences[0].chanblocks[1].freq, 11000.0);
    assert_abs_diff_eq!(fences[0].chanblocks[2].freq, 12000.0);
    assert_abs_diff_eq!(fences[0].chanblocks[3].freq, 13000.0);
    assert_abs_diff_eq!(fences[0].chanblocks[4].freq, 14000.0);
    assert_abs_diff_eq!(fences[0].first_freq, 10000.0);
    assert_abs_diff_eq!(fences[0].freq_res.unwrap(), 1000.0);

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
    assert_abs_diff_eq!(fences[0].chanblocks[0].freq, 10000.0);
    assert_abs_diff_eq!(fences[0].chanblocks[1].freq, 11000.0);
    assert_abs_diff_eq!(fences[0].chanblocks[2].freq, 12000.0);
    assert_abs_diff_eq!(fences[0].chanblocks[3].freq, 13000.0);
    assert_abs_diff_eq!(fences[0].chanblocks[4].freq, 14000.0);
    assert_abs_diff_eq!(fences[1].chanblocks[0].freq, 20000.0);
    assert_abs_diff_eq!(fences[0].first_freq, 10000.0);
    assert_abs_diff_eq!(fences[1].first_freq, 20000.0);
    assert_abs_diff_eq!(fences[0].freq_res.unwrap(), 1000.0);
    assert_abs_diff_eq!(fences[1].freq_res.unwrap(), 1000.0);

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
    assert_abs_diff_eq!(fences[0].chanblocks[0].freq, 10000.0);
    assert_abs_diff_eq!(fences[0].chanblocks[1].freq, 11000.0);
    assert_abs_diff_eq!(fences[0].chanblocks[2].freq, 12000.0);
    assert_abs_diff_eq!(fences[0].chanblocks[3].freq, 14000.0);
    assert_abs_diff_eq!(fences[1].chanblocks[0].freq, 20000.0);
    assert_abs_diff_eq!(fences[0].first_freq, 10000.0);
    assert_abs_diff_eq!(fences[1].first_freq, 20000.0);
    assert_abs_diff_eq!(fences[0].freq_res.unwrap(), 1000.0);
    assert_abs_diff_eq!(fences[1].freq_res.unwrap(), 1000.0);

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
    assert_abs_diff_eq!(fences[0].chanblocks[0].freq, 10500.0);
    assert_abs_diff_eq!(fences[0].chanblocks[1].freq, 12500.0);
    assert_abs_diff_eq!(fences[0].chanblocks[2].freq, 14500.0);
    assert_abs_diff_eq!(fences[1].chanblocks[0].freq, 20500.0);
    assert_abs_diff_eq!(fences[0].first_freq, 10000.0);
    assert_abs_diff_eq!(fences[1].first_freq, 20000.0);
    assert_abs_diff_eq!(fences[0].freq_res.unwrap(), 2000.0);
    assert_abs_diff_eq!(fences[1].freq_res.unwrap(), 2000.0);

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
    assert_abs_diff_eq!(fences[0].chanblocks[0].freq, 11000.0);
    assert_abs_diff_eq!(fences[0].chanblocks[1].freq, 14000.0);
    assert_abs_diff_eq!(fences[1].chanblocks[0].freq, 21000.0);
    assert_abs_diff_eq!(fences[0].first_freq, 10000.0);
    assert_abs_diff_eq!(fences[1].first_freq, 20000.0);
    assert_abs_diff_eq!(fences[0].freq_res.unwrap(), 3000.0);
    assert_abs_diff_eq!(fences[1].freq_res.unwrap(), 3000.0);
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
