// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Helper functions for averaging.

mod error;
#[cfg(test)]
mod tests;

pub(crate) use error::AverageFactorError;

use std::collections::HashSet;
use std::ops::Range;

use hifitime::{Duration, Epoch};
use vec1::Vec1;

use crate::unit_parsing::{parse_freq, parse_time, FreqFormat, TimeFormat};

/// A collection of timesteps.
#[derive(Debug, Clone)]
pub struct Timeblock {
    /// The timeblock index. e.g. If all observation timesteps are being used in
    /// a single calibration timeblock, then its index is 0.
    pub index: usize,

    /// The range of indices into an *unflagged* array of visibilities.
    ///
    /// The timesteps comprising a timeblock need not be contiguous, however, we
    /// want the timestep visibilities to be contiguous. Here, `range` indicates
    /// the *unflagged* timestep indices *for this timeblock*. e.g. If timeblock
    /// 0 represents timestep 10 and timeblock 1 represents timesteps 15 and 16
    /// (and these are the only timesteps used for calibration), then timeblock
    /// 0's range is 0..1 (only one index, 0), whereas timeblock 1's range is
    /// 1..3 (two indices starting at 1).
    ///
    /// We can use a range because the timesteps belonging to a timeblock are
    /// always contiguous.
    pub range: Range<usize>,

    /// The timestamps comprising this timeblock. These are determined by the
    /// timesteps into all available timestamps.
    pub timestamps: Vec1<Epoch>,

    /// The median timestamp of the *ideal* timeblock.
    ///
    /// e.g. If we have 9 timesteps and we're averaging 3, the averaged
    /// timeblocks look like this:
    ///
    /// [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
    ///
    /// But if we're only using timesteps [1, 3, 8], the timeblocks look like
    /// this.
    ///
    /// [[1, _, 3], [_, _, _], [_, 8]]
    ///
    /// In the first case, this `median` is [1, 4, 7] for each timeblock, [2, 5,
    /// 8] for the second. Note how missing timestamps don't affect it.
    pub median: Epoch,
}

/// A collection of fine-frequency channels.
#[derive(Debug, Clone)]
pub struct Chanblock {
    /// The chanblock index, regardless of flagging. e.g. If the first two
    /// calibration chanblocks are flagged, then the first unflagged chanblock
    /// has a chanblock_index of 2 but an unflagged_index of 0.
    pub chanblock_index: u16,

    /// The index into an *unflagged* array of visibilities. Regardless of the
    /// first unflagged chanblock's index, its unflagged index is 0.
    pub unflagged_index: u16,

    // TODO: Use frequency information. May become important for calibration
    // solutions and what frequencies they apply to.
    /// The centroid frequency for this chanblock \[Hz\].
    pub _freq: f64,
}

/// A spectral windows, a.k.a. a contiguous-band of fine-frequency channels
/// (possibly made up of multiple contiguous coarse channels). Multiple `Fence`s
/// allow a "picket fence" observation to be represented. Calibration is run on
/// each independent `Fence`.
#[derive(Debug)]
pub(crate) struct Fence {
    /// The unflagged calibration [Chanblock]s in this [Fence].
    pub(crate) chanblocks: Vec<Chanblock>,

    /// The indices of the flagged chanblocks.
    ///
    /// The type is `u16` to keep the memory usage down; these probably need to
    /// be promoted to `usize` when being used.
    pub(crate) flagged_chanblock_indices: Vec<u16>,

    /// The first chanblock's centroid frequency (may be flagged) \[Hz\].
    pub(crate) _first_freq: f64,

    /// The frequency gap between consecutive chanblocks \[Hz\]. If this isn't
    /// defined, it's because there's only one chanblock.
    pub(crate) _freq_res: Option<f64>,
}

impl Fence {
    fn _get_total_num_chanblocks(&self) -> usize {
        self.chanblocks.len() + self.flagged_chanblock_indices.len()
    }

    fn _get_freqs(&self) -> Vec<f64> {
        if let Some(freq_res) = self._freq_res {
            (0..self._get_total_num_chanblocks())
                .map(|i_chanblock| self._first_freq + i_chanblock as f64 * freq_res)
                .collect()
        } else {
            vec![self._first_freq]
        }
    }
}

/// Given *all* the available timestamps in some input data, the number of
/// timesteps to average together into a timeblock and which timesteps to use,
/// return timeblocks to be used for calibration. Timestamps and timesteps must
/// be ascendingly sorted.
///
/// The timestamps must be regular in some time resolution, but gaps are
/// allowed; e.g. [100, 101, 103, 104] is valid, can the code will determine a
/// time resolution of 1.
pub(super) fn timesteps_to_timeblocks(
    all_timestamps: &Vec1<Epoch>,
    time_average_factor: usize,
    timesteps_to_use: &Vec1<usize>,
) -> Vec1<Timeblock> {
    let time_res = all_timestamps
        .windows(2)
        .fold(Duration::from_seconds(f64::INFINITY), |a, t| {
            a.min(t[1] - t[0])
        });
    let timestamps_to_use = timesteps_to_use.mapped_ref(
        |&t_step|
            // TODO: Handle incorrect timestep indices.
            *all_timestamps.get(t_step).unwrap(), // Could use square brackets, but this way the unwrap is clear.
    );

    // Populate the median timestamps of all timeblocks based off of the first
    // timestamp. e.g. If there are 10 timestamps with an averaging factor of 3,
    // these are some possible situations:
    //
    // [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9]]
    //
    // [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    //
    // [[2, 3, 4], [5, 6, 7], [8, 9]]
    //
    // Each depends on the first timestep being used. For each of the respective
    // situations above, the results are:
    //
    // [(1, 0..=2), (4, 3..=5), (7, 6..=8), (10, 9..=11)]
    //
    // [(2, 1..=3), (5, 4..=6), (8, 7..=9)]
    //
    // [(3, 2..=4), (6, 5..=7), (9, 8..=10)]

    let mut timeblocks = vec![];
    let timeblock_length = Duration::from_total_nanoseconds(
        // time_average_factor as i128 * time_res.total_nanoseconds(),
        (time_average_factor - 1) as i128 * time_res.total_nanoseconds(),
    );
    let half_a_timeblock = timeblock_length / 2;
    let first_timestamp = *timestamps_to_use.first();
    let last_timestamp = *timestamps_to_use.last();
    let time_res = time_res.total_nanoseconds() as u128;
    let time_average_factor = time_average_factor as u128;
    let mut timeblock_index = 0;
    let mut timestep_index = 0;
    for i in 0.. {
        // `timeblock_start` and `timeblock_end` are not centroids but "leading
        // edge", however `timeblock_median` is a centroid.
        let timeblock_start = first_timestamp
            + Duration::from_total_nanoseconds(
                (time_res
                    .checked_mul(i)
                    .unwrap()
                    .checked_mul(time_average_factor)
                    .unwrap()) as i128,
            );
        let timeblock_end = timeblock_start + timeblock_length;
        let timeblock_median = timeblock_start + half_a_timeblock;

        if timeblock_start > last_timestamp {
            break;
        }

        let timeblock_timestamps = timestamps_to_use
            .iter()
            .filter(|ts| (timeblock_start..=timeblock_end).contains(ts))
            .copied()
            .collect::<Vec<_>>();
        if !timeblock_timestamps.is_empty() {
            let num_timeblock_timestamps = timeblock_timestamps.len();
            timeblocks.push(Timeblock {
                index: timeblock_index,
                range: timestep_index..timestep_index + num_timeblock_timestamps,
                timestamps: Vec1::try_from_vec(timeblock_timestamps).unwrap(),
                median: timeblock_median,
            });
            timeblock_index += 1;
            timestep_index += num_timeblock_timestamps;
        }
    }

    Vec1::try_from_vec(timeblocks).unwrap()
}

/// Returns a vector of [Fence]s (potentially multiple contiguous-bands of fine
/// channels) to use in calibration. If there's more than one [Fence], then this
/// is a "picket fence" observation.
pub(super) fn channels_to_chanblocks(
    all_channel_freqs: &[u64],
    frequency_resolution: Option<f64>,
    freq_average_factor: usize,
    flagged_channels: &HashSet<usize>,
) -> Vec<Fence> {
    // Handle 0 or 1 provided frequencies here.
    match all_channel_freqs {
        [] => return vec![],
        [f] => {
            let (chanblocks, flagged_chanblock_indices) = if flagged_channels.contains(&0) {
                (vec![], vec![0])
            } else {
                (
                    vec![Chanblock {
                        chanblock_index: 0,
                        unflagged_index: 0,
                        _freq: *f as f64,
                    }],
                    vec![],
                )
            };
            return vec![Fence {
                chanblocks,
                flagged_chanblock_indices,
                _first_freq: *f as f64,
                _freq_res: None,
            }];
        }
        _ => (), // More complicated logic needed.
    }

    // If the frequency resolution wasn't provided, we find the minimum gap
    // between consecutive frequencies and use this instead.
    let freq_res = frequency_resolution
        .map(|f| f.round() as u64)
        .unwrap_or_else(|| {
            // Iterate over all the frequencies and find the smallest gap between
            // any pair.
            all_channel_freqs.windows(2).fold(u64::MAX, |acc, window| {
                let diff = window[1] - window[0];
                acc.min(diff)
            })
        });

    // Find any picket fences here.
    let mut fence_index_ends = vec![];
    all_channel_freqs
        .windows(2)
        .enumerate()
        .for_each(|(i, window)| {
            if window[1] - window[0] > freq_res {
                fence_index_ends.push(i + 1);
            }
        });

    let mut fences = Vec::with_capacity(fence_index_ends.len() + 1);
    let biggest_freq_diff = freq_res * freq_average_factor as u64;
    let mut chanblocks = vec![];
    let mut flagged_chanblock_indices = vec![];
    let mut i_chanblock = 0;
    let mut i_unflagged_chanblock = 0;
    let mut current_freqs = vec![];
    let mut first_fence_freq = None;
    let mut first_freq = None;
    let mut all_flagged = true;

    for (i_chan, &freq) in all_channel_freqs.iter().enumerate() {
        match first_fence_freq {
            Some(_) => (),
            None => first_fence_freq = Some(freq),
        }
        match first_freq {
            Some(_) => (),
            None => first_freq = Some(freq),
        }

        if freq - first_freq.unwrap() >= biggest_freq_diff {
            if all_flagged {
                flagged_chanblock_indices.push(i_chanblock);
            } else {
                let centroid_freq =
                    first_freq.unwrap() + freq_res / 2 * (freq_average_factor - 1) as u64;
                chanblocks.push(Chanblock {
                    chanblock_index: i_chanblock,
                    unflagged_index: i_unflagged_chanblock,
                    _freq: centroid_freq as f64,
                });
                i_unflagged_chanblock += 1;
            }
            current_freqs.clear();
            first_freq = Some(freq);
            all_flagged = true;
            i_chanblock += 1;
        }

        current_freqs.push(freq as f64);
        if !flagged_channels.contains(&i_chan) {
            all_flagged = false;
        }

        if fence_index_ends.contains(&i_chan) {
            fences.push(Fence {
                chanblocks: chanblocks.clone(),
                flagged_chanblock_indices: flagged_chanblock_indices.clone(),
                _first_freq: first_fence_freq.unwrap() as f64,
                _freq_res: Some(biggest_freq_diff as f64),
            });
            first_fence_freq = Some(freq);
            chanblocks.clear();
            flagged_chanblock_indices.clear();
        }
    }
    // Deal with any leftover data.
    if let Some(first_freq) = first_freq {
        if all_flagged {
            flagged_chanblock_indices.push(i_chanblock);
        } else {
            let centroid_freq = first_freq + freq_res / 2 * (freq_average_factor - 1) as u64;
            chanblocks.push(Chanblock {
                chanblock_index: i_chanblock,
                unflagged_index: i_unflagged_chanblock,
                _freq: centroid_freq as f64,
            });
        }
        fences.push(Fence {
            chanblocks,
            flagged_chanblock_indices,
            _first_freq: first_fence_freq.unwrap() as f64,
            _freq_res: Some(biggest_freq_diff as f64),
        });
    }

    fences
}

/// Determine a time average factor given a time resolution and user input. Use
/// the default if the logic here is insufficient.
///
/// When averaging, the user input must be a multiple of the time resolution.
/// This function also checks that the user's input is sensible.
pub(super) fn parse_time_average_factor(
    time_resolution: Option<Duration>,
    user_input_time_factor: Option<&str>,
    default: usize,
) -> Result<usize, AverageFactorError> {
    match (time_resolution, user_input_time_factor.map(parse_time)) {
        (None, _) => {
            // If the time resolution is unknown, we assume it's because there's
            // only one timestep.
            Ok(1)
        }
        (_, None) => {
            // "None" indicates we should follow default behaviour.
            Ok(default)
        }
        // propagate any errors encountered during parsing.
        (_, Some(Err(e))) => Err(AverageFactorError::Parse(e)),

        // User input is OK but has no unit.
        (_, Some(Ok((factor, None)))) => {
            // Zero is not allowed.
            if factor < f64::EPSILON {
                return Err(AverageFactorError::Zero);
            }
            // Reject non-integer floats.
            if (factor - factor.round()).abs() > 1e-6 {
                return Err(AverageFactorError::NotInteger);
            }

            Ok(factor.round() as _)
        }

        // User input is OK and has a unit.
        (Some(time_res), Some(Ok((quantity, Some(time_format))))) => {
            // Zero is not allowed.
            if quantity < f64::EPSILON {
                return Err(AverageFactorError::Zero);
            }

            // Scale the quantity by the unit, if required.
            let quantity = match time_format {
                TimeFormat::s => quantity,
                TimeFormat::ms => quantity / 1e3,
            };
            let factor = quantity / time_res.to_seconds();
            // Reject non-integer floats.
            if factor.fract().abs() > 1e-6 {
                return Err(AverageFactorError::NotIntegerMultiple {
                    out: quantity,
                    inp: time_res.to_seconds(),
                });
            }

            Ok(factor.round() as _)
        }
    }
}

/// Determine a frequency average factor given a freq. resolution and user
/// input. Use the default if the logic here is insufficient.
///
/// When averaging, the user input must be a multiple of the freq. resolution.
/// This function also checks that the user's input is sensible.
pub(super) fn parse_freq_average_factor(
    freq_resolution: Option<f64>,
    user_input_freq_factor: Option<&str>,
    default: usize,
) -> Result<usize, AverageFactorError> {
    match (freq_resolution, user_input_freq_factor.map(parse_freq)) {
        (None, _) => {
            // If the freq. resolution is unknown, we assume it's because
            // there's only one channel.
            Ok(1)
        }
        (_, None) => {
            // "None" indicates we should follow default behaviour.
            Ok(default)
        }
        // propagate any errors encountered during parsing.
        (_, Some(Err(e))) => Err(AverageFactorError::Parse(e)),

        // User input is OK but has no unit.
        (_, Some(Ok((factor, None)))) => {
            // Zero is not allowed.
            if factor < f64::EPSILON {
                return Err(AverageFactorError::Zero);
            }
            // Reject non-integer floats.
            if (factor - factor.round()).abs() > 1e-6 {
                return Err(AverageFactorError::NotInteger);
            }

            Ok(factor.round() as _)
        }

        // User input is OK and has a unit.
        (Some(freq_res), Some(Ok((quantity, Some(time_format))))) => {
            // Zero is not allowed.
            if quantity < f64::EPSILON {
                return Err(AverageFactorError::Zero);
            }

            // Scale the quantity by the unit, if required.
            let quantity = match time_format {
                FreqFormat::Hz => quantity,
                FreqFormat::kHz => 1000.0 * quantity,
            };
            let factor = quantity / freq_res;
            // Reject non-integer floats.
            if factor.fract().abs() > 1e-6 {
                return Err(AverageFactorError::NotIntegerMultiple {
                    out: quantity,
                    inp: freq_res,
                });
            }

            Ok(factor.round() as _)
        }
    }
}
