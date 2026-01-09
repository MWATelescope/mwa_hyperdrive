// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Helper functions for averaging.

mod error;
#[cfg(test)]
mod tests;

pub(crate) use error::AverageFactorError;

use std::{collections::HashSet, num::NonZeroUsize, ops::Range};

use hifitime::{Duration, Epoch};
use itertools::Itertools;
use marlu::Jones;
use ndarray::prelude::*;
use vec1::Vec1;

use crate::unit_parsing::{parse_freq, parse_time, FreqFormat, TimeFormat};

/// A collection of timesteps.
#[derive(Debug, Clone)]
pub struct Timeblock {
    /// The timeblock index. e.g. If all observation timesteps are being used in
    /// a single timeblock, then this index is 0.
    pub index: usize,

    /// The range of indices into an *unflagged* array of visibilities.
    ///
    /// The timesteps comprising a timeblock need not be contiguous, however, we
    /// want the timestep visibilities to be contiguous. Here, `range` indicates
    /// the *unflagged* timestep indices *for this timeblock*. e.g. If timeblock
    /// 0 represents timestep 10 and timeblock 1 represents timesteps 15 and
    /// 16 , then timeblock 0's range is 0..1 (only one index, 0), whereas
    /// timeblock 1's range is 1..3 (indices 1 and 2).
    ///
    /// We can use a range because the timesteps belonging to a timeblock are
    /// always contiguous.
    pub range: Range<usize>,

    /// The timestamps comprising this timeblock. These are determined by the
    /// timesteps into all available timestamps.
    pub timestamps: Vec1<Epoch>,

    /// These are the indices (0 indexed) that map the incoming timestamps to
    /// the timestamps that are available in this `Timeblock`.
    pub timesteps: Vec1<usize>,

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
    /// chanblocks are flagged, then the first unflagged chanblock has a
    /// chanblock_index of 2 but an unflagged_index of 0.
    pub chanblock_index: u16,

    /// The index into an *unflagged* array of visibilities. Regardless of the
    /// first unflagged chanblock's `chanblock_index`, its `unflagged_index`
    /// is 0.
    pub unflagged_index: u16,

    /// The centroid frequency for this chanblock \[Hz\].
    pub freq: f64,
}

/// A spectral windows, a.k.a. a contiguous-band of fine-frequency channels
/// (possibly made up of multiple contiguous coarse channels). Multiple `Spw`s
/// allow a "picket fence" observation to be represented.
#[derive(Debug)]
pub(crate) struct Spw {
    /// The unflagged [`Chanblock`]s in this [`Spw`].
    pub(crate) chanblocks: Vec<Chanblock>,

    /// The indices of the flagged channels in the un-averaged input data.
    ///
    /// The type is `u16` to keep the memory usage down; these probably need to
    /// be promoted to `usize` when being used.
    pub(crate) flagged_chan_indices: HashSet<u16>,

    /// The indices of the flagged chanblocks.
    ///
    /// The type is `u16` to keep the memory usage down; these probably need to
    /// be promoted to `usize` when being used.
    pub(crate) flagged_chanblock_indices: HashSet<u16>,

    /// The number of channels to average per chanblock.
    pub(crate) chans_per_chanblock: NonZeroUsize,

    /// The frequency gap between consecutive chanblocks \[Hz\].
    pub(crate) freq_res: f64,

    /// The first chanblock's centroid frequency (may be flagged) \[Hz\].
    pub(crate) first_freq: f64,
}

impl Spw {
    /// Get all the frequencies of a spectral window (flagged and unflagged).
    pub(crate) fn get_all_freqs(&self) -> Vec1<f64> {
        let n = self.chanblocks.len() + self.flagged_chanblock_indices.len();
        let mut freqs = Vec::with_capacity(n);
        for i in 0..n {
            freqs.push((i as f64).mul_add(self.freq_res, self.first_freq));
        }
        Vec1::try_from_vec(freqs).expect("unlikely to fail as a SPW should have at least 1 channel")
    }
}

/// Given *all* the available timestamps in some input data, the number of
/// timesteps to average together into a timeblock and which timesteps to use,
/// return timeblocks. Timestamps and timesteps must be ascendingly sorted. If
/// `timesteps_to_use` isn't given, this function assumes all timestamps will
/// be used.
///
/// The timestamps must be regular in some time resolution, but gaps are
/// allowed; e.g. [100, 101, 103, 104] is valid, can the code will determine a
/// time resolution of 1.
pub(super) fn timesteps_to_timeblocks(
    all_timestamps: &Vec1<Epoch>,
    time_resolution: Duration,
    time_average_factor: NonZeroUsize,
    timesteps_to_use: Option<&Vec1<usize>>,
) -> Vec1<Timeblock> {
    let (timestamps_to_use, timesteps_to_use) = match timesteps_to_use {
        Some(timesteps_to_use) => {
            let timestamps_to_use = timesteps_to_use.mapped_ref(
                |&t_step|
                // TODO: Handle incorrect timestep indices.
                *all_timestamps.get(t_step).expect("timestep correctly indexes timestamps"), // Could use square brackets, but this way the potential error is clear.
            );
            (timestamps_to_use, timesteps_to_use.clone())
        }
        None => (
            all_timestamps.clone(),
            Vec1::try_from_vec((0..all_timestamps.len()).collect::<Vec<_>>())
                .expect("cannot be empty"),
        ),
    };

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

    // Safety check: if time resolution is zero, we can't create proper timeblocks
    // This prevents infinite loops in the following iteration
    if time_resolution.total_nanoseconds() == 0 {
        // For zero time resolution, create a single timeblock with all timestamps
        let timeblock = Timeblock {
            index: 0,
            range: 0..timestamps_to_use.len(),
            timestamps: timestamps_to_use.clone(),
            timesteps: timesteps_to_use.clone(),
            median: *timestamps_to_use.first(),
        };
        return Vec1::try_from_vec(vec![timeblock]).expect("cannot be empty");
    }

    let timeblock_length = Duration::from_total_nanoseconds(
        (time_average_factor.get() - 1) as i128 * time_resolution.total_nanoseconds(),
    );
    let half_a_timeblock = timeblock_length / 2;
    let first_timestamp = *timestamps_to_use.first();
    let last_timestamp = *timestamps_to_use.last();
    let time_res = time_resolution.total_nanoseconds() as u128;
    let time_average_factor = time_average_factor.get() as u128;
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

        let (timeblock_timestamps, timeblock_timesteps): (Vec<Epoch>, Vec<usize>) =
            timestamps_to_use
                .iter()
                .zip(timesteps_to_use.iter())
                .filter_map(|(timestamp, timestep)| {
                    if (timeblock_start..=timeblock_end).contains(timestamp) {
                        Some((*timestamp, *timestep))
                    } else {
                        None
                    }
                })
                .unzip();
        if !timeblock_timestamps.is_empty() {
            let num_timeblock_timestamps = timeblock_timestamps.len();
            timeblocks.push(Timeblock {
                index: timeblock_index,
                range: timestep_index..timestep_index + num_timeblock_timestamps,
                timestamps: Vec1::try_from_vec(timeblock_timestamps).expect("cannot be empty"),
                timesteps: Vec1::try_from_vec(timeblock_timesteps).expect("cannot be empty"),
                median: timeblock_median,
            });
            timeblock_index += 1;
            timestep_index += num_timeblock_timestamps;
        }
    }

    Vec1::try_from_vec(timeblocks).expect("cannot be empty")
}

/// Returns a vector of [`Spw`]s (potentially multiple contiguous-bands of fine
/// channels). If there's more than one [`Spw`], then this is a "picket fence"
/// observation.
pub(super) fn channels_to_chanblocks(
    all_channel_freqs: &[u64],
    freq_resolution: u64,
    freq_average_factor: NonZeroUsize,
    flagged_chan_indices: &HashSet<u16>,
) -> Vec<Spw> {
    // Handle 0 or 1 provided frequencies here.
    match all_channel_freqs {
        [] => return vec![],
        [f] => {
            let spw = if flagged_chan_indices.contains(&0) {
                Spw {
                    chanblocks: vec![],
                    flagged_chan_indices: HashSet::from([0]),
                    flagged_chanblock_indices: HashSet::from([0]),
                    chans_per_chanblock: freq_average_factor,
                    freq_res: freq_resolution as f64,
                    first_freq: *f as f64,
                }
            } else {
                Spw {
                    chanblocks: vec![Chanblock {
                        chanblock_index: 0,
                        unflagged_index: 0,
                        freq: *f as f64,
                    }],
                    flagged_chan_indices: HashSet::new(),
                    flagged_chanblock_indices: HashSet::new(),
                    chans_per_chanblock: freq_average_factor,
                    freq_res: freq_resolution as f64,
                    first_freq: *f as f64,
                }
            };
            return vec![spw];
        }
        _ => (), // More complicated logic needed.
    }

    // Find any picket SPWs here.
    let mut spw_index_ends = vec![];
    (0..)
        .zip(all_channel_freqs.windows(2))
        .for_each(|(i, window)| {
            if window[1] - window[0] > freq_resolution {
                spw_index_ends.push(i + 1);
            }
        });

    let mut spws = Vec::with_capacity(spw_index_ends.len() + 1);
    let biggest_freq_diff = freq_resolution * freq_average_factor.get() as u64;
    let mut chanblocks = vec![];
    let mut flagged_chanblock_indices = HashSet::new();
    let mut i_chanblock = 0;
    let mut i_unflagged_chanblock = 0;
    let mut current_freqs = vec![];
    let mut first_spw_freq = None;
    let mut first_freq = None;
    let mut all_flagged = true;
    let mut this_spw_flagged_chans = HashSet::new();

    for (i_chan, &freq) in (0..).zip(all_channel_freqs.iter()) {
        match first_spw_freq {
            Some(_) => (),
            None => first_spw_freq = Some(freq),
        }
        match first_freq {
            Some(_) => (),
            None => first_freq = Some(freq),
        }

        if freq - first_freq.unwrap() >= biggest_freq_diff {
            if all_flagged {
                flagged_chanblock_indices.insert(i_chanblock);
            } else {
                let centroid_freq = first_freq.unwrap()
                    + freq_resolution / 2 * (freq_average_factor.get() - 1) as u64;
                chanblocks.push(Chanblock {
                    chanblock_index: i_chanblock,
                    unflagged_index: i_unflagged_chanblock,
                    freq: centroid_freq as f64,
                });
                i_unflagged_chanblock += 1;
            }
            current_freqs.clear();
            first_freq = Some(freq);
            all_flagged = true;
            i_chanblock += 1;
        }

        current_freqs.push(freq as f64);
        if flagged_chan_indices.contains(&i_chan) {
            this_spw_flagged_chans.insert(i_chan);
        } else {
            all_flagged = false;
        }

        if spw_index_ends.contains(&i_chan) {
            spws.push(Spw {
                chanblocks: chanblocks.clone(),
                flagged_chan_indices: this_spw_flagged_chans.clone(),
                flagged_chanblock_indices: flagged_chanblock_indices.clone(),
                chans_per_chanblock: freq_average_factor,
                freq_res: biggest_freq_diff as f64,
                first_freq: (first_spw_freq.unwrap()
                    + freq_resolution / 2 * (freq_average_factor.get() - 1) as u64)
                    as f64,
            });
            first_spw_freq = Some(freq);
            chanblocks.clear();
            flagged_chanblock_indices.clear();
            this_spw_flagged_chans.clear();
        }
    }
    // Deal with any leftover data.
    if let Some(first_freq) = first_freq {
        if all_flagged {
            flagged_chanblock_indices.insert(i_chanblock);
        } else {
            let centroid_freq =
                first_freq + freq_resolution / 2 * (freq_average_factor.get() - 1) as u64;
            chanblocks.push(Chanblock {
                chanblock_index: i_chanblock,
                unflagged_index: i_unflagged_chanblock,
                freq: centroid_freq as f64,
            });
        }
        spws.push(Spw {
            chanblocks,
            flagged_chan_indices: this_spw_flagged_chans,
            flagged_chanblock_indices,
            chans_per_chanblock: freq_average_factor,
            freq_res: biggest_freq_diff as f64,
            first_freq: (first_spw_freq.unwrap()
                + freq_resolution / 2 * (freq_average_factor.get() - 1) as u64)
                as f64,
        });
    }

    spws
}

/// nasty hack because peel doesn't work with flagged channels
pub(super) fn unflag_spw(spw: Spw) -> Spw {
    let all_freqs: Vec<u64> = spw.get_all_freqs().iter().map(|&f| f as u64).collect_vec();
    channels_to_chanblocks(
        &all_freqs,
        spw.freq_res as u64,
        NonZeroUsize::new(1).unwrap(),
        &HashSet::new(),
    )
    .swap_remove(0)
}

/// Determine a time average factor given a time resolution and user input. Use
/// the default if the logic here is insufficient.
///
/// When averaging, the user input must be a multiple of the time resolution.
/// This function also checks that the user's input is sensible.
pub(super) fn parse_time_average_factor(
    time_resolution: Option<Duration>,
    user_input_time_factor: Option<&str>,
    default: NonZeroUsize,
) -> Result<NonZeroUsize, AverageFactorError> {
    match (time_resolution, user_input_time_factor.map(parse_time)) {
        (None, _) => {
            // If the time resolution is unknown, we assume it's because there's
            // only one timestep.
            Ok(NonZeroUsize::new(1).unwrap())
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

            let u = factor.round() as usize;
            Ok(NonZeroUsize::new(u).expect("is not 0"))
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

            let u = factor.round() as usize;
            Ok(NonZeroUsize::new(u).expect("is not 0"))
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
    default: NonZeroUsize,
) -> Result<NonZeroUsize, AverageFactorError> {
    match (freq_resolution, user_input_freq_factor.map(parse_freq)) {
        (None, _) => {
            // If the freq. resolution is unknown, we assume it's because
            // there's only one channel.
            Ok(NonZeroUsize::new(1).unwrap())
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

            let u = factor.round() as usize;
            Ok(NonZeroUsize::new(u).expect("is not 0"))
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
                FreqFormat::kHz => 1e3 * quantity,
                FreqFormat::MHz => 1e6 * quantity,
            };
            let factor = quantity / freq_res;
            // Reject non-integer floats.
            if factor.fract().abs() > 1e-6 {
                return Err(AverageFactorError::NotIntegerMultiple {
                    out: quantity,
                    inp: freq_res,
                });
            }

            let u = factor.round() as usize;
            Ok(NonZeroUsize::new(u).expect("is not 0"))
        }
    }
}

pub(crate) fn vis_average(
    jones_from_tfb: ArrayView3<Jones<f32>>,
    mut jones_to_fb: ArrayViewMut2<Jones<f32>>,
    weight_from_tfb: ArrayView3<f32>,
    mut weight_to_fb: ArrayViewMut2<f32>,
    flagged_chanblock_indices: &HashSet<u16>,
) {
    let avg_time = jones_from_tfb.len_of(Axis(0));
    let avg_freq = (jones_from_tfb.len_of(Axis(1)) as f64
        / (jones_to_fb.len_of(Axis(0)) + flagged_chanblock_indices.len()) as f64)
        .ceil() as usize;

    // {
    //     assert_eq!(jones_from_tfb.dim(), weight_from_tfb.dim());
    //     assert_eq!(jones_to_fb.dim(), weight_to_fb.dim());
    //     let (_time_from, freq_from, baseline_from) = jones_from_tfb.dim();
    //     let (freqs_to, baseline_to) = jones_to_fb.dim();
    //     assert_eq!(
    //         (freq_from as f64 / avg_freq as f64).floor() as usize,
    //         freqs_to + flagged_chan_indices.len(),
    //     );
    //     assert_eq!(
    //         avg_freq * (freqs_to + flagged_chan_indices.len()),
    //         freq_from
    //     );
    //     assert_eq!(baseline_from, baseline_to);
    // }

    // iterate along time axis in chunks of avg_time
    jones_from_tfb
        .axis_chunks_iter(Axis(0), avg_time)
        .zip(weight_from_tfb.axis_chunks_iter(Axis(0), avg_time))
        .for_each(|(jones_chunk_tfb, weight_chunk_tfb)| {
            jones_chunk_tfb
                .axis_iter(Axis(2))
                .zip(weight_chunk_tfb.axis_iter(Axis(2)))
                .zip(jones_to_fb.axis_iter_mut(Axis(1)))
                .zip(weight_to_fb.axis_iter_mut(Axis(1)))
                .for_each(
                    |(((jones_chunk_tf, weight_chunk_tf), mut jones_to_f), mut weight_to_f)| {
                        jones_chunk_tf
                            .axis_chunks_iter(Axis(1), avg_freq)
                            .zip(weight_chunk_tf.axis_chunks_iter(Axis(1), avg_freq))
                            .enumerate()
                            .filter(|(i, _)| !flagged_chanblock_indices.contains(&(*i as u16)))
                            .map(|(_, d)| d)
                            .zip(jones_to_f.iter_mut())
                            .zip(weight_to_f.iter_mut())
                            .for_each(
                                |(((jones_chunk_tf, weight_chunk_tf), jones_to), weight_to)| {
                                    vis_average_weights_non_zero(
                                        jones_chunk_tf,
                                        weight_chunk_tf,
                                        jones_to,
                                        weight_to,
                                    );
                                },
                            );
                    },
                );
        });
}

/// Average a chunk of visibilities and weights (both must have the same
/// dimensions) into an output vis and weight. This function allows the weights
/// to be negative; if all of the weights in the chunk are negative or 0, the
/// averaged visibility is considered "flagged".
#[inline]
pub(super) fn vis_average_weights_non_zero(
    jones_chunk_tf: ArrayView2<Jones<f32>>,
    weight_chunk_tf: ArrayView2<f32>,
    jones_to: &mut Jones<f32>,
    weight_to: &mut f32,
) {
    let mut jones_weighted_sum = Jones::default();
    let mut jones_sum = Jones::default();
    let mut unflagged_weight_sum = 0.0;
    let mut flagged_weight_sum = 0.0;
    let mut all_flagged = true;

    // iterate through time chunks
    jones_chunk_tf
        .iter()
        .zip_eq(weight_chunk_tf.iter())
        .for_each(|(jones, weight)| {
            let jones = Jones::<f64>::from(*jones);
            jones_sum += jones;

            let weight_abs_f64 = (*weight as f64).abs();
            if *weight > 0.0 {
                all_flagged = false;
                jones_weighted_sum += jones * weight_abs_f64;
                unflagged_weight_sum += weight_abs_f64;
            } else {
                flagged_weight_sum += weight_abs_f64;
            }
        });

    if all_flagged || unflagged_weight_sum <= 0.0 {
        *jones_to = Jones::from(jones_sum / jones_chunk_tf.len() as f64);
        *weight_to = -flagged_weight_sum as f32;
    } else {
        *jones_to = Jones::from(jones_weighted_sum / unflagged_weight_sum);
        *weight_to = unflagged_weight_sum as f32;
    }
}
