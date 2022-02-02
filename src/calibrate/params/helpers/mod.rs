// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Helper functions when converting arguments to parameters.
//!
//! The isolation of these functions is mostly useful for writing unit tests.

#[cfg(test)]
mod tests;

use std::collections::HashSet;

use hifitime::Epoch;
use thiserror::Error;

use crate::{
    calibrate::{Chanblock, Timeblock},
    math::average_epoch,
    unit_parsing::{parse_freq, parse_time, FreqFormat, TimeFormat},
};
use mwa_hyperdrive_common::{hifitime, thiserror};

/// Given *all* the available timestamps in some input data, the number of
/// timesteps to average together into a timeblock and which timesteps to use,
/// return timeblocks to be used for calibration.
pub(super) fn timesteps_to_timeblocks(
    all_timestamps: &[Epoch],
    time_average_factor: usize,
    timesteps_to_use: &[usize],
) -> Vec<Timeblock> {
    let mut timeblocks = vec![];
    let mut first_index = None;
    let mut first_timestamp = None;
    let mut current_indices = vec![];
    let mut current_timestamps = vec![];
    let mut timeblock_index = 0;
    for (i_timestep, &timestep) in timesteps_to_use.iter().enumerate() {
        // Set the first timestep and timestamp if it's not already set.
        match first_index {
            None => first_index = Some(i_timestep),
            Some(_) => (),
        }
        match first_timestamp {
            None => first_timestamp = Some(all_timestamps[timestep]),
            Some(_) => (),
        }

        // If the gap between this timestep and the first is too big...
        if i_timestep - first_index.unwrap() > time_average_factor - 1 {
            // ... put the current_timeblock in timeblocks along with
            // the average timestamp.
            timeblocks.push(Timeblock {
                index: timeblock_index,
                range: first_index.unwrap()..i_timestep,
                start: first_timestamp.unwrap(),
                end: *current_timestamps.last().unwrap(),
                average: average_epoch(&current_timestamps),
            });
            timeblock_index += 1;
            first_index = Some(i_timestep);
            first_timestamp = Some(all_timestamps[timestep]);
            current_indices.clear();
            current_timestamps.clear();
        }
        current_indices.push(i_timestep);
        current_timestamps.push(all_timestamps[timestep]);
    }
    // Push whatever is left in the last timeblock.
    if let Some(first_index) = first_index {
        timeblocks.push(Timeblock {
            index: timeblock_index,
            range: first_index..*current_indices.last().unwrap() + 1,
            start: first_timestamp.unwrap(),
            end: *current_timestamps.last().unwrap(),
            average: average_epoch(&current_timestamps),
        });
    }

    timeblocks
}

/// Returns a vector of unflagged chanblocks to use in calibration, as well as a
/// vector of flagged chanblock indices.
///
/// If the frequency resolution isn't given, then we guess based off of the
/// channel frequencies.
pub(super) fn channels_to_chanblocks(
    all_channel_freqs: &[f64],
    frequency_resolution: Option<f64>,
    freq_average_factor: usize,
    flagged_channels: &HashSet<usize>,
) -> (Vec<Chanblock>, Vec<usize>) {
    let freq_res = match frequency_resolution {
        Some(f) => f,
        None => {
            // Iterate over all the frequencies and find the smallest gap
            // between any pair.
            all_channel_freqs
                .iter()
                .skip(1)
                .zip(all_channel_freqs.iter())
                .fold(f64::INFINITY, |acc, (&f2, &f1)| {
                    let diff = f2 - f1;
                    acc.min(diff)
                })
        }
    };

    let biggest_freq_diff = freq_res * freq_average_factor as f64;
    let mut chanblocks = vec![];
    let mut flagged_chanblock_indices = vec![];
    let mut i_chanblock = 0;
    let mut i_unflagged_chanblock = 0;
    let mut current_freqs = vec![];
    let mut first_freq = None;
    let mut all_flagged = true;

    for (i_chan, &freq) in all_channel_freqs.iter().enumerate() {
        match first_freq {
            Some(_) => (),
            None => first_freq = Some(freq),
        }

        if freq - first_freq.unwrap() >= biggest_freq_diff {
            if all_flagged {
                flagged_chanblock_indices.push(i_chanblock);
            } else {
                chanblocks.push(Chanblock {
                    chanblock_index: i_chanblock,
                    unflagged_index: i_unflagged_chanblock,
                    freq: current_freqs.iter().sum::<f64>() / current_freqs.len() as f64,
                });
                i_unflagged_chanblock += 1;
            }
            current_freqs.clear();
            first_freq = Some(freq);
            all_flagged = true;
            i_chanblock += 1;
        }

        if !flagged_channels.contains(&i_chan) {
            all_flagged = false;
            current_freqs.push(freq);
        }
    }
    // Deal with any leftover data.
    if first_freq.is_some() {
        if all_flagged {
            flagged_chanblock_indices.push(i_chanblock);
        } else {
            chanblocks.push(Chanblock {
                chanblock_index: i_chanblock,
                unflagged_index: i_unflagged_chanblock,
                freq: current_freqs.iter().sum::<f64>() / current_freqs.len() as f64,
            })
        }
    }

    (chanblocks, flagged_chanblock_indices)
}

#[derive(Error, Debug)]
pub(super) enum AverageFactorError {
    #[error("The user input was 0; this is not permitted")]
    Zero,

    #[error("The user input has no units and isn't an integer; this is not permitted")]
    NotInteger,

    #[error("The user input isn't an integer multiple of the resolution: {out} vs {inp}")]
    NotIntegerMultiple { out: f64, inp: f64 },

    #[error("{0}")]
    Parse(#[from] crate::unit_parsing::UnitParseError),
}

/// Determine a time average factor given a time resolution and user input. Use
/// the default if the logic here is insufficient.
///
/// When averaging, the user input must be a multiple of the time resolution.
/// This function also checks that the user's input is sensible.
pub(super) fn parse_time_average_factor(
    time_resolution: Option<f64>,
    user_input_time_factor: Option<String>,
    default: usize,
) -> Result<usize, AverageFactorError> {
    match (
        time_resolution,
        user_input_time_factor.map(|f| parse_time(&f)),
    ) {
        (None, _) => {
            // If the time resolution is unknown, we assume it's because there's
            // only one timestep.
            Ok(1)
        }
        (_, None) => {
            // "None" indicates we should follow default behaviour.
            Ok(default)
        }
        // Propogate any errors encountered during parsing.
        (_, Some(Err(e))) => Err(AverageFactorError::Parse(e)),

        // User input is OK but has no unit.
        (_, Some(Ok((factor, TimeFormat::NoUnit)))) => {
            // Zero is not allowed.
            if factor < f64::EPSILON {
                return Err(AverageFactorError::Zero);
            }
            // Reject non-integer floats.
            if factor.fract().abs() > 1e-6 {
                return Err(AverageFactorError::NotInteger);
            }

            Ok(factor.round() as _)
        }

        // User input is OK and has a unit.
        (Some(time_res), Some(Ok((quantity, time_format)))) => {
            // Zero is not allowed.
            if quantity < f64::EPSILON {
                return Err(AverageFactorError::Zero);
            }

            // Scale the quantity by the unit, if required.
            let quantity = match time_format {
                TimeFormat::S => quantity,
                TimeFormat::Ms => 1000.0 * quantity,
                TimeFormat::NoUnit => unreachable!(),
            };
            let factor = quantity / time_res;
            // Reject non-integer floats.
            if factor.fract().abs() > 1e-6 {
                return Err(AverageFactorError::NotIntegerMultiple {
                    out: quantity,
                    inp: time_res,
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
    user_input_freq_factor: Option<String>,
    default: usize,
) -> Result<usize, AverageFactorError> {
    match (
        freq_resolution,
        user_input_freq_factor.map(|f| parse_freq(&f)),
    ) {
        (None, _) => {
            // If the freq. resolution is unknown, we assume it's because
            // there's only one channel.
            Ok(1)
        }
        (_, None) => {
            // "None" indicates we should follow default behaviour.
            Ok(default)
        }
        // Propogate any errors encountered during parsing.
        (_, Some(Err(e))) => Err(AverageFactorError::Parse(e)),

        // User input is OK but has no unit.
        (_, Some(Ok((factor, FreqFormat::NoUnit)))) => {
            // Zero is not allowed.
            if factor < f64::EPSILON {
                return Err(AverageFactorError::Zero);
            }
            // Reject non-integer floats.
            if factor.fract().abs() > 1e-6 {
                return Err(AverageFactorError::NotInteger);
            }

            Ok(factor.round() as _)
        }

        // User input is OK and has a unit.
        (Some(freq_res), Some(Ok((quantity, time_format)))) => {
            // Zero is not allowed.
            if quantity < f64::EPSILON {
                return Err(AverageFactorError::Zero);
            }

            // Scale the quantity by the unit, if required.
            let quantity = match time_format {
                FreqFormat::Hz => quantity,
                FreqFormat::kHz => 1000.0 * quantity,
                FreqFormat::NoUnit => unreachable!(),
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
