// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Metadata on an observation.

#[cfg(test)]
mod tests;

use std::collections::HashSet;

use hifitime::{Duration, Epoch, Unit};
use itertools::Itertools;
use log::{info, warn};
use marlu::{
    constants::{FREQ_WEIGHT_FACTOR, TIME_WEIGHT_FACTOR},
    LatLngHeight, RADec, XyzGeodetic,
};
use ndarray::Array2;
use thiserror::Error;
use vec1::Vec1;

use mwa_hyperdrive_beam::Delays;
use mwa_hyperdrive_common::{hifitime, itertools, log, marlu, ndarray, thiserror, vec1};

/// MWA observation metadata.
///
/// This can be thought of the state and contents of the input data. It may not
/// reflect its raw, non-preprocessed state, but this is all we can say about
/// it.
///
/// Tile information is ordered according to the "Antenna" column in HDU 1 of
/// the observation's metafits file.
pub struct ObsContext {
    /// The observation ID, which is also the observation's scheduled start GPS
    /// time (but shouldn't be used for this purpose).
    pub obsid: Option<u32>,

    /// The unique timestamps in the observation. These are stored as `hifitime`
    /// [Epoch] structs to help keep the code flexible. These include timestamps
    /// that are deemed "flagged" by the observation.
    pub timestamps: Vec1<Epoch>,

    /// The *available* timestep indices of the input data. This does not
    /// necessarily start at 0, and is not necessarily regular (e.g. a valid
    /// vector could be [1, 2, 4]).
    ///
    /// Allowing the indices to be non-regular means that we can represent input
    /// data that also isn't regular; naively reading in a dataset with 2
    /// timesteps that are separated by more than the time resolution of the
    /// data would give misleading results.
    pub all_timesteps: Vec1<usize>,

    /// The timestep indices of the input data that aren't totally flagged.
    ///
    /// This is allowed to be empty.
    pub(crate) unflagged_timesteps: Vec<usize>,

    /// The observation phase centre.
    pub(super) phase_centre: RADec,

    /// The observation pointing centre.
    ///
    /// This is typically not used, but if available is nice to report.
    pub(super) pointing_centre: Option<RADec>,

    /// The Earth position of the instrumental array.
    pub(crate) array_position: Option<LatLngHeight>,

    /// The difference between UT1 and UTC. If this is 0 seconds, then LSTs are
    /// wrong by up to 0.9 seconds. The code will assume that 0 seconds means
    /// that DUT1 wasn't provided and may warn the user.
    ///
    /// This is *probably* defined off of the obsid, but we don't expect DUT1 to
    /// change significantly across the course of an observation.
    pub(crate) dut1: Option<Duration>,

    /// The names of each of the tiles used in the array.
    pub(crate) tile_names: Vec1<String>,

    /// The [XyzGeodetic] coordinates of all tiles in the array (all coordinates
    /// are specified in \[metres\]). This includes tiles that have been flagged
    /// in the input data.
    pub tile_xyzs: Vec1<XyzGeodetic>,

    /// The flagged tiles, either already missing data or suggested to be
    /// flagged. Zero indexed.
    pub flagged_tiles: Vec<usize>,

    /// Are auto-correlations present in the visibility data?
    pub(crate) autocorrelations_present: bool,

    /// The dipole delays for each tile in the array. They are necessary for
    /// anything requiring beam responses. Not all input data specify the
    /// delays, but it still possible to continue if the delays are supplied
    /// another way (e.g. user-specified delays).
    pub(crate) dipole_delays: Option<Delays>,

    /// The dipole gains for each tile in the array. The first axis is unflagged
    /// antenna, the second dipole index. These will typically all be of value
    /// 1.0, except where a dipole is dead (0.0). If this is `None`, then it is
    /// assumed that all tiles are live.
    pub(crate) dipole_gains: Option<Array2<f64>>,

    /// The time resolution of the supplied data. This is not necessarily the
    /// native time resolution of the original observation's data, as it may
    /// have already been averaged. This is kept optional in case in the input
    /// data doesn't report the resolution and has only one timestep, and
    /// therefore no resolution.
    pub(crate) time_res: Option<Duration>,

    /// The coarse channel numbers (typically 1 to 24) that are present in the
    /// supplied data. This does not necessarily match the coarse channel
    /// numbers present in the full observation, as the input data may only
    /// reflect a fraction of it.
    pub(crate) coarse_chan_nums: Vec<u32>,

    /// The centre frequencies of each of the coarse channels in this
    /// observation \[Hz\].
    pub(crate) coarse_chan_freqs: Vec<f64>,

    /// The number of fine-frequency channels per coarse channel. For 40 kHz
    /// legacy MWA data, this is 32.
    pub(crate) num_fine_chans_per_coarse_chan: usize,

    /// The fine-channel resolution of the supplied data \[Hz\]. This is not
    /// necessarily the fine-channel resolution of the original observation's
    /// data; this data may have applied averaging to the original observation.
    pub(crate) freq_res: Option<f64>,

    /// All of the fine-channel frequencies within the data \[Hz\]. The values
    /// reflect the frequencies at the *centre* of each channel.
    ///
    /// These are kept as ints to help some otherwise error-prone calculations
    /// using floats. By using ints, we assume there is no sub-Hz structure.
    pub fine_chan_freqs: Vec1<u64>,

    /// The flagged fine channels for each baseline in the supplied data. Zero
    /// indexed.
    pub flagged_fine_chans: Vec<usize>,

    /// The fine channels per coarse channel already flagged in the supplied
    /// data. Zero indexed.
    pub(crate) flagged_fine_chans_per_coarse_chan: Vec<usize>,
}

impl ObsContext {
    /// Get the total number of tiles in the observation, i.e. flagged and
    /// unflagged.
    pub(crate) fn get_total_num_tiles(&self) -> usize {
        self.tile_xyzs.len()
    }

    /// Attempt to get time resolution using heuristics if it is not present.
    ///
    /// If `time_res` is `None`, then attempt to determine it from the minimum
    /// distance between timestamps. If there is no more than 1 timestamp, then
    /// return 1s, since the time resolution of single-timestep observations is
    /// not important anyway.
    pub fn guess_time_res(&self) -> Duration {
        match self.time_res {
            Some(t) => t,
            None => {
                warn!("No integration time specified; assuming {TIME_WEIGHT_FACTOR} second");
                Duration::from_f64(TIME_WEIGHT_FACTOR, Unit::Second)
            }
        }
    }

    pub fn guess_freq_res(&self) -> f64 {
        match self.freq_res {
            Some(f) => f,
            None => {
                warn!(
                    "No frequency resolution specified; assuming {} kHz",
                    FREQ_WEIGHT_FACTOR / 1e3
                );
                FREQ_WEIGHT_FACTOR
            }
        }
    }

    /// Given whether to use the [ObsContext]'s tile flags and additional tile
    /// flags (as strings or indices), return de-duplicated and sorted tile flag
    /// indices.
    pub(crate) fn get_tile_flags(
        &self,
        ignore_input_data_tile_flags: bool,
        additional_flags: Option<&[String]>,
    ) -> Result<Vec<usize>, InvalidTileFlag> {
        let mut flagged_tiles = HashSet::new();

        if !ignore_input_data_tile_flags {
            // Add tiles that have already been flagged by the input data.
            for &obs_tile_flag in &self.flagged_tiles {
                flagged_tiles.insert(obs_tile_flag);
            }
        }

        if let Some(flag_strings) = additional_flags {
            // We need to convert the strings into antenna indices. The strings
            // are either indices themselves or antenna names.
            for flag_string in flag_strings {
                // Try to parse a naked number.
                let result = match flag_string.trim().parse().ok() {
                    Some(n) => {
                        let total_num_tiles = self.get_total_num_tiles();
                        if n >= total_num_tiles {
                            Err(InvalidTileFlag::Index {
                                got: n,
                                max: total_num_tiles - 1,
                            })
                        } else {
                            flagged_tiles.insert(n);
                            Ok(())
                        }
                    }
                    None => {
                        // Check if this is an antenna name.
                        match self
                            .tile_names
                            .iter()
                            .enumerate()
                            .find(|(_, name)| name.to_lowercase() == flag_string.to_lowercase())
                        {
                            // If there are no matches, complain that the user input
                            // is no good.
                            None => Err(InvalidTileFlag::BadTileFlag(flag_string.to_string())),
                            Some((i, _)) => {
                                flagged_tiles.insert(i);
                                Ok(())
                            }
                        }
                    }
                };
                if result.is_err() {
                    // If there's a problem, show all the tile names and their
                    // indices to help out the user.
                    info!("All tile indices and names:");
                    self.tile_names.iter().enumerate().for_each(|(i, name)| {
                        info!("    {:3}: {:10}", i, name);
                    });
                    // Propagate the error.
                    result?;
                }
            }
        }

        // Convert the set to a vector.
        Ok(flagged_tiles.into_iter().sorted().collect())
    }
}

#[derive(Error, Debug)]
pub(crate) enum InvalidTileFlag {
    #[error("Got a tile flag {got}, but the biggest possible antenna index is {max}")]
    Index { got: usize, max: usize },

    #[error("Bad flag value: '{0}' is neither an integer or an available antenna name. Run with extra verbosity to see all tile names.")]
    BadTileFlag(String),
}
