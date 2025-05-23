// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Metadata on an observation.

#[cfg(test)]
mod tests;

use std::{
    fmt::{Display, Write},
    num::NonZeroU16,
};

use hifitime::{Duration, Epoch};
use log::{debug, error, info, trace, warn};
use marlu::{LatLngHeight, RADec, XyzGeodetic};
use ndarray::Array2;
use vec1::Vec1;

use crate::{beam::Delays, io::read::VisInputType};

/// Currently supported polarisations.
#[derive(Clone, Copy, Debug, PartialEq)]
#[allow(non_camel_case_types)]
pub enum Polarisations {
    XX_XY_YX_YY,
    XX,
    YY,
    XX_YY,
    XX_YY_XY,
}

impl Default for Polarisations {
    fn default() -> Self {
        Self::XX_XY_YX_YY
    }
}

impl Display for Polarisations {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}",
            match self {
                Polarisations::XX_XY_YX_YY => "XX XY YX YY",
                Polarisations::XX => "XX",
                Polarisations::YY => "YY",
                Polarisations::XX_YY => "XX YY",
                Polarisations::XX_YY_XY => "XX YY XY",
            }
        )
    }
}

impl Polarisations {
    pub(crate) fn num_pols(self) -> u8 {
        match self {
            Polarisations::XX_XY_YX_YY => 4,
            Polarisations::XX => 1,
            Polarisations::YY => 1,
            Polarisations::XX_YY => 2,
            Polarisations::XX_YY_XY => 3,
        }
    }
}

/// MWA observation metadata.
///
/// This can be thought of the state and contents of the input data. It may not
/// reflect its raw, non-preprocessed state, but this is all we can say about
/// it.
///
/// Tile information is ordered according to the "Antenna" column in HDU 1 of
/// the observation's metafits file.
pub(crate) struct ObsContext {
    /// The format of the file containing the visibilities (e.g. uvfits).
    pub(crate) input_data_type: VisInputType,

    /// The observation ID, which is also the observation's scheduled start GPS
    /// time (but shouldn't be used for this purpose).
    pub(crate) obsid: Option<u32>,

    /// The unique timestamps in the observation. These are stored as `hifitime`
    /// [Epoch] structs to help keep the code flexible. These include timestamps
    /// that are deemed "flagged" by the observation.
    pub(crate) timestamps: Vec1<Epoch>,

    /// The *available* timestep indices of the input data. This does not
    /// necessarily start at 0, and is not necessarily regular (e.g. a valid
    /// vector could be [1, 2, 4]).
    ///
    /// Allowing the indices to be non-regular means that we can represent input
    /// data that also isn't regular; naively reading in a dataset with 2
    /// timesteps that are separated by more than the time resolution of the
    /// data would give misleading results.
    pub(crate) all_timesteps: Vec1<usize>,

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

    /// The Earth position of the instrumental array *that is used*.
    /// Unfortunately this needs to be distinguished from the supplied array
    /// position because (1) there may not be an array position detailed in the
    /// input files and (2) it is needed to convert tile positions, and so the
    /// `tile_xyzs` described here depend on this value.
    pub(crate) array_position: LatLngHeight,

    /// The Earth position of the instrumental array *that is described by the
    /// input data*. It is provided here to distinguish from `array_position`,
    /// which may be different.
    pub(crate) supplied_array_position: LatLngHeight,

    /// The difference between UT1 and UTC. If this is 0 seconds, then LSTs are
    /// wrong by up to 0.9 seconds. The code will assume that 0 seconds means
    /// that DUT1 wasn't provided and may warn the user.
    ///
    /// This is *probably* defined off of the obsid, but we don't expect DUT1 to
    /// change significantly across the course of an observation.
    pub(crate) dut1: Option<Duration>,

    /// The names of each of the tiles in the input data. This includes flagged
    /// and unavailable tiles.
    pub(crate) tile_names: Vec1<String>,

    /// The [`XyzGeodetic`] coordinates of all tiles in the array (all
    /// coordinates are specified in \[metres\]). This includes flagged and
    /// unavailable tiles. The values described here may be affected by a
    /// user-supplied `array_position`.
    pub(crate) tile_xyzs: Vec1<XyzGeodetic>,

    /// The flagged tiles, i.e. what the observation data suggests to be
    /// flagged. Generally `hyperdrive` will discourage using flagged data, but
    /// the user can still access them if desired. Zero indexed.
    pub(crate) flagged_tiles: Vec<usize>,

    /// Tiles that are described by the observation data in some capacity, but
    /// their data is not available, regardless of what the user wants. Zero
    /// indexed.
    ///
    /// Why is this distinguished from flagged tiles? An observation with 128
    /// tiles *should* have solutions for 128 tiles. It would look weird if, for
    /// example, one tile was not included in the input data and the output data
    /// described 127 tiles.
    pub(crate) unavailable_tiles: Vec<usize>,

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

    /// The MWA receiver channel numbers (all in the range 0 to 255, e.g. 131 to
    /// 154 for EoR high-band) that are present in the supplied data. The vector
    /// is ascendingly sorted. These do not necessarily match the coarse channel
    /// numbers present in the full observation, as the input data may only
    /// reflect a fraction of it.
    pub(crate) mwa_coarse_chan_nums: Option<Vec1<u32>>,

    /// The number of fine-frequency channels per coarse channel. For 40 kHz
    /// legacy MWA data, this is 32.
    pub(crate) num_fine_chans_per_coarse_chan: Option<NonZeroU16>,

    /// The fine-channel resolution of the supplied data \[Hz\]. This is not
    /// necessarily the fine-channel resolution of the original observation's
    /// data; this data may have applied averaging to the original observation.
    pub(crate) freq_res: Option<f64>,

    /// All of the fine-channel frequencies within the data \[Hz\]. The values
    /// reflect the frequencies at the *centre* of each channel.
    ///
    /// These are kept as ints to help some otherwise error-prone calculations
    /// using floats. By using ints, we assume there is no sub-Hz structure.
    pub(crate) fine_chan_freqs: Vec1<u64>,

    /// The flagged fine channels for each baseline in the supplied data. Zero
    /// indexed.
    pub(crate) flagged_fine_chans: Vec<u16>,

    /// The fine channels per coarse channel already flagged in the supplied
    /// data. Zero indexed.
    pub(crate) flagged_fine_chans_per_coarse_chan: Option<Vec1<u16>>,

    /// The polarisations included in the data. Any combinations not listed are
    /// not supported.
    pub(crate) polarisations: Polarisations,
}

impl ObsContext {
    /// Get the total number of tiles in the observation, i.e. flagged and
    /// unflagged.
    pub(crate) fn get_total_num_tiles(&self) -> usize {
        self.tile_xyzs.len()
    }

    /// Return all frequencies within the fine frequency channel range that are
    /// multiples of 1.28 MHz.
    pub(crate) fn get_veto_freqs(&self) -> Vec<f64> {
        let mut veto_freqs = self.fine_chan_freqs.clone();
        veto_freqs.sort_unstable();
        let mut veto_freqs: Vec<f64> = veto_freqs
            .into_iter()
            .map(|f| (f as f64 / 1.28e6).round() * 1.28e6)
            .collect();
        veto_freqs.dedup();
        veto_freqs
    }

    /// Print information on the indices, names and statuses of all of the tiles
    /// in this observation at the indicated log level.
    pub(crate) fn print_tile_statuses(&self, level: log::Level) {
        let s = "All tile indices, names and default statuses:";
        match level {
            log::Level::Error => error!("{}", s),
            log::Level::Warn => warn!("{}", s),
            log::Level::Info => info!("{}", s),
            log::Level::Debug => debug!("{}", s),
            log::Level::Trace => trace!("{}", s),
        }

        let mut s = String::new();
        self.tile_names.iter().enumerate().for_each(|(i, name)| {
            let msg = match (
                self.unavailable_tiles.contains(&i),
                self.flagged_tiles.contains(&i),
            ) {
                (true, _) => "unavailable",
                (false, false) => "  unflagged",
                (false, true) => "    flagged",
            };

            write!(&mut s, "    {i:3}: {name:10}: {msg}").unwrap();
            match level {
                log::Level::Error => error!("{}", s),
                log::Level::Warn => warn!("{}", s),
                log::Level::Info => info!("{}", s),
                log::Level::Debug => debug!("{}", s),
                log::Level::Trace => trace!("{}", s),
            }
            s.clear();
        });
    }
}
