// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

/*!
Metadata on an observation.
 */

use std::f64::consts::TAU;
use std::ops::Range;

use ndarray::Array2;

use mwa_hyperdrive_core::{
    constants::{DS2R, SOLAR2SIDEREAL},
    RADec, XyzBaseline, XYZ,
};

/// Observation metadata.
///
/// This can be thought of as a substitute of mwalib context structs; mwalib
/// can't be used all the time (e.g. doesn't interface with measurement sets),
/// so this struct can be used as a common interface to some radio data.
///
/// Frequency information is deliberately kept aside in an effort to keep
/// complexity down; use [FreqContext] for that.
pub(crate) struct ObsContext {
    /// The observation ID, which is also the observation's scheduled start GPS
    /// time.
    pub(crate) obsid: Option<u32>,

    /// The unique timesteps in the observation. These are stored as
    /// `hifitime::Epoch` structs to help keep the code flexible.
    // TODO: Make these centroids.
    pub(crate) timesteps: Vec<hifitime::Epoch>,

    /// The timestep indices of the input data that aren't totally flagged
    /// (exclusive).
    pub(crate) timestep_indices: Range<usize>,

    /// The Local Sidereal Time in the __middle__ of the first timestep
    /// \[radians\].
    pub(crate) lst0: f64,

    /// The time resolution of the supplied data \[seconds\]. This is not
    /// necessarily the native time resolution of the original observation's
    /// data.
    pub(crate) time_res: f64,

    /// The observation phase centre.
    pub(super) pointing: RADec,

    /// The ideal dipole delays of each MWA tile in the observation (i.e. no
    /// values of 32).
    pub(crate) delays: Vec<u32>,

    /// The geocentric `XYZ` coordinates of all tiles in the array \[metres\].
    /// This includes tiles that have been flagged in the input data.
    pub(crate) tile_xyz: Vec<XYZ>,

    /// The `XyzBaselines` of the observations \[metres\]. This does not change
    /// over time; it is determined only by the telescope's tile layout.
    pub(crate) baseline_xyz: Vec<XyzBaseline>,

    /// The tiles already flagged in the supplied data. These values correspond
    /// to those from the "Antenna" column in HDU 1 of the metafits file. Zero
    /// indexed.
    pub(crate) tile_flags: Vec<usize>,

    /// The fine channels per coarse channel already flagged in the supplied
    /// data. Zero indexed.
    pub(crate) fine_chan_flags_per_coarse_band: Vec<usize>,

    /// The number of unflagged tiles in the input data.
    pub(crate) num_unflagged_tiles: usize,

    /// The number of unflagged cross-correlation baselines. e.g. if there are
    /// 128 unflagged tiles, then this is 8128.
    pub(crate) num_unflagged_baselines: usize,

    /// The dipole gains for each tile in the array. The first axis is unflagged
    /// antenna, the second dipole index. These will typically all be of value
    /// 1.0, except where a dipole is dead (0.0).
    pub(crate) dipole_gains: Array2<f64>,
}

impl ObsContext {
    pub(crate) fn lst_from_timestep(&self, timestep: usize) -> f64 {
        (self.lst0 + SOLAR2SIDEREAL * DS2R * timestep as f64 * self.time_res) % TAU
    }
}

/// Metadata on an observation's frequency setup.
pub(crate) struct FreqContext {
    /// The coarse band numbers (typically 1 to 24) that are present in the
    /// supplied data. This does not necessarily match the coarse band numbers
    /// present in the full observation, as the input data may only reflect a
    /// fraction of it.
    pub(crate) coarse_chan_nums: Vec<u32>,

    /// The centre frequencies of each of the coarse channels in this
    /// observation \[Hz\].
    pub(crate) coarse_chan_freqs: Vec<f64>,

    /// a.k.a coarse channel bandwidth \[Hz\].
    pub(crate) coarse_chan_width: f64,

    /// The bandwidth of the supplied data \[Hz\]. This is not necessarily the
    /// bandwidth of the full observation, as the input data may only reflect a
    /// fraction of it.
    pub(crate) total_bandwidth: f64,

    /// The fine-channel number range (exclusive). e.g. if a legacy MWA
    /// observation has 40 kHz fine-channel resolution, then this should be
    /// 0..768.
    pub(crate) fine_chan_range: Range<usize>,

    /// All of the fine-channel frequencies within the data \[Hz\].
    // TODO: Do the frequencies list the start edge or middle frequencies of these channels?
    pub(crate) fine_chan_freqs: Vec<f64>,

    /// The number of fine-frequency channels per coarse band. For 40 kHz legacy
    /// MWA data, this is 32.
    pub(crate) num_fine_chans_per_coarse_chan: usize,

    /// The fine-channel resolution of the supplied data \[Hz\]. This is not
    /// necessarily the fine-channel resolution of the original observation's
    /// data.
    pub(crate) native_fine_chan_width: f64,
}
