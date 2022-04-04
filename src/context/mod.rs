// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Metadata on an observation.

use hifitime::Epoch;
use marlu::{LatLngHeight, RADec, XyzGeodetic};
use ndarray::Array2;
use vec1::Vec1;

use mwa_hyperdrive_common::{hifitime, marlu, ndarray, vec1};

/// MWA observation metadata.
///
/// This can be thought of the state and contents of the input data. It may not
/// reflect its raw, non-preprocessed state, but this is all we can say about
/// it.
///
/// Tile information is ordered according to the "Antenna" column in HDU 1 of
/// the observation's metafits file.
pub(crate) struct ObsContext {
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
    /// Allowing the indices to non-regular means that we can represent input
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

    /// The names of each of the tiles used in the array.
    pub(crate) tile_names: Vec1<String>,

    /// The [XyzGeodetic] coordinates of all tiles in the array (all coordinates
    /// are specified in \[metres\]). This includes tiles that have been flagged
    /// in the input data.
    pub(crate) tile_xyzs: Vec1<XyzGeodetic>,

    /// The flagged tiles, either already missing data or suggested to be
    /// flagged. Zero indexed.
    pub(crate) flagged_tiles: Vec<usize>,

    /// Are auto-correlations present in the visibility data?
    pub(crate) autocorrelations_present: bool,

    /// The dipole gains for each tile in the array. The first axis is unflagged
    /// antenna, the second dipole index. These will typically all be of value
    /// 1.0, except where a dipole is dead (0.0). If this is `None`, then it is
    /// assumed that all tiles are live.
    pub(crate) dipole_gains: Option<Array2<f64>>,

    /// The time resolution of the supplied data \[seconds\]. This is not
    /// necessarily the native time resolution of the original observation's
    /// data, as it may have already been averaged. This is kept optional in
    /// case in the input data doesn't report the resolution and has only one
    /// timestep, and therefore no resolution.
    pub(crate) time_res: Option<f64>,

    /// The Earth position of the instrumental array.
    pub(crate) array_position: Option<LatLngHeight>,

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
    pub(crate) fine_chan_freqs: Vec1<u64>,

    /// The flagged fine channels for each baseline in the supplied data. Zero
    /// indexed.
    pub(crate) flagged_fine_chans: Vec<usize>,

    /// The fine channels per coarse channel already flagged in the supplied
    /// data. Zero indexed.
    pub(crate) flagged_fine_chans_per_coarse_chan: Vec<usize>,
}
