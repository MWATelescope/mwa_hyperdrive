// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

use mwa_hyperdrive_core::{RADec, XyzBaseline, XYZ};

/// Observation metadata. This acts as a substitute of mwalib context structs,
/// when mwalib can't be used.
pub(crate) struct Context {
    /// The observation ID, which is also the observation's scheduled start GPS
    /// time.
    pub(crate) obsid: u32,

    /// The unique timesteps in the observation. These are stored as
    /// `hifitime::Epoch` structs to help keep the code flexible.
    pub(crate) timesteps: Vec<hifitime::Epoch>,

    /// The time resolution of the supplied data [seconds]. This is not
    /// necessarily the native time resolution of the original observation's
    /// data.
    pub(crate) native_time_res: f64,

    /// The phase centre at the start of the observation.
    pub(super) pointing: RADec,

    /// The ideal dipole delays of each MWA tile in the observation (i.e. no values of 32).
    pub(crate) delays: Vec<u32>,

    /// The geocentric `XYZ` coordinates of all tiles in the array. This
    /// includes tiles that have been flagged in the input data.
    pub(crate) tile_xyz: Vec<XYZ>,

    /// The `XyzBaselines` of the observations [metres]. This does not change
    /// over time; it is determined only by the telescope's tile layout.
    pub(crate) baseline_xyz: Vec<XyzBaseline>,

    /// The tiles already flagged in the supplied data. These indices are
    /// antenna numbers.
    pub(crate) tile_flags: Vec<usize>,

    /// The fine channels per coarse channel already flagged in the supplied
    /// data. Zero indexed.
    pub(crate) fine_chan_flags: Vec<usize>,

    /// Frequency metadata.
    pub(crate) freq_context: FreqContext,
}

impl<'a> Context {
    pub(crate) fn new(
        obsid: u32,
        timesteps: Vec<hifitime::Epoch>,
        native_time_res_sec: f64,
        pointing: RADec,
        delays: Vec<u32>,
        tile_xyz: Vec<XYZ>,
        tile_flags: Vec<usize>,
        fine_chan_flags: Vec<usize>,
        coarse_chan_nums: Vec<u32>,
        total_bandwidth_hz: f64,
        fine_chan_freqs_hz: Vec<f64>,
    ) -> Self {
        let coarse_chan_width = total_bandwidth_hz / coarse_chan_nums.len() as f64;
        // TODO: Check that the length is enough.
        let native_fine_chan_width = fine_chan_freqs_hz[1] - fine_chan_freqs_hz[0];
        let num_fine_chans_per_coarse_chan =
            (total_bandwidth_hz / coarse_chan_nums.len() as f64 / native_fine_chan_width).round()
                as _;
        let coarse_chan_freqs: Vec<f64> = fine_chan_freqs_hz
            .chunks_exact(num_fine_chans_per_coarse_chan)
            // round is OK because these values are Hz, and we're not ever
            // looking at sub-Hz resolution.
            .map(|chunk| chunk[chunk.len() / 2].round())
            .collect();
        let freq_context = FreqContext {
            coarse_chan_nums,
            coarse_chan_freqs,
            coarse_chan_width,
            total_bandwidth: total_bandwidth_hz,
            fine_chan_freqs: fine_chan_freqs_hz,
            num_fine_chans_per_coarse_chan,
            native_fine_chan_width,
        };
        let baseline_xyz = XYZ::get_baselines(&tile_xyz);
        Self {
            obsid,
            timesteps,
            native_time_res: native_time_res_sec,
            pointing,
            delays,
            tile_xyz,
            baseline_xyz,
            tile_flags,
            fine_chan_flags,
            freq_context,
        }
    }

    pub(crate) fn get_obsid(&self) -> u32 {
        self.obsid
    }

    pub(crate) fn get_timesteps(&self) -> &[hifitime::Epoch] {
        &self.timesteps
    }

    pub(crate) fn get_native_time_res(&self) -> f64 {
        self.native_time_res
    }

    pub(crate) fn get_pointing(&self) -> &RADec {
        &self.pointing
    }

    pub(crate) fn get_freq_context(&self) -> &FreqContext {
        &self.freq_context
    }

    pub(crate) fn get_delays(&self) -> &[u32] {
        &self.delays
    }

    pub(crate) fn get_tile_xyz(&self) -> &[XYZ] {
        &self.tile_xyz
    }

    pub(crate) fn get_baseline_xyz(&self) -> &[XyzBaseline] {
        &self.baseline_xyz
    }

    pub(crate) fn get_tile_flags(&self) -> &[usize] {
        &self.tile_flags
    }

    pub(crate) fn get_fine_chan_flags(&self) -> &[usize] {
        &self.fine_chan_flags
    }
}

pub(crate) struct FreqContext {
    /// The coarse band numbers (typically 1 to 24) that are present in the
    /// supplied data. This does not necessarily match the coarse band numbers
    /// present in the full observation.
    pub(crate) coarse_chan_nums: Vec<u32>,

    /// [Hz]
    pub(crate) coarse_chan_freqs: Vec<f64>,

    /// [Hz]
    pub(crate) coarse_chan_width: f64,

    /// The bandwidth of the supplied data [Hz]. This is not necessarily the
    /// bandwidth of the full observation.
    pub(crate) total_bandwidth: f64,

    /// All of the fine-channel frequencies within the data.
    // TODO: Do the frequencies list the start edge or middle frequencies of these channels?
    pub(crate) fine_chan_freqs: Vec<f64>,

    pub(crate) num_fine_chans_per_coarse_chan: usize,

    /// The fine-channel resolution of the supplied data [Hz]. This is not
    /// necessarily the fine-channel resolution of the original observation's
    /// data.
    pub(crate) native_fine_chan_width: f64,
}
