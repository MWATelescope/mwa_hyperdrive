// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Metadata on an observation.

use std::f64::consts::TAU;
use std::ops::Range;

use hifitime::Epoch;
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

    /// The unique timesteps in the observation. These are stored as `hifitime`
    /// [Epoch] structs to help keep the code flexible. These include flagged
    /// timesteps.
    pub(crate) timesteps: Vec<Epoch>,

    /// The timestep indices of the input data that aren't totally flagged
    /// (exclusive).
    pub(crate) unflagged_timestep_indices: Range<usize>,

    // /// The Local Mean Sidereal Time in the __middle__ of the first timestep
    // /// (even if its flagged) \[radians\].
    // pub(crate) lst0: f64,
    /// The observation phase centre.
    pub(super) phase_centre: RADec,

    /// The observation pointing centre.
    ///
    /// This is typically not used, but if available is nice to report.
    pub(super) pointing_centre: Option<RADec>,

    /// The names of each of the tiles used in the array.
    pub(crate) names: Vec<String>,

    /// The geocentric [XYZ] coordinates of all tiles in the array \[metres\].
    /// This includes tiles that have been flagged in the input data.
    pub(crate) tile_xyz: Vec<XYZ>,

    /// The tiles already flagged in the supplied data. These values correspond
    /// to those from the "Antenna" column in HDU 1 of the metafits file. Zero
    /// indexed.
    pub(crate) tile_flags: Vec<usize>,

    /// The fine channels per coarse channel already flagged in the supplied
    /// data. Zero indexed.
    pub(crate) fine_chan_flags_per_coarse_chan: Vec<usize>,

    /// The number of unflagged tiles in the input data.
    pub(crate) num_unflagged_tiles: usize,

    /// The number of unflagged cross-correlation baselines. e.g. if there are
    /// 128 unflagged tiles, then this is 8128.
    pub(crate) num_unflagged_baselines: usize,

    /// The dipole gains for each tile in the array. The first axis is unflagged
    /// antenna, the second dipole index. These will typically all be of value
    /// 1.0, except where a dipole is dead (0.0).
    pub(crate) dipole_gains: Array2<f64>,

    /// The time resolution of the supplied data \[seconds\]. This is not
    /// necessarily the native time resolution of the original observation's
    /// data. This is kept optional in case in the input data has only one time
    /// step, and therefore no resolution.
    pub(crate) time_res: Option<f64>,

    /// The Earth longitude of the instrumental array \[radians\].
    pub(crate) array_longitude_rad: Option<f64>,

    /// The Earth latitude of the instrumental array \[radians\].
    pub(crate) array_latitude_rad: Option<f64>,
}

// impl ObsContext {
//     pub(crate) fn lst_from_timestep(&self, timestep: usize) -> f64 {
//         if let Some(tr) = self.time_res {
//             (self.lst0 + SOLAR2SIDEREAL * DS2R * timestep as f64 * tr) % TAU
//         } else {
//             // No time resolution implies no timesteps; just return the first
//             // LST.
//             self.lst0
//         }
//     }
// }

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

#[cfg(test)]
mod tests {
    use crate::constants::HIFITIME_GPS_FACTOR;
    use crate::tests::{reduced_obsids::*, *};

    // // astropy doesn't exactly agree with the numbers below, I think because the
    // // LST listed in MWA metafits files doesn't agree with what astropy thinks
    // // it should be. But, it's all very close.
    // #[test]
    // fn test_lst_from_timestep_native() {
    //     // Obsid 1090008640 actually starts at GPS time 1090008641.
    //     let args = get_1090008640();
    //     let params = args.into_params().unwrap();
    //     let obs_context = params.input_data.get_obs_context();
    //     // gpstime 1090008642
    //     assert_abs_diff_eq!(
    //         obs_context.lst_from_timestep(0),
    //         6.262123573853594,
    //         epsilon = 1e-10
    //     );

    //     // gpstime 1090008644
    //     assert_abs_diff_eq!(
    //         obs_context.lst_from_timestep(1),
    //         6.262269416170651,
    //         epsilon = 1e-10
    //     );
    // }

    // #[test]
    // // Unlike the test above, decrease the time resolution by averaging.
    // fn test_lst_from_timestep_averaged() {
    //     let args = get_1090008640();
    // args.time_res = ...
    // let params = args.into_params().unwrap();
    //     let context = match CorrelatorContext::new(&args.metafits.unwrap(), &args.gpuboxes.unwrap())
    //     {
    //         Ok(c) => c,
    //         Err(e) => panic!("{}", e),
    //     };
    //     // The native time res. is 2.0s, let's make our target 4.0s here.
    //     let time_res = 4.0;
    //     let new_lst = lst_from_timestep(0, &context, time_res);
    //     // gpstime 1090008643
    //     assert_abs_diff_eq!(new_lst, 6.2621966114770915, epsilon = 1e-10);

    //     let new_lst = lst_from_timestep(1, &context, time_res);
    //     // gpstime 1090008647
    //     assert_abs_diff_eq!(new_lst, 6.262488296111205, epsilon = 1e-10);
    // }

    #[test]
    fn hifitime_behaves_as_expected() {
        let gps = 1065880128.0;
        // https://en.wikipedia.org/wiki/Global_Positioning_System#Timekeeping
        // The difference between GPS and TAI time is always 19s, but hifitime
        // wants the number of TAI seconds since 1900. GPS time starts at 1980
        // Jan 5.
        let tai = gps + 19.0 + HIFITIME_GPS_FACTOR;
        let epoch = hifitime::Epoch::from_tai_seconds(tai);
        assert_abs_diff_eq!(epoch.as_gpst_seconds() - HIFITIME_GPS_FACTOR, 1065880128.0);
    }
}
