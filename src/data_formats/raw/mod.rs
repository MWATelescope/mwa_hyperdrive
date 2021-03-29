// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

/*!
Code to handle reading from raw MWA files.
 */

pub mod error;

pub use error::*;
use hifitime::Epoch;

use std::collections::HashSet;
use std::path::{Path, PathBuf};

use log::{debug, warn};
use ndarray::prelude::*;

use super::*;
use crate::flagging::aoflagger::AOFlags;
use crate::glob::*;
use crate::mwalib::{CorrelatorContext, Pol};
use mwa_hyperdrive_core::{c32, RADec, XyzBaseline, XYZ};

use super::InputData;
use crate::context::FreqContext;

/// Raw MWA data, i.e. gpubox files.
pub(crate) struct RawData {
    /// The interface to the raw data via mwalib.
    context: CorrelatorContext,

    // TODO: Rename to something more general. Don't need actual AOFlagger flags.
    /// AOFlagger flags.
    ///
    /// These aren't necessarily derived from mwaf files; if the user did
    /// not supply flags, then `aoflags` may contain no flags, or just
    /// default channel flags (the edges and centre channel, for example).
    aoflags: Option<AOFlags>,

    timesteps: Vec<hifitime::Epoch>,

    /// The phase centre at the start of the observation.
    pointing: RADec,

    /// The geocentric `XYZ` coordinates of all tiles in the array. This
    /// includes tiles that have been flagged in the input data.
    tile_xyz: Vec<XYZ>,

    /// The `XyzBaselines` of the observations [metres]. This does not change
    /// over time; it is determined only by the telescope's tile layout.
    baseline_xyz: Vec<XyzBaseline>,

    /// The dipole delays listed in the header of the observation's metafits
    /// file. If these are all 32, then this indicates possible problems with
    /// the observation.
    listed_delays: Vec<u32>,

    /// The ideal dipole delays of each MWA tile in the observation (i.e. no values of 32).
    ideal_delays: Vec<u32>,

    /// The tiles already flagged in the supplied data. These values correspond
    /// to those from the "Antenna" column in HDU 1 of the metafits file. Zero
    /// indexed.
    tile_flags: Vec<usize>,

    /// Which fine-frequency channels per coarse band are flagged? This is empty
    /// only if the user used `ignore_metafits_tile_flags` when creating this
    /// instance of `RawData`. Otherwise, 80 kHz is flagged at the edges as well
    /// as the centre channel. Zero indexed.
    fine_chan_flags: Vec<usize>,

    freq_context: super::FreqContext,
}

impl RawData {
    /// Create a new instance of the `InputData` trait with raw MWA data.
    pub(crate) fn new<T: AsRef<Path>>(
        metadata: &T,
        gpuboxes: &[T],
        mwafs: Option<&[T]>,
        ignore_metafits_tile_flags: bool,
        dont_flag_fine_channels: bool,
    ) -> Result<Self, NewRawError> {
        // The metafits argument could be a glob. If the specified
        // metafits file can't be found, treat it as a glob and expand
        // it to find a match.
        let meta_pb = {
            let pb = PathBuf::from(metadata.as_ref());
            if pb.exists() {
                pb
            } else {
                get_single_match_from_glob(metadata.as_ref().to_str().unwrap())?
            }
        };
        // TODO: Test existence.
        debug!("Using metafits: {}", meta_pb.display());

        let gpubox_pbs: Vec<PathBuf> = match gpuboxes.len() {
            0 => return Err(NewRawError::NoGpuboxes),

            // If a single gpubox file was specified, and it isn't a real
            // file, treat it as a glob and expand it to find matches.
            1 => {
                let pb = gpuboxes[0].as_ref().to_path_buf();
                if pb.exists() {
                    vec![pb]
                } else {
                    let entries = get_all_matches_from_glob(pb.as_os_str().to_str().unwrap())?;
                    if entries.is_empty() {
                        return Err(NewRawError::SingleGpuboxNotAFileOrGlob);
                    } else {
                        entries
                    }
                }
            }

            // TODO: Test existence.
            _ => gpuboxes.iter().map(|p| p.as_ref().to_path_buf()).collect(),
        };
        debug!("Using gpubox files: {:#?}", gpubox_pbs);

        debug!("Creating mwalib context");
        let context = CorrelatorContext::new(&meta_pb, &gpubox_pbs)?;
        let num_tiles = context.metafits_context.rf_inputs.len() / 2;
        debug!("There are {} total tiles", num_tiles);

        let mut tile_flags_set = HashSet::new();
        if !ignore_metafits_tile_flags {
            for flagged_meta_file in context
                .metafits_context
                .rf_inputs
                .iter()
                .filter(|rf| rf.pol == Pol::Y && rf.flagged)
            {
                tile_flags_set.insert(flagged_meta_file.ant as usize);
            }
            debug!("Using metafits tile flags; found {:?}", &tile_flags_set);
        } else {
            debug!("NOT using metafits tile flags");
        };
        let num_unflagged_tiles = num_tiles - tile_flags_set.len();
        for &f in &tile_flags_set {
            if f > num_tiles - 1 {
                return Err(NewRawError::InvalidTileFlag {
                    got: f,
                    max: num_tiles - 1,
                });
            }
        }

        // There's a chance that some or all tiles are flagged due to their
        // delays. Any delay == 32 is an indication that a dipole is "dead".
        let listed_delays = context.metafits_context.delays.clone();
        // TODO: This should probably fail instead of throw a warning. All the
        // user to proceed if they acknowledge the warning.
        debug!("Listed observation dipole delays: {:?}", &listed_delays);
        if listed_delays.iter().all(|&d| d == 32) {
            warn!("This observation has been flagged as \"do not use\", according to the metafits delays!");
        }
        let ideal_delays = if listed_delays.iter().all(|&d| d == 32) {
            listed_delays.clone()
        } else {
            // Even if DELAYS is all 32, the ideal delays are listed in HDU 2 of the
            // metafits file. Some dipoles might be dead, though, so iterate over RF
            // inputs until we have all non-32 delays.

            let mut ideal_delays = vec![32; 16];
            for rf in context
                .metafits_context
                .rf_inputs
                .iter()
                .filter(|rf| rf.pol == Pol::Y)
            {
                let dipole_delays = &rf.dipole_delays;
                if dipole_delays == &[32; 16] {
                    tile_flags_set.insert(rf.ant as usize);
                    continue;
                }
                for (i, &d) in dipole_delays.iter().enumerate() {
                    if d != 32 {
                        ideal_delays[i] = d;
                    }
                }

                // Are all delays non-32?
                if ideal_delays.iter().all(|&d| d != 32) {
                    break;
                }
            }
            ideal_delays
        };
        debug!("Ideal observation dipole delays: {:?}", &ideal_delays);

        let mut tile_flags: Vec<usize> = tile_flags_set.into_iter().collect();
        tile_flags.sort_unstable();
        let num_unflagged_tiles = num_tiles - tile_flags.len();
        for &f in &tile_flags {
            if f > num_tiles - 1 {
                return Err(NewRawError::InvalidTileFlag {
                    got: f,
                    max: num_tiles - 1,
                });
            }
        }

        // Are there any unflagged tiles?
        debug!("There are {} unflagged tiles", num_unflagged_tiles);
        if num_unflagged_tiles == 0 {
            return Err(NewRawError::AllTilesFlagged);
        }

        let fine_chan_flags: Vec<usize> = if dont_flag_fine_channels {
            vec![]
        } else {
            // If the flags aren't specified, use the observation's fine-channel
            // frequency resolution to set them.
            match context.metafits_context.corr_fine_chan_width_hz {
                // 10 kHz, 128 channels.
                10000 => vec![
                    0, 1, 2, 3, 4, 5, 6, 7, 64, 120, 121, 122, 123, 124, 125, 126, 127,
                ],

                // 20 kHz, 64 channels.
                20000 => vec![0, 1, 2, 3, 32, 60, 61, 62, 63],

                // 40 kHz, 32 channels.
                40000 => vec![0, 1, 16, 30, 31],

                f => return Err(NewRawError::UnhandledFreqResolutionForFlags(f)),
            }
        };

        let aoflags = if let Some(m) = mwafs {
            debug!("Reading AOFlagger mwaf files");
            let mut f = AOFlags::new_from_mwafs(&m)?;

            // The cotter flags are available for all times. Make them
            // match only those we'll use according to mwalib.
            f.trim(&context.metafits_context);

            // Ensure that there is a mwaf file for each specified gpubox file.
            for cc in &context.coarse_chans {
                if !f.gpubox_nums.contains(&(cc.gpubox_number as u8)) {
                    return Err(NewRawError::GpuboxFileMissingMwafFile(cc.gpubox_number));
                }
            }
            Some(f)
        } else {
            None
        };

        let timesteps: Vec<Epoch> = context
            .timesteps
            .iter()
            .map(|t| {
                let gps = t.gps_time_ms as f64 / 1e3;
                // https://en.wikipedia.org/wiki/Global_Positioning_System#Timekeeping
                // The difference between GPS and TAI time is always 19s, but
                // hifitime wants the number of TAI seconds since 1900. GPS time
                // starts at 1980 Jan 5.
                let tai = gps
                    + 19.0
                    + hifitime::SECONDS_PER_YEAR * 80.0
                    + hifitime::SECONDS_PER_DAY * 4.0;
                Epoch::from_tai_seconds(tai)
            })
            .collect::<Vec<_>>();

        // let lst = lst_from_timestep(timestep, &context, time_res);
        // let pointing = pointing_from_timestep(timestep, &context, time_res);

        // Populate a frequency context struct.
        let native_freq_res = context.metafits_context.corr_fine_chan_width_hz as f64;
        let mut fine_chan_freqs = Vec::with_capacity(
            context.metafits_context.num_corr_fine_chans_per_coarse
                * context.metafits_context.num_coarse_chans,
        );
        // TODO: I'm suspicious that the start channel freq is incorrect.
        for cc in &context.coarse_chans {
            let mut cc_freqs = Array1::range(
                cc.chan_start_hz as f64,
                cc.chan_end_hz as f64,
                context.metafits_context.corr_fine_chan_width_hz as f64,
            )
            .to_vec();
            fine_chan_freqs.append(&mut cc_freqs);
        }

        let freq_context = FreqContext {
            coarse_chan_nums: context
                .coarse_chans
                .iter()
                .map(|cc| cc.corr_chan_number as u32)
                .collect(),
            coarse_chan_freqs: context
                .coarse_chans
                .iter()
                .map(|cc| cc.chan_centre_hz as f64)
                .collect(),
            coarse_chan_width: context.coarse_chans.iter().next().unwrap().chan_width_hz as f64,
            total_bandwidth: context
                .coarse_chans
                .iter()
                .map(|cc| cc.chan_width_hz as f64)
                .sum(),
            fine_chan_freqs,
            num_fine_chans_per_coarse_chan: context.metafits_context.num_corr_fine_chans_per_coarse,
            native_fine_chan_width: context.metafits_context.corr_fine_chan_width_hz as f64,
        };

        let pointing = RADec::new(
            context
                .metafits_context
                .ra_phase_center_degrees
                .unwrap_or(context.metafits_context.ra_tile_pointing_degrees)
                .to_radians(),
            context
                .metafits_context
                .dec_phase_center_degrees
                .unwrap_or(context.metafits_context.dec_tile_pointing_degrees)
                .to_radians(),
        );
        let tile_xyz = XYZ::get_tiles_mwalib(&context.metafits_context);
        let baseline_xyz = XYZ::get_baselines(&tile_xyz);

        Ok(Self {
            context,
            aoflags,
            timesteps,
            pointing,
            tile_xyz,
            baseline_xyz,
            listed_delays,
            ideal_delays,
            tile_flags,
            fine_chan_flags,
            freq_context,
        })
    }
}

impl InputData for RawData {
    fn read(&self, time_range: Range<usize>) -> Result<Vec<Visibilities>, ReadInputDataError> {
        todo!();
    }

    fn get_obsid(&self) -> u32 {
        self.context.metafits_context.obs_id
    }

    fn get_timesteps(&self) -> &[hifitime::Epoch] {
        &self.timesteps
    }

    fn get_timestep_indices(&self) -> &Range<usize> {
        todo!()
    }

    fn get_native_time_res(&self) -> f64 {
        self.context.metafits_context.corr_int_time_ms as f64 / 1e3
    }

    fn get_pointing(&self) -> &RADec {
        &self.pointing
    }

    fn get_freq_context(&self) -> &FreqContext {
        &self.freq_context
    }

    fn get_tile_xyz(&self) -> &[XYZ] {
        &self.tile_xyz
    }

    fn get_baseline_xyz(&self) -> &[XyzBaseline] {
        &self.baseline_xyz
    }

    fn get_ideal_delays(&self) -> &[u32] {
        &self.ideal_delays
    }

    fn get_tile_flags(&self) -> &[usize] {
        &self.tile_flags
    }

    fn get_fine_chan_flags(&self) -> &[usize] {
        &self.fine_chan_flags
    }
}
