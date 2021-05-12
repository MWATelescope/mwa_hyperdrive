// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

/*!
Code to handle reading from raw MWA files.
 */

pub(crate) mod error;

pub use error::*;

use std::collections::HashSet;
use std::f64::consts::TAU;
use std::path::{Path, PathBuf};

use log::{debug, warn};
use ndarray::prelude::*;

use super::*;
use crate::context::{FreqContext, ObsContext};
use crate::flagging::aoflagger::AOFlags;
use crate::glob::*;
use crate::mwalib::{CorrelatorContext, Pol};
use mwa_hyperdrive_core::{erfa_sys, mwalib, Jones, RADec, XYZ};

/// Raw MWA data, i.e. gpubox files.
pub(crate) struct RawData {
    /// Observation metadata.
    obs_context: ObsContext,

    /// Frequency metadata.
    freq_context: FreqContext,

    // Raw-data-specific things follow.
    /// The interface to the raw data via mwalib.
    pub(crate) mwalib_context: CorrelatorContext,

    // TODO: Rename to something more general. Don't need actual AOFlagger flags.
    /// AOFlagger flags.
    ///
    /// These aren't necessarily derived from mwaf files; if the user did
    /// not supply flags, then `aoflags` may contain no flags, or just
    /// default channel flags (the edges and centre channel, for example).
    aoflags: Option<AOFlags>,
}

impl RawData {
    /// Create a new instance of the `InputData` trait with raw MWA data.
    pub(crate) fn new<T: AsRef<Path>>(
        metadata: &T,
        gpuboxes: &[T],
        mwafs: Option<&[T]>,
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
        let mwalib_context = CorrelatorContext::new(&meta_pb, &gpubox_pbs)?;
        let metafits_context = &mwalib_context.metafits_context;
        let total_num_tiles = metafits_context.rf_inputs.len() / 2;
        debug!("There are {} total tiles", total_num_tiles);

        let mut tile_flags_set: HashSet<usize> = metafits_context
            .rf_inputs
            .iter()
            .filter(|rf| rf.pol == Pol::Y && rf.flagged)
            .map(|rf_input| rf_input.ant as usize)
            .collect();
        debug!("Found tile flags {:?}", &tile_flags_set);
        let num_unflagged_tiles = total_num_tiles - tile_flags_set.len();
        for &f in &tile_flags_set {
            if f > total_num_tiles - 1 {
                return Err(NewRawError::InvalidTileFlag {
                    got: f,
                    max: total_num_tiles - 1,
                });
            }
        }

        // There's a chance that some or all tiles are flagged due to their
        // delays. Any delay == 32 is an indication that a dipole is "dead".
        let listed_delays = metafits_context.delays.clone();
        // TODO: This should probably fail instead of throw a warning. All the
        // user to proceed if they acknowledge the warning.
        debug!("Listed observation dipole delays: {:?}", &listed_delays);
        let bad_obs = if listed_delays.iter().all(|&d| d == 32) {
            warn!("This observation has been flagged as \"do not use\", according to the metafits delays!");
            true
        } else {
            false
        };
        let ideal_delays = if listed_delays.iter().all(|&d| d != 32) {
            listed_delays
        } else {
            // Even if DELAYS is all 32, the ideal delays are listed in HDU 2 of the
            // metafits file. Some dipoles might be dead, though, so iterate over RF
            // inputs until we have all non-32 delays.

            let mut ideal_delays = vec![32; 16];
            for rf in metafits_context
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
        if bad_obs {
            warn!("Using {:?} as dipole delays", &ideal_delays);
        }

        let mut tile_flags: Vec<usize> = tile_flags_set.into_iter().collect();
        tile_flags.sort_unstable();
        let num_unflagged_tiles = total_num_tiles - tile_flags.len();
        for &f in &tile_flags {
            if f > total_num_tiles - 1 {
                return Err(NewRawError::InvalidTileFlag {
                    got: f,
                    max: total_num_tiles - 1,
                });
            }
        }

        // Are there any unflagged tiles?
        debug!("There are {} unflagged tiles", num_unflagged_tiles);
        if num_unflagged_tiles == 0 {
            return Err(NewRawError::AllTilesFlagged);
        }

        let fine_chan_flags_per_coarse_chan: Vec<usize> = 
            // If the flags aren't specified, use the observation's fine-channel
            // frequency resolution to set them.
            match metafits_context.corr_fine_chan_width_hz {
                // 10 kHz, 128 channels.
                10000 => vec![
                    0, 1, 2, 3, 4, 5, 6, 7, 64, 120, 121, 122, 123, 124, 125, 126, 127,
                ],

                // 20 kHz, 64 channels.
                20000 => vec![0, 1, 2, 3, 32, 60, 61, 62, 63],

                // 40 kHz, 32 channels.
                40000 => vec![0, 1, 16, 30, 31],

                f => return Err(NewRawError::UnhandledFreqResolutionForFlags(f)),
        };

        let aoflags = if let Some(m) = mwafs {
            debug!("Reading AOFlagger mwaf files");
            let mut f = AOFlags::new_from_mwafs(&m)?;

            // The cotter flags are available for all times. Make them
            // match only those we'll use according to mwalib.
            f.trim(&metafits_context);

            // Ensure that there is a mwaf file for each specified gpubox file.
            for cc in &mwalib_context.coarse_chans {
                if !f.gpubox_nums.contains(&(cc.gpubox_number as u8)) {
                    return Err(NewRawError::GpuboxFileMissingMwafFile(cc.gpubox_number));
                }
            }
            Some(f)
        } else {
            None
        };

        let time_res = metafits_context.corr_int_time_ms as f64 / 1e3;

        // TODO: Which timesteps are good ones?
        let timesteps: Vec<hifitime::Epoch> = mwalib_context
            .timesteps
            .iter()
            .map(|t| {
                let gps = t.gps_time_ms as f64 / 1e3;
                // https://en.wikipedia.org/wiki/Global_Positioning_System#Timekeeping
                // The difference between GPS and TAI time is always 19s, but
                // hifitime wants the number of TAI seconds since 1900. GPS time
                // starts at 1980 Jan 5. Also adjust the time to be a centroid;
                // add half of the time resolution.
                let tai = gps + 19.0 + crate::constants::HIFITIME_GPS_FACTOR + time_res / 2.0;
                hifitime::Epoch::from_tai_seconds(tai)
            })
            .collect::<Vec<_>>();

        // Now that we have the timesteps, we can get the first LST. As our
        // timesteps are listed as centroids, the first timestep does not need
        // to be adjusted according to timewidth.
        let first_timestep_mjd = timesteps[0].as_mjd_utc_days();
        let lst0 = unsafe {
            let gst = erfa_sys::eraGst06a(
                erfa_sys::ERFA_DJM0,
                first_timestep_mjd,
                erfa_sys::ERFA_DJM0,
                first_timestep_mjd,
            );
            (gst + mwalib::MWA_LONGITUDE_RADIANS) % TAU
        };

        // Populate a frequency context struct.
        let mut fine_chan_freqs = Vec::with_capacity(
            metafits_context.num_corr_fine_chans_per_coarse * metafits_context.num_coarse_chans,
        );
        // TODO: I'm suspicious that the start channel freq is incorrect.
        for cc in &mwalib_context.coarse_chans {
            let mut cc_freqs = Array1::range(
                cc.chan_start_hz as f64,
                cc.chan_end_hz as f64,
                metafits_context.corr_fine_chan_width_hz as f64,
            )
            .to_vec();
            fine_chan_freqs.append(&mut cc_freqs);
        }

        let freq_context = FreqContext {
            coarse_chan_nums: mwalib_context
                .coarse_chans
                .iter()
                .map(|cc| cc.corr_chan_number as u32)
                .collect(),
            coarse_chan_freqs: mwalib_context
                .coarse_chans
                .iter()
                .map(|cc| cc.chan_centre_hz as f64)
                .collect(),
            coarse_chan_width: mwalib_context.coarse_chans[0].chan_width_hz as f64,
            total_bandwidth: mwalib_context
                .coarse_chans
                .iter()
                .map(|cc| cc.chan_width_hz as f64)
                .sum(),
            fine_chan_range: 0..mwalib_context.coarse_chans.len()
                * mwalib_context
                    .metafits_context
                    .num_corr_fine_chans_per_coarse,
            fine_chan_freqs,
            num_fine_chans_per_coarse_chan: metafits_context.num_corr_fine_chans_per_coarse,
            native_fine_chan_width: metafits_context.corr_fine_chan_width_hz as f64,
        };

        let pointing = RADec::new(
            metafits_context
                .ra_phase_center_degrees
                .unwrap_or(metafits_context.ra_tile_pointing_degrees)
                .to_radians(),
            metafits_context
                .dec_phase_center_degrees
                .unwrap_or(metafits_context.dec_tile_pointing_degrees)
                .to_radians(),
        );
        let tile_xyz = XYZ::get_tiles_mwalib(&metafits_context);
        let baseline_xyz = XYZ::get_baselines(&tile_xyz);

        let mut dipole_gains = Array2::from_elem(
            (
                metafits_context.rf_inputs.len() / 2,
                metafits_context.rf_inputs[0].dipole_gains.len(),
            ),
            1.0,
        );
        for (mut dipole_gains_for_one_tile, rf_input) in dipole_gains.outer_iter_mut().zip(
            metafits_context
                .rf_inputs
                .iter()
                .filter(|rf_input| rf_input.pol == Pol::Y),
        ) {
            dipole_gains_for_one_tile.assign(&ArrayView1::from(&rf_input.dipole_gains));
        }

        let obs_context = ObsContext {
            obsid: Some(metafits_context.obs_id),
            timesteps,
            timestep_indices: 0..mwalib_context.timesteps.len(),
            lst0,
            time_res,
            pointing,
            delays: ideal_delays,
            tile_xyz,
            baseline_xyz,
            tile_flags,
            fine_chan_flags_per_coarse_chan,
            num_unflagged_tiles,
            num_unflagged_baselines: num_unflagged_tiles * (num_unflagged_tiles - 1) / 2,
            dipole_gains,
        };

        Ok(Self {
            obs_context,
            freq_context,
            mwalib_context,
            aoflags,
        })
    }
}

impl InputData for RawData {
    fn get_obs_context(&self) -> &ObsContext {
        &self.obs_context
    }

    fn get_freq_context(&self) -> &FreqContext {
        &self.freq_context
    }

    fn read(
        &self,
        mut data_array: ArrayViewMut2<Jones<f32>>,
        timestep: usize,
        tile_to_unflagged_baseline_map: &HashMap<(usize, usize), usize>,
        flagged_fine_chans: &HashSet<usize>,
    ) -> Result<(Vec<UVW>, Array2<f32>), ReadInputDataError> {
        todo!();
    }
}
