// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Code to handle reading from raw MWA files.

pub(crate) mod error;

pub use error::*;

use std::collections::HashSet;
use std::path::Path;

use log::{debug, info, trace, warn};
use mwa_rust_core::{
    constants::{MWA_LAT_RAD, MWA_LONG_RAD},
    math::baseline_to_tiles,
    time::gps_to_epoch,
    Jones, RADec, XyzGeodetic,
};
use mwalib::{CorrelatorContext, Pol};
use ndarray::prelude::*;

use super::*;
use crate::context::{FreqContext, ObsContext};
use crate::flagging::aoflagger::AOFlags;
use crate::mwalib;
use mwa_hyperdrive_beam::Delays;

/// Raw MWA data, i.e. gpubox files.
pub(crate) struct RawData {
    /// Observation metadata.
    obs_context: ObsContext,

    /// Frequency metadata.
    freq_context: FreqContext,

    // Raw-data-specific things follow.
    /// The interface to the raw data via mwalib.
    mwalib_context: CorrelatorContext,

    // TODO: Rename to something more general. Don't need actual AOFlagger flags.
    /// AOFlagger flags.
    ///
    /// These aren't necessarily derived from mwaf files; if the user did
    /// not supply flags, then `aoflags` may contain no flags, or just
    /// default channel flags (the edges and centre channel, for example).
    _aoflags: Option<AOFlags>,
}

impl RawData {
    /// Create a new instance of the `InputData` trait with raw MWA data.
    pub(crate) fn new<T: AsRef<Path>>(
        metadata: &T,
        gpuboxes: &[T],
        mwafs: Option<&[T]>,
        dipole_delays: &mut Delays,
    ) -> Result<Self, NewRawError> {
        let meta_pb = metadata.as_ref();
        let gpubox_pbs: Vec<&Path> = gpuboxes.iter().map(|p| p.as_ref()).collect();
        trace!("Using metafits: {}", meta_pb.display());
        trace!("Using gpubox files: {:#?}", gpubox_pbs);

        trace!("Creating mwalib context");
        let mwalib_context = CorrelatorContext::new(&meta_pb, &gpubox_pbs)?;
        let metafits_context = &mwalib_context.metafits_context;
        let total_num_tiles = metafits_context.num_ants;
        trace!("There are {} total tiles", total_num_tiles);

        let tile_flags_set: HashSet<usize> = metafits_context
            .rf_inputs
            .iter()
            .filter(|rf| rf.pol == Pol::X && rf.flagged)
            .map(|rf_input| rf_input.ant as usize)
            .collect();
        debug!("Found metafits tile flags: {:?}", &tile_flags_set);

        // Are there any unflagged tiles?
        let num_unflagged_tiles = total_num_tiles - tile_flags_set.len();
        debug!("There are {} unflagged tiles", num_unflagged_tiles);
        if num_unflagged_tiles == 0 {
            return Err(NewRawError::AllTilesFlagged);
        }

        // Check that the tile flags are sensible.
        for &f in &tile_flags_set {
            if f > total_num_tiles - 1 {
                return Err(NewRawError::InvalidTileFlag {
                    got: f,
                    max: total_num_tiles - 1,
                });
            }
        }

        // All delays == 32 is an indication that this observation is "bad".
        let listed_delays = metafits_context.delays.clone();
        debug!("Listed observation dipole delays: {:?}", &listed_delays);
        if listed_delays.iter().all(|&d| d == 32) {
            warn!("This observation has been flagged as \"do not use\", according to the metafits delays!");
            true
        } else {
            false
        };
        if matches!(dipole_delays, Delays::None) {
            *dipole_delays = Delays::Full(metafits::get_dipole_delays(metafits_context));
        }

        let mut tile_flags: Vec<usize> = tile_flags_set.into_iter().collect();
        tile_flags.sort_unstable();
        for &f in &tile_flags {
            if f > total_num_tiles - 1 {
                return Err(NewRawError::InvalidTileFlag {
                    got: f,
                    max: total_num_tiles - 1,
                });
            }
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
            let mut f = AOFlags::new_from_mwafs(m)?;

            // The cotter flags are available for all times. Make them
            // match only those we'll use according to mwalib.
            f.trim(metafits_context);

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

        let time_res = Some(metafits_context.corr_int_time_ms as f64 / 1e3);

        // TODO: Which timesteps are good ones?
        let timesteps: Vec<hifitime::Epoch> = mwalib_context
            .timesteps
            .iter()
            .map(|t| gps_to_epoch(t.gps_time_ms as f64 / 1e3))
            .collect();

        // Populate a frequency context struct.
        let mut fine_chan_freqs = Vec::with_capacity(
            metafits_context.num_corr_fine_chans_per_coarse
                * metafits_context.metafits_coarse_chans.len(),
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
            native_fine_chan_width: Some(metafits_context.corr_fine_chan_width_hz as f64),
        };

        let phase_centre = RADec::new(
            metafits_context
                .ra_phase_center_degrees
                .unwrap_or_else(|| {
                    warn!("Assuming that the phase centre is the same as the pointing centre");
                    metafits_context.ra_tile_pointing_degrees
                })
                .to_radians(),
            metafits_context
                .dec_phase_center_degrees
                .unwrap_or(metafits_context.dec_tile_pointing_degrees)
                .to_radians(),
        );
        let pointing_centre = Some(RADec::new(
            metafits_context.ra_tile_pointing_degrees.to_radians(),
            metafits_context.dec_tile_pointing_degrees.to_radians(),
        ));
        let tile_xyzs = XyzGeodetic::get_tiles_mwa(metafits_context);
        let tile_names: Vec<String> = metafits_context
            .rf_inputs
            .iter()
            .filter(|rf| rf.pol == Pol::X)
            .map(|rf_input| rf_input.tile_name.clone())
            .collect();

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

        match mwalib_context.metafits_context.geometric_delays_applied {
            mwalib::GeometricDelaysApplied::No => (),

            g => info!(
                "No geometric delays will be applied; metafits indicates {}",
                g
            ),
        }

        let obs_context = ObsContext {
            obsid: Some(metafits_context.obs_id),
            timesteps,
            unflagged_timestep_indices: 0..mwalib_context.timesteps.len(),
            phase_centre,
            pointing_centre,
            tile_names,
            tile_xyzs,
            tile_flags,
            autocorrelations_present: true,
            fine_chan_flags_per_coarse_chan,
            dipole_gains: Some(dipole_gains),
            time_res,
            array_longitude_rad: Some(MWA_LONG_RAD),
            array_latitude_rad: Some(MWA_LAT_RAD),
        };

        Ok(Self {
            obs_context,
            freq_context,
            mwalib_context,
            _aoflags: aoflags,
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

    fn get_input_data_type(&self) -> VisInputType {
        VisInputType::Raw
    }

    fn read_crosses(
        &self,
        mut data_array: ArrayViewMut2<Jones<f32>>,
        mut weights_array: ArrayViewMut2<f32>,
        timestep: usize,
        tile_to_unflagged_baseline_map: &HashMap<(usize, usize), usize>,
        flagged_fine_chans: &HashSet<usize>,
    ) -> Result<(), ReadInputDataError> {
        // TODO: Handle non-contiguous coarse channels.
        // mwalib won't provide a context without gpubox files, so it is safe to
        // unwrap `first` and `last`.
        let coarse_chan_range = *self
            .mwalib_context
            .common_coarse_chan_indices
            .first()
            .unwrap()
            ..*self
                .mwalib_context
                .common_coarse_chan_indices
                .last()
                .unwrap()
                + 1;
        let timestep_range = timestep..timestep + 1;

        // Read in the data via Birli.
        let (mut vis, flags) = birli::context_to_jones_array(
            &self.mwalib_context,
            &timestep_range,
            &coarse_chan_range,
            None,
        );
        let weights = birli::flags::flag_to_weight_array(&self.mwalib_context, flags.view());

        // Correct the raw data.
        if !self.mwalib_context.metafits_context.cable_delays_applied {
            birli::correct_cable_lengths(&self.mwalib_context, &mut vis, &coarse_chan_range);
        }
        match self
            .mwalib_context
            .metafits_context
            .geometric_delays_applied
        {
            mwalib::GeometricDelaysApplied::No => birli::correct_geometry(
                &self.mwalib_context,
                &mut vis,
                &timestep_range,
                &coarse_chan_range,
                None,
            ),

            // Nothing to do; metafits indicates delay corrections have been
            // applied, and this is reported to the user in the new method.
            _ => (),
        }

        // Remove the extraneous time dimension; there's only ever one timestep.
        let mut vis = vis.remove_axis(Axis(0));
        let mut weights = weights.remove_axis(Axis(0));

        // Apply the weights to the just-read-in visibilities. We don't care for
        // negative weights; they just signal that the visibilities should be
        // flagged.
        ndarray::Zip::from(&mut vis)
            .and(&mut weights)
            .par_apply(|v, w| {
                if *w < 0.0 {
                    *w = 0.0;
                }
                *v *= *w;
            });

        // Write the visibilities to our `data_array`, ignoring any flagged
        // baselines.
        let mut data_array_bl_index = 0;
        for (i_bl, (vis_bl, weights_bl)) in vis
            .reversed_axes()
            .outer_iter()
            .zip(weights.reversed_axes().outer_iter())
            .enumerate()
        {
            let (tile1, tile2) =
                baseline_to_tiles(self.mwalib_context.metafits_context.num_ants, i_bl);

            if let Some(_) = tile_to_unflagged_baseline_map.get(&(tile1, tile2)) {
                let mut data_array_freq_index = 0;
                for (i_freq, (&vis, &weight)) in vis_bl.iter().zip(weights_bl.iter()).enumerate() {
                    if !flagged_fine_chans.contains(&i_freq) {
                        // data_array[(data_array_bl_index, data_array_freq_index)] = vis;
                        data_array[(data_array_bl_index, data_array_freq_index)] =
                            Jones::from([vis[0], vis[1], vis[2], vis[3]]);
                        weights_array[(data_array_bl_index, data_array_freq_index)] = weight;

                        data_array_freq_index += 1;
                    }
                }

                data_array_bl_index += 1;
            }
        }

        Ok(())
    }

    fn read_autos(
        &self,
        data_array: ArrayViewMut2<Jones<f32>>,
        weights_array: ArrayViewMut2<f32>,
        timestep: usize,
        flagged_tiles: &HashSet<usize>,
        flagged_fine_chans: &HashSet<usize>,
    ) -> Result<(), ReadInputDataError> {
        todo!()
    }
}
