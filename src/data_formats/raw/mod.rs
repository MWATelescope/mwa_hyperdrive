// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Code to handle reading from raw MWA files.

mod error;
mod helpers;
#[cfg(test)]
mod tests;

pub use error::*;
use helpers::*;

use std::collections::HashSet;
use std::ops::Range;
use std::path::{Path, PathBuf};

use hifitime::Epoch;
use log::{debug, info, trace, warn};
use marlu::{
    constants::{MWA_LAT_RAD, MWA_LONG_RAD},
    math::baseline_to_tiles,
    Jones, RADec, XyzGeodetic,
};
use mwalib::{CorrelatorContext, Pol};
use ndarray::prelude::*;
use rayon::prelude::*;
use vec1::Vec1;

use super::*;
use crate::{
    context::ObsContext,
    data_formats::metafits::get_dipole_gains,
    flagging::{AOFlags, MwafProducer},
    pfb_gains::{PfbFlavour, EMPIRICAL_40KHZ, LEVINE_40KHZ},
};
use mwa_hyperdrive_beam::Delays;
use mwa_hyperdrive_common::{hifitime, log, marlu, mwalib, ndarray, rayon};

/// Raw MWA data, i.e. gpubox files.
pub(crate) struct RawDataReader {
    /// Observation metadata.
    obs_context: ObsContext,

    // Raw-data-specific things follow.
    /// The interface to the raw data via mwalib.
    mwalib_context: CorrelatorContext,

    /// The poly-phase filter bank gains to be used to correct the bandpass
    /// shape for each coarse channel.
    pfb_gains: Option<Vec<f64>>,

    /// Should digital gains be applied?
    digital_gains: bool,

    /// Should cable length corrections be applied?
    cable_length_correction: bool,

    /// Should geometric corrections be applied?
    geometric_correction: bool,

    /// AOFlagger flags.
    aoflags: Option<AOFlags>,
}

impl RawDataReader {
    /// Create a new instance of the `InputData` trait with raw MWA data.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new<T: AsRef<Path>>(
        metadata: &T,
        gpuboxes: &[T],
        mwafs: Option<&[T]>,
        dipole_delays: &mut Delays,
        pfb_flavour: PfbFlavour,
        digital_gains: bool,
        cable_length_correction: bool,
        geometric_correction: bool,
    ) -> Result<RawDataReader, RawReadError> {
        // There are a lot of unwraps in this function. These are fine because
        // mwalib ensures that vectors aren't empty so when we convert a Vec to
        // Vec1, for example, we don't need to propagate a new error.

        let meta_pb = metadata.as_ref().to_path_buf();
        let gpubox_pbs: Vec<PathBuf> = gpuboxes.iter().map(|p| p.as_ref().to_path_buf()).collect();
        trace!("Using metafits: {}", meta_pb.display());
        trace!("Using gpubox files: {:#?}", gpubox_pbs);

        trace!("Creating mwalib context");
        let mwalib_context = get_mwalib_correlator_context(meta_pb, gpubox_pbs)?;
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
            return Err(RawReadError::AllTilesFlagged);
        }

        // Check that the tile flags are sensible.
        for &f in &tile_flags_set {
            if f > total_num_tiles - 1 {
                return Err(RawReadError::InvalidTileFlag {
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

        let mut flagged_tiles: Vec<usize> = tile_flags_set.into_iter().collect();
        flagged_tiles.sort_unstable();
        for &f in &flagged_tiles {
            if f > total_num_tiles - 1 {
                return Err(RawReadError::InvalidTileFlag {
                    got: f,
                    max: total_num_tiles - 1,
                });
            }
        }

        let mwax = matches!(mwalib_context.mwa_version, mwalib::MWAVersion::CorrMWAXv2);

        let flagged_fine_chans_per_coarse_chan: Vec<usize> =
            // If the flags aren't specified, use the observation's fine-channel
            // frequency resolution to set them.
            match (metafits_context.corr_fine_chan_width_hz, mwax) {
                // 10 kHz, 128 channels.
                (10000, true) => vec![
                    0, 1, 2, 3, 4, 5, 6, 7, 120, 121, 122, 123, 124, 125, 126, 127,
                ],
                // Include the centre channel.
                (10000, false) => vec![
                    0, 1, 2, 3, 4, 5, 6, 7, 64, 120, 121, 122, 123, 124, 125, 126, 127,
                ],

                // 20 kHz, 64 channels.
                (20000, true) => vec![0, 1, 2, 3, 60, 61, 62, 63],
                (20000, false) => vec![0, 1, 2, 3, 32, 60, 61, 62, 63],

                // 40 kHz, 32 channels.
                (40000, true) => vec![0, 1, 30, 31],
                (40000, false) => vec![0, 1, 16, 30, 31],

                (f, _) => return Err(RawReadError::UnhandledFreqResolutionForFlags(f)),
        };
        let flagged_fine_chans = {
            let mut flagged_fine_chans = Vec::with_capacity(
                flagged_fine_chans_per_coarse_chan.len() * mwalib_context.num_coarse_chans,
            );
            for i_cc in 0..mwalib_context.num_provided_coarse_chans {
                for &f in &flagged_fine_chans_per_coarse_chan {
                    flagged_fine_chans
                        .push(i_cc * metafits_context.num_corr_fine_chans_per_coarse + f);
                }
            }
            flagged_fine_chans
        };

        let time_res = Some(metafits_context.corr_int_time_ms as f64 / 1e3);

        let all_timesteps = Vec1::try_from_vec(mwalib_context.provided_timestep_indices.clone())
            .map_err(|_| RawReadError::NoTimesteps)?;
        let timestamps: Vec<Epoch> = mwalib_context
            .timesteps
            .iter()
            .map(|t| Epoch::from_gpst_seconds(t.gps_time_ms as f64 / 1e3))
            .collect();
        let timestamps = Vec1::try_from_vec(timestamps).map_err(|_| RawReadError::NoTimesteps)?;

        let common_good_coarse_chans: Vec<&mwalib::CoarseChannel> = mwalib_context
            .coarse_chans
            .iter()
            .enumerate()
            .filter(|(i, _)| mwalib_context.common_good_coarse_chan_indices.contains(i))
            .map(|(_, cc)| cc)
            .collect();
        let common_good_coarse_chans = Vec1::try_from_vec(common_good_coarse_chans).unwrap();

        let mut fine_chan_freqs = Vec::with_capacity(
            metafits_context.num_corr_fine_chans_per_coarse
                * mwalib_context.num_provided_coarse_chans,
        );
        for cc in &common_good_coarse_chans {
            for i_chan in 0..metafits_context.num_corr_fine_chans_per_coarse as u64 {
                fine_chan_freqs.push(
                    (cc.chan_start_hz as u64
                        + i_chan * metafits_context.corr_fine_chan_width_hz as u64)
                        as u64,
                )
            }
        }
        let fine_chan_freqs = Vec1::try_from_vec(fine_chan_freqs).unwrap();

        let phase_centre = RADec::new(
            metafits_context
                .ra_phase_center_degrees
                .unwrap_or_else(|| {
                    warn!(
                        "No phase centre specified; using the pointing centre as the phase centre"
                    );
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
        let tile_xyzs = Vec1::try_from_vec(tile_xyzs).unwrap();
        let tile_names: Vec<String> = metafits_context
            .rf_inputs
            .iter()
            .filter(|rf| rf.pol == Pol::X)
            .map(|rf_input| rf_input.tile_name.clone())
            .collect();
        let tile_names = Vec1::try_from_vec(tile_names).unwrap();

        let dipole_gains = get_dipole_gains(metafits_context);

        match (
            mwalib_context.metafits_context.geometric_delays_applied,
            geometric_correction,
        ) {
            (mwalib::GeometricDelaysApplied::No, true) => info!("Geometric delays will be applied"),

            (mwalib::GeometricDelaysApplied::No, false) => {
                info!("No geometric delays will be applied")
            }

            (g, _) => info!(
                "No geometric delays will be applied; metafits indicates {}",
                g
            ),
        }

        // TODO: Supply gains for all frequency resolutions. Delete this
        // frequency check when that's done.
        let pfb_gains = match (
            pfb_flavour,
            metafits_context.corr_fine_chan_width_hz == 40000,
        ) {
            // Not using any gains.
            (PfbFlavour::None, _) => None,

            (_, false) => {
                warn!("Not using any PFB gains (only 40kHz currently supported)");
                None
            }

            (PfbFlavour::Empirical, _) => {
                // Multiplication is cheaper than division; do the division
                // once so these values can be multiplied later.
                Some(EMPIRICAL_40KHZ.iter().map(|&g| 1.0 / g).collect())
            }

            (PfbFlavour::Levine, _) => Some(LEVINE_40KHZ.iter().map(|&g| 1.0 / g).collect()),
        };

        let aoflags = if let Some(m) = mwafs {
            trace!("Reading mwaf files");
            let mut f = AOFlags::new_from_mwafs(m)?;

            // Ensure that there is a mwaf file for each specified gpubox file.
            for gpubox_file in &mwalib_context.gpubox_batches[0].gpubox_files {
                if !f
                    .gpubox_nums
                    .contains(&(gpubox_file.channel_identifier as u8))
                {
                    return Err(RawReadError::GpuboxFileMissingMwafFile(
                        gpubox_file.channel_identifier,
                    ));
                }
            }

            // cotter has a nasty bug that can cause the start time listed in
            // mwaf files to be offset from data HDUs. Warn the user if this is
            // noticed.
            let time_res = mwalib_context.metafits_context.corr_int_time_ms as f64;
            let data_start = mwalib_context.common_start_gps_time_ms as f64;
            let data_end = mwalib_context.common_end_gps_time_ms as f64;
            let flags_start = f.start_time_milli as f64;
            let flags_end = flags_start + f.num_time_steps as f64 * time_res;
            let diff = (flags_start - data_start) / time_res;
            if diff.fract().abs() > 0.0 {
                warn!("These mwaf files do not have times corresponding to the data they were created from.");
                match f.software {
                    MwafProducer::Cotter => warn!("    This is a Cotter bug. You should probably use Birli to make new flags."),
                    MwafProducer::Birli => warn!("    These mwafs were made by Birli. Please file an issue!"),
                    MwafProducer::Unknown => warn!("    Unknown software made these mwafs."),
                }
                f.offset_bug = true;
            }

            // Warn the user if there are fewer timesteps in the aoflags than
            // there are in the raw data. This is good for signalling to the
            // user that attempting to use data timesteps without flag timesteps
            // will be a problem.
            let mut start_offset = ((data_start - flags_start) / time_res).ceil();
            let mut end_offset = ((data_end - flags_end) / time_res).ceil();
            if f.offset_bug {
                start_offset -= 1.0;
                end_offset -= 1.0;
            }
            if start_offset > 0.0 || end_offset > 0.0 {
                warn!("Not all MWA data timesteps have mwaf flags available");
                match (start_offset > 0.0, end_offset > 0.0) {
                    (true, true) => warn!(
                        "   {} timesteps at the start and {} at the end are not represented",
                        start_offset, end_offset,
                    ),
                    (true, false) => warn!(
                        "   {} timesteps at the start are not represented",
                        start_offset,
                    ),
                    (false, true) => {
                        warn!("   {} timesteps at the end are not represented", end_offset)
                    }
                    (false, false) => unreachable!(),
                }
            }

            Some(f)
        } else {
            None
        };

        let obs_context = ObsContext {
            obsid: Some(metafits_context.obs_id),
            timestamps,
            all_timesteps,
            unflagged_timesteps: mwalib_context.common_good_timestep_indices.clone(),
            phase_centre,
            pointing_centre,
            tile_names,
            tile_xyzs,
            flagged_tiles,
            autocorrelations_present: true,
            dipole_gains: Some(dipole_gains),
            time_res,
            array_longitude_rad: Some(MWA_LONG_RAD),
            array_latitude_rad: Some(MWA_LAT_RAD),
            coarse_chan_nums: common_good_coarse_chans
                .iter()
                .map(|cc| cc.corr_chan_number as u32)
                .collect(),
            coarse_chan_freqs: common_good_coarse_chans
                .iter()
                .map(|cc| cc.chan_centre_hz as f64)
                .collect(),
            coarse_chan_width: metafits_context.coarse_chan_width_hz as f64,
            num_fine_chans_per_coarse_chan: metafits_context.num_corr_fine_chans_per_coarse,
            total_bandwidth: common_good_coarse_chans
                .iter()
                .map(|cc| cc.chan_width_hz as f64)
                .sum(),
            freq_res: Some(metafits_context.corr_fine_chan_width_hz as f64),
            fine_chan_freqs,
            flagged_fine_chans,
            flagged_fine_chans_per_coarse_chan,
        };

        Ok(RawDataReader {
            obs_context,
            mwalib_context,
            pfb_gains,
            digital_gains,
            cable_length_correction,
            geometric_correction,
            aoflags,
        })
    }

    /// Uh... applies poly-phase filter bank gains (selected by the "new"
    /// method) to the supplied visibilities.
    fn apply_pfb_gains(&self, mut vis: ArrayViewMut2<Jones<f32>>) {
        if let Some(pfb_gains) = &self.pfb_gains {
            vis.outer_iter_mut()
                .into_par_iter()
                .enumerate()
                .for_each(|(i_freq, mut vis_bl)| {
                    let pfb_gain = pfb_gains[i_freq % pfb_gains.len()];
                    vis_bl.iter_mut().for_each(|v| {
                        // Do the multiplication at double precision.
                        let vd: Jones<f64> = Jones::from(*v) * pfb_gain;
                        *v = Jones::from(vd);
                    });
                });
        }
    }

    /// Apply digital gains to the supplied visibilities.
    fn apply_digital_gains(&self, mut vis: ArrayViewMut2<Jones<f32>>) {
        if self.digital_gains {
            vis.axis_iter_mut(Axis(1))
                .into_par_iter()
                .enumerate()
                .for_each(|(i_bl, mut vis)| {
                    let (tile1, tile2) =
                        baseline_to_tiles(self.mwalib_context.metafits_context.num_ants, i_bl);
                    // The digital gains in a metafits file are listed in
                    // increasing frequency; that is, the first gain corresponds
                    // to the lowest-frequency coarse channel, the last gain the
                    // highest-frequency coarse channel. Confirmed by Andrew
                    // Williams.

                    // tile_gains has at most two references to a bunch of
                    // digital gains. The references correspond to the tiles
                    // making up this baseline. If this is an auto-correlation,
                    // there's only one tile.

                    // Set tile_gains to anything to make Rust happy. I don't
                    // want to allocate a new vector even though it's probably
                    // not significantly affecting performance.
                    let mut tile_gains: [&Vec<f64>; 2] = [
                        &self.mwalib_context.metafits_context.rf_inputs[0].digital_gains,
                        &self.mwalib_context.metafits_context.rf_inputs[0].digital_gains,
                    ];
                    self.mwalib_context
                        .metafits_context
                        .rf_inputs
                        .iter()
                        // Discard every second RF input; the digital gains are
                        // the same for both.
                        .filter(|rf| rf.pol == Pol::X)
                        // RFs are ordered by increasing antenna number. So
                        // tile1 always comes before tile2.
                        .filter(|rf| rf.ant == tile1 as u32 || rf.ant == tile2 as u32)
                        .map(|rf| &rf.digital_gains)
                        .enumerate()
                        .for_each(|(i, rf)| tile_gains[i] = rf);

                    vis.iter_mut().enumerate().for_each(|(i_chan, vis)| {
                        let i_cc = i_chan
                            / self
                                .mwalib_context
                                .metafits_context
                                .num_corr_fine_chans_per_coarse;
                        let mut gain = tile_gains[0][i_cc];
                        let i = if tile1 != tile2 { 1 } else { 0 };
                        gain *= tile_gains[i][i_cc];

                        // Do the division at double precision.
                        let vd: Jones<f64> = Jones::from(*vis) / gain;
                        *vis = Jones::from(vd);
                    });
                });
        }
    }

    fn apply_mwaf_flags(
        &self,
        mwaf_timestep: Option<usize>,
        gpubox_channels: Range<usize>,
        mut vis: ArrayViewMut2<Jones<f32>>,
        mut weights: ArrayViewMut2<f32>,
    ) {
        if let Some(timestep) = mwaf_timestep {
            for (i_gpubox_chan, gpubox_channel) in gpubox_channels.into_iter().enumerate() {
                let flags = self.aoflags.as_ref().unwrap().flags[&(gpubox_channel as u8)]
                    .slice(s![timestep, .., ..,]);
                // Select only the applicable frequencies.
                let n = self.obs_context.num_fine_chans_per_coarse_chan;
                let selection = s![
                    (i_gpubox_chan * n)..((i_gpubox_chan + 1) * n),
                    .. // All baselines
                ];

                vis.slice_mut(selection)
                    .outer_iter_mut()
                    // .into_par_iter()
                    .zip(weights.slice_mut(selection).outer_iter_mut())
                    .enumerate()
                    .for_each(|(i_freq, (mut vis, mut weights))| {
                        // Get the right bit for this frequency channel.
                        let bit = i_freq % 8;
                        // Get the right flags for this frequency channel. There
                        // are 8 flags per value in flags (an 8-bit byte).
                        let flags_for_bls = flags.slice(s![.., i_freq / 8]).mapv(|f| {
                            if ((f.reverse_bits() >> bit) & 0x01) == 0x01 {
                                // If we are flagging, keep 0, otherwise 1. This
                                // saves a conditional when actually applying
                                // the flag.
                                0_u8
                            } else {
                                1
                            }
                        });

                        vis.iter_mut()
                            .zip(weights.iter_mut())
                            .zip(flags_for_bls)
                            .for_each(|((vis, weight), flag)| {
                                let f = flag as f32;
                                *vis *= f;
                                *weight *= f;
                            });
                    });
            }
        }
    }

    /// An internal method for reading visibilities. Cross- and/or
    /// auto-correlation visibilities and weights are written to the supplied
    /// arrays.
    fn read_inner(
        &self,
        crosses: Option<CrossData>,
        autos: Option<AutoData>,
        timestep: usize,
        flagged_fine_chans: &HashSet<usize>,
    ) -> Result<(), ReadInputDataError> {
        // TODO: Handle non-contiguous coarse channels.

        // Check that mwaf flags are available for this timestep.
        let mwaf_timestep = match &self.aoflags {
            Some(aoflags) => {
                // The time resolution is always specified for raw MWA data. Use
                // units of milliseconds with the flags.
                let time_res = self.obs_context.time_res.unwrap();
                let flags_start = aoflags.start_time_milli as f64 / 1e3;
                let flags_end = flags_start + aoflags.num_time_steps as f64 * time_res;
                let gps = self.obs_context.timestamps[timestep].as_gpst_seconds();
                if !(flags_start..flags_end).contains(&gps) {
                    return Err(ReadInputDataError::MwafFlagsMissingForTimestep { timestep, gps });
                }
                // Find the flags timestep index corresponding to the data
                // timestep index.
                let offset = (gps - flags_start) / time_res;
                let offset = if aoflags.offset_bug {
                    offset.floor() as usize
                } else {
                    offset.round() as usize
                };

                Some(offset)
            }
            None => None,
        };

        // mwalib won't provide a context without gpubox files, so it is safe to
        // unwrap `first` and `last`.
        let coarse_chan_range = *self
            .mwalib_context
            .common_good_coarse_chan_indices
            .first()
            .unwrap()
            ..*self
                .mwalib_context
                .common_good_coarse_chan_indices
                .last()
                .unwrap()
                + 1;
        let gpubox_channels = self.mwalib_context.gpubox_batches[0]
            .gpubox_files
            .first()
            .unwrap()
            .channel_identifier
            ..self.mwalib_context.gpubox_batches[0]
                .gpubox_files
                .last()
                .unwrap()
                .channel_identifier
                + 1;
        let timestep_range = timestep..timestep + 1;

        // Read in the data via Birli.
        let (mut vis, flags) = birli::context_to_jones_array(
            &self.mwalib_context,
            &timestep_range,
            &coarse_chan_range,
            None,
            false,
        )?;
        let weight_factor = birli::flags::get_weight_factor(&self.mwalib_context);
        let weights = birli::flags::flag_to_weight_array(flags.view(), weight_factor);

        // Correct the raw data.
        if !self.mwalib_context.metafits_context.cable_delays_applied
            && self.cable_length_correction
        {
            birli::correct_cable_lengths(&self.mwalib_context, &mut vis, &coarse_chan_range, false);
        }
        match (
            self.mwalib_context
                .metafits_context
                .geometric_delays_applied,
            self.geometric_correction,
        ) {
            // Nothing to do; user has spoken!
            (_, false) => (),

            (mwalib::GeometricDelaysApplied::No, true) => birli::correct_geometry(
                &self.mwalib_context,
                &mut vis,
                &timestep_range,
                &coarse_chan_range,
                None,
                Some(self.obs_context.phase_centre),
                false,
            ),

            // Nothing to do; metafits indicates delay corrections have been
            // applied, and this is reported to the user in the new method.
            (
                mwalib::GeometricDelaysApplied::AzElTracking
                | mwalib::GeometricDelaysApplied::TilePointing
                | mwalib::GeometricDelaysApplied::Zenith,
                _,
            ) => (),
        }
        // Remove the extraneous time dimension; there's only ever one timestep.
        let mut vis = vis.remove_axis(Axis(0));
        let mut weights = weights.remove_axis(Axis(0));

        // TODO: It's (probably) a lot more efficient to traverse the
        // visibilities once, applying PFB gains, digital gains and weights all
        // at the same time.
        self.apply_pfb_gains(vis.view_mut());
        self.apply_digital_gains(vis.view_mut());
        self.apply_mwaf_flags(
            mwaf_timestep,
            gpubox_channels,
            vis.view_mut(),
            weights.view_mut(),
        );

        // If applicable, write the cross-correlation visibilities to our
        // `data_array`, ignoring any flagged baselines.
        if let Some(CrossData {
            mut data_array,
            mut weights_array,
            tile_to_unflagged_baseline_map,
        }) = crosses
        {
            vis.outer_iter()
                .zip(weights.outer_iter())
                .enumerate()
                // Let only unflagged channels proceed.
                .filter(|(i_chan, _)| !flagged_fine_chans.contains(i_chan))
                // Discard the channel index and get the unflagged
                // channel index.
                .map(|(_, data)| data)
                .enumerate()
                .for_each(|(i_unflagged_chan, (vis, weights))| {
                    vis.iter()
                        .zip(weights)
                        .enumerate()
                        // Let only unflagged baselines proceed.
                        .filter(|(i_baseline, _)| {
                            let (tile1, tile2) = baseline_to_tiles(
                                self.mwalib_context.metafits_context.num_ants,
                                *i_baseline,
                            );
                            tile_to_unflagged_baseline_map
                                .get(&(tile1, tile2))
                                .is_some()
                        })
                        // Discard the baseline index and get the unflagged baseline
                        // index.
                        .map(|(_, data)| data)
                        .enumerate()
                        .for_each(|(i_unflagged_baseline, (vis, weight))| {
                            data_array[(i_unflagged_baseline, i_unflagged_chan)] = *vis;
                            weights_array[(i_unflagged_baseline, i_unflagged_chan)] = *weight;
                        });
                });
        }

        // If applicable, write the auto-correlation visibilities to our
        // `data_array`, ignoring any flagged baselines.
        if let Some(AutoData {
            mut data_array,
            mut weights_array,
            flagged_tiles,
        }) = autos
        {
            vis.outer_iter()
                .zip(weights.outer_iter())
                .enumerate()
                // Let only unflagged channels proceed.
                .filter(|(i_chan, _)| !flagged_fine_chans.contains(i_chan))
                // Discard the channel index and get the unflagged channel
                // index.
                .map(|(_, data)| data)
                .enumerate()
                .for_each(|(i_unflagged_chan, (vis, weights))| {
                    vis.iter()
                        .zip(weights)
                        .enumerate()
                        // Let only unflagged autos proceed.
                        .filter(|(i_baseline, _)| {
                            let (tile1, tile2) = baseline_to_tiles(
                                self.mwalib_context.metafits_context.num_ants,
                                *i_baseline,
                            );
                            tile1 == tile2 && !flagged_tiles.contains(&tile1)
                        })
                        // Discard the baseline index and get the unflagged tile
                        // index.
                        .map(|(_, data)| data)
                        .enumerate()
                        .for_each(|(i_unflagged_tile, (vis, weight))| {
                            data_array[(i_unflagged_tile, i_unflagged_chan)] = *vis;
                            weights_array[(i_unflagged_tile, i_unflagged_chan)] = *weight;
                        });
                });
        }

        Ok(())
    }
}

/// A private container for cross-correlation data. It only exists to give
/// meaning to the types.
struct CrossData<'a, 'b, 'c> {
    data_array: ArrayViewMut2<'a, Jones<f32>>,
    weights_array: ArrayViewMut2<'b, f32>,
    tile_to_unflagged_baseline_map: &'c HashMap<(usize, usize), usize>,
}

/// A private container for auto-correlation data. It only exists to give
/// meaning to the types.
struct AutoData<'a, 'b, 'c> {
    data_array: ArrayViewMut2<'a, Jones<f32>>,
    weights_array: ArrayViewMut2<'b, f32>,
    flagged_tiles: &'c [usize],
}

impl InputData for RawDataReader {
    fn get_obs_context(&self) -> &ObsContext {
        &self.obs_context
    }

    fn get_input_data_type(&self) -> VisInputType {
        VisInputType::Raw
    }

    fn read_crosses_and_autos(
        &self,
        cross_data_array: ArrayViewMut2<Jones<f32>>,
        cross_weights_array: ArrayViewMut2<f32>,
        auto_data_array: ArrayViewMut2<Jones<f32>>,
        auto_weights_array: ArrayViewMut2<f32>,
        timestep: usize,
        tile_to_unflagged_baseline_map: &HashMap<(usize, usize), usize>,
        flagged_tiles: &[usize],
        flagged_fine_chans: &HashSet<usize>,
    ) -> Result<(), ReadInputDataError> {
        self.read_inner(
            Some(CrossData {
                data_array: cross_data_array,
                weights_array: cross_weights_array,
                tile_to_unflagged_baseline_map,
            }),
            Some(AutoData {
                data_array: auto_data_array,
                weights_array: auto_weights_array,
                flagged_tiles,
            }),
            timestep,
            flagged_fine_chans,
        )
    }

    fn read_crosses(
        &self,
        data_array: ArrayViewMut2<Jones<f32>>,
        weights_array: ArrayViewMut2<f32>,
        timestep: usize,
        tile_to_unflagged_baseline_map: &HashMap<(usize, usize), usize>,
        flagged_fine_chans: &HashSet<usize>,
    ) -> Result<(), ReadInputDataError> {
        self.read_inner(
            Some(CrossData {
                data_array,
                weights_array,
                tile_to_unflagged_baseline_map,
            }),
            None,
            timestep,
            flagged_fine_chans,
        )
    }

    fn read_autos(
        &self,
        data_array: ArrayViewMut2<Jones<f32>>,
        weights_array: ArrayViewMut2<f32>,
        timestep: usize,
        flagged_tiles: &[usize],
        flagged_fine_chans: &HashSet<usize>,
    ) -> Result<(), ReadInputDataError> {
        self.read_inner(
            None,
            Some(AutoData {
                data_array,
                weights_array,
                flagged_tiles,
            }),
            timestep,
            flagged_fine_chans,
        )
    }
}
