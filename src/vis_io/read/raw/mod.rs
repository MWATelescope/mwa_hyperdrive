// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Code to handle reading from raw MWA files.

mod error;
mod helpers;
#[cfg(test)]
mod tests;

pub(crate) use error::*;
use helpers::*;

use std::{
    collections::HashSet,
    ops::Range,
    path::{Path, PathBuf},
};

use birli::PreprocessContext;
use hifitime::{Duration, Epoch, Unit};
use itertools::izip;
use log::{debug, trace, warn};
use marlu::{math::baseline_to_tiles, Jones, LatLngHeight, RADec, VisSelection, XyzGeodetic};
use mwalib::{CorrelatorContext, GeometricDelaysApplied, MWAVersion, Pol};
use ndarray::prelude::*;
use vec1::Vec1;

use super::*;
use crate::{
    beam::Delays,
    context::ObsContext,
    flagging::{MwafFlags, MwafProducer},
    math::TileBaselineFlags,
    metafits,
    pfb_gains::{PfbFlavour, PfbParseError},
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RawDataCorrections {
    /// What 'flavour' are the PFB gains?
    pub(crate) pfb_flavour: PfbFlavour,

    /// Should digital gains be applied?
    pub(crate) digital_gains: bool,

    /// Should cable length corrections be applied?
    pub(crate) cable_length: bool,

    /// Should geometric corrections be applied?
    pub(crate) geometric: bool,
}

impl RawDataCorrections {
    /// Create a new [RawDataCorrections]. This is mostly useful for parsing the
    /// text representing the PFB flavour; if you already have the [PfbFlavour]
    /// enum, then you may as well populate your own `RawDataCorrections`
    /// struct. If a string isn't provided for `pfb_flavour`, then the
    /// [default](crate::pfb_gains::DEFAULT_PFB_FLAVOUR) is used.
    pub(crate) fn new(
        pfb_flavour: Option<&str>,
        digital_gains: bool,
        cable_length: bool,
        geometric: bool,
    ) -> Result<RawDataCorrections, PfbParseError> {
        Ok(Self {
            pfb_flavour: match pfb_flavour {
                None => crate::pfb_gains::DEFAULT_PFB_FLAVOUR,
                Some(s) => PfbFlavour::parse(s)?,
            },
            digital_gains,
            cable_length,
            geometric,
        })
    }
}

impl Default for RawDataCorrections {
    fn default() -> Self {
        Self {
            pfb_flavour: crate::pfb_gains::DEFAULT_PFB_FLAVOUR,
            digital_gains: true,
            cable_length: true,
            geometric: true,
        }
    }
}

/// Raw MWA data, i.e. gpubox files.
pub(crate) struct RawDataReader {
    /// Observation metadata.
    obs_context: ObsContext,

    // Raw-data-specific things follow.
    /// The interface to the raw data via mwalib.
    pub(crate) mwalib_context: CorrelatorContext,

    /// The poly-phase filter bank gains to be used to correct the bandpass
    /// shape for each coarse channel.
    pfb_gains: Option<&'static [f64]>,

    /// The corrections to be applied.
    corrections: RawDataCorrections,

    /// A pair of tiles for each baseline in the data (including autos and
    /// ignoring all flags). e.g. The first element is (0, 0).
    all_baseline_tile_pairs: Vec<(usize, usize)>,

    /// Mwaf flags.
    mwaf_flags: Option<MwafFlags>,
}

impl RawDataReader {
    /// Create a new [`RawDataReader`].
    pub(crate) fn new<T: AsRef<Path>>(
        metadata: &T,
        gpuboxes: &[T],
        mwafs: Option<&[T]>,
        corrections: RawDataCorrections,
    ) -> Result<RawDataReader, VisReadError> {
        Self::new_inner(metadata, gpuboxes, mwafs, corrections).map_err(VisReadError::from)
    }

    /// Create a new [`RawDataReader`].
    fn new_inner<T: AsRef<Path>>(
        metadata: &T,
        gpuboxes: &[T],
        mwafs: Option<&[T]>,
        corrections: RawDataCorrections,
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

        let is_mwax = match mwalib_context.mwa_version {
            MWAVersion::CorrMWAXv2 => true,
            MWAVersion::CorrLegacy | MWAVersion::CorrOldLegacy => false,
            MWAVersion::VCSLegacyRecombined | MWAVersion::VCSMWAXv2 => {
                return Err(RawReadError::Vcs)
            }
        };

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
        let listed_delays = &metafits_context.delays;
        debug!("Listed observation dipole delays: {listed_delays:?}");
        if listed_delays.iter().all(|&d| d == 32) {
            warn!("This observation has been flagged as \"do not use\", according to the metafits delays!");
            true
        } else {
            false
        };
        let dipole_delays = Delays::Full(metafits::get_dipole_delays(metafits_context));

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

        let flagged_fine_chans_per_coarse_chan: Vec<usize> =
            // If the flags aren't specified, use the observation's fine-channel
            // frequency resolution to set them.
            match (metafits_context.corr_fine_chan_width_hz, is_mwax) {
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

        let time_res = Some(Duration::from_f64(
            metafits_context.corr_int_time_ms as f64,
            Unit::Millisecond,
        ));

        let all_timesteps = Vec1::try_from_vec(mwalib_context.provided_timestep_indices.clone())
            .map_err(|_| RawReadError::NoTimesteps)?;
        let timestamps: Vec<Epoch> = mwalib_context
            .timesteps
            .iter()
            .map(|t| {
                Epoch::from_gpst_seconds(
                    (t.gps_time_ms + metafits_context.corr_int_time_ms / 2) as f64 / 1e3,
                )
            })
            .collect();
        let timestamps = Vec1::try_from_vec(timestamps).map_err(|_| RawReadError::NoTimesteps)?;

        // Use the "common good" coarse channels. If there aren't any, complain.
        let coarse_chan_indices = &mwalib_context.common_good_coarse_chan_indices;
        if coarse_chan_indices.is_empty() {
            return Err(RawReadError::NoGoodCoarseChannels);
        }
        let (coarse_chan_nums, coarse_chan_freqs) = mwalib_context
            .coarse_chans
            .iter()
            .enumerate()
            .filter(|(i, _)| coarse_chan_indices.contains(i))
            .map(|(_, cc)| (cc.corr_chan_number as u32, cc.chan_centre_hz as f64))
            .unzip();
        debug!("Coarse channel numbers: {:?}", coarse_chan_nums);
        debug!(
            "Coarse channel centre frequencies [Hz]: {:?}",
            coarse_chan_freqs
        );

        let fine_chan_freqs = mwalib_context
            .get_fine_chan_freqs_hz_array(coarse_chan_indices)
            .into_iter()
            .map(|f| f.round() as u64)
            .collect();
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

        let dipole_gains = metafits::get_dipole_gains(metafits_context);

        match mwalib_context.metafits_context.geometric_delays_applied {
            mwalib::GeometricDelaysApplied::No => (),

            g => debug!(
                "No geometric delays will be applied; metafits indicates {}",
                g
            ),
        }

        let pfb_gains = corrections.pfb_flavour.get_gains();

        let mwaf_flags = if let Some(m) = mwafs {
            trace!("Reading mwaf files");
            let mut f = MwafFlags::new_from_mwafs(m)?;

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
            let time_res = Duration::from_f64(
                mwalib_context.metafits_context.corr_int_time_ms as f64,
                Unit::Millisecond,
            );
            let data_start =
                Epoch::from_gpst_seconds(mwalib_context.common_start_gps_time_ms as f64 / 1e3)
                    + time_res / 2.0;
            let data_end =
                Epoch::from_gpst_seconds(mwalib_context.common_end_gps_time_ms as f64 / 1e3)
                    + time_res / 2.0;
            let flags_start = f.start_time;
            let flags_end = flags_start + f.num_time_steps as f64 * time_res;
            let diff = (flags_start - data_start).in_seconds() / time_res.in_seconds();
            debug!("Data start time (GPS): {}", data_start.as_gpst_seconds());
            debug!("Flag start time (GPS): {}", flags_start.as_gpst_seconds());
            debug!("(flags_start - data_start).in_seconds() / time_res.in_seconds(): {diff}");
            if diff.fract().abs() > 0.0 {
                warn!("These mwaf files do not have times corresponding to the data they were created from.");
                match f.software {
                    MwafProducer::Cotter => warn!("    This is a Cotter bug. You should probably use Birli to make new flags."),
                    MwafProducer::Birli => warn!("    These mwafs were made by Birli. Please file an issue!"),
                    MwafProducer::Unknown => warn!("    Unknown software made these mwafs."),
                }
                f.offset_bug = true;
            }

            // Warn the user if there are fewer timesteps in the mwaf flags than
            // there are in the raw data. This is good for signalling to the
            // user that attempting to use data timesteps without flag timesteps
            // will be a problem.
            let mut start_offset =
                ((data_start - flags_start).in_seconds() / time_res.in_seconds()).ceil();
            let mut end_offset =
                ((data_end - flags_end).in_seconds() / time_res.in_seconds()).ceil();
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
            array_position: Some(LatLngHeight::new_mwa()),
            dut1: metafits_context
                .dut1
                .map(|dut1| Duration::from_f64(dut1, Unit::Second)),
            tile_names,
            tile_xyzs,
            flagged_tiles,
            unavailable_tiles: vec![],
            autocorrelations_present: true,
            dipole_delays: Some(dipole_delays),
            dipole_gains: Some(dipole_gains),
            time_res,
            coarse_chan_nums,
            coarse_chan_freqs,
            num_fine_chans_per_coarse_chan: metafits_context.num_corr_fine_chans_per_coarse,
            freq_res: Some(metafits_context.corr_fine_chan_width_hz as f64),
            fine_chan_freqs,
            flagged_fine_chans,
            flagged_fine_chans_per_coarse_chan,
        };

        let all_baseline_tile_pairs = metafits_context
            .baselines
            .iter()
            .map(|bl| (bl.ant1_index, bl.ant2_index))
            .collect();

        Ok(RawDataReader {
            obs_context,
            mwalib_context,
            pfb_gains,
            corrections,
            all_baseline_tile_pairs,
            mwaf_flags,
        })
    }

    /// From the given mwaf flags for a timestep, make weights negative
    /// (flagged) or leave them as is.
    fn apply_mwaf_flags(
        &self,
        mwaf_timestep: Option<usize>,
        gpubox_channels: Range<usize>,
        mut weights: ArrayViewMut2<f32>,
    ) {
        if let Some(timestep) = mwaf_timestep {
            for (i_gpubox_chan, gpubox_channel) in gpubox_channels.into_iter().enumerate() {
                let flags = self.mwaf_flags.as_ref().unwrap().flags[&(gpubox_channel as u8)]
                    .slice(s![timestep, .., ..,]);
                // Select only the applicable frequencies.
                let n = self.obs_context.num_fine_chans_per_coarse_chan;
                let selection = s![
                    (i_gpubox_chan * n)..((i_gpubox_chan + 1) * n),
                    .. // All baselines
                ];

                weights
                    .slice_mut(selection)
                    .outer_iter_mut()
                    .enumerate()
                    .for_each(|(i_freq, mut weights)| {
                        // Get the right bit for this frequency channel.
                        let bit = i_freq % 8;
                        // Get the right flags for this frequency channel. There
                        // are 8 flags per value in flags (an 8-bit byte).
                        let flags_for_bls = flags
                            .slice(s![.., i_freq / 8])
                            .mapv(|f| ((f.reverse_bits() >> bit) & 0x01) == 0x01);

                        weights
                            .iter_mut()
                            .zip(flags_for_bls)
                            .for_each(|(weight, flag)| {
                                if flag {
                                    *weight = -weight.abs();
                                }
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
    ) -> Result<(), VisReadError> {
        // TODO: Handle non-contiguous coarse channels.

        // Check that mwaf flags are available for this timestep.
        let mwaf_timestep = match &self.mwaf_flags {
            Some(mwaf_flags) => {
                // The time resolution is always specified for raw MWA data.
                let time_res = self.obs_context.time_res.unwrap();
                let flags_start = mwaf_flags.start_time;
                let flags_end = flags_start + mwaf_flags.num_time_steps as f64 * time_res;
                let timestamp = self.obs_context.timestamps[timestep];
                if !(flags_start..flags_end).contains(&timestamp) {
                    return Err(VisReadError::MwafFlagsMissingForTimestep {
                        timestep,
                        gps: timestamp.as_gpst_seconds(),
                    });
                }
                // Find the flags timestep index corresponding to the data
                // timestep index.
                let offset = (timestamp - flags_start).in_seconds() / time_res.in_seconds();
                let offset = if mwaf_flags.offset_bug {
                    offset.floor() as usize
                } else {
                    offset.round() as usize
                };

                debug!("timestep {timestep}, mwaf timestep {offset}");
                Some(offset)
            }
            None => None,
        };

        // We checked mwalib has some common good coarse channels in the new
        // method, so it is safe to unwrap `first` and `last`.
        let coarse_chan_indices = &self.mwalib_context.common_good_coarse_chan_indices;
        let vis_sel = VisSelection {
            timestep_range: timestep..timestep + 1,
            coarse_chan_range: *coarse_chan_indices.first().unwrap()
                ..*coarse_chan_indices.last().unwrap() + 1,
            baseline_idxs: (0..self.all_baseline_tile_pairs.len()).collect(),
        };

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

        // Read in the data via Birli.
        let fine_chans_per_coarse = self
            .mwalib_context
            .metafits_context
            .num_corr_fine_chans_per_coarse;
        let mut jones_array = vis_sel.allocate_jones(fine_chans_per_coarse)?;
        let mut flag_array = vis_sel.allocate_flags(fine_chans_per_coarse)?;
        let mut weight_array = vis_sel.allocate_weights(fine_chans_per_coarse)?;
        vis_sel.read_mwalib(
            &self.mwalib_context,
            jones_array.view_mut(),
            flag_array.view_mut(),
            false,
        )?;

        let weight_factor = birli::flags::get_weight_factor(&self.mwalib_context);
        // populate weights
        weight_array.fill(weight_factor as _);

        // Correct the raw data.
        let metafits_context = &self.mwalib_context.metafits_context;

        let prep_ctx = PreprocessContext {
            array_pos: self
                .obs_context
                .array_position
                .unwrap_or_else(LatLngHeight::new_mwa),
            phase_centre: self.obs_context.phase_centre,
            correct_cable_lengths: self.corrections.cable_length
                && match metafits_context.cable_delays_applied {
                    mwalib::CableDelaysApplied::NoCableDelaysApplied => true,
                    mwalib::CableDelaysApplied::CableAndRecClock
                    | mwalib::CableDelaysApplied::CableAndRecClockAndBeamformerDipoleDelays => {
                        false
                    }
                },
            correct_digital_gains: self.corrections.digital_gains,
            passband_gains: self.pfb_gains,
            correct_geometry: self.corrections.geometric
                && matches!(
                    metafits_context.geometric_delays_applied,
                    GeometricDelaysApplied::No
                ),
            ..PreprocessContext::default()
        };

        prep_ctx.preprocess(
            &self.mwalib_context,
            jones_array.view_mut(),
            weight_array.view_mut(),
            flag_array.view_mut(),
            &vis_sel,
        )?;

        // Remove the extraneous time dimension; there's only ever one timestep.
        let vis = jones_array.remove_axis(Axis(0));

        // bake flags into weights
        for (weight, flag) in izip!(weight_array.iter_mut(), flag_array.iter()) {
            *weight = if *flag {
                -(*weight).abs()
            } else {
                (*weight).abs()
            } as f32;
        }

        let mut weights = weight_array.remove_axis(Axis(0));

        self.apply_mwaf_flags(mwaf_timestep, gpubox_channels, weights.view_mut());

        // If applicable, write the cross-correlation visibilities to our
        // `data_array`, ignoring any flagged baselines.
        if let Some(CrossData {
            mut data_array,
            mut weights_array,
            tile_baseline_flags,
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
                            let (tile1, tile2) =
                                baseline_to_tiles(metafits_context.num_ants, *i_baseline);
                            tile_baseline_flags
                                .tile_to_unflagged_cross_baseline_map
                                .contains_key(&(tile1, tile2))
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
        // `data_array`, ignoring any flagged tiles.
        if let Some(AutoData {
            mut data_array,
            mut weights_array,
            tile_baseline_flags,
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
                            let (tile1, tile2) =
                                baseline_to_tiles(metafits_context.num_ants, *i_baseline);
                            tile1 == tile2 && !tile_baseline_flags.flagged_tiles.contains(&tile1)
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

impl VisRead for RawDataReader {
    fn get_obs_context(&self) -> &ObsContext {
        &self.obs_context
    }

    fn get_input_data_type(&self) -> VisInputType {
        VisInputType::Raw
    }

    fn get_metafits_context(&self) -> Option<&MetafitsContext> {
        Some(&self.mwalib_context.metafits_context)
    }

    fn get_flags(&self) -> Option<&MwafFlags> {
        self.mwaf_flags.as_ref()
    }

    fn read_crosses_and_autos(
        &self,
        cross_data_array: ArrayViewMut2<Jones<f32>>,
        cross_weights_array: ArrayViewMut2<f32>,
        auto_data_array: ArrayViewMut2<Jones<f32>>,
        auto_weights_array: ArrayViewMut2<f32>,
        timestep: usize,
        tile_baseline_flags: &TileBaselineFlags,
        flagged_fine_chans: &HashSet<usize>,
    ) -> Result<(), VisReadError> {
        self.read_inner(
            Some(CrossData {
                data_array: cross_data_array,
                weights_array: cross_weights_array,
                tile_baseline_flags,
            }),
            Some(AutoData {
                data_array: auto_data_array,
                weights_array: auto_weights_array,
                tile_baseline_flags,
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
        tile_baseline_flags: &TileBaselineFlags,
        flagged_fine_chans: &HashSet<usize>,
    ) -> Result<(), VisReadError> {
        self.read_inner(
            Some(CrossData {
                data_array,
                weights_array,
                tile_baseline_flags,
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
        tile_baseline_flags: &TileBaselineFlags,
        flagged_fine_chans: &HashSet<usize>,
    ) -> Result<(), VisReadError> {
        self.read_inner(
            None,
            Some(AutoData {
                data_array,
                weights_array,
                tile_baseline_flags,
            }),
            timestep,
            flagged_fine_chans,
        )
    }
}
