// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Code to handle reading from raw MWA files.

mod error;
pub(crate) mod pfb_gains;
#[cfg(test)]
mod tests;

pub(crate) use error::RawReadError;

use std::{
    collections::HashSet,
    fmt::Debug,
    num::NonZeroU16,
    ops::Range,
    path::{Path, PathBuf},
};

use birli::PreprocessContext;
use hifitime::{Duration, Epoch};
use itertools::Itertools;
use log::{debug, trace};
use marlu::{math::baseline_to_tiles, Jones, LatLngHeight, RADec, VisSelection, XyzGeodetic};
use mwalib::{
    CorrelatorContext, GeometricDelaysApplied, GpuboxError, MWAVersion, MetafitsContext,
    MwalibError, Pol,
};
use ndarray::prelude::*;
use vec1::Vec1;

use super::{AutoData, CrossData, MarluMwaObsContext, VisInputType, VisRead, VisReadError};
use crate::{
    beam::Delays,
    cli::Warn,
    context::ObsContext,
    flagging::{MwafFlags, MwafProducer},
    metafits,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RawDataCorrections {
    /// What 'flavour' are the PFB gains?
    pub(crate) pfb_flavour: pfb_gains::PfbFlavour,

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
    ) -> Result<RawDataCorrections, pfb_gains::PfbParseError> {
        Ok(Self {
            pfb_flavour: match pfb_flavour {
                None => pfb_gains::DEFAULT_PFB_FLAVOUR,
                Some(s) => pfb_gains::PfbFlavour::parse(s)?,
            },
            digital_gains,
            cable_length,
            geometric,
        })
    }

    fn nothing_to_do(self) -> bool {
        let Self {
            pfb_flavour,
            digital_gains,
            cable_length,
            geometric,
        } = self;
        matches!(pfb_flavour, pfb_gains::PfbFlavour::None)
            && !digital_gains
            && !cable_length
            && !geometric
    }

    /// Return a [`RawDataCorrections`] that won't do any corrections.
    pub fn do_nothing() -> RawDataCorrections {
        RawDataCorrections {
            pfb_flavour: pfb_gains::PfbFlavour::None,
            digital_gains: false,
            cable_length: false,
            geometric: false,
        }
    }
}

impl Default for RawDataCorrections {
    fn default() -> Self {
        Self {
            pfb_flavour: pfb_gains::DEFAULT_PFB_FLAVOUR,
            digital_gains: true,
            cable_length: true,
            geometric: true,
        }
    }
}

/// Raw MWA data, i.e. gpubox files.
pub struct RawDataReader {
    /// Observation metadata.
    obs_context: ObsContext,

    /// The interface to the raw data via mwalib.
    mwalib_context: CorrelatorContext,

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
    pub fn new(
        metafits: &Path,
        gpuboxes: &[PathBuf],
        mwafs: Option<&[PathBuf]>,
        corrections: RawDataCorrections,
        array_position: Option<LatLngHeight>,
    ) -> Result<RawDataReader, RawReadError> {
        // There are a lot of unwraps in this function. These are fine because
        // mwalib ensures that vectors aren't empty so when we convert a Vec to
        // Vec1, for example, we don't need to propagate a new error.

        trace!("Using metafits: {}", metafits.display());
        trace!("Using gpubox files: {:#?}", gpuboxes);

        trace!("Creating mwalib context");
        let mwalib_context = crate::misc::expensive_op(
            || CorrelatorContext::new(metafits, gpuboxes).map_err(Box::new),
            "Still waiting to inspect all gpubox metadata",
        )?;
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
            "All of this observation's tiles are flagged".warn();
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
            "This observation has been flagged as \"do not use\", according to the metafits delays!".warn();
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

        let time_res = {
            let int_time_ns = i128::from(metafits_context.corr_int_time_ms)
                .checked_mul(1_000_000)
                .expect("does not overflow i128");
            Duration::from_total_nanoseconds(int_time_ns)
        };

        let all_timesteps = Vec1::try_from_vec(mwalib_context.provided_timestep_indices.clone())
            .map_err(|_| RawReadError::NoTimesteps)?;
        let timestamps: Vec<Epoch> = mwalib_context
            .timesteps
            .iter()
            .map(|t| {
                let gps_nanoseconds = t
                    .gps_time_ms
                    .checked_mul(1_000_000)
                    .expect("does not overflow u64");
                Epoch::from_gpst_nanoseconds(gps_nanoseconds) + time_res / 2
            })
            .collect();
        let timestamps = Vec1::try_from_vec(timestamps).map_err(|_| RawReadError::NoTimesteps)?;

        // Use the "common good" coarse channels. If there aren't any, complain.
        let coarse_chan_indices = &mwalib_context.common_good_coarse_chan_indices;
        if coarse_chan_indices.is_empty() {
            return Err(RawReadError::NoGoodCoarseChannels);
        }
        let mut coarse_chan_nums: Vec<u32> = mwalib_context
            .coarse_chans
            .iter()
            .enumerate()
            .filter(|(i, _)| coarse_chan_indices.contains(i))
            .map(|(_, cc)| {
                cc.rec_chan_number
                    .try_into()
                    .expect("not larger than u32::MAX")
            })
            .collect();
        coarse_chan_nums.sort_unstable();
        debug!("MWA coarse channel numbers: {coarse_chan_nums:?}");
        let mwa_coarse_chan_nums =
            Vec1::try_from_vec(coarse_chan_nums).expect("MWA data always has coarse channel info");

        let num_corr_fine_chans_per_coarse = NonZeroU16::new(
            metafits_context
                .num_corr_fine_chans_per_coarse
                .try_into()
                .expect("is smaller than u16::MAX"),
        )
        .expect("never 0");
        let flagged_fine_chans_per_coarse_chan = get_80khz_fine_chan_flags_per_coarse_chan(
            metafits_context.corr_fine_chan_width_hz,
            num_corr_fine_chans_per_coarse,
            is_mwax,
        );
        // Given the provided "common good" coarse channels, find the missing
        // coarse channels and flag their channels.
        let mut missing_coarse_chans = Vec::with_capacity(24);
        let coarse_chan_span = *coarse_chan_indices
            .first()
            .expect("at least one coarse channel provided")
            ..=*coarse_chan_indices
                .last()
                .expect("at least one coarse channel provided");
        for i_cc in coarse_chan_span.clone() {
            if !coarse_chan_indices.contains(&i_cc) {
                missing_coarse_chans.push(i_cc);
            }
        }

        let mut flagged_fine_chans = Vec::with_capacity(
            flagged_fine_chans_per_coarse_chan.len() * coarse_chan_indices.len()
                + missing_coarse_chans.len() * metafits_context.num_corr_fine_chans_per_coarse,
        );
        for i_cc in coarse_chan_span.clone() {
            if missing_coarse_chans.contains(&i_cc) {
                for f in 0..num_corr_fine_chans_per_coarse.get() {
                    // The flagged channels are relative to the start of the
                    // frequency band we're interested in. So if this is the
                    // first coarse channel we're interested in, the flags
                    // should start from 0, not wherever the coarse channel sits
                    // within the whole observation band.
                    flagged_fine_chans.push(
                        (i_cc - *coarse_chan_span.start()) as u16
                            * num_corr_fine_chans_per_coarse.get()
                            + f,
                    );
                }
            } else {
                for &f in &flagged_fine_chans_per_coarse_chan {
                    flagged_fine_chans.push(
                        (i_cc - *coarse_chan_span.start()) as u16
                            * num_corr_fine_chans_per_coarse.get()
                            + f,
                    );
                }
            }
        }

        let fine_chan_freqs = mwalib_context
            .get_fine_chan_freqs_hz_array(&coarse_chan_span.collect::<Vec<_>>())
            .into_iter()
            .map(|f| f.round() as u64)
            .collect();
        let fine_chan_freqs = Vec1::try_from_vec(fine_chan_freqs).unwrap();

        let phase_centre = RADec::from_degrees(
            metafits_context.ra_phase_center_degrees.unwrap_or_else(|| {
                "No phase centre specified; using the pointing centre as the phase centre".warn();
                metafits_context.ra_tile_pointing_degrees
            }),
            metafits_context
                .dec_phase_center_degrees
                .unwrap_or(metafits_context.dec_tile_pointing_degrees),
        );
        let pointing_centre = Some(RADec::from_degrees(
            metafits_context.ra_tile_pointing_degrees,
            metafits_context.dec_tile_pointing_degrees,
        ));
        let supplied_array_position = LatLngHeight::mwa();
        let array_position = array_position.unwrap_or(supplied_array_position);
        let tile_xyzs = XyzGeodetic::get_tiles(metafits_context, array_position.latitude_rad);
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
            let mut f = MwafFlags::new_from_mwafs(m).map_err(Box::new)?;

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
            let data_start = {
                let gps_nanoseconds = mwalib_context
                    .common_start_gps_time_ms
                    .checked_mul(1_000_000)
                    .expect("does not overflow u64");
                Epoch::from_gpst_nanoseconds(gps_nanoseconds) + time_res / 2
            };
            let data_end = {
                let gps_nanoseconds = mwalib_context
                    .common_end_gps_time_ms
                    .checked_mul(1_000_000)
                    .expect("does not overflow u64");
                Epoch::from_gpst_nanoseconds(gps_nanoseconds) + time_res / 2
            };
            let flags_start = f.start_time;
            let flags_end = flags_start + f.num_time_steps as f64 * time_res;
            let diff = (flags_start - data_start).to_seconds() / time_res.to_seconds();
            debug!("Data start time (GPS): {}", data_start.to_gpst_seconds());
            debug!("Flag start time (GPS): {}", flags_start.to_gpst_seconds());
            debug!("(flags_start - data_start).to_seconds() / time_res.to_seconds(): {diff}");
            if diff.fract().abs() > 0.0 {
                let mut block = vec!["These mwaf files do not have times corresponding to the data they were created from.".into()];
                match f.software {
                    MwafProducer::Cotter => block.push(
                        "This is a Cotter bug. You should probably use Birli to make new flags."
                            .into(),
                    ),
                    MwafProducer::Birli => {
                        block.push("These mwafs were made by Birli. Please file an issue!".into())
                    }
                    MwafProducer::Unknown => {
                        block.push("Unknown software made these mwafs.".into())
                    }
                }
                block.warn();
                f.offset_bug = true;
            }

            // Warn the user if there are fewer timesteps in the mwaf flags than
            // there are in the raw data. This is good for signalling to the
            // user that attempting to use data timesteps without flag timesteps
            // will be a problem.
            let mut start_offset =
                ((data_start - flags_start).to_seconds() / time_res.to_seconds()).ceil();
            let mut end_offset =
                ((data_end - flags_end).to_seconds() / time_res.to_seconds()).ceil();
            if f.offset_bug {
                start_offset -= 1.0;
                end_offset -= 1.0;
            }
            if start_offset > 0.0 || end_offset > 0.0 {
                let mut block = vec!["Not all MWA data timesteps have mwaf flags available".into()];
                match (start_offset > 0.0, end_offset > 0.0) {
                    (true, true) => block.push(
                        format!(
                            "{} timesteps at the start and {} at the end are not represented",
                            start_offset, end_offset,
                        )
                        .into(),
                    ),
                    (true, false) => block.push(
                        format!(
                            "{} timesteps at the start are not represented",
                            start_offset,
                        )
                        .into(),
                    ),
                    (false, true) => block.push(
                        format!("{} timesteps at the end are not represented", end_offset).into(),
                    ),
                    (false, false) => unreachable!(),
                }
                block.warn();
            }

            Some(f)
        } else {
            None
        };

        let obs_context = ObsContext {
            input_data_type: VisInputType::Raw,
            obsid: Some(metafits_context.obs_id),
            timestamps,
            all_timesteps,
            unflagged_timesteps: mwalib_context.common_good_timestep_indices.clone(),
            phase_centre,
            pointing_centre,
            array_position,
            supplied_array_position,
            dut1: metafits_context.dut1.map(Duration::from_seconds),
            tile_names,
            tile_xyzs,
            flagged_tiles,
            unavailable_tiles: vec![],
            autocorrelations_present: true,
            dipole_delays: Some(dipole_delays),
            dipole_gains: Some(dipole_gains),
            time_res: Some(time_res),
            mwa_coarse_chan_nums: Some(mwa_coarse_chan_nums),
            num_fine_chans_per_coarse_chan: Some(num_corr_fine_chans_per_coarse),
            freq_res: Some(metafits_context.corr_fine_chan_width_hz as f64),
            fine_chan_freqs,
            flagged_fine_chans,
            flagged_fine_chans_per_coarse_chan: Vec1::try_from_vec(
                flagged_fine_chans_per_coarse_chan,
            )
            .ok(),
            polarisations: crate::context::Polarisations::XX_XY_YX_YY,
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
                let n = usize::from(
                    self.obs_context
                        .num_fine_chans_per_coarse_chan
                        .expect("raw MWA data always specifies this")
                        .get(),
                );
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
                            .zip_eq(flags_for_bls)
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
    pub fn read_inner(
        &self,
        crosses: Option<CrossData>,
        autos: Option<AutoData>,
        timestep: usize,
        flagged_fine_chans: &HashSet<u16>,
    ) -> Result<(), RawReadError> {
        // Check that mwaf flags are available for this timestep.
        let mwaf_timestep = match &self.mwaf_flags {
            Some(mwaf_flags) => {
                // The time resolution is always specified for raw MWA data.
                let time_res = self.obs_context.time_res.unwrap();
                let flags_start = mwaf_flags.start_time;
                let flags_end = flags_start + mwaf_flags.num_time_steps as f64 * time_res;
                let timestamp = self.obs_context.timestamps[timestep];
                if !(flags_start..flags_end).contains(&timestamp) {
                    return Err(RawReadError::MwafFlagsMissingForTimestep {
                        timestep,
                        gps: timestamp.to_gpst_seconds(),
                    });
                }
                // Find the flags timestep index corresponding to the data
                // timestep index.
                let offset = (timestamp - flags_start).to_seconds() / time_res.to_seconds();
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

        let gpubox_channels = self.mwalib_context.gpubox_batches[0]
            .gpubox_files
            .first()
            .expect("at least one coarse channel provided")
            .channel_identifier
            ..self.mwalib_context.gpubox_batches[0]
                .gpubox_files
                .last()
                .expect("at least one coarse channel provided")
                .channel_identifier
                + 1;
        let coarse_chan_indices = &self.mwalib_context.common_good_coarse_chan_indices;
        let coarse_chan_range = *coarse_chan_indices
            .first()
            .expect("at least one coarse channel provided")
            ..*coarse_chan_indices
                .last()
                .expect("at least one coarse channel provided")
                + 1;

        // Read in the data via mwalib.
        let metafits_context = &self.mwalib_context.metafits_context;
        let fine_chans_per_coarse = metafits_context.num_corr_fine_chans_per_coarse;
        let size = fine_chans_per_coarse * metafits_context.num_baselines;
        let full_size = size * gpubox_channels.len();
        let mut jones_array_tfb = vec![Jones::default(); full_size];
        let mut flag_array_tfb = vec![false; full_size];

        for ((jones_array_fb, flag_array_fb), i_cc) in jones_array_tfb
            .chunks_exact_mut(size)
            .zip_eq(flag_array_tfb.chunks_exact_mut(size))
            .zip_eq(coarse_chan_range.clone())
        {
            // Skip unavailable coarse channels.
            if !coarse_chan_indices.contains(&i_cc) {
                continue;
            }

            // Cast the Jones slice to a float slice so mwalib can use it.
            let jones_array_fb = unsafe {
                let ptr = jones_array_fb.as_mut_ptr();
                std::slice::from_raw_parts_mut(ptr.cast(), jones_array_fb.len() * 8)
            };

            match self
                .mwalib_context
                .read_by_frequency_into_buffer(timestep, i_cc, jones_array_fb)
            {
                Ok(()) => (),

                Err(GpuboxError::NoDataForTimeStepCoarseChannel {
                    timestep_index,
                    coarse_chan_index,
                }) => {
                    format!(
                        "Flagging missing data at timestep {timestep_index}, coarse channel {coarse_chan_index}"
                    ).warn();
                    flag_array_fb.fill(true);
                }

                Err(e) => return Err(RawReadError::from(Box::new(MwalibError::from(e)))),
            }
        }

        let shape = (
            1,
            fine_chans_per_coarse * gpubox_channels.len(),
            metafits_context.num_baselines,
        );
        let mut jones_array_tfb =
            Array3::from_shape_vec(shape, jones_array_tfb).expect("correct shape");
        let mut flag_array_tfb =
            Array3::from_shape_vec(shape, flag_array_tfb).expect("correct shape");
        let mut weight_array_tfb = Array3::from_elem(
            shape,
            birli::flags::get_weight_factor(&self.mwalib_context) as f32,
        );

        // Correct the raw data.
        if !self.corrections.nothing_to_do() {
            let prep_ctx = PreprocessContext {
                array_pos: self.obs_context.array_position,
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
            let vis_sel = VisSelection {
                timestep_range: timestep..timestep + 1,
                coarse_chan_range,
                baseline_idxs: (0..self.all_baseline_tile_pairs.len()).collect(),
            };
            prep_ctx
                .preprocess(
                    &self.mwalib_context,
                    jones_array_tfb.view_mut(),
                    weight_array_tfb.view_mut(),
                    flag_array_tfb.view_mut(),
                    &vis_sel,
                )
                .map_err(Box::new)?;
        }

        // Convert the data array into a vector so we can use `chunks_exact`
        // below (this is 0 cost). This is measurably faster than using
        // `outer_iter` on an `ndarray`. Ignore the time dimension; there's only
        // ever one timestep.
        let (_, num_chans, num_baselines) = jones_array_tfb.dim();
        let (data_vis_fb, data_vis_offset) = jones_array_tfb.into_raw_vec_and_offset();
        assert!(data_vis_offset.unwrap_or(0) == 0);

        // bake flags into weights
        for (weight, flag) in weight_array_tfb.iter_mut().zip_eq(flag_array_tfb) {
            *weight = if flag {
                -(*weight).abs()
            } else {
                (*weight).abs()
            };
        }

        let mut data_weights_fb = weight_array_tfb.remove_axis(Axis(0));
        self.apply_mwaf_flags(mwaf_timestep, gpubox_channels, data_weights_fb.view_mut());
        let (data_weights_fb, data_weights_offset) = data_weights_fb.into_raw_vec_and_offset();
        assert!(data_weights_offset.unwrap_or(0) == 0);

        let chan_flags = (0..num_chans)
            .map(|i_chan| flagged_fine_chans.contains(&(i_chan as u16)))
            .collect::<Vec<_>>();

        // If applicable, write the cross-correlation visibilities to our
        // `data_array`, ignoring any flagged baselines.
        if let Some(CrossData {
            mut vis_fb,
            mut weights_fb,
            tile_baseline_flags,
        }) = crosses
        {
            let baseline_flags = (0..num_baselines)
                .map(|i_bl| {
                    let (tile1, tile2) = baseline_to_tiles(metafits_context.num_ants, i_bl);
                    (tile1 == tile2)
                        || !tile_baseline_flags
                            .tile_to_unflagged_cross_baseline_map
                            .contains_key(&(tile1, tile2))
                })
                .collect::<Vec<_>>();
            let num_unflagged_baselines = tile_baseline_flags
                .tile_to_unflagged_cross_baseline_map
                .len();

            data_vis_fb
                .chunks_exact(num_baselines)
                .zip(data_weights_fb.chunks_exact(num_baselines))
                // Let only unflagged channels proceed.
                .zip(chan_flags.iter())
                .filter(|((_, _), &flag)| !flag)
                // Discard the flag and then zip with the outgoing array.
                .map(|((data, weights), _)| (data, weights))
                .zip(
                    vis_fb
                        .as_slice_mut()
                        .expect("is_contiguous")
                        .chunks_exact_mut(num_unflagged_baselines),
                )
                .zip(
                    weights_fb
                        .as_slice_mut()
                        .expect("is_contiguous")
                        .chunks_exact_mut(num_unflagged_baselines),
                )
                .for_each(|(((data_vis_b, data_weights_b), vis_b), weight_b)| {
                    data_vis_b
                        .iter()
                        .zip_eq(data_weights_b)
                        // Let only unflagged baselines proceed.
                        .zip_eq(baseline_flags.iter())
                        .filter(|((_, _), &bl_flag)| !bl_flag)
                        // Discard the baseline flag and then zip with the outgoing array.
                        .map(|(data, _)| data)
                        .zip_eq(vis_b.iter_mut())
                        .zip_eq(weight_b.iter_mut())
                        .for_each(|(((data_vis, data_weight), vis), weight)| {
                            *vis = *data_vis;
                            *weight = *data_weight;
                        });
                });
        }

        // If applicable, write the auto-correlation visibilities to our
        // `data_array`, ignoring any flagged tiles.
        if let Some(AutoData {
            mut vis_fb,
            mut weights_fb,
            tile_baseline_flags,
        }) = autos
        {
            let baseline_flags = (0..num_baselines)
                .map(|i_bl| {
                    let (tile1, tile2) = baseline_to_tiles(metafits_context.num_ants, i_bl);
                    (tile1 != tile2) || tile_baseline_flags.flagged_tiles.contains(&tile1)
                })
                .collect::<Vec<_>>();
            let num_unflagged_tiles = vis_fb.len_of(Axis(1));

            data_vis_fb
                .chunks_exact(num_baselines)
                .zip(data_weights_fb.chunks_exact(num_baselines))
                // Let only unflagged channels proceed.
                .zip(chan_flags.iter())
                .filter(|((_, _), &flag)| !flag)
                // Discard the flag and then zip with the outgoing array.
                .map(|((data, weights), _)| (data, weights))
                .zip_eq(
                    vis_fb
                        .as_slice_mut()
                        .expect("is contiguous")
                        .chunks_exact_mut(num_unflagged_tiles),
                )
                .zip_eq(
                    weights_fb
                        .as_slice_mut()
                        .expect("is contiguous")
                        .chunks_exact_mut(num_unflagged_tiles),
                )
                .for_each(|(((data_vis_b, data_weight_b), vis_b), weights_b)| {
                    data_vis_b
                        .iter()
                        .zip_eq(data_weight_b)
                        // Let only unflagged baselines proceed.
                        .zip_eq(baseline_flags.iter())
                        .filter(|((_, _), &bl_flag)| !bl_flag)
                        // Discard the baseline flag and then zip with the outgoing array.
                        .map(|(data, _)| data)
                        .zip_eq(vis_b.iter_mut())
                        .zip_eq(weights_b.iter_mut())
                        .for_each(|(((data_vis, data_weight), vis), weight)| {
                            *vis = *data_vis;
                            *weight = *data_weight;
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

    fn get_raw_data_corrections(&self) -> Option<RawDataCorrections> {
        Some(self.corrections)
    }

    fn set_raw_data_corrections(&mut self, corrections: RawDataCorrections) {
        self.corrections = corrections;
    }

    fn read_inner_dispatch(
        &self,
        cross_data: Option<CrossData>,
        auto_data: Option<AutoData>,
        timestep: usize,
        flagged_fine_chans: &HashSet<u16>,
    ) -> Result<(), VisReadError> {
        self.read_inner(cross_data, auto_data, timestep, flagged_fine_chans)?;
        Ok(())
    }

    fn get_marlu_mwa_info(&self) -> Option<MarluMwaObsContext> {
        Some(MarluMwaObsContext::from_mwalib(
            &self.mwalib_context.metafits_context,
        ))
    }
}

fn get_80khz_fine_chan_flags_per_coarse_chan(
    fine_chan_width: u32,
    num_fine_chans_per_coarse_chan: NonZeroU16,
    is_mwax: bool,
) -> Vec<u16> {
    let mut flags = vec![];

    // Any fractional parts are discarded, meaning e.g. if the resolution was
    // 79kHz per channel, only 1 edge channel is flagged rather than 2.
    let num_flagged_fine_chans_per_edge = (80000 / fine_chan_width)
        .try_into()
        .expect("smaller than u16::MAX");
    for i in 0..num_flagged_fine_chans_per_edge {
        flags.push(i);
        flags.push(
            (num_fine_chans_per_coarse_chan.get() - 1)
                .checked_sub(i)
                .expect("algorithm is sound"),
        );
    }
    // Also put the centre channel in if this isn't an MWAX obs.
    if !is_mwax {
        flags.push(num_fine_chans_per_coarse_chan.get() / 2);
    }
    flags.sort_unstable();
    flags
}
