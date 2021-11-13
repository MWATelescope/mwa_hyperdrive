// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Parameters required for calibration and associated functions.
//!
//! Strategy: Users give arguments to hyperdrive (handled by calibrate::args).
//! hyperdrive turns arguments into parameters (handled by calibrate::params). Using
//! this terminology, the code to handle arguments and parameters (and associated
//! errors) can be neatly split.

pub(crate) mod error;
pub(crate) mod freq;

mod filenames;

pub(crate) use error::*;
use filenames::InputDataTypes;
pub(crate) use freq::*;

use std::collections::{HashMap, HashSet};
use std::fs::OpenOptions;
use std::ops::Deref;
use std::path::{Path, PathBuf};
use std::str::FromStr;

use itertools::Itertools;
use log::{debug, info, log_enabled, trace, warn, Level::Debug};
use mwa_rust_core::{
    constants::{MWA_LAT_RAD, MWA_LONG_RAD},
    pos::precession::{precess_time, PrecessionInfo},
    time::epoch_as_gps_seconds,
    XyzGeodetic,
};
use rayon::prelude::*;

use super::solutions::CalSolutionType;
use crate::{
    constants::*, data_formats::VisOutputType, data_formats::*, glob::*, math::TileBaselineMaps,
};
use mwa_hyperdrive_beam::{create_fee_beam_object, create_no_beam_object, Beam, Delays};
use mwa_hyperdrive_srclist::{
    constants::*, veto_sources, FluxDensityType, SourceList, SourceListType,
};

/// Parameters needed to perform calibration.
pub struct CalibrateParams {
    /// Interface to the MWA data, and metadata on the input data.
    // pub(crate) input_data: Box<dyn InputData>,
    pub(crate) input_data: Box<dyn InputData>,

    /// Beam object.
    pub(crate) beam: Box<dyn Beam>,

    /// The sky-model source list.
    pub(crate) source_list: SourceList,

    /// The optional sky-model visibilities file. If specified, it will be
    /// written to during calibration.
    pub(crate) model_file: Option<PathBuf>,

    /// Which tiles are flagged? This field contains flags that are
    /// user-specified as well as whatever was already flagged in the supplied
    /// data.
    ///
    /// These values correspond to those from the "Antenna" column in HDU 2 of
    /// the metafits file. Zero indexed.
    pub(crate) tile_flags: HashSet<usize>,

    /// Which baselines are flagged? These indices apply only to
    /// cross-correlation baselines.
    ///
    /// These values correspond to tiles derived from the "Antenna" column in
    /// HDU 2 of the metafits file. Zero indexed.
    pub(crate) baseline_flags: HashSet<usize>,

    /// Channel- and frequency-related parameters required for calibration.
    pub(crate) freq: FrequencyParams,

    /// The number of time samples to average together before calibrating.
    ///
    /// e.g. If the input data is in 0.5s resolution and this variable was 4,
    /// then we average 2s worth of data together during calibration.
    pub(crate) time_average_factor: usize,

    /// The timestep indices to use in calibration.
    pub(crate) timesteps: Vec<usize>,

    /// Given two antenna numbers, get the unflagged baseline number. e.g. If
    /// antenna 1 (i.e. the second antenna) is flagged, then the first baseline
    /// (i.e. 0) is between antenna 0 and antenna 2.
    ///
    /// This exists because some tiles may be flagged, so some baselines may be
    /// flagged.
    pub(crate) tile_to_unflagged_baseline_map: HashMap<(usize, usize), usize>,

    /// Given an unflagged baseline number, get the tile number pair that
    /// contribute to it. e.g. If tile 1 (i.e. the second tile) is flagged, then
    /// the second unflagged baseline (i.e. 0) is between tile 0 and tile 2 (the
    /// first is tile 0 and tile 0).
    ///
    /// This exists because some tiles may be flagged, so some baselines may be
    /// flagged.
    pub(crate) unflagged_baseline_to_tile_map: HashMap<usize, (usize, usize)>,

    /// Are auto-correlations being included?
    pub(crate) using_autos: bool,

    /// The names of the unflagged tiles.
    pub(crate) unflagged_tile_names: Vec<String>,

    /// The unflagged [XyzGeodetic] coordinates of each tile \[metres\]. This
    /// does not change over time; it is determined only by the telescope's tile
    /// layout.
    pub(crate) unflagged_tile_xyzs: Vec<XyzGeodetic>,

    /// The Earth longitude of the array \[radians\]. This is populated by user
    /// input or the input data.
    pub(crate) array_longitude: f64,

    /// The Earth latitude of the array \[radians\]. This is populated by user
    /// input or the input data.
    pub(crate) array_latitude: f64,

    /// The maximum number of times to iterate when performing "MitchCal".
    pub(crate) max_iterations: usize,

    /// The threshold at which we stop convergence when performing "MitchCal".
    /// This is smaller than `min_threshold`.
    pub(crate) stop_threshold: f64,

    /// The minimum threshold to satisfy convergence when performing "MitchCal".
    /// Reaching this threshold counts as "converged", but it's not as good as
    /// the stop threshold. This is bigger than `stop_threshold`.
    pub(crate) min_threshold: f64,

    /// The paths to the files where the calibration solutions are written. The
    /// same solutions are written to each file here, but the format may be
    /// different (indicated by the file extension). Supported formats are
    /// detailed by [super::solutions::CalSolutionType].
    pub(crate) output_solutions_filenames: Vec<(CalSolutionType, PathBuf)>,

    /// The paths to the files where calibrated visibilities are written. The
    /// same visibilities are written to each file here, but the format may be
    /// different (indicated by the file extension). Supported formats are
    /// detailed by [crate::data_formats::VisOutputType].
    pub(crate) output_vis_filenames: Vec<(VisOutputType, PathBuf)>,

    #[cfg(feature = "cuda")]
    /// If enabled, use the CPU to generate sky-model visibilites. Otherwise,
    /// use the GPU.
    pub(crate) cpu_vis_gen: bool,
}

impl CalibrateParams {
    /// Create a new params struct from arguments.
    ///
    /// If the time or frequency resolution aren't specified, they default to
    /// the observation's native resolution.
    ///
    /// Source list vetoing is performed in this function, using the specified
    /// number of sources and/or the veto threshold.
    pub(crate) fn new(
        super::args::CalibrateUserArgs {
            data,
            outputs,
            model_filename,
            beam_file,
            unity_dipole_gains,
            delays,
            no_beam,
            source_list,
            source_list_type,
            num_sources,
            source_dist_cutoff,
            veto_threshold,
            time_average_factor,
            // freq_average_factor,
            timesteps,
            tile_flags,
            ignore_input_data_tile_flags,
            ignore_input_data_fine_channels_flags,
            ignore_autos,
            fine_chan_flags_per_coarse_chan,
            fine_chan_flags,
            max_iterations,
            stop_thresh,
            min_thresh,
            array_longitude_deg,
            array_latitude_deg,
            #[cfg(feature = "cuda")]
            cpu,
        }: super::args::CalibrateUserArgs,
    ) -> Result<Self, InvalidArgsError> {
        let mut dipole_delays = match (delays, no_beam) {
            // Check that delays are sensible, regardless if we actually need
            // them.
            (Some(d), _) => {
                if d.len() != 16 || d.iter().any(|&v| v > 32) {
                    return Err(InvalidArgsError::BadDelays);
                }
                Delays::Partial(d)
            }

            // No delays were provided, but because we're not using beam code,
            // we don't need them.
            (None, true) => Delays::NotNecessary,

            // No delays were provided, but they'll be necessary eventually.
            // Other code should fail if no delays can be found.
            (None, false) => Delays::None,
        };

        // Handle input data. We expect one of three possibilities:
        // - gpubox files, a metafits file (and maybe mwaf files),
        // - a measurement set (and maybe a metafits file), or
        // - uvfits files.
        // If none or multiple of these possibilities are met, then we must fail.
        let input_data_types = match data {
            Some(strings) => InputDataTypes::new(&strings)?,
            None => return Err(InvalidArgsError::NoInputData),
        };
        let input_data: Box<dyn InputData> = match (
            input_data_types.metafits,
            input_data_types.gpuboxes,
            input_data_types.mwafs,
            input_data_types.ms,
            input_data_types.uvfits,
        ) {
            // Valid input for reading raw data.
            (Some(meta), Some(gpuboxes), mwafs, None, None) => {
                let input_data =
                    RawData::new(&meta, &gpuboxes, mwafs.as_deref(), &mut dipole_delays)?;

                // Print some high-level information.
                let obs_context = input_data.get_obs_context();
                // obsid is always present because we must have a metafits file;
                // unwrap is safe.
                info!("Calibrating obsid {}", obs_context.obsid.unwrap());
                info!("Using metafits: {}", meta.display());
                info!("Using {} gpubox files", gpuboxes.len());
                debug!("gpubox files: {:?}", &gpuboxes);
                match mwafs {
                    Some(_) => info!("Using supplied mwaf flags"),
                    None => warn!("No mwaf flags files supplied"),
                }
                Box::new(input_data)
            }

            // Valid input for reading a measurement set.
            (meta, None, None, Some(ms), None) => {
                let input_data = MS::new(&ms, meta.as_ref(), &mut dipole_delays)?;
                match input_data.get_obs_context().obsid {
                    Some(o) => info!(
                        "Calibrating obsid {} from measurement set {}",
                        o,
                        input_data.ms.canonicalize()?.display()
                    ),
                    None => info!(
                        "Calibrating measurement set {}",
                        input_data.ms.canonicalize()?.display()
                    ),
                }
                Box::new(input_data)
            }

            // Valid input for reading uvfits files.
            (meta, None, None, None, Some(uvfits_strs)) => {
                let input_data = match uvfits_strs.len() {
                    0 => panic!("whatcha doin?"),
                    1 => Uvfits::new(&uvfits_strs[0], meta.as_ref(), &mut dipole_delays)?,
                    _ => todo!(),
                };
                match input_data.get_obs_context().obsid {
                    Some(o) => info!(
                        "Calibrating obsid {} from uvfits {}",
                        o,
                        input_data.uvfits.canonicalize()?.display()
                    ),
                    None => info!(
                        "Calibrating uvfits {}",
                        input_data.uvfits.canonicalize()?.display()
                    ),
                }
                Box::new(input_data)
            }

            _ => return Err(InvalidArgsError::InvalidDataInput),
        };

        let obs_context = input_data.get_obs_context();
        let freq_context = input_data.get_freq_context();

        let beam: Box<dyn Beam> = if no_beam {
            create_no_beam_object(obs_context.tile_xyzs.len())
        } else {
            create_fee_beam_object(
                beam_file,
                obs_context.tile_xyzs.len(),
                dipole_delays,
                if unity_dipole_gains {
                    None
                } else {
                    obs_context.dipole_gains.clone()
                },
            )?
        };

        // Designate calibration outputs.
        let (output_solutions_filenames, output_vis_filenames) = {
            match outputs {
                // Defaults.
                None => {
                    let pb = PathBuf::from(DEFAULT_OUTPUT_SOLUTIONS_FILENAME);
                    let sol_type = pb
                        .extension()
                        .and_then(|os_str| os_str.to_str())
                        .and_then(|s| CalSolutionType::from_str(s).ok())
                        // Tests should pick up a bad default filename.
                        .expect("DEFAULT_OUTPUT_SOLUTIONS_FILENAME has an unhandled extension!");
                    (vec![(sol_type, pb)], vec![])
                }
                Some(outputs) => {
                    let mut cal_sols = vec![];
                    let mut vis_out = vec![];
                    for file in outputs {
                        // Is the output file type supported?
                        let ext = file.extension().and_then(|os_str| os_str.to_str());
                        match (
                            ext.and_then(|s| CalSolutionType::from_str(s).ok()),
                            ext.and_then(|s| VisOutputType::from_str(s).ok()),
                        ) {
                            (Some(sol_type), None) => {
                                can_write_to_file(&file)?;
                                cal_sols.push((sol_type, file));
                            },
                            (None, Some(vis_type)) => {
                                can_write_to_file(&file)?;
                                vis_out.push((vis_type, file));
                            },
                            (None, None) => return Err(InvalidArgsError::CalibrationOutputFile { ext: ext.unwrap_or("<no extension>").to_string()}),
                            (Some(_), Some(_)) => panic!("Programmer error: File extension '{}' is valid for both calibration solutions and visibility outputs, but this shouldn't be possible.", ext.unwrap()),
                        }
                    }
                    (cal_sols, vis_out)
                }
            }
        };
        if output_solutions_filenames.is_empty() && output_vis_filenames.is_empty() {
            return Err(InvalidArgsError::NoOutput);
        }

        // Handle the output model file, if specified.
        let model_file = match model_filename {
            None => None,
            Some(file) => {
                // Is the output file type supported?
                let ext = file.extension().and_then(|os_str| os_str.to_str());
                match ext.and_then(|s| VisOutputType::from_str(s).ok()) {
                    Some(_) => {
                        can_write_to_file(&file)?;
                        Some(file)
                    }
                    None => {
                        return Err(InvalidArgsError::VisFileType {
                            ext: ext.unwrap_or("<no extension>").to_string(),
                        })
                    }
                }
            }
        };

        let timesteps_to_use = {
            let input_data_unflagged_timesteps: Vec<usize> = obs_context
                .unflagged_timestep_indices
                .clone()
                .into_iter()
                .collect();
            match timesteps {
                None => input_data_unflagged_timesteps,
                Some(mut ts) => {
                    // Make sure there are no duplicates.
                    let timesteps_hashset: HashSet<&usize> = ts.iter().collect();
                    if timesteps_hashset.len() != ts.len() {
                        return Err(InvalidArgsError::DuplicateTimesteps);
                    }

                    // Ensure that all specified timesteps are actually available.
                    for t in &ts {
                        if !(0..obs_context.timesteps.len()).contains(t) {
                            return Err(InvalidArgsError::UnavailableTimestep {
                                got: *t,
                                last: obs_context.timesteps.len() - 1,
                            });
                        }
                    }

                    ts.sort_unstable();
                    ts
                }
            }
        };

        let array_longitude = match (array_longitude_deg, obs_context.array_longitude_rad) {
            (Some(array_longitude_deg), _) => array_longitude_deg.to_radians(),
            (None, Some(input_data_long)) => input_data_long,
            (None, None) => {
                warn!("Assuming that the input array is at the MWA Earth coordinates");
                MWA_LONG_RAD
            }
        };
        let array_latitude = match (array_latitude_deg, obs_context.array_latitude_rad) {
            (Some(array_latitude_deg), _) => array_latitude_deg.to_radians(),
            (None, Some(input_data_lat)) => input_data_lat,
            (None, None) => MWA_LAT_RAD,
        };

        // The length of the tile XYZ collection is the total number of tiles in
        // the array, even if some tiles are flagged.
        let total_num_tiles = obs_context.tile_xyzs.len();

        // Assign the tile flags.
        let mut tile_flags: HashSet<usize> = match tile_flags {
            Some(flags) => flags.into_iter().collect(),
            None => HashSet::new(),
        };
        if !ignore_input_data_tile_flags {
            // Add tiles that have already been flagged by the input data.
            for &obs_tile_flag in &obs_context.tile_flags {
                tile_flags.insert(obs_tile_flag);
            }
        }
        let baseline_flags = get_flagged_baselines_set(total_num_tiles, &tile_flags);

        // Assign the per-coarse-channel fine-channel flags.
        // TODO: Rename "coarse band" to "coarse channel".
        let mut fine_chan_flags_per_coarse_chan: HashSet<usize> =
            match fine_chan_flags_per_coarse_chan {
                Some(flags) => flags.into_iter().collect(),
                None => HashSet::new(),
            };
        if !ignore_input_data_fine_channels_flags {
            for &obs_fine_chan_flag in &obs_context.fine_chan_flags_per_coarse_chan {
                fine_chan_flags_per_coarse_chan.insert(obs_fine_chan_flag);
            }
        }

        // Determine all of the fine-channel flags.
        let mut fine_chan_flags: HashSet<usize> = match fine_chan_flags {
            Some(flags) => flags.into_iter().collect(),
            None => HashSet::new(),
        };
        for (i, _) in freq_context.coarse_chan_nums.iter().enumerate() {
            // Add the per-coarse-channel flags to the observation-wide flags.
            for f in &fine_chan_flags_per_coarse_chan {
                fine_chan_flags.insert(f + freq_context.num_fine_chans_per_coarse_chan * i);
            }
        }

        // Set up frequency information.
        let total_num_fine_channels = freq_context.fine_chan_freqs.len();
        let num_fine_chans_per_coarse_band =
            total_num_fine_channels / freq_context.coarse_chan_freqs.len();
        let num_unflagged_fine_chans_per_coarse_band =
            num_fine_chans_per_coarse_band - fine_chan_flags_per_coarse_chan.len();
        let mut unflagged_fine_chans = HashSet::new();
        let mut unflagged_fine_chan_freqs = vec![];
        for (&freq, chan_num) in freq_context
            .fine_chan_freqs
            .iter()
            .zip((0..freq_context.fine_chan_freqs.len()).into_iter())
        {
            if !fine_chan_flags.contains(&chan_num) {
                unflagged_fine_chans.insert(chan_num);
                unflagged_fine_chan_freqs.push(freq.round());
            }
        }

        // let freq_average_factor = match (freq_context.native_fine_chan_width, freq_average_factor) {
        //     (None, _) => {
        //         // If the input data has unknown frequency resolution, it's
        //         // because there's only one frequency.
        //         1
        //     }
        //     (Some(_), None) => {
        //         // "None" indicates we should follow default behaviour: Each
        //         // fine-frequency channel is calibrated independently.
        //         1
        //     }
        //     (_, Some(f)) => {
        //         // In all other cases, just use the input as is, but check that
        //         // it's not too big.
        //         if f > unflagged_fine_chans.len() {
        //             warn!(
        //                 "Cannot average {} channels; only {} are being used. Capping.",
        //                 f,
        //                 unflagged_fine_chans.len()
        //             );
        //             unflagged_fine_chans.len()
        //         } else {
        //             f
        //         }
        //     }
        // };

        let freq_struct = FrequencyParams {
            freq_average_factor: 1,
            num_fine_chans: total_num_fine_channels,
            num_unflagged_fine_chans_per_coarse_band,
            num_unflagged_fine_chans: num_unflagged_fine_chans_per_coarse_band
                * freq_context.coarse_chan_freqs.len(),
            unflagged_fine_chans,
            unflagged_fine_chan_freqs,
            fine_chan_flags,
        };

        // Print out some coordinates, including their precessed counterparts
        // (if applicable).
        let precession_info = precess_time(
            obs_context.phase_centre,
            obs_context.timesteps[0],
            array_longitude,
            array_latitude,
        );

        let mut source_list: SourceList = {
            // Handle the source list argument.
            let sl_pb: PathBuf = match source_list {
                None => return Err(InvalidArgsError::NoSourceList),
                Some(sl) => {
                    // If the specified source list file can't be found, treat
                    // it as a glob and expand it to find a match.
                    let pb = PathBuf::from(&sl);
                    if pb.exists() {
                        pb
                    } else {
                        get_single_match_from_glob(&sl)?
                    }
                }
            };

            // Read the source list file. If the type was manually specified,
            // use that, otherwise the reading code will try all available
            // kinds.
            let sl_type_specified = source_list_type.is_none();
            let sl_type = source_list_type.and_then(|t| SourceListType::from_str(t.as_ref()).ok());
            let (sl, sl_type) =
                match mwa_hyperdrive_srclist::read::read_source_list_file(&sl_pb, sl_type) {
                    Ok((sl, sl_type)) => (sl, sl_type),
                    Err(e) => {
                        eprintln!("Error when trying to read source list:");
                        return Err(InvalidArgsError::from(e));
                    }
                };

            // If the user didn't specify the source list type, then print out
            // what we found.
            if sl_type_specified {
                trace!("Successfully parsed {}-style source list", sl_type);
            }

            sl
        };
        trace!("Found {} sources in the source list", source_list.len());
        // Veto any sources that may be troublesome, and/or cap the total number
        // of sources. If the user doesn't specify how many source-list sources
        // to use, then all sources are used.
        if num_sources == Some(0) || source_list.is_empty() {
            return Err(InvalidArgsError::NoSources);
        }
        veto_sources(
            &mut source_list,
            precession_info
                .hadec_j2000
                .to_radec(precession_info.lmst_j2000),
            precession_info.lmst_j2000,
            precession_info.array_latitude_j2000,
            &freq_context.coarse_chan_freqs,
            beam.deref(),
            num_sources,
            source_dist_cutoff.unwrap_or(DEFAULT_CUTOFF_DISTANCE),
            veto_threshold.unwrap_or(DEFAULT_VETO_THRESHOLD),
        )?;
        if source_list.is_empty() {
            return Err(InvalidArgsError::NoSourcesAfterVeto);
        }

        // Convert list flux densities into a power law if possible.
        // TODO: Make this user controllable.
        let num_list_types = source_list
            .par_iter()
            .flat_map(|(_, source)| &source.components)
            .filter(|comp| matches!(comp.flux_type, FluxDensityType::List { .. }))
            .count();
        source_list
            .par_iter_mut()
            .flat_map(|(_, source)| &mut source.components)
            .for_each(|comp| {
                if let FluxDensityType::List { .. } = comp.flux_type {
                    comp.flux_type.convert_list_to_power_law();
                }
            });
        let new_num_list_types = source_list
            .par_iter()
            .flat_map(|(_, source)| &source.components)
            .filter(|comp| matches!(comp.flux_type, FluxDensityType::List { .. }))
            .count();
        debug!(
            "{} components converted from flux density lists to power laws",
            num_list_types - new_num_list_types
        );

        let time_average_factor = match (obs_context.time_res, time_average_factor) {
            (None, _) => {
                // If the input data has unknown time resolution, it's because
                // there's only one timestep.
                1
            }
            (Some(_), None) => {
                // "None" indicates we should follow default behaviour: All data
                // should be averaged in time.
                timesteps_to_use.len()
            }
            (_, Some(f)) => {
                // In all other cases, just use the input as is, but check that
                // it's not too big.
                if f > timesteps_to_use.len() {
                    warn!(
                        "Cannot average {} timesteps; only {} are being used. Capping.",
                        f,
                        timesteps_to_use.len()
                    );
                    timesteps_to_use.len()
                } else {
                    f
                }
            }
        };

        let using_autos = if ignore_autos {
            false
        } else {
            obs_context.autocorrelations_present
        };
        let tile_baseline_maps = TileBaselineMaps::new(total_num_tiles, &tile_flags);

        let (unflagged_tile_xyzs, unflagged_tile_names) = obs_context
            .tile_xyzs
            .par_iter()
            .zip(obs_context.tile_names.par_iter())
            .enumerate()
            .filter(|(tile_index, _)| !tile_flags.contains(tile_index))
            .map(|(_, (xyz, name))| (*xyz, name.clone()))
            .unzip();

        // Make sure the thresholds are sensible.
        let mut stop_threshold = stop_thresh.unwrap_or(DEFAULT_STOP_THRESHOLD);
        let min_threshold = min_thresh.unwrap_or(DEFAULT_MIN_THRESHOLD);
        if stop_threshold > min_threshold {
            warn!("Specified stop threshold ({}) is bigger than the min. threshold ({}); capping the stop threshold.", stop_threshold, min_threshold);
            stop_threshold = min_threshold;
        }

        let params = Self {
            input_data,
            beam,
            source_list,
            model_file,
            tile_flags,
            baseline_flags,
            freq: freq_struct,
            time_average_factor,
            timesteps: timesteps_to_use,
            tile_to_unflagged_baseline_map: tile_baseline_maps.tile_to_unflagged_baseline_map,
            unflagged_baseline_to_tile_map: tile_baseline_maps.unflagged_baseline_to_tile_map,
            using_autos,
            unflagged_tile_names,
            unflagged_tile_xyzs,
            array_longitude,
            array_latitude,
            max_iterations: max_iterations.unwrap_or(DEFAULT_MAX_ITERATIONS),
            stop_threshold,
            min_threshold,
            output_solutions_filenames,
            output_vis_filenames,
            #[cfg(feature = "cuda")]
            cpu_vis_gen: cpu,
        };
        params.log_param_info(&precession_info);
        Ok(params)
    }

    fn log_param_info(&self, precession_info: &PrecessionInfo) {
        let obs_context = self.input_data.get_obs_context();
        let freq_context = self.input_data.get_freq_context();

        info!(
            "Array longitude, latitude:     ({:.4}°, {:.4}°)",
            self.array_longitude.to_degrees(),
            self.array_latitude.to_degrees()
        );
        info!(
            "Array latitude (J2000):                    {:.4}°",
            precession_info.array_latitude_j2000.to_degrees()
        );
        info!(
            "Phase centre:                  ({:.4}°, {:.4}°)",
            obs_context.phase_centre.ra.to_degrees(),
            obs_context.phase_centre.dec.to_degrees()
        );
        let pc_j2000 = precession_info
            .hadec_j2000
            .to_radec(precession_info.lmst_j2000);
        info!(
            "Phase centre (J2000):          ({:.4}°, {:.4}°)",
            pc_j2000.ra.to_degrees(),
            pc_j2000.dec.to_degrees()
        );
        if let Some(pc) = obs_context.pointing_centre {
            info!(
                "Pointing centre:               ({:.4}°, {:.4}°)",
                pc.ra.to_degrees(),
                pc.dec.to_degrees()
            );
        }
        info!(
            "LMST of first timestep:         {:.4}°",
            precession_info.lmst.to_degrees()
        );
        info!(
            "LMST of first timestep (J2000): {:.4}°",
            precession_info.lmst_j2000.to_degrees()
        );

        let total_num_tiles = obs_context.tile_xyzs.len();
        info!("Total number of tiles:           {}", total_num_tiles);
        let num_unflagged_tiles = total_num_tiles - self.tile_flags.len();
        info!("Total number of unflagged tiles: {}", num_unflagged_tiles);
        {
            // Print out the tile flags. Use a vector to sort ascendingly.
            let mut tile_flags = self.tile_flags.iter().collect::<Vec<_>>();
            tile_flags.sort_unstable();
            info!("Tile flags: {:?}", tile_flags);
        }

        info!("Available timesteps: {:?}", 0..obs_context.timesteps.len());
        info!(
            "Unflagged timesteps: {:?}",
            obs_context.unflagged_timestep_indices
        );
        // We don't require the timesteps to be used in calibration to be
        // sequential. But if they are, it looks a bit neater to print them out
        // as a range rather than individual indicies.
        info!(
            "{}",
            range_or_comma_separated(&self.timesteps, Some("Using timesteps:    "))
        );

        match self.timesteps.as_slice() {
            [] => unreachable!(), // Handled by data-reading code.
            [t] => info!(
                "Only timestep (GPS): {:.2}",
                epoch_as_gps_seconds(obs_context.timesteps[*t])
            ),
            [t0, .., tn] => {
                info!(
                    "First timestep (GPS): {:.2}",
                    epoch_as_gps_seconds(obs_context.timesteps[*t0])
                );
                info!(
                    "Last timestep  (GPS): {:.2}",
                    epoch_as_gps_seconds(obs_context.timesteps[*tn])
                );
            }
        }

        match obs_context.time_res {
            Some(native) => {
                info!("Input data time resolution:  {:.2} seconds", native);
            }
            None => info!("Input data time resolution unknown"),
        }
        match freq_context.native_fine_chan_width {
            Some(freq_res) => {
                info!("Input data freq. resolution: {:.2} kHz", freq_res / 1e3);
            }
            None => info!("Input data freq. resolution unknown"),
        }

        info!(
            "Total number of fine channels:     {}",
            self.freq.num_fine_chans
        );
        info!(
            "Number of unflagged fine channels: {}",
            self.freq.unflagged_fine_chans.len()
        );
        if log_enabled!(Debug) {
            let mut unflagged_fine_chans: Vec<_> = self.freq.unflagged_fine_chans.iter().collect();
            unflagged_fine_chans.sort_unstable();
            match unflagged_fine_chans.as_slice() {
                [] => unreachable!(), // Handled by data-reading code.
                [f] => info!("Only unflagged fine-channel: {}", f),
                [f_0, .., f_n] => {
                    info!("First unflagged fine-channel: {}", f_0);
                    info!("Last unflagged fine-channel:  {}", f_n);
                }
            }
        }
        info!(
            "Input data's fine-channel flags per coarse channel: {:?}",
            obs_context.fine_chan_flags_per_coarse_chan
        );
        {
            let mut fine_chan_flags_vec = self.freq.fine_chan_flags.iter().collect::<Vec<_>>();
            fine_chan_flags_vec.sort_unstable();
            debug!("Flagged fine-channels: {:?}", fine_chan_flags_vec);
        }
        match self.freq.unflagged_fine_chan_freqs.as_slice() {
            [] => unreachable!(), // Handled by data-reading code.
            [f] => info!("Only unflagged fine-channel frequency: {:.2} MHz", f / 1e6),
            [f_0, .., f_n] => {
                info!(
                    "First unflagged fine-channel frequency: {:.2} MHz",
                    f_0 / 1e6
                );
                info!(
                    "Last unflagged fine-channel frequency:  {:.2} MHz",
                    f_n / 1e6
                );
            }
        }

        info!(
            "Averaging {} timesteps into each timeblock",
            self.time_average_factor,
        );
        info!(
            "Averaging {} fine-freq. channels into each chanblock",
            self.freq.freq_average_factor,
        );
        info!(
            "Number of calibration timeblocks: {}",
            self.get_num_timeblocks()
        );
        info!(
            "Number of calibration chanblocks: {}",
            self.get_num_chanblocks()
        );

        let (num_points, num_gaussians, num_shapelets) = self.source_list.get_counts();
        let num_components = num_points + num_gaussians + num_shapelets;
        info!(
            "Using {} sources with a total of {} components",
            self.source_list.len(),
            num_components
        );
        info!("{} point components", num_points);
        info!("{} Gaussian components", num_gaussians);
        info!("{} shapelet components", num_shapelets);
        if num_components > 10000 {
            warn!("Using more than 10,000 components!");
        }
        trace!("Using sources: {:?}", self.source_list.keys());

        if !self.output_solutions_filenames.is_empty() {
            info!(
                "Writing calibration solutions to: {}",
                self.output_solutions_filenames
                    .iter()
                    .map(|(_, pb)| pb.display())
                    .join(", ")
            );
        }
        if !self.output_vis_filenames.is_empty() {
            info!(
                "Writing calibrated visibilities to: {}",
                self.output_vis_filenames
                    .iter()
                    .map(|(_, pb)| pb.display())
                    .join(", ")
            );
        }
    }

    pub(crate) fn get_num_timeblocks(&self) -> usize {
        (self.timesteps.len() as f64 / self.time_average_factor as f64).ceil() as _
    }

    pub(crate) fn get_num_chanblocks(&self) -> usize {
        (self.freq.unflagged_fine_chans.len() as f64 / self.freq.freq_average_factor as f64).ceil()
            as _
    }

    /// The number of baselines, including auto-correlation "baselines" if these
    /// are included.
    pub(crate) fn get_num_baselines(&self) -> usize {
        let n = self.unflagged_tile_xyzs.len();
        if self.using_autos {
            (n * (n + 1)) / 2
        } else {
            (n * (n - 1)) / 2
        }
    }
}

/// Get the flagged cross-correlation baseline indices from flagged tile
/// indices. If all 128 tiles are flagged in a 128-element array, then this set
/// will have 8128 flags.
fn get_flagged_baselines_set(
    total_num_tiles: usize,
    tile_flags: &HashSet<usize>,
) -> HashSet<usize> {
    let mut flagged_baselines: HashSet<usize> = HashSet::new();
    let mut bl = 0;
    for tile1 in 0..total_num_tiles {
        for tile2 in tile1 + 1..total_num_tiles {
            if tile_flags.contains(&tile1) || tile_flags.contains(&tile2) {
                flagged_baselines.insert(bl);
            }
            bl += 1;
        }
    }
    flagged_baselines
}

// It looks a bit neater to print out a collection of numbers as a range rather
// than individual indicies if they're sequential. This function inspects a
// collection and returns a string to be printed.
fn range_or_comma_separated(collection: &[usize], prefix: Option<&str>) -> String {
    if collection.is_empty() {
        return "".to_string();
    }

    let mut iter = collection.iter();
    let mut prev = *iter.next().unwrap();
    // Innocent until proven guilty.
    let mut is_sequential = true;
    for next in iter {
        if *next == prev + 1 {
            prev = *next;
        } else {
            is_sequential = false;
            break;
        }
    }

    if is_sequential {
        let suffix = format!("{:?}", (collection[0]..collection.last().unwrap() + 1));
        if let Some(p) = prefix {
            format!("{} {}", p, suffix)
        } else {
            suffix
        }
    } else {
        let suffix = collection
            .iter()
            .map(|t| t.to_string())
            .collect::<Vec<_>>()
            .join(", ");
        if let Some(p) = prefix {
            format!("{} {}", p, suffix)
        } else {
            suffix
        }
    }
}

/// Check if we are able to write to a file path. If we aren't able to write to
/// the file, it's either because the directory containing the file doesn't
/// exist, or there's another issue (probably bad permissions). In the former
/// case, create the parent directories, otherwise return an error.
/// Additionally, if the file exists, emit a warning that it will be
/// overwritten.
///
/// With this approach, we potentially avoid doing a whole run of calibration
/// only to be unable to write to a file at the end. This code _doesn't_ alter
/// the file if it exists.
fn can_write_to_file<T: AsRef<Path>>(file: T) -> Result<(), InvalidArgsError> {
    let file_exists = file.as_ref().exists();

    match OpenOptions::new()
        .write(true)
        .create(true)
        .open(&file)
        .map_err(|e| e.kind())
    {
        // File is writable.
        Ok(_) => {
            if file_exists {
                warn!(
                    "Will overwrite the existing file '{}'",
                    file.as_ref().display()
                )
            }
        }

        // File doesn't exist. Attempt to make the directories leading up to the
        // file; if this fails, then we can't write the file anyway.
        Err(std::io::ErrorKind::NotFound) => {
            if let Some(p) = file.as_ref().parent() {
                match std::fs::DirBuilder::new()
                    .recursive(true)
                    .create(p)
                    .map_err(|e| e.kind())
                {
                    Ok(()) => (),
                    Err(std::io::ErrorKind::PermissionDenied) => {
                        return Err(InvalidArgsError::NewDirectory(p.to_path_buf()))
                    }
                    Err(e) => return Err(InvalidArgsError::IO(e.into())),
                }
            }
        }

        Err(std::io::ErrorKind::PermissionDenied) => {
            return Err(InvalidArgsError::FileNotWritable {
                file: file.as_ref().display().to_string(),
            })
        }

        Err(e) => {
            return Err(InvalidArgsError::IO(e.into()));
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tests::{full_obsids::*, reduced_obsids::*, *};

    #[test]
    fn test_get_flagged_baselines_set() {
        let total_num_tiles = 128;
        let mut tile_flags = HashSet::new();
        let flagged_baselines = get_flagged_baselines_set(total_num_tiles, &tile_flags);
        assert!(flagged_baselines.is_empty());

        tile_flags.insert(127);
        let flagged_baselines = get_flagged_baselines_set(total_num_tiles, &tile_flags);
        assert_eq!(flagged_baselines.len(), 127);
        assert!(flagged_baselines.contains(&126));
        assert!(flagged_baselines.contains(&252));
        assert!(flagged_baselines.contains(&8127));
    }

    // #[test]
    // fn test_new_params() {
    //     let args = get_1090008640_smallest();
    //     let params = match args.into_params() {
    //         Ok(p) => p,
    //         Err(e) => panic!("{}", e),
    //     };
    //     // The default time resolution should be 2.0s, as per the metafits.
    //     assert_abs_diff_eq!(params.time_res.unwrap(), 2.0);
    //     // The default freq resolution should be 40kHz, as per the metafits.
    //     assert_abs_diff_eq!(params.freq.res.unwrap(), 40e3, epsilon = 1e-10);
    // }

    // #[test]
    // fn test_new_params_time_averaging() {
    //     // The native time resolution is 2.0s.
    //     let mut args = get_1090008640_smallest();
    //     // 4.0 should be a multiple of 2.0s
    //     args.time_res = Some(4.0);
    //     let params = match args.into_params() {
    //         Ok(p) => p,
    //         Err(e) => panic!("{}", e),
    //     };
    //     assert_abs_diff_eq!(params.time_res.unwrap(), 4.0);

    //     let mut args = get_1090008640();
    //     // 8.0 should be a multiple of 2.0s
    //     args.time_res = Some(8.0);
    //     let params = match args.into_params() {
    //         Ok(p) => p,
    //         Err(e) => panic!("{}", e),
    //     };
    //     assert_abs_diff_eq!(params.time_res.unwrap(), 8.0);
    // }

    // #[test]
    // fn test_new_params_time_averaging_fail() {
    //     // The native time resolution is 2.0s.
    //     let mut args = get_1090008640_smallest();
    //     // 2.01 is not a multiple of 2.0s
    //     args.time_res = Some(2.01);
    //     let result = args.into_params();
    //     assert!(
    //         result.is_err(),
    //         "Expected CalibrateParams to have not been successfully created"
    //     );

    //     let mut args = get_1090008640_smallest();
    //     // 3.0 is not a multiple of 2.0s
    //     args.time_res = Some(3.0);
    //     let result = args.into_params();
    //     assert!(
    //         result.is_err(),
    //         "Expected CalibrateParams to have not been successfully created"
    //     );
    // }

    // #[test]
    // fn test_new_params_freq_averaging() {
    //     // The native freq. resolution is 40kHz.
    //     let mut args = get_1090008640_smallest();
    //     // 80e3 should be a multiple of 40kHz
    //     args.freq_res = Some(80e3);
    //     let params = match args.into_params() {
    //         Ok(p) => p,
    //         Err(e) => panic!("{}", e),
    //     };
    //     assert_abs_diff_eq!(params.freq.res, 80e3, epsilon = 1e-10);

    //     let mut args = get_1090008640_smallest();
    //     // 200e3 should be a multiple of 40kHz
    //     args.freq_res = Some(200e3);
    //     let params = match args.into_params() {
    //         Ok(p) => p,
    //         Err(e) => panic!("{}", e),
    //     };
    //     assert_abs_diff_eq!(params.freq.res, 200e3, epsilon = 1e-10);
    // }

    // #[test]
    // fn test_new_params_freq_averaging_fail() {
    //     // The native freq. resolution is 40kHz.
    //     let mut args = get_1090008640_smallest();
    //     // 10e3 is not a multiple of 40kHz
    //     args.freq_res = Some(10e3);
    //     let result = args.into_params();
    //     assert!(
    //         result.is_err(),
    //         "Expected CalibrateParams to have not been successfully created"
    //     );

    //     let mut args = get_1090008640_smallest();

    //     // 79e3 is not a multiple of 40kHz
    //     args.freq_res = Some(79e3);
    //     let result = args.into_params();
    //     assert!(
    //         result.is_err(),
    //         "Expected CalibrateParams to have not been successfully created"
    //     );
    // }

    #[test]
    fn test_new_params_tile_flags() {
        // 1090008640 has no flagged tiles in its metafits.
        let mut args = get_1090008640_smallest();
        // Manually flag antennas 1, 2 and 3.
        args.tile_flags = Some(vec![1, 2, 3]);
        let params = match args.into_params() {
            Ok(p) => p,
            Err(e) => panic!("{}", e),
        };
        assert_eq!(params.tile_flags.len(), 3);
        assert!(params.tile_flags.contains(&1));
        assert!(params.tile_flags.contains(&2));
        assert!(params.tile_flags.contains(&3));
        assert_eq!(params.unflagged_baseline_to_tile_map.len(), 7750);
        assert_eq!(params.tile_to_unflagged_baseline_map.len(), 7750);

        assert_eq!(params.unflagged_baseline_to_tile_map[&0], (0, 4));
        assert_eq!(params.unflagged_baseline_to_tile_map[&1], (0, 5));
        assert_eq!(params.unflagged_baseline_to_tile_map[&2], (0, 6));
        assert_eq!(params.unflagged_baseline_to_tile_map[&3], (0, 7));

        assert_eq!(params.tile_to_unflagged_baseline_map[&(0, 4)], 0);
        assert_eq!(params.tile_to_unflagged_baseline_map[&(0, 5)], 1);
        assert_eq!(params.tile_to_unflagged_baseline_map[&(0, 6)], 2);
        assert_eq!(params.tile_to_unflagged_baseline_map[&(0, 7)], 3);
    }

    // The following tests use full MWA data.

    #[test]
    #[serial]
    #[ignore]
    fn test_new_params_real_data() {
        let args = get_1065880128();
        let result = args.into_params();
        assert!(
            result.is_ok(),
            "Expected CalibrateParams to have been successfully created"
        );
    }

    // #[test]
    // #[serial]
    // #[ignore]
    // fn test_lst_from_timestep_native_real() {
    //     let args = get_1065880128();
    //     let context = match CorrelatorContext::new(&args.metafits.unwrap(), &args.gpuboxes.unwrap())
    //     {
    //         Ok(c) => c,
    //         Err(e) => panic!("{}", e),
    //     };
    //     let time_res = context.metafits_context.corr_int_time_ms as f64 / 1e3;
    //     let new_lst = lst_from_timestep(0, &context, time_res);
    //     // gpstime 1065880126.25
    //     assert_abs_diff_eq!(new_lst, 6.074695614533638, epsilon = 1e-10);

    //     let new_lst = lst_from_timestep(1, &context, time_res);
    //     // gpstime 1065880126.75
    //     assert_abs_diff_eq!(new_lst, 6.074732075112903, epsilon = 1e-10);
    // }

    // #[test]
    // #[serial]
    // #[ignore]
    // fn test_lst_from_timestep_averaged_real() {
    //     let args = get_1065880128();
    //     let context = match CorrelatorContext::new(&args.metafits.unwrap(), &args.gpuboxes.unwrap())
    //     {
    //         Ok(c) => c,
    //         Err(e) => panic!("{}", e),
    //     };
    //     // The native time res. is 0.5s, let's make our target 2s here.
    //     let time_res = 2.0;
    //     let new_lst = lst_from_timestep(0, &context, time_res);
    //     // gpstime 1065880127
    //     assert_abs_diff_eq!(new_lst, 6.074750305402534, epsilon = 1e-10);

    //     let new_lst = lst_from_timestep(1, &context, time_res);
    //     // gpstime 1065880129
    //     assert_abs_diff_eq!(new_lst, 6.074896147719591, epsilon = 1e-10);
    // }
}
