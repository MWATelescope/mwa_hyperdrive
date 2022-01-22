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
mod filenames;
pub(crate) mod freq;
#[cfg(test)]
mod tests;

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
use marlu::{
    constants::{MWA_LAT_RAD, MWA_LONG_RAD},
    pos::{
        precession::{precess_time, PrecessionInfo},
        xyz::xyzs_to_cross_uvws_parallel,
    },
    time::epoch_as_gps_seconds,
    Jones, XyzGeodetic,
};
use ndarray::ArrayViewMut2;
use rayon::prelude::*;

use super::solutions::CalSolutionType;
use crate::{
    constants::*,
    context::{FreqContext, ObsContext},
    data_formats::*,
    glob::*,
    math::TileBaselineMaps,
    pfb_gains::{PfbFlavour, DEFAULT_PFB_FLAVOUR},
    unit_parsing::{
        parse_freq, parse_time, parse_wavelength, FreqFormat, TimeFormat, WavelengthUnit,
    },
};
use mwa_hyperdrive_beam::{create_fee_beam_object, create_no_beam_object, Beam, Delays};
use mwa_hyperdrive_common::{itertools, log, marlu, ndarray, rayon};
use mwa_hyperdrive_srclist::{
    constants::*, veto_sources, FluxDensityType, SourceList, SourceListType,
};

/// Parameters needed to perform calibration.
pub struct CalibrateParams {
    /// Interface to the MWA data, and metadata on the input data.
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

    /// Multiplicative factors to apply to unflagged baselines. These are mostly
    /// all 1.0, but flagged baselines (perhaps due to a UVW cutoff) have values
    /// of 0.0.
    pub(crate) baseline_weights: Vec<f64>,

    /// Channel- and frequency-related parameters required for calibration.
    pub(crate) freq: FrequencyParams,

    /// The number of time samples to average together before calibrating.
    ///
    /// e.g. If the input data is in 0.5s resolution and this variable was 4,
    /// then we average 2s worth of data together during calibration.
    pub(crate) time_average_factor: usize,

    /// The timestep indices to use in calibration. These are sorted
    /// ascendingly.
    pub(crate) timesteps: Vec<usize>,

    /// Given two antenna indices, get the unflagged cross-correlation baseline
    /// index. e.g. If antenna 1 (i.e. the second antenna) is flagged, then the
    /// first baseline (i.e. 0) is between antenna 0 and antenna 2.
    ///
    /// This exists because some tiles may be flagged, so some baselines may be
    /// flagged.
    pub(crate) tile_to_unflagged_cross_baseline_map: HashMap<(usize, usize), usize>,

    /// Given an unflagged baseline index, get the tile index pair that
    /// contribute to it. e.g. If tile 1 (i.e. the second tile) is flagged, then
    /// the first unflagged baseline (i.e. 0) is between tile 0 and tile 2.
    ///
    /// This exists because some tiles may be flagged, so some baselines may be
    /// flagged.
    pub(crate) unflagged_cross_baseline_to_tile_map: HashMap<usize, (usize, usize)>,

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

    /// The number of calibrated time samples to average together before writing
    /// out calibrated visibilities.
    pub(crate) output_vis_time_average_factor: usize,

    /// The number of calibrated frequencies samples to average together before
    /// writing out calibrated visibilities.
    pub(crate) output_vis_freq_average_factor: usize,

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
            source_list,
            source_list_type,
            num_sources,
            source_dist_cutoff,
            veto_threshold,
            beam_file,
            unity_dipole_gains,
            delays,
            no_beam,
            time_average_factor,
            // freq_average_factor,
            timesteps,
            tile_flags,
            ignore_input_data_tile_flags,
            ignore_input_data_fine_channel_flags,
            ignore_autos,
            fine_chan_flags_per_coarse_chan,
            fine_chan_flags,
            pfb_flavour,
            no_digital_gains,
            no_geometric_correction,
            no_cable_length_correction,
            output_vis_time_average,
            output_vis_freq_average,
            uvw_min,
            uvw_max,
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
                let pfb_flavour = match pfb_flavour {
                    None => DEFAULT_PFB_FLAVOUR,
                    Some(s) => match PfbFlavour::from_str(&s.to_lowercase()) {
                        Err(_) => return Err(InvalidArgsError::ParsePfbFlavour(s)),
                        Ok(p) => p,
                    },
                };

                let input_data = RawData::new(
                    &meta,
                    &gpuboxes,
                    mwafs.as_deref(),
                    &mut dipole_delays,
                    pfb_flavour,
                    !no_digital_gains,
                    !no_cable_length_correction,
                    !no_geometric_correction,
                )?;

                // Print some high-level information.
                let obs_context = input_data.get_obs_context();
                // obsid is always present because we must have a metafits file;
                // unwrap is safe.
                info!("Calibrating obsid {}", obs_context.obsid.unwrap());
                info!("Using metafits: {}", meta.display());
                info!("Using {} gpubox files", gpuboxes.len());
                match pfb_flavour {
                    PfbFlavour::None => info!("Not doing any PFB correction"),
                    PfbFlavour::Empirical => info!("Using 'RTS empirical' PFB gains"),
                    PfbFlavour::Levine => info!("Using 'Alan Levine' PFB gains"),
                }
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
        if log_enabled!(Debug) {
            debug!(
                "All tile indices and names: {:?}",
                obs_context
                    .tile_names
                    .iter()
                    .enumerate()
                    .collect::<Vec<_>>()
            );
        }
        let mut tile_flags: HashSet<usize> = match tile_flags {
            Some(flags) => {
                // We need to convert the strings into antenna indices. The
                // strings are either indicies themselves or antenna names.
                let mut tile_flags = HashSet::new();

                for flag in flags {
                    // Try to parse a naked number.
                    match flag.trim().parse().ok() {
                        Some(n) => {
                            if n >= total_num_tiles {
                                return Err(InvalidArgsError::InvalidTileFlag {
                                    got: n,
                                    max: total_num_tiles - 1,
                                });
                            }
                            tile_flags.insert(n);
                        }
                        None => {
                            // Check if this is an antenna name.
                            match obs_context
                                .tile_names
                                .iter()
                                .enumerate()
                                .find(|(_, name)| name.to_lowercase() == flag.to_lowercase())
                            {
                                // If there are no matches, complain that
                                // the user input is no good.
                                None => return Err(InvalidArgsError::BadTileFlag(flag)),
                                Some((i, _)) => tile_flags.insert(i),
                            };
                        }
                    }
                }

                tile_flags
            }
            None => HashSet::new(),
        };
        if !ignore_input_data_tile_flags {
            // Add tiles that have already been flagged by the input data.
            for &obs_tile_flag in &obs_context.tile_flags {
                tile_flags.insert(obs_tile_flag);
            }
        }

        // Assign the per-coarse-channel fine-channel flags.
        // TODO: Rename "coarse band" to "coarse channel".
        let mut fine_chan_flags_per_coarse_chan: HashSet<usize> =
            match fine_chan_flags_per_coarse_chan {
                Some(flags) => flags.into_iter().collect(),
                None => HashSet::new(),
            };
        if !ignore_input_data_fine_channel_flags {
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
        // Print out some coordinates, including their precessed counterparts
        // (if applicable).
        let precession_info = precess_time(
            obs_context.phase_centre,
            obs_context.timesteps[0],
            array_longitude,
            array_latitude,
        );
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

        // Handle output visibility arguments.
        let (output_vis_time_average_factor, output_vis_freq_average_factor) =
            if output_vis_filenames.is_empty() {
                // If we're not writing out calibrated visibilities but arguments
                // are set for them, issue warnings.
                match (output_vis_time_average, output_vis_freq_average) {
                    (Some(_), Some(_)) => {
                        warn!("Not writing out calibrated visibilities, but");
                        warn!("    \"output_vis_time_average_factor\" and");
                        warn!("    \"output_vis_freq_average_factor\" are set.");
                    }

                    (Some(_), None) => {
                        warn!("Not writing out calibrated visibilities, but");
                        warn!("    \"output_vis_time_average_factor\" is set.");
                    }

                    (None, Some(_)) => {
                        warn!("Not writing out calibrated visibilities, but");
                        warn!("    \"output_vis_freq_average_factor\" is set.");
                    }

                    (None, None) => (),
                }
                (1, 1)
            } else {
                // Parse and verify user input (specified resolutions must
                // evenly divide the input data's resolutions).
                let time_factor = match output_vis_time_average.map(|f| parse_time(&f)) {
                    None => 1.0,
                    Some(Ok((factor, TimeFormat::NoUnit))) => {
                        // Reject non-integer floats.
                        if factor.fract().abs() > 1e-6 {
                            return Err(InvalidArgsError::OutputVisTimeFactorNotInteger);
                        }
                        // Zero is not allowed.
                        if factor < f64::EPSILON {
                            return Err(InvalidArgsError::OutputVisTimeAverageFactorZero);
                        }

                        factor
                    }
                    Some(Ok((out_res, time_format))) => {
                        let out_res_s = out_res
                            * match time_format {
                                TimeFormat::S | TimeFormat::NoUnit => 1.0,
                                TimeFormat::Ms => 1000.0,
                            };
                        // If the time resolution isn't determined, it's because
                        // there's only one timestep. In this case, it doesn't
                        // matter what the output time resolution is; only one
                        // timestep goes into it.
                        let ratio = out_res_s / obs_context.time_res.unwrap_or(1.0);
                        // Reject non-integer floats.
                        if ratio.fract().abs() > 1e-6 {
                            return Err(InvalidArgsError::OutputVisTimeResNotMulitple {
                                out: out_res_s,
                                inp: obs_context.time_res.unwrap(),
                            });
                        }
                        // Zero is not allowed.
                        if ratio < f64::EPSILON {
                            return Err(InvalidArgsError::OutputVisTimeResZero);
                        }

                        ratio
                    }
                    Some(Err(e)) => {
                        return Err(InvalidArgsError::ParseOutputVisTimeAverageFactor(e))
                    }
                };
                let freq_factor = match output_vis_freq_average.map(|f| parse_freq(&f)) {
                    None => 1.0,
                    Some(Ok((factor, FreqFormat::NoUnit))) => {
                        // Reject non-integer floats.
                        if factor.fract().abs() > 1e-6 {
                            return Err(InvalidArgsError::OutputVisFreqFactorNotInteger);
                        }
                        // Zero is not allowed.
                        if factor < f64::EPSILON {
                            return Err(InvalidArgsError::OutputVisFreqAverageFactorZero);
                        }

                        factor
                    }
                    Some(Ok((out_res, freq_format))) => {
                        let out_res_hz = out_res
                            * match freq_format {
                                FreqFormat::Hz | FreqFormat::NoUnit => 1.0,
                                FreqFormat::kHz => 1000.0,
                            };
                        // If the frequency resolution isn't determined, it's
                        // because there's only one fine channel. In this case,
                        // it doesn't matter what the output freq. resolution
                        // is; only one channel goes into it.
                        let ratio = out_res_hz / freq_context.native_fine_chan_width.unwrap_or(1.0);
                        if ratio.fract().abs() > 1e-6 {
                            return Err(InvalidArgsError::OutputVisFreqResNotMulitple {
                                out: out_res_hz,
                                inp: freq_context.native_fine_chan_width.unwrap(),
                            });
                        }
                        // Zero is not allowed.
                        if ratio < f64::EPSILON {
                            return Err(InvalidArgsError::OutputVisFreqResZero);
                        }

                        ratio
                    }
                    Some(Err(e)) => {
                        return Err(InvalidArgsError::ParseOutputVisFreqAverageFactor(e))
                    }
                };

                (time_factor as usize, freq_factor as usize)
            };

        let using_autos = if ignore_autos {
            false
        } else {
            obs_context.autocorrelations_present
        };
        let tile_baseline_maps = TileBaselineMaps::new(total_num_tiles, &tile_flags);

        let (unflagged_tile_xyzs, unflagged_tile_names): (Vec<XyzGeodetic>, Vec<String>) =
            obs_context
                .tile_xyzs
                .par_iter()
                .zip(obs_context.tile_names.par_iter())
                .enumerate()
                .filter(|(tile_index, _)| !tile_flags.contains(tile_index))
                .map(|(_, (xyz, name))| (*xyz, name.clone()))
                .unzip();

        // Set baseline weights from UVW cuts.
        let uvw_min = uvw_min.or_else(|| Some(DEFAULT_UVW_MIN.to_string()));
        let baseline_weights = {
            let mut baseline_weights = vec![
                1.0;
                tile_baseline_maps
                    .unflagged_cross_baseline_to_tile_map
                    .len()
            ];
            if uvw_min.is_some() || uvw_max.is_some() {
                // Parse the arguments. If a lambda was used, then we use the
                // centroid frequency of the observation.
                let freq_centroid = freq_context.fine_chan_freqs.iter().sum::<f64>()
                    / freq_context.fine_chan_freqs.len() as f64;
                let mut lambda_used = false;
                // Let the new uvw_min and uvw_max values be in metres.
                let uvw_min = match uvw_min {
                    None => 0.0,
                    Some(uvw_min) => {
                        let (quantity, unit) =
                            parse_wavelength(&uvw_min).map_err(InvalidArgsError::ParseUvwMin)?;
                        match unit {
                            WavelengthUnit::M => {
                                info!("Minimum UVW cutoff: {}m", quantity);
                                quantity
                            }
                            WavelengthUnit::L => {
                                if !lambda_used {
                                    info!("Using observation centroid frequency {} MHz to convert lambdas to metres", freq_centroid/1e6);
                                    lambda_used = true;
                                }
                                let metres = marlu::constants::VEL_C / freq_centroid * quantity;
                                info!("Minimum UVW cutoff: {}λ ({:.3}m)", quantity, metres);
                                metres
                            }
                        }
                    }
                };
                let uvw_max = match uvw_max {
                    None => f64::INFINITY,
                    Some(uvw_max) => {
                        let (quantity, unit) =
                            parse_wavelength(&uvw_max).map_err(InvalidArgsError::ParseUvwMax)?;
                        match unit {
                            WavelengthUnit::M => {
                                info!("Maximum UVW cutoff: {}m", quantity);
                                quantity
                            }
                            WavelengthUnit::L => {
                                if !lambda_used {
                                    info!("Using observation centroid frequency {} MHz to convert lambdas to metres", freq_centroid/1e6);
                                    // lambda_used = true;
                                }
                                let metres = marlu::constants::VEL_C / freq_centroid * quantity;
                                info!("Maximum UVW cutoff: {}λ ({:.3}m)", quantity, metres);
                                metres
                            }
                        }
                    }
                };

                let uvws = xyzs_to_cross_uvws_parallel(
                    &unflagged_tile_xyzs,
                    obs_context
                        .phase_centre
                        .to_hadec(precession_info.lmst_j2000),
                );
                debug_assert_eq!(baseline_weights.len(), uvws.len());
                uvws.into_par_iter()
                    .zip(baseline_weights.par_iter_mut())
                    .for_each(|(uvw, baseline_weight)| {
                        let uvw_length = uvw.u * uvw.u + uvw.v * uvw.v + uvw.w * uvw.w;
                        if uvw_length < uvw_min * uvw_min || uvw_length > uvw_max * uvw_max {
                            *baseline_weight = 0.0;
                        }
                    });
            }
            baseline_weights
        };

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
            baseline_weights,
            freq: freq_struct,
            time_average_factor,
            timesteps: timesteps_to_use,
            tile_to_unflagged_cross_baseline_map: tile_baseline_maps
                .tile_to_unflagged_cross_baseline_map,
            unflagged_cross_baseline_to_tile_map: tile_baseline_maps
                .unflagged_cross_baseline_to_tile_map,
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
            output_vis_time_average_factor,
            output_vis_freq_average_factor,
            #[cfg(feature = "cuda")]
            cpu_vis_gen: cpu,
        };
        params.log_param_info(&precession_info)?;
        Ok(params)
    }

    pub(crate) fn get_obs_context(&self) -> &ObsContext {
        self.input_data.get_obs_context()
    }

    pub(crate) fn get_freq_context(&self) -> &FreqContext {
        self.input_data.get_freq_context()
    }

    pub(crate) fn read_crosses(
        &self,
        vis: ArrayViewMut2<Jones<f32>>,
        weights: ArrayViewMut2<f32>,
        timestep: usize,
    ) -> Result<(), ReadInputDataError> {
        self.input_data.read_crosses(
            vis,
            weights,
            timestep,
            &self.tile_to_unflagged_cross_baseline_map,
            &self.freq.fine_chan_flags,
        )
    }

    fn log_param_info(&self, precession_info: &PrecessionInfo) -> Result<(), InvalidArgsError> {
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
            "Phase centre (J2000):          ({:.4}°, {:.4}°)",
            obs_context.phase_centre.ra.to_degrees(),
            obs_context.phase_centre.dec.to_degrees()
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
        info!("Number of unflagged tiles:       {}", num_unflagged_tiles);
        {
            // Print out the tile flags. Use a vector to sort ascendingly.
            let mut tile_flags = self.tile_flags.iter().collect::<Vec<_>>();
            tile_flags.sort_unstable();
            info!("Tile flags: {:?}", tile_flags);
        }
        if num_unflagged_tiles == 0 {
            return Err(InvalidArgsError::NoTiles);
        }
        if log_enabled!(Debug) {
            debug!("Tile indices, names and statuses:");
            obs_context
                .tile_names
                .iter()
                .enumerate()
                .map(|(i, name)| {
                    let flagged = self.tile_flags.contains(&i);
                    (i, name, if flagged { "  flagged" } else { "unflagged" })
                })
                .for_each(|(i, name, status)| {
                    debug!("    {:3}: {:10}: {}", i, name, status);
                })
        }

        info!(
            "{}",
            range_or_comma_separated(
                &obs_context.all_timestep_indices,
                Some("Available timesteps:")
            )
        );
        info!(
            "{}",
            range_or_comma_separated(
                &obs_context.unflagged_timestep_indices,
                Some("Unflagged timesteps:")
            )
        );
        // We don't require the timesteps to be used in calibration to be
        // sequential. But if they are, it looks a bit neater to print them out
        // as a range rather than individual indicies.
        info!(
            "{}",
            range_or_comma_separated(&self.timesteps, Some("Using timesteps:    "))
        );

        match self.timesteps.as_slice() {
            [] => {
                info!("No timesteps being used!");
                return Err(InvalidArgsError::NoTimesteps);
            }
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
                [f] => debug!("Only unflagged fine-channel: {}", f),
                [f_0, .., f_n] => {
                    debug!("First unflagged fine-channel: {}", f_0);
                    debug!("Last unflagged fine-channel:  {}", f_n);
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
            [] => return Err(InvalidArgsError::NoChannels),
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

            info!("Averaging output calibrated visibilities");
            if let Some(tr) = obs_context.time_res {
                info!(
                    "    {}x in time  ({}s)",
                    self.output_vis_time_average_factor,
                    tr * self.output_vis_time_average_factor as f64
                );
            } else {
                info!(
                    "    {}x (only one timestep)",
                    self.output_vis_time_average_factor
                );
            }

            if let Some(fr) = freq_context.native_fine_chan_width {
                info!(
                    "    {}x in freq. ({}kHz)",
                    self.output_vis_freq_average_factor,
                    fr * self.output_vis_freq_average_factor as f64 / 1000.0
                );
            } else {
                info!(
                    "    {}x (only one fine channel)",
                    self.output_vis_freq_average_factor
                );
            }
        }

        Ok(())
    }

    pub(crate) fn get_num_timeblocks(&self) -> usize {
        (self.timesteps.len() as f64 / self.time_average_factor as f64).ceil() as _
    }

    pub(crate) fn get_num_chanblocks(&self) -> usize {
        (self.freq.unflagged_fine_chans.len() as f64 / self.freq.freq_average_factor as f64).ceil()
            as _
    }

    /// The number of unflagged baselines, including auto-correlation
    /// "baselines" if these are included.
    pub(crate) fn get_num_unflagged_baselines(&self) -> usize {
        let n = self.unflagged_tile_xyzs.len();
        if self.using_autos {
            (n * (n + 1)) / 2
        } else {
            (n * (n - 1)) / 2
        }
    }
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
        let suffix = if collection.len() == 1 {
            format!("[{}]", collection[0])
        } else {
            format!(
                "[{:?})",
                (*collection.first().unwrap()..*collection.last().unwrap() + 1)
            )
        };
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
            format!("{} [{}]", p, suffix)
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
