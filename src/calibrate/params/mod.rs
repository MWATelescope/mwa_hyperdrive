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
mod helpers;
#[cfg(test)]
mod tests;

pub(crate) use error::*;
use helpers::*;

use std::collections::{HashMap, HashSet};
use std::fs::OpenOptions;
use std::ops::Deref;
use std::path::{Path, PathBuf};
use std::str::FromStr;

use itertools::Itertools;
use log::{debug, info, log_enabled, trace, warn, Level::Debug};
use marlu::{
    pos::{precession::precess_time, xyz::xyzs_to_cross_uvws_parallel},
    Jones, LatLngHeight, XyzGeodetic,
};
use ndarray::ArrayViewMut2;
use rayon::prelude::*;
use vec1::Vec1;

use super::{messages, solutions::CalSolutionType, CalibrateUserArgs, Fence, Timeblock};
use crate::{
    constants::*,
    context::ObsContext,
    data_formats::*,
    filenames::InputDataTypes,
    glob::*,
    math::TileBaselineMaps,
    pfb_gains::{PfbFlavour, DEFAULT_PFB_FLAVOUR},
    unit_parsing::{parse_wavelength, WavelengthUnit},
};
use mwa_hyperdrive_beam::{create_fee_beam_object, create_no_beam_object, Beam, Delays};
use mwa_hyperdrive_common::{itertools, log, marlu, ndarray, rayon, vec1};
use mwa_hyperdrive_srclist::{
    veto_sources, SourceList, SourceListType, DEFAULT_CUTOFF_DISTANCE, DEFAULT_VETO_THRESHOLD,
};

/// Parameters needed to perform calibration.
pub(crate) struct CalibrateParams {
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
    pub(crate) flagged_tiles: Vec<usize>,

    /// Multiplicative factors to apply to unflagged baselines. These are mostly
    /// all 1.0, but flagged baselines (perhaps due to a UVW cutoff) have values
    /// of 0.0.
    pub(crate) baseline_weights: Vec<f64>,

    /// Blocks of timesteps used for calibration. Each timeblock contains
    /// indices of the input data to average together during calibration. Each
    /// timeblock may have a different number of timesteps; the number of blocks
    /// and their lengths depends on which input data timesteps are being used
    /// as well as the `time_average_factor` (i.e. the number of timesteps to
    /// average during calibration; by default we average all timesteps).
    ///
    /// Simple examples: If we are averaging all data over time to form
    /// calibration solutions, there will only be one timeblock, and that block
    /// will contain all input data timestep indices. On the other hand, if
    /// `time_average_factor` is 1, then there are as many timeblocks as there
    /// are timesteps, and each block contains 1 timestep index.
    ///
    /// A more complicated example: If we are using input data timesteps 10, 11,
    /// 12 and 15 with a `time_average_factor` of 4, then there will be 2
    /// timeblocks, even though there are only 4 timesteps. This is because
    /// timestep 10 and 15 can't occupy the same timeblock with the "length" is
    /// 4. So the first timeblock contains 10, 11 and 12, whereas the second
    /// contains only 15.
    pub(super) timeblocks: Vec1<Timeblock>,

    /// The timestep indicies into the input data to be used for calibration.
    pub(crate) timesteps: Vec1<usize>,

    /// The number of frequency samples to average together during calibration.
    ///
    /// e.g. If the input data is in 40kHz resolution and this variable was 2,
    /// then we average 80kHz worth of data together during calibration.
    pub(crate) freq_average_factor: usize,

    /// Spectral windows, or, groups of contiguous-bands of channels to be
    /// calibrated. Multiple [Fence]s can represent a "picket fence"
    /// observation. Each [Fence] is composed of chanblocks, and the unflagged
    /// chanblocks are calibrated. Each chanblock may represent of multiple
    /// channels, depending on `freq_average_factor`; when visibilities are read
    /// from `input_data`, the channels are averaged according to
    /// `freq_average_factor`. If no frequency channels are flagged, then these
    /// chanblocks will represent all frequency channels. However, it's likely
    /// at least some channels are flagged, so the `flagged_chanblock_indices`
    /// in every [Fence] may be needed.
    pub(crate) fences: Vec1<Fence>,

    /// The frequencies of each of the observation's unflagged fine channels
    /// \[Hz\].
    pub(crate) unflagged_fine_chan_freqs: Vec<f64>,

    /// The fine channels to be flagged across the entire observation. e.g. For
    /// a 40 kHz observation, there are 768 fine channels, and this could
    /// contain 0 and 767.
    pub(crate) flagged_fine_chans: HashSet<usize>,

    /// Given two antenna indices, get the unflagged cross-correlation baseline
    /// index. e.g. If antenna 1 (i.e. the second antenna) is flagged, then the
    /// first baseline (i.e. 0) is between antenna 0 and antenna 2.
    ///
    /// This exists because some tiles may be flagged, so some baselines may be
    /// flagged.
    pub(crate) tile_to_unflagged_cross_baseline_map: HashMap<(usize, usize), usize>,

    /// The names of the unflagged tiles.
    pub(crate) unflagged_tile_names: Vec<String>,

    /// The unflagged [XyzGeodetic] coordinates of each tile \[metres\]. This
    /// does not change over time; it is determined only by the telescope's tile
    /// layout.
    pub(crate) unflagged_tile_xyzs: Vec<XyzGeodetic>,

    /// The Earth position of the array. This is populated by user input or the input data.
    pub(crate) array_position: LatLngHeight,

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

    /// When reading in visibilities and generating sky-model visibilities,
    /// don't draw progress bars.
    pub(crate) no_progress_bars: bool,

    #[cfg(feature = "cuda")]
    /// If true, use the CPU to generate sky-model visibilites. Otherwise, use
    /// the GPU.
    pub(crate) use_cpu_for_modelling: bool,
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
        CalibrateUserArgs {
            data,
            source_list,
            source_list_type,
            outputs,
            model_filename,
            output_vis_time_average,
            output_vis_freq_average,
            num_sources,
            source_dist_cutoff,
            veto_threshold,
            beam_file,
            unity_dipole_gains,
            delays,
            no_beam,
            time_average_factor,
            freq_average_factor,
            timesteps,
            uvw_min,
            uvw_max,
            max_iterations,
            stop_thresh,
            min_thresh,
            array_position,
            #[cfg(feature = "cuda")]
            cpu,
            tile_flags,
            ignore_input_data_tile_flags,
            ignore_input_data_fine_channel_flags,
            fine_chan_flags_per_coarse_chan,
            fine_chan_flags,
            pfb_flavour,
            no_digital_gains,
            no_cable_length_correction,
            no_geometric_correction,
            no_progress_bars,
        }: CalibrateUserArgs,
    ) -> Result<CalibrateParams, InvalidArgsError> {
        // Handle input data. We expect one of three possibilities:
        // - gpubox files, a metafits file (and maybe mwaf files),
        // - a measurement set (and maybe a metafits file), or
        // - uvfits files.
        // If none or multiple of these possibilities are met, then we must fail.
        let input_data_types = match data {
            Some(strings) => InputDataTypes::new(&strings)
                .map_err(|e| InvalidArgsError::InputFile(e.to_string()))?,
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
                // Ensure that there's only one metafits.
                let meta = if meta.len() > 1 {
                    return Err(InvalidArgsError::MultipleMetafits(meta));
                } else {
                    meta.first()
                };

                let pfb_flavour = match pfb_flavour {
                    None => DEFAULT_PFB_FLAVOUR,
                    Some(s) => match PfbFlavour::from_str(&s.to_lowercase()) {
                        Err(_) => return Err(InvalidArgsError::ParsePfbFlavour(s)),
                        Ok(p) => p,
                    },
                };

                debug!("gpubox files: {:?}", &gpuboxes);
                debug!("mwaf files: {:?}", &mwafs);

                let input_data = RawDataReader::new(
                    meta,
                    &gpuboxes,
                    mwafs.as_deref(),
                    pfb_flavour,
                    !no_digital_gains,
                    !no_cable_length_correction,
                    !no_geometric_correction,
                )?;

                messages::InputFileDetails::Raw {
                    obsid: input_data.get_obs_context().obsid.unwrap(),
                    gpubox_count: gpuboxes.len(),
                    metafits_file_name: meta.display().to_string(),
                    mwaf: mwafs.as_ref().map(|m| m.len()),
                    pfb: pfb_flavour,
                    digital_gains: !no_digital_gains,
                    cable_length: !no_cable_length_correction,
                    geometric: !no_geometric_correction,
                }
                .print();

                Box::new(input_data)
            }

            // Valid input for reading a measurement set.
            (meta, None, None, Some(ms), None) => {
                // Only one MS is supported at the moment.
                let ms: PathBuf = if ms.len() > 1 {
                    return Err(InvalidArgsError::MultipleMeasurementSets(ms));
                } else {
                    ms.first().clone()
                };

                // Ensure that there's only one metafits.
                let meta: Option<&PathBuf> = match meta.as_ref() {
                    None => None,
                    Some(m) => {
                        if m.len() > 1 {
                            return Err(InvalidArgsError::MultipleMetafits(m.clone()));
                        } else {
                            Some(m.first())
                        }
                    }
                };

                let input_data = MS::new(&ms, meta)?;

                messages::InputFileDetails::MeasurementSet {
                    obsid: input_data.get_obs_context().obsid,
                    file_name: ms.display().to_string(),
                    metafits_file_name: meta.map(|m| m.display().to_string()),
                }
                .print();

                Box::new(input_data)
            }

            // Valid input for reading uvfits files.
            (meta, None, None, None, Some(uvfits)) => {
                // Only one uvfits is supported at the moment.
                let uvfits: PathBuf = if uvfits.len() > 1 {
                    return Err(InvalidArgsError::MultipleUvfits(uvfits));
                } else {
                    uvfits.first().clone()
                };

                // Ensure that there's only one metafits.
                let meta: Option<&PathBuf> = match meta.as_ref() {
                    None => None,
                    Some(m) => {
                        if m.len() > 1 {
                            return Err(InvalidArgsError::MultipleMetafits(m.clone()));
                        } else {
                            Some(m.first())
                        }
                    }
                };

                let input_data = UvfitsReader::new(&uvfits, meta)?;

                messages::InputFileDetails::UvfitsFile {
                    obsid: input_data.get_obs_context().obsid,
                    file_name: uvfits.display().to_string(),
                    metafits_file_name: meta.map(|m| m.display().to_string()),
                }
                .print();

                Box::new(input_data)
            }

            _ => return Err(InvalidArgsError::InvalidDataInput),
        };

        let obs_context = input_data.get_obs_context();

        let array_position = match array_position {
            None => LatLngHeight::new_mwa(),
            Some(pos) => {
                if pos.len() != 3 {
                    return Err(InvalidArgsError::BadArrayPosition { pos });
                }
                LatLngHeight {
                    longitude_rad: pos[0].to_radians(),
                    latitude_rad: pos[1].to_radians(),
                    height_metres: pos[2],
                }
            }
        };

        let timesteps_to_use = {
            match timesteps {
                None => Vec1::try_from_vec(obs_context.unflagged_timesteps.clone())
                    .map_err(|_| InvalidArgsError::NoTimesteps)?,
                Some(mut ts) => {
                    // Make sure there are no duplicates.
                    let timesteps_hashset: HashSet<&usize> = ts.iter().collect();
                    if timesteps_hashset.len() != ts.len() {
                        return Err(InvalidArgsError::DuplicateTimesteps);
                    }

                    // Ensure that all specified timesteps are actually available.
                    for t in &ts {
                        if !(0..obs_context.timestamps.len()).contains(t) {
                            return Err(InvalidArgsError::UnavailableTimestep {
                                got: *t,
                                last: obs_context.timestamps.len() - 1,
                            });
                        }
                    }

                    ts.sort_unstable();
                    Vec1::try_from_vec(ts).map_err(|_| InvalidArgsError::NoTimesteps)?
                }
            }
        };

        let precession_info = precess_time(
            obs_context.phase_centre,
            obs_context.timestamps[*timesteps_to_use.first()],
            array_position.longitude_rad,
            array_position.latitude_rad,
        );

        // The length of the tile XYZ collection is the total number of tiles in
        // the array, even if some tiles are flagged.
        let total_num_tiles = obs_context.tile_xyzs.len();

        // Assign the tile flags.
        let mut flagged_tiles: Vec<usize> =
            match tile_flags {
                Some(flags) => {
                    // We need to convert the strings into antenna indices. The
                    // strings are either indicies themselves or antenna names.
                    let mut flagged_tiles = HashSet::new();

                    for flag in flags {
                        // Try to parse a naked number.
                        let result =
                            match flag.trim().parse().ok() {
                                Some(n) => {
                                    if n >= total_num_tiles {
                                        Err(InvalidArgsError::InvalidTileFlag {
                                            got: n,
                                            max: total_num_tiles - 1,
                                        })
                                    } else {
                                        flagged_tiles.insert(n);
                                        Ok(())
                                    }
                                }
                                None => {
                                    // Check if this is an antenna name.
                                    match obs_context.tile_names.iter().enumerate().find(
                                        |(_, name)| name.to_lowercase() == flag.to_lowercase(),
                                    ) {
                                        // If there are no matches, complain that
                                        // the user input is no good.
                                        None => Err(InvalidArgsError::BadTileFlag(flag)),
                                        Some((i, _)) => {
                                            flagged_tiles.insert(i);
                                            Ok(())
                                        }
                                    }
                                }
                            };
                        if result.is_err() {
                            // If there's a problem, show all the tile names
                            // and their indices to help out the user.
                            info!("All tile indices and names:");
                            obs_context
                                .tile_names
                                .iter()
                                .enumerate()
                                .for_each(|(i, name)| {
                                    info!("    {:3}: {:10}", i, name);
                                });
                            // Propagate the error.
                            result?;
                        }
                    }

                    flagged_tiles.into_iter().sorted().collect::<Vec<_>>()
                }
                None => vec![],
            };
        if !ignore_input_data_tile_flags {
            // Add tiles that have already been flagged by the input data.
            for &obs_tile_flag in &obs_context.flagged_tiles {
                flagged_tiles.push(obs_tile_flag);
            }
        }
        let num_unflagged_tiles = total_num_tiles - flagged_tiles.len();
        if log_enabled!(Debug) {
            debug!("Tile indices, names and statuses:");
            obs_context
                .tile_names
                .iter()
                .enumerate()
                .map(|(i, name)| {
                    let flagged = flagged_tiles.contains(&i);
                    (i, name, if flagged { "  flagged" } else { "unflagged" })
                })
                .for_each(|(i, name, status)| {
                    debug!("    {:3}: {:10}: {}", i, name, status);
                })
        }
        if num_unflagged_tiles == 0 {
            return Err(InvalidArgsError::NoTiles);
        }
        messages::ArrayDetails {
            array_position,
            array_latitude_j2000: Some(precession_info.array_latitude_j2000),
            total_num_tiles,
            num_unflagged_tiles,
            flagged_tiles: flagged_tiles
                .iter()
                .cloned()
                .sorted()
                .map(|i| (obs_context.tile_names[i].clone(), i))
                .collect(),
        }
        .print();

        let dipole_delays = match delays {
            // We have user-provided delays; check that they're are sensible,
            // regardless of whether we actually need them.
            Some(d) => {
                if d.len() != 16 || d.iter().any(|&v| v > 32) {
                    return Err(InvalidArgsError::BadDelays);
                }
                Delays::Partial(d)
            }

            // No delays were provided; use whatever was in the input data.
            None => match obs_context.dipole_delays.as_ref() {
                Some(d) => d.clone(),
                None => return Err(InvalidArgsError::NoDelays),
            },
        };
        let ideal_delays = dipole_delays.get_ideal_delays();
        debug!("Ideal dipole delays: {:?}", ideal_delays);

        let beam: Box<dyn Beam> = if no_beam {
            create_no_beam_object(total_num_tiles)
        } else {
            create_fee_beam_object(
                beam_file,
                total_num_tiles,
                dipole_delays,
                if unity_dipole_gains {
                    None
                } else {
                    // If we don't have dipole gains from the input data, then
                    // we issue a warning that we must assume no dead dipoles.
                    if obs_context.dipole_gains.is_none() {
                        match input_data.get_input_data_type() {
                            VisInputType::MeasurementSet => {
                                warn!("Measurement sets cannot supply dead dipole information.");
                                warn!("Without a metafits file, we must assume all dipoles are alive.");
                                warn!("This will make beam Jones matrices inaccurate in sky-model generation.");
                            }
                            VisInputType::Uvfits => {
                                warn!("uvfits files cannot supply dead dipole information.");
                                warn!("Without a metafits file, we must assume all dipoles are alive.");
                                warn!("This will make beam Jones matrices inaccurate in sky-model generation.");
                            }
                            VisInputType::Raw => unreachable!(),
                        }
                    }
                    obs_context.dipole_gains.clone()
                },
            )?
        };
        let beam_file = beam.get_beam_file();
        debug!("Beam file: {beam_file:?}");

        // Set up frequency information. Determine all of the fine-channel flags.
        let mut flagged_fine_chans: HashSet<usize> = match fine_chan_flags {
            Some(flags) => flags.into_iter().collect(),
            None => HashSet::new(),
        };
        if !ignore_input_data_fine_channel_flags {
            for &f in &obs_context.flagged_fine_chans {
                flagged_fine_chans.insert(f);
            }
        }
        // Assign the per-coarse-channel fine-channel flags.
        let fine_chan_flags_per_coarse_chan: HashSet<usize> = {
            let mut out_flags = match fine_chan_flags_per_coarse_chan {
                Some(flags) => flags.into_iter().collect(),
                None => HashSet::new(),
            };
            if !ignore_input_data_fine_channel_flags {
                for &obs_fine_chan_flag in &obs_context.flagged_fine_chans_per_coarse_chan {
                    out_flags.insert(obs_fine_chan_flag);
                }
            }
            out_flags
        };
        if !ignore_input_data_fine_channel_flags {
            for (i, _) in obs_context.coarse_chan_nums.iter().enumerate() {
                for f in &fine_chan_flags_per_coarse_chan {
                    flagged_fine_chans.insert(f + obs_context.num_fine_chans_per_coarse_chan * i);
                }
            }
        }
        let mut unflagged_fine_chan_freqs = vec![];
        for (i_chan, &freq) in obs_context.fine_chan_freqs.iter().enumerate() {
            if !flagged_fine_chans.contains(&i_chan) {
                unflagged_fine_chan_freqs.push(freq as f64);
            }
        }
        if log_enabled!(Debug) {
            let unflagged_fine_chans: Vec<_> = (0..obs_context.fine_chan_freqs.len())
                .into_iter()
                .filter(|i_chan| !flagged_fine_chans.contains(i_chan))
                .collect();
            match unflagged_fine_chans.as_slice() {
                [] => (),
                [f] => debug!("Only unflagged fine-channel: {}", f),
                [f_0, .., f_n] => {
                    debug!("First unflagged fine-channel: {}", f_0);
                    debug!("Last unflagged fine-channel:  {}", f_n);
                }
            }

            let fine_chan_flags_vec = flagged_fine_chans.iter().sorted().collect::<Vec<_>>();
            debug!("Flagged fine-channels: {:?}", fine_chan_flags_vec);
        }
        if unflagged_fine_chan_freqs.is_empty() {
            return Err(InvalidArgsError::NoChannels);
        }

        messages::ObservationDetails {
            dipole_delays: ideal_delays,
            beam_file,
            num_tiles_with_dead_dipoles: if unity_dipole_gains {
                None
            } else {
                obs_context.dipole_gains.as_ref().map(|array| {
                    array
                        .outer_iter()
                        .filter(|tile_dipole_gains| {
                            tile_dipole_gains.iter().any(|g| g.abs() < f64::EPSILON)
                        })
                        .count()
                })
            },
            phase_centre: obs_context.phase_centre,
            pointing_centre: obs_context.pointing_centre,
            lmst: precession_info.lmst,
            lmst_j2000: precession_info.lmst_j2000,
            available_timesteps: &obs_context.all_timesteps,
            unflagged_timesteps: &obs_context.unflagged_timesteps,
            using_timesteps: &timesteps_to_use,
            first_timestep: Some(obs_context.timestamps[*timesteps_to_use.first()]),
            last_timestep: if timesteps_to_use.len() > 1 {
                Some(obs_context.timestamps[*timesteps_to_use.last()])
            } else {
                None
            },
            time_res_seconds: obs_context.time_res,
            total_num_channels: obs_context.fine_chan_freqs.len(),
            num_unflagged_channels: unflagged_fine_chan_freqs.len(),
            flagged_chans_per_coarse_chan: &obs_context.flagged_fine_chans_per_coarse_chan,
            first_freq_hz: Some(unflagged_fine_chan_freqs[0]),
            last_freq_hz: if unflagged_fine_chan_freqs.len() > 1 {
                Some(*unflagged_fine_chan_freqs.last().unwrap())
            } else {
                None
            },
            freq_res_hz: obs_context.freq_res,
        }
        .print();

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
                                trace!("{} is a solution output", file.display());
                                can_write_to_file(&file)?;
                                cal_sols.push((sol_type, file));
                            },
                            (None, Some(vis_type)) => {
                                trace!("{} is a visibility output", file.display());
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

        // Set up the timeblocks.
        let time_average_factor = parse_time_average_factor(
            obs_context.time_res,
            time_average_factor,
            *timesteps_to_use.last() - *timesteps_to_use.first() + 1,
        )
        .map_err(|e| match e {
            AverageFactorError::Zero => InvalidArgsError::CalTimeFactorZero,
            AverageFactorError::NotInteger => InvalidArgsError::CalTimeFactorNotInteger,
            AverageFactorError::NotIntegerMultiple { out, inp } => {
                InvalidArgsError::CalTimeResNotMulitple { out, inp }
            }
            AverageFactorError::Parse(e) => InvalidArgsError::ParseCalTimeAverageFactor(e),
        })?;
        // Check that the factor is not too big.
        let time_average_factor = if time_average_factor > timesteps_to_use.len() {
            warn!(
                "Cannot average {} timesteps during calibration; only {} are being used. Capping.",
                time_average_factor,
                timesteps_to_use.len()
            );
            timesteps_to_use.len()
        } else {
            time_average_factor
        };

        let timeblocks = timesteps_to_timeblocks(
            &obs_context.timestamps,
            time_average_factor,
            &timesteps_to_use,
        );
        // There must be at least one timeblock for calibration.
        let timeblocks =
            Vec1::try_from_vec(timeblocks).map_err(|_| InvalidArgsError::NoTimesteps)?;

        // Set up the chanblocks.
        let freq_average_factor =
            parse_freq_average_factor(obs_context.freq_res, freq_average_factor, 1).map_err(
                |e| match e {
                    AverageFactorError::Zero => InvalidArgsError::CalFreqFactorZero,
                    AverageFactorError::NotInteger => InvalidArgsError::CalFreqFactorNotInteger,
                    AverageFactorError::NotIntegerMultiple { out, inp } => {
                        InvalidArgsError::CalFreqResNotMulitple { out, inp }
                    }
                    AverageFactorError::Parse(e) => InvalidArgsError::ParseCalFreqAverageFactor(e),
                },
            )?;
        // Check that the factor is not too big.
        let freq_average_factor = if freq_average_factor > unflagged_fine_chan_freqs.len() {
            warn!(
                "Cannot average {} channels; only {} are being used. Capping.",
                freq_average_factor,
                unflagged_fine_chan_freqs.len()
            );
            unflagged_fine_chan_freqs.len()
        } else {
            freq_average_factor
        };

        let fences = channels_to_chanblocks(
            &obs_context.fine_chan_freqs,
            obs_context.freq_res,
            freq_average_factor,
            &flagged_fine_chans,
        );
        // There must be at least one chanblock for calibration.
        match fences.as_slice() {
            // No fences is the same as no chanblocks.
            [] => return Err(InvalidArgsError::NoChannels),
            [f] => {
                // Check that the chanblocks aren't all flagged.
                if f.chanblocks.is_empty() {
                    return Err(InvalidArgsError::NoChannels);
                }
            }
            [f, ..] => {
                // Check that the chanblocks aren't all flagged.
                if f.chanblocks.is_empty() {
                    return Err(InvalidArgsError::NoChannels);
                }
                warn!("\"Picket fence\" data detected. Only the first contiguous band will be used as this is not well supported right now.");
            }
        }
        let fences = Vec1::try_from_vec(fences).unwrap();

        // XXX(Dev): TileBaselineMaps logic might fit inside FlagContext
        let tile_baseline_maps = TileBaselineMaps::new(total_num_tiles, &flagged_tiles);

        let (unflagged_tile_xyzs, unflagged_tile_names): (Vec<XyzGeodetic>, Vec<String>) =
            obs_context
                .tile_xyzs
                .par_iter()
                .zip(obs_context.tile_names.par_iter())
                .enumerate()
                .filter(|(tile_index, _)| !flagged_tiles.contains(tile_index))
                .map(|(_, (xyz, name))| (*xyz, name.clone()))
                .unzip();

        // Set baseline weights from UVW cuts. Use a lambda from the centroid
        // frequency if UVW cutoffs are specified as wavelengths.
        let freq_centroid = obs_context
            .fine_chan_freqs
            .iter()
            .map(|&u| u as f64)
            .sum::<f64>()
            / obs_context.fine_chan_freqs.len() as f64;
        let lambda = marlu::constants::VEL_C / freq_centroid;
        let (uvw_min, uvw_min_metres) = {
            let (quantity, unit) = parse_wavelength(uvw_min.as_deref().unwrap_or(DEFAULT_UVW_MIN))
                .map_err(InvalidArgsError::ParseUvwMin)?;
            match unit {
                WavelengthUnit::M => ((quantity, unit), quantity),
                WavelengthUnit::L => ((quantity, unit), quantity * lambda),
            }
        };
        let (uvw_max, uvw_max_metres) = match uvw_max {
            None => ((f64::INFINITY, WavelengthUnit::M), f64::INFINITY),
            Some(s) => {
                let (quantity, unit) =
                    parse_wavelength(&s).map_err(InvalidArgsError::ParseUvwMax)?;
                match unit {
                    WavelengthUnit::M => ((quantity, unit), quantity),
                    WavelengthUnit::L => ((quantity, unit), quantity * lambda),
                }
            }
        };

        let (baseline_weights, num_flagged_baselines) = {
            let mut baseline_weights = vec![
                1.0;
                tile_baseline_maps
                    .unflagged_cross_baseline_to_tile_map
                    .len()
            ];
            let uvws = xyzs_to_cross_uvws_parallel(
                &unflagged_tile_xyzs,
                obs_context
                    .phase_centre
                    .to_hadec(precession_info.lmst_j2000),
            );
            assert_eq!(baseline_weights.len(), uvws.len());
            let uvw_min = uvw_min_metres.powi(2);
            let uvw_max = uvw_max_metres.powi(2);
            let mut num_flagged_baselines = 0;
            for (uvw, baseline_weight) in uvws.into_iter().zip(baseline_weights.iter_mut()) {
                let uvw_length = uvw.u.powi(2) + uvw.v.powi(2) + uvw.w.powi(2);
                if uvw_length < uvw_min || uvw_length > uvw_max {
                    *baseline_weight = 0.0;
                    num_flagged_baselines += 1;
                }
            }
            (baseline_weights, num_flagged_baselines)
        };

        // Make sure the calibration thresholds are sensible.
        let mut stop_threshold = stop_thresh.unwrap_or(DEFAULT_STOP_THRESHOLD);
        let min_threshold = min_thresh.unwrap_or(DEFAULT_MIN_THRESHOLD);
        if stop_threshold > min_threshold {
            warn!("Specified stop threshold ({}) is bigger than the min. threshold ({}); capping the stop threshold.", stop_threshold, min_threshold);
            stop_threshold = min_threshold;
        }
        let max_iterations = max_iterations.unwrap_or(DEFAULT_MAX_ITERATIONS);

        messages::CalibrationDetails {
            timesteps_per_timeblock: time_average_factor,
            channels_per_chanblock: freq_average_factor,
            num_timeblocks: timeblocks.len(),
            num_chanblocks: fences.first().chanblocks.len(),
            uvw_min,
            uvw_max,
            num_calibration_baselines: baseline_weights.len() - num_flagged_baselines,
            total_num_baselines: baseline_weights.len(),
            lambda,
            freq_centroid,
            min_threshold,
            stop_threshold,
            max_iterations,
        }
        .print();

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
                    Err(e) => return Err(InvalidArgsError::from(e)),
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
            &obs_context.coarse_chan_freqs,
            beam.deref(),
            num_sources,
            source_dist_cutoff.unwrap_or(DEFAULT_CUTOFF_DISTANCE),
            veto_threshold.unwrap_or(DEFAULT_VETO_THRESHOLD),
        )?;
        if source_list.is_empty() {
            return Err(InvalidArgsError::NoSourcesAfterVeto);
        }

        messages::SkyModelDetails {
            source_list: &source_list,
        }
        .print();

        // Handle output visibility arguments.
        let (output_vis_time_average_factor, output_vis_freq_average_factor) =
            if output_vis_filenames.is_empty() {
                // If we're not writing out calibrated visibilities but arguments
                // are set for them, issue warnings.
                match (output_vis_time_average, output_vis_freq_average) {
                    (None, None) => (),
                    (time, freq) => {
                        warn!("Not writing out calibrated visibilities, but");
                        if time.is_some() {
                            warn!("  output_vis_time_average is set");
                        }
                        if freq.is_some() {
                            warn!("  output_vis_freq_average is set");
                        }
                    }
                }
                (1, 1)
            } else {
                // Parse and verify user input (specified resolutions must
                // evenly divide the input data's resolutions).
                let time_factor =
                    parse_time_average_factor(obs_context.time_res, output_vis_time_average, 1)
                        .map_err(|e| match e {
                            AverageFactorError::Zero => {
                                InvalidArgsError::OutputVisTimeAverageFactorZero
                            }
                            AverageFactorError::NotInteger => {
                                InvalidArgsError::OutputVisTimeFactorNotInteger
                            }
                            AverageFactorError::NotIntegerMultiple { out, inp } => {
                                InvalidArgsError::OutputVisTimeResNotMulitple { out, inp }
                            }
                            AverageFactorError::Parse(e) => {
                                InvalidArgsError::ParseOutputVisTimeAverageFactor(e)
                            }
                        })?;
                let freq_factor =
                    parse_freq_average_factor(obs_context.freq_res, output_vis_freq_average, 1)
                        .map_err(|e| match e {
                            AverageFactorError::Zero => {
                                InvalidArgsError::OutputVisFreqAverageFactorZero
                            }
                            AverageFactorError::NotInteger => {
                                InvalidArgsError::OutputVisFreqFactorNotInteger
                            }
                            AverageFactorError::NotIntegerMultiple { out, inp } => {
                                InvalidArgsError::OutputVisFreqResNotMulitple { out, inp }
                            }
                            AverageFactorError::Parse(e) => {
                                InvalidArgsError::ParseOutputVisFreqAverageFactor(e)
                            }
                        })?;

                (time_factor, freq_factor)
            };

        messages::OutputFileDetails {
            output_solutions: &output_solutions_filenames
                .iter()
                .map(|p| p.1.clone())
                .collect::<Vec<_>>(),
            output_vis: &output_vis_filenames
                .iter()
                .map(|p| p.1.clone())
                .collect::<Vec<_>>(),
            input_vis_time_res: obs_context.time_res,
            input_vis_freq_res: obs_context.freq_res,
            output_vis_time_average_factor,
            output_vis_freq_average_factor,
        }
        .print();

        Ok(CalibrateParams {
            input_data,
            beam,
            source_list,
            model_file,
            flagged_tiles,
            baseline_weights,
            timeblocks,
            timesteps: timesteps_to_use,
            freq_average_factor,
            fences,
            unflagged_fine_chan_freqs,
            flagged_fine_chans,
            tile_to_unflagged_cross_baseline_map: tile_baseline_maps
                .tile_to_unflagged_cross_baseline_map,
            unflagged_tile_names,
            unflagged_tile_xyzs,
            array_position,
            max_iterations,
            stop_threshold,
            min_threshold,
            output_solutions_filenames,
            output_vis_filenames,
            output_vis_time_average_factor,
            output_vis_freq_average_factor,
            no_progress_bars,
            #[cfg(feature = "cuda")]
            use_cpu_for_modelling: cpu,
        })
    }

    pub(crate) fn get_obs_context(&self) -> &ObsContext {
        self.input_data.get_obs_context()
    }

    pub(crate) fn get_total_num_tiles(&self) -> usize {
        self.get_obs_context().tile_xyzs.len()
    }

    pub(crate) fn get_num_unflagged_tiles(&self) -> usize {
        self.get_total_num_tiles() - self.flagged_tiles.len()
    }

    /// The number of calibration timesteps.
    pub(crate) fn get_num_timesteps(&self) -> usize {
        self.timeblocks
            .iter()
            .fold(0, |acc, tb| acc + tb.range.len())
    }

    /// The number of unflagged cross-correlation baselines.
    pub(crate) fn get_num_unflagged_cross_baselines(&self) -> usize {
        let n = self.unflagged_tile_xyzs.len();
        (n * (n - 1)) / 2
    }

    /// Get the sorted *cross-correlation* pairs of antennas for all unflagged
    /// *cross-correlation* baselines. e.g. In a 128T observation, if tiles 0
    /// and 1 are unflagged, then the first baseline is (0,1), and the first
    /// element here is (0,1).
    pub(crate) fn get_ant_pairs(&self) -> Vec<(usize, usize)> {
        self.tile_to_unflagged_cross_baseline_map
            .keys()
            .cloned()
            .sorted()
            .collect()
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
            &self.flagged_fine_chans,
        )
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
fn can_write_to_file(file: &Path) -> Result<(), InvalidArgsError> {
    trace!("Testing whether we can write to {}", file.display());

    if file.is_dir() {
        let exists = can_write_to_dir(file)?;
        if exists {
            warn!("Will overwrite the existing directory '{}'", file.display());
        }
    } else {
        let exists = can_write_to_file_inner(file)?;
        if exists {
            warn!("Will overwrite the existing file '{}'", file.display());
        }
    }

    Ok(())
}

/// Iterate over all of the files and subdirectories of a directory and test
/// whether we can write to them. Note that testing whether directories are
/// writable is very weak; in my testing, changing a subdirectories owner to
/// root and running this function suggested that the file was writable, but it
/// was not. This appears to be a limitation of operating systems, and there's
/// not even a reliable way of checking if *your* user is able to write to a
/// directory. Files are much more rigorously tested.
fn can_write_to_dir(dir: &Path) -> Result<bool, InvalidArgsError> {
    let exists = dir.exists();

    let metadata = std::fs::metadata(dir)?;
    let permissions = metadata.permissions();
    if permissions.readonly() {
        return Err(InvalidArgsError::FileNotWritable {
            file: dir.display().to_string(),
        });
    }

    // Test whether every single entry in `dir` is writable.
    for entry in std::fs::read_dir(dir)? {
        let entry = entry?.path();
        if entry.is_file() {
            can_write_to_file_inner(&entry)?;
        } else if entry.is_dir() {
            can_write_to_dir(&entry)?;
        }
    }

    Ok(exists)
}

fn can_write_to_file_inner(file: &Path) -> Result<bool, InvalidArgsError> {
    let file_exists = file.exists();

    match OpenOptions::new()
        .write(true)
        .create(true)
        .open(&file)
        .map_err(|e| e.kind())
    {
        // File is writable.
        Ok(_) => {
            // If the file in question didn't already exist, `OpenOptions::new`
            // creates it as part of its work. We don't want to keep the 0-sized
            // file; remove it if it didn't exist before.
            if !file_exists {
                std::fs::remove_file(file).map_err(InvalidArgsError::IO)?;
            }
        }

        // File doesn't exist. Attempt to make the directories leading up to the
        // file; if this fails, then we can't write the file anyway.
        Err(std::io::ErrorKind::NotFound) => {
            if let Some(p) = file.parent() {
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
                file: file.display().to_string(),
            })
        }

        Err(e) => {
            return Err(InvalidArgsError::IO(e.into()));
        }
    }

    Ok(file_exists)
}
