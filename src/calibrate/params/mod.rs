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
mod helpers;
#[cfg(test)]
mod tests;

pub(crate) use error::*;
use filenames::InputDataTypes;
use helpers::*;

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
    Jones, XyzGeodetic,
};
use ndarray::ArrayViewMut2;
use rayon::prelude::*;
use vec1::Vec1;

use super::{solutions::CalSolutionType, CalibrateUserArgs, Fence, Timeblock};
use crate::{
    constants::*,
    context::ObsContext,
    data_formats::*,
    glob::*,
    math::TileBaselineMaps,
    pfb_gains::{PfbFlavour, DEFAULT_PFB_FLAVOUR},
    unit_parsing::{parse_wavelength, WavelengthUnit},
};
use mwa_hyperdrive_beam::{create_fee_beam_object, create_no_beam_object, Beam, Delays};
use mwa_hyperdrive_common::{itertools, log, marlu, ndarray, rayon, vec1};
use mwa_hyperdrive_srclist::{
    constants::*, veto_sources, ComponentCounts, SourceList, SourceListType,
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

    /// The number of time samples to average together during calibration.
    ///
    /// e.g. If the input data is in 0.5s resolution and this variable was 4,
    /// then we average 2s worth of data together during calibration.
    pub(crate) time_average_factor: usize,

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
            ignore_autos,
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
            array_longitude_deg,
            array_latitude_deg,
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

                let input_data = RawDataReader::new(
                    meta,
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
                    PfbFlavour::Jake => info!("Using 'Jake Jones' PFB gains"),
                    PfbFlavour::Cotter2014 => info!("Using 'Cotter 2014' PFB gains"),
                    PfbFlavour::Empirical => info!("Using 'RTS empirical' PFB gains"),
                    PfbFlavour::Levine => info!("Using 'Alan Levine' PFB gains"),
                }
                debug!("gpubox files: {:?}", &gpuboxes);
                match mwafs {
                    Some(_) => info!("Using supplied mwaf flags"),
                    None => warn!("No mwaf flag files supplied"),
                }
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

                let input_data = MS::new(&ms, meta, &mut dipole_delays)?;
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

                let input_data = UvfitsReader::new(&uvfits, meta, &mut dipole_delays)?;
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
        let mut flagged_tiles: Vec<usize> = match tile_flags {
            Some(flags) => {
                // We need to convert the strings into antenna indices. The
                // strings are either indicies themselves or antenna names.
                let mut flagged_tiles = HashSet::new();

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
                            flagged_tiles.insert(n);
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
                                Some((i, _)) => flagged_tiles.insert(i),
                            };
                        }
                    }
                }

                let mut flagged_tiles: Vec<_> = flagged_tiles.into_iter().collect();
                flagged_tiles.sort_unstable();
                flagged_tiles
            }
            None => vec![],
        };
        if !ignore_input_data_tile_flags {
            // Add tiles that have already been flagged by the input data.
            for &obs_tile_flag in &obs_context.flagged_tiles {
                flagged_tiles.push(obs_tile_flag);
            }
        }

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
        // Print out some coordinates, including their precessed counterparts
        // (if applicable).
        let precession_info = precess_time(
            obs_context.phase_centre,
            obs_context.timestamps[*timesteps_to_use.first()],
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
            &obs_context.coarse_chan_freqs,
            beam.deref(),
            num_sources,
            source_dist_cutoff.unwrap_or(DEFAULT_CUTOFF_DISTANCE),
            veto_threshold.unwrap_or(DEFAULT_VETO_THRESHOLD),
        )?;
        if source_list.is_empty() {
            return Err(InvalidArgsError::NoSourcesAfterVeto);
        }

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
                let time_factor = parse_time_average_factor(
                    obs_context
                        .time_res
                        .map(|res| res * time_average_factor as f64),
                    output_vis_time_average,
                    1,
                )
                .map_err(|e| match e {
                    AverageFactorError::Zero => InvalidArgsError::OutputVisTimeAverageFactorZero,
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
                let freq_factor = parse_freq_average_factor(
                    obs_context
                        .freq_res
                        .map(|res| res * freq_average_factor as f64),
                    output_vis_freq_average,
                    1,
                )
                .map_err(|e| match e {
                    AverageFactorError::Zero => InvalidArgsError::OutputVisFreqAverageFactorZero,
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

        let using_autos = if ignore_autos {
            false
        } else {
            obs_context.autocorrelations_present
        };
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
                let freq_centroid = obs_context
                    .fine_chan_freqs
                    .iter()
                    .map(|&u| u as f64)
                    .sum::<f64>()
                    / obs_context.fine_chan_freqs.len() as f64;
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
                assert_eq!(baseline_weights.len(), uvws.len());
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

        let params = CalibrateParams {
            input_data,
            beam,
            source_list,
            model_file,
            flagged_tiles,
            baseline_weights,
            time_average_factor,
            timeblocks,
            timesteps: timesteps_to_use,
            freq_average_factor,
            fences,
            unflagged_fine_chan_freqs,
            flagged_fine_chans,
            tile_to_unflagged_cross_baseline_map: tile_baseline_maps
                .tile_to_unflagged_cross_baseline_map,
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
            no_progress_bars,
            #[cfg(feature = "cuda")]
            use_cpu_for_modelling: cpu,
        };
        let extra_info = ExtraInfo {
            precession_info,
            params: &params,
        };
        extra_info.log_info()?;
        Ok(params)
    }

    pub(crate) fn get_obs_context(&self) -> &ObsContext {
        self.input_data.get_obs_context()
    }

    pub(crate) fn get_total_num_tiles(&self) -> usize {
        self.get_obs_context().tile_xyzs.len()
    }

    pub(crate) fn get_num_unflagged_tiles(&self) -> usize {
        self.get_obs_context().tile_xyzs.len() - self.flagged_tiles.len()
    }

    /// The number of calibration timesteps.
    pub(crate) fn get_num_timesteps(&self) -> usize {
        self.timeblocks
            .iter()
            .fold(0, |acc, tb| acc + tb.range.len())
    }

    /// The number of unflagged baselines, including auto-correlation
    /// "baselines" if these are included.
    // TODO(dev): this is only used in tests
    #[allow(dead_code)]
    pub(crate) fn get_num_unflagged_baselines(&self) -> usize {
        let n = self.unflagged_tile_xyzs.len();
        if self.using_autos {
            (n * (n + 1)) / 2
        } else {
            (n * (n - 1)) / 2
        }
    }

    /// The number of unflagged cross-correlation baselines.
    pub(crate) fn get_num_unflagged_cross_baselines(&self) -> usize {
        let n = self.unflagged_tile_xyzs.len();
        (n * (n - 1)) / 2
    }

    pub(crate) fn get_ant_pairs(&self) -> Vec<(usize, usize)> {
        // TODO(Dev): support autos
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

/// Extra metadata that is useful to report.
struct ExtraInfo<'a> {
    precession_info: PrecessionInfo,

    /// The rest of the parameters.
    params: &'a CalibrateParams,
}

impl<'a> ExtraInfo<'a> {
    fn log_info(self) -> Result<(), InvalidArgsError> {
        let params = self.params;
        let obs_context = params.input_data.get_obs_context();

        info!(
            "Array longitude, latitude:     ({:.4}°, {:.4}°)",
            params.array_longitude.to_degrees(),
            params.array_latitude.to_degrees()
        );
        info!(
            "Array latitude (J2000):                    {:.4}°",
            self.precession_info.array_latitude_j2000.to_degrees()
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
            self.precession_info.lmst.to_degrees()
        );
        info!(
            "LMST of first timestep (J2000): {:.4}°",
            self.precession_info.lmst_j2000.to_degrees()
        );

        let total_num_tiles = params.get_total_num_tiles();
        let num_unflagged_tiles = params.get_num_unflagged_tiles();
        info!("Total number of tiles:           {}", total_num_tiles);
        info!("Number of unflagged tiles:       {}", num_unflagged_tiles);
        {
            // Print out the tile flags. Use a vector to sort ascendingly.
            let mut tile_flags = params.flagged_tiles.iter().collect::<Vec<_>>();
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
                    let flagged = params.flagged_tiles.contains(&i);
                    (i, name, if flagged { "  flagged" } else { "unflagged" })
                })
                .for_each(|(i, name, status)| {
                    debug!("    {:3}: {:10}: {}", i, name, status);
                })
        }

        info!(
            "{}",
            range_or_comma_separated(&obs_context.all_timesteps, Some("Available timesteps:"))
        );
        info!(
            "{}",
            range_or_comma_separated(
                &obs_context.unflagged_timesteps,
                Some("Unflagged timesteps:")
            )
        );
        // We don't require the timesteps to be used in calibration to be
        // sequential. But if they are, it looks a bit neater to print them out
        // as a range rather than individual indicies.
        info!(
            "{}",
            range_or_comma_separated(&self.params.timesteps, Some("Using timesteps:    "))
        );

        match self.params.timesteps.as_slice() {
            [] => unreachable!(),
            [t] => info!(
                "Only timestep (GPS): {:.2}",
                obs_context.timestamps[*t].as_gpst_seconds()
            ),
            [t0, .., tn] => {
                info!(
                    "First timestep (GPS): {:.2}",
                    obs_context.timestamps[*t0].as_gpst_seconds()
                );
                info!(
                    "Last timestep  (GPS): {:.2}",
                    obs_context.timestamps[*tn].as_gpst_seconds()
                );
            }
        }

        match obs_context.time_res {
            Some(native) => {
                info!("Input data time resolution:  {:.2} seconds", native);
            }
            None => info!("Input data time resolution unknown"),
        }
        match obs_context.freq_res {
            Some(freq_res) => {
                info!("Input data freq. resolution: {:.2} kHz", freq_res / 1e3);
            }
            None => info!("Input data freq. resolution unknown"),
        }

        info!(
            "Total number of fine channels:     {}",
            obs_context.fine_chan_freqs.len()
        );
        info!(
            "Number of unflagged fine channels: {}",
            params.unflagged_fine_chan_freqs.len()
        );
        if log_enabled!(Debug) {
            let unflagged_fine_chans: Vec<_> = (0..obs_context.fine_chan_freqs.len())
                .into_iter()
                .filter(|i_chan| !params.flagged_fine_chans.contains(i_chan))
                .collect();
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
            obs_context.flagged_fine_chans_per_coarse_chan
        );
        if log_enabled!(Debug) {
            let mut fine_chan_flags_vec = params.flagged_fine_chans.iter().collect::<Vec<_>>();
            fine_chan_flags_vec.sort_unstable();
            debug!("Flagged fine-channels: {:?}", fine_chan_flags_vec);
        }
        match params.unflagged_fine_chan_freqs.as_slice() {
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
            params.time_average_factor,
        );
        info!(
            "Averaging {} fine-freq. channels into each chanblock",
            params.freq_average_factor,
        );
        info!(
            "Number of calibration timeblocks: {}",
            params.timeblocks.len()
        );
        info!(
            "Number of calibration chanblocks: {}",
            params.fences.first().chanblocks.len()
        );

        let ComponentCounts {
            num_points,
            num_gaussians,
            num_shapelets,
            ..
        } = params.source_list.get_counts();
        let num_components = num_points + num_gaussians + num_shapelets;
        info!(
            "Using {} sources with a total of {} components",
            params.source_list.len(),
            num_components
        );
        info!("{num_points} point components");
        info!("{num_gaussians} Gaussian components");
        info!("{num_shapelets} shapelet components");
        if num_components > 10000 {
            warn!("Using more than 10,000 components!");
        }
        trace!("Using sources: {:?}", params.source_list.keys());

        if !params.output_solutions_filenames.is_empty() {
            info!(
                "Writing calibration solutions to: {}",
                params
                    .output_solutions_filenames
                    .iter()
                    .map(|(_, pb)| pb.display())
                    .join(", ")
            );
        }
        if !params.output_vis_filenames.is_empty() {
            info!(
                "Writing calibrated visibilities to: {}",
                params
                    .output_vis_filenames
                    .iter()
                    .map(|(_, pb)| pb.display())
                    .join(", ")
            );

            info!("Averaging output calibrated visibilities");
            if let Some(tr) = obs_context.time_res {
                info!(
                    "    {}x in time  ({}s)",
                    params.output_vis_time_average_factor,
                    tr * params.output_vis_time_average_factor as f64
                );
            } else {
                info!(
                    "    {}x (only one timestep)",
                    params.output_vis_time_average_factor
                );
            }

            if let Some(fr) = obs_context.freq_res {
                info!(
                    "    {}x in freq. ({}kHz)",
                    params.output_vis_freq_average_factor,
                    fr * params.output_vis_freq_average_factor as f64 / 1000.0
                );
            } else {
                info!(
                    "    {}x (only one fine channel)",
                    params.output_vis_freq_average_factor
                );
            }
        }

        Ok(())
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
fn can_write_to_file(file: &Path) -> Result<(), InvalidArgsError> {
    let file_exists = file.exists();

    match OpenOptions::new()
        .write(true)
        .create(true)
        .open(&file)
        .map_err(|e| e.kind())
    {
        // File is writable.
        Ok(_) => {
            if file_exists {
                warn!("Will overwrite the existing file '{}'", file.display())
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

    Ok(())
}
