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
pub(crate) mod ranked_source;

pub(crate) use error::*;
use filenames::InputDataTypes;
pub(crate) use freq::*;
pub(crate) use ranked_source::*;

use std::collections::{HashMap, HashSet};
use std::fs::OpenOptions;
use std::ops::Deref;
use std::path::{Path, PathBuf};
use std::str::FromStr;

use log::{debug, info, trace, warn};
use mwalib::{MWA_LATITUDE_RADIANS, MWA_LONGITUDE_RADIANS};
use rayon::prelude::*;

use crate::calibrate::veto::veto_sources;
use crate::data_formats::*;
use crate::precession::precess_time;
use crate::{glob::*, *};
use mwa_hyperdrive_core::beam::*;
use mwa_hyperdrive_core::jones::cache::JonesCache;
use mwa_hyperdrive_core::*;
use mwa_hyperdrive_srclist::SourceListType;

/// Parameters needed to perform calibration.
pub struct CalibrateParams {
    /// Interface to the MWA data, and metadata on the input data.
    // pub(crate) input_data: Box<dyn InputData>,
    pub(crate) input_data: Box<dyn InputData>,

    /// Beam struct.
    pub(crate) beam: Box<dyn Beam>,

    /// A shared cache of Jones matrices. This field should be used to generate
    /// Jones matrices and populate the cache.
    pub(crate) jones_cache: mwa_hyperdrive_core::jones::cache::JonesCache,

    /// The sky-model source list.
    pub(crate) source_list: mwa_hyperdrive_core::SourceList,

    /// The optional sky-model visibilities file. If specified, it will be
    /// written to during calibration.
    pub(crate) model_file: Option<PathBuf>,

    /// A list of source names sorted by flux density (brightest to dimmest).
    ///
    /// `source_list` can't be sorted, so this is used to index the source list.
    // Not currently used.
    _ranked_sources: Vec<RankedSource>,

    /// Which tiles are flagged? This field contains flags that are
    /// user-specified as well as whatever was already flagged in the supplied
    /// data.
    ///
    /// These values correspond to those from the "Antenna" column in HDU 2 of
    /// the metafits file. Zero indexed.
    pub(crate) tile_flags: HashSet<usize>,

    /// Channel- and frequency-related parameters required for calibration.
    pub(crate) freq: FrequencyParams,

    /// The target time resolution \[seconds\]. This is kept optional in case in
    /// the input data has only one time step, and therefore no resolution.
    ///
    /// e.g. If the input data is in 0.5s resolution and this variable was 4s,
    /// then we average 8 scans worth of time data when calibrating.
    pub(crate) time_res: Option<f64>,

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
    /// the first unflagged baseline (i.e. 0) is between tile 0 and tile 2.
    ///
    /// This exists because some tiles may be flagged, so some baselines may be
    /// flagged.
    pub(crate) unflagged_baseline_to_tile_map: HashMap<usize, (usize, usize)>,

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

    /// The path to the file where the calibration solutions are written.
    /// Supported formats are .fits and .bin (which is the "André calibrate
    /// format")
    pub(crate) output_solutions_filename: PathBuf,
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
            output_solutions_filename,
            model_filename,
            beam_file,
            no_beam,
            source_list,
            source_list_type,
            num_sources,
            source_dist_cutoff,
            veto_threshold,
            // time_res,
            // freq_res,
            timesteps,
            tile_flags,
            ignore_input_data_tile_flags,
            ignore_input_data_fine_channels_flags,
            delays,
            fine_chan_flags_per_coarse_chan,
            fine_chan_flags,
            max_iterations,
            stop_thresh,
            min_thresh,
            array_longitude_deg,
            array_latitude_deg,
        }: super::args::CalibrateUserArgs,
    ) -> Result<Self, InvalidArgsError> {
        let mut dipole_delays = match (delays, no_beam) {
            // Check that delays are sensible, regardless if we actually need
            // them.
            (Some(d), _) => {
                if d.len() != 16 || d.iter().any(|&v| v > 32) {
                    return Err(InvalidArgsError::BadDelays);
                }
                Delays::Available(d)
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
                    _ => panic!("TODO: support many uvfits files"),
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

        let beam: Box<dyn Beam> = create_beam_object(no_beam, dipole_delays, beam_file)?;

        // Handle potential issues with the output calibration solutions file.
        let output_solutions_filename = PathBuf::from(
            output_solutions_filename
                .unwrap_or_else(|| DEFAULT_OUTPUT_SOLUTIONS_FILENAME.to_string()),
        );
        // Is the output file type supported?
        match &output_solutions_filename
            .extension()
            .and_then(|os_str| os_str.to_str())
        {
            Some("fits") | Some("bin") => (),
            ext => {
                return Err(InvalidArgsError::OutputSolutionsFileType {
                    ext: ext.unwrap_or("<no extension>").to_string(),
                })
            }
        }
        // Check if the file exists. If so, check we can write to it. If we
        // aren't able to write to the file, then handle the problem here. The
        // alternative is complaining that the file can't be written to after
        // calibration is finished, which would mean that the time spent
        // calibrating was wasted. This code _doesn't_ alter the file if it
        // exists.
        if output_solutions_filename.exists() {
            match OpenOptions::new()
                .write(true)
                .create(true)
                .open(&output_solutions_filename)
                .map_err(|io_error| io_error.kind())
            {
                Ok(_) => warn!(
                    "Will overwrite the existing calibration solutions file '{}'",
                    output_solutions_filename.display()
                ),
                Err(std::io::ErrorKind::PermissionDenied) => {
                    return Err(InvalidArgsError::FileNotWritable {
                        file: output_solutions_filename.display().to_string(),
                    })
                }
                Err(e) => return Err(InvalidArgsError::IO(e.into())),
            }
        }

        // Handle potential issues with the output model file.
        let model_file = match model_filename {
            None => None,
            Some(filename) => {
                let pb = PathBuf::from(filename);
                // Is the output file type supported?
                match &pb.extension().and_then(|os_str| os_str.to_str()) {
                    Some("uvfits") => (),
                    ext => {
                        return Err(InvalidArgsError::ModelFileType {
                            ext: ext.unwrap_or("<no extension>").to_string(),
                        })
                    }
                }
                // Check if the file exists. If so, check we can write to it. If we
                // aren't able to write to the file, then handle the problem here.
                if pb.exists() {
                    match OpenOptions::new()
                        .write(true)
                        .create(true)
                        .open(&pb)
                        .map_err(|io_error| io_error.kind())
                    {
                        Ok(_) => warn!(
                            "Will overwrite the existing sky-model file '{}'",
                            pb.display()
                        ),
                        Err(std::io::ErrorKind::PermissionDenied) => {
                            return Err(InvalidArgsError::FileNotWritable {
                                file: pb.display().to_string(),
                            })
                        }
                        Err(e) => return Err(InvalidArgsError::IO(e.into())),
                    }
                };

                Some(pb)
            }
        };

        let obs_context = input_data.get_obs_context();
        let freq_context = input_data.get_freq_context();

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
        info!(
            "Available timesteps (exclusive): {:?}",
            0..obs_context.timesteps.len()
        );
        info!(
            "Unflagged timesteps (exclusive): {:?}",
            obs_context.unflagged_timestep_indices
        );
        {
            // We don't require the timesteps to be used in calibration to be
            // sequential. But if they are, it looks a bit neater to print them out
            // as a range rather than individual indicies.
            info!(
                "{}",
                range_or_comma_separated(&timesteps_to_use, Some("Using timesteps"))
            );
        }

        let array_longitude = match (array_longitude_deg, obs_context.array_longitude_rad) {
            (Some(array_longitude_deg), _) => array_longitude_deg.to_radians(),
            (None, Some(input_data_long)) => input_data_long,
            (None, None) => {
                warn!("Assuming that the input array is at the MWA Earth coordinates");
                MWA_LONGITUDE_RADIANS
            }
        };
        let array_latitude = match (array_latitude_deg, obs_context.array_latitude_rad) {
            (Some(array_latitude_deg), _) => array_latitude_deg.to_radians(),
            (None, Some(input_data_lat)) => input_data_lat,
            (None, None) => MWA_LATITUDE_RADIANS,
        };

        // The length of the tile XYZ collection is the total number of tiles in
        // the array, even if some tiles are flagged.
        let total_num_tiles = obs_context.tile_xyzs.len();
        info!("Total number of tiles: {}", total_num_tiles);

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
        let num_unflagged_tiles = total_num_tiles - tile_flags.len();
        info!("Total number of unflagged tiles: {}", num_unflagged_tiles);
        {
            // Print out the tile flags. Use a vector to sort ascendingly.
            let mut tile_flags = tile_flags.iter().collect::<Vec<_>>();
            tile_flags.sort_unstable();
            info!("Tile flags: {:?}", tile_flags);
        }

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
        {
            let mut fine_chan_flags_per_coarse_chan_vec =
                fine_chan_flags_per_coarse_chan.iter().collect::<Vec<_>>();
            fine_chan_flags_per_coarse_chan_vec.sort_unstable();
            info!(
                "Fine-channel flags per coarse channel: {:?}",
                fine_chan_flags_per_coarse_chan_vec
            );
        }
        debug!(
            "Observation's fine-channel flags per coarse channel: {:?}",
            obs_context.fine_chan_flags_per_coarse_chan
        );

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
        {
            let mut fine_chan_flags_vec = fine_chan_flags.iter().collect::<Vec<_>>();
            fine_chan_flags_vec.sort_unstable();
            debug!("All fine-channel flags: {:?}", fine_chan_flags_vec);
        }

        // Set up frequency information.
        let native_freq_res = freq_context.native_fine_chan_width;
        let freq_res = native_freq_res;
        // let freq_res = freq_res.unwrap_or(native_freq_res);
        // if freq_res % native_freq_res != 0.0 {
        //     return Err(InvalidArgsError::InvalidFreqResolution {
        //         got: freq_res,
        //         native: native_freq_res,
        //     });
        // }

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
        let freq_struct = FrequencyParams {
            res: freq_res,
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
            &obs_context.phase_centre,
            &obs_context.timesteps[0],
            array_longitude,
            array_latitude,
        );

        info!(
            "Phase centre:                   {}",
            &obs_context.phase_centre
        );
        info!(
            "Phase centre (J2000):           {}",
            precession_info
                .hadec_j2000
                .to_radec(precession_info.lmst_j2000)
        );
        if let Some(pc) = &obs_context.pointing_centre {
            info!("Pointing centre:                {}", pc);
        }
        info!(
            "LMST of first timestep:         {:.4}°",
            precession_info.lmst.to_degrees()
        );
        info!(
            "LMST of first timestep (J2000): {:.4}°",
            precession_info.lmst_j2000.to_degrees()
        );
        info!(
            "Array longitude:                {:.4}°",
            array_longitude.to_degrees()
        );
        info!(
            "Array latitude:                 {:.4}°",
            array_latitude.to_degrees()
        );
        info!(
            "Array latitude (J2000):         {:.4}°",
            precession_info.array_latitude_j2000.to_degrees()
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
                debug!("Successfully parsed {}-style source list", sl_type);
            }

            sl
        };
        info!("Found {} sources in the source list", source_list.len());
        // Veto any sources that may be troublesome, and/or cap the total number
        // of sources. If the user doesn't specify how many source-list sources
        // to use, then all sources are used.
        if num_sources == Some(0) || source_list.is_empty() {
            return Err(InvalidArgsError::NoSources);
        }
        let _ranked_sources = veto_sources(
            &mut source_list,
            &precession_info
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
        let num_components = source_list
            .values()
            .fold(0, |a, src| a + src.components.len());
        info!(
            "Using {} sources with a total of {} components",
            source_list.len(),
            num_components
        );
        trace!("Using sources: {:?}", source_list.keys());
        if source_list.len() > 10000 {
            warn!("Using more than 10,000 sources!");
        } else if source_list.is_empty() {
            return Err(InvalidArgsError::NoSourcesAfterVeto);
        }

        let native_time_res = obs_context.time_res;
        // let time_res = time_res.or(native_time_res);
        let time_res = native_time_res;
        if let Some(ntr) = native_time_res {
            info!("Input data time resolution: {:.2}s", ntr);
            if let Some(tr) = time_res {
                if tr % ntr != 0.0 {
                    return Err(InvalidArgsError::InvalidTimeResolution {
                        got: tr,
                        native: ntr,
                    });
                }
            }
        }
        // let num_time_steps_to_average = time_res / native_time_res;
        // let timesteps = (0..context.timesteps.len() / num_time_steps_to_average as usize).collect();

        let tile_baseline_maps = generate_tile_baseline_maps(total_num_tiles, &tile_flags);

        let (unflagged_tile_xyzs, unflagged_tile_names) = obs_context
            .tile_xyzs
            .par_iter()
            .zip(obs_context.tile_names.par_iter())
            .enumerate()
            .filter(|(tile_index, _)| !tile_flags.contains(tile_index))
            .map(|(_, (xyz, name))| (xyz.clone(), name.clone()))
            .unzip();

        // Make sure the thresholds are sensible.
        let mut stop_threshold = stop_thresh.unwrap_or(DEFAULT_STOP_THRESHOLD);
        let min_threshold = min_thresh.unwrap_or(DEFAULT_MIN_THRESHOLD);
        if stop_threshold > min_threshold {
            warn!("Specified stop threshold ({}) is bigger than the min. threshold ({}); capping the stop threshold.", stop_threshold, min_threshold);
            stop_threshold = min_threshold;
        }

        Ok(Self {
            input_data,
            beam,
            jones_cache: JonesCache::new(),
            source_list,
            model_file,
            _ranked_sources,
            tile_flags,
            freq: freq_struct,
            time_res,
            timesteps: timesteps_to_use,
            tile_to_unflagged_baseline_map: tile_baseline_maps.tile_to_unflagged_baseline_map,
            unflagged_baseline_to_tile_map: tile_baseline_maps.unflagged_baseline_to_tile_map,
            unflagged_tile_names,
            unflagged_tile_xyzs,
            array_longitude,
            array_latitude,
            max_iterations: max_iterations.unwrap_or(DEFAULT_MAX_ITERATIONS),
            stop_threshold,
            min_threshold,
            output_solutions_filename,
        })
    }
}

fn create_beam_object<T: AsRef<Path>>(
    no_beam: bool,
    delays: Delays,
    beam_file: Option<T>,
) -> Result<Box<dyn Beam>, BeamError> {
    if no_beam {
        info!("Not using a beam");
        Ok(Box::new(NoBeam))
    } else {
        trace!("Setting up FEE beam");
        let beam = if let Some(bf) = beam_file {
            // Set up the FEE beam struct from the specified beam file.
            Box::new(FEEBeam::new(&bf, delays)?)
        } else {
            // Set up the FEE beam struct from the MWA_BEAM_FILE environment
            // variable.
            Box::new(FEEBeam::new_from_env(delays)?)
        };
        info!("Using FEE beam with delays {:?}", beam.get_delays());
        Ok(beam)
    }
}

// /// Get the LST (in radians) from a timestep. `time_res_seconds` refers to the
// /// target time resolution of calibration, *not* the observation's time
// /// resolution.
// ///
// /// The LST is calculated for the middle of the timestep, not the start of it.
// fn lst_from_timestep(timestep: usize, context: &CorrelatorContext, time_res_seconds: f64) -> f64 {
//     let start_lst = context.metafits_context.lst_rad;
//     let factor = SOLAR2SIDEREAL * DS2R;
//     start_lst
//         + factor * (get_diff_in_start_time(context) + time_res_seconds * (timestep as f64 + 0.5))
// }

struct TileBaselineMaps {
    tile_to_unflagged_baseline_map: HashMap<(usize, usize), usize>,
    unflagged_baseline_to_tile_map: HashMap<usize, (usize, usize)>,
}

fn generate_tile_baseline_maps(
    total_num_tiles: usize,
    tile_flags: &HashSet<usize>,
) -> TileBaselineMaps {
    let mut tile_to_unflagged_baseline_map = HashMap::new();
    let mut unflagged_baseline_to_tile_map = HashMap::new();
    let mut bl = 0;
    for tile1 in 0..total_num_tiles {
        if tile_flags.contains(&tile1) {
            continue;
        }
        for tile2 in tile1 + 1..total_num_tiles {
            if tile_flags.contains(&tile2) {
                continue;
            }
            tile_to_unflagged_baseline_map.insert((tile1, tile2), bl);
            unflagged_baseline_to_tile_map.insert(bl, (tile1, tile2));
            bl += 1;
        }
    }

    TileBaselineMaps {
        tile_to_unflagged_baseline_map,
        unflagged_baseline_to_tile_map,
    }
}

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
            format!("{} (exclusive): {}", p, suffix)
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
            format!("{}: {}", p, suffix)
        } else {
            suffix
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tests::{full_obsids::*, reduced_obsids::*, *};

    #[test]
    fn test_generate_tile_baseline_maps() {
        let total_num_tiles = 128;
        let mut tile_flags = HashSet::new();
        let maps = generate_tile_baseline_maps(total_num_tiles, &tile_flags);
        assert_eq!(maps.tile_to_unflagged_baseline_map[&(0, 1)], 0);
        assert_eq!(maps.unflagged_baseline_to_tile_map[&0], (0, 1));

        tile_flags.insert(1);
        let maps = generate_tile_baseline_maps(total_num_tiles, &tile_flags);
        assert_eq!(maps.tile_to_unflagged_baseline_map[&(0, 2)], 0);
        assert_eq!(maps.tile_to_unflagged_baseline_map[&(2, 3)], 126);
        assert_eq!(maps.unflagged_baseline_to_tile_map[&0], (0, 2));
        assert_eq!(maps.unflagged_baseline_to_tile_map[&126], (2, 3));
    }

    #[test]
    fn test_get_flagged_baselines_set() {
        let total_num_tiles = 128;
        let mut tile_flags = HashSet::new();
        let flagged_baselines = get_flagged_baselines_set(total_num_tiles, &tile_flags);
        assert!(flagged_baselines.is_empty());

        tile_flags.insert(127);
        let flagged_baselines = get_flagged_baselines_set(total_num_tiles, &tile_flags);
        assert!(flagged_baselines.contains(&126));
        assert!(flagged_baselines.contains(&252));
        assert!(flagged_baselines.contains(&8127));
    }

    #[test]
    fn test_new_params() {
        let args = get_1090008640_smallest();
        let params = match args.into_params() {
            Ok(p) => p,
            Err(e) => panic!("{}", e),
        };
        // The default time resolution should be 2.0s, as per the metafits.
        assert_abs_diff_eq!(params.time_res.unwrap(), 2.0);
        // The default freq resolution should be 40kHz, as per the metafits.
        assert_abs_diff_eq!(params.freq.res, 40e3, epsilon = 1e-10);
    }

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
