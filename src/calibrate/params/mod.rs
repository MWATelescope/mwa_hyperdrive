// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

/*!
Parameters required for calibration and associated functions.

Strategy: Users give arguments to hyperdrive (handled by calibrate::args).
hyperdrive turns arguments into parameters (handled by calibrate::params). Using
this terminology, the code to handle arguments and parameters (and associated
errors) can be neatly split.
 */

pub(crate) mod error;
mod filenames;
pub(crate) mod freq;
pub(crate) mod ranked_source;

pub use error::*;
use filenames::InputDataTypes;
pub(crate) use freq::*;
pub(crate) use ranked_source::*;

use std::collections::{HashMap, HashSet};
use std::fs::OpenOptions;
use std::path::PathBuf;

use log::{debug, info, trace, warn};
use rayon::prelude::*;

use crate::beam::Beam;
use crate::calibrate::veto::veto_sources;
use crate::data_formats::*;
use crate::{glob::*, *};
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

    /// The sky-model source list.
    pub(crate) source_list: mwa_hyperdrive_core::SourceList,

    /// A list of source names sorted by flux density (brightest to dimmest).
    ///
    /// `source_list` can't be sorted, so this is used to index the source list.
    pub(crate) ranked_sources: Vec<RankedSource>,

    /// The number of components in the source list. If all sources have only
    /// one component, then this is equal to the length of the source list.
    pub(crate) num_components: usize,

    /// Which tiles are flagged? This field contains flags that are
    /// user-specified as well as whatever was already flagged in the supplied
    /// data.
    ///
    /// These values correspond to those from the "Antenna" column in HDU 1 of
    /// the metafits file. Zero indexed.
    pub(crate) tile_flags: HashSet<usize>,

    /// Channel- and frequency-related parameters required for calibration.
    pub(crate) freq: FrequencyParams,

    /// The target time resolution \[seconds\].
    ///
    /// e.g. If the input data is in 0.5s resolution and this variable was 4s,
    /// then we average 8 scans worth of time data when calibrating.
    pub(crate) time_res: f64,

    /// A shared cache of Jones matrices. This field should be used to generate
    /// Jones matrices and populate the cache.
    pub(crate) jones_cache: mwa_hyperdrive_core::jones::cache::JonesCache,
    // /// The observation timestep indices, respecting the target time resolution.
    // ///
    // /// If time_res is the same as the observations native resolution, then
    // /// these timesteps are the same as mwalib's timesteps. Otherwise, they need
    // /// to be adjusted.
    // timesteps: Vec<usize>,
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

    /// The unflagged [XyzBaseline]s of the observation \[metres\]. This does
    /// not change over time; it is determined only by the telescope's tile
    /// layout.
    pub(crate) unflagged_baseline_xyz: Vec<XyzBaseline>,

    /// The maximum number of times to iterate when performing "MitchCal".
    pub(crate) max_iterations: usize,

    /// The threshold at which we stop convergence when performing "MitchCal".
    /// This is smaller than `min_threshold`.
    pub(crate) stop_threshold: f32,

    /// The minimum threshold to satisfy convergence when performing "MitchCal".
    /// Reaching this threshold counts as "converged", but it's not as good as
    /// the stop threshold. This is bigger than `stop_threshold`.
    pub(crate) min_threshold: f32,

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
            beam_file,
            no_beam,
            source_list,
            source_list_type,
            num_sources,
            source_dist_cutoff,
            veto_threshold,
            time_res,
            freq_res,
            tile_flags,
            ignore_input_data_tile_flags,
            ignore_input_data_fine_channels_flags,
            fine_chan_flags_per_coarse_chan,
            fine_chan_flags,
            max_iterations,
            stop_thresh,
            min_thresh,
        }: super::args::CalibrateUserArgs,
    ) -> Result<Self, InvalidArgsError> {
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
                let input_data = RawData::new(&meta, &gpuboxes, mwafs.as_deref())?;

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
                let input_data = MS::new(&ms, meta.as_ref())?;
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
            // TODO: Might need a metafits here.
            (None, None, None, None, Some(uvfits_strs)) => {
                todo!();

                // let mut uvfits_pbs = Vec::with_capacity(uvfits_strs.len());
                // for s in uvfits_strs {
                //     let pb = PathBuf::from(s);
                //     // Check that the file actually exists and is readable.
                //     if !pb.exists() || !is_readable(&pb)? {
                //         return Err(InvalidArgsError::BadFile(pb));
                //     }
                //     uvfits_pbs.push(pb);
                // }
                // InputData::Uvfits { paths: uvfits_pbs }
            }

            _ => return Err(InvalidArgsError::InvalidDataInput),
        };

        let beam: Box<dyn Beam> = {
            if no_beam {
                info!("Not using a beam");
                Box::new(beam::NoBeam)
            } else {
                info!("Using FEE beam");
                if let Some(bf) = beam_file {
                    // Set up the beam struct from the specified beam file.
                    Box::new(beam::FEEBeam::new(&bf)?)
                } else {
                    // Set up the beam struct from the MWA_BEAM_FILE environment
                    // variable.
                    Box::new(beam::FEEBeam::new_from_env()?)
                }
            }
        };

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
                    return Err(InvalidArgsError::OutputSolutionsFileNotWritable {
                        file: output_solutions_filename.display().to_string(),
                    })
                }
                Err(e) => return Err(InvalidArgsError::IO(e.into())),
            }
        }

        let obs_context = input_data.get_obs_context();
        let freq_context = input_data.get_freq_context();

        // The length of the tile XYZ collection is the total number of tiles in
        // the array, even if some tiles are flagged.
        let total_num_tiles = obs_context.tile_xyz.len();
        info!("Total number of tiles: {}", total_num_tiles);

        // Assign the tile flags.
        let mut tile_flags: HashSet<usize> = match tile_flags {
            Some(flags) => flags.into_iter().collect(),
            None => HashSet::new(),
        };
        // Add tiles that have already been flagged by the input data.
        for &obs_tile_flag in &obs_context.tile_flags {
            tile_flags.insert(obs_tile_flag);
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
        for &obs_fine_chan_flag in &obs_context.fine_chan_flags_per_coarse_chan {
            fine_chan_flags_per_coarse_chan.insert(obs_fine_chan_flag);
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
        let freq_res = freq_res.unwrap_or(native_freq_res);
        if freq_res % native_freq_res != 0.0 {
            return Err(InvalidArgsError::InvalidFreqResolution {
                got: freq_res,
                native: native_freq_res,
            });
        }

        let num_fine_chans_per_coarse_band =
            freq_context.fine_chan_freqs.len() / freq_context.coarse_chan_freqs.len();
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
            let sl_type = match source_list_type.as_ref() {
                Some(t) => mwa_hyperdrive_srclist::read::parse_source_list_type(&t)?,
                None => SourceListType::Unspecified,
            };
            let (sl, sl_type) =
                match mwa_hyperdrive_srclist::read::read_source_list_file(&sl_pb, Some(sl_type)) {
                    Ok((sl, sl_type)) => (sl, sl_type),
                    Err(e) => {
                        eprintln!("Error when trying to read source list:");
                        return Err(InvalidArgsError::from(e));
                    }
                };

            // If the user didn't specify the source list type, then print out
            // what we found.
            if source_list_type.is_none() {
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
        let ranked_sources = veto_sources(
            &mut source_list,
            &obs_context.pointing,
            obs_context.lst0,
            &obs_context.delays,
            &freq_context.coarse_chan_freqs,
            &beam,
            num_sources,
            source_dist_cutoff.unwrap_or(CUTOFF_DISTANCE),
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
        let time_res = time_res.unwrap_or(native_time_res);
        if time_res % native_time_res != 0.0 {
            return Err(InvalidArgsError::InvalidTimeResolution {
                got: time_res,
                native: native_time_res,
            });
        }
        // let num_time_steps_to_average = time_res / native_time_res;
        // let timesteps = (0..context.timesteps.len() / num_time_steps_to_average as usize).collect();

        let tile_baseline_maps = generate_tile_baseline_maps(total_num_tiles, &tile_flags);
        let flagged_baselines = get_flagged_baselines_set(total_num_tiles, &tile_flags);

        let unflagged_baseline_xyz = obs_context
            .baseline_xyz
            .par_iter()
            .enumerate()
            .filter(|(baseline_index, _)| !flagged_baselines.contains(baseline_index))
            .map(|(_, xyz)| xyz.clone())
            .collect();

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
            source_list,
            ranked_sources,
            num_components,
            tile_flags,
            freq: freq_struct,
            time_res,
            jones_cache: JonesCache::new(),
            tile_to_unflagged_baseline_map: tile_baseline_maps.tile_to_unflagged_baseline_map,
            unflagged_baseline_to_tile_map: tile_baseline_maps.unflagged_baseline_to_tile_map,
            unflagged_baseline_xyz,
            max_iterations: max_iterations.unwrap_or(DEFAULT_MAX_ITERATIONS),
            stop_threshold,
            min_threshold,
            output_solutions_filename,
        })
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
        assert_abs_diff_eq!(params.time_res, 2.0);
        // The default freq resolution should be 40kHz, as per the metafits.
        assert_abs_diff_eq!(params.freq.res, 40e3, epsilon = 1e-10);
    }

    #[test]
    fn test_new_params_time_averaging() {
        // The native time resolution is 2.0s.
        let mut args = get_1090008640_smallest();
        // 4.0 should be a multiple of 2.0s
        args.time_res = Some(4.0);
        let params = match args.into_params() {
            Ok(p) => p,
            Err(e) => panic!("{}", e),
        };
        assert_abs_diff_eq!(params.time_res, 4.0);

        let mut args = get_1090008640();
        // 8.0 should be a multiple of 2.0s
        args.time_res = Some(8.0);
        let params = match args.into_params() {
            Ok(p) => p,
            Err(e) => panic!("{}", e),
        };
        assert_abs_diff_eq!(params.time_res, 8.0);
    }

    #[test]
    fn test_new_params_time_averaging_fail() {
        // The native time resolution is 2.0s.
        let mut args = get_1090008640_smallest();
        // 2.01 is not a multiple of 2.0s
        args.time_res = Some(2.01);
        let result = args.into_params();
        assert!(
            result.is_err(),
            "Expected CalibrateParams to have not been successfully created"
        );

        let mut args = get_1090008640_smallest();
        // 3.0 is not a multiple of 2.0s
        args.time_res = Some(3.0);
        let result = args.into_params();
        assert!(
            result.is_err(),
            "Expected CalibrateParams to have not been successfully created"
        );
    }

    #[test]
    fn test_new_params_freq_averaging() {
        // The native freq. resolution is 40kHz.
        let mut args = get_1090008640_smallest();
        // 80e3 should be a multiple of 40kHz
        args.freq_res = Some(80e3);
        let params = match args.into_params() {
            Ok(p) => p,
            Err(e) => panic!("{}", e),
        };
        assert_abs_diff_eq!(params.freq.res, 80e3, epsilon = 1e-10);

        let mut args = get_1090008640_smallest();
        // 200e3 should be a multiple of 40kHz
        args.freq_res = Some(200e3);
        let params = match args.into_params() {
            Ok(p) => p,
            Err(e) => panic!("{}", e),
        };
        assert_abs_diff_eq!(params.freq.res, 200e3, epsilon = 1e-10);
    }

    #[test]
    fn test_new_params_freq_averaging_fail() {
        // The native freq. resolution is 40kHz.
        let mut args = get_1090008640_smallest();
        // 10e3 is not a multiple of 40kHz
        args.freq_res = Some(10e3);
        let result = args.into_params();
        assert!(
            result.is_err(),
            "Expected CalibrateParams to have not been successfully created"
        );

        let mut args = get_1090008640_smallest();

        // 79e3 is not a multiple of 40kHz
        args.freq_res = Some(79e3);
        let result = args.into_params();
        assert!(
            result.is_err(),
            "Expected CalibrateParams to have not been successfully created"
        );
    }

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
