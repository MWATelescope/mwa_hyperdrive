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
pub(crate) mod freq;
pub(crate) mod ranked_source;

pub use error::*;
pub(crate) use freq::*;
pub(crate) use ranked_source::*;

use std::collections::HashSet;
use std::path::PathBuf;

use log::{debug, info, warn};
use mwa_hyperbeam::fee::FEEBeam;
use mwalib::{CorrelatorContext, MWA_LONGITUDE_RADIANS};
use permissions::is_readable;

use crate::calibrate::veto::veto_sources;
use crate::data_formats::*;
use crate::{glob::*, *};
use mwa_hyperdrive_core::jones::cache::JonesCache;
use mwa_hyperdrive_core::*;
use mwa_hyperdrive_srclist::{SourceListFileType, SourceListType};

/// Parameters needed to perform calibration.
pub struct CalibrateParams {
    /// Interface to the MWA data, and metadata on the input data.
    pub(crate) input_data: Box<dyn InputData>,

    /// Beam struct.
    pub(crate) beam: mwa_hyperbeam::fee::FEEBeam,

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
    pub(crate) tile_flags: Vec<usize>,

    /// Which tiles are unflagged?
    pub(crate) unflagged_tiles: Vec<usize>,

    /// Channel- and frequency-related parameters required for calibration.
    pub(crate) freq: FrequencyParams,

    /// The target time resolution [seconds].
    ///
    /// e.g. If the input data is in 0.5s resolution and this variable was 4s,
    /// then we average 8 scans worth of time data for calibration.
    ///
    /// In a perfect world, this variable would be an integer, but it's
    /// primarily used in floating-point calculations, so it's more convenient
    /// to store it as a float.
    pub(crate) time_res: f64,

    /// A shared cache of Jones matrices. This field should be used to generate
    /// Jones matrices and populate the cache.
    pub(crate) jones_cache: mwa_hyperdrive_core::jones::cache::JonesCache,

    /// The timestep index into the observation.
    ///
    /// This is used to map to mwalib timesteps. hyperdrive's timesteps must be
    /// different, because hyperdrive might be doing time averaging, whereas
    /// mwalib has no notion of that.
    ///
    /// The first timestep does not necessarily correspond to the scheduled
    /// start of the observation; mwalib may ignore the start of the observation
    /// because e.g. not all data is available.
    ///
    /// To prevent this variable from being misused, it is private. Getting or
    /// setting requires methods.
    // timestep: usize,

    /// The local sidereal time [radians]. This variable is kept in lockstep with
    /// `timestep`.
    ///
    /// To prevent this variable from being misused, it is private. Getting or
    /// setting requires methods.
    lst: f64,

    /// The current pointing.
    ///
    /// To prevent this variable from being misused, it is private. Getting or
    /// setting requires methods.
    pointing: HADec,
    // /// The observation timestep indices, respecting the target time resolution.
    // ///
    // /// If time_res is the same as the observations native resolution, then
    // /// these timesteps are the same as mwalib's timesteps. Otherwise, they need
    // /// to be adjusted.
    // timesteps: Vec<usize>,
}

impl CalibrateParams {
    /// Create a new params struct from arguments.
    ///
    /// If the time or frequency resolution aren't specified, they default to
    /// the observation's native resolution.
    ///
    /// Source list vetoing is performed in this function, using the specified
    /// number of sources and/or the veto threshold.
    pub(crate) fn new(args: super::args::CalibrateUserArgs) -> Result<Self, InvalidArgsError> {
        // Set up the beam (requires the MWA_BEAM_FILE variable to be set).
        debug!("Creating beam object");
        let beam = FEEBeam::new_from_env()?;

        // Handle input data. We expect one of three possibilities:
        // - gpubox files, a metafits files, and maybe mwaf files,
        // - a measurement set, or
        // - uvfits files.
        // If none or multiple of these possibilities are met, then we must fail.
        let input_data: Box<dyn InputData> = match (
            args.metafits,
            args.gpuboxes,
            args.mwafs,
            args.ms,
            args.uvfits,
        ) {
            // Valid input for reading raw data.
            (Some(meta), Some(gpuboxes), mwafs, None, None) => {
                let input_data = RawData::new(
                    &meta,
                    &gpuboxes,
                    mwafs.as_deref(),
                    args.ignore_metafits_flags.unwrap_or_default(),
                    args.dont_flag_fine_channels.unwrap_or_default(),
                )?;

                // Print some high-level information.
                let obs_context = input_data.get_obs_context();
                info!("Calibrating obsid {}", obs_context.obsid);
                info!("Using metafits: {}", PathBuf::from(meta).display());
                info!("Using {} gpubox files", gpuboxes.len());
                match mwafs {
                    Some(_) => info!("Using supplied mwaf flags"),
                    None => warn!("No mwaf flags files supplied"),
                }

                Box::new(input_data)
            }

            // Valid input for reading a measurement set.
            (None, None, None, Some(ms), None) => {
                let input_data = MS::new(&ms)?;
                info!(
                    "Calibrating obsid {} from measurement set {}",
                    input_data.get_obs_context().obsid,
                    input_data.ms.canonicalize()?.display()
                );
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

        let obs_context = input_data.get_obs_context();
        let freq_context = input_data.get_freq_context();

        // Assign the tile flags. Add tiles that have already been flagged by
        // the input data.
        let mut tile_flags_set: HashSet<usize> = match args.tile_flags {
            Some(flags) => flags.into_iter().collect(),
            None => HashSet::new(),
        };
        for &obs_tile_flag in &obs_context.tile_flags {
            tile_flags_set.insert(obs_tile_flag);
        }
        let mut tile_flags: Vec<usize> = tile_flags_set.into_iter().collect();
        tile_flags.sort_unstable();
        // The length of the tile XYZ collection is the number of tiles in the
        // array, even if some tiles are flagged.
        let unflagged_tiles = (0..obs_context.tile_xyz.len())
            .into_iter()
            .filter(|ant| tile_flags.contains(ant))
            .collect();
        info!("Tile flags: {:?}", tile_flags);

        // Assign the per-coarse-channel fine-channel flags.
        let mut fine_chan_flags_set: HashSet<usize> = match args.fine_chan_flags {
            Some(flags) => flags.into_iter().collect(),
            None => HashSet::new(),
        };
        for &obs_fine_chan_flag in &obs_context.fine_chan_flags {
            fine_chan_flags_set.insert(obs_fine_chan_flag);
        }
        let mut fine_chan_flags: Vec<usize> = fine_chan_flags_set.into_iter().collect();
        fine_chan_flags.sort_unstable();
        info!("Fine-channel flags per coarse band: {:?}", fine_chan_flags);
        debug!(
            "Observation's fine-channel flags per coarse band: {:?}",
            obs_context.fine_chan_flags
        );

        let first_timestep = obs_context.timesteps[0];
        let lst = unsafe {
            let gmst = erfa_sys::eraGmst06(
                erfa_sys::ERFA_DJM0,
                first_timestep.as_mjd_utc_days(),
                erfa_sys::ERFA_DJM0,
                first_timestep.as_mjd_utc_days(),
            );
            (gmst + MWA_LONGITUDE_RADIANS) % TAU
        };

        // Set up frequency information.
        let native_freq_res = freq_context.native_fine_chan_width;
        let freq_res = args.freq_res.unwrap_or(native_freq_res);
        if freq_res % native_freq_res != 0.0 {
            return Err(InvalidArgsError::InvalidFreqResolution {
                got: freq_res,
                native: native_freq_res,
            });
        }

        let num_fine_chans_per_coarse_band =
            freq_context.fine_chan_freqs.len() / freq_context.coarse_chan_freqs.len();
        let num_unflagged_fine_chans_per_coarse_band =
            num_fine_chans_per_coarse_band - fine_chan_flags.len();
        let unflagged_fine_chan_freqs: Vec<f64> = freq_context
            .fine_chan_freqs
            .iter()
            .zip(
                (0..freq_context.num_fine_chans_per_coarse_chan)
                    .into_iter()
                    .cycle(),
            )
            .filter_map(|(&freq, chan_num)| {
                if fine_chan_flags.contains(&chan_num) {
                    None
                } else {
                    Some(freq)
                }
            })
            .collect();
        let freq_struct = FrequencyParams {
            res: freq_res,
            num_fine_chans_per_coarse_band,
            num_fine_chans: freq_context.fine_chan_freqs.len(),
            fine_chan_freqs: freq_context.fine_chan_freqs.clone(),
            num_unflagged_fine_chans_per_coarse_band,
            num_unflagged_fine_chans: num_unflagged_fine_chans_per_coarse_band
                * freq_context.coarse_chan_freqs.len(),
            // TODO
            unflagged_fine_chan_freqs: vec![],
            fine_chan_flags,
        };

        let mut source_list: SourceList = {
            // Handle the source list argument.
            let sl_pb: PathBuf = match args.source_list {
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
            // use that, otherwise guess from the file.
            let sl_type = match args.source_list_type {
                Some(t) => mwa_hyperdrive_srclist::read::parse_source_list_type(&t)?,
                None => match mwa_hyperdrive_srclist::read::parse_file_type(&sl_pb)? {
                    SourceListFileType::Json | SourceListFileType::Yaml => {
                        SourceListType::Hyperdrive
                    }
                    SourceListFileType::Txt => {
                        warn!(
                            "Assuming that {} is an RTS-style source list",
                            sl_pb.display()
                        );
                        SourceListType::Rts
                    }
                },
            };
            match mwa_hyperdrive_srclist::read::read_source_list_file(&sl_pb, &sl_type) {
                Ok(sl) => sl,
                Err(e) => {
                    eprintln!("Error when trying to read source list:");
                    return Err(InvalidArgsError::from(e));
                }
            }
        };
        info!("Found {} sources in the source list", source_list.len());

        // Veto any sources that may be troublesome, and/or cap the total number
        // of sources. If the user doesn't specify how many source-list sources
        // to use, then all sources are used.
        if args.num_sources == Some(0) || source_list.is_empty() {
            return Err(InvalidArgsError::NoSources);
        }
        dbg!(&freq_context.coarse_chan_freqs);
        let pointing = obs_context.pointing.to_owned();
        let ranked_sources = veto_sources(
            &mut source_list,
            &pointing,
            lst,
            &obs_context.delays,
            &freq_context.coarse_chan_freqs,
            &beam,
            args.num_sources,
            args.source_dist_cutoff.unwrap_or(CUTOFF_DISTANCE),
            args.veto_threshold.unwrap_or(DEFAULT_VETO_THRESHOLD),
        )?;
        let num_components = source_list
            .values()
            .fold(0, |a, src| a + src.components.len());
        info!(
            "Using {} sources with a total of {} components",
            source_list.len(),
            num_components
        );
        debug!("Using sources: {:#?}", source_list.keys());
        if source_list.len() > 10000 {
            warn!("Using more than 10,000 sources!");
        } else if source_list.is_empty() {
            return Err(InvalidArgsError::NoSourcesAfterVeto);
        }

        let native_time_res = obs_context.native_time_res;
        let time_res = args.time_res.unwrap_or(native_time_res);
        if time_res % native_time_res != 0.0 {
            return Err(InvalidArgsError::InvalidTimeResolution {
                got: time_res,
                native: native_time_res,
            });
        }
        // let num_time_steps_to_average = time_res / native_time_res;
        // let timesteps = (0..context.timesteps.len() / num_time_steps_to_average as usize).collect();

        Ok(Self {
            input_data,
            beam,
            source_list,
            ranked_sources,
            num_components,
            tile_flags,
            unflagged_tiles,
            freq: freq_struct,
            time_res,
            jones_cache: JonesCache::new(),
            // timestep,
            lst,
            // TODO: Should this be RADec or HADec?
            pointing: pointing.to_hadec(lst),
            // timesteps,
        })
    }

    // /// Get the current timestep.
    // pub fn get_timestep(&self) -> usize {
    //     self.timestep
    // }

    // /// Get all of the available timesteps.
    // pub fn get_timesteps(&self) -> &[usize] {
    //     &self.timesteps
    // }

    // /// Get the current LST [radians].
    // pub(crate) fn get_lst(&self) -> f64 {
    //     self.lst
    // }

    // /// Get the current pointing.
    // pub(crate) fn get_pointing(&self) -> &HADec {
    //     &self.pointing
    // }

    // /// Increment the timestep and LST.
    // pub(crate) fn next_timestep(&mut self) {
    //     self.timestep += 1;
    //     self.lst = lst_from_timestep(self.timestep, &self.context, self.time_res);
    // }

    /// Get the antenna index from the antenna number.
    ///
    /// Consider an array with 3 antennas. If all are being used, then an
    /// "antenna array" would have a dimension of size 3. But, if one of those
    /// antennas is flagged, then this array would have only a size of 2.
    /// Naively using antenna number 3 (index 2) to index into the array would
    /// obviously be a problem when considering flagged antennas; that's what
    /// this function is for.
    ///
    /// This function will panic if the antenna number is in the tile flags.
    pub(crate) fn get_ant_index(&self, ant_number: usize) -> usize {
        if self.tile_flags.contains(&ant_number) {
            panic!(
                "Tried to get the antenna index for antenna number {}, but that antenna is flagged",
                ant_number
            );
        }
        ant_number - self.tile_flags.iter().filter(|&f| f < &ant_number).count()
    }
}

/// Get the difference between when the observation actually starts and when it
/// was scheduled to start [seconds].
///
/// MWA observations don't necessarily start when they should! mwalib provides
/// the information needed to determine when the observation *actually* starts,
/// and with this value, other values such as the start LST can be determined.
/// mwalib determines the true start time based on the timesteps available in
/// the gpubox files.
fn get_diff_in_start_time(context: &CorrelatorContext) -> f64 {
    // Convert to i64 to prevent trying to subtract a u64 from a smaller u64.
    (context.start_unix_time_ms as i64 - context.metafits_context.sched_start_unix_time_ms as i64)
        as f64
        / 1e3
}

/// Get the LST (in radians) from a timestep. `time_res_seconds` refers to the
/// target time resolution of calibration, *not* the observation's time
/// resolution.
///
/// The LST is calculated for the middle of the timestep, not the start of it.
fn lst_from_timestep(timestep: usize, context: &CorrelatorContext, time_res_seconds: f64) -> f64 {
    let start_lst = context.metafits_context.lst_rad;
    let factor = SOLAR2SIDEREAL * DS2R;
    start_lst
        + factor * (get_diff_in_start_time(context) + time_res_seconds * (timestep as f64 + 0.5))
}

/// Get the pointing from a timestep. `time_res_seconds` refers to the target
/// time resolution of calibration, *not* the observation's time resolution.
///
/// The pointing is calculated for the middle of the timestep, not the start of
/// it.
fn pointing_from_timestep(
    timestep: usize,
    context: &CorrelatorContext,
    time_res_seconds: f64,
) -> HADec {
    let start_lst = context.metafits_context.lst_rad;
    let true_start_lst = lst_from_timestep(timestep, context, time_res_seconds);
    let lst_diff = true_start_lst - start_lst;

    let mut pointing = HADec::from_radec(
        &RADec::new_degrees(
            context
                .metafits_context
                // If the phase centre isn't specified, this is probably a
                // "drift" observation. Use the tile pointing instead.
                // TODO: Let the user override this.
                .ra_phase_center_degrees
                .unwrap_or(context.metafits_context.ra_tile_pointing_degrees),
            context
                .metafits_context
                .dec_phase_center_degrees
                .unwrap_or(context.metafits_context.dec_tile_pointing_degrees),
        ),
        context.metafits_context.lst_rad,
    );
    pointing.ha += lst_diff;
    pointing
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::calibrate::args::CalibrateUserArgs;
    use mwa_hyperdrive_tests::{full_obsids::*, reduced_obsids::*};

    use approx::*;
    // Need to use serial tests because HDF5 is not necessarily reentrant.
    use serial_test::serial;

    #[test]
    #[serial]
    fn test_new_params() {
        let data = get_1090008640();
        let args = CalibrateUserArgs {
            metafits: Some(data.metafits),
            gpuboxes: Some(data.gpuboxes),
            mwafs: Some(data.mwafs),
            source_list: data.source_list,
            ..Default::default()
        };
        let params = match CalibrateParams::new(args) {
            Ok(p) => p,
            Err(e) => panic!("{}", e),
        };
        // The default time resolution should be 2.0s, as per the metafits.
        assert_abs_diff_eq!(params.time_res, 2.0);
        // The default freq resolution should be 40kHz, as per the metafits.
        assert_abs_diff_eq!(params.freq.res, 40e3, epsilon = 1e-10);
    }

    #[test]
    #[serial]
    fn test_new_params_time_averaging() {
        // The native time resolution is 2.0s.
        let data = get_1090008640();
        let args = CalibrateUserArgs {
            metafits: Some(data.metafits),
            gpuboxes: Some(data.gpuboxes),
            mwafs: Some(data.mwafs),
            source_list: data.source_list,
            // 4.0 should be a multiple of 2.0s
            time_res: Some(4.0),
            ..Default::default()
        };
        let params = match CalibrateParams::new(args) {
            Ok(p) => p,
            Err(e) => panic!("{}", e),
        };
        assert_abs_diff_eq!(params.time_res, 4.0);

        let data = get_1090008640();
        let args = CalibrateUserArgs {
            metafits: Some(data.metafits),
            gpuboxes: Some(data.gpuboxes),
            mwafs: Some(data.mwafs),
            source_list: data.source_list,
            // 8.0 should be a multiple of 2.0s
            time_res: Some(8.0),
            ..Default::default()
        };
        let params = match CalibrateParams::new(args) {
            Ok(p) => p,
            Err(e) => panic!("{}", e),
        };
        assert_abs_diff_eq!(params.time_res, 8.0);
    }

    #[test]
    #[serial]
    fn test_new_params_time_averaging_fail() {
        // The native time resolution is 2.0s.
        let data = get_1090008640();
        let args = CalibrateUserArgs {
            metafits: Some(data.metafits),
            gpuboxes: Some(data.gpuboxes),
            mwafs: Some(data.mwafs),
            source_list: data.source_list,
            // 2.01 is not a multiple of 2.0s
            time_res: Some(2.01),
            ..Default::default()
        };
        let result = CalibrateParams::new(args);
        assert!(
            result.is_err(),
            "Expected CalibrateParams to have not been successfully created"
        );

        let data = get_1090008640();
        let args = CalibrateUserArgs {
            metafits: Some(data.metafits),
            gpuboxes: Some(data.gpuboxes),
            mwafs: Some(data.mwafs),
            source_list: data.source_list,
            // 3.0 is not a multiple of 2.0s
            time_res: Some(3.0),
            ..Default::default()
        };
        let result = CalibrateParams::new(args);
        assert!(
            result.is_err(),
            "Expected CalibrateParams to have not been successfully created"
        );
    }

    #[test]
    #[serial]
    fn test_new_params_freq_averaging() {
        // The native freq. resolution is 40kHz.
        let data = get_1090008640();
        let args = CalibrateUserArgs {
            metafits: Some(data.metafits),
            gpuboxes: Some(data.gpuboxes),
            mwafs: Some(data.mwafs),
            source_list: data.source_list,
            // 80e3 should be a multiple of 40kHz
            freq_res: Some(80e3),
            ..Default::default()
        };
        let params = match CalibrateParams::new(args) {
            Ok(p) => p,
            Err(e) => panic!("{}", e),
        };
        assert_abs_diff_eq!(params.freq.res, 80e3, epsilon = 1e-10);

        let data = get_1090008640();
        let args = CalibrateUserArgs {
            metafits: Some(data.metafits),
            gpuboxes: Some(data.gpuboxes),
            mwafs: Some(data.mwafs),
            source_list: data.source_list,
            // 200e3 should be a multiple of 40kHz
            freq_res: Some(200e3),
            ..Default::default()
        };
        let params = match CalibrateParams::new(args) {
            Ok(p) => p,
            Err(e) => panic!("{}", e),
        };
        assert_abs_diff_eq!(params.freq.res, 200e3, epsilon = 1e-10);
    }

    #[test]
    #[serial]
    fn test_new_params_freq_averaging_fail() {
        // The native freq. resolution is 40kHz.
        let data = get_1090008640();
        let args = CalibrateUserArgs {
            metafits: Some(data.metafits),
            gpuboxes: Some(data.gpuboxes),
            mwafs: Some(data.mwafs),
            source_list: data.source_list,
            // 10e3 is not a multiple of 40kHz
            freq_res: Some(10e3),
            ..Default::default()
        };
        let result = CalibrateParams::new(args);
        assert!(
            result.is_err(),
            "Expected CalibrateParams to have not been successfully created"
        );

        let data = get_1090008640();
        let args = CalibrateUserArgs {
            metafits: Some(data.metafits),
            gpuboxes: Some(data.gpuboxes),
            mwafs: Some(data.mwafs),
            source_list: data.source_list,
            // 79e3 is not a multiple of 40kHz
            freq_res: Some(79e3),
            ..Default::default()
        };
        let result = CalibrateParams::new(args);
        assert!(
            result.is_err(),
            "Expected CalibrateParams to have not been successfully created"
        );
    }

    #[test]
    #[serial]
    fn test_new_params_tile_flags() {
        // 1090008640 has no flagged tiles in its metafits.
        let data = get_1090008640();
        let args = CalibrateUserArgs {
            metafits: Some(data.metafits),
            gpuboxes: Some(data.gpuboxes),
            mwafs: Some(data.mwafs),
            source_list: data.source_list,
            // Manually flag antennas 1, 2 and 3.
            tile_flags: Some(vec![1, 2, 3]),
            ..Default::default()
        };
        let params = match CalibrateParams::new(args) {
            Ok(p) => p,
            Err(e) => panic!("{}", e),
        };
        assert_eq!(params.tile_flags.len(), 3);
        assert!(params.tile_flags.contains(&1));
        assert!(params.tile_flags.contains(&2));
        assert!(params.tile_flags.contains(&3));
        assert_eq!(params.num_unflagged_tiles, 125);
        assert_eq!(params.get_ant_index(0), 0);
        assert_eq!(params.get_ant_index(4), 1);
        assert_eq!(params.get_ant_index(5), 2);
        assert_eq!(params.get_ant_index(6), 3);
    }

    #[test]
    #[serial]
    #[should_panic]
    fn test_new_params_tile_flags_fail2() {
        // Try to get an antenna index for a tile that is flagged (should
        // panic).
        let data = get_1090008640();
        let args = CalibrateUserArgs {
            metafits: Some(data.metafits),
            gpuboxes: Some(data.gpuboxes),
            mwafs: Some(data.mwafs),
            source_list: data.source_list,
            tile_flags: Some(vec![1, 2]),
            ..Default::default()
        };
        let params = match CalibrateParams::new(args) {
            Ok(p) => p,
            Err(e) => panic!("{}", e),
        };
        // Should panic.
        assert_eq!(params.get_ant_index(1), 1);
    }

    // astropy doesn't exactly agree with the numbers below, I think because the
    // LST listed in MWA metafits files doesn't agree with what astropy thinks
    // it should be. But, it's all very close.
    #[test]
    fn test_lst_from_mwalib_timestep_native() {
        // Obsid 1090008640 actually starts at 1090008641.
        let data = get_1090008640();
        let context = match CorrelatorContext::new(&data.metafits, &data.gpuboxes) {
            Ok(c) => c,
            Err(e) => panic!("{}", e),
        };
        let time_res = context.metafits_context.corr_int_time_ms as f64 / 1e3;
        let new_lst = lst_from_timestep(0, &context, time_res);
        // gpstime 1090008642
        assert_abs_diff_eq!(new_lst, 6.262123690318563, epsilon = 1e-10);

        let new_lst = lst_from_timestep(1, &context, time_res);
        // gpstime 1090008644
        assert_abs_diff_eq!(new_lst, 6.26226953263562, epsilon = 1e-10);
    }

    #[test]
    fn test_lst_from_mwalib_timestep_averaged() {
        let data = get_1090008640();
        let context = match CorrelatorContext::new(&data.metafits, &data.gpuboxes) {
            Ok(c) => c,
            Err(e) => panic!("{}", e),
        };
        // The native time res. is 2.0s, let's make our target 4.0s here.
        let time_res = 4.0;
        let new_lst = lst_from_timestep(0, &context, time_res);
        // gpstime 1090008643
        assert_abs_diff_eq!(new_lst, 6.2621966114770915, epsilon = 1e-10);

        let new_lst = lst_from_timestep(1, &context, time_res);
        // gpstime 1090008647
        assert_abs_diff_eq!(new_lst, 6.262488296111205, epsilon = 1e-10);
    }

    #[test]
    fn test_pointing_from_timestep_native() {
        // Obsid 1090008640 actually starts at 1090008641.
        let data = get_1090008640();
        let context = match CorrelatorContext::new(&data.metafits, &data.gpuboxes) {
            Ok(c) => c,
            Err(e) => panic!("{}", e),
        };
        let time_res = context.metafits_context.corr_int_time_ms as f64 / 1e3;
        let pointing = pointing_from_timestep(0, &context, time_res);
        // gpstime 1090008642
        assert_abs_diff_eq!(pointing.ha, 6.262123690318563, epsilon = 1e-10);
        assert_abs_diff_eq!(pointing.dec, -0.47123889803846897, epsilon = 1e-10);

        let pointing = pointing_from_timestep(1, &context, time_res);
        // gpstime 1090008644
        assert_abs_diff_eq!(pointing.ha, 6.26226953263562, epsilon = 1e-10);
        assert_abs_diff_eq!(pointing.dec, -0.47123889803846897, epsilon = 1e-10);
    }

    #[test]
    fn test_pointing_from_timestep_averaged() {
        let data = get_1090008640();
        let context = match CorrelatorContext::new(&data.metafits, &data.gpuboxes) {
            Ok(c) => c,
            Err(e) => panic!("{}", e),
        };
        // The native time res. is 2.0s, let's make our target 4.0s here.
        let time_res = 4.0;
        let pointing = pointing_from_timestep(0, &context, time_res);
        // gpstime 1090008643
        assert_abs_diff_eq!(pointing.ha, 6.2621966114770915, epsilon = 1e-10);
        assert_abs_diff_eq!(pointing.dec, -0.47123889803846897, epsilon = 1e-10);

        let pointing = pointing_from_timestep(1, &context, time_res);
        // gpstime 1090008647
        assert_abs_diff_eq!(pointing.ha, 6.262488296111205, epsilon = 1e-10);
        assert_abs_diff_eq!(pointing.dec, -0.47123889803846897, epsilon = 1e-10);
    }

    // The following tests use full MWA data.

    #[test]
    #[serial]
    #[ignore]
    fn test_new_params_real_data() {
        let data = get_1065880128();
        let args = CalibrateUserArgs {
            metafits: Some(data.metafits),
            gpuboxes: Some(data.gpuboxes),
            mwafs: Some(data.mwafs),
            source_list: data.source_list,
            ..Default::default()
        };
        let result = CalibrateParams::new(args);
        assert!(
            result.is_ok(),
            "Expected CalibrateParams to have been successfully created"
        );
    }

    #[test]
    #[ignore]
    fn test_lst_from_timestep_native_real() {
        let data = get_1065880128();
        let context = match CorrelatorContext::new(&data.metafits, &data.gpuboxes) {
            Ok(c) => c,
            Err(e) => panic!("{}", e),
        };
        let time_res = context.metafits_context.corr_int_time_ms as f64 / 1e3;
        let new_lst = lst_from_timestep(0, &context, time_res);
        // gpstime 1065880126.25
        assert_abs_diff_eq!(new_lst, 6.074695614533638, epsilon = 1e-10);

        let new_lst = lst_from_timestep(1, &context, time_res);
        // gpstime 1065880126.75
        assert_abs_diff_eq!(new_lst, 6.074732075112903, epsilon = 1e-10);
    }

    #[test]
    #[ignore]
    fn test_lst_from_timestep_averaged_real() {
        let data = get_1065880128();
        let context = match CorrelatorContext::new(&data.metafits, &data.gpuboxes) {
            Ok(c) => c,
            Err(e) => panic!("{}", e),
        };
        // The native time res. is 0.5s, let's make our target 2s here.
        let time_res = 2.0;
        let new_lst = lst_from_timestep(0, &context, time_res);
        // gpstime 1065880127
        assert_abs_diff_eq!(new_lst, 6.074750305402534, epsilon = 1e-10);

        let new_lst = lst_from_timestep(1, &context, time_res);
        // gpstime 1065880129
        assert_abs_diff_eq!(new_lst, 6.074896147719591, epsilon = 1e-10);
    }

    #[test]
    #[ignore]
    fn test_pointing_from_timestep_native_real() {
        let data = get_1065880128();
        let context = match CorrelatorContext::new(&data.metafits, &data.gpuboxes) {
            Ok(c) => c,
            Err(e) => panic!("{}", e),
        };
        let time_res = context.metafits_context.corr_int_time_ms as f64 / 1e3;
        let pointing = pointing_from_timestep(0, &context, time_res);
        assert_abs_diff_eq!(pointing.ha, 6.074695614533638, epsilon = 1e-10);
        assert_abs_diff_eq!(pointing.dec, -0.47123889803846897, epsilon = 1e-10);

        let pointing = pointing_from_timestep(1, &context, time_res);
        assert_abs_diff_eq!(pointing.ha, 6.074732075112903, epsilon = 1e-10);
        assert_abs_diff_eq!(pointing.dec, -0.47123889803846897, epsilon = 1e-10);
    }

    #[test]
    #[ignore]
    fn test_pointing_from_timestep_averaged_real() {
        let data = get_1065880128();
        let context = match CorrelatorContext::new(&data.metafits, &data.gpuboxes) {
            Ok(c) => c,
            Err(e) => panic!("{}", e),
        };
        // The native time res. is 0.5s, let's make our target 2s here.
        let time_res = 2.0;
        let pointing = pointing_from_timestep(0, &context, time_res);
        // gpstime 1065880127
        assert_abs_diff_eq!(pointing.ha, 6.074750305402534, epsilon = 1e-10);
        assert_abs_diff_eq!(pointing.dec, -0.47123889803846897, epsilon = 1e-10);

        let pointing = pointing_from_timestep(1, &context, time_res);
        // gpstime 1065880129
        assert_abs_diff_eq!(pointing.ha, 6.074896147719591, epsilon = 1e-10);
        assert_abs_diff_eq!(pointing.dec, -0.47123889803846897, epsilon = 1e-10);
    }
}
