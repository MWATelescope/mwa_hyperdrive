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

pub mod error;
pub(crate) mod freq;

pub use error::*;
pub(crate) use freq::*;

use mwa_hyperbeam::fee::FEEBeam;
use mwalib::mwalibContext;
use ndarray::Array1;

use crate::calibrate::veto::veto_sources;
use crate::flagging::cotter::CotterFlags;
use crate::{glob::*, *};
use mwa_hyperdrive_core::jones::cache::JonesCache;
use mwa_hyperdrive_core::*;
use mwa_hyperdrive_srclist::{SourceListFileType, SourceListType};

/// A source's name as well as its apparent flux density.
pub struct RankedSource {
    /// The name of the source. This can be used as a key for a `SourceList`.
    pub name: String,

    /// The apparent flux density [Jy].
    pub flux_density: f64,
}

/// Parameters needed to perform calibration.
pub struct CalibrateParams {
    /// mwalib context struct.
    pub(crate) context: mwalibContext,

    /// If provided, information on RFI flags.
    pub(crate) cotter_flags: Option<CotterFlags>,

    /// Which tiles are flagged? These values correspond to those from the
    /// "Antenna" column in HDU 1 of the metafits file.
    pub(crate) tile_flags: Vec<usize>,

    /// How many tiles are unflagged?
    pub(crate) num_unflagged_tiles: usize,

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
    timestep: usize,

    /// The observation timesteps, respecting the target time resolution.
    ///
    /// If time_res is the same as the observations native resolution, then
    /// these timesteps are the same as mwalib's timesteps. Otherwise, they need
    /// to be adjusted.
    timesteps: Vec<usize>,

    /// The local sidereal time [radians]. This variable is kept in lockstep with
    /// `timestep`.
    ///
    /// To prevent this variable from being misused, it is private. Getting or
    /// setting requires methods.
    lst: f64,
}

impl CalibrateParams {
    /// Create a new params struct from arguments and an existing mwalib
    /// context. This function is intentionally private; it should only be
    /// directly used for testing. The only way for a caller to create a
    /// `CalibrateParams` is with `CalibrateParams::from_args`, which requires
    /// the presence of gpubox files. To allow testing *without* gpubox files,
    /// this function does any work not required by the gpubox files.
    ///
    /// If the time or frequency resolution aren't specified, they default to
    /// the observation's native resolution.
    ///
    /// Source list vetoing is performed in this function, using the specified
    /// number of sources and/or the veto threshold.
    fn new(
        args: crate::calibrate::args::CalibrateUserArgs,
        context: mwalibContext,
    ) -> Result<Self, InvalidArgsError> {
        // Set up the beam (requires the MWA_BEAM_FILE variable to be set).
        debug!("Creating beam object");
        let beam = FEEBeam::new_from_env()?;

        // mwaf files are optional; don't bail if none were specified.
        debug!("Reading mwaf cotter flag files");
        let cotter_flags = {
            let mwaf_pbs: Option<Vec<PathBuf>> = match args.mwafs {
                None => None,

                Some(m) => match m.len() {
                    0 => None,

                    // If a single mwaf "file" was specified, and it isn't a real
                    // file, treat it as a glob and expand it to find matches.
                    1 => {
                        let pb = PathBuf::from(&m[0]);
                        if pb.exists() {
                            Some(vec![pb])
                        } else {
                            let entries = get_all_matches_from_glob(&m[0])?;
                            if entries.is_empty() {
                                return Err(InvalidArgsError::SingleMwafNotAFileOrGlob);
                            } else {
                                Some(entries)
                            }
                        }
                    }

                    _ => Some(m.iter().map(PathBuf::from).collect()),
                },
            };

            // If some mwaf files were given, unpack them.
            if let Some(m) = mwaf_pbs {
                let mut f = CotterFlags::new_from_mwafs(&m)?;

                // The cotter flags are available for all times. Make them match
                // only those we'll use according to mwalib.
                f.trim(&context);

                // Ensure that there is a mwaf file for each specified gpubox file.
                for cc in &context.coarse_channels {
                    if !f.gpubox_nums.contains(&(cc.gpubox_number as u8)) {
                        return Err(InvalidArgsError::GpuboxFileMissingMwafFile(
                            cc.gpubox_number,
                        ));
                    }
                }

                Some(f)
            } else {
                None
            }
        };

        // Assign the tile flags. Need to handle explicit user input as well as
        // whether to use the metafits file or not.
        let mut tile_flags: Vec<usize> = match args.tile_flags {
            Some(flags) => flags,
            None => vec![],
        };
        match args.ignore_metafits_flags {
            Some(true) => debug!("NOT using metafits tile flags"),
            _ => {
                // Iterate over the RF inputs, and add the tile flags if
                // necessary.
                let mut meta_tile_flags = vec![];
                for rf in context
                    .rf_inputs
                    .iter()
                    .filter(|rf| rf.pol == mwalib::Pol::Y)
                {
                    let a = rf.antenna as usize;
                    if rf.flagged && !tile_flags.contains(&a) {
                        meta_tile_flags.push(a);
                        tile_flags.push(a);
                    }
                }
                debug!("Using metafits tile flags; found {:?}", &meta_tile_flags);
            }
        }
        // Sort the tile flags.
        tile_flags.sort_unstable();
        // Validate.
        let num_tiles = context.rf_inputs.len() / 2;
        debug!("There are {} total tiles", num_tiles);
        let num_unflagged_tiles = num_tiles - tile_flags.len();
        for &f in &tile_flags {
            if f > num_tiles - 1 {
                return Err(InvalidArgsError::InvalidTileFlag {
                    got: f,
                    max: num_tiles - 1,
                });
            }
        }
        debug!("There are {} unflagged tiles", num_unflagged_tiles);

        let fine_channel_flags: Vec<usize> = match args.fine_chan_flags {
            Some(flags) => flags,
            // If the flags aren't specified, use the observation's fine-channel
            // frequency resolution to set them.
            None => match context.fine_channel_width_hz {
                // 10 kHz, 128 channels.
                10000 => vec![
                    0, 1, 2, 3, 4, 5, 6, 7, 64, 120, 121, 122, 123, 124, 125, 126, 127,
                ],

                // 20 kHz, 64 channels.
                20000 => vec![0, 1, 2, 3, 32, 60, 61, 62, 63],

                // 40 kHz, 32 channels.
                40000 => vec![0, 1, 16, 30, 31],
                f => return Err(InvalidArgsError::UnhandledFreqResolutionForFlags(f)),
            },
        };

        // Print some high-level information.
        info!("Calibrating obsid {}", context.obsid);
        info!("Using metafits: {}", context.metafits_filename);
        info!("Using {} gpubox files", context.num_gpubox_files);
        match &cotter_flags {
            Some(_) => info!("Using supplied cotter flags"),
            None => warn!("No cotter flags files specified"),
        }
        info!("Tile flags: {:?}", &tile_flags);
        info!(
            "Fine-channel flags per coarse band: {:?}",
            fine_channel_flags
        );

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
        let ranked_sources = veto_sources(
            &mut source_list,
            &context,
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

        let native_time_res = context.integration_time_milliseconds as f64 / 1e3;
        let time_res = args.time_res.unwrap_or(native_time_res);
        if time_res % native_time_res != 0.0 {
            return Err(InvalidArgsError::InvalidTimeResolution {
                got: time_res,
                native: native_time_res,
            });
        }
        let num_time_steps_to_average = time_res / native_time_res;
        let timesteps = (0..context.timesteps.len() / num_time_steps_to_average as usize).collect();

        let native_freq_res = context.fine_channel_width_hz as f64;
        let freq_res = args.freq_res.unwrap_or(native_freq_res);
        if freq_res % native_freq_res != 0.0 {
            return Err(InvalidArgsError::InvalidFreqResolution {
                got: freq_res,
                native: native_freq_res,
            });
        }

        // Start at the first timestep.
        let timestep = 0;
        let lst = lst_from_timestep(timestep, &context, time_res);

        let mut fine_chan_freqs =
            Vec::with_capacity(context.num_fine_channels_per_coarse * context.num_coarse_channels);
        // TODO: I'm suspicious that the start channel freq is incorrect.
        for cc in &context.coarse_channels {
            let mut cc_freqs = Array1::range(
                cc.channel_start_hz as f64,
                cc.channel_end_hz as f64,
                context.fine_channel_width_hz as f64,
            )
            .to_vec();
            fine_chan_freqs.append(&mut cc_freqs);
        }

        let unflagged_fine_chan_freqs = fine_chan_freqs
            .iter()
            .zip(
                (0..context.num_fine_channels_per_coarse)
                    .into_iter()
                    .cycle(),
            )
            .filter_map(|(&freq, chan_num)| {
                if fine_channel_flags.contains(&chan_num) {
                    None
                } else {
                    Some(freq)
                }
            })
            .collect();
        let num_unflagged_fine_chans_per_coarse_band =
            context.num_fine_channels_per_coarse - fine_channel_flags.len();

        let freq_struct = FrequencyParams {
            res: freq_res,
            num_fine_chans_per_coarse_band: context.num_fine_channels_per_coarse,
            num_fine_chans: context.num_fine_channels_per_coarse * context.num_coarse_channels,
            fine_chan_freqs,
            num_unflagged_fine_chans_per_coarse_band,
            num_unflagged_fine_chans: num_unflagged_fine_chans_per_coarse_band
                * context.num_coarse_channels,
            unflagged_fine_chan_freqs,
            fine_channel_flags,
        };

        Ok(Self {
            context,
            cotter_flags,
            tile_flags,
            num_unflagged_tiles,
            beam,
            source_list,
            ranked_sources,
            num_components,
            freq: freq_struct,
            time_res,
            timesteps,
            timestep,
            lst,
            jones_cache: JonesCache::new(),
        })
    }

    /// Create a new params struct from user arguments.
    ///
    /// Most of the work is delegated to `CalibrateParams::new`; this function
    /// requires the presence of gpubox files, whereas `new` does not. However,
    /// `new` is not publicly visible; it is only used so we can test the code
    /// without needing gpubox files.
    pub fn from_args(
        mut args: crate::calibrate::args::CalibrateUserArgs,
    ) -> Result<Self, InvalidArgsError> {
        let metafits: PathBuf = match args.metafits {
            None => return Err(InvalidArgsError::NoMetafits),
            Some(m) => {
                // The metafits argument could be a glob. If the specified
                // metafits file can't be found, treat it as a glob and expand
                // it to find a match.
                let pb = PathBuf::from(&m);
                if pb.exists() {
                    pb
                } else {
                    get_single_match_from_glob(&m)?
                }
            }
        };
        debug!("Using metafits: {}", metafits.display());

        let gpuboxes: Vec<PathBuf> = match args.gpuboxes {
            None => return Err(InvalidArgsError::NoGpuboxes),
            Some(g) => {
                match g.len() {
                    0 => return Err(InvalidArgsError::NoGpuboxes),

                    // If a single gpubox file was specified, and it isn't a real
                    // file, treat it as a glob and expand it to find matches.
                    1 => {
                        let pb = PathBuf::from(&g[0]);
                        if pb.exists() {
                            vec![pb]
                        } else {
                            let entries = get_all_matches_from_glob(&g[0])?;
                            if entries.is_empty() {
                                return Err(InvalidArgsError::SingleGpuboxNotAFileOrGlob);
                            } else {
                                entries
                            }
                        }
                    }

                    _ => g.iter().map(PathBuf::from).collect(),
                }
            }
        };
        debug!("Using gpubox files: {:#?}", gpuboxes);

        debug!("Creating mwalib context");
        let context = mwalibContext::new(&metafits, &gpuboxes)?;

        // Plug up the missing parts of the arguments.
        args.metafits = None;
        args.gpuboxes = None;
        CalibrateParams::new(args, context)
    }

    // /// Get the timestep needed to read in data.
    // pub fn get_timestep(&self) -> usize {
    //     self.timestep
    // }

    /// Get the LST [radians].
    pub(crate) fn get_lst(&self) -> f64 {
        self.lst
    }

    /// Increment the timestep and LST.
    pub(crate) fn next_timestep(&mut self) {
        self.timestep += 1;
        self.lst = lst_from_timestep(self.timestep, &self.context, self.time_res);
    }

    /// In terms of antenna number, how many tiles are flagged before this one?
    pub(crate) fn count_flagged_before_this_ant(&self, ant: usize) -> usize {
        self.tile_flags.iter().filter(|&f| f < &ant).count()
    }
}

/// Get the LST (in radians) from a timestep.
///
/// The LST is calculated for the middle of the timestep, not the start.
fn lst_from_timestep(timestep: usize, context: &mwalibContext, time_res_seconds: f64) -> f64 {
    let start_lst = context.lst_degrees.to_radians();
    // Convert to i64 to prevent trying to subtract a u64 from a smaller u64.
    let diff_in_start_time = (context.start_unix_time_milliseconds as i64
        - context.scheduled_start_unix_time_milliseconds as i64)
        as f64
        / 1e3;
    let offset = diff_in_start_time / time_res_seconds;
    let factor = time_res_seconds * SOLAR2SIDEREAL * DS2R;
    start_lst + factor * (offset + (timestep as f64 + 0.5))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::calibrate::args::CalibrateUserArgs;
    use mwa_hyperdrive_tests::{gpuboxes::*, no_gpuboxes::*};

    use approx::*;
    // Need to use serial tests because HDF5 is not necessarily reentrant.
    use serial_test::serial;

    #[test]
    #[serial]
    fn test_new_params() {
        let data = get_1065880128_meta();
        let context = mwalibContext::new(&data.metafits, &data.gpuboxes)
            .expect("Failed to create mwalib context");
        let args = CalibrateUserArgs {
            metafits: Some(data.metafits),
            gpuboxes: Some(data.gpuboxes),
            mwafs: Some(data.mwafs),
            source_list: data.source_list,
            ..Default::default()
        };
        let result = CalibrateParams::new(args, context);
        assert!(result.is_ok());
        let params = result.unwrap();
        // The default time resolution should be 0.5s, as per the metafits.
        assert_abs_diff_eq!(params.time_res, 0.5, epsilon = 1e-10);
        // The default freq resolution should be 40kHz, as per the metafits.
        assert_abs_diff_eq!(params.freq.res, 40e3, epsilon = 1e-10);
    }

    #[test]
    #[serial]
    fn test_new_params_time_averaging() {
        // The native time resolution is 0.5s.
        let data = get_1065880128_meta();
        let context = mwalibContext::new(&data.metafits, &data.gpuboxes)
            .expect("Failed to create mwalib context");
        let args = CalibrateUserArgs {
            metafits: Some(data.metafits),
            gpuboxes: Some(data.gpuboxes),
            mwafs: Some(data.mwafs),
            source_list: data.source_list,
            // 1.0 should be a multiple of 0.5s
            time_res: Some(1.0),
            ..Default::default()
        };
        let result = CalibrateParams::new(args, context);
        assert!(
            result.is_ok(),
            "Expected CalibrateParams to have been successfully created"
        );
        let params = result.unwrap();
        assert_abs_diff_eq!(params.time_res, 1.0, epsilon = 1e-10);

        let data = get_1065880128_meta();
        let context = mwalibContext::new(&data.metafits, &data.gpuboxes)
            .expect("Failed to create mwalib context");
        let args = CalibrateUserArgs {
            metafits: Some(data.metafits),
            gpuboxes: Some(data.gpuboxes),
            mwafs: Some(data.mwafs),
            source_list: data.source_list,
            // 2.0 should be a multiple of 0.5s
            time_res: Some(2.0),
            ..Default::default()
        };
        let result = CalibrateParams::new(args, context);
        assert!(
            result.is_ok(),
            "Expected CalibrateParams to have been successfully created"
        );
        let params = result.unwrap();
        assert_abs_diff_eq!(params.time_res, 2.0, epsilon = 1e-10);
    }

    #[test]
    #[serial]
    fn test_new_params_time_averaging_fail() {
        // The native time resolution is 0.5s.
        let data = get_1065880128_meta();
        let context = mwalibContext::new(&data.metafits, &data.gpuboxes)
            .expect("Failed to create mwalib context");
        let args = CalibrateUserArgs {
            metafits: Some(data.metafits),
            gpuboxes: Some(data.gpuboxes),
            mwafs: Some(data.mwafs),
            source_list: data.source_list,
            // 1.01 is not a multiple of 0.5s
            time_res: Some(1.01),
            ..Default::default()
        };
        let result = CalibrateParams::new(args, context);
        assert!(
            result.is_err(),
            "Expected CalibrateParams to have not been successfully created"
        );

        let data = get_1065880128_meta();
        let context = mwalibContext::new(&data.metafits, &data.gpuboxes)
            .expect("Failed to create mwalib context");
        let args = CalibrateUserArgs {
            metafits: Some(data.metafits),
            gpuboxes: Some(data.gpuboxes),
            mwafs: Some(data.mwafs),
            source_list: data.source_list,
            // 0.75 is not a multiple of 0.5s
            time_res: Some(0.75),
            ..Default::default()
        };
        let result = CalibrateParams::new(args, context);
        assert!(
            result.is_err(),
            "Expected CalibrateParams to have not been successfully created"
        );
    }

    #[test]
    #[serial]
    fn test_new_params_freq_averaging() {
        // The native freq. resolution is 40kHz.
        let data = get_1065880128_meta();
        let context = mwalibContext::new(&data.metafits, &data.gpuboxes)
            .expect("Failed to create mwalib context");
        let args = CalibrateUserArgs {
            metafits: Some(data.metafits),
            gpuboxes: Some(data.gpuboxes),
            mwafs: Some(data.mwafs),
            source_list: data.source_list,
            // 80e3 should be a multiple of 40kHz
            freq_res: Some(80e3),
            ..Default::default()
        };
        let result = CalibrateParams::new(args, context);
        assert!(
            result.is_ok(),
            "Expected CalibrateParams to have been successfully created"
        );
        let params = result.unwrap();
        assert_abs_diff_eq!(params.freq.res, 80e3, epsilon = 1e-10);

        let data = get_1065880128_meta();
        let context = mwalibContext::new(&data.metafits, &data.gpuboxes)
            .expect("Failed to create mwalib context");
        let args = CalibrateUserArgs {
            metafits: Some(data.metafits),
            gpuboxes: Some(data.gpuboxes),
            mwafs: Some(data.mwafs),
            source_list: data.source_list,
            // 200e3 should be a multiple of 40kHz
            freq_res: Some(200e3),
            ..Default::default()
        };
        let result = CalibrateParams::new(args, context);
        assert!(
            result.is_ok(),
            "Expected CalibrateParams to have been successfully created"
        );
        let params = result.unwrap();
        assert_abs_diff_eq!(params.freq.res, 200e3, epsilon = 1e-10);
    }

    #[test]
    #[serial]
    fn test_new_params_freq_averaging_fail() {
        // The native freq. resolution is 40kHz.
        let data = get_1065880128_meta();
        let context = mwalibContext::new(&data.metafits, &data.gpuboxes)
            .expect("Failed to create mwalib context");
        let args = CalibrateUserArgs {
            metafits: Some(data.metafits),
            gpuboxes: Some(data.gpuboxes),
            mwafs: Some(data.mwafs),
            source_list: data.source_list,
            // 10e3 is not a multiple of 40kHz
            freq_res: Some(10e3),
            ..Default::default()
        };
        let result = CalibrateParams::new(args, context);
        assert!(
            result.is_err(),
            "Expected CalibrateParams to have not been successfully created"
        );

        let data = get_1065880128_meta();
        let context = mwalibContext::new(&data.metafits, &data.gpuboxes)
            .expect("Failed to create mwalib context");
        let args = CalibrateUserArgs {
            metafits: Some(data.metafits),
            gpuboxes: Some(data.gpuboxes),
            mwafs: Some(data.mwafs),
            source_list: data.source_list,
            // 79e3 is not a multiple of 40kHz
            freq_res: Some(79e3),
            ..Default::default()
        };
        let result = CalibrateParams::new(args, context);
        assert!(
            result.is_err(),
            "Expected CalibrateParams to have not been successfully created"
        );
    }

    #[test]
    #[serial]
    fn test_new_params_tile_flags() {
        // 1065880128 has two flagged tiles in its metafits (82 and 123).
        let data = get_1065880128_meta();
        let context = mwalibContext::new(&data.metafits, &data.gpuboxes)
            .expect("Failed to create mwalib context");
        let args = CalibrateUserArgs {
            metafits: Some(data.metafits),
            gpuboxes: Some(data.gpuboxes),
            mwafs: Some(data.mwafs),
            source_list: data.source_list,
            tile_flags: Some(vec![1, 2, 3]),
            ..Default::default()
        };
        let result = CalibrateParams::new(args, context);
        assert!(
            result.is_ok(),
            "Expected CalibrateParams to have been successfully created"
        );

        let params = result.unwrap();
        assert_eq!(params.tile_flags.len(), 5);
        assert!(params.tile_flags.contains(&1));
        assert!(params.tile_flags.contains(&2));
        assert!(params.tile_flags.contains(&3));
        assert!(params.tile_flags.contains(&82));
        assert!(params.tile_flags.contains(&123));
        assert_eq!(params.num_unflagged_tiles, 123);
        assert_eq!(params.count_flagged_before_this_ant(1), 0);
        assert_eq!(params.count_flagged_before_this_ant(2), 1);
        assert_eq!(params.count_flagged_before_this_ant(3), 2);
        assert_eq!(params.count_flagged_before_this_ant(4), 3);
        assert_eq!(params.count_flagged_before_this_ant(127), 5);
    }

    #[test]
    #[serial]
    fn test_new_params_tile_flags_fail() {
        let data = get_1065880128_meta();
        let context = mwalibContext::new(&data.metafits, &data.gpuboxes)
            .expect("Failed to create mwalib context");
        let args = CalibrateUserArgs {
            metafits: Some(data.metafits),
            gpuboxes: Some(data.gpuboxes),
            mwafs: Some(data.mwafs),
            source_list: data.source_list,
            tile_flags: Some(vec![1, 128]),
            ..Default::default()
        };
        let result = CalibrateParams::new(args, context);
        assert!(
            result.is_err(),
            "Expected CalibrateParams to have not been successfully created"
        );
    }

    // astropy doesn't exactly agree with the numbers below, I think because the
    // LST listed in MWA metafits files doesn't agree with what astropy thinks
    // it should be. But, it's all very close.
    #[test]
    fn test_lst_from_mwalib_timestep_native() {
        let context = mwalibContext::new(&"tests/1065880128.metafits", &[]).unwrap();
        let time_res = context.integration_time_milliseconds as f64 / 1e3;
        let new_lst = lst_from_timestep(0, &context, time_res);
        // gpstime 1065880128.75
        assert_abs_diff_eq!(new_lst, 6.074877917424663, epsilon = 1e-10);

        let new_lst = lst_from_timestep(1, &context, time_res);
        // gpstime 1065880129.25
        assert_abs_diff_eq!(new_lst, 6.074914378000397, epsilon = 1e-10);
    }

    #[test]
    fn test_lst_from_mwalib_timestep_averaged() {
        let context = mwalibContext::new(&"tests/1065880128.metafits", &[]).unwrap();
        // The native time res. is 0.5s, let's make our target 2s here.
        let time_res = 2.0;
        let new_lst = lst_from_timestep(0, &context, time_res);
        // gpstime 1065880129.5
        assert_abs_diff_eq!(new_lst, 6.074932608288263, epsilon = 1e-10);

        let new_lst = lst_from_timestep(1, &context, time_res);
        // gpstime 1065880131.5
        assert_abs_diff_eq!(new_lst, 6.075078450591198, epsilon = 1e-10);
    }

    #[test]
    #[serial]
    #[ignore]
    fn test_new_params_real_data() {
        let data = get_1065880128();
        let context = mwalibContext::new(&data.metafits, &data.gpuboxes)
            .expect("Failed to create mwalib context");
        let args = CalibrateUserArgs {
            metafits: Some(data.metafits),
            gpuboxes: Some(data.gpuboxes),
            mwafs: Some(data.mwafs),
            source_list: data.source_list,
            ..Default::default()
        };
        let result = CalibrateParams::new(args, context);
        assert!(
            result.is_ok(),
            "Expected CalibrateParams to have been successfully created"
        );
    }
}
