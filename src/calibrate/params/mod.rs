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

    /// Optional paths to mwaf files.
    pub(crate) cotter_flags: Option<CotterFlags>,

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
    /// Create a new params struct from Rust structs directly.
    ///
    /// If the time or frequency resolution aren't specified, they default to
    /// the observation's native resolution.
    ///
    /// Source list vetoing is not performed in this function.
    pub fn new(
        context: mwalibContext,
        cotter_flags: Option<CotterFlags>,
        beam: mwa_hyperbeam::fee::FEEBeam,
        source_list: mwa_hyperdrive_core::SourceList,
        ranked_sources: Vec<RankedSource>,
        fine_channel_flags: Vec<usize>,
        time_res_seconds: Option<f64>,
        fine_chan_freq_res_hz: Option<f64>,
    ) -> Result<Self, InvalidArgsError> {
        let native_time_res = context.integration_time_milliseconds as f64 / 1e3;
        let time_res = time_res_seconds.unwrap_or(native_time_res);
        if time_res % native_time_res != 0.0 {
            return Err(InvalidArgsError::InvalidTimeResolution {
                got: time_res,
                native: native_time_res,
            });
        }
        let num_time_steps_to_average = time_res / native_time_res;
        let timesteps = (0..context.timesteps.len() / num_time_steps_to_average as usize).collect();

        let native_freq_res = context.fine_channel_width_hz as f64;
        let freq_res = fine_chan_freq_res_hz.unwrap_or(native_freq_res);
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
        let num_components = source_list
            .values()
            .fold(0, |a, src| a + src.components.len());
        debug!("There are {} components across all sources", num_components);

        Ok(Self {
            context,
            cotter_flags,
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
    /// If the time or frequency resolution aren't specified, they default to
    /// the observation's native resolution.
    ///
    /// Source list vetoing is performed in this function, using the specified
    /// number of sources and/or the veto threshold.
    pub fn new_from_args(
        args: crate::calibrate::args::CalibrateUserArgs,
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

        // Set up the beam (requires the MWA_BEAM_FILE variable to be set).
        debug!("Creating beam object");
        let beam = FEEBeam::new_from_env()?;

        debug!("Creating mwalib context");
        let context = mwalibContext::new(&metafits, &gpuboxes)?;

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

        // Print some high-level information.
        info!("Calibrating obsid {}", context.obsid);
        info!("Using metafits: {}", context.metafits_filename);
        info!("Using {} gpubox files", gpuboxes.len());
        match &cotter_flags {
            Some(_) => info!("Using supplied cotter flags"),
            None => warn!("No cotter flags files specified"),
        }

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
        info!("Found {} sources from the source list", source_list.len());

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
        info!("Using {} sources", source_list.len());
        debug!("Using sources: {:#?}", source_list.keys());
        if source_list.len() > 10000 {
            warn!("Using more than 10,000 sources!");
        } else if source_list.is_empty() {
            return Err(InvalidArgsError::NoSourcesAfterVeto);
        }

        CalibrateParams::new(
            context,
            cotter_flags,
            beam,
            source_list,
            ranked_sources,
            fine_channel_flags,
            args.time_res,
            args.freq_res,
        )
    }

    // /// Get the timestep needed to read in data.
    // pub fn get_timestep(&self) -> usize {
    //     self.timestep
    // }

    /// Get the LST [radians].
    pub fn get_lst(&self) -> f64 {
        self.lst
    }

    /// Increment the timestep and LST.
    pub fn next_timestep(&mut self) {
        self.timestep += 1;
        self.lst = lst_from_timestep(self.timestep, &self.context, self.time_res);
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

    use approx::*;
    // Need to use serial tests because HDF5 is not necessarily reentrant.
    use serial_test::serial;

    fn get_srclist() -> SourceList {
        mwa_hyperdrive_srclist::read::read_source_list_file(
            &PathBuf::from("tests/srclist_pumav3_EoR0aegean_EoR1pietro+ForA_1065880128_100.yaml"),
            &SourceListType::Hyperdrive,
        )
        .unwrap()
    }

    fn get_context_and_beam() -> (mwalibContext, FEEBeam) {
        (
            mwalibContext::new(&"tests/1065880128.metafits", &[]).unwrap(),
            mwa_hyperbeam::fee::FEEBeam::new_from_env().unwrap(),
        )
    }

    #[test]
    #[serial]
    fn test_new_params() {
        let (context, beam) = get_context_and_beam();
        let srclist = get_srclist();
        let result = CalibrateParams::new(
            context,
            None,
            beam,
            srclist.clone(),
            // ranked_sources isn't tested.
            vec![],
            vec![],
            None,
            None,
        );
        assert!(result.is_ok());
        let params = result.unwrap();
        // The default time resolution should be 0.5s, as per the metafits.
        assert_abs_diff_eq!(params.time_res, 0.5, epsilon = 1e-10);
        // The default freq resolution should be 40kHz, as per the metafits.
        assert_abs_diff_eq!(params.freq.res, 40e3, epsilon = 1e-10);

        let (context, beam) = get_context_and_beam();
        let result = CalibrateParams::new(
            context,
            None,
            beam,
            srclist.clone(),
            // ranked_sources isn't tested.
            vec![],
            vec![],
            // 1.0 should be a multiple of 0.5s
            Some(1.0),
            None,
        );
        assert!(
            result.is_ok(),
            "Expected CalibrateParams to have been successfully created"
        );
        let params = result.unwrap();
        assert_abs_diff_eq!(params.time_res, 1.0, epsilon = 1e-10);

        let (context, beam) = get_context_and_beam();
        let result = CalibrateParams::new(
            context,
            None,
            beam,
            srclist.clone(),
            // ranked_sources isn't tested.
            vec![],
            vec![],
            // 2.0 should be a multiple of 0.5s
            Some(2.0),
            None,
        );
        assert!(
            result.is_ok(),
            "Expected CalibrateParams to have been successfully created"
        );
        let params = result.unwrap();
        assert_abs_diff_eq!(params.time_res, 2.0, epsilon = 1e-10);

        let (context, beam) = get_context_and_beam();
        let result = CalibrateParams::new(
            context,
            None,
            beam,
            srclist.clone(),
            // ranked_sources isn't tested.
            vec![],
            vec![],
            None,
            // 80e3 should be a multiple of 40kHz
            Some(80e3),
        );
        assert!(
            result.is_ok(),
            "Expected CalibrateParams to have been successfully created"
        );
        let params = result.unwrap();
        assert_abs_diff_eq!(params.freq.res, 80e3, epsilon = 1e-10);

        let (context, beam) = get_context_and_beam();
        let result = CalibrateParams::new(
            context,
            None,
            beam,
            srclist,
            // ranked_sources isn't tested.
            vec![],
            vec![],
            None,
            // 200e3 should be a multiple of 40kHz
            Some(200e3),
        );
        assert!(
            result.is_ok(),
            "Expected CalibrateParams to have been successfully created"
        );
        let params = result.unwrap();
        assert_abs_diff_eq!(params.freq.res, 200e3, epsilon = 1e-10);
    }

    #[test]
    #[serial]
    fn bad_time_params() {
        let (context, beam) = get_context_and_beam();
        let srclist = get_srclist();
        let result = CalibrateParams::new(
            context,
            None,
            beam,
            srclist.clone(),
            // ranked_sources isn't tested.
            vec![],
            vec![],
            // 0.75 is not a multiple of 0.5s
            Some(0.75),
            None,
        );
        assert!(result.is_err());

        let (context, beam) = get_context_and_beam();
        let result = CalibrateParams::new(
            context,
            None,
            beam,
            srclist,
            // ranked_sources isn't tested.
            vec![],
            vec![],
            // 1.01 is not a multiple of 0.5s
            Some(1.01),
            None,
        );
        assert!(result.is_err());
    }

    #[test]
    #[serial]
    fn bad_freq_params() {
        let (context, beam) = get_context_and_beam();
        let srclist = get_srclist();
        let result = CalibrateParams::new(
            context,
            None,
            beam,
            srclist.clone(),
            // ranked_sources isn't tested.
            vec![],
            vec![],
            None,
            // 10e3 is not a multiple of 40kHz
            Some(10e3),
        );
        assert!(result.is_err());

        let (context, beam) = get_context_and_beam();
        let result = CalibrateParams::new(
            context,
            None,
            beam,
            srclist,
            // ranked_sources isn't tested.
            vec![],
            vec![],
            None,
            // 50e3 is not a multiple of 40kHz
            Some(50e3),
        );
        assert!(result.is_err());
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
}
