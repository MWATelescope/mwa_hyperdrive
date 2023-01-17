// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

use std::{
    collections::{HashMap, HashSet},
    f64::consts::TAU,
    io::Write,
    ops::{Deref, Div, Sub},
    path::PathBuf,
    str::FromStr,
    thread::{self, ScopedJoinHandle},
};

use clap::Parser;
use crossbeam_channel::{bounded, unbounded};
use crossbeam_utils::atomic::AtomicCell;
use hifitime::{Duration, Epoch};
use indicatif::{MultiProgress, ProgressBar, ProgressDrawTarget, ProgressStyle};
use itertools::Itertools;
use log::{debug, info, log_enabled, trace, warn, Level::Debug};
use marlu::{
    constants::{FREQ_WEIGHT_FACTOR, TIME_WEIGHT_FACTOR, VEL_C},
    math::num_tiles_from_num_cross_correlation_baselines,
    pos::xyz::xyzs_to_cross_uvws,
    precession::precess_time,
    HADec, Jones, LatLngHeight, MwaObsContext, RADec, XyzGeodetic, UVW,
};
use ndarray::{prelude::*, Zip};
use num_complex::Complex;
use rayon::prelude::*;
use scopeguard::defer_on_unwind;
use thiserror::Error;
use vec1::Vec1;

use crate::{
    averaging::{
        channels_to_chanblocks, parse_freq_average_factor, parse_time_average_factor,
        timesteps_to_timeblocks, AverageFactorError, Chanblock, Timeblock,
    },
    beam::{create_fee_beam_object, create_no_beam_object, Beam, Delays},
    constants::{DEFAULT_CUTOFF_DISTANCE, DEFAULT_VETO_THRESHOLD},
    context::ObsContext,
    cuda,
    di_calibrate::calibrate_timeblock,
    filenames::{InputDataTypes, SUPPORTED_CALIBRATED_INPUT_FILE_COMBINATIONS},
    glob::get_single_match_from_glob,
    help_texts::*,
    math::{average_epoch, TileBaselineFlags},
    messages,
    model::{
        new_sky_modeller, ModelError, ModellerInfo, SkyModeller, SkyModellerCpu, SkyModellerCuda,
    },
    srclist::{
        read::read_source_list_file, veto_sources, ComponentList, Source, SourceList,
        SourceListType,
    },
    unit_parsing::WAVELENGTH_FORMATS,
    vis_io::{
        read::{
            MsReader, RawDataCorrections, RawDataReader, UvfitsReader, VisInputType, VisRead,
            VisReadError,
        },
        write::{write_vis, VisOutputType, VisTimestep, VIS_OUTPUT_EXTENSIONS},
    },
    HyperdriveError,
};

pub(crate) const DEFAULT_OUTPUT_PEEL_FILENAME: &str = "hyperdrive_peeled.uvfits";
pub(crate) const DEFAULT_OUTPUT_IONO_CONSTS: &str = "hyperdrive_iono_consts.json";
pub(crate) const DEFAULT_TIME_AVERAGE_FACTOR: &str = "8s";
pub(crate) const DEFAULT_FREQ_AVERAGE_FACTOR: &str = "80kHz";
pub(crate) const DEFAULT_IONO_FREQ_AVERAGE_FACTOR: &str = "1.28MHz";
pub(crate) const DEFAULT_OUTPUT_TIME_AVERAGE_FACTOR: &str = "8s";
pub(crate) const DEFAULT_OUTPUT_FREQ_AVERAGE_FACTOR: &str = "80kHz";
pub(crate) const DEFAULT_UVW_MIN: &str = "0λ";

lazy_static::lazy_static! {
    static ref DUT1: Duration = Duration::from_seconds(0.0);

    static ref VIS_OUTPUTS_HELP: String = format!("The paths to the files where the peeled visibilities are written. Supported formats: {}", *VIS_OUTPUT_EXTENSIONS);

    static ref TIME_AVERAGE_FACTOR_HELP: String = format!("The number of timesteps to use per timeblock *during* peeling. Also supports a target time resolution (e.g. 8s). If this is 0, then all data are averaged together. Default: {DEFAULT_TIME_AVERAGE_FACTOR}. e.g. If this variable is 4, then peeling is performed with 4 timesteps per timeblock. If the variable is instead 4s, then each timeblock contains up to 4s worth of data.");

    static ref FREQ_AVERAGE_FACTOR_HELP: String = format!("The number of fine-frequency channels to average together *before* peeling. Also supports a target time resolution (e.g. 80kHz). If this is 0, then all data is averaged together. Default: {DEFAULT_FREQ_AVERAGE_FACTOR}. e.g. If the input data is in 20kHz resolution and this variable was 2, then we average 40kHz worth of data into a chanblock before peeling. If the variable is instead 40kHz, then each chanblock contains up to 40kHz worth of data.");

    static ref IONO_FREQ_AVERAGE_FACTOR_HELP: String = format!("The number of fine-frequency channels to average together *during* peeling. Also supports a target time resolution (e.g. 1.28MHz). Cannot be 0. Default: {DEFAULT_IONO_FREQ_AVERAGE_FACTOR}. e.g. If the input data is in 40kHz resolution and this variable was 2, then we average 80kHz worth of data into a chanblock during peeling. If the variable is instead 1.28MHz, then each chanblock contains 32 fine channels.");

    static ref OUTPUT_TIME_AVERAGE_FACTOR_HELP: String = format!("The number of timeblocks to average together when writing out visibilities. Also supports a target time resolution (e.g. 8s). If this is 0, then all data are averaged together. Default: {DEFAULT_OUTPUT_TIME_AVERAGE_FACTOR}. e.g. If this variable is 4, then 8 timesteps are averaged together as a timeblock in the output visibilities.");

    static ref OUTPUT_FREQ_AVERAGE_FACTOR_HELP: String = format!("The number of fine-frequency channels to average together when writing out visibilities. Also supports a target time resolution (e.g. 80kHz). If this is 0, then all data are averaged together. Default: {DEFAULT_OUTPUT_FREQ_AVERAGE_FACTOR}. This is multiplicative with the freq average factor; e.g. If this variable is 4, and the freq average factor is 2, then 8 fine-frequency channels are averaged together as a chanblock in the output visibilities.");

    static ref UVW_MIN_HELP: String = format!("The minimum UVW length to use. This value must have a unit annotated. Allowed units: {}. Default: {DEFAULT_UVW_MIN}", *WAVELENGTH_FORMATS);

    static ref UVW_MAX_HELP: String = format!("The maximum UVW length to use. This value must have a unit annotated. Allowed units: {}. No default.", *WAVELENGTH_FORMATS);
}

// Arguments that are exposed to users. All arguments except bools should be
// optional.
#[derive(Parser, Debug, Clone, Default)]
pub struct PeelArgs {
    /// Paths to input data files to be calibrated. These can include a metafits
    /// file, gpubox files, mwaf files, a measurement set and/or uvfits files.
    #[clap(short, long, multiple_values(true), help_heading = "INPUT FILES")]
    pub data: Option<Vec<String>>,

    /// Path to the sky-model source list file.
    #[clap(short, long, help_heading = "INPUT FILES")]
    pub source_list: Option<String>,

    #[clap(long, help = SOURCE_LIST_TYPE_HELP.as_str(), help_heading = "INPUT FILES")]
    pub source_list_type: Option<String>,

    /// Use a DUT1 value of 0 seconds rather than what is in the input data.
    #[clap(long, help_heading = "INPUT FILES")]
    pub ignore_dut1: bool,

    #[clap(short, long, multiple_values(true), help = VIS_OUTPUTS_HELP.as_str(), help_heading = "OUTPUT FILES")]
    pub outputs: Option<Vec<PathBuf>>,

    /// The number of sources to peel. Peel sources are treated the same as
    /// "ionospherically subtracted" sources, except before subtracting, a "DI
    /// calibration" is done between the iono-rotated model and the data. This
    /// allows for scintillation and any other phase shift to be corrected.
    #[clap(long = "peel", help_heading = "SKY-MODEL SOURCES", default_value = "0")]
    pub num_sources_to_peel: usize,

    /// The number of sources to "ionospherically subtract". That is, a λ²
    /// dependence is found for each of these sources and removed. The number of
    /// iono subtract sources cannot be more than the number of sources to
    /// subtract. The default is the number of sources in the source list, after
    /// vetoing.
    #[clap(long = "iono-sub", help_heading = "SKY-MODEL SOURCES")]
    pub num_sources_to_iono_subtract: Option<usize>,

    /// The number of sources to "ionospherically subtract" in serial, not
    /// parallel. The larger this number is, the better the quality of iono
    /// subtraction, but the slower the whole process takes.
    #[clap(
        long = "iono-sub-serial",
        help_heading = "PEELING",
        default_value = "10"
    )]
    pub num_sources_to_iono_subtract_in_serial: usize,

    /// The number of sources to subtract. This subtraction just uses the sky
    /// model directly; no peeling or ionospheric λ² is found. There must be at
    /// least as many sources subtracted as there are ionospherically
    /// subtracted. The default is the number of sources in the source list,
    /// after vetoing.
    #[clap(long = "sub", help_heading = "SKY-MODEL SOURCES")]
    pub num_sources_to_subtract: Option<usize>,

    #[clap(long, help = SOURCE_DIST_CUTOFF_HELP.as_str(), help_heading = "SKY-MODEL SOURCES")]
    pub source_dist_cutoff: Option<f64>,

    #[clap(long, help = VETO_THRESHOLD_HELP.as_str(), help_heading = "SKY-MODEL SOURCES")]
    pub veto_threshold: Option<f64>,

    #[cfg(feature = "cuda")]
    /// Use the CPU for visibility generation. This is deliberately made
    /// non-default because using a GPU is much faster.
    #[clap(long, help_heading = "MODELLING")]
    pub cpu: bool,

    /// The path to the HDF5 MWA FEE beam file. If not specified, this must be
    /// provided by the MWA_BEAM_FILE environment variable.
    #[clap(long, help_heading = "MODELLING")]
    pub beam_file: Option<PathBuf>,

    /// Pretend that all MWA dipoles are alive and well, ignoring whatever is in
    /// the metafits file.
    #[clap(long, help_heading = "MODELLING")]
    pub unity_dipole_gains: bool,

    #[clap(long, multiple_values(true), help = DIPOLE_DELAYS_HELP.as_str(), help_heading = "MODELLING")]
    pub delays: Option<Vec<u32>>,

    /// Don't apply a beam response when generating a sky model. The default is
    /// to use the FEE beam.
    #[clap(long, help_heading = "MODELLING")]
    pub no_beam: bool,

    #[clap(short, long, help = TIME_AVERAGE_FACTOR_HELP.as_str(), help_heading = "AVERAGING")]
    pub time_average_factor: Option<String>,

    #[clap(short, long, help = FREQ_AVERAGE_FACTOR_HELP.as_str(), help_heading = "AVERAGING")]
    pub freq_average_factor: Option<String>,

    #[clap(short, long, help = IONO_FREQ_AVERAGE_FACTOR_HELP.as_str(), help_heading = "AVERAGING")]
    pub iono_freq_average_factor: Option<String>,

    #[clap(long, help = OUTPUT_TIME_AVERAGE_FACTOR_HELP.as_str(), help_heading = "AVERAGING")]
    pub output_time_average_factor: Option<String>,

    #[clap(long, help = OUTPUT_FREQ_AVERAGE_FACTOR_HELP.as_str(), help_heading = "AVERAGING")]
    pub output_freq_average_factor: Option<String>,

    /// The timesteps to use from the input data. The timesteps will be
    /// ascendingly sorted for calibration. No duplicates are allowed. The
    /// default is to use all unflagged timesteps.
    #[clap(long, multiple_values(true), help_heading = "CALIBRATION")]
    pub timesteps: Option<Vec<usize>>,

    /// Use all timesteps in the data, including flagged ones. The default is to
    /// use all unflagged timesteps.
    #[clap(long, conflicts_with("timesteps"), help_heading = "CALIBRATION")]
    pub use_all_timesteps: bool,

    #[clap(long, help = UVW_MIN_HELP.as_str(), help_heading = "CALIBRATION")]
    pub uvw_min: Option<String>,

    #[clap(long, help = UVW_MAX_HELP.as_str(), help_heading = "CALIBRATION")]
    pub uvw_max: Option<String>,

    // #[clap(long, help = MAX_ITERATIONS_HELP.as_str(), help_heading = "CALIBRATION")]
    // pub max_iterations: Option<u32>,

    // #[clap(long, help = STOP_THRESHOLD_HELP.as_str(), help_heading = "CALIBRATION")]
    // pub stop_thresh: Option<f64>,

    // #[clap(long, help = MIN_THRESHOLD_HELP.as_str(), help_heading = "CALIBRATION")]
    // pub min_thresh: Option<f64>,
    #[clap(
        long, help = ARRAY_POSITION_HELP.as_str(), help_heading = "ARRAY",
        number_of_values = 3,
        allow_hyphen_values = true,
        value_names = &["LONG_RAD", "LAT_RAD", "HEIGHT_M"]
    )]
    pub array_position: Option<Vec<f64>>,

    /// If specified, don't precess the array to J2000. We assume that sky-model
    /// sources are specified in the J2000 epoch.
    #[clap(long, help_heading = "ARRAY")]
    pub no_precession: bool,

    /// Additional tiles to be flagged. These values correspond to either the
    /// values in the "Antenna" column of HDU 2 in the metafits file (e.g. 0 3
    /// 127), or the "TileName" (e.g. Tile011).
    #[clap(long, multiple_values(true), help_heading = "FLAGGING")]
    pub tile_flags: Option<Vec<String>>,

    /// If specified, pretend that all tiles are unflagged in the input data.
    #[clap(long, help_heading = "FLAGGING")]
    pub ignore_input_data_tile_flags: bool,

    /// If specified, pretend all fine channels in the input data are unflagged.
    #[clap(long, help_heading = "FLAGGING")]
    pub ignore_input_data_fine_channel_flags: bool,

    /// The fine channels to be flagged in each coarse channel. e.g. 0 1 16 30
    /// 31 are typical for 40 kHz data. If this is not specified, it defaults to
    /// flagging 80 kHz (or as close to this as possible) at the edges, as well
    /// as the centre channel for non-MWAX data.
    #[clap(long, multiple_values(true), help_heading = "FLAGGING")]
    pub fine_chan_flags_per_coarse_chan: Option<Vec<usize>>,

    /// The fine channels to be flagged across the whole observation band. e.g.
    /// 0 767 are the first and last fine channels for 40 kHz data.
    #[clap(long, multiple_values(true), help_heading = "FLAGGING")]
    pub fine_chan_flags: Option<Vec<usize>>,

    #[clap(long, help = PFB_FLAVOUR_HELP.as_str(), help_heading = "RAW MWA DATA")]
    pub pfb_flavour: Option<String>,

    /// When reading in raw MWA data, don't apply digital gains.
    #[clap(long, help_heading = "RAW MWA DATA")]
    pub no_digital_gains: bool,

    /// When reading in raw MWA data, don't apply cable length corrections. Note
    /// that some data may have already had the correction applied before it was
    /// written.
    #[clap(long, help_heading = "RAW MWA DATA")]
    pub no_cable_length_correction: bool,

    /// When reading in raw MWA data, don't apply geometric corrections. Note
    /// that some data may have already had the correction applied before it was
    /// written.
    #[clap(long, help_heading = "RAW MWA DATA")]
    pub no_geometric_correction: bool,

    /// When reading in visibilities and generating sky-model visibilities,
    /// don't draw progress bars.
    #[clap(long, help_heading = "USER INTERFACE")]
    pub no_progress_bars: bool,
}

impl PeelArgs {
    pub fn run(self, dry_run: bool) -> Result<(), HyperdriveError> {
        self.run_inner(dry_run)?;
        Ok(())
    }

    fn run_inner(self, dry_run: bool) -> Result<(), PeelError> {
        let PeelArgs {
            data,
            source_list,
            source_list_type,
            ignore_dut1,
            outputs,
            num_sources_to_peel,
            num_sources_to_iono_subtract,
            num_sources_to_iono_subtract_in_serial,
            num_sources_to_subtract,
            source_dist_cutoff,
            veto_threshold,
            cpu,
            beam_file,
            unity_dipole_gains,
            delays,
            no_beam,
            time_average_factor,
            freq_average_factor,
            iono_freq_average_factor,
            output_time_average_factor,
            output_freq_average_factor,
            timesteps,
            use_all_timesteps,
            uvw_min,
            uvw_max,
            array_position,
            no_precession,
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
        } = self;

        // Check that the number of sources to peel, iono subtract and subtract
        // are sensible. When that's done, veto up to the maximum number of
        // sources to subtract.
        let max_num_sources = match (num_sources_to_iono_subtract, num_sources_to_subtract) {
            (Some(is), Some(s)) => {
                if s < is {
                    panic!("The number of sources to subtract ({s}) must be at least equal to the number of sources to iono subtract ({is})");
                }
                Some(s)
            }
            (None, Some(s)) => Some(s),
            (Some(_), None) => None,
            (None, None) => None,
        };

        // If we're going to use a GPU for modelling, get the device info so we
        // can ensure a CUDA-capable device is available, and so we can report
        // it to the user later.
        #[cfg(feature = "cuda")]
        let modeller_info = if cpu {
            ModellerInfo::Cpu
        } else {
            let (device_info, driver_info) = crate::cuda::get_device_info()?;
            ModellerInfo::Cuda {
                device_info,
                driver_info,
            }
        };
        #[cfg(not(feature = "cuda"))]
        let modeller_info = ModellerInfo::Cpu;

        // If the user supplied the array position, unpack it here.
        let array_position = match array_position {
            Some(pos) => {
                if pos.len() != 3 {
                    return Err(PeelError::BadArrayPosition { pos });
                }
                Some(LatLngHeight {
                    longitude_rad: pos[0].to_radians(),
                    latitude_rad: pos[1].to_radians(),
                    height_metres: pos[2],
                })
            }
            None => None,
        };

        // Handle input data. We expect one of three possibilities:
        // - gpubox files, a metafits file (and maybe mwaf files),
        // - a measurement set (and maybe a metafits file), or
        // - uvfits files.
        // If none or multiple of these possibilities are met, then we must fail.
        let input_data_types = match data {
            Some(strings) => InputDataTypes::new(&strings)?,
            None => return Err(PeelError::NoInputData),
        };
        let (input_data, raw_data_corrections): (Box<dyn VisRead>, Option<RawDataCorrections>) =
            match (
                input_data_types.metafits,
                input_data_types.gpuboxes,
                input_data_types.mwafs,
                input_data_types.ms,
                input_data_types.uvfits,
            ) {
                // Valid input for reading raw data.
                (Some(meta), Some(gpuboxes), mwafs, None, None) => {
                    todo!("need user to supply calibration solutions");

                    // Ensure that there's only one metafits.
                    let meta = if meta.len() > 1 {
                        return Err(PeelError::MultipleMetafits(meta));
                    } else {
                        meta.first()
                    };

                    debug!("gpubox files: {:?}", &gpuboxes);
                    debug!("mwaf files: {:?}", &mwafs);

                    let corrections = RawDataCorrections::new(
                        pfb_flavour.as_deref(),
                        !no_digital_gains,
                        !no_cable_length_correction,
                        !no_geometric_correction,
                    )?;
                    let input_data =
                        RawDataReader::new(meta, &gpuboxes, mwafs.as_deref(), corrections)?;

                    messages::InputFileDetails::Raw {
                        obsid: input_data.mwalib_context.metafits_context.obs_id,
                        gpubox_count: gpuboxes.len(),
                        metafits_file_name: meta.display().to_string(),
                        mwaf: input_data.get_flags(),
                        raw_data_corrections: corrections,
                    }
                    .print("Peeling");

                    (Box::new(input_data), Some(corrections))
                }

                // Valid input for reading a measurement set.
                (meta, None, None, Some(ms), None) => {
                    // Only one MS is supported at the moment.
                    let ms: PathBuf = if ms.len() > 1 {
                        return Err(PeelError::MultipleMeasurementSets(ms));
                    } else {
                        ms.first().clone()
                    };

                    // Ensure that there's only one metafits.
                    let meta: Option<&PathBuf> = match meta.as_ref() {
                        None => None,
                        Some(m) => {
                            if m.len() > 1 {
                                return Err(PeelError::MultipleMetafits(m.clone()));
                            } else {
                                Some(m.first())
                            }
                        }
                    };

                    let input_data = MsReader::new(&ms, meta, array_position)?;

                    messages::InputFileDetails::MeasurementSet {
                        obsid: input_data.get_obs_context().obsid,
                        file_name: ms.display().to_string(),
                        metafits_file_name: meta.map(|m| m.display().to_string()),
                    }
                    .print("Peeling");

                    (Box::new(input_data), None)
                }

                // Valid input for reading uvfits files.
                (meta, None, None, None, Some(uvfits)) => {
                    // Only one uvfits is supported at the moment.
                    let uvfits: PathBuf = if uvfits.len() > 1 {
                        return Err(PeelError::MultipleUvfits(uvfits));
                    } else {
                        uvfits.first().clone()
                    };

                    // Ensure that there's only one metafits.
                    let meta: Option<&PathBuf> = match meta.as_ref() {
                        None => None,
                        Some(m) => {
                            if m.len() > 1 {
                                return Err(PeelError::MultipleMetafits(m.clone()));
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
                    .print("Peeling");

                    (Box::new(input_data), None)
                }

                // The following matches are for invalid combinations of input
                // files. Make an error message for the user.
                (Some(_), _, None, None, None) => {
                    let msg = "Received only a metafits file; a uvfits file, a measurement set or gpubox files are required.";
                    return Err(PeelError::InvalidDataInput(msg));
                }
                (Some(_), _, Some(_), None, None) => {
                    let msg =
                        "Received only a metafits file and mwaf files; gpubox files are required.";
                    return Err(PeelError::InvalidDataInput(msg));
                }
                (None, Some(_), _, None, None) => {
                    let msg = "Received gpuboxes without a metafits file; this is not supported.";
                    return Err(PeelError::InvalidDataInput(msg));
                }
                (None, None, Some(_), None, None) => {
                    let msg = "Received mwaf files without gpuboxes and a metafits file; this is not supported.";
                    return Err(PeelError::InvalidDataInput(msg));
                }
                (_, Some(_), _, Some(_), None) => {
                    let msg = "Received gpuboxes and measurement set files; this is not supported.";
                    return Err(PeelError::InvalidDataInput(msg));
                }
                (_, Some(_), _, None, Some(_)) => {
                    let msg = "Received gpuboxes and uvfits files; this is not supported.";
                    return Err(PeelError::InvalidDataInput(msg));
                }
                (_, _, _, Some(_), Some(_)) => {
                    let msg = "Received uvfits and measurement set files; this is not supported.";
                    return Err(PeelError::InvalidDataInput(msg));
                }
                (_, _, Some(_), Some(_), _) => {
                    let msg = "Received mwafs and measurement set files; this is not supported.";
                    return Err(PeelError::InvalidDataInput(msg));
                }
                (_, _, Some(_), _, Some(_)) => {
                    let msg = "Received mwafs and uvfits files; this is not supported.";
                    return Err(PeelError::InvalidDataInput(msg));
                }
                (None, None, None, None, None) => return Err(PeelError::NoInputData),
            };

        let obs_context = input_data.get_obs_context();

        // If the array position wasn't user defined, try the input data.
        let array_position = array_position.unwrap_or_else(|| {
            trace!("The array position was not specified in the input data; assuming MWA");
            LatLngHeight::mwa()
        });
        let dut1 = if ignore_dut1 { None } else { obs_context.dut1 }
            .unwrap_or_else(|| Duration::from_seconds(0.0));

        let timesteps_to_use = {
            match (use_all_timesteps, timesteps) {
                (true, _) => obs_context.all_timesteps.clone(),
                (false, None) => Vec1::try_from_vec(obs_context.unflagged_timesteps.clone())
                    .map_err(|_| PeelError::NoTimesteps)?,
                (false, Some(mut ts)) => {
                    // Make sure there are no duplicates.
                    let timesteps_hashset: HashSet<&usize> = ts.iter().collect();
                    if timesteps_hashset.len() != ts.len() {
                        return Err(PeelError::DuplicateTimesteps);
                    }

                    // Ensure that all specified timesteps are actually available.
                    for t in &ts {
                        if !(0..obs_context.timestamps.len()).contains(t) {
                            return Err(PeelError::UnavailableTimestep {
                                got: *t,
                                last: obs_context.timestamps.len() - 1,
                            });
                        }
                    }

                    ts.sort_unstable();
                    Vec1::try_from_vec(ts).map_err(|_| PeelError::NoTimesteps)?
                }
            }
        };
        let timestamps = timesteps_to_use.mapped_ref(|i| obs_context.timestamps[*i]);

        let precession_info = precess_time(
            array_position.longitude_rad,
            array_position.latitude_rad,
            obs_context.phase_centre,
            obs_context.timestamps[*timesteps_to_use.first()],
            dut1,
        );
        let (lmst, latitude) = if no_precession {
            (precession_info.lmst, array_position.latitude_rad)
        } else {
            (
                precession_info.lmst_j2000,
                precession_info.array_latitude_j2000,
            )
        };

        // The length of the tile XYZ collection is the total number of tiles in
        // the array, even if some tiles are flagged.
        let total_num_tiles = obs_context.get_total_num_tiles();

        // Assign the tile flags.
        let flagged_tiles =
            obs_context.get_tile_flags(ignore_input_data_tile_flags, tile_flags.as_deref())?;
        let num_unflagged_tiles = total_num_tiles - flagged_tiles.len();
        let num_unflagged_cross_baselines = (num_unflagged_tiles * (num_unflagged_tiles - 1)) / 2;
        if log_enabled!(Debug) {
            obs_context.print_debug_tile_statuses();
        }
        if num_unflagged_tiles == 0 {
            return Err(PeelError::NoTiles);
        }
        messages::ArrayDetails {
            array_position: Some(array_position),
            array_latitude_j2000: if no_precession {
                None
            } else {
                Some(precession_info.array_latitude_j2000)
            },
            total_num_tiles,
            num_unflagged_tiles,
            flagged_tiles: &flagged_tiles
                .iter()
                .sorted()
                .map(|&i| (obs_context.tile_names[i].as_str(), i))
                .collect::<Vec<_>>(),
        }
        .print();

        let dipole_delays = match delays {
            // We have user-provided delays; check that they're are sensible,
            // regardless of whether we actually need them.
            Some(d) => {
                if d.len() != 16 || d.iter().any(|&v| v > 32) {
                    return Err(PeelError::BadDelays);
                }
                Some(Delays::Partial(d))
            }

            // No delays were provided; use whatever was in the input data.
            None => obs_context.dipole_delays.as_ref().cloned(),
        };

        let beam: Box<dyn Beam> = if no_beam {
            create_no_beam_object(total_num_tiles)
        } else {
            let mut dipole_delays = dipole_delays.ok_or(PeelError::NoDelays)?;
            let dipole_gains = if unity_dipole_gains {
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
            };
            if dipole_gains.is_none() {
                // If we don't have dipole gains, we must assume all dipoles are
                // "alive". But, if any dipole delays are 32, then the beam code
                // will still ignore those dipoles. So use ideal dipole delays
                // for all tiles.
                let ideal_delays = dipole_delays.get_ideal_delays();

                // Warn the user if they wanted unity dipole gains but the ideal
                // dipole delays contain 32.
                if unity_dipole_gains && ideal_delays.iter().any(|&v| v == 32) {
                    warn!(
                        "Some ideal dipole delays are 32; these dipoles will not have unity gains"
                    );
                }
                dipole_delays.set_to_ideal_delays();
            }

            create_fee_beam_object(beam_file, total_num_tiles, dipole_delays, dipole_gains)?
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
            return Err(PeelError::NoChannels);
        }

        messages::ObservationDetails {
            dipole_delays: beam.get_ideal_dipole_delays(),
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
            dut1: obs_context.dut1,
            lmst: Some(precession_info.lmst),
            lmst_j2000: if no_precession {
                Some(precession_info.lmst_j2000)
            } else {
                None
            },
            available_timesteps: Some(&obs_context.all_timesteps),
            unflagged_timesteps: Some(&obs_context.unflagged_timesteps),
            using_timesteps: Some(&timesteps_to_use),
            first_timestamp: Some(obs_context.timestamps[*timesteps_to_use.first()]),
            last_timestamp: if timesteps_to_use.len() > 1 {
                Some(obs_context.timestamps[*timesteps_to_use.last()])
            } else {
                None
            },
            time_res: obs_context.time_res,
            total_num_channels: obs_context.fine_chan_freqs.len(),
            num_unflagged_channels: Some(unflagged_fine_chan_freqs.len()),
            flagged_chans_per_coarse_chan: Some(&obs_context.flagged_fine_chans_per_coarse_chan),
            first_freq_hz: Some(*obs_context.fine_chan_freqs.first() as f64),
            last_freq_hz: Some(*obs_context.fine_chan_freqs.last() as f64),
            first_unflagged_freq_hz: unflagged_fine_chan_freqs.first().copied(),
            last_unflagged_freq_hz: unflagged_fine_chan_freqs.last().copied(),
            freq_res_hz: obs_context.freq_res,
        }
        .print();

        // Parse vis and iono const outputs.
        let (vis_outputs, iono_outputs) = {
            let mut vis_outputs = vec![];
            let mut iono_outputs = vec![];
            match outputs {
                // Defaults.
                None => {
                    let pb = PathBuf::from(DEFAULT_OUTPUT_PEEL_FILENAME);
                    let vis_type = pb
                        .extension()
                        .and_then(|os_str| os_str.to_str())
                        .and_then(|s| VisOutputType::from_str(s).ok())
                        // Tests should pick up a bad default filename.
                        .expect("DEFAULT_OUTPUT_PEEL_FILENAME has an unhandled extension!");
                    vis_outputs.push((pb, vis_type));
                    // TODO: Type this and clean up
                    let pb = PathBuf::from(DEFAULT_OUTPUT_IONO_CONSTS);
                    if pb.extension().and_then(|os_str| os_str.to_str()) != Some("json") {
                        // Tests should pick up a bad default filename.
                        panic!("DEFAULT_OUTPUT_IONO_CONSTS has an unhandled extension!");
                    }
                    iono_outputs.push(pb);
                }
                Some(os) => {
                    for file in os {
                        // Is the output file type supported?
                        let ext = file.extension().and_then(|os_str| os_str.to_str());
                        match (
                            ext.and_then(|s| VisOutputType::from_str(s).ok()),
                            ext.map(|s| s == "json"),
                        ) {
                            (Some(vis_type), _) => {
                                crate::vis_io::write::can_write_to_file(&file).unwrap();
                                vis_outputs.push((file, vis_type));
                            }
                            (None, Some(true)) => {
                                iono_outputs.push(file);
                            }
                            (None, _) => {
                                return Err(PeelError::VisFileType {
                                    ext: ext.unwrap_or("<no extension>").to_string(),
                                })
                            }
                        }
                    }
                }
            };
            (vis_outputs, iono_outputs)
        };
        if vis_outputs.len() + iono_outputs.len() == 0 {
            return Err(PeelError::NoOutput);
        }
        let vis_outputs = Vec1::try_from_vec(vis_outputs).ok();

        // Set up the timeblocks.
        let time_average_factor = {
            let default_time_average_factor = parse_time_average_factor(
                obs_context.time_res,
                Some(DEFAULT_TIME_AVERAGE_FACTOR),
                1,
            )
            .expect("default is sensible");

            parse_time_average_factor(
                obs_context.time_res,
                time_average_factor.as_deref(),
                default_time_average_factor,
            )
            .map_err(|e| match e {
                AverageFactorError::Zero => PeelError::CalTimeFactorZero,
                AverageFactorError::NotInteger => PeelError::CalTimeFactorNotInteger,
                AverageFactorError::NotIntegerMultiple { out, inp } => {
                    PeelError::CalTimeResNotMultiple { out, inp }
                }
                AverageFactorError::Parse(e) => PeelError::ParseCalTimeAverageFactor(e),
            })?
        };
        // Check that the factor is not too big.
        let time_average_factor = if time_average_factor > timesteps_to_use.len() {
            warn!(
                "Cannot average {time_average_factor} timesteps; only {} are being used. Capping.",
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

        // Set up the chanblocks.
        let freq_average_factor = {
            let default_freq_average_factor = parse_freq_average_factor(
                obs_context.freq_res,
                Some(DEFAULT_FREQ_AVERAGE_FACTOR),
                1,
            )
            .expect("default is sensible");

            parse_freq_average_factor(
                obs_context.freq_res,
                freq_average_factor.as_deref(),
                default_freq_average_factor,
            )
            .map_err(|e| match e {
                AverageFactorError::Zero => PeelError::CalFreqFactorZero,
                AverageFactorError::NotInteger => PeelError::CalFreqFactorNotInteger,
                AverageFactorError::NotIntegerMultiple { out, inp } => {
                    PeelError::CalFreqResNotMultiple { out, inp }
                }
                AverageFactorError::Parse(e) => PeelError::ParseCalFreqAverageFactor(e),
            })?
        };
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
            &HashSet::new(),
        );
        // There must be at least one chanblock for calibration.
        match fences.as_slice() {
            // No fences is the same as no chanblocks.
            [] => return Err(PeelError::NoChannels),
            [f] => {
                // Check that the chanblocks aren't all flagged.
                if f.chanblocks.is_empty() {
                    return Err(PeelError::NoChannels);
                }
            }
            [f, ..] => {
                // Check that the chanblocks aren't all flagged.
                if f.chanblocks.is_empty() {
                    return Err(PeelError::NoChannels);
                }
                // TODO: Allow picket fence.
                eprintln!("\"Picket fence\" data detected. hyperdrive does not support this right now -- exiting.");
                eprintln!("See for more info: https://MWATelescope.github.io/mwa_hyperdrive/defs/mwa/picket_fence.html");
                std::process::exit(1);
            }
        }
        let fences = Vec1::try_from_vec(fences).map_err(|_| PeelError::NoChannels)?;
        let all_fine_chan_freqs_hz =
            Vec1::try_from_vec(fences[0].chanblocks.iter().map(|c| c._freq).collect()).unwrap();
        let all_fine_chan_lambdas_m = all_fine_chan_freqs_hz.mapped_ref(|f| VEL_C / *f);

        let output_time_average_factor = {
            let default_output_time_average_factor = parse_time_average_factor(
                obs_context.time_res,
                Some(DEFAULT_OUTPUT_TIME_AVERAGE_FACTOR),
                1,
            )
            .expect("default is sensible");

            parse_time_average_factor(
                obs_context.time_res,
                output_time_average_factor.as_deref(),
                default_output_time_average_factor,
            )
            .map_err(|e| match e {
                AverageFactorError::Zero => PeelError::OutputVisTimeAverageFactorZero,
                AverageFactorError::NotInteger => PeelError::OutputVisTimeFactorNotInteger,
                AverageFactorError::NotIntegerMultiple { out, inp } => {
                    PeelError::OutputVisTimeResNotMultiple { out, inp }
                }
                AverageFactorError::Parse(e) => PeelError::ParseOutputVisTimeAverageFactor(e),
            })?
        };
        let output_freq_average_factor = {
            let default_output_freq_average_factor = parse_freq_average_factor(
                obs_context.freq_res.map(|f| f * freq_average_factor as f64),
                Some(DEFAULT_OUTPUT_FREQ_AVERAGE_FACTOR),
                1,
            )
            .expect("default is sensible");

            parse_freq_average_factor(
                obs_context.freq_res,
                output_freq_average_factor.as_deref(),
                default_output_freq_average_factor,
            )
            .map_err(|e| match e {
                AverageFactorError::Zero => PeelError::OutputVisFreqAverageFactorZero,
                AverageFactorError::NotInteger => PeelError::OutputVisFreqFactorNotInteger,
                AverageFactorError::NotIntegerMultiple { out, inp } => {
                    PeelError::OutputVisFreqResNotMultiple { out, inp }
                }
                AverageFactorError::Parse(e) => PeelError::ParseOutputVisFreqAverageFactor(e),
            })?
        };

        let tile_baseline_flags = TileBaselineFlags::new(total_num_tiles, flagged_tiles);
        let flagged_tiles = &tile_baseline_flags.flagged_tiles;

        let unflagged_tile_xyzs: Vec<XyzGeodetic> = obs_context
            .tile_xyzs
            .par_iter()
            .enumerate()
            .filter(|(tile_index, _)| !flagged_tiles.contains(tile_index))
            .map(|(_, xyz)| *xyz)
            .collect();

        // // Set baseline weights from UVW cuts. Use a lambda from the centroid
        // // frequency if UVW cutoffs are specified as wavelengths.
        // let freq_centroid = obs_context
        //     .fine_chan_freqs
        //     .iter()
        //     .map(|&u| u as f64)
        //     .sum::<f64>()
        //     / obs_context.fine_chan_freqs.len() as f64;
        // let lambda = marlu::constants::VEL_C / freq_centroid;
        // let (uvw_min, uvw_min_metres) = {
        //     let (quantity, unit) = parse_wavelength(
        //         uvw_min
        //             .as_deref()
        //             .unwrap_or(crate::cli::di_calibrate::DEFAULT_UVW_MIN),
        //     )
        //     .map_err(PeelError::ParseUvwMin)?;
        //     match unit {
        //         WavelengthUnit::M => ((quantity, unit), quantity),
        //         WavelengthUnit::L => ((quantity, unit), quantity * lambda),
        //     }
        // };
        // let (uvw_max, uvw_max_metres) = match uvw_max {
        //     None => ((f64::INFINITY, WavelengthUnit::M), f64::INFINITY),
        //     Some(s) => {
        //         let (quantity, unit) = parse_wavelength(&s).map_err(PeelError::ParseUvwMax)?;
        //         match unit {
        //             WavelengthUnit::M => ((quantity, unit), quantity),
        //             WavelengthUnit::L => ((quantity, unit), quantity * lambda),
        //         }
        //     }
        // };

        // let (baseline_weights, num_flagged_baselines) = {
        //     let mut baseline_weights = Vec1::try_from_vec(vec![
        //         1.0;
        //         tile_baseline_flags
        //             .unflagged_cross_baseline_to_tile_map
        //             .len()
        //     ])
        //     .map_err(|_| PeelError::NoTiles)?;
        //     let uvws = xyzs_to_cross_uvws(
        //         &unflagged_tile_xyzs,
        //         obs_context.phase_centre.to_hadec(lmst),
        //     );
        //     assert_eq!(baseline_weights.len(), uvws.len());
        //     let uvw_min = uvw_min_metres.powi(2);
        //     let uvw_max = uvw_max_metres.powi(2);
        //     let mut num_flagged_baselines = 0;
        //     for (uvw, baseline_weight) in uvws.into_iter().zip(baseline_weights.iter_mut()) {
        //         let uvw_length = uvw.u.powi(2) + uvw.v.powi(2) + uvw.w.powi(2);
        //         if uvw_length < uvw_min || uvw_length > uvw_max {
        //             *baseline_weight = 0.0;
        //             num_flagged_baselines += 1;
        //         }
        //     }
        //     (baseline_weights, num_flagged_baselines)
        // };

        // // Make sure the calibration thresholds are sensible.
        // let mut stop_threshold =
        //     stop_thresh.unwrap_or(crate::cli::di_calibrate::DEFAULT_STOP_THRESHOLD);
        // let min_threshold = min_thresh.unwrap_or(crate::cli::di_calibrate::DEFAULT_MIN_THRESHOLD);
        // if stop_threshold > min_threshold {
        //     warn!("Specified stop threshold ({}) is bigger than the min. threshold ({}); capping stop threshold.", stop_threshold, min_threshold);
        //     stop_threshold = min_threshold;
        // }
        // let max_iterations =
        //     max_iterations.unwrap_or(crate::cli::di_calibrate::DEFAULT_MAX_ITERATIONS);

        // messages::CalibrationDetails {
        //     timesteps_per_timeblock: time_average_factor,
        //     channels_per_chanblock: freq_average_factor,
        //     num_timeblocks: timeblocks.len(),
        //     num_chanblocks: fences.first().chanblocks.len(),
        //     uvw_min,
        //     uvw_max,
        //     num_calibration_baselines: baseline_weights.len() - num_flagged_baselines,
        //     total_num_baselines: baseline_weights.len(),
        //     lambda,
        //     freq_centroid,
        //     min_threshold,
        //     stop_threshold,
        //     max_iterations,
        // }
        // .print();

        let mut source_list: SourceList = {
            // Handle the source list argument.
            let sl_pb: PathBuf = match source_list {
                None => return Err(PeelError::NoSourceList),
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
            let (sl, sl_type) = match read_source_list_file(sl_pb, sl_type) {
                Ok((sl, sl_type)) => (sl, sl_type),
                Err(e) => return Err(PeelError::from(e)),
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
        if source_list.is_empty() {
            return Err(PeelError::NoSources);
        }
        veto_sources(
            &mut source_list,
            obs_context.phase_centre,
            lmst,
            latitude,
            &obs_context.coarse_chan_freqs,
            beam.deref(),
            num_sources_to_subtract,
            source_dist_cutoff.unwrap_or(DEFAULT_CUTOFF_DISTANCE),
            veto_threshold.unwrap_or(DEFAULT_VETO_THRESHOLD),
        )?;
        if source_list.is_empty() {
            return Err(PeelError::NoSourcesAfterVeto);
        }

        messages::SkyModelDetails {
            source_list: &source_list,
        }
        .print();

        messages::print_modeller_info(&modeller_info);

        {
            messages::OutputFileDetails {
                output_solutions: &[],
                vis_type: "peeled",
                output_vis: vis_outputs.as_ref(),
                input_vis_time_res: obs_context.time_res,
                input_vis_freq_res: obs_context.freq_res,
                output_vis_time_average_factor: output_time_average_factor,
                output_vis_freq_average_factor: output_freq_average_factor * freq_average_factor,
            }
            .print();
        }

        let iono_freq_average_factor = {
            let default_iono_freq_average_factor = parse_freq_average_factor(
                obs_context.freq_res,
                Some(DEFAULT_IONO_FREQ_AVERAGE_FACTOR),
                1,
            )
            .expect("default is sensible");

            parse_freq_average_factor(
                obs_context.freq_res,
                iono_freq_average_factor.as_deref(),
                default_iono_freq_average_factor,
            )
            .unwrap()
        };
        let low_res_fences = channels_to_chanblocks(
            &obs_context.fine_chan_freqs,
            obs_context.freq_res,
            iono_freq_average_factor,
            &HashSet::new(),
        );
        assert_eq!(
            low_res_fences.len(),
            1,
            "Picket fence observations are not supported!"
        );
        let (low_res_freqs_hz, low_res_lambdas_m): (Vec<_>, Vec<_>) = low_res_fences
            .into_iter()
            .flat_map(|f| f.chanblocks.into_iter().map(|c| c._freq))
            .map(|f| (f, VEL_C / f))
            .unzip();

        // TODO: Ensure that the order of the sources are brightest first,
        // dimmest last.
        let num_sources_to_iono_subtract =
            num_sources_to_iono_subtract.unwrap_or(source_list.len());
        // Finding the Stokes-I-weighted `RADec` of each source.
        let source_weighted_positions = {
            let mut component_radecs = vec![];
            let mut component_stokes_is = vec![];
            let mut source_weighted_positions = Vec::with_capacity(num_sources_to_iono_subtract);
            for source in source_list.values().take(num_sources_to_iono_subtract) {
                component_radecs.clear();
                component_stokes_is.clear();
                for comp in &source.components {
                    component_radecs.push(comp.radec);
                    // TODO: Do this properly.
                    component_stokes_is.push(1.0);
                }

                source_weighted_positions.push(
                    RADec::weighted_average(&component_radecs, &component_stokes_is)
                        .expect("component RAs aren't too far apart from one another"),
                );
            }
            source_weighted_positions
        };

        if dry_run {
            info!("Dry run -- exiting now.");
            return Ok(());
        }

        let error = AtomicCell::new(false);
        let (tx_data, rx_data) = bounded(1);
        let (tx_residual, rx_residual) = bounded(1);
        let (tx_write, rx_write) = bounded(2);
        let (tx_iono_consts, rx_iono_consts) = unbounded();

        // Progress bars. Courtesy Dev.
        let multi_progress = MultiProgress::with_draw_target(if no_progress_bars {
            ProgressDrawTarget::hidden()
        } else {
            ProgressDrawTarget::stdout()
        });
        let read_progress = multi_progress.add(
            ProgressBar::new(timestamps.len() as _)
                .with_style(
                    ProgressStyle::default_bar()
                        .template("{msg:17}: [{wide_bar:.blue}] {pos:2}/{len:2} timesteps ({elapsed_precise}<{eta_precise})").unwrap()
                        .progress_chars("=> "),
                )
                .with_position(0)
                .with_message("Reading data"),
        );
        let model_progress = multi_progress.add(
            ProgressBar::new(timestamps.len() as _)
                .with_style(
                    ProgressStyle::default_bar()
                        .template("{msg:17}: [{wide_bar:.blue}] {pos:2}/{len:2} timesteps ({elapsed_precise}<{eta_precise})").unwrap()
                        .progress_chars("=> "),
                )
                .with_position(0)
                .with_message("Sky modelling"),
        );
        let overall_peel_progress = multi_progress.add(
            ProgressBar::new(timeblocks.len() as _)
                .with_style(
                    ProgressStyle::default_bar()
                        .template("{msg:17}: [{wide_bar:.blue}] {pos:2}/{len:2} timeblocks ({elapsed_precise}<{eta_precise})").unwrap()
                        .progress_chars("=> "),
                )
                .with_position(0)
                .with_message("Peeling timeblocks"),
        );
        let write_progress = multi_progress.add(
            ProgressBar::new(timeblocks.len() as _)
                .with_style(
                    ProgressStyle::default_bar()
                        .template("{msg:17}: [{wide_bar:.blue}] {pos:2}/{len:2} timeblocks ({elapsed_precise}<{eta_precise})").unwrap()
                        .progress_chars("=> "),
                )
                .with_position(0)
                .with_message("Writing visibilities"),
        );
        read_progress.tick();
        model_progress.tick();
        overall_peel_progress.tick();
        write_progress.tick();

        thread::scope(|scope| {
            let read_handle: ScopedJoinHandle<Result<(), VisReadError>> = thread::Builder::new()
                .name("read".to_string())
                .spawn_scoped(
                    scope,
                    (|| {
                        defer_on_unwind! { error.store(true); }

                        let mut vis_data_full_res = Array3::zeros((
                            1,
                            obs_context.fine_chan_freqs.len(),
                            num_unflagged_cross_baselines,
                        ));
                        let mut vis_weights_full_res = Array3::zeros(vis_data_full_res.dim());
                        for timeblock in &timeblocks {
                            // Make a new block of data to be passed along; this
                            // contains the target number of timesteps per peel cadence
                            // but is averaged to the `freq_average_factor`.
                            let mut out_data = Array3::zeros((
                                timeblock.timestamps.len(),
                                all_fine_chan_freqs_hz.len(),
                                num_unflagged_cross_baselines,
                            ));
                            let mut out_weights = Array3::zeros(out_data.dim());

                            for (i_timestep, timestep) in timeblock.range.clone().enumerate() {
                                // Should we continue?
                                if error.load() {
                                    return Ok(());
                                }

                                let result = input_data.read_crosses(
                                    vis_data_full_res.slice_mut(s![0, .., ..]),
                                    vis_weights_full_res.slice_mut(s![0, .., ..]),
                                    timestep,
                                    &tile_baseline_flags,
                                    // Do *not* pass flagged channels to the reader; we
                                    // want all of the visibilities so we can do
                                    // averaging.
                                    &HashSet::new(),
                                );
                                if let Err(e) = result {
                                    error.store(true);
                                    return Err(e);
                                }

                                // Apply flagged channels and cap flagged weights at 0.
                                vis_weights_full_res
                                    .axis_iter_mut(Axis(1))
                                    .enumerate()
                                    .for_each(|(i_chan, mut vis_weights_full_res)| {
                                        if flagged_fine_chans.contains(&i_chan) {
                                            vis_weights_full_res.mapv_inplace(|_| 0.0);
                                        } else {
                                            vis_weights_full_res.iter_mut().for_each(|w| {
                                                if *w < 0.0 {
                                                    *w = 0.0;
                                                }
                                            });
                                        }
                                    });

                                // Average the full-res data into the averaged data.
                                if freq_average_factor != 1 {
                                    vis_average(
                                        vis_data_full_res.view(),
                                        out_data.slice_mut(s![i_timestep..i_timestep + 1, .., ..]),
                                        vis_weights_full_res.view(),
                                        out_weights.slice_mut(s![
                                            i_timestep..i_timestep + 1,
                                            ..,
                                            ..
                                        ]),
                                    );
                                } else {
                                    out_data
                                        .slice_mut(s![i_timestep..i_timestep + 1, .., ..])
                                        .assign(&vis_data_full_res);
                                    out_weights
                                        .slice_mut(s![i_timestep..i_timestep + 1, .., ..])
                                        .assign(&vis_weights_full_res);
                                }

                                read_progress.inc(1);
                            }

                            tx_data.send((out_data, out_weights, timeblock)).unwrap();
                        }

                        read_progress.abandon_with_message("Finished reading input data");
                        drop(tx_data);
                        Ok(())
                    }),
                )
                .expect("OS can create threads");

            let model_handle: ScopedJoinHandle<Result<(), ModelError>> = thread::Builder::new()
                .name("model".to_string())
                .spawn_scoped(scope, || {
                    defer_on_unwind! { error.store(true); }

                    let mut modeller = new_sky_modeller(
                        matches!(modeller_info, ModellerInfo::Cpu),
                        beam.deref(),
                        &source_list,
                        &unflagged_tile_xyzs,
                        &all_fine_chan_freqs_hz,
                        &tile_baseline_flags.flagged_tiles,
                        obs_context.phase_centre,
                        array_position.longitude_rad,
                        array_position.latitude_rad,
                        dut1,
                        !no_precession,
                    )
                    .unwrap();

                    let mut vis_model = Array3::zeros((
                        time_average_factor,
                        all_fine_chan_freqs_hz.len(),
                        num_unflagged_cross_baselines,
                    ));
                    for timeblock in &timeblocks {
                        for (vis_model_slice, timestamp) in
                            vis_model.outer_iter_mut().zip(timeblock.timestamps.iter())
                        {
                            // Should we continue?
                            if error.load() {
                                return Ok(());
                            }

                            let result = modeller.model_timestep(vis_model_slice, *timestamp);
                            if let Err(e) = result {
                                error.store(true);
                                return Err(e);
                            }

                            model_progress.inc(1);
                        }

                        // We call the received data "residual", but the model needs
                        // to be subtracted for this to be a true "residual". That's
                        // happens next.
                        let (mut vis_residual, vis_weights, timeblock) = rx_data.recv().unwrap();

                        // TODO: Make this available behind a flag.
                        // // for each source in the sourcelist, rotate to it and
                        // // subtract the model
                        // for (source_name, source) in srclist.iter() {
                        //     info!("Rotating to {source_name} and subtracting its model");
                        //     modeller
                        //         .update_with_a_source(source, source.weighted_radec)
                        //         .unwrap();

                        //     vis_rotate(
                        //         vis_residual.view_mut(),
                        //         source.weighted_radec,
                        //         precessed_tile_xyzs.view(),
                        //         &mut tile_ws_from,
                        //         &mut tile_ws_to,
                        //         &lmsts,
                        //         all_fine_chan_freqs_hz,
                        //         true,
                        //     );

                        //     vis_residual
                        //         .outer_iter_mut()
                        //         .zip(timestamps.iter())
                        //         .for_each(|(mut vis_residual, epoch)| {
                        //             // model into vis_slice: (bl, ch), a 2d slice of vis_residual_tmp: (1, bl, ch)
                        //             // vis slice for modelling needs to be 2d, so we take a slice of vis_residual_tmp
                        //             let mut vis_slice = vis_residual_tmp.slice_mut(s![0, .., ..]);
                        //             modeller
                        //                 .model_timestep(vis_slice.view_mut(), *epoch)
                        //                 .unwrap();

                        //             // TODO: This is currently useful as simulate_accumulate_iono is only in
                        //             // one place. But it might be useful in Shintaro feedback when the
                        //             // source already has constants?
                        //             // // apply iono to temp model
                        //             // if source.iono_consts.0.abs() > 1e-9 || source.iono_consts.1.abs() > 1e-9 {
                        //             //     let part_uvws = calc_part_uvws(
                        //             //         unflagged_tile_xyzs.len(),
                        //             //         timestamps,
                        //             //         source_pos,
                        //             //         array_pos,
                        //             //         unflagged_tile_xyzs,
                        //             //     );
                        //             //     apply_iono(
                        //             //         vis_residual_tmp.view_mut(),
                        //             //         part_uvws,
                        //             //         source.iono_consts,
                        //             //         all_fine_chan_freqs_hz,
                        //             //     );
                        //             // }
                        //             // after apply iono, copy to residual array
                        //             vis_residual -= &vis_residual_tmp.slice(s![0, .., ..]);
                        //         });
                        // }
                        // // rotate back to original phase centre
                        // vis_rotate(
                        //     vis_residual.view_mut(),
                        //     obs_context.phase_centre,
                        //     precessed_tile_xyzs.view(),
                        //     &mut tile_ws_from,
                        //     &mut tile_ws_to,
                        //     &lmsts,
                        //     all_fine_chan_freqs_hz,
                        //     true,
                        // );

                        // Don't rotate to each source and subtract; just subtract
                        // the full model.
                        vis_residual -= &vis_model;

                        tx_residual
                            .send((vis_residual, vis_weights, timeblock))
                            .unwrap();
                    }

                    drop(tx_residual);
                    model_progress.abandon_with_message("Finished generating sky model");
                    Ok(())
                })
                .expect("OS can create threads");

            let peel_handle = thread::Builder::new()
                .name("peel".to_string())
                .spawn_scoped(scope, || {
                    defer_on_unwind! { error.store(true); }

                    for (i, (mut vis_residual, vis_weights, timeblock)) in
                        rx_residual.iter().enumerate()
                    {
                        let mut iono_consts = vec![(0.0, 0.0); num_sources_to_iono_subtract];
                        if num_sources_to_iono_subtract > 0 {
                            peel(
                                vis_residual.view_mut(),
                                vis_weights.view(),
                                timeblock,
                                &fences[0].chanblocks,
                                &source_list,
                                &mut iono_consts,
                                &source_weighted_positions,
                                num_sources_to_peel,
                                num_sources_to_iono_subtract,
                                num_sources_to_iono_subtract_in_serial,
                                &all_fine_chan_freqs_hz,
                                &low_res_freqs_hz,
                                &all_fine_chan_lambdas_m,
                                &low_res_lambdas_m,
                                obs_context,
                                array_position,
                                &unflagged_tile_xyzs,
                                &tile_baseline_flags.flagged_tiles,
                                beam.deref(),
                                dut1,
                                &modeller_info,
                                no_precession,
                                &multi_progress,
                            )
                            .unwrap();

                            if i == 0 {
                                source_list
                                    .iter()
                                    .take(10)
                                    .zip(iono_consts.iter())
                                    .for_each(|((name, src), iono_consts)| {
                                        multi_progress
                                            .println(format!(
                                                "{name}: {iono_consts:?} ({})",
                                                src.components.first().radec
                                            ))
                                            .unwrap();
                                    });
                            }
                        }

                        tx_iono_consts.send(iono_consts).unwrap();

                        for ((cross_data, cross_weights), timestamp) in vis_residual
                            .outer_iter()
                            .zip(vis_weights.outer_iter())
                            .zip(timeblock.timestamps.iter())
                        {
                            // TODO: Puke.
                            let cross_data = cross_data.to_owned().into_shared();
                            let cross_weights = cross_weights.to_owned().into_shared();
                            if vis_outputs.is_some() {
                                tx_write
                                    .send(VisTimestep {
                                        cross_data,
                                        cross_weights,
                                        autos: None,
                                        timestamp: *timestamp,
                                    })
                                    .unwrap();
                            }
                        }

                        overall_peel_progress.inc(1);
                    }
                    overall_peel_progress.abandon_with_message("Finished peeling");
                    drop(tx_write);
                    drop(tx_iono_consts);
                })
                .expect("OS can create threads");

            let write_handle = thread::Builder::new()
                .name("write".to_string())
                .spawn_scoped(scope, || {
                    defer_on_unwind! { error.store(true); }

                    let marlu_mwa_obs_context = input_data.get_metafits_context().map(|c| {
                        (
                            MwaObsContext::from_mwalib(c),
                            0..obs_context.coarse_chan_freqs.len(),
                        )
                    });

                    if let Some(vis_outputs) = vis_outputs.as_ref() {
                        let time_res = obs_context.guess_time_res();
                        let freq_res = obs_context.guess_freq_res() * freq_average_factor as f64;

                        let output_timeblocks = timesteps_to_timeblocks(
                            &obs_context.timestamps,
                            output_time_average_factor,
                            &timesteps_to_use,
                        );

                        let result = write_vis(
                            &vis_outputs,
                            array_position,
                            obs_context.phase_centre,
                            obs_context.pointing_centre,
                            &obs_context.tile_xyzs,
                            &obs_context.tile_names,
                            obs_context.obsid,
                            &obs_context.timestamps,
                            &timesteps_to_use,
                            &output_timeblocks,
                            time_res,
                            dut1,
                            freq_res,
                            &all_fine_chan_freqs_hz,
                            &tile_baseline_flags
                                .unflagged_cross_baseline_to_tile_map
                                .values()
                                .copied()
                                .sorted()
                                .collect::<Vec<_>>(),
                            &HashSet::new(),
                            time_average_factor * output_time_average_factor,
                            output_freq_average_factor,
                            marlu_mwa_obs_context.as_ref().map(|(c, r)| (c, r)),
                            rx_write,
                            &error,
                            Some(write_progress),
                        );
                        match result {
                            Ok(m) => info!("{m}"),
                            Err(e) => {
                                error.store(true);
                                return Err(e);
                            }
                        }
                    }

                    // Write out the iono consts.
                    let mut output_iono_consts: HashMap<_, _> = source_list
                        .keys()
                        .take(num_sources_to_iono_subtract)
                        .map(|name| {
                            (
                                name,
                                (
                                    Vec::with_capacity(timeblocks.len()),
                                    Vec::with_capacity(timeblocks.len()),
                                ),
                            )
                        })
                        .collect();
                    while let Ok(iono_consts) = rx_iono_consts.recv() {
                        output_iono_consts.iter_mut().zip_eq(iono_consts).for_each(
                            |((_, (output_alphas, output_betas)), iono_consts)| {
                                output_alphas.push(iono_consts.0);
                                output_betas.push(iono_consts.1);
                            },
                        );
                    }
                    let output_json_string =
                        serde_json::to_string_pretty(&output_iono_consts).unwrap();
                    for iono_output in iono_outputs {
                        let mut file = std::fs::File::create(iono_output).unwrap();
                        file.write_all(output_json_string.as_bytes()).unwrap();
                    }

                    Ok(())
                })
                .expect("OS can create threads");

            read_handle.join().unwrap().unwrap();
            model_handle.join().unwrap().unwrap();
            peel_handle.join().unwrap();
            write_handle.join().unwrap().unwrap();
        });

        Ok(())
    }
}

fn get_weights_rts(
    tile_uvs: ArrayView2<UV>,
    lambdas_m: &[f64],
    short_sigma: f64,
    weight_factor: f64,
) -> Array3<f32> {
    let (num_timesteps, num_tiles) = tile_uvs.dim();
    let num_cross_baselines = (num_tiles * (num_tiles - 1)) / 2;

    let mut weights = Array3::zeros((num_timesteps, lambdas_m.len(), num_cross_baselines));
    weights
        .outer_iter_mut()
        .into_par_iter()
        .zip_eq(tile_uvs.outer_iter())
        .for_each(|(mut weights, tile_uvs)| {
            let mut i_tile1 = 0;
            let mut i_tile2 = 0;
            let mut tile1_uv = tile_uvs[i_tile1];
            let mut tile2_uv = tile_uvs[i_tile2];
            weights.axis_iter_mut(Axis(1)).for_each(|mut weights| {
                i_tile2 += 1;
                if i_tile2 == num_tiles {
                    i_tile1 += 1;
                    i_tile2 = i_tile1 + 1;
                    tile1_uv = tile_uvs[i_tile1];
                }
                tile2_uv = tile_uvs[i_tile2];
                let uv = tile1_uv - tile2_uv;

                weights
                    .iter_mut()
                    .zip_eq(lambdas_m)
                    .for_each(|(weight, lambda_m)| {
                        let UV { u, v } = uv / *lambda_m;
                        // 1 - exp(-(u*u+v*v)/(2*sig^2))
                        let uv_sq = u * u + v * v;
                        let exp = (-uv_sq / (2.0 * short_sigma * short_sigma)).exp();
                        *weight = (weight_factor * (1.0 - exp)) as f32;
                    });
            });
        });
    weights
}

/// Average "high-res" data to "low-res" data.
fn vis_average(
    jones_from: ArrayView3<Jones<f32>>,
    mut jones_to: ArrayViewMut3<Jones<f32>>,
    weight_from: ArrayView3<f32>,
    mut weight_to: ArrayViewMut3<f32>,
) {
    let from_dims = jones_from.dim();
    let (time_axis, freq_axis, baseline_axis) = (Axis(0), Axis(1), Axis(2));
    let avg_time = jones_from.len_of(time_axis) / jones_to.len_of(time_axis);
    let avg_freq = jones_from.len_of(freq_axis) / jones_to.len_of(freq_axis);

    assert_eq!(from_dims, weight_from.dim());
    let to_dims = jones_to.dim();
    assert_eq!(
        to_dims,
        (
            (from_dims.0 as f64 / avg_time as f64).floor() as usize,
            (from_dims.1 as f64 / avg_freq as f64).floor() as usize,
            from_dims.2,
        )
    );
    assert_eq!(to_dims, weight_to.dim());

    let num_tiles =
        num_tiles_from_num_cross_correlation_baselines(jones_from.len_of(baseline_axis));
    assert_eq!(
        (num_tiles * (num_tiles - 1)) / 2,
        jones_from.len_of(baseline_axis)
    );

    // iterate along time axis in chunks of avg_time
    jones_from
        .axis_chunks_iter(time_axis, avg_time)
        .zip(weight_from.axis_chunks_iter(time_axis, avg_time))
        .zip(jones_to.outer_iter_mut())
        .zip(weight_to.outer_iter_mut())
        .for_each(
            |(((jones_chunk, weight_chunk), mut jones_to), mut weight_to)| {
                // iterate along baseline axis
                let mut i_tile1 = 0;
                let mut i_tile2 = 0;
                jones_chunk
                    .axis_iter(Axis(1))
                    .zip(weight_chunk.axis_iter(Axis(1)))
                    .zip(jones_to.outer_iter_mut())
                    .zip(weight_to.outer_iter_mut())
                    .for_each(
                        |(((jones_chunk, weight_chunk), mut jones_to), mut weight_to)| {
                            i_tile2 += 1;
                            if i_tile2 == num_tiles {
                                i_tile1 += 1;
                                i_tile2 = i_tile1 + 1;
                            }

                            jones_chunk
                                .axis_chunks_iter(Axis(1), avg_freq)
                                .zip(weight_chunk.axis_chunks_iter(Axis(1), avg_freq))
                                .zip(jones_to.iter_mut())
                                .zip(weight_to.iter_mut())
                                .for_each(
                                    |(((jones_chunk, weight_chunk), jones_to), weight_to)| {
                                        let mut jones_weighted_sum = Jones::default();
                                        let mut weight_sum = 0.0;

                                        // iterate through time chunks
                                        jones_chunk
                                            .outer_iter()
                                            .zip(weight_chunk.outer_iter())
                                            .for_each(|(jones_chunk, weights_chunk)| {
                                                jones_chunk
                                                    .iter()
                                                    .zip(weights_chunk.iter())
                                                    .for_each(|(jones, weight)| {
                                                        // Any flagged
                                                        // visibilities would
                                                        // have a weight <= 0,
                                                        // but we've already
                                                        // capped them to 0.
                                                        // This means we don't
                                                        // need to check the
                                                        // value of the weight
                                                        // when accumulating
                                                        // unflagged
                                                        // visibilities; the
                                                        // flagged ones
                                                        // contribute nothing.

                                                        let jones = Jones::<f64>::from(*jones);
                                                        let weight = *weight as f64;
                                                        jones_weighted_sum += jones * weight;
                                                        weight_sum += weight;
                                                    });
                                            });

                                        if weight_sum > 0.0 {
                                            *jones_to =
                                                Jones::from(jones_weighted_sum / weight_sum);
                                            *weight_to = weight_sum as f32;
                                        }
                                    },
                                );
                        },
                    );
            },
        );
}

fn vis_average2(
    jones_from: ArrayView3<Jones<f32>>,
    mut jones_to: ArrayViewMut3<Jones<f32>>,
    weight_from: ArrayView3<f32>,
) {
    let from_dims = jones_from.dim();
    let (time_axis, baseline_axis, freq_axis) = (Axis(0), Axis(1), Axis(2));
    let avg_time = jones_from.len_of(time_axis) / jones_to.len_of(time_axis);
    let avg_freq = jones_from.len_of(freq_axis) / jones_to.len_of(freq_axis);

    assert_eq!(from_dims, weight_from.dim());
    let to_dims = jones_to.dim();
    assert_eq!(
        to_dims,
        (
            (from_dims.0 as f64 / avg_time as f64).floor() as usize,
            (from_dims.1 as f64 / avg_freq as f64).floor() as usize,
            from_dims.2,
        )
    );

    let num_tiles =
        num_tiles_from_num_cross_correlation_baselines(jones_from.len_of(baseline_axis));
    assert_eq!(
        (num_tiles * (num_tiles - 1)) / 2,
        jones_from.len_of(baseline_axis)
    );

    // iterate along time axis in chunks of avg_time
    jones_from
        .axis_chunks_iter(time_axis, avg_time)
        .zip(weight_from.axis_chunks_iter(time_axis, avg_time))
        .zip(jones_to.outer_iter_mut())
        .for_each(|((jones_chunk, weight_chunk), mut jones_to)| {
            // iterate along baseline axis
            let mut i_tile1 = 0;
            let mut i_tile2 = 0;
            jones_chunk
                .axis_iter(Axis(1))
                .zip(weight_chunk.axis_iter(Axis(1)))
                .zip(jones_to.outer_iter_mut())
                .for_each(|((jones_chunk, weight_chunk), mut jones_to)| {
                    i_tile2 += 1;
                    if i_tile2 == num_tiles {
                        i_tile1 += 1;
                        i_tile2 = i_tile1 + 1;
                    }

                    jones_chunk
                        .axis_chunks_iter(Axis(1), avg_freq)
                        .zip(weight_chunk.axis_chunks_iter(Axis(1), avg_freq))
                        .zip(jones_to.iter_mut())
                        .for_each(|((jones_chunk, weight_chunk), jones_to)| {
                            let mut jones_weighted_sum = Jones::default();
                            let mut weight_sum = 0.0;

                            // iterate through time chunks
                            jones_chunk
                                .outer_iter()
                                .zip(weight_chunk.outer_iter())
                                .for_each(|(jones_chunk, weights_chunk)| {
                                    jones_chunk.iter().zip(weights_chunk.iter()).for_each(
                                        |(jones, weight)| {
                                            // Any flagged
                                            // visibilities would
                                            // have a weight <= 0,
                                            // but we've already
                                            // capped them to 0.
                                            // This means we don't
                                            // need to check the
                                            // value of the weight
                                            // when accumulating
                                            // unflagged
                                            // visibilities; the
                                            // flagged ones
                                            // contribute nothing.

                                            let jones = Jones::<f64>::from(*jones);
                                            let weight = *weight as f64;
                                            jones_weighted_sum += jones * weight;
                                            weight_sum += weight;
                                        },
                                    );
                                });

                            if weight_sum > 0.0 {
                                *jones_to = Jones::from(jones_weighted_sum / weight_sum);
                            }
                        });
                });
        });
}

fn weights_average(weight_from: ArrayView3<f32>, mut weight_to: ArrayViewMut3<f32>) {
    let from_dims = weight_from.dim();
    let (time_axis, freq_axis, baseline_axis) = (Axis(0), Axis(1), Axis(2));
    let avg_time = weight_from.len_of(time_axis) / weight_to.len_of(time_axis);
    let avg_freq = weight_from.len_of(freq_axis) / weight_to.len_of(freq_axis);

    assert_eq!(from_dims, weight_from.dim());
    let to_dims = weight_to.dim();
    assert_eq!(
        to_dims,
        (
            (from_dims.0 as f64 / avg_time as f64).floor() as usize,
            (from_dims.1 as f64 / avg_freq as f64).floor() as usize,
            from_dims.2
        )
    );
    assert_eq!(to_dims, weight_to.dim());

    let num_tiles =
        num_tiles_from_num_cross_correlation_baselines(weight_from.len_of(baseline_axis));
    assert_eq!(
        (num_tiles * (num_tiles - 1)) / 2,
        weight_from.len_of(baseline_axis)
    );

    // iterate along time axis in chunks of avg_time
    weight_from
        .axis_chunks_iter(time_axis, avg_time)
        .zip(weight_to.outer_iter_mut())
        .for_each(|(weight_chunk, mut weight_to)| {
            // iterate along baseline axis
            let mut i_tile1 = 0;
            let mut i_tile2 = 0;
            weight_chunk
                .axis_iter(Axis(1))
                .zip(weight_to.outer_iter_mut())
                .for_each(|(weight_chunk, mut weight_to)| {
                    i_tile2 += 1;
                    if i_tile2 == num_tiles {
                        i_tile1 += 1;
                        i_tile2 = i_tile1 + 1;
                    }

                    weight_chunk
                        .axis_chunks_iter(Axis(1), avg_freq)
                        .zip(weight_to.iter_mut())
                        .for_each(|(weight_chunk, weight_to)| {
                            let mut weight_sum = 0.0;

                            // iterate through time chunks
                            weight_chunk.outer_iter().for_each(|weights_chunk| {
                                weights_chunk.iter().for_each(|weight| {
                                    // Any flagged visibilities would have a
                                    // weight <= 0, but we've already capped
                                    // them to 0. This means we don't need to
                                    // check the value of the weight when
                                    // accumulating unflagged visibilities; the
                                    // flagged ones contribute nothing.

                                    let weight = *weight as f64;
                                    weight_sum += weight;
                                });
                            });

                            if weight_sum > 0.0 {
                                *weight_to = weight_sum as f32;
                            }
                        });
                });
        });
}

// /// Rotate the provided visibilities to the given phase centre. This function
// /// expects:
// ///
// /// 1) `tile_xyzs` to have already been precessed,
// /// 2) `tile_ws_from` to already be populated with the correct [`W`]s for where
// ///    the data is currently phased,
// /// 3) An equal number of timesteps in `jones_array`, `tile_xyzs`,
// ///    `tile_ws_from`, `tile_ws_to` and `lmsts`.
// ///
// /// After the visibilities have been "rotated", the memory of `tile_ws_from` and
// /// `tile_ws_to` is swapped. This allows this function to be called again with
// /// the same arrays and a new phase centre without new allocations.
// #[allow(clippy::too_many_arguments)]
// fn vis_rotate(
//     mut jones_array: ArrayViewMut3<Jones<f32>>,
//     phase_to: RADec,
//     tile_xyzs: ArrayView2<XyzGeodetic>,
//     tile_ws_from: &mut Array2<W>,
//     tile_ws_to: &mut Array2<W>,
//     lmsts: &[f64],
//     fine_chan_freqs: &[f64],
//     swap: bool,
// ) {
//     let num_tiles = tile_xyzs.len_of(Axis(1));
//     assert_eq!(tile_ws_from.len_of(Axis(1)), num_tiles);
//     assert_eq!(tile_ws_to.len_of(Axis(1)), num_tiles);

//     // iterate along time axis in chunks of avg_time
//     jones_array
//         .outer_iter_mut()
//         .into_par_iter()
//         .zip(tile_ws_from.outer_iter())
//         .zip(tile_ws_to.outer_iter_mut())
//         .zip(tile_xyzs.outer_iter())
//         .zip(lmsts.par_iter())
//         .for_each(
//             |((((mut jones_array, tile_ws_from), mut tile_ws_to), tile_xyzs), lmst)| {
//                 assert_eq!(tile_ws_from.len(), num_tiles);
//                 // Generate the "to" Ws.
//                 let phase_to = phase_to.to_hadec(*lmst);
//                 setup_ws(tile_ws_to.view_mut(), tile_xyzs.view(), phase_to);

//                 // iterate along baseline axis
//                 let mut i_tile1 = 0;
//                 let mut i_tile2 = 0;
//                 let mut tile1_w_from = tile_ws_from[i_tile1];
//                 let mut tile2_w_from = tile_ws_from[i_tile2];
//                 let mut tile1_w_to = tile_ws_to[i_tile1];
//                 let mut tile2_w_to = tile_ws_to[i_tile2];
//                 jones_array.outer_iter_mut().for_each(|mut jones_array| {
//                     i_tile2 += 1;
//                     if i_tile2 == num_tiles {
//                         i_tile1 += 1;
//                         i_tile2 = i_tile1 + 1;
//                         tile1_w_from = tile_ws_from[i_tile1];
//                         tile1_w_to = tile_ws_to[i_tile1];
//                     }
//                     tile2_w_from = tile_ws_from[i_tile2];
//                     tile2_w_to = tile_ws_to[i_tile2];

//                     let w_diff = (tile1_w_to - tile2_w_to) - (tile1_w_from - tile2_w_from);
//                     let arg = -TAU * w_diff / VEL_C;
//                     // iterate along frequency axis
//                     jones_array.iter_mut().zip(fine_chan_freqs.iter()).for_each(
//                         |(jones, &freq_hz)| {
//                             let rotation = Complex::cis(arg * freq_hz);
//                             *jones = Jones::<f32>::from(Jones::<f64>::from(*jones) * rotation);
//                         },
//                     );
//                 });
//             },
//         );

//     if swap {
//         // Swap the arrays, so that for the next source, the "from" Ws are our "to"
//         // Ws.
//         std::mem::swap(tile_ws_from, tile_ws_to);
//     }
// }

#[allow(clippy::too_many_arguments)]
fn vis_rotate2(
    jones_array: ArrayView3<Jones<f32>>,
    mut jones_array2: ArrayViewMut3<Jones<f32>>,
    phase_to: RADec,
    tile_xyzs: ArrayView2<XyzGeodetic>,
    tile_ws_from: ArrayView2<W>,
    mut tile_ws_to: ArrayViewMut2<W>,
    lmsts: &[f64],
    fine_chan_lambdas_m: &[f64],
) {
    let num_tiles = tile_xyzs.len_of(Axis(1));
    assert_eq!(tile_ws_from.len_of(Axis(1)), num_tiles);
    assert_eq!(tile_ws_to.len_of(Axis(1)), num_tiles);

    // iterate along time axis in chunks of avg_time
    jones_array
        .outer_iter()
        .into_par_iter()
        .zip(jones_array2.outer_iter_mut())
        .zip(tile_ws_from.outer_iter())
        .zip(tile_ws_to.outer_iter_mut())
        .zip(tile_xyzs.outer_iter())
        .zip(lmsts.par_iter())
        .for_each(
            |(
                ((((jones_array, mut jones_array2), tile_ws_from), mut tile_ws_to), tile_xyzs),
                lmst,
            )| {
                assert_eq!(tile_ws_from.len(), num_tiles);
                // Generate the "to" Ws.
                let phase_to = phase_to.to_hadec(*lmst);
                setup_ws(tile_ws_to.view_mut(), tile_xyzs.view(), phase_to);

                // iterate along baseline axis
                let mut i_tile1 = 0;
                let mut i_tile2 = 0;
                let mut tile1_w_from = tile_ws_from[i_tile1];
                let mut tile2_w_from = tile_ws_from[i_tile2];
                let mut tile1_w_to = tile_ws_to[i_tile1];
                let mut tile2_w_to = tile_ws_to[i_tile2];
                jones_array
                    .axis_iter(Axis(1))
                    .zip(jones_array2.axis_iter_mut(Axis(1)))
                    .for_each(|(jones_array, mut jones_array2)| {
                        i_tile2 += 1;
                        if i_tile2 == num_tiles {
                            i_tile1 += 1;
                            i_tile2 = i_tile1 + 1;
                            tile1_w_from = tile_ws_from[i_tile1];
                            tile1_w_to = tile_ws_to[i_tile1];
                        }
                        tile2_w_from = tile_ws_from[i_tile2];
                        tile2_w_to = tile_ws_to[i_tile2];

                        let w_diff = (tile1_w_to - tile2_w_to) - (tile1_w_from - tile2_w_from);
                        let arg = -TAU * w_diff;
                        // iterate along frequency axis
                        jones_array
                            .iter()
                            .zip(jones_array2.iter_mut())
                            .zip(fine_chan_lambdas_m.iter())
                            .for_each(|((jones, jones2), lambda_m)| {
                                let rotation = Complex::cis(arg / *lambda_m);
                                *jones2 = Jones::<f32>::from(Jones::<f64>::from(*jones) * rotation);
                            });
                    });
            },
        );
}

#[allow(clippy::too_many_arguments)]
fn vis_rotate2_serial(
    jones_array: ArrayView3<Jones<f32>>,
    mut jones_array2: ArrayViewMut3<Jones<f32>>,
    phase_to: RADec,
    tile_xyzs: ArrayView2<XyzGeodetic>,
    tile_ws_from: ArrayView2<W>,
    mut tile_ws_to: ArrayViewMut2<W>,
    lmsts: &[f64],
    fine_chan_lambdas_m: &[f64],
) {
    let num_tiles = tile_xyzs.len_of(Axis(1));
    assert_eq!(tile_ws_from.len_of(Axis(1)), num_tiles);
    assert_eq!(tile_ws_to.len_of(Axis(1)), num_tiles);

    // iterate along time axis in chunks of avg_time
    jones_array
        .outer_iter()
        .zip(jones_array2.outer_iter_mut())
        .zip(tile_ws_from.outer_iter())
        .zip(tile_ws_to.outer_iter_mut())
        .zip(tile_xyzs.outer_iter())
        .zip(lmsts.iter())
        .for_each(
            |(
                ((((jones_array, mut jones_array2), tile_ws_from), mut tile_ws_to), tile_xyzs),
                lmst,
            )| {
                assert_eq!(tile_ws_from.len(), num_tiles);
                // Generate the "to" Ws.
                let phase_to = phase_to.to_hadec(*lmst);
                setup_ws(tile_ws_to.view_mut(), tile_xyzs.view(), phase_to);

                // iterate along baseline axis
                let mut i_tile1 = 0;
                let mut i_tile2 = 0;
                let mut tile1_w_from = tile_ws_from[i_tile1];
                let mut tile2_w_from = tile_ws_from[i_tile2];
                let mut tile1_w_to = tile_ws_to[i_tile1];
                let mut tile2_w_to = tile_ws_to[i_tile2];
                jones_array
                    .axis_iter(Axis(1))
                    .zip(jones_array2.axis_iter_mut(Axis(1)))
                    .for_each(|(jones_array, mut jones_array2)| {
                        i_tile2 += 1;
                        if i_tile2 == num_tiles {
                            i_tile1 += 1;
                            i_tile2 = i_tile1 + 1;
                            tile1_w_from = tile_ws_from[i_tile1];
                            tile1_w_to = tile_ws_to[i_tile1];
                        }
                        tile2_w_from = tile_ws_from[i_tile2];
                        tile2_w_to = tile_ws_to[i_tile2];

                        let w_diff = (tile1_w_to - tile2_w_to) - (tile1_w_from - tile2_w_from);
                        let arg = -TAU * w_diff;
                        // iterate along frequency axis
                        jones_array
                            .iter()
                            .zip(jones_array2.iter_mut())
                            .zip(fine_chan_lambdas_m.iter())
                            .for_each(|((jones, jones2), lambda_m)| {
                                let rotation = Complex::cis(arg / *lambda_m);
                                *jones2 = Jones::<f32>::from(Jones::<f64>::from(*jones) * rotation);
                            });
                    });
            },
        );
}

/// Rotate the supplied visibilities according to the `λ²` constants of
/// proportionality with `exp(-2πi(αu+βv)λ²)`.
fn apply_iono(
    mut jones: ArrayViewMut3<Jones<f32>>,
    tile_uvs: ArrayView2<UV>,
    const_lm: (f64, f64),
    lambdas_m: &[f64],
) {
    let num_tiles = tile_uvs.len_of(Axis(1));

    // iterate along time axis
    jones
        .outer_iter_mut()
        .into_par_iter()
        .zip(tile_uvs.outer_iter())
        .for_each(|(mut jones, tile_uvs)| {
            // Just in case the compiler can't understand how an ndarray is laid
            // out.
            assert_eq!(tile_uvs.len(), num_tiles);

            // iterate along baseline axis
            let mut i_tile1 = 0;
            let mut i_tile2 = 0;
            jones.axis_iter_mut(Axis(1)).for_each(|mut jones| {
                i_tile2 += 1;
                if i_tile2 == num_tiles {
                    i_tile1 += 1;
                    i_tile2 = i_tile1 + 1;
                }

                let UV { u, v } = tile_uvs[i_tile1] - tile_uvs[i_tile2];
                let arg = -TAU * (u * const_lm.0 + v * const_lm.1);
                // iterate along frequency axis
                jones
                    .iter_mut()
                    .zip(lambdas_m.iter())
                    .for_each(|(jones, lambda_m)| {
                        let j = Jones::<f64>::from(*jones);
                        // The baseline UV is in units of metres, so we need to
                        // divide by λ to use it in an exponential. But we're
                        // also multiplying by λ², so just multiply by λ.
                        let rotation = Complex::cis(arg * *lambda_m);
                        *jones = Jones::from(j * rotation);
                    });
            });
        });
}

/// Rotate the supplied visibilities according to the `λ²` constants of
/// proportionality with `exp(-2πi(αu+βv)λ²)`.
fn apply_iono2(
    jones_from: ArrayView3<Jones<f32>>,
    mut jones_to: ArrayViewMut3<Jones<f32>>,
    tile_uvs: ArrayView2<UV>,
    const_lm: (f64, f64),
    lambdas_m: &[f64],
) {
    let num_tiles = tile_uvs.len_of(Axis(1));

    // iterate along time axis
    jones_from
        .outer_iter()
        .zip(jones_to.outer_iter_mut())
        .zip(tile_uvs.outer_iter())
        .for_each(|((jones_from, mut jones_to), tile_uvs)| {
            // Just in case the compiler can't understand how an ndarray is laid
            // out.
            assert_eq!(tile_uvs.len(), num_tiles);

            // iterate along baseline axis
            let mut i_tile1 = 0;
            let mut i_tile2 = 0;
            jones_from
                .axis_iter(Axis(1))
                .zip(jones_to.axis_iter_mut(Axis(1)))
                .for_each(|(jones_from, mut jones_to)| {
                    i_tile2 += 1;
                    if i_tile2 == num_tiles {
                        i_tile1 += 1;
                        i_tile2 = i_tile1 + 1;
                    }

                    let UV { u, v } = tile_uvs[i_tile1] - tile_uvs[i_tile2];
                    let arg = -TAU * (u * const_lm.0 + v * const_lm.1);
                    // iterate along frequency axis
                    jones_from
                        .iter()
                        .zip(jones_to.iter_mut())
                        .zip(lambdas_m.iter())
                        .for_each(|((jones_from, jones_to), lambda_m)| {
                            let j = Jones::<f64>::from(*jones_from);
                            // The baseline UV is in units of metres, so we need
                            // to divide by λ to use it in an exponential. But
                            // we're also multiplying by λ², so just multiply by
                            // λ.
                            let rotation = Complex::cis(arg * *lambda_m);
                            *jones_to = Jones::from(j * rotation);
                        });
                });
        });
}

fn apply_iono3(
    vis_model: ArrayView3<Jones<f32>>,
    mut vis_residual: ArrayViewMut3<Jones<f32>>,
    tile_uvs: ArrayView2<UV>,
    const_lm: (f64, f64),
    lambdas_m: &[f64],
) {
    let num_tiles = tile_uvs.len_of(Axis(1));

    // iterate along time axis
    vis_model
        .outer_iter()
        .into_par_iter()
        .zip(vis_residual.outer_iter_mut())
        .zip(tile_uvs.outer_iter())
        .for_each(|((vis_model, mut vis_residual), tile_uvs)| {
            // Just in case the compiler can't understand how an ndarray is laid
            // out.
            assert_eq!(tile_uvs.len(), num_tiles);

            // iterate along baseline axis
            let mut i_tile1 = 0;
            let mut i_tile2 = 0;
            vis_model
                .axis_iter(Axis(1))
                .zip(vis_residual.axis_iter_mut(Axis(1)))
                .for_each(|(vis_model, mut vis_residual)| {
                    i_tile2 += 1;
                    if i_tile2 == num_tiles {
                        i_tile1 += 1;
                        i_tile2 = i_tile1 + 1;
                    }

                    let UV { u, v } = tile_uvs[i_tile1] - tile_uvs[i_tile2];
                    let arg = -TAU * (u * const_lm.0 + v * const_lm.1);
                    // iterate along frequency axis
                    vis_model
                        .iter()
                        .zip(vis_residual.iter_mut())
                        .zip(lambdas_m.iter())
                        .for_each(|((vis_model, vis_residual), lambda_m)| {
                            let mut j = Jones::<f64>::from(*vis_residual);
                            let mut m = Jones::<f64>::from(*vis_model);
                            j += m;

                            // The baseline UV is in units of metres, so we need
                            // to divide by λ to use it in an exponential. But
                            // we're also multiplying by λ², so just multiply by
                            // λ.
                            let rotation = Complex::cis(arg * *lambda_m);
                            m *= rotation;
                            *vis_residual = Jones::from(j - m);
                        });
                });
        });
}

// TODO: CHJ: Ask Dev if this is useful
// // apply ionospheric rotation approximation by `order` taylor expansion terms
// fn apply_iono_approx<F>(
//     mut jones: ArrayViewMut3<Jones<F>>,
//     vis_ctx: &VisContext,
//     obs_ctx: &MarluObsContext,
//     // constants of proportionality for ionospheric offset in l,m
//     const_lm: (f64, f64),
//     order: usize,
// ) where
//     F: Float + Num + NumAssign + Default,
// {
//     let jones_dims = jones.dim();
//     let freqs_hz = vis_ctx.frequencies_hz();
//     let tile_xyzs: Vec<XyzGeodetic> = obs_ctx.ant_positions_geodetic().collect();
//     let centroid_timestamps: Vec<Epoch> = vis_ctx.timeseries(false, true).collect();
//     let phase_centre = obs_ctx.phase_centre;
//     let array_pos = obs_ctx.array_pos;
//     // let ant_pairs = vis_ctx.sel_baselines.clone();
//     let num_tiles = num_tiles_from_num_cross_correlation_baselines(jones.dim().1);

//     assert_eq!(jones_dims.0, centroid_timestamps.len());
//     // assert_eq!(jones_dims.1, ant_pairs.len());
//     assert_eq!(jones_dims.2, freqs_hz.len());

//     // pre-compute partial uvws:
//     let part_uvws = calc_part_uvws(
//         num_tiles,
//         &centroid_timestamps,
//         phase_centre,
//         array_pos,
//         &tile_xyzs,
//     );

//     let lambdas = freqs_hz
//         .iter()
//         .map(|freq_hz| VEL_C / freq_hz)
//         .collect::<Vec<_>>();

//     // iterate along time axis
//     for (mut jones, part_uvws) in jones.outer_iter_mut().zip(part_uvws.outer_iter()) {
//         let mut i_tile1 = 0;
//         let mut i_tile2 = 1;

//         // iterate along baseline axis
//         for mut jones in jones.outer_iter_mut() {
//             i_tile2 += 1;
//             if i_tile2 == num_tiles {
//                 i_tile1 += 1;
//                 i_tile2 = i_tile1 + 1;
//             }

//             let uvw = part_uvws[[i_tile1]] - part_uvws[[i_tile2]];
//             let uv_lm = uvw.u * const_lm.0 + uvw.v * const_lm.1;
//             // iterate along frequency axis
//             for (jones, &lambda) in jones.iter_mut().zip(&lambdas) {
//                 // in RTS, uvw is in units of λ but pal uvw is in meters, so divide by wavelength,
//                 // but we're also multiplying by λ², so just multiply by λ

//                 // first order taylor expansion, data D from rotation of model M
//                 // D = M * exp(-i * phi * lambda^2 )
//                 //   = M * (1 - i * phi * lambda^2 + ... )
//                 let exponent = -Complex::i() * F::from(TAU * uv_lm * lambda).unwrap();
//                 let rotation: Complex<F> = (0..=order)
//                     .map(|n| exponent.powi(n as i32) / F::from(factorial(&n)).unwrap())
//                     .sum();
//                 *jones *= rotation;
//             }
//         }
//     }
// }

// the offsets as defined by the RTS code
// TODO: Assume there's only 1 timestep, because this is low res data?
fn iono_fit(
    residual: ArrayView3<Jones<f32>>,
    weights: ArrayView3<f32>,
    model: ArrayView3<Jones<f32>>,
    lambdas_m: &[f64],
    tile_uvs_low_res: &[UV],
) -> [f64; 4] {
    let num_tiles = tile_uvs_low_res.len();

    // a-terms used in least-squares estimator
    let (mut a_uu, mut a_uv, mut a_vv) = (0.0, 0.0, 0.0);
    // A-terms used in least-squares estimator
    let (mut aa_u, mut aa_v) = (0.0, 0.0);
    // Excess amplitude in the visibilities (V) over the models (M)
    let (mut s_vm, mut s_mm) = (0.0, 0.0);

    // iterate over time
    residual
        .outer_iter()
        .zip(weights.outer_iter())
        .zip(model.outer_iter())
        .for_each(|((residual, weights), model)| {
            // iterate over frequency
            residual
                .outer_iter()
                .zip(weights.outer_iter())
                .zip(model.outer_iter())
                .zip(lambdas_m.iter())
                .for_each(|(((residual, weights), model), &lambda)| {
                    // lambda^2
                    let lambda_2 = lambda * lambda;
                    // lambda^4
                    let lambda_4 = lambda_2 * lambda_2;

                    let mut i_tile1 = 0;
                    let mut i_tile2 = 0;
                    let mut uv_tile1 = tile_uvs_low_res[i_tile1];
                    let mut uv_tile2 = tile_uvs_low_res[i_tile2];

                    let mut a_uu_bl = 0.0;
                    let mut a_uv_bl = 0.0;
                    let mut a_vv_bl = 0.0;
                    let mut aa_u_bl = 0.0;
                    let mut aa_v_bl = 0.0;
                    let mut s_vm_bl = 0.0;
                    let mut s_mm_bl = 0.0;

                    // iterate over baseline
                    residual
                        .iter()
                        .zip(weights.iter())
                        .zip(model.iter())
                        .for_each(|((residual, weight), model)| {
                            i_tile2 += 1;
                            if i_tile2 == num_tiles {
                                i_tile1 += 1;
                                i_tile2 = i_tile1 + 1;
                                uv_tile1 = tile_uvs_low_res[i_tile1];
                            }

                            if *weight > 0.0 {
                                uv_tile2 = tile_uvs_low_res[i_tile2];
                                // Divide by λ to get dimensionless UV.
                                let UV { u, v } = (uv_tile1 - uv_tile2) / lambda;

                                // Stokes I of the residual visibilities and
                                // model visibilities. It doesn't matter if the
                                // convention is to divide by 2 or not; the
                                // algorithm's result is algebraically the same.
                                let residual_i = residual[0] + residual[3];
                                let model_i = model[0] + model[3];

                                let model_i_re = model_i.re as f64;
                                let mr = model_i_re * (residual_i.im as f64 - model_i.im as f64);
                                let mm = model_i_re * model_i_re;
                                let s_vm = model_i_re * residual_i.re as f64;
                                let s_mm = mm;

                                let weight = *weight as f64;

                                // To avoid accumulating floating-point errors
                                // (and save some multiplies), multiplications
                                // with powers of lambda are done outside the
                                // loop.
                                a_uu_bl += weight * mm * u * u;
                                a_uv_bl += weight * mm * u * v;
                                a_vv_bl += weight * mm * v * v;
                                aa_u_bl += weight * mr * u;
                                aa_v_bl += weight * mr * v;
                                s_vm_bl += weight * s_vm;
                                s_mm_bl += weight * s_mm;
                            }
                        });

                    a_uu += a_uu_bl * lambda_4;
                    a_uv += a_uv_bl * lambda_4;
                    a_vv += a_vv_bl * lambda_4;
                    aa_u += aa_u_bl * -lambda_2;
                    aa_v += aa_v_bl * -lambda_2;
                    s_vm += s_vm_bl;
                    s_mm += s_mm_bl;
                });
        });

    let denom = TAU * (a_uu * a_vv - a_uv * a_uv);
    [
        (aa_u * a_vv - aa_v * a_uv) / denom,
        (aa_v * a_uu - aa_u * a_uv) / denom,
        s_vm,
        s_mm,
    ]
}

fn setup_ws(
    mut tile_ws: ArrayViewMut1<W>,
    tile_xyzs: ArrayView1<XyzGeodetic>,
    phase_centre: HADec,
) {
    let (s_ha, c_ha) = phase_centre.ha.sin_cos();
    let (s_dec, c_dec) = phase_centre.dec.sin_cos();
    tile_ws
        .iter_mut()
        .zip(tile_xyzs.iter().copied())
        .for_each(|(tile_w, tile_xyz)| {
            *tile_w = W::from_xyz(tile_xyz, s_ha, c_ha, s_dec, c_dec);
        });
}

fn setup_uvs(tile_uvs: &mut [UV], tile_xyzs: &[XyzGeodetic], phase_centre: HADec) {
    let (s_ha, c_ha) = phase_centre.ha.sin_cos();
    let (s_dec, c_dec) = phase_centre.dec.sin_cos();
    tile_uvs
        .iter_mut()
        .zip(tile_xyzs.iter().copied())
        .for_each(|(tile_uv, tile_xyz)| {
            *tile_uv = UV::from_xyz(tile_xyz, s_ha, c_ha, s_dec, c_dec);
        });
}

fn model_timesteps(
    modeller: &mut dyn SkyModeller,
    timestamps: &[Epoch],
    mut vis_result: ArrayViewMut3<Jones<f32>>,
) -> Result<(), ModelError> {
    vis_result
        .outer_iter_mut()
        .zip(timestamps.iter())
        .try_for_each(|(mut vis_result, epoch)| {
            modeller
                .model_timestep(vis_result.view_mut(), *epoch)
                .map(|_| ())
        })
}

// fn di_cal(
//     mut vis_unpeeled: ArrayViewMut3<Jones<f32>>,
//     vis_model: ArrayView3<Jones<f32>>,
//     params: &DiCalParams,
// ) {
//     let num_tiles = params.unflagged_tile_xyzs.len();
//     let shape = (1, num_tiles, params.fences.first().chanblocks.len());
//     let mut di_jones = Array3::from_elem(shape, Jones::identity());
//     let pb = ProgressBar::new(params.fences.first().chanblocks.len() as _);
//     calibrate_timeblock(
//         vis_unpeeled.view(),
//         vis_model.view(),
//         di_jones.view_mut(),
//         params.timeblocks.first(),
//         &params.fences.first().chanblocks,
//         DEFAULT_MAX_ITERATIONS,
//         DEFAULT_STOP_THRESHOLD,
//         DEFAULT_MIN_THRESHOLD,
//         pb,
//         false,
//     );

//     vis_unpeeled.outer_iter_mut().for_each(|mut vis_unpeeled| {
//         let mut i_tile1 = 0;
//         let mut i_tile2 = 0;
//         let mut tile1_sol = di_jones.slice(s![0, i_tile1, ..]);
//         let mut tile2_sol = di_jones.slice(s![0, i_tile2, ..]);
//         vis_unpeeled.outer_iter_mut().for_each(|mut vis_unpeeled| {
//             i_tile2 += 1;
//             if i_tile2 == num_tiles {
//                 i_tile1 += 1;
//                 i_tile2 = i_tile1 + 1;
//                 tile1_sol = di_jones.slice(s![0, i_tile1, ..]);
//             }
//             tile2_sol = di_jones.slice(s![0, i_tile2, ..]);

//             vis_unpeeled
//                 .iter_mut()
//                 .zip(tile1_sol.iter())
//                 .zip(tile2_sol.iter())
//                 .for_each(|((vis_unpeeled, sol_tile1), sol_tile2)| {
//                     *vis_unpeeled = Jones::<f32>::from(
//                         sol_tile1.inv() * Jones::<f64>::from(*vis_unpeeled) * sol_tile2.inv().h(),
//                     );
//                 });
//         });
//     });
// }

#[allow(clippy::too_many_arguments)]
fn peel(
    mut vis_residual: ArrayViewMut3<Jones<f32>>,
    vis_weights: ArrayView3<f32>,
    timeblock: &Timeblock,
    chanblocks: &[Chanblock],
    source_list: &SourceList,
    iono_consts: &mut [(f64, f64)],
    source_weighted_positions: &[RADec],
    num_sources_to_peel: usize,
    num_sources_to_iono_subtract: usize,
    num_sources_to_iono_subtract_in_serial: usize,
    all_fine_chan_freqs_hz: &[f64],
    low_res_freqs_hz: &[f64],
    all_fine_chan_lambdas_m: &[f64],
    low_res_lambdas_m: &[f64],
    obs_context: &ObsContext,
    array_position: LatLngHeight,
    unflagged_tile_xyzs: &[XyzGeodetic],
    flagged_tiles: &HashSet<usize>,
    beam: &dyn Beam,
    dut1: Duration,
    modeller_info: &ModellerInfo,
    no_precession: bool,
    multi_progress_bar: &MultiProgress,
) -> Result<(), PeelError> {
    let num_threads = rayon::current_num_threads();
    // let n_threads = 12;

    let timestamps = &timeblock.timestamps;
    let peel_progress = multi_progress_bar.add(
        ProgressBar::new(num_sources_to_iono_subtract as _)
            .with_style(
                ProgressStyle::default_bar()
                    .template("{msg:17}: [{wide_bar:.blue}] {pos:2}/{len:2} sources ({elapsed_precise}<{eta_precise})").unwrap()
                    .progress_chars("=> "),
            )
            .with_position(0)
            .with_message(format!("Peeling timeblock {}", timeblock.index + 1)),
    );
    peel_progress.tick();

    let num_timesteps = vis_residual.len_of(Axis(0));
    let num_tiles = unflagged_tile_xyzs.len();
    let num_cross_baselines = (num_tiles * (num_tiles - 1)) / 2;

    let precession_infos = timestamps
        .iter()
        .copied()
        .map(|time| {
            precess_time(
                array_position.longitude_rad,
                array_position.latitude_rad,
                obs_context.phase_centre,
                time,
                dut1,
            )
        })
        .collect::<Vec<_>>();
    let (lmsts, latitudes): (Vec<f64>, Vec<f64>) = precession_infos
        .iter()
        .map(|p| {
            if no_precession {
                (p.lmst, array_position.latitude_rad)
            } else {
                (p.lmst_j2000, p.array_latitude_j2000)
            }
        })
        .unzip();
    let mut precessed_tile_xyzs =
        Array2::from_elem((timestamps.len(), num_tiles), XyzGeodetic::default());
    precessed_tile_xyzs
        .outer_iter_mut()
        .zip(precession_infos.into_iter())
        .for_each(|(mut precessed_tile_xyzs, precession_info)| {
            let xyzs = precession_info.precess_xyz(unflagged_tile_xyzs);
            precessed_tile_xyzs.assign(&Array1::from_vec(xyzs));
        });

    let mut tile_uvs_high_res = Array2::default((timestamps.len(), num_tiles));
    let mut tile_uvs_low_res: Array2<UV> = Array2::default((1, num_tiles));

    // Pre-compute tile UVs.
    tile_uvs_high_res
        .outer_iter_mut()
        .zip(precessed_tile_xyzs.outer_iter())
        .zip(lmsts.iter())
        .for_each(|((mut tile_uvs, unflagged_tile_xyzs), lmst)| {
            let obs_phase_centre = obs_context.phase_centre.to_hadec(*lmst);
            setup_uvs(
                tile_uvs.as_slice_mut().unwrap(),
                unflagged_tile_xyzs.as_slice().unwrap(),
                obs_phase_centre,
            );
        });

    // use the baseline taper from the RTS, 1-exp(-(u*u+v*v)/(2*sig^2));
    let short_baseline_sigma = 20.;
    // TODO: Do we care about weights changing over time?
    let iono_taper_weights = {
        let mut iono_taper = get_weights_rts(
            tile_uvs_high_res.view(),
            all_fine_chan_lambdas_m,
            short_baseline_sigma,
            (obs_context.guess_freq_res() / FREQ_WEIGHT_FACTOR)
                * (obs_context.guess_time_res().to_seconds() / TIME_WEIGHT_FACTOR),
        );
        iono_taper *= &vis_weights;
        iono_taper
    };

    let mut tile_ws_from = Array2::default((timestamps.len(), num_tiles));
    let mut tile_ws_to = tile_ws_from.clone();
    // Set up the first set of Ws at the phase centre.
    tile_ws_from
        .outer_iter_mut()
        .into_par_iter()
        .zip(precessed_tile_xyzs.outer_iter())
        .zip(lmsts.par_iter())
        .for_each(|((mut tile_ws_from, precessed_tile_xyzs), lmst)| {
            let obs_phase_centre = obs_context.phase_centre.to_hadec(*lmst);
            setup_ws(
                tile_ws_from.view_mut(),
                precessed_tile_xyzs.view(),
                obs_phase_centre,
            );
        });

    // Temporary visibility array, re-used for each timestep
    let mut vis_residual_tmp = vis_residual.to_owned();
    let high_res_vis_dims = vis_residual.dim();

    // TODO: Do we allow multiple timesteps in the low-res data?

    let average_timestamp = average_epoch(timestamps);
    let average_precession_info = precess_time(
        array_position.longitude_rad,
        array_position.latitude_rad,
        obs_context.phase_centre,
        average_timestamp,
        dut1,
    );
    let average_lmst = if no_precession {
        average_precession_info.lmst
    } else {
        average_precession_info.lmst_j2000
    };
    let average_latitude = if no_precession {
        array_position.latitude_rad
    } else {
        average_precession_info.array_latitude_j2000
    };
    let average_precessed_tile_xyzs = Array2::from_shape_vec(
        (1, num_tiles),
        average_precession_info.precess_xyz(unflagged_tile_xyzs),
    )
    .expect("correct shape");

    // temporary arrays for accumulation
    // TODO: Do a stocktake of arrays that are lying around!
    // These are time, bl, channel
    let mut vis_residual_low_res: Array3<Jones<f32>> =
        Array3::zeros((1, low_res_freqs_hz.len(), num_cross_baselines));
    let mut vis_weights_low_res: Array3<f32> = Array3::zeros(vis_residual_low_res.dim());
    let low_res_vis_dims = vis_residual_low_res.dim();

    // The low-res weights only need to be populated once.
    weights_average(vis_weights.view(), vis_weights_low_res.view_mut());

    // let mut di_jones = if num_sources_to_peel > 0 {
    //     Array3::from_elem(
    //         (1, num_tiles, all_fine_chan_lambdas_m.len()),
    //         Jones::identity(),
    //     )
    // } else {
    //     Array3::default((0, 0, 0))
    // };

    unsafe {
        use cuda::{CudaFloat, DevicePointer};

        let cuda_xyzs: Vec<_> = precessed_tile_xyzs
            .iter()
            .copied()
            .map(|XyzGeodetic { x, y, z }| cuda::XYZ {
                x: x as _,
                y: y as _,
                z: z as _,
            })
            .collect();
        let mut cuda_uvws = Array2::from_elem(
            (num_timesteps, num_cross_baselines),
            cuda::UVW {
                u: 0.0,
                v: 0.0,
                w: 0.0,
            },
        );
        cuda_uvws
            .outer_iter_mut()
            .zip(precessed_tile_xyzs.outer_iter())
            .zip(lmsts.iter())
            .for_each(|((mut cuda_uvws, xyzs), lmst)| {
                let phase_centre = obs_context.phase_centre.to_hadec(*lmst);
                let v = xyzs_to_cross_uvws(xyzs.as_slice().unwrap(), phase_centre)
                    .into_iter()
                    .map(|uvw| cuda::UVW {
                        u: uvw.u as CudaFloat,
                        v: uvw.v as CudaFloat,
                        w: uvw.w as CudaFloat,
                    })
                    .collect::<Vec<_>>();
                cuda_uvws.assign(&ArrayView1::from(&v));
            });
        let cuda_lmsts: Vec<CudaFloat> = lmsts.iter().map(|l| *l as CudaFloat).collect();
        let cuda_lambdas: Vec<CudaFloat> = all_fine_chan_lambdas_m
            .iter()
            .map(|l| *l as CudaFloat)
            .collect();
        let cuda_xyzs_low_res: Vec<_> = average_precessed_tile_xyzs
            .iter()
            .copied()
            .map(|XyzGeodetic { x, y, z }| cuda::XYZ {
                x: x as _,
                y: y as _,
                z: z as _,
            })
            .collect();
        let cuda_low_res_lambdas: Vec<CudaFloat> =
            low_res_lambdas_m.iter().map(|l| *l as CudaFloat).collect();

        let d_xyzs = DevicePointer::copy_to_device(&cuda_xyzs).unwrap();
        let d_uvws_from = DevicePointer::copy_to_device(cuda_uvws.as_slice().unwrap()).unwrap();
        let mut d_uvws_to =
            DevicePointer::malloc(cuda_uvws.len() * std::mem::size_of::<cuda::UVW>()).unwrap();
        let d_lmsts = DevicePointer::copy_to_device(&cuda_lmsts).unwrap();
        let d_lambdas = DevicePointer::copy_to_device(&cuda_lambdas).unwrap();
        let d_xyzs_low_res = DevicePointer::copy_to_device(&cuda_xyzs_low_res).unwrap();
        let d_average_lmsts = DevicePointer::copy_to_device(&[average_lmst as CudaFloat]).unwrap();
        let mut d_uvws_low_res: DevicePointer<cuda::UVW> =
            DevicePointer::malloc(cuda_uvws.len() * std::mem::size_of::<cuda::UVW>()).unwrap();
        let d_low_res_lambdas = DevicePointer::copy_to_device(&cuda_low_res_lambdas).unwrap();
        // Make the amount of elements in `d_iono_fits` a power of 2, for
        // efficiency.
        let mut d_iono_fits = {
            let min_size =
                num_cross_baselines * low_res_freqs_hz.len() * std::mem::size_of::<Jones<f64>>();
            let n = (min_size as f64).log2().ceil() as u32;
            let size = 2_usize.pow(n);
            // dbg!(min_size, size);
            let mut d: DevicePointer<Jones<f64>> = DevicePointer::malloc(size).unwrap();
            d.clear();
            d
        };

        // let mut d_low_res_vis = DevicePointer::malloc(
        //     num_cross_baselines * low_res_freqs_hz.len() * std::mem::size_of::<Jones<CudaFloat>>(),
        // );
        // let mut d_low_res_weights = DevicePointer::malloc(
        //     num_cross_baselines * low_res_freqs_hz.len() * std::mem::size_of::<CudaFloat>(),
        // );

        let mut d_high_res_vis =
            DevicePointer::copy_to_device(vis_residual.as_slice().unwrap()).unwrap();
        let d_high_res_weights =
            DevicePointer::copy_to_device(vis_weights.as_slice().unwrap()).unwrap();

        let mut d_low_res_vis =
            DevicePointer::copy_to_device(vis_residual_low_res.as_slice().unwrap()).unwrap();
        let d_low_res_weights =
            DevicePointer::copy_to_device(vis_weights_low_res.as_slice().unwrap()).unwrap();
        let mut d_low_res_model_rotated =
            DevicePointer::copy_to_device(vis_residual_low_res.as_slice().unwrap()).unwrap();

        let freq_average_factor = all_fine_chan_freqs_hz.len() / low_res_freqs_hz.len();
        let n_serial = num_sources_to_iono_subtract;

        let mut high_res_modeller = SkyModellerCuda::new(
            beam.deref(),
            &SourceList::new(),
            unflagged_tile_xyzs,
            all_fine_chan_freqs_hz,
            flagged_tiles,
            RADec::default(),
            array_position.longitude_rad,
            array_position.latitude_rad,
            dut1,
            !no_precession,
        )?;
        let mut d_high_res_model: DevicePointer<Jones<f32>> = DevicePointer::malloc(
            timestamps.len()
                * num_cross_baselines
                * all_fine_chan_freqs_hz.len()
                * std::mem::size_of::<Jones<f32>>(),
        )
        .unwrap();

        // The UVWs for every timestep will be the same (because the phase
        // centres are always the same). Make these ahead of time for
        // efficiency.
        let high_res_uvws = {
            let mut uvws = Array2::default((timestamps.len(), num_cross_baselines));
            let mut tile_uvws = vec![UVW::default(); precessed_tile_xyzs.len_of(Axis(1))];
            uvws.outer_iter_mut()
                .zip(precessed_tile_xyzs.outer_iter())
                .zip(lmsts.iter())
                .for_each(|((mut uvws, precessed_tile_xyzs), lmst)| {
                    let phase_centre = obs_context.phase_centre.to_hadec(*lmst);
                    let (s_ha, c_ha) = phase_centre.ha.sin_cos();
                    let (s_dec, c_dec) = phase_centre.dec.sin_cos();
                    // Get a UVW for each tile.
                    tile_uvws
                        .iter_mut()
                        .zip(precessed_tile_xyzs.iter())
                        .for_each(|(uvw, xyz)| {
                            *uvw = UVW::from_xyz_inner(*xyz, s_ha, c_ha, s_dec, c_dec)
                        });
                    // Take the difference of every pair of UVWs.
                    let mut count = 0;
                    for (i, t1) in tile_uvws.iter().enumerate() {
                        for t2 in tile_uvws.iter().skip(i + 1) {
                            uvws[count] = *t1 - *t2;
                            count += 1;
                        }
                    }
                });
            uvws
        };
        // One pointer per timestep.
        let mut d_uvws = Vec::with_capacity(high_res_uvws.len_of(Axis(0)));
        // Temp vector to store results.
        let mut cuda_uvws = vec![
            cuda::UVW {
                u: 0.0,
                v: 0.0,
                w: 0.0
            };
            high_res_uvws.len_of(Axis(1))
        ];
        for uvws in high_res_uvws.outer_iter() {
            // Convert the type and push the results to the device,
            // saving the resulting pointer.
            uvws.iter()
                .zip_eq(cuda_uvws.iter_mut())
                .for_each(|(uvw, cuda_uvw)| {
                    *cuda_uvw = cuda::UVW {
                        u: uvw.u as _,
                        v: uvw.v as _,
                        w: uvw.w as _,
                    }
                });
            d_uvws.push(cuda::DevicePointer::copy_to_device(&cuda_uvws).unwrap());
        }

        let mut low_res_modeller = SkyModellerCuda::new(
            beam.deref(),
            &SourceList::new(),
            unflagged_tile_xyzs,
            low_res_freqs_hz,
            flagged_tiles,
            RADec::default(),
            array_position.longitude_rad,
            array_position.latitude_rad,
            dut1,
            !no_precession,
        )?;

        for (i_source, (((source_name, source), iono_consts), source_phase_centre)) in source_list
            .iter()
            .take(n_serial)
            .zip_eq(iono_consts.iter_mut().take(n_serial))
            .zip_eq(source_weighted_positions.iter().take(n_serial).copied())
            .enumerate()
        {
            debug!("peel loop: {source_name} at {source_phase_centre} (has iono {iono_consts:?})");
            let start = std::time::Instant::now();
            let status = cuda::rotate_average(
                d_high_res_vis.get().cast(),
                d_high_res_weights.get(),
                d_low_res_vis.get_mut().cast(),
                cuda::RADec {
                    ra: source_phase_centre.ra as _,
                    dec: source_phase_centre.dec as _,
                },
                timestamps.len().try_into().unwrap(),
                num_tiles.try_into().unwrap(),
                num_cross_baselines.try_into().unwrap(),
                all_fine_chan_freqs_hz.len().try_into().unwrap(),
                freq_average_factor.try_into().unwrap(),
                d_lmsts.get(),
                d_xyzs.get(),
                d_uvws_from.get(),
                d_uvws_to.get_mut(),
                d_lambdas.get(),
            );
            assert_eq!(status, 0);
            trace!("{:?}: rotate_average", std::time::Instant::now() - start);

            low_res_modeller.update_with_a_source(source, source_phase_centre)?;
            low_res_modeller.clear_vis();
            trace!(
                "{:?}: low res update and clear",
                std::time::Instant::now() - start
            );

            let status = cuda::xyzs_to_uvws(
                d_xyzs_low_res.get(),
                d_average_lmsts.get(),
                d_uvws_low_res.get_mut(),
                cuda::RADec {
                    ra: source_phase_centre.ra as CudaFloat,
                    dec: source_phase_centre.dec as CudaFloat,
                },
                num_tiles.try_into().unwrap(),
                num_cross_baselines.try_into().unwrap(),
                1,
            );
            assert_eq!(status, 0);
            trace!(
                "{:?}: low res xyzs_to_uvws",
                std::time::Instant::now() - start
            );

            low_res_modeller.model_with_uvws2(&d_uvws_low_res, average_lmst, average_latitude)?;
            trace!("{:?}: low res model", std::time::Instant::now() - start);

            let status = cuda::iono_loop(
                d_low_res_vis.get().cast(),
                d_low_res_weights.get(),
                low_res_modeller.d_vis.get_mut().cast(),
                d_low_res_model_rotated.get_mut().cast(),
                d_iono_fits.get_mut().cast(),
                &mut iono_consts.0,
                &mut iono_consts.1,
                num_timesteps.try_into().unwrap(),
                num_tiles.try_into().unwrap(),
                num_cross_baselines.try_into().unwrap(),
                low_res_freqs_hz.len().try_into().unwrap(),
                10,
                d_average_lmsts.get(),
                d_uvws_low_res.get(),
                d_low_res_lambdas.get(),
            );
            assert_eq!(status, 0);
            // dbg!(iono_consts);
            trace!("{:?}: iono_loop", std::time::Instant::now() - start);

            high_res_modeller.update_with_a_source(source, obs_context.phase_centre)?;
            // high_res_modeller.clear_vis();
            // Clear the old memory before reusing the buffer.
            cuda_runtime_sys::cudaMemset(
                d_high_res_model.get_mut().cast(),
                0,
                timestamps.len()
                    * num_cross_baselines
                    * all_fine_chan_freqs_hz.len()
                    * std::mem::size_of::<Jones<f32>>(),
            );
            d_uvws
                .iter()
                .zip(lmsts.iter())
                .zip(latitudes.iter())
                .enumerate()
                .try_for_each(|(i_time, ((d_uvws, lmst), latitude))| {
                    high_res_modeller.model_with_uvws3(
                        d_high_res_model
                            .get_mut()
                            .add(i_time * num_cross_baselines * all_fine_chan_freqs_hz.len()),
                        d_uvws,
                        *lmst,
                        *latitude,
                    )
                })?;
            trace!("{:?}: high res model", std::time::Instant::now() - start);

            let status = cuda::subtract_iono(
                d_high_res_vis.get_mut().cast(),
                d_high_res_model.get().cast(),
                iono_consts.0,
                iono_consts.1,
                d_uvws_from.get(),
                d_lambdas.get(),
                num_timesteps.try_into().unwrap(),
                num_cross_baselines.try_into().unwrap(),
                all_fine_chan_freqs_hz.len().try_into().unwrap(),
            );
            assert_eq!(status, 0);
            trace!("{:?}: subtract_iono", std::time::Instant::now() - start);
            debug!("peel loop finished: {source_name} at {source_phase_centre} (has iono {iono_consts:?})");

            peel_progress.inc(1);
        }

        // d_low_res_vis
        //     .copy_from_device(vis_residual_low_res_gpu.as_slice_mut().unwrap())
        //     .unwrap();
        // info!("{:?}: copy from device", std::time::Instant::now() - start);
        // dbg!(
        //     vis_residual_low_res.slice(s![0, 0, 0..3]),
        //     vis_residual_low_res_gpu.slice(s![0, 0..3, 0])
        // );
        // let mut d = 0.0_f32;
        // dbg!(vis_residual_low_res.dim(), vis_residual_low_res_gpu.dim());
        // vis_residual_low_res
        //     .outer_iter()
        //     .zip(vis_residual_low_res_gpu.outer_iter())
        //     .for_each(|(v, v2)| {
        //         v.outer_iter()
        //             .zip_eq(v2.axis_iter(Axis(1)))
        //             .for_each(|(v, v2)| {
        //                 v.iter().zip_eq(v2.iter()).for_each(|(v, v2)| {
        //                     let d2 = *v - v2;
        //                     d = d
        //                         .max(d2[0].re)
        //                         .max(d2[0].im)
        //                         .max(d2[1].re)
        //                         .max(d2[1].im)
        //                         .max(d2[2].re)
        //                         .max(d2[2].im)
        //                         .max(d2[3].re)
        //                         .max(d2[3].im);
        //                 });
        //             });
        //     });
        // dbg!(d);
    }

    // ///////// //
    // PEEL LOOP //
    // ///////// //

    // let error = AtomicCell::new(false);
    // let (tx_high_res_model, rx_high_res_model) = bounded(num_threads);
    // let (tx_low_res_model, rx_low_res_model) = bounded(2);
    // thread::scope(|scope| {
    //     let low_res_model_handle: ScopedJoinHandle<Result<(), BeamError>> = scope.spawn(|| {
    //         defer_on_unwind! { error.store(true); }

    //         let mut low_res_modeller = new_sky_modeller(
    //             matches!(modeller_info, ModellerInfo::Cpu),
    //             beam.deref(),
    //             &SourceList::new(),
    //             unflagged_tile_xyzs,
    //             low_res_freqs_hz,
    //             flagged_tiles,
    //             RADec::default(),
    //             array_position.longitude_rad,
    //             array_position.latitude_rad,
    //             dut1,
    //             !no_precession,
    //         )?;

    //         for ((source_name, source), source_phase_centre) in source_list
    //             .iter()
    //             .take(num_sources_to_iono_subtract)
    //             .zip_eq(source_weighted_positions.iter())
    //         {
    //             // Should we continue?
    //             if error.load() {
    //                 return Ok(());
    //             }

    //             trace!(
    //                 "Modelling '{source_name}' in low res, weighted position {source_phase_centre}"
    //             );
    //             let mut vis_model_low_res = Array3::zeros(low_res_vis_dims);
    //             low_res_modeller.update_with_a_source(source, *source_phase_centre)?;
    //             model_timesteps(
    //                 low_res_modeller.deref_mut(),
    //                 &[average_timestamp],
    //                 vis_model_low_res.view_mut(),
    //             )?;
    //             tx_low_res_model.send(vis_model_low_res).unwrap();
    //         }

    //         drop(tx_low_res_model);
    //         Ok(())
    //     });

    //     let high_res_model_handle: ScopedJoinHandle<Result<(), BeamError>> = scope.spawn(|| {
    //         defer_on_unwind! { error.store(true); }

    // let mut high_res_modeller = match modeller_info {
    //     ModellerInfo::Cpu => {
    //         let phase_centre = RADec::default();
    //         let components =
    //             ComponentList::new(source_list, all_fine_chan_freqs_hz, phase_centre);
    //         let maps = crate::math::TileBaselineFlags::new(
    //             unflagged_tile_xyzs.len() + flagged_tiles.len(),
    //             flagged_tiles.clone(),
    //         );

    //         let modeller = SkyModellerCpu {
    //             beam,
    //             phase_centre,
    //             array_longitude: array_position.longitude_rad,
    //             array_latitude: array_position.latitude_rad,
    //             dut1,
    //             precess: !no_precession,
    //             unflagged_fine_chan_freqs: all_fine_chan_freqs_hz,
    //             unflagged_tile_xyzs,
    //             flagged_tiles,
    //             unflagged_baseline_to_tile_map: maps.unflagged_cross_baseline_to_tile_map,
    //             components,
    //         };
    //         Either::Left(modeller)
    //     }

    //     #[cfg(feature = "cuda")]
    //     ModellerInfo::Cuda { .. } => unsafe {
    //         let modeller = SkyModellerCuda::new(
    //             beam,
    //             &SourceList::new(),
    //             unflagged_tile_xyzs,
    //             all_fine_chan_freqs_hz,
    //             flagged_tiles,
    //             RADec::default(),
    //             array_position.longitude_rad,
    //             array_position.latitude_rad,
    //             dut1,
    //             !no_precession,
    //         )?;
    //         Either::Right((modeller, vec![]))
    //     },
    // };

    //         // The UVWs for every timestep will be the same (because the phase
    //         // centres are always the same). Make these ahead of time for
    //         // efficiency.
    //         let high_res_uvws = {
    //             let mut uvws = Array2::default((timestamps.len(), num_cross_baselines));
    //             let mut tile_uvws =
    //                 vec![UVW::default(); precessed_tile_xyzs.len_of(Axis(1))];
    //             uvws.outer_iter_mut()
    //                 .zip(precessed_tile_xyzs.outer_iter())
    //                 .zip(lmsts.iter())
    //                 .for_each(|((mut uvws, precessed_tile_xyzs), lmst)| {
    //                     let phase_centre = obs_context.phase_centre.to_hadec(*lmst);
    //                     let (s_ha, c_ha) = phase_centre.ha.sin_cos();
    //                     let (s_dec, c_dec) = phase_centre.dec.sin_cos();
    //                     // Get a UVW for each tile.
    //                     tile_uvws
    //                         .iter_mut()
    //                         .zip(precessed_tile_xyzs.iter())
    //                         .for_each(|(uvw, xyz)| {
    //                             *uvw = UVW::from_xyz_inner(*xyz, s_ha, c_ha, s_dec, c_dec)
    //                         });
    //                     // Take the difference of every pair of UVWs.
    //                     let mut count = 0;
    //                     for (i, t1) in tile_uvws.iter().enumerate() {
    //                         for t2 in tile_uvws.iter().skip(i + 1) {
    //                             uvws[count] = *t1 - *t2;
    //                             count += 1;
    //                         }
    //                     }
    //                 });
    //             uvws
    //         };

    //         // If we're modelling with CUDA, we can allocate UVWs to the device
    //         // once, now, and re-use the pointers.
    //         #[cfg(feature = "cuda")]
    //         match high_res_modeller.as_mut() {
    //             Either::Right((_, d_uvws)) => {
    //                 // One pointer per timestep.
    //                 *d_uvws = Vec::with_capacity(high_res_uvws.len_of(Axis(0)));
    //                 // Temp vector to store results.
    //                 let mut cuda_uvws = vec![
    //                     cuda::UVW {
    //                         u: 0.0,
    //                         v: 0.0,
    //                         w: 0.0
    //                     };
    //                     high_res_uvws.len_of(Axis(1))
    //                 ];
    //                 for uvws in high_res_uvws.outer_iter() {
    //                     // Convert the type and push the results to the device,
    //                     // saving the resulting pointer.
    //                     uvws.iter()
    //                         .zip_eq(cuda_uvws.iter_mut())
    //                         .for_each(|(uvw, cuda_uvw)| {
    //                             *cuda_uvw = cuda::UVW {
    //                                 u: uvw.u as _,
    //                                 v: uvw.v as _,
    //                                 w: uvw.w as _,
    //                             }
    //                         });
    //                     unsafe {
    //                         d_uvws.push(cuda::DevicePointer::copy_to_device(&cuda_uvws)?);
    //                     }
    //                 }
    //             }

    //             Either::Left(_) => (),
    //         };

    //         for ((source_name, source), source_phase_centre) in source_list
    //             .iter()
    //             .take(num_sources_to_iono_subtract)
    //             .zip_eq(source_weighted_positions.iter())
    //         {
    //             // Should we continue?
    //             if error.load() {
    //                 return Ok(());
    //             }

    //             trace!("Modelling '{source_name}' in high res, weighted position {source_phase_centre}");
    //             let mut vis_model_high_res = Array3::zeros(high_res_vis_dims);
    //             match high_res_modeller.as_mut() {
    //                 Either::Left(high_res_modeller) => {
    //                     high_res_modeller.update_with_a_source(source, obs_context.phase_centre)?;
    //                     model_timesteps(
    //                         high_res_modeller,
    //                         timestamps,
    //                         vis_model_high_res.view_mut(),
    //                     )?;
    //                 }
    //                 Either::Right((high_res_cuda_modeller, d_uvws)) => {
    //                     high_res_cuda_modeller
    //                         .update_with_a_source(source, obs_context.phase_centre)?;
    //                     vis_model_high_res
    //                         .outer_iter_mut()
    //                         .zip(d_uvws.iter())
    //                         .zip(lmsts.iter())
    //                         .zip(latitudes.iter())
    //                         .try_for_each(|(((vis_model, d_uvws), lmst), latitude)| {
    //                             high_res_cuda_modeller
    //                                 .model_with_uvws(vis_model, d_uvws, *lmst, *latitude)
    //                         })?;
    //                 }
    //             }
    //             tx_high_res_model.send(vis_model_high_res).unwrap();
    //         }

    //         drop(tx_high_res_model);
    //         Ok(())
    //     });

    //     let peel_handle: ScopedJoinHandle<Result<(), BeamError>> = scope.spawn(|| {
    //         defer_on_unwind! { error.store(true); }

    //         let mut vis_model_low_res_tmp: Array3<Jones<f32>> = vis_residual_low_res.clone();
    //         // let n_serial = num_sources_to_iono_subtract_in_serial.min(num_sources_to_iono_subtract);
    //         let n_serial = num_sources_to_iono_subtract;

    //         for (i_source, ((source_name, iono_consts), source_phase_centre)) in source_list
    //             .keys()
    //             .take(n_serial)
    //             .zip_eq(iono_consts.iter_mut().take(n_serial))
    //             .zip_eq(source_weighted_positions.iter().take(n_serial).copied())
    //             .enumerate()
    //         {
    //             // Should we continue?
    //             if error.load() {
    //                 return Ok(());
    //             }

    //             let start = std::time::Instant::now();

    //             debug!(
    //                 "peel loop: {source_name} at {source_phase_centre} (has iono {iono_consts:?})",
    //             );

    //             // /////////////////// //
    //             // ROTATE, AVERAGE VIS //
    //             // /////////////////// //

    //             // Rotate the residual visibilities to the source phase centre and
    //             // average into vis_residual_low_res.
    //             trace!("{:?}: vis_rotate2", std::time::Instant::now() - start);
    //             vis_rotate2(
    //                 vis_residual.view(),
    //                 vis_residual_tmp.view_mut(),
    //                 source_phase_centre,
    //                 precessed_tile_xyzs.view(),
    //                 tile_ws_from.view(),
    //                 tile_ws_to.view_mut(),
    //                 &lmsts,
    //                 all_fine_chan_lambdas_m,
    //             );
    //             trace!("{:?}: vis_average", std::time::Instant::now() - start);
    //             vis_average2(
    //                 vis_residual_tmp.view(),
    //                 vis_residual_low_res.view_mut(),
    //                 iono_taper_weights.view(),
    //             );
    //             trace!("{:?}: add low-res model", std::time::Instant::now() - start);

    //             // ///////////// //
    //             // UNPEEL SOURCE //
    //             // ///////////// //
    //             // at lower resolution

    //             let mut vis_model_low_res = rx_low_res_model.recv().unwrap();
    //             Zip::from(&mut vis_residual_low_res)
    //                 .and(&mut vis_model_low_res_tmp)
    //                 .and(&vis_model_low_res)
    //                 .for_each(|r, t, m| {
    //                     *r += *m;
    //                     *t = *m;
    //                 });
    //             trace!(
    //                 "{:?}: pre-compute tile UVs",
    //                 std::time::Instant::now() - start
    //             );

    //             // Pre-compute tile UVs.
    //             tile_uvs_low_res
    //                 .outer_iter_mut()
    //                 .zip(average_precessed_tile_xyzs.outer_iter())
    //                 .for_each(|(mut tile_uvs, tile_xyzs)| {
    //                     let phase_centre = source_phase_centre.to_hadec(average_lmst);
    //                     setup_uvs(
    //                         tile_uvs.as_slice_mut().unwrap(),
    //                         tile_xyzs.as_slice().unwrap(),
    //                         phase_centre,
    //                     );
    //                 });
    //             trace!("{:?}: alpha/beta loop", std::time::Instant::now() - start);

    //             // ///////////////// //
    //             // CALCULATE OFFSETS //
    //             // ///////////////// //
    //             // iterate towards a convergent solution for ɑ, β
    //             let mut gain_update = 1.0;

    //             let mut iteration = 0;
    //             while iteration != 10 {
    //                 iteration += 1;
    //                 debug!("iter {iteration}, consts: {iono_consts:?}");

    //                 // iono rotate model using existing iono consts (if they're
    //                 // non-zero)
    //                 if iono_consts.0.abs() > 0.0 || iono_consts.1.abs() > 0.0 {
    //                     apply_iono2(
    //                         vis_model_low_res.view(),
    //                         vis_model_low_res_tmp.view_mut(),
    //                         tile_uvs_low_res.view(),
    //                         *iono_consts,
    //                         low_res_lambdas_m,
    //                     );
    //                 }

    //                 let iono_fits = iono_fit(
    //                     vis_residual_low_res.view(),
    //                     weight_residual_low_res.view(),
    //                     vis_model_low_res_tmp.view(),
    //                     low_res_lambdas_m,
    //                     tile_uvs_low_res.as_slice().unwrap(),
    //                 );
    //                 trace!("iono_fits: {iono_fits:?}");

    //                 iono_consts.0 += iono_fits[0];
    //                 iono_consts.1 += iono_fits[1];
    //                 gain_update *= iono_fits[2] / iono_fits[3];
    //                 vis_model_low_res
    //                     .iter_mut()
    //                     .for_each(|v| *v *= gain_update as f32);

    //                 // if the offset is small, we've converged.
    //                 if iono_fits[0].abs() < 1e-12 && iono_fits[1].abs() < 1e-12 {
    //                     debug!("iter {iteration}, consts: {iono_consts:?}, finished");
    //                     break;
    //                 }
    //             }

    //             // /////////////// //
    //             // UPDATE RESIDUAL //
    //             // /////////////// //
    //             // at higher resolution, unpeel old model, then re-peel correctly rotated model.
    //             let mut vis_model = rx_high_res_model.recv().unwrap();

    //             if i_source < num_sources_to_peel {
    //                 trace!("{:?}: apply_iono", std::time::Instant::now() - start);
    //                 apply_iono(
    //                     vis_model.view_mut(),
    //                     tile_uvs_high_res.view(),
    //                     *iono_consts,
    //                     all_fine_chan_lambdas_m,
    //                 );

    //                 trace!(
    //                     "{:?}: calibrate_timeblock",
    //                     std::time::Instant::now() - start
    //                 );
    //                 let pb = ProgressBar::with_draw_target(
    //                     Some(chanblocks.len() as _),
    //                     ProgressDrawTarget::hidden(),
    //                 )
    //                 .with_position(0)
    //                 .with_message("DI cal {}");

    //                 let results = calibrate_timeblock(
    //                     vis_residual.view(),
    //                     vis_model.view(),
    //                     di_jones.view_mut(),
    //                     timeblock,
    //                     chanblocks,
    //                     50,
    //                     1e-8,
    //                     1e-2,
    //                     pb,
    //                     false,
    //                 );
    //                 // Apply the solutions.
    //                 trace!("{:?}: apply solutions", std::time::Instant::now() - start);

    //                 vis_residual
    //                     .outer_iter_mut()
    //                     .zip(vis_model.outer_iter())
    //                     .for_each(|(mut vis_residual, vis_model)| {
    //                         let mut i_tile1 = 0;
    //                         let mut i_tile2 = 0;

    //                         vis_residual
    //                             .outer_iter_mut()
    //                             .zip(vis_model.outer_iter())
    //                             .for_each(|(mut vis_residual, vis_model)| {
    //                                 i_tile2 += 1;
    //                                 if i_tile2 == num_tiles {
    //                                     i_tile1 += 1;
    //                                     i_tile2 = i_tile1 + 1;
    //                                 }
    //                                 let sol_tile1 = di_jones.slice(s![0, i_tile1, ..]);
    //                                 let sol_tile2 = di_jones.slice(s![0, i_tile2, ..]);

    //                                 vis_residual
    //                                     .iter_mut()
    //                                     .zip(vis_model.iter())
    //                                     .zip(sol_tile1.iter())
    //                                     .zip(sol_tile2.iter())
    //                                     .for_each(
    //                                         |(
    //                                             ((vis_residual, vis_model), sol_tile1),
    //                                             sol_tile2,
    //                                         )| {
    //                                             if !sol_tile1.any_nan() && !sol_tile2.any_nan() {
    //                                                 *vis_residual -= Jones::<f32>::from(
    //                                                     *sol_tile1
    //                                                         * Jones::<f64>::from(vis_model)
    //                                                         * sol_tile2.h(),
    //                                                 );
    //                                             }
    //                                         },
    //                                     );
    //                             });
    //                     });
    //             } else {
    //                 vis_model.iter_mut().for_each(|v| *v *= gain_update as f32);

    //                 trace!("{:?}: apply_iono3", std::time::Instant::now() - start);
    //                 apply_iono3(
    //                     vis_model.view(),
    //                     vis_residual.view_mut(),
    //                     tile_uvs_high_res.view(),
    //                     *iono_consts,
    //                     all_fine_chan_lambdas_m,
    //                 );
    //             }
    //             trace!("{:?}: end source loop", std::time::Instant::now() - start);

    //             peel_progress.inc(1);
    //         }
    //         // drop(vis_residual_tmp);
    //         // drop(vis_residual_low_res);
    //         // drop(tile_ws_to);

    //         // // Parallel iono subtraction.
    //         // let mut n_skip = n_serial;

    //         // // Prepare temp arrays and modellers for each thread.
    //         // let mut vis_residual_tmps: Vec<Array3<Jones<f32>>> = (0..num_threads)
    //         //     .into_iter()
    //         //     .map(|_| Array3::zeros(vis_residual.dim()))
    //         //     .collect();
    //         // let mut vis_residual_low_ress: Vec<Array3<Jones<f32>>> = (0..num_threads)
    //         //     .into_iter()
    //         //     .map(|_| Array3::zeros(low_res_vis_dims))
    //         //     .collect();
    //         // let mut vis_model_low_ress: Vec<Array3<Jones<f32>>> = (0..num_threads)
    //         //     .into_iter()
    //         //     .map(|_| Array3::zeros(low_res_vis_dims))
    //         //     .collect();
    //         // let mut vis_model_low_res_tmps: Vec<Array3<Jones<f32>>> = (0..num_threads)
    //         //     .into_iter()
    //         //     .map(|_| Array3::zeros(low_res_vis_dims))
    //         //     .collect();
    //         // let mut tile_uvs_low_ress: Vec<Array2<UV>> = (0..num_threads)
    //         //     .into_iter()
    //         //     .map(|_| Array2::default((1, num_tiles)))
    //         //     .collect();
    //         // let mut tile_ws_tos: Vec<Array2<W>> = (0..num_threads)
    //         //     .into_iter()
    //         //     .map(|_| Array2::default((timestamps.len(), num_tiles)))
    //         //     .collect();
    //         // let mut low_res_modellers: Vec<Box<dyn SkyModeller>> = (0..num_threads)
    //         //     .into_iter()
    //         //     .map(|_| {
    //         //         new_sky_modeller(
    //         //             matches!(modeller_info, ModellerInfo::Cpu),
    //         //             beam.deref(),
    //         //             &SourceList::new(),
    //         //             unflagged_tile_xyzs,
    //         //             low_res_freqs_hz,
    //         //             flagged_tiles,
    //         //             RADec::default(),
    //         //             array_position.longitude_rad,
    //         //             array_position.latitude_rad,
    //         //             dut1,
    //         //             !no_precession,
    //         //         )
    //         //         .unwrap()
    //         //     })
    //         //     .collect();

    //         // for source_list_chunk in &source_list.iter().skip(n_skip).chunks(num_threads) {
    //         //     // TODO: clean this mess up.
    //         //     let source_list_chunk: Vec<(&String, &Source)> =
    //         //         source_list_chunk.into_iter().collect();
    //         //     source_list_chunk
    //         //         .par_iter()
    //         //         .zip_eq(
    //         //             iono_consts
    //         //                 .par_iter_mut()
    //         //                 .skip(n_skip)
    //         //                 .take(source_list_chunk.len()),
    //         //         )
    //         //         .zip_eq(
    //         //             source_weighted_positions
    //         //                 .par_iter()
    //         //                 .skip(n_skip)
    //         //                 .take(source_list_chunk.len())
    //         //                 .copied(),
    //         //         )
    //         //         .zip_eq(
    //         //             vis_residual_tmps
    //         //                 .par_iter_mut()
    //         //                 .take(source_list_chunk.len()),
    //         //         )
    //         //         .zip_eq(
    //         //             vis_residual_low_ress
    //         //                 .par_iter_mut()
    //         //                 .take(source_list_chunk.len()),
    //         //         )
    //         //         .zip_eq(
    //         //             vis_model_low_ress
    //         //                 .par_iter_mut()
    //         //                 .take(source_list_chunk.len()),
    //         //         )
    //         //         .zip_eq(
    //         //             vis_model_low_res_tmps
    //         //                 .par_iter_mut()
    //         //                 .take(source_list_chunk.len()),
    //         //         )
    //         //         .zip_eq(
    //         //             tile_uvs_low_ress
    //         //                 .par_iter_mut()
    //         //                 .take(source_list_chunk.len()),
    //         //         )
    //         //         .zip_eq(tile_ws_tos.par_iter_mut().take(source_list_chunk.len()))
    //         //         .zip_eq(
    //         //             low_res_modellers
    //         //                 .par_iter_mut()
    //         //                 .take(source_list_chunk.len()),
    //         //         )
    //         //         .for_each(
    //         //             |(
    //         //                 (
    //         //                     (
    //         //                         (
    //         //                             (
    //         //                                 (
    //         //                                     (
    //         //                                         (
    //         //                                             ((source_name, source), iono_consts),
    //         //                                             source_phase_centre,
    //         //                                         ),
    //         //                                         vis_residual_tmp,
    //         //                                     ),
    //         //                                     vis_residual_low_res,
    //         //                                 ),
    //         //                                 vis_model_low_res,
    //         //                             ),
    //         //                             vis_model_low_res_tmp,
    //         //                         ),
    //         //                         tile_uvs_low_res,
    //         //                     ),
    //         //                     tile_ws_to,
    //         //                 ),
    //         //                 low_res_modeller,
    //         //             )| {
    //         //                 low_res_modeller
    //         //                     .update_with_a_source(source, source_phase_centre)
    //         //                     .unwrap();
    //         //                 low_res_modeller
    //         //                     .model_timestep(
    //         //                         vis_model_low_res.slice_mut(s![0, .., ..]),
    //         //                         average_timestamp,
    //         //                     )
    //         //                     .unwrap();

    //         //                 // Pre-compute tile UVs.
    //         //                 tile_uvs_low_res
    //         //                     .outer_iter_mut()
    //         //                     .zip(average_precessed_tile_xyzs.outer_iter())
    //         //                     .for_each(|(mut tile_uvs, tile_xyzs)| {
    //         //                         let phase_centre = source_phase_centre.to_hadec(average_lmst);
    //         //                         setup_uvs(
    //         //                             tile_uvs.as_slice_mut().unwrap(),
    //         //                             tile_xyzs.as_slice().unwrap(),
    //         //                             phase_centre,
    //         //                         );
    //         //                     });

    //         //                 let start = std::time::Instant::now();
    //         //                 iono_sub_parallel(
    //         //                     vis_residual.view(),
    //         //                     vis_residual_low_res.view_mut(),
    //         //                     weight_residual_low_res.view(),
    //         //                     iono_taper_weights.view(),
    //         //                     vis_residual_tmp.view_mut(),
    //         //                     vis_model_low_res.view(),
    //         //                     vis_model_low_res_tmp.view_mut(),
    //         //                     iono_consts,
    //         //                     source_phase_centre,
    //         //                     precessed_tile_xyzs.view(),
    //         //                     &lmsts,
    //         //                     average_precessed_tile_xyzs.view(),
    //         //                     tile_ws_from.view(),
    //         //                     tile_ws_to.view_mut(),
    //         //                     tile_uvs_low_res.view(),
    //         //                     all_fine_chan_lambdas_m,
    //         //                     low_res_lambdas_m,
    //         //                 );
    //         //                 trace!(
    //         //                     "iono subbed {source_name} in {:?}",
    //         //                     std::time::Instant::now() - start
    //         //                 );

    //         //                 peel_progress.inc(1);
    //         //             },
    //         //         );

    //         //     // /////////////// //
    //         //     // UPDATE RESIDUAL //
    //         //     // /////////////// //
    //         //     // at higher resolution, unpeel old model, then re-peel correctly rotated model.
    //         //     for iono_consts in iono_consts
    //         //         .iter()
    //         //         .skip(n_skip)
    //         //         .take(source_list_chunk.len())
    //         //     {
    //         //         let vis_model = rx_high_res_model.recv().unwrap();
    //         //         apply_iono3(
    //         //             vis_model.view(),
    //         //             vis_residual.view_mut(),
    //         //             tile_uvs_high_res.view(),
    //         //             *iono_consts,
    //         //             all_fine_chan_lambdas_m,
    //         //         );
    //         //     }

    //         //     n_skip += source_list_chunk.len();
    //         // }

    //         Ok(())
    //     });

    //     low_res_model_handle.join().unwrap().unwrap();
    //     high_res_model_handle.join().unwrap().unwrap();
    //     peel_handle.join().unwrap().unwrap();
    // });

    Ok(())
}

/// Just the W terms of [`UVW`] coordinates.
#[derive(Clone, Copy, Default)]
struct W(f64);

impl W {
    fn from_xyz(xyz: XyzGeodetic, s_ha: f64, c_ha: f64, s_dec: f64, c_dec: f64) -> W {
        W(c_dec * c_ha * xyz.x - c_dec * s_ha * xyz.y + s_dec * xyz.z)
    }
}

impl Sub for W {
    type Output = f64;

    fn sub(self, rhs: Self) -> Self::Output {
        self.0 - rhs.0
    }
}

/// Just the U and V terms of [`UVW`] coordinates.
#[derive(Clone, Copy, Default)]
struct UV {
    u: f64,
    v: f64,
}

impl UV {
    fn from_xyz(xyz: XyzGeodetic, s_ha: f64, c_ha: f64, s_dec: f64, c_dec: f64) -> UV {
        UV {
            u: s_ha * xyz.x + c_ha * xyz.y,
            v: -s_dec * c_ha * xyz.x + s_dec * s_ha * xyz.y + c_dec * xyz.z,
        }
    }
}

impl Sub for UV {
    type Output = UV;

    fn sub(self, rhs: Self) -> Self::Output {
        UV {
            u: self.u - rhs.u,
            v: self.v - rhs.v,
        }
    }
}

impl Div<f64> for UV {
    type Output = UV;

    fn div(self, rhs: f64) -> Self::Output {
        UV {
            u: self.u / rhs,
            v: self.v / rhs,
        }
    }
}

#[derive(Error, Debug)]
pub(crate) enum PeelError {
    #[error("No input data was given!")]
    NoInputData,

    #[error("{0}\n\nSupported combinations of file formats:\n{SUPPORTED_CALIBRATED_INPUT_FILE_COMBINATIONS}")]
    InvalidDataInput(&'static str),

    #[error("Multiple metafits files were specified: {0:?}\nThis is unsupported.")]
    MultipleMetafits(Vec1<PathBuf>),

    #[error("Multiple measurement sets were specified: {0:?}\nThis is currently unsupported.")]
    MultipleMeasurementSets(Vec1<PathBuf>),

    #[error("Multiple uvfits files were specified: {0:?}\nThis is currently unsupported.")]
    MultipleUvfits(Vec1<PathBuf>),

    #[error("No calibration output was specified. There must be at least one calibration solution file.")]
    NoOutput,

    #[error("No sky-model source list file supplied")]
    NoSourceList,

    #[error("Tried to create a beam object, but MWA dipole delay information isn't available!")]
    NoDelays,

    #[error(
        "The specified MWA dipole delays aren't valid; there should be 16 values between 0 and 32"
    )]
    BadDelays,

    #[error("The data either contains no tiles or all tiles are flagged")]
    NoTiles,

    #[error("The data either contains no frequency channels or all channels are flagged")]
    NoChannels,

    #[error("The data either contains no timesteps or no timesteps are being used")]
    NoTimesteps,

    #[error("The number of specified sources was 0, or the size of the source list was 0")]
    NoSources,

    #[error("After vetoing sources, none were left. Decrease the veto threshold, or supply more sources")]
    NoSourcesAfterVeto,

    #[error("Duplicate timesteps were specified; this is invalid")]
    DuplicateTimesteps,

    #[error("Timestep {got} was specified but it isn't available; the last timestep is {last}")]
    UnavailableTimestep { got: usize, last: usize },

    #[error(
        "Cannot write visibilities to a file type '{ext}'. Supported formats are: {}", *crate::vis_io::write::VIS_OUTPUT_EXTENSIONS
    )]
    VisFileType { ext: String },

    #[error(transparent)]
    TileFlag(#[from] crate::context::InvalidTileFlag),

    #[error(transparent)]
    ParsePfbFlavour(#[from] crate::pfb_gains::PfbParseError),

    #[error("Error when parsing time average factor: {0}")]
    ParseCalTimeAverageFactor(crate::unit_parsing::UnitParseError),

    #[error("Error when parsing freq. average factor: {0}")]
    ParseCalFreqAverageFactor(crate::unit_parsing::UnitParseError),

    #[error("Calibration time average factor isn't an integer")]
    CalTimeFactorNotInteger,

    #[error("Calibration freq. average factor isn't an integer")]
    CalFreqFactorNotInteger,

    #[error("Calibration time resolution isn't a multiple of input data's: {out} seconds vs {inp} seconds")]
    CalTimeResNotMultiple { out: f64, inp: f64 },

    #[error("Calibration freq. resolution isn't a multiple of input data's: {out} Hz vs {inp} Hz")]
    CalFreqResNotMultiple { out: f64, inp: f64 },

    #[error("Calibration time average factor cannot be 0")]
    CalTimeFactorZero,

    #[error("Calibration freq. average factor cannot be 0")]
    CalFreqFactorZero,

    #[error("Error when parsing output vis. time average factor: {0}")]
    ParseOutputVisTimeAverageFactor(crate::unit_parsing::UnitParseError),

    #[error("Error when parsing output vis. freq. average factor: {0}")]
    ParseOutputVisFreqAverageFactor(crate::unit_parsing::UnitParseError),

    #[error("Output vis. time average factor isn't an integer")]
    OutputVisTimeFactorNotInteger,

    #[error("Output vis. freq. average factor isn't an integer")]
    OutputVisFreqFactorNotInteger,

    #[error("Output vis. time average factor cannot be 0")]
    OutputVisTimeAverageFactorZero,

    #[error("Output vis. freq. average factor cannot be 0")]
    OutputVisFreqAverageFactorZero,

    #[error("Output vis. time resolution isn't a multiple of input data's: {out} seconds vs {inp} seconds")]
    OutputVisTimeResNotMultiple { out: f64, inp: f64 },

    #[error("Output vis. freq. resolution isn't a multiple of input data's: {out} Hz vs {inp} Hz")]
    OutputVisFreqResNotMultiple { out: f64, inp: f64 },

    #[error("Error when parsing minimum UVW cutoff: {0}")]
    ParseUvwMin(crate::unit_parsing::UnitParseError),

    #[error("Error when parsing maximum UVW cutoff: {0}")]
    ParseUvwMax(crate::unit_parsing::UnitParseError),

    #[error("Array position specified as {pos:?}, not [<Longitude>, <Latitude>, <Height>]")]
    BadArrayPosition { pos: Vec<f64> },

    #[error(transparent)]
    Glob(#[from] crate::glob::GlobError),

    #[error(transparent)]
    VisRead(#[from] crate::vis_io::read::VisReadError),

    #[error(transparent)]
    FileWrite(#[from] crate::vis_io::write::FileWriteError),

    #[error(transparent)]
    Veto(#[from] crate::srclist::VetoError),

    #[error("Error when trying to read source list: {0}")]
    SourceList(#[from] crate::srclist::ReadSourceListError),

    #[error(transparent)]
    Beam(#[from] crate::beam::BeamError),

    #[error(transparent)]
    Model(#[from] crate::model::ModelError),

    #[error(transparent)]
    IO(#[from] std::io::Error),

    #[cfg(feature = "cuda")]
    #[error(transparent)]
    Cuda(#[from] crate::cuda::CudaError),
}

fn iono_sub_parallel(
    vis_residual: ArrayView3<Jones<f32>>,
    mut vis_residual_low_res: ArrayViewMut3<Jones<f32>>,
    weight_residual_low_res: ArrayView3<f32>,
    iono_taper_weights: ArrayView3<f32>,
    mut vis_residual_tmp: ArrayViewMut3<Jones<f32>>,
    vis_model_low_res: ArrayView3<Jones<f32>>,
    mut vis_model_low_res_tmp: ArrayViewMut3<Jones<f32>>,
    iono_consts: &mut (f64, f64),
    source_phase_centre: RADec,
    tile_xyzs: ArrayView2<XyzGeodetic>,
    lmsts: &[f64],
    average_tile_xyzs: ArrayView2<XyzGeodetic>,
    tile_ws_from: ArrayView2<W>,
    mut tile_ws_to: ArrayViewMut2<W>,
    tile_uvs_low_res: ArrayView2<UV>,
    all_fine_chan_lambdas_m: &[f64],
    low_res_lambdas_m: &[f64],
) {
    // /////////////////// //
    // ROTATE, AVERAGE VIS //
    // /////////////////// //

    // Rotate the residual visibilities to the source phase centre and
    // average into vis_residual_low_res.
    vis_rotate2_serial(
        vis_residual.view(),
        vis_residual_tmp.view_mut(),
        source_phase_centre,
        tile_xyzs,
        tile_ws_from.view(),
        tile_ws_to.view_mut(),
        lmsts,
        all_fine_chan_lambdas_m,
    );
    vis_average2(
        vis_residual_tmp.view(),
        vis_residual_low_res.view_mut(),
        iono_taper_weights,
    );

    // ///////////// //
    // UNPEEL SOURCE //
    // ///////////// //
    // at lower resolution

    Zip::from(&mut vis_residual_low_res)
        .and(&mut vis_model_low_res_tmp)
        .and(&vis_model_low_res)
        .for_each(|r, t, m| {
            *r += *m;
            *t = *m;
        });

    // ///////////////// //
    // CALCULATE OFFSETS //
    // ///////////////// //
    // iterate towards a convergent solution for ɑ, β

    let mut iteration = 0;
    while iteration != 10 {
        iteration += 1;

        // iono rotate model using existing iono consts (if they're
        // non-zero)
        if iono_consts.0.abs() > 0.0 || iono_consts.1.abs() > 0.0 {
            apply_iono2(
                vis_model_low_res.view(),
                vis_model_low_res_tmp.view_mut(),
                tile_uvs_low_res.view(),
                *iono_consts,
                low_res_lambdas_m,
            );
        }

        let offsets = iono_fit(
            vis_residual_low_res.view(),
            weight_residual_low_res.view(),
            vis_model_low_res_tmp.view(),
            low_res_lambdas_m,
            tile_uvs_low_res.as_slice().unwrap(),
        );

        // if the offset is small, we've converged.
        iono_consts.0 += offsets[0];
        iono_consts.1 += offsets[1];
        if offsets[0].abs() < 1e-12 && offsets[1].abs() < 1e-12 {
            break;
        }
    }
}
