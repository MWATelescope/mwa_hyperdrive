// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

mod error;
#[cfg(test)]
mod tests;
pub(crate) use error::PeelError;

use std::{
    collections::HashSet,
    f64::consts::TAU,
    io::Write,
    ops::{Deref, DerefMut, Div, Neg, Sub},
    path::PathBuf,
    str::FromStr,
    thread::{self, ScopedJoinHandle},
};

use clap::Parser;
use crossbeam_channel::{bounded, unbounded};
use crossbeam_utils::atomic::AtomicCell;
use hifitime::{Duration, Epoch};
use indexmap::IndexMap;
use indicatif::{MultiProgress, ProgressBar, ProgressDrawTarget, ProgressStyle};
use itertools::{izip, Itertools};
use log::{debug, info, log_enabled, trace, warn, Level::Debug};
#[cfg(feature = "cuda")]
use marlu::pos::xyz::xyzs_to_cross_uvws;
use marlu::{
    constants::{FREQ_WEIGHT_FACTOR, TIME_WEIGHT_FACTOR, VEL_C},
    math::num_tiles_from_num_cross_correlation_baselines,
    precession::{get_lmst, precess_time},
    HADec, Jones, LatLngHeight, MwaObsContext, RADec, XyzGeodetic, UVW,
};
use ndarray::{prelude::*, Zip};
use num_complex::Complex;
use num_traits::Zero;
use rayon::prelude::*;
use scopeguard::defer_on_unwind;
use serde::Serialize;
use vec1::Vec1;

use crate::{
    averaging::{
        channels_to_chanblocks, parse_freq_average_factor, parse_time_average_factor,
        timesteps_to_timeblocks, AverageFactorError, Timeblock,
    },
    beam::{create_fee_beam_object, create_no_beam_object, Beam, Delays},
    constants::DEFAULT_VETO_THRESHOLD,
    context::ObsContext,
    filenames::InputDataTypes,
    help_texts::{
        ARRAY_POSITION_HELP, DIPOLE_DELAYS_HELP, MS_DATA_COL_NAME_HELP, PFB_FLAVOUR_HELP,
        SOURCE_LIST_TYPE_HELP, VETO_THRESHOLD_HELP,
    },
    io::{
        get_single_match_from_glob,
        read::{MsReader, RawDataCorrections, UvfitsReader, VisInputType, VisRead, VisReadError},
        write::{write_vis, VisOutputType, VisTimestep, VIS_OUTPUT_EXTENSIONS},
    },
    math::{average_epoch, TileBaselineFlags},
    messages,
    model::{new_sky_modeller, ModelError, ModellerInfo, SkyModeller},
    srclist::{read::read_source_list_file, veto_sources, SourceList, SourceListType},
    unit_parsing::WAVELENGTH_FORMATS,
    HyperdriveError,
};
#[cfg(feature = "cuda")]
use crate::{
    cuda::{self, CudaError, CudaFloat, DevicePointer},
    model::SkyModellerCuda,
};

const DEFAULT_OUTPUT_PEEL_FILENAME: &str = "hyperdrive_peeled.uvfits";
const DEFAULT_OUTPUT_IONO_CONSTS: &str = "hyperdrive_iono_consts.json";
#[cfg(not(feature = "cuda"))]
const DEFAULT_NUM_PASSES: usize = 1;
#[cfg(feature = "cuda")]
const DEFAULT_NUM_PASSES: usize = 3;
const DEFAULT_TIME_AVERAGE_FACTOR: &str = "8s";
const DEFAULT_FREQ_AVERAGE_FACTOR: &str = "80kHz";
const DEFAULT_IONO_FREQ_AVERAGE_FACTOR: &str = "1.28MHz";
const DEFAULT_OUTPUT_TIME_AVERAGE_FACTOR: &str = "8s";
const DEFAULT_OUTPUT_FREQ_AVERAGE_FACTOR: &str = "80kHz";
const DEFAULT_UVW_MIN: &str = "0λ";
/// Sources that are separated by more than this value from the phase centre are
/// discarded from sky-model source lists \[degrees\].
const DEFAULT_CUTOFF_DISTANCE: f64 = 90.0;

lazy_static::lazy_static! {
    static ref DUT1: Duration = Duration::from_seconds(0.0);

    static ref VIS_OUTPUTS_HELP: String = format!("The paths to the files where the peeled visibilities are written. Supported formats: {}", *VIS_OUTPUT_EXTENSIONS);

    static ref NUM_PASSES_HELP: String = format!("The number of times to iterate over all sources per timeblock. Default: {DEFAULT_NUM_PASSES}");

    static ref TIME_AVERAGE_FACTOR_HELP: String = format!("The number of timesteps to use per timeblock *during* peeling. Also supports a target time resolution (e.g. 8s). If this is 0, then all data are averaged together. Default: {DEFAULT_TIME_AVERAGE_FACTOR}. e.g. If this variable is 4, then peeling is performed with 4 timesteps per timeblock. If the variable is instead 4s, then each timeblock contains up to 4s worth of data.");

    static ref FREQ_AVERAGE_FACTOR_HELP: String = format!("The number of fine-frequency channels to average together *before* peeling. Also supports a target time resolution (e.g. 80kHz). If this is 0, then all data is averaged together. Default: {DEFAULT_FREQ_AVERAGE_FACTOR}. e.g. If the input data is in 20kHz resolution and this variable was 2, then we average 40kHz worth of data into a chanblock before peeling. If the variable is instead 40kHz, then each chanblock contains up to 40kHz worth of data.");

    static ref IONO_FREQ_AVERAGE_FACTOR_HELP: String = format!("The number of fine-frequency channels to average together *during* peeling. Also supports a target time resolution (e.g. 1.28MHz). Cannot be 0. Default: {DEFAULT_IONO_FREQ_AVERAGE_FACTOR}. e.g. If the input data is in 40kHz resolution and this variable was 2, then we average 80kHz worth of data into a chanblock during peeling. If the variable is instead 1.28MHz, then each chanblock contains 32 fine channels.");

    static ref OUTPUT_TIME_AVERAGE_FACTOR_HELP: String = format!("The number of timeblocks to average together when writing out visibilities. Also supports a target time resolution (e.g. 8s). If this is 0, then all data are averaged together. Default: {DEFAULT_OUTPUT_TIME_AVERAGE_FACTOR}. e.g. If this variable is 4, then 8 timesteps are averaged together as a timeblock in the output visibilities.");

    static ref OUTPUT_FREQ_AVERAGE_FACTOR_HELP: String = format!("The number of fine-frequency channels to average together when writing out visibilities. Also supports a target time resolution (e.g. 80kHz). If this is 0, then all data are averaged together. Default: {DEFAULT_OUTPUT_FREQ_AVERAGE_FACTOR}. This is multiplicative with the freq average factor; e.g. If this variable is 4, and the freq average factor is 2, then 8 fine-frequency channels are averaged together as a chanblock in the output visibilities.");

    static ref UVW_MIN_HELP: String = format!("The minimum UVW length to use. This value must have a unit annotated. Allowed units: {}. Default: {DEFAULT_UVW_MIN}", *WAVELENGTH_FORMATS);

    static ref UVW_MAX_HELP: String = format!("The maximum UVW length to use. This value must have a unit annotated. Allowed units: {}. No default.", *WAVELENGTH_FORMATS);

    static ref SOURCE_DIST_CUTOFF_HELP: String =
        format!("Specifies the maximum distance from the phase centre a source can be [degrees]. Default: {DEFAULT_CUTOFF_DISTANCE}");
}

#[derive(Debug, Default, Clone, Copy)]
struct IonoConsts {
    alpha: f64,
    beta: f64,
    s_vm: f64,
    s_mm: f64,
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

    #[clap(long, help = MS_DATA_COL_NAME_HELP, help_heading = "INPUT FILES")]
    pub ms_data_column_name: Option<String>,

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

    #[clap(long, help = NUM_PASSES_HELP.as_str(), help_heading = "PEELING")]
    pub num_passes: Option<usize>,

    /// Use the CPU for peeling and ionospheric subraction. This is deliberately
    /// made non-default because using a GPU is much faster.
    #[cfg(feature = "cuda")]
    #[clap(long, help_heading = "PEELING")]
    pub cpu_peel: bool,

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

    /// Use the CPU for visibility generation. This is deliberately made
    /// non-default because using a GPU is much faster.
    #[cfg(feature = "cuda")]
    #[clap(long, help_heading = "MODELLING")]
    pub cpu_vis: bool,

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
            ms_data_column_name,
            ignore_dut1,
            outputs,
            num_sources_to_peel: _,
            num_sources_to_iono_subtract,
            num_passes,
            #[cfg(feature = "cuda")]
            cpu_peel,
            num_sources_to_subtract,
            source_dist_cutoff,
            veto_threshold,
            #[cfg(feature = "cuda")]
            cpu_vis,
            beam_file,
            unity_dipole_gains,
            delays,
            no_beam,
            time_average_factor,
            freq_average_factor,
            iono_freq_average_factor,
            output_time_average_factor,
            output_freq_average_factor,
            // todo (dev): this could be match!(timesteps, None)
            timesteps,
            use_all_timesteps,
            uvw_min: _,
            uvw_max: _,
            array_position,
            no_precession,
            tile_flags,
            ignore_input_data_tile_flags,
            ignore_input_data_fine_channel_flags,
            fine_chan_flags_per_coarse_chan,
            fine_chan_flags,
            pfb_flavour: _,
            no_digital_gains: _,
            no_cable_length_correction: _,
            no_geometric_correction: _,
            no_progress_bars,
        } = self;

        // Check that the number of sources to peel, iono subtract and subtract
        // are sensible. When that's done, veto up to the maximum number of
        // sources to subtract.
        let _max_num_sources = match (num_sources_to_iono_subtract, num_sources_to_subtract) {
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

        let num_passes = num_passes.unwrap_or(DEFAULT_NUM_PASSES);

        // If we're going to use a GPU for modelling, get the device info so we
        // can ensure a CUDA-capable device is available, and so we can report
        // it to the user later.
        #[cfg(feature = "cuda")]
        let modeller_info = if cpu_vis {
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

        #[cfg(feature = "cuda")]
        if let (false, true) = (cpu_peel, cpu_vis) {
            return Err(PeelError::ModellerMismatch);
        }

        // Handle input data. We expect one of three possibilities:
        // - gpubox files, a metafits file (and maybe mwaf files),
        // - a measurement set (and maybe a metafits file), or
        // - uvfits files.
        // If none or multiple of these possibilities are met, then we must fail.
        let input_data_types = match data {
            Some(strings) => InputDataTypes::new(&strings)?,
            None => return Err(PeelError::NoInputData),
        };
        let (input_data, _raw_data_corrections): (Box<dyn VisRead>, Option<RawDataCorrections>) =
            match (
                input_data_types.metafits,
                input_data_types.gpuboxes,
                input_data_types.mwafs,
                input_data_types.ms,
                input_data_types.uvfits,
            ) {
                // Valid input for reading raw data.
                (Some(_meta), Some(_gpuboxes), _mwafs, None, None) => {
                    todo!("need user to supply calibration solutions");

                    // // Ensure that there's only one metafits.
                    // let meta = if meta.len() > 1 {
                    //     return Err(PeelError::MultipleMetafits(meta));
                    // } else {
                    //     meta.first()
                    // };

                    // debug!("gpubox files: {:?}", &gpuboxes);
                    // debug!("mwaf files: {:?}", &mwafs);

                    // let corrections = RawDataCorrections::new(
                    //     pfb_flavour.as_deref(),
                    //     !no_digital_gains,
                    //     !no_cable_length_correction,
                    //     !no_geometric_correction,
                    // )?;
                    // let input_data =
                    //     RawDataReader::new(meta, &gpuboxes, mwafs.as_deref(), corrections)?;

                    // messages::InputFileDetails::Raw {
                    //     obsid: input_data.mwalib_context.metafits_context.obs_id,
                    //     gpubox_count: gpuboxes.len(),
                    //     metafits_file_name: meta.display().to_string(),
                    //     mwaf: input_data.get_flags(),
                    //     raw_data_corrections: corrections,
                    // }
                    // .print("Peeling");

                    // (Box::new(input_data), Some(corrections))
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

                    let input_data = MsReader::new(&ms, ms_data_column_name, meta, array_position)?;

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

                    let input_data = UvfitsReader::new(&uvfits, meta, array_position)?;

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
                                crate::io::write::can_write_to_file(&file).unwrap();
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
        //     for (uvw, baseline_weight) in uvws.into_iter().zip_eq(baseline_weights.iter_mut()) {
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
            ProgressBar::with_draw_target(Some(timeblocks.len() as _), if vis_outputs.is_some() && !multi_progress.is_hidden() {
                ProgressDrawTarget::stdout()
            } else {
                ProgressDrawTarget::hidden()
            })
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
                .spawn_scoped(scope, || {
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
                                    out_weights.slice_mut(s![i_timestep..i_timestep + 1, .., ..]),
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
                })
                .expect("OS can create threads");

            let model_handle: ScopedJoinHandle<Result<(), ModelError>> = thread::Builder::new()
                .name("model".to_string())
                .spawn_scoped(scope, || {
                    defer_on_unwind! { error.store(true); }

                    let mut modeller = new_sky_modeller(
                        #[cfg(feature = "cuda")]
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
                        // The number of timestamps in a timeblock can vary;
                        // don't use zip_eq.
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

                        // Don't rotate to each source and subtract; just subtract
                        // the full model.
                        for (mut vis_residual_slice, vis_model_slice) in
                            vis_residual.outer_iter_mut().zip(vis_model.outer_iter())
                        {
                            vis_residual_slice -= &vis_model_slice;
                        }

                        tx_residual
                            .send((vis_residual, vis_weights, timeblock))
                            .unwrap();
                    }

                    drop(tx_residual);
                    model_progress.abandon_with_message("Finished generating sky model");
                    Ok(())
                })
                .expect("OS can create threads");

            let peel_handle: ScopedJoinHandle<Result<(), PeelError>> = thread::Builder::new()
                .name("peel".to_string())
                .spawn_scoped(scope, || {
                    defer_on_unwind! { error.store(true); }

                    for (i, (mut vis_residual, vis_weights, timeblock)) in
                        rx_residual.iter().enumerate()
                    {
                        // Should we continue?
                        if error.load() {
                            return Ok(());
                        }

                        let mut iono_consts =
                            vec![IonoConsts::default(); num_sources_to_iono_subtract];
                        if num_sources_to_iono_subtract > 0 {
                            #[cfg(feature = "cuda")]
                            if cpu_peel {
                                let mut low_res_modeller = new_sky_modeller(
                                    #[cfg(feature = "cuda")]
                                    cpu_vis,
                                    beam.deref(),
                                    &SourceList::new(),
                                    &unflagged_tile_xyzs,
                                    &low_res_freqs_hz,
                                    flagged_tiles,
                                    RADec::default(),
                                    array_position.longitude_rad,
                                    array_position.latitude_rad,
                                    dut1,
                                    !no_precession,
                                )?;
                                let mut high_res_modeller = new_sky_modeller(
                                    #[cfg(feature = "cuda")]
                                    cpu_vis,
                                    beam.deref(),
                                    &SourceList::new(),
                                    &unflagged_tile_xyzs,
                                    &all_fine_chan_freqs_hz,
                                    flagged_tiles,
                                    RADec::default(),
                                    array_position.longitude_rad,
                                    array_position.latitude_rad,
                                    dut1,
                                    !no_precession,
                                )?;
                                peel_cpu(
                                    vis_residual.view_mut(),
                                    vis_weights.view(),
                                    timeblock,
                                    &source_list,
                                    &mut iono_consts,
                                    &source_weighted_positions,
                                    num_passes,
                                    &low_res_freqs_hz,
                                    &all_fine_chan_lambdas_m,
                                    &low_res_lambdas_m,
                                    obs_context,
                                    array_position,
                                    &unflagged_tile_xyzs,
                                    low_res_modeller.deref_mut(),
                                    high_res_modeller.deref_mut(),
                                    dut1,
                                    no_precession,
                                    &multi_progress,
                                )
                                .unwrap();
                            } else {
                                // TODO (Dev): forgive me Ferris for I have sinned.
                                // {high|low} could be instantiated once if we could figure out a
                                // way to downcast to SkyModellerCuda from SkyModeller, but it looks
                                // like this is impoissible because of lifetime stuff :(
                                let mut low_res_modeller = SkyModellerCuda::new(
                                    beam.deref(),
                                    &SourceList::new(),
                                    &unflagged_tile_xyzs,
                                    &low_res_freqs_hz,
                                    flagged_tiles,
                                    RADec::default(),
                                    array_position.longitude_rad,
                                    array_position.latitude_rad,
                                    dut1,
                                    !no_precession,
                                )?;
                                let mut high_res_modeller = SkyModellerCuda::new(
                                    beam.deref(),
                                    &SourceList::new(),
                                    &unflagged_tile_xyzs,
                                    &all_fine_chan_freqs_hz,
                                    flagged_tiles,
                                    RADec::default(),
                                    array_position.longitude_rad,
                                    array_position.latitude_rad,
                                    dut1,
                                    !no_precession,
                                )?;
                                peel_cuda(
                                    vis_residual.view_mut(),
                                    vis_weights.view(),
                                    timeblock,
                                    &source_list,
                                    &mut iono_consts,
                                    &source_weighted_positions,
                                    num_passes,
                                    &low_res_freqs_hz,
                                    &all_fine_chan_lambdas_m,
                                    &low_res_lambdas_m,
                                    obs_context,
                                    array_position,
                                    &unflagged_tile_xyzs,
                                    &mut low_res_modeller,
                                    &mut high_res_modeller,
                                    dut1,
                                    no_precession,
                                    &multi_progress,
                                )
                                .unwrap();
                            }
                            #[cfg(not(feature = "cuda"))]
                            {
                                let mut low_res_modeller = new_sky_modeller(
                                    #[cfg(feature = "cuda")]
                                    cpu_vis,
                                    beam.deref(),
                                    &SourceList::new(),
                                    &unflagged_tile_xyzs,
                                    &low_res_freqs_hz,
                                    flagged_tiles,
                                    RADec::default(),
                                    array_position.longitude_rad,
                                    array_position.latitude_rad,
                                    dut1,
                                    !no_precession,
                                )?;
                                let mut high_res_modeller = new_sky_modeller(
                                    #[cfg(feature = "cuda")]
                                    cpu_vis,
                                    beam.deref(),
                                    &SourceList::new(),
                                    &unflagged_tile_xyzs,
                                    &all_fine_chan_freqs_hz,
                                    flagged_tiles,
                                    RADec::default(),
                                    array_position.longitude_rad,
                                    array_position.latitude_rad,
                                    dut1,
                                    !no_precession,
                                )?;
                                peel_cpu(
                                    vis_residual.view_mut(),
                                    vis_weights.view(),
                                    timeblock,
                                    &source_list,
                                    &mut iono_consts,
                                    &source_weighted_positions,
                                    num_passes,
                                    &low_res_freqs_hz,
                                    &all_fine_chan_lambdas_m,
                                    &low_res_lambdas_m,
                                    obs_context,
                                    array_position,
                                    &unflagged_tile_xyzs,
                                    low_res_modeller.deref_mut(),
                                    high_res_modeller.deref_mut(),
                                    dut1,
                                    no_precession,
                                    &multi_progress,
                                )
                                .unwrap();
                            }

                            // dev: what's with this?
                            if i == 0 {
                                source_list
                                    .iter()
                                    .take(10)
                                    .zip(iono_consts.iter())
                                    .for_each(|((name, src), iono_consts)| {
                                        multi_progress
                                            .println(format!(
                                                "{name}: {:e} {:e} ({})",
                                                iono_consts.alpha,
                                                iono_consts.beta,
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

                    Ok(())
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
                            vis_outputs,
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

                    if !iono_outputs.is_empty() {
                        // Write out the iono consts. First, allocate a space
                        // for all the results. We use an IndexMap to keep the
                        // order of the sources preserved while also being able
                        // to write out a "HashMap-style" json.
                        #[derive(Debug, Serialize)]
                        struct SourceIonoConsts {
                            alphas: Vec<f64>,
                            betas: Vec<f64>,
                            weighted_catalogue_pos_j2000: RADec,
                        }

                        let mut output_iono_consts: IndexMap<&str, SourceIonoConsts> = source_list
                            .iter()
                            .take(num_sources_to_iono_subtract)
                            .zip_eq(source_weighted_positions.iter().copied())
                            .map(|((name, _src), weighted_pos)| {
                                (
                                    name.as_str(),
                                    SourceIonoConsts {
                                        alphas: Vec::with_capacity(timeblocks.len()),
                                        betas: Vec::with_capacity(timeblocks.len()),
                                        weighted_catalogue_pos_j2000: weighted_pos,
                                    },
                                )
                            })
                            .collect();

                        // Store the results as they are received on the
                        // channel.
                        while let Ok(incoming_iono_consts) = rx_iono_consts.recv() {
                            incoming_iono_consts
                                .into_iter()
                                .zip_eq(output_iono_consts.iter_mut())
                                .for_each(
                                    |(
                                        IonoConsts {
                                            alpha,
                                            beta,
                                            s_vm: _,
                                            s_mm: _,
                                        },
                                        (_src_name, src_iono_consts),
                                    )| {
                                        src_iono_consts.alphas.push(alpha);
                                        src_iono_consts.betas.push(beta);
                                    },
                                );
                        }

                        // The channel has stopped sending results; write them
                        // out to a file.
                        let output_json_string =
                            serde_json::to_string_pretty(&output_iono_consts).unwrap();
                        for iono_output in iono_outputs {
                            let mut file = std::fs::File::create(iono_output)?;
                            file.write_all(output_json_string.as_bytes())?;
                        }
                    }

                    Ok(())
                })
                .expect("OS can create threads");

            read_handle.join().unwrap().unwrap();
            model_handle.join().unwrap().unwrap();
            peel_handle.join().unwrap().unwrap();
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

/// Average "high-res" vis and weights to "low-res" vis and weights
/// arguments are all 3D arrays with axes (time, freq, baseline)
// TODO (Dev): rename to vis_weight_average_tfb
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
        .zip_eq(weight_from.axis_chunks_iter(time_axis, avg_time))
        .zip_eq(jones_to.outer_iter_mut())
        .zip_eq(weight_to.outer_iter_mut())
        .for_each(
            |(((jones_chunk, weight_chunk), mut jones_to), mut weight_to)| {
                // iterate along baseline axis
                let mut i_tile1 = 0;
                let mut i_tile2 = 0;
                jones_chunk
                    .axis_iter(Axis(2))
                    .zip_eq(weight_chunk.axis_iter(Axis(2)))
                    .zip_eq(jones_to.axis_iter_mut(Axis(1)))
                    .zip_eq(weight_to.axis_iter_mut(Axis(1)))
                    .for_each(
                        |(((jones_chunk, weight_chunk), mut jones_to), mut weight_to)| {
                            i_tile2 += 1;
                            if i_tile2 == num_tiles {
                                i_tile1 += 1;
                                i_tile2 = i_tile1 + 1;
                            }

                            jones_chunk
                                .axis_chunks_iter(Axis(1), avg_freq)
                                .zip_eq(weight_chunk.axis_chunks_iter(Axis(1), avg_freq))
                                .zip_eq(jones_to.iter_mut())
                                .zip_eq(weight_to.iter_mut())
                                .for_each(
                                    |(((jones_chunk, weight_chunk), jones_to), weight_to)| {
                                        let mut jones_weighted_sum = Jones::default();
                                        let mut weight_sum = 0.0;

                                        // iterate through time chunks
                                        jones_chunk
                                            .outer_iter()
                                            .zip_eq(weight_chunk.outer_iter())
                                            .for_each(|(jones_chunk, weights_chunk)| {
                                                jones_chunk
                                                    .iter()
                                                    .zip_eq(weights_chunk.iter())
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

// TODO (dev): a.div_ceil(b) would be better, but it's nightly:
// https://doc.rust-lang.org/std/primitive.i32.html#method.div_ceil
pub(super) fn div_ceil(a: usize, b: usize) -> usize {
    (a + b - 1) / b
}

/// Like `vis_weight_average_tfb`, but for when we don't need to keep the low-res weights
/// Average "high-res" vis and weights to "low-res" vis (no low-res weights)
/// arguments are all 3D arrays with axes (time, freq, baseline).
/// assumes weights are capped to 0
// TODO (Dev): rename to vis_average_tfb
fn vis_average2(
    jones_from_tfb: ArrayView3<Jones<f32>>,
    mut jones_to_tfb: ArrayViewMut3<Jones<f32>>,
    weight_tfb: ArrayView3<f32>,
) {
    let from_dims = jones_from_tfb.dim();
    let (time_axis, freq_axis, baseline_axis) = (Axis(0), Axis(1), Axis(2));
    let avg_time = div_ceil(
        jones_from_tfb.len_of(time_axis),
        jones_to_tfb.len_of(time_axis),
    );
    let avg_freq = div_ceil(
        jones_from_tfb.len_of(freq_axis),
        jones_to_tfb.len_of(freq_axis),
    );

    assert_eq!(from_dims, weight_tfb.dim());
    let to_dims = jones_to_tfb.dim();
    assert_eq!(
        to_dims,
        (
            div_ceil(from_dims.0, avg_time),
            div_ceil(from_dims.1, avg_freq),
            from_dims.2,
        )
    );

    // iterate along time axis in chunks of avg_time
    for (jones_chunk_tfb, weight_chunk_tfb, mut jones_to_fb) in izip!(
        jones_from_tfb.axis_chunks_iter(time_axis, avg_time),
        weight_tfb.axis_chunks_iter(time_axis, avg_time),
        jones_to_tfb.outer_iter_mut()
    ) {
        for (jones_chunk_tfb, weight_chunk_tfb, mut jones_to_b) in izip!(
            jones_chunk_tfb.axis_chunks_iter(freq_axis, avg_freq),
            weight_chunk_tfb.axis_chunks_iter(freq_axis, avg_freq),
            jones_to_fb.outer_iter_mut()
        ) {
            // iterate along baseline axis
            for (jones_chunk_tf, weight_chunk_tf, jones_to) in izip!(
                jones_chunk_tfb.axis_iter(baseline_axis),
                weight_chunk_tfb.axis_iter(baseline_axis),
                jones_to_b.iter_mut()
            ) {
                let mut jones_weighted_sum = Jones::zero();
                let mut weight_sum: f64 = 0.0;
                for (&jones, &weight) in jones_chunk_tf.iter().zip_eq(weight_chunk_tf.iter()) {
                    // assumes weights are capped to 0. otherwise we would need to check weight >= 0
                    debug_assert!(weight >= 0.0, "weight was not capped to zero: {weight}");
                    jones_weighted_sum += Jones::<f64>::from(jones) * weight as f64;
                    weight_sum += weight as f64;
                }

                if weight_sum > 0.0 {
                    *jones_to = Jones::from(jones_weighted_sum / weight_sum);
                }
            }
        }
    }
}

fn weights_average(weight_tfb: ArrayView3<f32>, mut weight_avg_tfb: ArrayViewMut3<f32>) {
    let from_dims = weight_tfb.dim();
    let (time_axis, freq_axis, baseline_axis) = (Axis(0), Axis(1), Axis(2));
    let avg_time = div_ceil(
        weight_tfb.len_of(time_axis),
        weight_avg_tfb.len_of(time_axis),
    );
    let avg_freq = div_ceil(
        weight_tfb.len_of(freq_axis),
        weight_avg_tfb.len_of(freq_axis),
    );

    let to_dims = weight_avg_tfb.dim();
    assert_eq!(
        to_dims,
        (
            div_ceil(from_dims.0, avg_time),
            div_ceil(from_dims.1, avg_freq),
            from_dims.2,
        )
    );

    // iterate along time axis in chunks of avg_time
    for (weight_chunk_tfb, mut weight_avg_fb) in izip!(
        weight_tfb.axis_chunks_iter(time_axis, avg_time),
        weight_avg_tfb.outer_iter_mut()
    ) {
        // iterate along frequency axis in chunks of avg_freq
        for (weight_chunk_tfb, mut weight_avg_b) in izip!(
            weight_chunk_tfb.axis_chunks_iter(freq_axis, avg_freq),
            weight_avg_fb.outer_iter_mut()
        ) {
            // iterate along baseline axis
            for (weight_chunk_tf, weight_avg) in izip!(
                weight_chunk_tfb.axis_iter(baseline_axis),
                weight_avg_b.iter_mut()
            ) {
                let mut weight_sum: f64 = 0.0;
                for &weight in weight_chunk_tf.iter() {
                    weight_sum += weight as f64;
                }

                *weight_avg = (weight_sum as f32).max(0.);
            }
        }
    }
}

// #[allow(clippy::too_many_arguments)]
// #[deprecated = "doesn't support --no-precession"]
// fn vis_rotate2(
//     jones_array: ArrayView3<Jones<f32>>,
//     mut jones_array2: ArrayViewMut3<Jones<f32>>,
//     phase_to: RADec,
//     tile_xyzs: ArrayView2<XyzGeodetic>,
//     tile_ws_from: ArrayView2<W>,
//     mut tile_ws_to: ArrayViewMut2<W>,
//     lmsts: &[f64],
//     fine_chan_lambdas_m: &[f64],
// ) {
//     let num_tiles = tile_xyzs.len_of(Axis(1));
//     assert_eq!(tile_ws_from.len_of(Axis(1)), num_tiles);
//     assert_eq!(tile_ws_to.len_of(Axis(1)), num_tiles);

//     // iterate along time axis in chunks of avg_time
//     jones_array
//         .outer_iter()
//         .into_par_iter()
//         .zip(jones_array2.outer_iter_mut())
//         .zip(tile_ws_from.outer_iter())
//         .zip(tile_ws_to.outer_iter_mut())
//         .zip(tile_xyzs.outer_iter())
//         .zip(lmsts.par_iter())
//         .for_each(
//             |(((((vis_tfb, mut vis_rot_tfb), tile_ws_from), mut tile_ws_to), tile_xyzs), lmst)| {
//                 assert_eq!(tile_ws_from.len(), num_tiles);
//                 // Generate the "to" Ws.
//                 let phase_to = phase_to.to_hadec(*lmst);
//                 setup_ws(
//                     tile_ws_to.as_slice_mut().unwrap(),
//                     tile_xyzs.as_slice().unwrap(),
//                     phase_to,
//                 );

//                 vis_rotate_fb(
//                     vis_tfb,
//                     vis_rot_tfb,
//                     tile_ws_from.as_slice().unwrap(),
//                     tile_ws_to.as_slice().unwrap(),
//                     fine_chan_lambdas_m,
//                 );
//             },
//         );
// }

fn vis_rotate_fb(
    vis_fb: ArrayView2<Jones<f32>>,
    mut vis_rot_fb: ArrayViewMut2<Jones<f32>>,
    tile_ws_from: &[W],
    tile_ws_to: &[W],
    fine_chan_lambdas_m: &[f64],
) {
    let num_tiles = tile_ws_from.len();
    let mut i_tile1 = 0;
    let mut i_tile2 = 0;
    let mut tile1_w_from = tile_ws_from[i_tile1];
    let mut tile2_w_from = tile_ws_from[i_tile2];
    let mut tile1_w_to = tile_ws_to[i_tile1];
    let mut tile2_w_to = tile_ws_to[i_tile2];
    // iterate along baseline axis
    vis_fb
        .axis_iter(Axis(1))
        .zip(vis_rot_fb.axis_iter_mut(Axis(1)))
        .for_each(|(vis_f, mut vis_rot_f)| {
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
            vis_f
                .iter()
                .zip(vis_rot_f.iter_mut())
                .zip(fine_chan_lambdas_m.iter())
                .for_each(|((jones, jones_rot), lambda_m)| {
                    let rotation = Complex::cis(arg / *lambda_m);
                    *jones_rot = Jones::<f32>::from(Jones::<f64>::from(*jones) * rotation);
                });
        });
}

/// Rotate the supplied visibilities according to the `λ²` constants of
/// proportionality with `exp(-2πi(αu+βv)λ²)`.
fn apply_iono2(
    vis_tfb: ArrayView3<Jones<f32>>,
    mut vis_iono_tfb: ArrayViewMut3<Jones<f32>>,
    tile_uvs: ArrayView2<UV>,
    iono_consts: IonoConsts,
    lambdas_m: &[f64],
) {
    let num_tiles = tile_uvs.len_of(Axis(1));

    // iterate along time axis
    vis_tfb
        .outer_iter()
        .zip(vis_iono_tfb.outer_iter_mut())
        .zip(tile_uvs.outer_iter())
        .for_each(|((vis_fb, mut vis_iono_fb), tile_uvs)| {
            // Just in case the compiler can't understand how an ndarray is laid
            // out.
            assert_eq!(tile_uvs.len(), num_tiles);

            // iterate along baseline axis
            let mut i_tile1 = 0;
            let mut i_tile2 = 0;
            vis_fb
                .axis_iter(Axis(1))
                .zip(vis_iono_fb.axis_iter_mut(Axis(1)))
                .for_each(|(vis_f, mut vis_iono_f)| {
                    i_tile2 += 1;
                    if i_tile2 == num_tiles {
                        i_tile1 += 1;
                        i_tile2 = i_tile1 + 1;
                    }

                    let UV { u, v } = tile_uvs[i_tile1] - tile_uvs[i_tile2];
                    let arg = -TAU * (u * iono_consts.alpha + v * iono_consts.beta);
                    // iterate along frequency axis
                    vis_f
                        .iter()
                        .zip(vis_iono_f.iter_mut())
                        .zip(lambdas_m.iter())
                        .for_each(|((jones, jones_iono), lambda_m)| {
                            let j = Jones::<f64>::from(*jones);
                            // The baseline UV is in units of metres, so we need
                            // to divide by λ to use it in an exponential. But
                            // we're also multiplying by λ², so just multiply by
                            // λ.
                            let rotation = Complex::cis(arg * *lambda_m);
                            *jones_iono = Jones::from(j * rotation);
                        });
                });
        });
}

/// unpeel model, peel iono model
/// this is useful when vis_model has already been subtraced from vis_residual
/// TODO (Dev): rename to unpeel_model
fn apply_iono3(
    vis_model: ArrayView3<Jones<f32>>,
    mut vis_residual: ArrayViewMut3<Jones<f32>>,
    tile_uvs: ArrayView2<UV>,
    iono_consts: IonoConsts,
    old_iono_consts: IonoConsts,
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
                    let arg = -TAU * (u * iono_consts.alpha + v * iono_consts.beta);
                    let old_arg = -TAU * (u * old_iono_consts.alpha + v * old_iono_consts.beta);
                    // iterate along frequency axis
                    vis_model
                        .iter()
                        .zip(vis_residual.iter_mut())
                        .zip(lambdas_m.iter())
                        .for_each(|((vis_model, vis_residual), lambda_m)| {
                            let mut j = Jones::<f64>::from(*vis_residual);
                            let m = Jones::<f64>::from(*vis_model);
                            // The baseline UV is in units of metres, so we need
                            // to divide by λ to use it in an exponential. But
                            // we're also multiplying by λ², so just multiply by
                            // λ.
                            let old_rotation = Complex::cis(old_arg * *lambda_m);
                            j += m * old_rotation;

                            let rotation = Complex::cis(arg * *lambda_m);
                            j -= m * rotation;
                            *vis_residual = Jones::from(j);
                        });
                });
        });
}

// the offsets as defined by the RTS code
// TODO: Assume there's only 1 timestep, because this is low res data?
fn iono_fit(
    residual: ArrayView3<Jones<f32>>,
    weights: ArrayView3<f32>,
    model: ArrayView3<Jones<f32>>,
    lambdas_m: &[f64],
    tile_uvs_low_res: ArrayView2<UV>,
) -> [f64; 4] {
    let num_tiles = tile_uvs_low_res.len_of(Axis(1));

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
        .zip(tile_uvs_low_res.outer_iter())
        .for_each(|(((residual, weights), model), tile_uvs_low_res)| {
            // iterate over frequency
            residual
                .outer_iter()
                .zip(weights.outer_iter())
                .zip(model.outer_iter())
                .zip(lambdas_m.iter())
                .for_each(|(((residual, weights), model), &lambda)| {
                    let lambda_2 = lambda * lambda;

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
                                // Normally, we would divide by λ to get
                                // dimensionless UV. However, UV are only used
                                // to determine a_uu, a_uv, a_vv, which are also
                                // scaled by lambda. So... don't divide by λ.
                                let UV { u, v } = uv_tile1 - uv_tile2;

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

                                #[cfg(test)]
                                {
                                    println!("uv ({:+1.5}, {:+1.5}) l{:+1.3} | RI {:+1.5} @{:+1.5}pi | MI {:+1.5} @{:+1.5}pi", u, v, lambda, residual_i.norm(), residual_i.arg(), model_i.norm(), model_i.arg());
                                    if i_tile1 == 0 && i_tile2 == 1 {
                                        let a_uu_asdf = weight * mm * u * u;
                                        let a_uv_asdf = weight * mm * u * v;
                                        let a_vv_asdf = weight * mm * v * v;
                                        dbg!(residual_i, model_i, weight, u, v, mr, mm, a_uu_asdf, a_uv_asdf, a_vv_asdf);
                                    }
                                }

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

                    // As above, we didn't divide UV by lambda, so below we use
                    // λ² for λ⁴, and λ for λ².
                    a_uu += a_uu_bl * lambda_2;
                    a_uv += a_uv_bl * lambda_2;
                    a_vv += a_vv_bl * lambda_2;
                    aa_u += aa_u_bl * -lambda;
                    aa_v += aa_v_bl * -lambda;
                    s_vm += s_vm_bl;
                    s_mm += s_mm_bl;
                });
        });

    let denom = TAU * (a_uu * a_vv - a_uv * a_uv);
    #[cfg(test)]
    dbg!(a_uu, a_vv, a_uv, denom);
    [
        (aa_u * a_vv - aa_v * a_uv) / denom,
        (aa_v * a_uu - aa_u * a_uv) / denom,
        s_vm,
        s_mm,
    ]
}

#[cfg(test)]
fn setup_ws(tile_ws: &mut [W], tile_xyzs: &[XyzGeodetic], phase_centre: HADec) {
    assert_eq!(tile_ws.len(), tile_xyzs.len());
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
    assert_eq!(tile_uvs.len(), tile_xyzs.len());
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

#[allow(clippy::too_many_arguments)]
fn peel_cpu(
    // TODO (Dev): I would name this vis_residual_tfb
    mut vis_residual: ArrayViewMut3<Jones<f32>>,
    // TODO (Dev): I would name this vis_weights
    vis_weights: ArrayView3<f32>,
    timeblock: &Timeblock,
    source_list: &SourceList,
    iono_consts: &mut [IonoConsts],
    source_weighted_positions: &[RADec],
    num_passes: usize,
    // TODO (dev): Why do we need both this and low_res_lambdas_m? it's not even used
    _low_res_freqs_hz: &[f64],
    all_fine_chan_lambdas_m: &[f64],
    low_res_lambdas_m: &[f64],
    obs_context: &ObsContext,
    // TODO (dev): array_position is available from obs_context
    array_position: LatLngHeight,
    // TODO (dev): unflagged_tile_xyzs is available from obs_context
    unflagged_tile_xyzs: &[XyzGeodetic],
    low_res_modeller: &mut dyn SkyModeller,
    high_res_modeller: &mut dyn SkyModeller,
    // TODO (dev): dut1 is available from obs_context
    dut1: Duration,
    no_precession: bool,
    multi_progress_bar: &MultiProgress,
) -> Result<(), PeelError> {
    // TODO: Do we allow multiple timesteps in the low-res data?

    let timestamps = &timeblock.timestamps;
    let num_timestamps_high_res = timestamps.len();
    let num_timestamps_low_res = 1;
    let avg_time = num_timestamps_high_res / num_timestamps_low_res;

    let num_tiles = unflagged_tile_xyzs.len();
    let num_cross_baselines = (num_tiles * (num_tiles - 1)) / 2;

    let num_freqs_high_res = all_fine_chan_lambdas_m.len();
    let num_freqs_low_res = low_res_lambdas_m.len();

    let num_sources = source_list.len();
    let num_sources_to_iono_subtract = iono_consts.len();

    // TODO: these assertions should be actual errors.
    let (time_axis, freq_axis, baseline_axis) = (Axis(0), Axis(1), Axis(2));

    assert_eq!(vis_residual.len_of(time_axis), num_timestamps_high_res);
    assert_eq!(vis_weights.len_of(time_axis), num_timestamps_high_res);

    assert_eq!(vis_residual.len_of(baseline_axis), num_cross_baselines);
    assert_eq!(vis_weights.len_of(baseline_axis), num_cross_baselines);

    assert_eq!(vis_residual.len_of(freq_axis), num_freqs_high_res);
    assert_eq!(vis_weights.len_of(freq_axis), num_freqs_high_res);

    assert_eq!(iono_consts.len(), num_sources_to_iono_subtract);
    assert!(num_sources_to_iono_subtract <= num_sources);

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

    // observation phase center
    let mut tile_uvs_high_res = Array2::<UV>::default((timestamps.len(), num_tiles));
    // TODO (Dev): I would name this "tile_ws_high_res"
    let mut tile_ws_from = Array2::<W>::default((timestamps.len(), num_tiles));
    // source phase center
    // TODO (Dev): I would name this tile_uvs_high_res_rot
    let mut tile_uvs_high_res_rot = tile_uvs_high_res.clone();
    // TODO (Dev): I would name this tile_ws_high_res_rot
    let mut tile_ws_to = tile_ws_from.clone();
    // TODO (Dev): I would name this tile_uvs_low_res_rot
    let mut tile_uvs_low_res = Array2::<UV>::default((1, num_tiles));

    // Pre-compute high-res tile UVs and Ws at observation phase centre.
    for (&time, mut tile_uvs, mut tile_ws) in izip!(
        timestamps.iter(),
        tile_uvs_high_res.outer_iter_mut(),
        tile_ws_from.outer_iter_mut(),
    ) {
        let (lmst, precessed_xyzs) = if !no_precession {
            let precession_info = precess_time(
                array_position.longitude_rad,
                array_position.latitude_rad,
                obs_context.phase_centre,
                time,
                dut1,
            );
            let precessed_xyzs = precession_info.precess_xyz(unflagged_tile_xyzs);
            (precession_info.lmst_j2000, precessed_xyzs)
        } else {
            let lmst = get_lmst(array_position.longitude_rad, time, dut1);
            (lmst, unflagged_tile_xyzs.into())
        };
        let hadec_phase = obs_context.phase_centre.to_hadec(lmst);
        let (s_ha, c_ha) = hadec_phase.ha.sin_cos();
        let (s_dec, c_dec) = hadec_phase.dec.sin_cos();
        for (tile_uv, tile_w, &precessed_xyzs) in izip!(
            tile_uvs.iter_mut(),
            tile_ws.iter_mut(),
            precessed_xyzs.iter(),
        ) {
            let uvw = UVW::from_xyz_inner(precessed_xyzs, s_ha, c_ha, s_dec, c_dec);
            *tile_uv = UV { u: uvw.u, v: uvw.v };
            *tile_w = W(uvw.w);
        }
    }

    // TODO (Dev): iono_taper_weights could be supplied to peel
    // use the baseline taper from the RTS, 1-exp(-(u*u+v*v)/(2*sig^2));
    // let short_baseline_sigma = 20.;
    // TODO: Do we care about weights changing over time?
    // let vis_weights = {
    //     let mut iono_taper = get_weights_rts(
    //         tile_uvs_high_res.view(),
    //         all_fine_chan_lambdas_m,
    //         short_baseline_sigma,
    //         (obs_context.guess_freq_res() / FREQ_WEIGHT_FACTOR)
    //             * (obs_context.guess_time_res().to_seconds() / TIME_WEIGHT_FACTOR),
    //     );
    //     iono_taper *= &vis_weights;
    //     iono_taper
    // };

    // Temporary visibility array, re-used for each timestep
    // TODO (Dev): I would name this vis_residual_rot_tfb
    let mut vis_residual_tmp = vis_residual.to_owned();
    let high_res_vis_dims = vis_residual.dim();
    let mut vis_model_high_res = Array3::default(high_res_vis_dims);

    // temporary arrays for accumulation
    // TODO: Do a stocktake of arrays that are lying around!
    // TODO (Dev): I would name this vis_residual_low_res_rot
    let mut vis_residual_low_res: Array3<Jones<f32>> = Array3::zeros((
        num_timestamps_low_res,
        num_freqs_low_res,
        num_cross_baselines,
    ));
    let mut vis_model_low_res = vis_residual_low_res.clone();
    // TODO (Dev): I would name this vis_model_low_res_rot
    let mut vis_model_low_res_tmp = vis_residual_low_res.clone();
    let mut vis_weights_low_res: Array3<f32> = Array3::zeros(vis_residual_low_res.dim());

    // The low-res weights only need to be populated once.
    weights_average(vis_weights.view(), vis_weights_low_res.view_mut());

    for pass in 0..num_passes {
        for (((source_name, source), iono_consts), source_phase_centre) in source_list
            .iter()
            .take(num_sources_to_iono_subtract)
            .zip_eq(iono_consts.iter_mut())
            .zip_eq(source_weighted_positions.iter().copied())
        {
            multi_progress_bar.suspend(|| {
                debug!(
                    "peel loop {pass}: {source_name} at {source_phase_centre} (has iono ({} {}))",
                    iono_consts.alpha, iono_consts.beta
                )
            });
            let start = std::time::Instant::now();
            let old_iono_consts = *iono_consts;

            low_res_modeller.update_with_a_source(source, source_phase_centre)?;
            high_res_modeller.update_with_a_source(source, obs_context.phase_centre)?;
            // this is only necessary for cpu modeller.
            vis_model_low_res.fill(Jones::zero());

            multi_progress_bar.suspend(|| {
                trace!(
                    "{:?}: initialize modellers",
                    std::time::Instant::now() - start
                )
            });
            // TODO (dev): model_timestep returns uvws, could re-use these here.
            // iterate along time chunks:
            // - calculate high-res uvws in source phase centre
            // - rotate residuals to source phase centre
            // - model low res visibilities in source phase centre
            // - calculate low-res uvws in source phase centre
            for (
                timestamps,
                vis_residual_tfb,
                mut vis_residual_rot_tfb,
                mut vis_model_low_res_rot_fb,
                tile_ws_high_res,
                mut tile_uvs_high_res_rot,
                mut tile_ws_high_res_rot,
                mut tile_uvs_low_res_rot,
            ) in izip!(
                timestamps.chunks(avg_time),
                vis_residual.axis_chunks_iter(time_axis, avg_time),
                vis_residual_tmp.axis_chunks_iter_mut(time_axis, avg_time),
                vis_model_low_res.outer_iter_mut(),
                tile_ws_from.axis_chunks_iter(time_axis, avg_time),
                tile_uvs_high_res_rot.axis_chunks_iter_mut(time_axis, avg_time),
                tile_ws_to.axis_chunks_iter_mut(time_axis, avg_time),
                tile_uvs_low_res.outer_iter_mut(),
            ) {
                multi_progress_bar.suspend(|| {
                    trace!(
                        "{:?}: calc source uvw, rotate residual",
                        std::time::Instant::now() - start
                    )
                });
                // iterate along high res times
                for (
                    &time,
                    vis_residual_fb,
                    mut vis_residual_rot_fb,
                    tile_ws_high_res,
                    mut tile_uvs_high_res_rot,
                    mut tile_ws_high_res_rot,
                ) in izip!(
                    timestamps,
                    vis_residual_tfb.outer_iter(),
                    vis_residual_rot_tfb.outer_iter_mut(),
                    tile_ws_high_res.outer_iter(),
                    tile_uvs_high_res_rot.outer_iter_mut(),
                    tile_ws_high_res_rot.outer_iter_mut(),
                ) {
                    let (lmst, precessed_xyzs) = if !no_precession {
                        let precession_info = precess_time(
                            array_position.longitude_rad,
                            array_position.latitude_rad,
                            obs_context.phase_centre,
                            time,
                            dut1,
                        );
                        let precessed_xyzs = precession_info.precess_xyz(unflagged_tile_xyzs);
                        (precession_info.lmst_j2000, precessed_xyzs)
                    } else {
                        let lmst = get_lmst(array_position.longitude_rad, time, dut1);
                        (lmst, unflagged_tile_xyzs.into())
                    };
                    let hadec_source = source_phase_centre.to_hadec(lmst);
                    let (s_ha, c_ha) = hadec_source.ha.sin_cos();
                    let (s_dec, c_dec) = hadec_source.dec.sin_cos();
                    for (tile_uv, tile_w, &precessed_xyz) in izip!(
                        tile_uvs_high_res_rot.iter_mut(),
                        tile_ws_high_res_rot.iter_mut(),
                        precessed_xyzs.iter(),
                    ) {
                        let UVW { u, v, w } =
                            UVW::from_xyz_inner(precessed_xyz, s_ha, c_ha, s_dec, c_dec);
                        *tile_uv = UV { u, v };
                        *tile_w = W(w);
                    }

                    vis_rotate_fb(
                        vis_residual_fb.view(),
                        vis_residual_rot_fb.view_mut(),
                        tile_ws_high_res.as_slice().unwrap(),
                        tile_ws_high_res_rot.as_slice().unwrap(),
                        all_fine_chan_lambdas_m,
                    );
                }
                multi_progress_bar
                    .suspend(|| trace!("{:?}: low-res uvws", std::time::Instant::now() - start));

                let low_res_epoch = average_epoch(timestamps);
                // compute low-res tile UVs at source phase centre.
                let (lmst, precessed_xyzs) = if !no_precession {
                    let precession_info = precess_time(
                        array_position.longitude_rad,
                        array_position.latitude_rad,
                        obs_context.phase_centre,
                        low_res_epoch,
                        dut1,
                    );
                    let precessed_xyzs = precession_info.precess_xyz(unflagged_tile_xyzs);
                    (precession_info.lmst_j2000, precessed_xyzs)
                } else {
                    let lmst = get_lmst(array_position.longitude_rad, low_res_epoch, dut1);
                    (lmst, unflagged_tile_xyzs.into())
                };
                let hadec_source = source_phase_centre.to_hadec(lmst);
                setup_uvs(
                    tile_uvs_low_res_rot.as_slice_mut().unwrap(),
                    &precessed_xyzs,
                    hadec_source,
                );

                multi_progress_bar
                    .suspend(|| trace!("{:?}: low-res model", std::time::Instant::now() - start));
                low_res_modeller
                    .model_timestep(vis_model_low_res_rot_fb.view_mut(), low_res_epoch)?;
            }

            multi_progress_bar
                .suspend(|| trace!("{:?}: vis_average", std::time::Instant::now() - start));
            vis_average2(
                vis_residual_tmp.view(),
                vis_residual_low_res.view_mut(),
                vis_weights.view(),
            );

            // Add the low-res model to the residuals. If the iono consts are
            // non-zero, then also shift the model before adding it.
            multi_progress_bar
                .suspend(|| trace!("{:?}: add low-res model", std::time::Instant::now() - start));
            if iono_consts.alpha.abs() > f64::EPSILON && iono_consts.beta.abs() > f64::EPSILON {
                apply_iono2(
                    vis_model_low_res.view(),
                    vis_model_low_res_tmp.view_mut(),
                    tile_uvs_low_res.view(),
                    *iono_consts,
                    low_res_lambdas_m,
                );
                Zip::from(&mut vis_residual_low_res)
                    .and(&vis_model_low_res_tmp)
                    .for_each(|r, m| {
                        *r += *m;
                    });
            } else {
                Zip::from(&mut vis_residual_low_res)
                    .and(&vis_model_low_res)
                    .for_each(|r, m| {
                        *r += *m;
                    });
            }

            multi_progress_bar
                .suspend(|| trace!("{:?}: alpha/beta loop", std::time::Instant::now() - start));
            // let mut gain_update = 1.0;
            let mut iteration = 0;
            while iteration != 10 {
                iteration += 1;
                multi_progress_bar.suspend(|| debug!("iter {iteration}, consts: {iono_consts:?}"));

                // iono rotate model using existing iono consts
                apply_iono2(
                    vis_model_low_res.view(),
                    vis_model_low_res_tmp.view_mut(),
                    tile_uvs_low_res.view(),
                    *iono_consts,
                    low_res_lambdas_m,
                );

                let iono_fits = iono_fit(
                    vis_residual_low_res.view(),
                    vis_weights_low_res.view(),
                    vis_model_low_res_tmp.view(),
                    low_res_lambdas_m,
                    tile_uvs_low_res.view(),
                );
                multi_progress_bar.suspend(|| trace!("iono_fits: {iono_fits:?}"));

                iono_consts.alpha += iono_fits[0];
                iono_consts.beta += iono_fits[1];
                // gain_update *= iono_fits[2] / iono_fits[3];
                // vis_model_low_res
                //     .iter_mut()
                //     .for_each(|v| *v *= gain_update as f32);

                // if the offset is small, we've converged.
                if iono_fits[0].abs() < 1e-12 && iono_fits[1].abs() < 1e-12 {
                    debug!("iter {iteration}, consts: {iono_consts:?}, finished");
                    break;
                }
            }

            multi_progress_bar
                .suspend(|| trace!("{:?}: high res model", std::time::Instant::now() - start));
            vis_model_high_res.fill(Jones::default());
            model_timesteps(high_res_modeller, timestamps, vis_model_high_res.view_mut())?;

            multi_progress_bar
                .suspend(|| trace!("{:?}: apply_iono3", std::time::Instant::now() - start));
            // add the model to residual, and subtract the iono rotated model
            apply_iono3(
                vis_model_high_res.view(),
                vis_residual.view_mut(),
                // TODO: pretty sure this needs to be tile_uvs_high_res_rot
                // tile_uvs_high_res.view(),
                tile_uvs_high_res_rot.view(),
                *iono_consts,
                old_iono_consts,
                all_fine_chan_lambdas_m,
            );

            multi_progress_bar.suspend(|| {
                debug!(
                    "peel loop finished: {source_name} at {source_phase_centre} (has iono {iono_consts:?})"
                )
            });
            peel_progress.inc(1);
        }
    }

    Ok(())
}

#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
fn peel_cuda(
    mut vis_residual: ArrayViewMut3<Jones<f32>>,
    vis_weights: ArrayView3<f32>,
    timeblock: &Timeblock,
    source_list: &SourceList,
    iono_consts: &mut [IonoConsts],
    source_weighted_positions: &[RADec],
    num_passes: usize,
    low_res_freqs_hz: &[f64],
    all_fine_chan_lambdas_m: &[f64],
    low_res_lambdas_m: &[f64],
    obs_context: &ObsContext,
    array_position: LatLngHeight,
    unflagged_tile_xyzs: &[XyzGeodetic],
    low_res_modeller: &mut SkyModellerCuda,
    high_res_modeller: &mut SkyModellerCuda,
    dut1: Duration,
    no_precession: bool,
    multi_progress_bar: &MultiProgress,
) -> Result<(), PeelError> {
    use std::ffi::CStr;

    let num_sources_to_iono_subtract = iono_consts.len();

    // TODO: Do we allow multiple timesteps in the low-res data?

    let timestamps = &timeblock.timestamps;
    let peel_progress = multi_progress_bar.add(
        ProgressBar::new(num_sources_to_iono_subtract as _)
            .with_style(
                ProgressStyle::default_bar()
                    .template("{msg:17}: [{wide_bar:.blue}] {pos:2}/{len:2} sources ({elapsed_precise}<{eta_precise})").unwrap()
                    .progress_chars("=> "),
            )
            .with_position(0)
    );
    peel_progress.tick();

    let num_timesteps = vis_residual.len_of(Axis(0));
    let num_tiles = unflagged_tile_xyzs.len();
    let num_cross_baselines = (num_tiles * (num_tiles - 1)) / 2;

    let mut lmsts = vec![0.; timestamps.len()];
    let mut latitudes = vec![0.; timestamps.len()];
    let mut tile_xyzs_high_res = Array2::<XyzGeodetic>::default((timestamps.len(), num_tiles));
    let mut high_res_uvws = Array2::default((timestamps.len(), num_cross_baselines));
    let mut tile_uvs_high_res = Array2::<UV>::default((timestamps.len(), num_tiles));
    let mut tile_ws_high_res = Array2::<W>::default((timestamps.len(), num_tiles));

    // Pre-compute high-res tile UVs and Ws at observation phase centre.
    for (
        &time,
        lmst,
        latitude,
        mut tile_xyzs_high_res,
        mut high_res_uvws,
        mut tile_uvs_high_res,
        mut tile_ws_high_res,
    ) in izip!(
        timestamps.iter(),
        lmsts.iter_mut(),
        latitudes.iter_mut(),
        tile_xyzs_high_res.outer_iter_mut(),
        high_res_uvws.outer_iter_mut(),
        tile_uvs_high_res.outer_iter_mut(),
        tile_ws_high_res.outer_iter_mut(),
    ) {
        if !no_precession {
            let precession_info = precess_time(
                array_position.longitude_rad,
                array_position.latitude_rad,
                obs_context.phase_centre,
                time,
                dut1,
            );
            tile_xyzs_high_res
                .iter_mut()
                .zip_eq(&precession_info.precess_xyz(unflagged_tile_xyzs))
                .for_each(|(a, b)| *a = *b);
            *lmst = precession_info.lmst_j2000;
            *latitude = precession_info.array_latitude_j2000;
        } else {
            tile_xyzs_high_res
                .iter_mut()
                .zip_eq(unflagged_tile_xyzs)
                .for_each(|(a, b)| *a = *b);
            *lmst = get_lmst(array_position.longitude_rad, time, dut1);
            *latitude = array_position.latitude_rad;
        };
        let hadec_phase = obs_context.phase_centre.to_hadec(*lmst);
        let (s_ha, c_ha) = hadec_phase.ha.sin_cos();
        let (s_dec, c_dec) = hadec_phase.dec.sin_cos();
        let mut tile_uvws_high_res = vec![UVW::default(); num_tiles];
        for (tile_uvw, tile_uv, tile_w, &tile_xyz) in izip!(
            tile_uvws_high_res.iter_mut(),
            tile_uvs_high_res.iter_mut(),
            tile_ws_high_res.iter_mut(),
            tile_xyzs_high_res.iter(),
        ) {
            let uvw = UVW::from_xyz_inner(tile_xyz, s_ha, c_ha, s_dec, c_dec);
            *tile_uvw = uvw;
            *tile_uv = UV { u: uvw.u, v: uvw.v };
            *tile_w = W(uvw.w);
        }

        // The UVWs for every timestep will be the same (because the phase
        // centres are always the same). Make these ahead of time for
        // efficiency.
        let mut count = 0;
        for (i, t1) in tile_uvws_high_res.iter().enumerate() {
            for t2 in tile_uvws_high_res.iter().skip(i + 1) {
                high_res_uvws[count] = *t1 - *t2;
                count += 1;
            }
        }
    }

    // use the baseline taper from the RTS, 1-exp(-(u*u+v*v)/(2*sig^2));
    // let short_baseline_sigma = 20.;
    // TODO: Do we care about weights changing over time?
    // let vis_weights = {
    //     let mut iono_taper = get_weights_rts(
    //         tile_uvs_high_res.view(),
    //         all_fine_chan_lambdas_m,
    //         short_baseline_sigma,
    //         (obs_context.guess_freq_res() / FREQ_WEIGHT_FACTOR)
    //             * (obs_context.guess_time_res().to_seconds() / TIME_WEIGHT_FACTOR),
    //     );
    //     iono_taper *= &vis_weights;
    //     iono_taper
    // };

    let (average_lmst, average_latitude, average_tile_xyzs) = if no_precession {
        let average_timestamp = average_epoch(timestamps);
        let average_tile_xyzs =
            ArrayView2::from_shape((1, num_tiles), unflagged_tile_xyzs).expect("correct shape");
        (
            get_lmst(array_position.longitude_rad, average_timestamp, dut1),
            array_position.latitude_rad,
            CowArray::from(average_tile_xyzs),
        )
    } else {
        let average_timestamp = average_epoch(timestamps);
        let average_precession_info = precess_time(
            array_position.longitude_rad,
            array_position.latitude_rad,
            obs_context.phase_centre,
            average_timestamp,
            dut1,
        );
        let average_precessed_tile_xyzs = Array2::from_shape_vec(
            (1, num_tiles),
            average_precession_info.precess_xyz(unflagged_tile_xyzs),
        )
        .expect("correct shape");

        (
            average_precession_info.lmst_j2000,
            average_precession_info.array_latitude_j2000,
            CowArray::from(average_precessed_tile_xyzs),
        )
    };

    // temporary arrays for accumulation
    // TODO: Do a stocktake of arrays that are lying around!
    // These are time, bl, channel
    let vis_residual_low_res: Array3<Jones<f32>> =
        Array3::zeros((1, low_res_freqs_hz.len(), num_cross_baselines));
    let mut vis_weights_low_res: Array3<f32> = Array3::zeros(vis_residual_low_res.dim());

    // The low-res weights only need to be populated once.
    weights_average(vis_weights.view(), vis_weights_low_res.view_mut());

    let freq_average_factor = all_fine_chan_lambdas_m.len() / low_res_freqs_hz.len();

    unsafe {
        let cuda_xyzs_high_res: Vec<_> = tile_xyzs_high_res
            .iter()
            .copied()
            .map(|XyzGeodetic { x, y, z }| cuda::XYZ {
                x: x as CudaFloat,
                y: y as CudaFloat,
                z: z as CudaFloat,
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
            .zip(tile_xyzs_high_res.outer_iter())
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
        let cuda_xyzs_low_res: Vec<_> = average_tile_xyzs
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

        let d_xyzs = DevicePointer::copy_to_device(&cuda_xyzs_high_res).unwrap();
        let d_uvws_from = DevicePointer::copy_to_device(cuda_uvws.as_slice().unwrap()).unwrap();
        let mut d_uvws_to =
            DevicePointer::malloc(cuda_uvws.len() * std::mem::size_of::<cuda::UVW>()).unwrap();
        let d_lmsts = DevicePointer::copy_to_device(&cuda_lmsts).unwrap();
        let d_lambdas = DevicePointer::copy_to_device(&cuda_lambdas).unwrap();
        let d_xyzs_low_res = DevicePointer::copy_to_device(&cuda_xyzs_low_res).unwrap();
        let d_average_lmsts = DevicePointer::copy_to_device(&[average_lmst as CudaFloat]).unwrap();
        let mut d_uvws_source_low_res: DevicePointer<cuda::UVW> =
            DevicePointer::malloc(cuda_uvws.len() * std::mem::size_of::<cuda::UVW>()).unwrap();
        let d_low_res_lambdas = DevicePointer::copy_to_device(&cuda_low_res_lambdas).unwrap();
        let mut d_iono_fits: DevicePointer<Jones<f64>> = DevicePointer::malloc(
            num_cross_baselines * low_res_freqs_hz.len() * std::mem::size_of::<Jones<f64>>(),
        )?;
        let mut d_iono_consts = DevicePointer::malloc(std::mem::size_of::<cuda::IonoConsts>())?;
        let mut d_old_iono_consts = DevicePointer::malloc(std::mem::size_of::<cuda::IonoConsts>())?;

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

        let mut d_high_res_model: DevicePointer<Jones<f32>> = DevicePointer::malloc(
            timestamps.len()
                * num_cross_baselines
                * all_fine_chan_lambdas_m.len()
                * std::mem::size_of::<Jones<f32>>(),
        )
        .unwrap();

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
                .for_each(|(&UVW { u, v, w }, cuda_uvw)| {
                    *cuda_uvw = cuda::UVW {
                        u: u as CudaFloat,
                        v: v as CudaFloat,
                        w: w as CudaFloat,
                    }
                });
            d_uvws.push(cuda::DevicePointer::copy_to_device(&cuda_uvws).unwrap());
        }

        for pass in 0..num_passes {
            peel_progress.reset();
            peel_progress.set_message(format!(
                "Peeling timeblock {}, pass {}",
                timeblock.index + 1,
                pass + 1
            ));

            for (((source_name, source), iono_consts), source_phase_centre) in source_list
                .iter()
                .take(num_sources_to_iono_subtract)
                .zip_eq(iono_consts.iter_mut())
                .zip_eq(source_weighted_positions.iter().copied())
            {
                let start = std::time::Instant::now();
                multi_progress_bar.suspend(|| {
                    debug!(
                        "peel loop {pass}: {source_name} at {source_phase_centre} (has iono {iono_consts:?})"
                    )
                });

                let old_iono_consts = *iono_consts;
                d_iono_consts.overwrite(&[cuda::IonoConsts {
                    alpha: iono_consts.alpha,
                    beta: iono_consts.beta,
                    s_vm: iono_consts.s_vm,
                    s_mm: iono_consts.s_mm,
                }])?;
                d_old_iono_consts.overwrite(&[cuda::IonoConsts {
                    alpha: old_iono_consts.alpha,
                    beta: old_iono_consts.beta,
                    s_vm: old_iono_consts.s_vm,
                    s_mm: old_iono_consts.s_mm,
                }])?;

                let error_message_ptr = cuda::rotate_average(
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
                    all_fine_chan_lambdas_m.len().try_into().unwrap(),
                    freq_average_factor.try_into().unwrap(),
                    d_lmsts.get(),
                    d_xyzs.get(),
                    d_uvws_from.get(),
                    d_uvws_to.get_mut(),
                    d_lambdas.get(),
                );
                if !error_message_ptr.is_null() {
                    // Get the CUDA error message associated with the enum variant.
                    let error_message = CStr::from_ptr(error_message_ptr)
                        .to_str()
                        .unwrap_or("<cannot read CUDA error string>");
                    let our_error_str =
                        format!("{}:{}: rotate_average: {error_message}", file!(), line!());
                    return Err(CudaError::Kernel(our_error_str).into());
                }

                multi_progress_bar
                    .suspend(|| trace!("{:?}: rotate_average", std::time::Instant::now() - start));

                low_res_modeller.update_with_a_source(source, source_phase_centre)?;
                low_res_modeller.clear_vis();
                multi_progress_bar.suspend(|| {
                    trace!(
                        "{:?}: low res update and clear",
                        std::time::Instant::now() - start
                    )
                });

                let error_message_ptr = cuda::xyzs_to_uvws(
                    d_xyzs_low_res.get(),
                    d_average_lmsts.get(),
                    d_uvws_source_low_res.get_mut(),
                    cuda::RADec {
                        ra: source_phase_centre.ra as CudaFloat,
                        dec: source_phase_centre.dec as CudaFloat,
                    },
                    num_tiles.try_into().unwrap(),
                    num_cross_baselines.try_into().unwrap(),
                    1,
                );
                if !error_message_ptr.is_null() {
                    let error_message = CStr::from_ptr(error_message_ptr)
                        .to_str()
                        .unwrap_or("<cannot read CUDA error string>");
                    let our_error_str =
                        format!("{}:{}: xyzs_to_uvws: {error_message}", file!(), line!());
                    return Err(CudaError::Kernel(our_error_str).into());
                }

                multi_progress_bar.suspend(|| {
                    trace!(
                        "{:?}: low res xyzs_to_uvws",
                        std::time::Instant::now() - start
                    )
                });

                low_res_modeller.model_with_uvws2(
                    &d_uvws_source_low_res,
                    average_lmst,
                    average_latitude,
                )?;
                multi_progress_bar
                    .suspend(|| trace!("{:?}: low res model", std::time::Instant::now() - start));

                let error_message_ptr = cuda::add_model(
                    d_low_res_vis.get_mut().cast(),
                    low_res_modeller.d_vis.get().cast(),
                    d_iono_consts.get(),
                    d_low_res_lambdas.get(),
                    d_uvws_source_low_res.get(),
                    vis_residual_low_res.len_of(Axis(0)).try_into().unwrap(),
                    num_cross_baselines.try_into().unwrap(),
                    low_res_freqs_hz.len().try_into().unwrap(),
                );
                if !error_message_ptr.is_null() {
                    let error_message = CStr::from_ptr(error_message_ptr)
                        .to_str()
                        .unwrap_or("<cannot read CUDA error string>");
                    let our_error_str =
                        format!("{}:{}: add_model: {error_message}", file!(), line!());
                    return Err(CudaError::Kernel(our_error_str).into());
                }

                #[cfg(test)]
                {
                    let vis_residual_low_res = d_low_res_vis.copy_from_device_new().unwrap();
                    let vis_residual_low_res = Array3::from_shape_vec(
                        (1, low_res_freqs_hz.len(), num_cross_baselines),
                        vis_residual_low_res,
                    )
                    .unwrap();

                    let vis_model_low_res = low_res_modeller.d_vis.copy_from_device_new().unwrap();
                    let vis_model_low_res = Array3::from_shape_vec(
                        (1, low_res_freqs_hz.len(), num_cross_baselines),
                        vis_model_low_res,
                    )
                    .unwrap();

                    let uvws_low_res = d_uvws_source_low_res.copy_from_device_new()?;

                    for (vis_residual_low_res, vis_model_low_res, &lambda) in izip!(
                        vis_residual_low_res.slice(s![0, .., ..]).outer_iter(),
                        vis_model_low_res.slice(s![0, .., ..]).outer_iter(),
                        low_res_lambdas_m
                    ) {
                        for (residual, model, &cuda::UVW { u, v, w: _ }) in izip!(
                            vis_residual_low_res.iter(),
                            vis_model_low_res.iter(),
                            &uvws_low_res,
                        ) {
                            let residual_i = residual[0] + residual[3];
                            let model_i = model[0] + model[3];
                            let u = u / lambda as CudaFloat;
                            let v = v / lambda as CudaFloat;

                            println!("uv ({:+1.5}, {:+1.5}) l{:+1.3} | RI {:+1.5} @{:+1.5}pi | MI {:+1.5} @{:+1.5}pi", u, v, lambda, residual_i.norm(), residual_i.arg(), model_i.norm(), model_i.arg());
                        }
                    }
                }

                let error_message_ptr = cuda::iono_loop(
                    d_low_res_vis.get().cast(),
                    d_low_res_weights.get(),
                    low_res_modeller.d_vis.get().cast(),
                    d_low_res_model_rotated.get_mut().cast(),
                    d_iono_fits.get_mut().cast(),
                    d_iono_consts.get_mut(),
                    num_timesteps.try_into().unwrap(),
                    num_tiles.try_into().unwrap(),
                    num_cross_baselines.try_into().unwrap(),
                    low_res_freqs_hz.len().try_into().unwrap(),
                    10,
                    d_average_lmsts.get(),
                    d_uvws_source_low_res.get(),
                    d_low_res_lambdas.get(),
                );
                if !error_message_ptr.is_null() {
                    let error_message = CStr::from_ptr(error_message_ptr)
                        .to_str()
                        .unwrap_or("<cannot read CUDA error string>");
                    let our_error_str =
                        format!("{}:{}: iono_loop: {error_message}", file!(), line!());
                    return Err(CudaError::Kernel(our_error_str).into());
                }

                // dbg!(iono_consts);
                multi_progress_bar
                    .suspend(|| trace!("{:?}: iono_loop", std::time::Instant::now() - start));

                high_res_modeller.update_with_a_source(source, obs_context.phase_centre)?;
                // high_res_modeller.clear_vis();
                // Clear the old memory before reusing the buffer.
                cuda_runtime_sys::cudaMemset(
                    d_high_res_model.get_mut().cast(),
                    0,
                    timestamps.len()
                        * num_cross_baselines
                        * all_fine_chan_lambdas_m.len()
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
                                .add(i_time * num_cross_baselines * all_fine_chan_lambdas_m.len()),
                            d_uvws,
                            *lmst,
                            *latitude,
                        )
                    })?;
                multi_progress_bar
                    .suspend(|| trace!("{:?}: high res model", std::time::Instant::now() - start));

                // TODO: This is ugly.
                let mut tmp = [cuda::IonoConsts {
                    alpha: 0.0,
                    beta: 0.0,
                    s_vm: 0.0,
                    s_mm: 0.0,
                }];
                d_iono_consts.copy_from_device(&mut tmp)?;
                *iono_consts = IonoConsts {
                    alpha: tmp[0].alpha,
                    beta: tmp[0].beta,
                    s_vm: tmp[0].s_vm,
                    s_mm: tmp[0].s_mm,
                };

                // TODO: Check the iono consts. If they're not sensible, reset
                // them and don't subtract the shifted model.
                let error_message_ptr = cuda::subtract_iono(
                    d_high_res_vis.get_mut().cast(),
                    d_high_res_model.get().cast(),
                    d_iono_consts.get(),
                    d_old_iono_consts.get(),
                    d_uvws_to.get(),
                    d_lambdas.get(),
                    num_timesteps.try_into().unwrap(),
                    num_cross_baselines.try_into().unwrap(),
                    all_fine_chan_lambdas_m.len().try_into().unwrap(),
                );
                if !error_message_ptr.is_null() {
                    let error_message = CStr::from_ptr(error_message_ptr)
                        .to_str()
                        .unwrap_or("<cannot read CUDA error string>");
                    let our_error_str =
                        format!("{}:{}: subtract_iono: {error_message}", file!(), line!());
                    return Err(CudaError::Kernel(our_error_str).into());
                }

                multi_progress_bar
                    .suspend(|| trace!("{:?}: subtract_iono", std::time::Instant::now() - start));
                debug!("peel loop finished: {source_name} at {source_phase_centre} (has iono {iono_consts:?})");

                peel_progress.inc(1);
            }
        }

        // copy results back to host
        d_high_res_vis
            .copy_from_device(vis_residual.as_slice_mut().unwrap())
            .unwrap();
    }

    Ok(())
}

/// Just the W terms of [`UVW`] coordinates.
#[derive(Clone, Copy, Default, PartialEq, Debug)]
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

impl Neg for W {
    type Output = Self;

    fn neg(self) -> Self::Output {
        W(-self.0)
    }
}

#[cfg(test)]
impl approx::AbsDiffEq for W {
    type Epsilon = f64;

    fn default_epsilon() -> Self::Epsilon {
        f64::EPSILON
    }

    fn abs_diff_eq(&self, other: &Self, epsilon: Self::Epsilon) -> bool {
        f64::abs_diff_eq(&self.0, &other.0, epsilon)
    }
}

/// Just the U and V terms of [`UVW`] coordinates.
#[derive(Clone, Copy, Default, PartialEq, Debug)]
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

#[cfg(test)]
impl approx::AbsDiffEq for UV {
    type Epsilon = f64;

    fn default_epsilon() -> Self::Epsilon {
        f64::EPSILON
    }

    fn abs_diff_eq(&self, other: &Self, epsilon: Self::Epsilon) -> bool {
        f64::abs_diff_eq(&self.u, &other.u, epsilon) && f64::abs_diff_eq(&self.v, &other.v, epsilon)
    }
}
