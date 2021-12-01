// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Handling of calibration arguments.
//!
//! Strategy: Users give arguments to hyperdrive (handled by [calibrate::args]).
//! hyperdrive turns arguments into parameters (handled by [calibrate::params]).
//! Using this paradigm, the code to handle arguments and parameters (and
//! associated errors) can be neatly split.

use std::fs::File;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::str::FromStr;

use itertools::Itertools;
use log::debug;
use mwa_rust_core::constants::{MWA_LAT_RAD, MWA_LONG_RAD};
use serde::{Deserialize, Serialize};
use structopt::StructOpt;
use strum::IntoEnumIterator;
use strum_macros::{Display, EnumIter, EnumString};
use thiserror::Error;

use crate::{
    calibrate::{
        params::{CalibrateParams, InvalidArgsError},
        solutions::CalSolutionType,
    },
    data_formats::VisOutputType,
    pfb_gains::{DEFAULT_PFB_FLAVOUR, PFB_FLAVOURS},
    *,
};
use mwa_hyperdrive_srclist::{SOURCE_DIST_CUTOFF_HELP, VETO_THRESHOLD_HELP};

#[derive(Debug, Display, EnumIter, EnumString)]
enum ArgFileTypes {
    #[strum(serialize = "toml")]
    Toml,
    #[strum(serialize = "json")]
    Json,
}

lazy_static::lazy_static! {
    static ref ARG_FILE_TYPES_COMMA_SEPARATED: String = ArgFileTypes::iter().join(", ");

    pub(super) static ref VIS_OUTPUT_EXTENSIONS: String = VisOutputType::iter().join(", ");

    pub(super) static ref CAL_SOLUTION_EXTENSIONS: String = CalSolutionType::iter().join(", ");

    static ref DI_CALIBRATE_OUTPUT_HELP: String =
        format!("Paths to the calibration output files. Supported calibrated visibility outputs: {}. Supported calibration solution formats: {}. Default: {}", *VIS_OUTPUT_EXTENSIONS, *CAL_SOLUTION_EXTENSIONS, DEFAULT_OUTPUT_SOLUTIONS_FILENAME);

    static ref MODEL_FILENAME_HELP: String = format!("The path to the file where the generated sky-model visibilities are written. If this argument isn't supplied, then no file is written. Supported formats: {}", *VIS_OUTPUT_EXTENSIONS);

    static ref SOURCE_LIST_TYPE_HELP: String =
        format!("The type of sky-model source list. Valid types are: {}. If not specified, all types are attempted", *mwa_hyperdrive_srclist::SOURCE_LIST_TYPES_COMMA_SEPARATED);

    static ref PFB_FLAVOUR_HELP: String =
        format!("The 'flavour' of poly-phase filter bank corrections applied to raw MWA data. The default is '{}'. Valid flavours are: {}", DEFAULT_PFB_FLAVOUR, *PFB_FLAVOURS);

    static ref MAX_ITERATIONS_HELP: String =
        format!("The maximum number of times to iterate when performing \"MitchCal\". Default: {}", DEFAULT_MAX_ITERATIONS);

    static ref STOP_THRESHOLD_HELP: String =
        format!("The threshold at which we stop iterating when performing \"MitchCal\". Default: {:e}", DEFAULT_STOP_THRESHOLD);

    static ref MIN_THRESHOLD_HELP: String =
        format!("The minimum threshold to satisfy convergence when performing \"MitchCal\". Even when this threshold is exceeded, iteration will continue until max iterations or the stop threshold is reached. Default: {:e}", DEFAULT_MIN_THRESHOLD);

    static ref ARRAY_LONGITUDE_HELP: String =
        format!("The Earth longitude of the instrumental array [degrees]. Default (MWA): {}°", MWA_LONG_RAD.to_degrees());

    static ref ARRAY_LATITUDE_HELP: String =
        format!("The Earth latitude of the instrumental array [degrees]. Default (MWA): {}°", MWA_LAT_RAD.to_degrees());
}

// Arguments that are exposed to users. All arguments except bools should be
// optional.
//
// These are digested by hyperdrive and used to eventually populate
// [CalibrateParams], which is used throughout hyperdrive's calibration code.
#[derive(StructOpt, Debug, Clone, Default, Serialize, Deserialize)]
pub struct CalibrateUserArgs {
    /// Paths to input data files to be calibrated. These can include a metafits
    /// file, gpubox files, mwaf files, a measurement set and/or uvfits files.
    #[structopt(short, long)]
    pub data: Option<Vec<String>>,

    #[structopt(short, long, help = DI_CALIBRATE_OUTPUT_HELP.as_str())]
    pub outputs: Option<Vec<PathBuf>>,

    #[structopt(short, long, help = MODEL_FILENAME_HELP.as_str())]
    pub model_filename: Option<PathBuf>,

    /// Path to the sky-model source list file.
    #[structopt(short, long)]
    pub source_list: Option<String>,

    #[structopt(long, help = SOURCE_LIST_TYPE_HELP.as_str())]
    pub source_list_type: Option<String>,

    /// The number of sources to use in the source list. The default is to use
    /// them all. Example: If 1000 sources are specified here, then the top 1000
    /// sources are used (based on their flux densities after the beam
    /// attenuation) within the specified source distance cutoff.
    #[structopt(short, long)]
    pub num_sources: Option<usize>,

    #[structopt(long, help = SOURCE_DIST_CUTOFF_HELP.as_str())]
    pub source_dist_cutoff: Option<f64>,

    #[structopt(long, help = VETO_THRESHOLD_HELP.as_str())]
    pub veto_threshold: Option<f64>,

    /// The path to the HDF5 MWA FEE beam file. If not specified, this must be
    /// provided by the MWA_BEAM_FILE environment variable.
    #[structopt(long)]
    pub beam_file: Option<PathBuf>,

    /// Pretend that all MWA dipoles are alive and well, ignoring whatever is in
    /// the metafits file.
    #[structopt(long)]
    pub unity_dipole_gains: bool,

    /// If specified, use these dipole delays for the MWA pointing.
    #[structopt(long)]
    pub delays: Option<Vec<u32>>,

    /// Don't apply a beam response when generating a sky model. The default is
    /// to use the FEE beam.
    #[structopt(long)]
    pub no_beam: bool,

    /// The number of time samples to average together before calibrating. If
    /// this is 0, then all data are averaged together. Default: 0. e.g. If the
    /// input data is in 0.5s resolution and this variable was 4, then we
    /// produce calibration solutions for every 2s worth of data.
    #[structopt(short, long)]
    pub time_average_factor: Option<usize>,

    // /// The number of fine-frequency channels to average together before
    // /// calibrating. If this is 0, then all data is averaged together. Default:
    // /// 1
    // ///
    // /// e.g. If the input data is in 20kHz resolution and this variable was 2,
    // /// then we average 40kHz worth of data together during calibration.
    // #[structopt(short, long)]
    // pub freq_average_factor: Option<usize>,
    /// The timesteps to use from the input data. The timesteps will be
    /// ascendingly sorted for calibration. No duplicates are allowed. The
    /// default is to use all unflagged timesteps.
    #[structopt(long)]
    pub timesteps: Option<Vec<usize>>,

    /// Additional tiles to be flagged. These values correspond to either the
    /// values in the "Antenna" column of HDU 2 in the metafits file (e.g. 0 3
    /// 127), or the "TileName" (e.g. Tile011).
    #[structopt(long)]
    pub tile_flags: Option<Vec<String>>,

    /// If specified, pretend that all tiles are unflagged in the input data.
    #[structopt(long)]
    pub ignore_input_data_tile_flags: bool,

    /// If specified, pretend all fine channels in the input data are unflagged.
    #[structopt(long)]
    pub ignore_input_data_fine_channel_flags: bool,

    /// When writing out calibrated visibilities, don't include
    /// auto-correlations.
    #[structopt(long)]
    pub ignore_autos: bool,

    /// The fine channels to be flagged in each coarse channel. e.g. 0 1 16 30
    /// 31 are typical for 40 kHz data. If this is not specified, it defaults to
    /// flagging 80 kHz (or as close to this as possible) at the edges, as well
    /// as the centre channel for non-MWAX data.
    #[structopt(long)]
    pub fine_chan_flags_per_coarse_chan: Option<Vec<usize>>,

    /// The fine channels to be flagged across the whole observation band. e.g.
    /// 0 767 are the first and last fine channels for 40 kHz data.
    #[structopt(long)]
    pub fine_chan_flags: Option<Vec<usize>>,

    #[structopt(long, help = PFB_FLAVOUR_HELP.as_str())]
    pub pfb_flavour: Option<String>,

    /// When reading in raw MWA data, don't apply digital gains.
    #[structopt(long)]
    pub no_digital_gains: bool,

    /// When reading in raw MWA data, don't apply cable length corrections. Note
    /// that some data may have already had the correction applied before it was
    /// written.
    #[structopt(long)]
    pub no_cable_length_correction: bool,

    /// When reading in raw MWA data, don't apply geometric corrections. Note
    /// that some data may have already had the correction applied before it was
    /// written.
    #[structopt(long)]
    pub no_geometric_correction: bool,

    /// When writing out calibrated visibilities, average this many timesteps
    /// together. Also supports a target time resolution (e.g. 8s). The value
    /// must be a multiple of the input data's time resolution. The default is
    /// to preserve the input data's time resolution. e.g. If the input data is
    /// in 0.5s resolution and this variable is 4, then we average 2s worth of
    /// calibrated data together before writing the data out. If the variable is
    /// instead 4s, then 8 calibrated timesteps are averaged together before
    /// writing the data out.
    #[structopt(long)]
    pub output_vis_time_average: Option<String>,

    /// When writing out calibrated visibilities, average this many fine freq.
    /// channels together. Also supports a target freq. resolution (e.g. 80kHz).
    /// The value must be a multiple of the input data's freq. resolution. The
    /// default is to preserve the input data's freq. resolution. e.g. If the
    /// input data is in 40kHz resolution and this variable is 4, then we
    /// average 160kHz worth of calibrated data together before writing the data
    /// out. If the variable is instead 80kHz, then 2 calibrated fine freq.
    /// channels are averaged together before writing the data out.
    #[structopt(long)]
    pub output_vis_freq_average: Option<String>,

    #[structopt(long, help = MAX_ITERATIONS_HELP.as_str())]
    pub max_iterations: Option<usize>,

    #[structopt(long, help = STOP_THRESHOLD_HELP.as_str())]
    pub stop_thresh: Option<f64>,

    #[structopt(long, help = MIN_THRESHOLD_HELP.as_str())]
    pub min_thresh: Option<f64>,

    #[structopt(long = "array_longitude", help = ARRAY_LONGITUDE_HELP.as_str())]
    pub array_longitude_deg: Option<f64>,

    #[structopt(long = "array_latitude", help = ARRAY_LATITUDE_HELP.as_str())]
    pub array_latitude_deg: Option<f64>,

    #[cfg(feature = "cuda")]
    /// Use the CPU for visibility generation. This is deliberately made
    /// non-default because using a GPU is much faster.
    #[structopt(long)]
    pub cpu: bool,
}

impl CalibrateUserArgs {
    /// Both command-line and file arguments overlap in terms of what is
    /// available; this function consolidates everything that was specified into
    /// a single struct. Where applicable, it will prefer CLI parameters over
    /// those in the file.
    ///
    /// The argument to this function is the path to the arguments file.
    ///
    /// This function should only ever merge arguments, and not try to make
    /// sense of them.
    pub fn merge<T: AsRef<Path>>(self, arg_file: &T) -> Result<Self, CalibrateArgsError> {
        // Make it abundantly clear that "self" should be considered the
        // command-line arguments.
        let cli_args = self;

        // Read in the file arguments.
        let file_args: Self = {
            let file_args_path = PathBuf::from(arg_file.as_ref());
            debug!(
                "Attempting to parse argument file {} ...",
                file_args_path.display()
            );

            let mut contents = String::new();
            let file_args_extension = file_args_path
                .extension()
                .and_then(|e| e.to_str())
                .map(|e| e.to_lowercase())
                .and_then(|e| ArgFileTypes::from_str(&e).ok());
            match file_args_extension {
                Some(ArgFileTypes::Toml) => {
                    debug!("Parsing toml file...");
                    let mut fh = File::open(&arg_file)?;
                    fh.read_to_string(&mut contents)?;
                    match toml::from_str(&contents) {
                        Ok(p) => p,
                        Err(e) => {
                            return Err(CalibrateArgsError::TomlDecode {
                                file: file_args_path.display().to_string(),
                                err: e.to_string(),
                            })
                        }
                    }
                }

                Some(ArgFileTypes::Json) => {
                    debug!("Parsing json file...");
                    let mut fh = File::open(&arg_file)?;
                    fh.read_to_string(&mut contents)?;
                    match serde_json::from_str(&contents) {
                        Ok(p) => p,
                        Err(e) => {
                            return Err(CalibrateArgsError::JsonDecode {
                                file: file_args_path.display().to_string(),
                                err: e.to_string(),
                            })
                        }
                    }
                }

                _ => {
                    return Err(CalibrateArgsError::UnrecognisedArgFileExt(
                        file_args_path.display().to_string(),
                    ))
                }
            }
        };

        // Ensure all of the file args are accounted for by pattern matching.
        let Self {
            data,
            outputs,
            model_filename,
            beam_file,
            unity_dipole_gains,
            delays,
            no_beam,
            source_list,
            source_list_type,
            num_sources,
            source_dist_cutoff,
            veto_threshold,
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
            no_cable_length_correction,
            no_geometric_correction,
            output_vis_time_average,
            output_vis_freq_average,
            max_iterations,
            stop_thresh,
            min_thresh,
            array_longitude_deg,
            array_latitude_deg,
            #[cfg(feature = "cuda")]
            cpu,
        } = file_args;
        // Merge all the arguments, preferring the CLI args when available.
        Ok(Self {
            data: cli_args.data.or(data),
            outputs: cli_args.outputs.or(outputs),
            model_filename: cli_args.model_filename.or(model_filename),
            beam_file: cli_args.beam_file.or(beam_file),
            unity_dipole_gains: cli_args.unity_dipole_gains || unity_dipole_gains,
            no_beam: cli_args.no_beam || no_beam,
            source_list: cli_args.source_list.or(source_list),
            source_list_type: cli_args.source_list_type.or(source_list_type),
            num_sources: cli_args.num_sources.or(num_sources),
            source_dist_cutoff: cli_args.source_dist_cutoff.or(source_dist_cutoff),
            veto_threshold: cli_args.veto_threshold.or(veto_threshold),
            time_average_factor: cli_args.time_average_factor.or(time_average_factor),
            // freq_average_factor: cli_args.freq_average_factor.or(freq_average_factor),
            timesteps: cli_args.timesteps.or(timesteps),
            tile_flags: cli_args.tile_flags.or(tile_flags),
            ignore_input_data_tile_flags: cli_args.ignore_input_data_tile_flags
                || ignore_input_data_tile_flags,
            delays: cli_args.delays.or(delays),
            ignore_input_data_fine_channel_flags: cli_args.ignore_input_data_fine_channel_flags
                || ignore_input_data_fine_channel_flags,
            ignore_autos: cli_args.ignore_autos || ignore_autos,
            fine_chan_flags_per_coarse_chan: cli_args
                .fine_chan_flags_per_coarse_chan
                .or(fine_chan_flags_per_coarse_chan),
            fine_chan_flags: cli_args.fine_chan_flags.or(fine_chan_flags),
            pfb_flavour: cli_args.pfb_flavour.or(pfb_flavour),
            no_digital_gains: cli_args.no_digital_gains || no_digital_gains,
            no_cable_length_correction: cli_args.no_cable_length_correction
                || no_cable_length_correction,
            no_geometric_correction: cli_args.no_geometric_correction || no_geometric_correction,
            output_vis_time_average: cli_args.output_vis_time_average.or(output_vis_time_average),
            output_vis_freq_average: cli_args.output_vis_freq_average.or(output_vis_freq_average),
            max_iterations: cli_args.max_iterations.or(max_iterations),
            stop_thresh: cli_args.stop_thresh.or(stop_thresh),
            min_thresh: cli_args.min_thresh.or(min_thresh),
            array_longitude_deg: cli_args.array_longitude_deg.or(array_longitude_deg),
            array_latitude_deg: cli_args.array_latitude_deg.or(array_latitude_deg),
            #[cfg(feature = "cuda")]
            cpu: cli_args.cpu || cpu,
        })
    }

    pub fn into_params(self) -> Result<CalibrateParams, InvalidArgsError> {
        CalibrateParams::new(self)
    }
}

/// Errors associated with merging `CalibrateUserArgs` structs.
#[derive(Error, Debug)]
pub enum CalibrateArgsError {
    #[error("Argument file '{0}' doesn't have a recognised file extension! Valid extensions are: {}", *ARG_FILE_TYPES_COMMA_SEPARATED)]
    UnrecognisedArgFileExt(String),

    #[error("Couldn't decode toml structure from {file}:\n{err}")]
    TomlDecode { file: String, err: String },

    #[error("Couldn't decode json structure from {file}:\n{err}")]
    JsonDecode { file: String, err: String },

    #[error("IO error when trying to read argument file: {0}")]
    IO(#[from] std::io::Error),
}
