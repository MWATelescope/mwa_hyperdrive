// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Handling of calibration arguments.
//!
//! Strategy: Users give arguments to hyperdrive (handled by calibrate::args).
//! hyperdrive turns arguments into parameters (handled by calibrate::params).
//! Using this paradigm, the code to handle arguments and parameters (and
//! associated errors) can be neatly split.

use std::fs::File;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::str::FromStr;

use itertools::Itertools;
use log::debug;
use serde::{Deserialize, Serialize};
use structopt::StructOpt;
use strum::IntoEnumIterator;
use strum_macros::{Display, EnumIter, EnumString};
use thiserror::Error;

use crate::calibrate::params::{CalibrateParams, InvalidArgsError};
use crate::*;
use mwa_hyperdrive_core::mwalib::{MWA_LATITUDE_RADIANS, MWA_LONGITUDE_RADIANS};
use mwa_hyperdrive_srclist::{SOURCE_DIST_CUTOFF_HELP, VETO_THRESHOLD_HELP};

#[derive(Display, EnumIter, EnumString)]
enum ArgFileTypes {
    #[strum(serialize = "toml")]
    Toml,
    #[strum(serialize = "json")]
    Json,
}

lazy_static::lazy_static! {
    pub(crate) static ref ARG_FILE_TYPES_COMMA_SEPARATED: String = ArgFileTypes::iter().join(", ");

    static ref OUTPUT_SOLUTIONS_HELP: String =
        format!("The path to the file where the calibration solutions are written. Supported formats are .fits and .bin (which is the \"André calibrate format\"). Default: {}", DEFAULT_OUTPUT_SOLUTIONS_FILENAME);

    static ref SOURCE_LIST_TYPE_HELP: String =
        format!(r#"The type of sky-model source list. Valid types are: {}

If not specified, the program will try all types"#, *mwa_hyperdrive_srclist::SOURCE_LIST_TYPES_COMMA_SEPARATED);

    static ref MAX_ITERATIONS_HELP: String =
        format!("The maximum number of times to iterate when performing \"MitchCal\". Default: {}", DEFAULT_MAX_ITERATIONS);

    static ref STOP_THRESHOLD_HELP: String =
        format!("The threshold at which we stop convergence when performing \"MitchCal\". Default: {:e}", DEFAULT_STOP_THRESHOLD);

    static ref MIN_THRESHOLD_HELP: String =
        format!("The minimum threshold to satisfy convergence when performing \"MitchCal\". Reaching this threshold counts as converged, but iteration will continue until max iterations or the stop threshold is reached. Default: {:e}", DEFAULT_MIN_THRESHOLD);

    static ref ARRAY_LONGITUDE_HELP: String =
        format!("The Earth longitude of the instrumental array [degrees]. Default (MWA): {}", MWA_LONGITUDE_RADIANS.to_degrees());

    static ref ARRAY_LATITUDE_HELP: String =
        format!("The Earth latitude of the instrumental array [degrees]. Default (MWA): {}", MWA_LATITUDE_RADIANS.to_degrees());
}

// Arguments that are exposed to users. All arguments except bools should be
// optional.
//
// These are digested by hyperdrive and used to eventually populate
// `CalibrateParams`, which is used throughout hyperdrive's calibrate.
#[derive(StructOpt, Debug, Default, Serialize, Deserialize)]
pub struct CalibrateUserArgs {
    /// Paths to input data files to be calibrated. These can include a metafits
    /// file, gpubox files, mwaf files, a measurement set and/or uvfits files.
    #[structopt(short, long)]
    pub data: Option<Vec<String>>,

    #[structopt(short, long, help = OUTPUT_SOLUTIONS_HELP.as_str())]
    pub output_solutions_filename: Option<String>,

    /// The path to the file where the generated sky-model visibilities are
    /// written. Only uvfits is currently supported. If this argument isn't
    /// supplied, then no file is written.
    #[structopt(short, long)]
    pub model_filename: Option<String>,

    /// Path to the sky-model source list file.
    #[structopt(short, long)]
    pub source_list: Option<String>,

    #[structopt(long, help = SOURCE_LIST_TYPE_HELP.as_str())]
    pub source_list_type: Option<String>,

    /// The number of sources to use in the source list. The default is to use
    /// them all.
    ///
    /// Example: If 1000 sources are specified here, then the top 1000 sources
    /// are used (based on their flux densities after the beam attenuation)
    /// within the specified source distance cutoff.
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

    /// Don't apply a beam response when generating a sky model. The default is
    /// to use the FEE beam.
    #[structopt(long)]
    pub no_beam: bool,

    // /// The calibration time resolution [seconds]. This must be a multiple of
    // /// the input data's native time resolution. If not supplied, then the
    // /// observation's native time resolution is used.
    // #[structopt(short, long)]
    // pub time_res: Option<f64>,

    // /// The calibration fine-channel frequency resolution [Hz]. This must be a
    // /// multiple of the input data's native frequency resolution. If not
    // /// supplied, then the observation's native frequency resolution is used.
    // #[structopt(short, long)]
    // pub freq_res: Option<f64>,
    /// The timesteps to use from the input data. The timesteps will be
    /// ascendingly sorted for calibration. No duplicates are allowed. The
    /// default is to use all unflagged timesteps.
    #[structopt(long)]
    pub timesteps: Option<Vec<usize>>,

    /// Additional tiles to be flagged. These values correspond to values in the
    /// "Antenna" column of HDU 2 in the metafits file, e.g. 0 3 127. These
    /// values should also be the same as FHD tile flags.
    #[structopt(long)]
    pub tile_flags: Option<Vec<usize>>,

    /// If specified, pretend that all tiles are unflagged in the input data.
    #[structopt(long)]
    pub ignore_input_data_tile_flags: bool,

    /// If specified, use these dipole delays for the MWA pointing.
    #[structopt(long)]
    pub delays: Option<Vec<u32>>,

    /// If specified, pretend all fine channels in the input data are unflagged.
    #[structopt(long)]
    pub ignore_input_data_fine_channels_flags: bool,

    /// The fine channels to be flagged in each coarse channel. e.g. 0 1 16 30
    /// 31 are typical for 40 kHz data.
    ///
    /// If this is not specified, it defaults to flagging the centre channel, as
    /// well as 80 kHz (or as close to this as possible) at the edges.
    #[structopt(long)]
    pub fine_chan_flags_per_coarse_chan: Option<Vec<usize>>,

    /// The fine channels to be flagged across the whole observation band. e.g.
    /// 0 767 are the first and last fine channels for 40 kHz data.
    #[structopt(long)]
    pub fine_chan_flags: Option<Vec<usize>>,

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
            delays,
            ignore_input_data_fine_channels_flags,
            fine_chan_flags_per_coarse_chan,
            fine_chan_flags,
            max_iterations,
            stop_thresh,
            min_thresh,
            array_longitude_deg,
            array_latitude_deg,
        } = file_args;
        // Merge all the arguments, preferring the CLI args when available.
        Ok(Self {
            data: cli_args.data.or(data),
            output_solutions_filename: cli_args
                .output_solutions_filename
                .or(output_solutions_filename),
            model_filename: cli_args.model_filename.or(model_filename),
            beam_file: cli_args.beam_file.or(beam_file),
            no_beam: cli_args.no_beam || no_beam,
            source_list: cli_args.source_list.or(source_list),
            source_list_type: cli_args.source_list_type.or(source_list_type),
            num_sources: cli_args.num_sources.or(num_sources),
            source_dist_cutoff: cli_args.source_dist_cutoff.or(source_dist_cutoff),
            veto_threshold: cli_args.veto_threshold.or(veto_threshold),
            // time_res: cli_args.time_res.or(time_res),
            // freq_res: cli_args.freq_res.or(freq_res),
            timesteps: cli_args.timesteps.or(timesteps),
            tile_flags: cli_args.tile_flags.or(tile_flags),
            ignore_input_data_tile_flags: cli_args.ignore_input_data_tile_flags
                || ignore_input_data_tile_flags,
            delays: cli_args.delays.or(delays),
            ignore_input_data_fine_channels_flags: cli_args.ignore_input_data_fine_channels_flags
                || ignore_input_data_fine_channels_flags,
            fine_chan_flags_per_coarse_chan: cli_args
                .fine_chan_flags_per_coarse_chan
                .or(fine_chan_flags_per_coarse_chan),
            fine_chan_flags: cli_args.fine_chan_flags.or(fine_chan_flags),
            max_iterations: cli_args.max_iterations.or(max_iterations),
            stop_thresh: cli_args.stop_thresh.or(stop_thresh),
            min_thresh: cli_args.min_thresh.or(min_thresh),
            array_longitude_deg: cli_args.array_longitude_deg.or(array_longitude_deg),
            array_latitude_deg: cli_args.array_latitude_deg.or(array_latitude_deg),
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
