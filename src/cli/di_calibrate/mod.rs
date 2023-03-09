// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Handling of calibration arguments.
//!
//! Strategy: Users give arguments to hyperdrive (handled by [calibrate::args]).
//! hyperdrive turns arguments into parameters (handled by [calibrate::params]).
//! Using this paradigm, the code to handle arguments and parameters (and
//! associated errors) can be neatly split.

mod error;
mod params;
#[cfg(test)]
pub(crate) mod tests;

pub(crate) use error::DiCalArgsError;
pub(crate) use params::DiCalParams;

use std::{fs::File, io::Read, path::PathBuf, str::FromStr};

use clap::Parser;
use itertools::Itertools;
use log::{debug, info, trace};
use serde::{Deserialize, Serialize};
use strum::IntoEnumIterator;
use strum_macros::{Display, EnumIter, EnumString};

use crate::{
    di_calibrate::DiCalibrateError,
    help_texts::*,
    io::write::VIS_OUTPUT_EXTENSIONS,
    solutions::{self, CalSolutionType, CalibrationSolutions, CAL_SOLUTION_EXTENSIONS},
    unit_parsing::WAVELENGTH_FORMATS,
    HyperdriveError,
};

#[derive(Debug, Display, EnumIter, EnumString)]
enum ArgFileTypes {
    #[strum(serialize = "toml")]
    Toml,
    #[strum(serialize = "json")]
    Json,
}

// The default minimum baseline cutoff.
pub(crate) const DEFAULT_UVW_MIN: &str = "50λ";

/// The maximum number of times to iterate when performing "MitchCal" in
/// direction-independent calibration.
pub(crate) const DEFAULT_MAX_ITERATIONS: u32 = 50;

/// The threshold to satisfy convergence when performing "MitchCal" in
/// direction-independent calibration.
pub(crate) const DEFAULT_STOP_THRESHOLD: f64 = 1e-8;

/// The minimum threshold to satisfy convergence when performing "MitchCal" in
/// direction-independent calibration. Reaching this threshold counts as
/// "converged", but it's not as good as the stop threshold.
pub(crate) const DEFAULT_MIN_THRESHOLD: f64 = 1e-4;

pub(crate) const DEFAULT_OUTPUT_SOLUTIONS_FILENAME: &str = "hyperdrive_solutions.fits";

lazy_static::lazy_static! {
    static ref ARG_FILE_TYPES_COMMA_SEPARATED: String = ArgFileTypes::iter().join(", ");

    static ref ARG_FILE_HELP: String =
        format!("All of the arguments to di-calibrate may be specified in a file. Any CLI arguments override parameters set in the file. Supported formats: {}", *ARG_FILE_TYPES_COMMA_SEPARATED);

    static ref OUTPUTS_HELP: String =
        format!("Paths to the output calibration solution files. Supported formats: {}. Default: {}", *CAL_SOLUTION_EXTENSIONS, DEFAULT_OUTPUT_SOLUTIONS_FILENAME);

    static ref MODEL_FILENAME_HELP: String =
        format!("The paths to the files where the generated sky-model visibilities are written. If this argument isn't supplied, then no file is written. Supported formats: {}", *VIS_OUTPUT_EXTENSIONS);

    static ref UVW_MIN_HELP: String =
        format!("The minimum UVW length to use. This value must have a unit annotated. Allowed units: {}. Default: {}", *WAVELENGTH_FORMATS, DEFAULT_UVW_MIN);

    static ref UVW_MAX_HELP: String =
        format!("The maximum UVW length to use. This value must have a unit annotated. Allowed units: {}. No default.", *WAVELENGTH_FORMATS);

    static ref MAX_ITERATIONS_HELP: String =
        format!("The maximum number of times to iterate when performing \"MitchCal\". Default: {DEFAULT_MAX_ITERATIONS}");

    static ref STOP_THRESHOLD_HELP: String =
        format!("The threshold at which we stop iterating when performing \"MitchCal\". Default: {DEFAULT_STOP_THRESHOLD:e}");

    static ref MIN_THRESHOLD_HELP: String =
        format!("The minimum threshold to satisfy convergence when performing \"MitchCal\". Even when this threshold is exceeded, iteration will continue until max iterations or the stop threshold is reached. Default: {DEFAULT_MIN_THRESHOLD:e}");
}

#[derive(Parser, Debug, Clone, Default, Serialize, Deserialize)]
pub struct DiCalArgs {
    #[clap(name = "ARGUMENTS_FILE", help = ARG_FILE_HELP.as_str(), parse(from_os_str))]
    pub args_file: Option<PathBuf>,

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

    #[clap(short, long, multiple_values(true), help = OUTPUTS_HELP.as_str(), help_heading = "OUTPUT FILES")]
    pub outputs: Option<Vec<PathBuf>>,

    #[clap(short, long, multiple_values(true), help = MODEL_FILENAME_HELP.as_str(), help_heading = "OUTPUT FILES")]
    pub model_filenames: Option<Vec<PathBuf>>,

    /// When writing out model visibilities, average this many timesteps
    /// together. Also supports a target time resolution (e.g. 8s). The value
    /// must be a multiple of the input data's time resolution. The default is
    /// to preserve the input data's time resolution. e.g. If the input data is
    /// in 0.5s resolution and this variable is 4, then we average 2s worth of
    /// model data together before writing the data out. If the variable is
    /// instead 4s, then 8 model timesteps are averaged together before writing
    /// the data out.
    #[clap(long, help_heading = "OUTPUT FILES")]
    pub output_model_time_average: Option<String>,

    /// When writing out model visibilities, average this many fine freq.
    /// channels together. Also supports a target freq. resolution (e.g. 80kHz).
    /// The value must be a multiple of the input data's freq. resolution. The
    /// default is to preserve the input data's freq. resolution multiplied by
    /// the frequency average factor. e.g. If the input data is in 40kHz
    /// resolution, the frequency average factor is 2 and this variable is 4,
    /// then we average 320kHz worth of model data together before writing the
    /// data out. If the variable is instead 80kHz, then 4 model fine freq.
    /// channels are averaged together before writing the data out.
    #[clap(long, help_heading = "OUTPUT FILES")]
    pub output_model_freq_average: Option<String>,

    /// The number of sources to use in the source list. The default is to use
    /// them all. Example: If 1000 sources are specified here, then the top 1000
    /// sources are used (based on their flux densities after the beam
    /// attenuation) within the specified source distance cutoff.
    #[clap(short, long, help_heading = "SKY-MODEL SOURCES")]
    pub num_sources: Option<usize>,

    #[clap(long, help = SOURCE_DIST_CUTOFF_HELP.as_str(), help_heading = "SKY-MODEL SOURCES")]
    pub source_dist_cutoff: Option<f64>,

    #[clap(long, help = VETO_THRESHOLD_HELP.as_str(), help_heading = "SKY-MODEL SOURCES")]
    pub veto_threshold: Option<f64>,

    /// The path to the HDF5 MWA FEE beam file. If not specified, this must be
    /// provided by the MWA_BEAM_FILE environment variable.
    #[clap(long, help_heading = "BEAM")]
    pub beam_file: Option<PathBuf>,

    /// Pretend that all MWA dipoles are alive and well, ignoring whatever is in
    /// the metafits file.
    #[clap(long, help_heading = "BEAM")]
    pub unity_dipole_gains: bool,

    #[clap(long, multiple_values(true), help = DIPOLE_DELAYS_HELP.as_str(), help_heading = "BEAM")]
    pub delays: Option<Vec<u32>>,

    /// Don't apply a beam response when generating a sky model. The default is
    /// to use the FEE beam.
    #[clap(long, help_heading = "BEAM")]
    pub no_beam: bool,

    /// The number of timesteps to average together during calibration. Also
    /// supports a target time resolution (e.g. 8s). If this is 0, then all data
    /// are averaged together. Default: 0. e.g. If this variable is 4, then we
    /// produce calibration solutions in timeblocks with up to 4 timesteps each.
    /// If the variable is instead 4s, then each timeblock contains up to 4s
    /// worth of data.
    #[clap(short, long, help_heading = "CALIBRATION")]
    pub timesteps_per_timeblock: Option<String>,

    /// The number of fine-frequency channels to average together before
    /// calibration. If this is 0, then all data is averaged together. Default:
    /// 1. e.g. If the input data is in 20kHz resolution and this variable was
    /// 2, then we average 40kHz worth of data into a chanblock before
    /// calibration. If the variable is instead 40kHz, then each chanblock
    /// contains up to 40kHz worth of data.
    #[clap(short, long, help_heading = "CALIBRATION")]
    pub freq_average_factor: Option<String>,

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

    #[clap(long, help = MAX_ITERATIONS_HELP.as_str(), help_heading = "CALIBRATION")]
    pub max_iterations: Option<u32>,

    #[clap(long, help = STOP_THRESHOLD_HELP.as_str(), help_heading = "CALIBRATION")]
    pub stop_thresh: Option<f64>,

    #[clap(long, help = MIN_THRESHOLD_HELP.as_str(), help_heading = "CALIBRATION")]
    pub min_thresh: Option<f64>,

    #[clap(
        long, help = ARRAY_POSITION_HELP.as_str(), help_heading = "CALIBRATION",
        number_of_values = 3,
        allow_hyphen_values = true,
        value_names = &["LONG_DEG", "LAT_DEG", "HEIGHT_M"]
    )]
    pub array_position: Option<Vec<f64>>,

    /// If specified, don't precess the array to J2000. We assume that sky-model
    /// sources are specified in the J2000 epoch.
    #[clap(long, help_heading = "CALIBRATION")]
    pub no_precession: bool,

    #[cfg(feature = "cuda")]
    /// Use the CPU for visibility generation. This is deliberately made
    /// non-default because using a GPU is much faster.
    #[clap(long, help_heading = "CALIBRATION")]
    pub cpu: bool,

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

impl DiCalArgs {
    /// Both command-line and file arguments overlap in terms of what is
    /// available; this function consolidates everything that was specified into
    /// a single struct. Where applicable, it will prefer CLI parameters over
    /// those in the file.
    ///
    /// The argument to this function is the path to the arguments file.
    ///
    /// This function should only ever merge arguments, and not try to make
    /// sense of them.
    pub(crate) fn merge(self) -> Result<DiCalArgs, DiCalArgsError> {
        let cli_args = self;

        if let Some(arg_file) = cli_args.args_file {
            // Read in the file arguments.
            let file_args: DiCalArgs = {
                debug!(
                    "Attempting to parse argument file {} ...",
                    arg_file.display()
                );

                let mut contents = String::new();
                let file_args_extension = arg_file
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
                                return Err(DiCalArgsError::TomlDecode {
                                    file: arg_file.display().to_string(),
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
                                return Err(DiCalArgsError::JsonDecode {
                                    file: arg_file.display().to_string(),
                                    err: e.to_string(),
                                })
                            }
                        }
                    }

                    _ => {
                        return Err(DiCalArgsError::UnrecognisedArgFileExt(
                            arg_file.display().to_string(),
                        ))
                    }
                }
            };

            // Ensure all of the file args are accounted for by pattern
            // matching.
            let DiCalArgs {
                args_file: _,
                data,
                source_list,
                source_list_type,
                ms_data_column_name,
                ignore_dut1,
                outputs,
                model_filenames,
                output_model_time_average,
                output_model_freq_average,
                num_sources,
                source_dist_cutoff,
                veto_threshold,
                beam_file,
                unity_dipole_gains,
                delays,
                no_beam,
                timesteps_per_timeblock,
                freq_average_factor,
                timesteps,
                use_all_timesteps,
                uvw_min,
                uvw_max,
                max_iterations,
                stop_thresh,
                min_thresh,
                array_position,
                no_precession,
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
            } = file_args;
            // Merge all the arguments, preferring the CLI args when available.
            Ok(DiCalArgs {
                args_file: None,
                data: cli_args.data.or(data),
                source_list: cli_args.source_list.or(source_list),
                source_list_type: cli_args.source_list_type.or(source_list_type),
                ms_data_column_name: cli_args.ms_data_column_name.or(ms_data_column_name),
                ignore_dut1: cli_args.ignore_dut1 || ignore_dut1,
                outputs: cli_args.outputs.or(outputs),
                model_filenames: cli_args.model_filenames.or(model_filenames),
                output_model_time_average: cli_args
                    .output_model_time_average
                    .or(output_model_time_average),
                output_model_freq_average: cli_args
                    .output_model_freq_average
                    .or(output_model_freq_average),
                num_sources: cli_args.num_sources.or(num_sources),
                source_dist_cutoff: cli_args.source_dist_cutoff.or(source_dist_cutoff),
                veto_threshold: cli_args.veto_threshold.or(veto_threshold),
                beam_file: cli_args.beam_file.or(beam_file),
                unity_dipole_gains: cli_args.unity_dipole_gains || unity_dipole_gains,
                delays: cli_args.delays.or(delays),
                no_beam: cli_args.no_beam || no_beam,
                timesteps_per_timeblock: cli_args
                    .timesteps_per_timeblock
                    .or(timesteps_per_timeblock),
                freq_average_factor: cli_args.freq_average_factor.or(freq_average_factor),
                timesteps: cli_args.timesteps.or(timesteps),
                use_all_timesteps: cli_args.use_all_timesteps || use_all_timesteps,
                uvw_min: cli_args.uvw_min.or(uvw_min),
                uvw_max: cli_args.uvw_max.or(uvw_max),
                max_iterations: cli_args.max_iterations.or(max_iterations),
                stop_thresh: cli_args.stop_thresh.or(stop_thresh),
                min_thresh: cli_args.min_thresh.or(min_thresh),
                array_position: cli_args.array_position.or(array_position),
                no_precession: cli_args.no_precession || no_precession,
                #[cfg(feature = "cuda")]
                cpu: cli_args.cpu || cpu,
                tile_flags: cli_args.tile_flags.or(tile_flags),
                ignore_input_data_tile_flags: cli_args.ignore_input_data_tile_flags
                    || ignore_input_data_tile_flags,
                ignore_input_data_fine_channel_flags: cli_args.ignore_input_data_fine_channel_flags
                    || ignore_input_data_fine_channel_flags,
                fine_chan_flags_per_coarse_chan: cli_args
                    .fine_chan_flags_per_coarse_chan
                    .or(fine_chan_flags_per_coarse_chan),
                fine_chan_flags: cli_args.fine_chan_flags.or(fine_chan_flags),
                pfb_flavour: cli_args.pfb_flavour.or(pfb_flavour),
                no_digital_gains: cli_args.no_digital_gains || no_digital_gains,
                no_cable_length_correction: cli_args.no_cable_length_correction
                    || no_cable_length_correction,
                no_geometric_correction: cli_args.no_geometric_correction
                    || no_geometric_correction,
                no_progress_bars: cli_args.no_progress_bars || no_progress_bars,
            })
        } else {
            Ok(cli_args)
        }
    }

    pub(crate) fn into_params(self) -> Result<DiCalParams, DiCalArgsError> {
        DiCalParams::new(self)
    }

    pub fn run(self, dry_run: bool) -> Result<Option<CalibrationSolutions>, HyperdriveError> {
        let args = if self.args_file.is_some() {
            trace!("Merging command-line arguments with the argument file");
            self.merge().map_err(DiCalibrateError::from)?
        } else {
            self
        };

        debug!("{:#?}", &args);
        trace!("Converting arguments into calibration parameters");
        let parameters = args.into_params()?;

        if dry_run {
            info!("Dry run -- exiting now.");
            return Ok(None);
        }

        let sols = parameters.calibrate()?;

        // Write out the solutions.
        if parameters.output_solutions_filenames.len() == 1 {
            let (sol_type, file) = &parameters.output_solutions_filenames[0];
            match sol_type {
                CalSolutionType::Fits => solutions::hyperdrive::write(&sols, file)?,
                CalSolutionType::Bin => solutions::ao::write(&sols, file)?,
            }
            info!("Calibration solutions written to {}", file.display());
        } else {
            for (i, (sol_type, file)) in parameters
                .output_solutions_filenames
                .into_iter()
                .enumerate()
            {
                match sol_type {
                    CalSolutionType::Fits => solutions::hyperdrive::write(&sols, &file)?,
                    CalSolutionType::Bin => solutions::ao::write(&sols, &file)?,
                }
                if i == 0 {
                    info!("Calibration solutions written to:");
                }
                info!("  {}", file.display());
            }
        }

        Ok(Some(sols))
    }
}
