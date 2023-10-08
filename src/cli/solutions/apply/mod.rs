// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Given input data and a calibration solutions file, apply the solutions and
//! write out the calibrated visibilities.

#[cfg(test)]
mod tests;

use std::path::PathBuf;

use clap::Parser;
use log::{debug, info, trace};
use serde::{Deserialize, Serialize};

use crate::{
    cli::common::{display_warnings, InputVisArgs, OutputVisArgs, ARG_FILE_HELP},
    io::write::VIS_OUTPUT_EXTENSIONS,
    params::SolutionsApplyParams,
    solutions::CAL_SOLUTION_EXTENSIONS,
    HyperdriveError,
};

pub(crate) const DEFAULT_OUTPUT_VIS_FILENAME: &str = "hyperdrive_calibrated.uvfits";

lazy_static::lazy_static! {
    static ref SOLS_INPUT_HELP: String =
        format!("Path to the calibration solutions file to be applied. Supported formats: {}", *CAL_SOLUTION_EXTENSIONS);

    static ref OUTPUTS_HELP: String =
        format!("Paths to the output calibrated visibility files. Supported formats: {}. Default: {}", *VIS_OUTPUT_EXTENSIONS, DEFAULT_OUTPUT_VIS_FILENAME);
}

#[derive(Parser, Debug, Clone, Default, Serialize, Deserialize)]
pub(crate) struct SolutionsApplyArgs {
    #[clap(name = "ARGUMENTS_FILE", help = ARG_FILE_HELP.as_str(), parse(from_os_str))]
    args_file: Option<PathBuf>,

    #[clap(flatten)]
    #[serde(rename = "data")]
    #[serde(default)]
    data_args: InputVisArgs,

    /// Path to the calibration solutions file to be applied.
    #[clap(short, long, help = SOLS_INPUT_HELP.as_str(), help_heading = "INPUT DATA")]
    solutions: Option<String>,

    #[clap(
        short = 'o',
        long,
        multiple_values(true),
        help = OUTPUTS_HELP.as_str(),
        help_heading = "OUTPUT FILES"
    )]
    outputs: Option<Vec<PathBuf>>,

    /// When writing out visibilities, average this many timesteps together.
    /// Also supports a target time resolution (e.g. 8s). The value must be a
    /// multiple of the input data's time resolution. The default is no
    /// averaging, i.e. a value of 1. Examples: If the input data is in 0.5s
    /// resolution and this variable is 4, then we average 2s worth of data
    /// together before writing the data out. If the variable is instead 4s,
    /// then 8 timesteps are averaged together before writing the data out.
    #[clap(long, help_heading = "OUTPUT FILES")]
    output_vis_time_average: Option<String>,

    /// When writing out visibilities, average this many fine freq. channels
    /// together. Also supports a target freq. resolution (e.g. 80kHz). The
    /// value must be a multiple of the input data's freq. resolution. The
    /// default is no averaging, i.e. a value of 1. Examples: If the input data
    /// is in 40kHz resolution and this variable is 4, then we average 160kHz
    /// worth of data together before writing the data out. If the variable is
    /// instead 80kHz, then 2 fine freq. channels are averaged together before
    /// writing the data out.
    #[clap(long, help_heading = "OUTPUT FILES")]
    output_vis_freq_average: Option<String>,

    /// Don't write auto-correlatsions to the output visibilities.
    #[clap(long, help_heading = "OUTPUT FILES")]
    output_no_autos: bool,

    /// Rather than writing out the entire input bandwidth, write out only the
    /// smallest contiguous band. e.g. Typical 40 kHz MWA data has 768 channels,
    /// but the first 2 and last 2 channels are usually flagged. Turning this
    /// option on means that 764 channels would be written out instead of 768.
    /// Note that other flagged channels in the band are unaffected, because the
    /// data written out must be contiguous.
    #[clap(long, help_heading = "OUTPUT FILES")]
    #[serde(default)]
    output_smallest_contiguous_band: bool,
}

impl SolutionsApplyArgs {
    /// Both command-line and file arguments overlap in terms of what is
    /// available; this function consolidates everything that was specified into
    /// a single struct. Where applicable, it will prefer CLI parameters over
    /// those in the file.
    ///
    /// The argument to this function is the path to the arguments file.
    ///
    /// This function should only ever merge arguments, and not try to make
    /// sense of them.
    pub(crate) fn merge(self) -> Result<SolutionsApplyArgs, HyperdriveError> {
        debug!("Merging command-line arguments with the argument file");

        let cli_args = self;

        if let Some(arg_file) = cli_args.args_file {
            // Read in the file arguments. Ensure all of the file args are
            // accounted for by pattern matching.
            let SolutionsApplyArgs {
                args_file: _,
                data_args,
                solutions,
                outputs,
                output_vis_time_average,
                output_vis_freq_average,
                output_no_autos,
                output_smallest_contiguous_band,
            } = unpack_arg_file!(arg_file);

            // Merge all the arguments, preferring the CLI args when available.
            Ok(SolutionsApplyArgs {
                args_file: None,
                data_args: cli_args.data_args.merge(data_args),
                solutions: cli_args.solutions.or(solutions),
                outputs: cli_args.outputs.or(outputs),
                output_vis_time_average: cli_args
                    .output_vis_time_average
                    .or(output_vis_time_average),
                output_vis_freq_average: cli_args
                    .output_vis_freq_average
                    .or(output_vis_freq_average),
                output_no_autos: cli_args.output_no_autos || output_no_autos,
                output_smallest_contiguous_band: cli_args.output_smallest_contiguous_band
                    || output_smallest_contiguous_band,
            })
        } else {
            Ok(cli_args)
        }
    }

    fn parse(self) -> Result<SolutionsApplyParams, HyperdriveError> {
        debug!("{:#?}", self);

        let Self {
            args_file: _,
            mut data_args,
            solutions,
            outputs,
            output_vis_time_average,
            output_vis_freq_average,
            output_no_autos,
            output_smallest_contiguous_band,
        } = self;

        match (solutions, data_args.files.as_mut()) {
            // Add the user-specified solutions to the file list.
            (Some(s), Some(f)) => f.push(s),

            // No solutions specified to solutions-apply; if no solutions were
            // given to data_args, then we'll need to complain.
            (None, _) => (),

            // Solutions were given, but no data_args. Well, we need
            // visibilities, so parsing data_args will fail.
            (Some(_), None) => (),
        }

        let input_vis_params = data_args.parse("Applying solutions")?;
        if input_vis_params.solutions.is_none() {
            return Err(SolutionsApplyArgsError::NoSolutions.into());
        }

        let output_vis_params = OutputVisArgs {
            outputs,
            output_vis_time_average,
            output_vis_freq_average,
            output_autos: input_vis_params.using_autos && !output_no_autos,
        }
        .parse(
            input_vis_params.time_res,
            input_vis_params.spw.freq_res,
            &input_vis_params.timeblocks.mapped_ref(|tb| tb.median),
            output_smallest_contiguous_band,
            DEFAULT_OUTPUT_VIS_FILENAME,
            Some("calibrated"),
        )?;

        display_warnings();

        Ok(SolutionsApplyParams {
            input_vis_params,
            output_vis_params,
        })
    }

    pub(crate) fn run(self, dry_run: bool) -> Result<(), HyperdriveError> {
        debug!("Converting arguments into parameters");
        trace!("{:#?}", self);
        let params = self.parse()?;

        if dry_run {
            info!("Dry run -- exiting now.");
            return Ok(());
        }

        params.run()?;
        Ok(())
    }
}

#[derive(thiserror::Error, Debug)]
pub(crate) enum SolutionsApplyArgsError {
    #[error("No calibration solutions were supplied")]
    NoSolutions,
}
