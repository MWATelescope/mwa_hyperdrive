// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

#[cfg(test)]
mod tests;

use std::path::PathBuf;

use clap::Parser;
use log::{debug, info, trace};
use serde::{Deserialize, Serialize};

use super::common::{InputVisArgs, OutputVisArgs, ARG_FILE_HELP};
use crate::{
    cli::common::display_warnings, io::write::VIS_OUTPUT_EXTENSIONS, params::VisConvertParams,
    HyperdriveError,
};

lazy_static::lazy_static! {
    static ref OUTPUTS_HELP: String =
        format!("Paths to the output visibility files. Supported formats: {}", *VIS_OUTPUT_EXTENSIONS);
}

#[derive(Parser, Debug, Clone, Default, Serialize, Deserialize)]
pub(super) struct VisConvertArgs {
    #[clap(name = "ARGUMENTS_FILE", help = ARG_FILE_HELP.as_str(), parse(from_os_str))]
    pub(super) args_file: Option<PathBuf>,

    #[clap(flatten)]
    #[serde(rename = "data")]
    #[serde(default)]
    pub(super) data_args: InputVisArgs,

    #[clap(
        short = 'o',
        long,
        multiple_values(true),
        help = OUTPUTS_HELP.as_str(),
        help_heading = "OUTPUT FILES"
    )]
    pub(super) outputs: Option<Vec<PathBuf>>,

    /// When writing out visibilities, average this many timesteps together.
    /// Also supports a target time resolution (e.g. 8s). The value must be a
    /// multiple of the input data's time resolution. The default is no
    /// averaging, i.e. a value of 1. Examples: If the input data is in 0.5s
    /// resolution and this variable is 4, then we average 2s worth of data
    /// together before writing the data out. If the variable is instead 4s,
    /// then 8 timesteps are averaged together before writing the data out.
    #[clap(long, help_heading = "OUTPUT FILES")]
    pub(super) output_vis_time_average: Option<String>,

    /// When writing out visibilities, average this many fine freq. channels
    /// together. Also supports a target freq. resolution (e.g. 80kHz). The
    /// value must be a multiple of the input data's freq. resolution. The
    /// default is no averaging, i.e. a value of 1. Examples: If the input data
    /// is in 40kHz resolution and this variable is 4, then we average 160kHz
    /// worth of data together before writing the data out. If the variable is
    /// instead 80kHz, then 2 fine freq. channels are averaged together before
    /// writing the data out.
    #[clap(long, help_heading = "OUTPUT FILES")]
    pub(super) output_vis_freq_average: Option<String>,

    /// Rather than writing out the entire input bandwidth, write out only the
    /// smallest contiguous band. e.g. Typical 40 kHz MWA data has 768 channels,
    /// but the first 2 and last 2 channels are usually flagged. Turning this
    /// option on means that 764 channels would be written out instead of 768.
    /// Note that other flagged channels in the band are unaffected, because the
    /// data written out must be contiguous.
    #[clap(long, help_heading = "OUTPUT FILES")]
    #[serde(default)]
    pub(super) output_smallest_contiguous_band: bool,
}

impl VisConvertArgs {
    /// Both command-line and file arguments overlap in terms of what is
    /// available; this function consolidates everything that was specified into
    /// a single struct. Where applicable, it will prefer CLI parameters over
    /// those in the file.
    ///
    /// The argument to this function is the path to the arguments file.
    ///
    /// This function should only ever merge arguments, and not try to make
    /// sense of them.
    pub(super) fn merge(self) -> Result<VisConvertArgs, HyperdriveError> {
        debug!("Merging command-line arguments with the argument file");

        let cli_args = self;

        if let Some(arg_file) = cli_args.args_file {
            // Read in the file arguments. Ensure all of the file args are
            // accounted for by pattern matching.
            let VisConvertArgs {
                args_file: _,
                data_args,
                outputs,
                output_vis_time_average,
                output_vis_freq_average,
                output_smallest_contiguous_band,
            } = unpack_arg_file!(arg_file);

            // Merge all the arguments, preferring the CLI args when available.
            Ok(VisConvertArgs {
                args_file: None,
                data_args: cli_args.data_args.merge(data_args),
                outputs: cli_args.outputs.or(outputs),
                output_vis_time_average: cli_args
                    .output_vis_time_average
                    .or(output_vis_time_average),
                output_vis_freq_average: cli_args
                    .output_vis_freq_average
                    .or(output_vis_freq_average),
                output_smallest_contiguous_band: cli_args.output_smallest_contiguous_band
                    || output_smallest_contiguous_band,
            })
        } else {
            Ok(cli_args)
        }
    }

    pub(super) fn parse(self) -> Result<VisConvertParams, HyperdriveError> {
        debug!("{:#?}", self);

        let Self {
            args_file: _,
            data_args,
            outputs,
            output_vis_time_average,
            output_vis_freq_average,
            output_smallest_contiguous_band,
        } = self;

        if outputs.is_none() {
            return Err(VisConvertArgsError::NoOutputs.into());
        }

        let input_vis_params = data_args.parse("Converting")?;
        let output_vis_params = OutputVisArgs {
            outputs,
            output_vis_time_average,
            output_vis_freq_average,
            output_autos: input_vis_params.using_autos,
        }
        .parse(
            input_vis_params.time_res,
            input_vis_params.spw.freq_res,
            &input_vis_params.timeblocks.mapped_ref(|tb| tb.median),
            output_smallest_contiguous_band,
            "", // Won't be used because the outputs are checked above.
            None,
        )?;

        display_warnings();

        Ok(VisConvertParams {
            input_vis_params,
            output_vis_params,
        })
    }

    pub(super) fn run(self, dry_run: bool) -> Result<(), HyperdriveError> {
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
pub(super) enum VisConvertArgsError {
    #[error("No output visibility files were specified")]
    NoOutputs,
}
