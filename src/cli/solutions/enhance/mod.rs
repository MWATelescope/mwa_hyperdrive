// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Given input data with auto-correlations and a calibration solutions file,
//! take the fine frequency structure of the solutions from the
//! auto-correlations and write out the enhanced solutions.

#[cfg(test)]
mod tests;

use std::{path::PathBuf, str::FromStr};

use clap::Parser;
use log::{debug, info, trace};
use serde::{Deserialize, Serialize};
use vec1::Vec1;

use crate::{
    cli::common::{display_warnings, InputVisArgs, ARG_FILE_HELP},
    params::SolutionsEnhanceParams,
    solutions::{enhance::EnhanceOptions, CalSolutionType, CAL_SOLUTION_EXTENSIONS},
    HyperdriveError,
};

pub(crate) const DEFAULT_OUTPUT_SOLUTIONS_FILENAME: &str = "hyperdrive_solutions_enhanced.fits";
pub(crate) const DEFAULT_DELAY_CUT_US: f64 = 0.1;
pub(crate) const DEFAULT_MAX_MISMATCH: f64 = 5.0;

lazy_static::lazy_static! {
    static ref SOLS_INPUT_HELP: String =
        format!("Path to the calibration solutions file to be enhanced. Supported formats: {}", *CAL_SOLUTION_EXTENSIONS);

    static ref OUTPUTS_HELP: String =
        format!("Paths to the output calibration solution files. Supported formats: {}. Default: {}", *CAL_SOLUTION_EXTENSIONS, DEFAULT_OUTPUT_SOLUTIONS_FILENAME);

    static ref DELAY_CUT_HELP: String =
        format!("The delay [microseconds] below which the input solutions are kept and above which the frequency structure of the auto-correlations is used (this is the half-power point of a Gaussian low-pass filter). Default: {DEFAULT_DELAY_CUT_US}");

    static ref MAX_MISMATCH_HELP: String =
        format!("A tile and polarisation keeps its input solutions when the difference between the fine frequency structure of its solutions and its auto-correlations is more than this many times the median difference over tiles. 0 means all tiles are enhanced. Default: {DEFAULT_MAX_MISMATCH}");
}

#[derive(Parser, Debug, Clone, Default, Serialize, Deserialize)]
pub(crate) struct SolutionsEnhanceArgs {
    #[clap(value_name = "ARGUMENTS_FILE", help = ARG_FILE_HELP.as_str())]
    args_file: Option<PathBuf>,

    #[clap(flatten)]
    #[serde(rename = "data")]
    #[serde(default)]
    data_args: InputVisArgs,

    /// Path to the calibration solutions file to be enhanced.
    #[clap(short, long, help = SOLS_INPUT_HELP.as_str(), help_heading = "INPUT DATA")]
    solutions: Option<String>,

    #[clap(
        short = 'o',
        long,
        num_args(1..),
        help = OUTPUTS_HELP.as_str(),
        help_heading = "OUTPUT FILES"
    )]
    outputs: Option<Vec<PathBuf>>,

    #[clap(long, help = DELAY_CUT_HELP.as_str(), help_heading = "ENHANCEMENT")]
    delay_cut: Option<f64>,

    #[clap(long, help = MAX_MISMATCH_HELP.as_str(), help_heading = "ENHANCEMENT")]
    max_mismatch: Option<f64>,

    /// Only take the amplitudes of the solutions from the auto-correlations;
    /// leave the phases as they are. By default, the phase that goes with the
    /// fine amplitude structure (for a causal response, e.g. cable
    /// reflections) is also applied.
    #[clap(long, help_heading = "ENHANCEMENT")]
    #[serde(default)]
    amplitude_only: bool,
}

impl SolutionsEnhanceArgs {
    /// Both command-line and file arguments overlap in terms of what is
    /// available; this function consolidates everything that was specified into
    /// a single struct. Where applicable, it will prefer CLI parameters over
    /// those in the file.
    ///
    /// This function should only ever merge arguments, and not try to make
    /// sense of them.
    pub(crate) fn merge(self) -> Result<SolutionsEnhanceArgs, HyperdriveError> {
        debug!("Merging command-line arguments with the argument file");

        let cli_args = self;

        if let Some(arg_file) = cli_args.args_file {
            // Read in the file arguments. Ensure all of the file args are
            // accounted for by pattern matching.
            let SolutionsEnhanceArgs {
                args_file: _,
                data_args,
                solutions,
                outputs,
                delay_cut,
                max_mismatch,
                amplitude_only,
            } = unpack_arg_file!(arg_file);

            // Merge all the arguments, preferring the CLI args when available.
            Ok(SolutionsEnhanceArgs {
                args_file: None,
                data_args: cli_args.data_args.merge(data_args),
                solutions: cli_args.solutions.or(solutions),
                outputs: cli_args.outputs.or(outputs),
                delay_cut: cli_args.delay_cut.or(delay_cut),
                max_mismatch: cli_args.max_mismatch.or(max_mismatch),
                amplitude_only: cli_args.amplitude_only || amplitude_only,
            })
        } else {
            Ok(cli_args)
        }
    }

    fn parse(self) -> Result<SolutionsEnhanceParams, HyperdriveError> {
        debug!("{:#?}", self);

        let Self {
            args_file: _,
            mut data_args,
            solutions,
            outputs,
            delay_cut,
            max_mismatch,
            amplitude_only,
        } = self;

        // Add the user-specified solutions to the file list. If there are no
        // data files, parsing the data args fails.
        if let (Some(s), Some(f)) = (solutions, data_args.files.as_mut()) {
            f.push(s);
        }

        let delay_cut_us = delay_cut.unwrap_or(DEFAULT_DELAY_CUT_US);
        if !(delay_cut_us.is_finite() && delay_cut_us > 0.0) {
            return Err(SolutionsEnhanceArgsError::BadDelayCut(delay_cut_us).into());
        }
        let max_mismatch = max_mismatch.unwrap_or(DEFAULT_MAX_MISMATCH);
        if !(max_mismatch.is_finite() && max_mismatch >= 0.0) {
            return Err(SolutionsEnhanceArgsError::BadMaxMismatch(max_mismatch).into());
        }

        let output_solution_files = {
            let outputs =
                outputs.unwrap_or_else(|| vec![PathBuf::from(DEFAULT_OUTPUT_SOLUTIONS_FILENAME)]);
            for file in &outputs {
                let ext = file.extension().and_then(|e| e.to_str());
                if ext
                    .and_then(|e| CalSolutionType::from_str(e).ok())
                    .is_none()
                {
                    return Err(SolutionsEnhanceArgsError::InvalidOutputFormat {
                        file: file.clone(),
                    }
                    .into());
                }
            }
            Vec1::try_from_vec(outputs).map_err(|_| SolutionsEnhanceArgsError::NoOutput)?
        };

        let mut input_vis_params = data_args.parse("Enhancing solutions")?;
        // The solutions aren't to be applied to the data.
        let Some(solutions) = input_vis_params.solutions.take() else {
            return Err(SolutionsEnhanceArgsError::NoSolutions.into());
        };

        display_warnings();

        Ok(SolutionsEnhanceParams {
            input_vis_params,
            solutions,
            output_solution_files,
            options: EnhanceOptions {
                delay_cut_us,
                max_mismatch,
                amplitude_only,
            },
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
        display_warnings();
        Ok(())
    }
}

#[derive(thiserror::Error, Debug)]
pub(crate) enum SolutionsEnhanceArgsError {
    #[error("No calibration solutions were supplied")]
    NoSolutions,

    #[error("No output was specified. There must be at least one calibration solutions output.")]
    NoOutput,

    #[error(
        "An invalid output format was specified ({file}). Supported:\n{}",
        *CAL_SOLUTION_EXTENSIONS,
    )]
    InvalidOutputFormat { file: PathBuf },

    #[error("The delay cut must be a positive number of microseconds; got {0}")]
    BadDelayCut(f64),

    #[error("The maximum mismatch must be 0 or a positive number; got {0}")]
    BadMaxMismatch(f64),
}
