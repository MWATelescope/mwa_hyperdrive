// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

use std::path::PathBuf;

use clap::Parser;
use log::{debug, info, trace};
use marlu::{precession::precess_time, LatLngHeight};
use serde::{Deserialize, Serialize};

use super::common::{
    display_warnings, BeamArgs, InputVisArgs, ModellingArgs, OutputVisArgs, SkyModelWithVetoArgs,
    ARG_FILE_HELP,
};
use crate::{
    io::write::VIS_OUTPUT_EXTENSIONS,
    params::{ModellingParams, VisSubtractParams},
    srclist::SourceList,
    HyperdriveError,
};

const DEFAULT_OUTPUT_VIS_FILENAME: &str = "hyp_subtracted.uvfits";

lazy_static::lazy_static! {
    static ref OUTPUTS_HELP: String =
        format!("Paths to the subtracted visibility files. Supported formats: {}. Default: {}", *VIS_OUTPUT_EXTENSIONS, DEFAULT_OUTPUT_VIS_FILENAME);
}

#[derive(Parser, Debug, Clone, Default, Serialize, Deserialize)]
struct VisSubtractCliArgs {
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

    /// Don't write auto-correlations to the output visibilities.
    /// Default: output if present
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

#[derive(Parser, Debug, Clone, Default, Serialize, Deserialize)]
pub(super) struct VisSubtractArgs {
    #[clap(name = "ARGUMENTS_FILE", help = ARG_FILE_HELP.as_str(), parse(from_os_str))]
    args_file: Option<PathBuf>,

    #[clap(flatten)]
    #[serde(rename = "data")]
    #[serde(default)]
    data_args: InputVisArgs,

    #[clap(flatten)]
    #[serde(rename = "sky-model")]
    #[serde(default)]
    srclist_args: SkyModelWithVetoArgs,

    #[clap(flatten)]
    #[serde(rename = "model")]
    #[serde(default)]
    modelling_args: ModellingArgs,

    #[clap(flatten)]
    #[serde(rename = "beam")]
    #[serde(default)]
    beam_args: BeamArgs,

    #[clap(flatten)]
    #[serde(rename = "vis-subtract")]
    #[serde(default)]
    vis_subtract_args: VisSubtractCliArgs,
}

impl VisSubtractArgs {
    /// Both command-line and file arguments overlap in terms of what is
    /// available; this function consolidates everything that was specified into
    /// a single struct. Where applicable, it will prefer CLI parameters over
    /// those in the file.
    ///
    /// The argument to this function is the path to the arguments file.
    ///
    /// This function should only ever merge arguments, and not try to make
    /// sense of them.
    pub(super) fn merge(self) -> Result<VisSubtractArgs, HyperdriveError> {
        debug!("Merging command-line arguments with the argument file");

        let cli_args = self;

        if let Some(arg_file) = cli_args.args_file {
            // Read in the file arguments. Ensure all of the file args are
            // accounted for by pattern matching.
            let VisSubtractArgs {
                args_file: _,
                data_args,
                srclist_args,
                modelling_args,
                beam_args,
                vis_subtract_args,
            } = unpack_arg_file!(arg_file);

            // Merge all the arguments, preferring the CLI args when available.
            Ok(VisSubtractArgs {
                args_file: None,
                data_args: cli_args.data_args.merge(data_args),
                srclist_args: cli_args.srclist_args.merge(srclist_args),
                modelling_args: cli_args.modelling_args.merge(modelling_args),
                beam_args: cli_args.beam_args.merge(beam_args),
                vis_subtract_args: cli_args.vis_subtract_args.merge(vis_subtract_args),
            })
        } else {
            Ok(cli_args)
        }
    }

    fn parse(self) -> Result<VisSubtractParams, HyperdriveError> {
        debug!("{:#?}", self);

        let Self {
            args_file: _,
            data_args,
            srclist_args,
            modelling_args,
            beam_args,
            vis_subtract_args:
                VisSubtractCliArgs {
                    outputs,
                    output_vis_time_average,
                    output_vis_freq_average,
                    output_smallest_contiguous_band,
                    output_no_autos,
                },
        } = self;

        let input_vis_params = data_args.parse("Vis subtracting")?;
        let obs_context = input_vis_params.get_obs_context();
        let total_num_tiles = obs_context.get_total_num_tiles();

        let beam = beam_args.parse(
            total_num_tiles,
            obs_context.dipole_delays.clone(),
            obs_context.dipole_gains.clone(),
            Some(obs_context.input_data_type),
        )?;
        let modelling_params @ ModellingParams {
            apply_precession, ..
        } = modelling_args.parse();

        let LatLngHeight {
            longitude_rad,
            latitude_rad,
            height_metres: _,
        } = obs_context.array_position;
        let precession_info = precess_time(
            longitude_rad,
            latitude_rad,
            obs_context.phase_centre,
            input_vis_params.timeblocks.first().median,
            input_vis_params.dut1,
        );
        let (lmst, latitude) = if apply_precession {
            (
                precession_info.lmst_j2000,
                precession_info.array_latitude_j2000,
            )
        } else {
            (precession_info.lmst, latitude_rad)
        };

        // Read in the source list and remove all but the specified sources.
        let source_list: SourceList = srclist_args.parse(
            obs_context.phase_centre,
            lmst,
            latitude,
            &obs_context.get_veto_freqs(),
            &*beam,
        )?;

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
            Some("subtracted"),
        )?;

        display_warnings();

        Ok(VisSubtractParams {
            input_vis_params,
            output_vis_params,
            beam,
            source_list,
            modelling_params,
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

impl VisSubtractCliArgs {
    fn merge(self, other: Self) -> Self {
        Self {
            outputs: self.outputs.or(other.outputs),
            output_vis_time_average: self
                .output_vis_time_average
                .or(other.output_vis_time_average),
            output_vis_freq_average: self
                .output_vis_freq_average
                .or(other.output_vis_freq_average),
            output_no_autos: self.output_no_autos || other.output_no_autos,
            output_smallest_contiguous_band: self.output_smallest_contiguous_band
                || other.output_smallest_contiguous_band,
        }
    }
}
