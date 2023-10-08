// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

use std::{borrow::Cow, path::PathBuf, str::FromStr};

use clap::Parser;
use log::{debug, info, trace};
use marlu::{precession::precess_time, LatLngHeight};
use serde::{Deserialize, Serialize};

use super::common::{
    display_warnings, BeamArgs, InputVisArgs, ModellingArgs, OutputVisArgs, SkyModelWithVetoArgs,
    ARG_FILE_HELP,
};
use crate::{
    cli::common::InfoPrinter,
    constants::{DEFAULT_CUTOFF_DISTANCE, DEFAULT_VETO_THRESHOLD},
    io::{get_single_match_from_glob, write::VIS_OUTPUT_EXTENSIONS},
    params::{ModellingParams, VisSubtractParams},
    srclist::{
        read::read_source_list_file, veto_sources, ComponentCounts, ReadSourceListError,
        SourceList, SourceListType,
    },
    HyperdriveError,
};

const DEFAULT_OUTPUT_VIS_FILENAME: &str = "hyp_subtracted.uvfits";

lazy_static::lazy_static! {
    static ref OUTPUTS_HELP: String =
        format!("Paths to the subtracted visibility files. Supported formats: {}. Default: {}", *VIS_OUTPUT_EXTENSIONS, DEFAULT_OUTPUT_VIS_FILENAME);
}

#[derive(Parser, Debug, Clone, Default, Serialize, Deserialize)]
struct VisSubtractCliArgs {
    /// Invert the subtraction; sources *not* specified in sources-to-subtract
    /// will be subtracted from the input data.
    #[clap(short, long, help_heading = "SKY-MODEL SOURCES")]
    #[serde(default)]
    invert: bool,

    /// The names of the sources in the sky-model source list that will be
    /// subtracted from the input data.
    #[clap(long, multiple_values(true), help_heading = "SKY-MODEL SOURCES")]
    sources_to_subtract: Option<Vec<String>>,

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

    /// Don't write subtracted auto-correlatsions to the output visibilities.
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
                    invert,
                    sources_to_subtract,
                    outputs,
                    output_vis_time_average,
                    output_vis_freq_average,
                    output_no_autos,
                    output_smallest_contiguous_band,
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
            Some(&obs_context.tile_names),
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

        // If we're not inverted but `sources_to_subtract` is empty, then there's
        // nothing to do.
        let sources_to_subtract = sources_to_subtract.unwrap_or_default();
        if !invert && sources_to_subtract.is_empty() {
            return Err(VisSubtractArgsError::NoSources.into());
        }

        // Read in the source list and remove all but the specified sources. We
        // have to parse the arguments manually as we're doing custom stuff here
        // in vis-subtract.
        let SkyModelWithVetoArgs {
            source_list,
            source_list_type,
            num_sources,
            source_dist_cutoff,
            veto_threshold,
        } = srclist_args;

        let source_list: SourceList = {
            let source_list = source_list.ok_or(ReadSourceListError::NoSourceList)?;
            // If the specified source list file can't be found, treat it as a glob
            // and expand it to find a match.
            let pb = PathBuf::from(&source_list);
            let pb = if pb.exists() {
                pb
            } else {
                get_single_match_from_glob(&source_list)
                    .map_err(|e| HyperdriveError::Generic(e.to_string()))?
            };

            // Read the source list file. If the type was manually specified,
            // use that, otherwise the reading code will try all available
            // kinds.
            let sl_type_not_specified = source_list_type.is_none();
            let sl_type = source_list_type
                .as_ref()
                .and_then(|t| SourceListType::from_str(t.as_ref()).ok());
            let (sl, sl_type) = read_source_list_file(pb, sl_type)?;

            // If the user didn't specify the source list type, then print out
            // what we found.
            if sl_type_not_specified {
                trace!("Successfully parsed {}-style source list", sl_type);
            }
            if num_sources == Some(0) || sl.is_empty() {
                return Err(ReadSourceListError::NoSources.into());
            }
            sl
        };
        debug!("Found {} sources in the source list", source_list.len());
        let ComponentCounts {
            num_points,
            num_gaussians,
            num_shapelets,
            ..
        } = source_list.get_counts();
        let mut sl_printer = InfoPrinter::new("Sky model info".into());
        sl_printer.push_block(vec![
            format!("Source list contains {} sources", source_list.len()).into(),
            format!("({} components, {num_points} points, {num_gaussians} Gaussians, {num_shapelets} shapelets)", num_points + num_gaussians + num_shapelets).into()
        ]);

        // Ensure that all specified sources are actually in the source list.
        for name in &sources_to_subtract {
            if !source_list.contains_key(name) {
                return Err(HyperdriveError::from(VisSubtractArgsError::MissingSource {
                    name: name.to_string().into(),
                }));
            }
        }
        // Handle the invert option.
        let source_list: SourceList = if invert {
            let mut sl: SourceList = source_list
                .into_iter()
                .filter(|(name, _)| !sources_to_subtract.contains(name))
                .collect();
            if sl.is_empty() {
                // Nothing to do.
                return Err(VisSubtractArgsError::AllSourcesFiltered.into());
            }
            veto_sources(
                &mut sl,
                obs_context.phase_centre,
                lmst,
                latitude,
                &obs_context.get_veto_freqs(),
                &*beam,
                num_sources,
                source_dist_cutoff.unwrap_or(DEFAULT_CUTOFF_DISTANCE),
                veto_threshold.unwrap_or(DEFAULT_VETO_THRESHOLD),
            )?;
            if sl.is_empty() {
                return Err(ReadSourceListError::NoSourcesAfterVeto.into());
            }
            sl
        } else {
            source_list
                .into_iter()
                .filter(|(name, _)| sources_to_subtract.contains(name))
                .collect()
        };
        let ComponentCounts {
            num_points,
            num_gaussians,
            num_shapelets,
            num_power_laws: _,
            num_curved_power_laws: _,
            num_lists: _,
        } = source_list.get_counts();
        sl_printer.push_block(vec![
            format!(
                "Subtracting {} sources with a total of {} components",
                source_list.len(),
                num_points + num_gaussians + num_shapelets
            )
            .into(),
            format!("{num_points} points, {num_gaussians} Gaussians, {num_shapelets} shapelets")
                .into(),
        ]);
        sl_printer.display();

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

#[derive(thiserror::Error, Debug)]
pub(super) enum VisSubtractArgsError {
    #[error("Specified source {name} is not in the input source list; can't subtract it")]
    MissingSource { name: Cow<'static, str> },

    #[error("No sources were specified for subtraction. Did you want to subtract all sources? See the \"invert\" option.")]
    NoSources,

    #[error("No sources were left after removing specified sources from the source list.")]
    AllSourcesFiltered,
}

impl VisSubtractCliArgs {
    fn merge(self, other: Self) -> Self {
        Self {
            invert: self.invert || other.invert,
            sources_to_subtract: self.sources_to_subtract.or(other.sources_to_subtract),
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
