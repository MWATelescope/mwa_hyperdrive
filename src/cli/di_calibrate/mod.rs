// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Parse calibration arguments into parameters.

#[cfg(test)]
mod tests;

use std::{num::NonZeroUsize, path::PathBuf, str::FromStr};

use clap::Parser;
use itertools::Itertools;
use log::{debug, info, log_enabled, trace, Level::Debug};
use marlu::{
    pos::{precession::precess_time, xyz::xyzs_to_cross_uvws},
    LatLngHeight, XyzGeodetic,
};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use vec1::{vec1, Vec1};

use super::common::{
    display_warnings, BeamArgs, InfoPrinter, InputVisArgs, ModellingArgs, OutputVisArgs,
    SkyModelWithVetoArgs, Warn, ARG_FILE_HELP,
};
use crate::{
    averaging::{parse_time_average_factor, timesteps_to_timeblocks, AverageFactorError},
    io::write::{can_write_to_file, VIS_OUTPUT_EXTENSIONS},
    params::{DiCalParams, ModellingParams},
    solutions::{self, CalSolutionType, CalibrationSolutions, CAL_SOLUTION_EXTENSIONS},
    unit_parsing::{parse_wavelength, WavelengthUnit, WAVELENGTH_FORMATS},
    HyperdriveError,
};

// The default minimum baseline cutoff.
const DEFAULT_UVW_MIN: &str = "50λ";

/// The maximum number of times to iterate when performing calibration in
/// direction-independent calibration.
const DEFAULT_MAX_ITERATIONS: u32 = 50;

/// The threshold to satisfy convergence when performing calibration in
/// direction-independent calibration.
const DEFAULT_STOP_THRESHOLD: f64 = 1e-8;

/// The minimum threshold to satisfy convergence when performing calibration in
/// direction-independent calibration. Reaching this threshold counts as
/// "converged", but it's not as good as the stop threshold.
const DEFAULT_MIN_THRESHOLD: f64 = 1e-4;

const DEFAULT_OUTPUT_SOLUTIONS_FILENAME: &str = "hyperdrive_solutions.fits";

lazy_static::lazy_static! {
    static ref DI_SOLS_OUTPUTS_HELP: String =
        format!("Paths to the output calibration solution files. Supported formats: {}. Default: {}", *CAL_SOLUTION_EXTENSIONS, DEFAULT_OUTPUT_SOLUTIONS_FILENAME);

    static ref MODEL_FILENAME_HELP: String =
        format!("The paths to the files where the generated sky-model visibilities are written. If this argument isn't supplied, then no file is written. Supported formats: {}", *VIS_OUTPUT_EXTENSIONS);

    static ref UVW_MIN_HELP: String =
        format!("The minimum UVW length to use. This value must have a unit annotated. Allowed units: {}. Default: {}", *WAVELENGTH_FORMATS, DEFAULT_UVW_MIN);

    static ref UVW_MAX_HELP: String =
        format!("The maximum UVW length to use. This value must have a unit annotated. Allowed units: {}. No default.", *WAVELENGTH_FORMATS);

    static ref MAX_ITERATIONS_HELP: String =
        format!("The maximum number of times to iterate during calibration. Default: {DEFAULT_MAX_ITERATIONS}");

    static ref STOP_THRESHOLD_HELP: String =
        format!("The threshold at which we stop iterating during calibration. Default: {DEFAULT_STOP_THRESHOLD:e}");

    static ref MIN_THRESHOLD_HELP: String =
        format!("The minimum threshold to satisfy convergence during calibration. Even when this threshold is exceeded, iteration will continue until max iterations or the stop threshold is reached. Default: {DEFAULT_MIN_THRESHOLD:e}");
}

#[derive(Parser, Debug, Clone, Default, Serialize, Deserialize)]
struct DiCalCliArgs {
    #[clap(short='o', long="outputs", multiple_values(true), help = DI_SOLS_OUTPUTS_HELP.as_str(), help_heading = "OUTPUT FILES")]
    solutions: Option<Vec<PathBuf>>,

    /// The number of timesteps to average together during calibration. Also
    /// supports a target time resolution (e.g. 8s). If this is 0, then all data
    /// are averaged together. Default: 0. e.g. If this variable is 4, then we
    /// produce calibration solutions in timeblocks with up to 4 timesteps each.
    /// If the variable is instead 4s, then each timeblock contains up to 4s
    /// worth of data.
    #[clap(short, long, help_heading = "CALIBRATION")]
    timesteps_per_timeblock: Option<String>,

    #[clap(long, help = UVW_MIN_HELP.as_str(), help_heading = "CALIBRATION")]
    uvw_min: Option<String>,

    #[clap(long, help = UVW_MAX_HELP.as_str(), help_heading = "CALIBRATION")]
    uvw_max: Option<String>,

    #[clap(long, help = MAX_ITERATIONS_HELP.as_str(), help_heading = "CALIBRATION")]
    max_iterations: Option<u32>,

    #[clap(long, help = STOP_THRESHOLD_HELP.as_str(), help_heading = "CALIBRATION")]
    stop_threshold: Option<f64>,

    #[clap(long, help = MIN_THRESHOLD_HELP.as_str(), help_heading = "CALIBRATION")]
    min_threshold: Option<f64>,

    #[clap(long, multiple_values(true), help = MODEL_FILENAME_HELP.as_str(), help_heading = "OUTPUT FILES")]
    model_filenames: Option<Vec<PathBuf>>,

    /// When writing out model visibilities, average this many timesteps
    /// together. Also supports a target time resolution (e.g. 8s). The value
    /// must be a multiple of the input data's time resolution. The default is
    /// to preserve the input data's time resolution. e.g. If the input data is
    /// in 0.5s resolution and this variable is 4, then we average 2s worth of
    /// model data together before writing the data out. If the variable is
    /// instead 4s, then 8 model timesteps are averaged together before writing
    /// the data out.
    #[clap(long, help_heading = "OUTPUT FILES")]
    output_model_time_average: Option<String>,

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
    output_model_freq_average: Option<String>,

    /// When writing out model visibilities, rather than writing out the entire
    /// input bandwidth, write out only the smallest contiguous band. e.g.
    /// Typical 40 kHz MWA data has 768 channels, but the first 2 and last 2
    /// channels are usually flagged. Turning this option on means that 764
    /// channels would be written out instead of 768. Note that other flagged
    /// channels in the band are unaffected, because the data written out must
    /// be contiguous.
    #[clap(long, help_heading = "OUTPUT FILES")]
    #[serde(default)]
    output_smallest_contiguous_band: bool,
}

#[derive(Parser, Debug, Clone, Default, Serialize, Deserialize)]
pub(super) struct DiCalArgs {
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
    model_args: ModellingArgs,

    #[clap(flatten)]
    #[serde(rename = "beam")]
    #[serde(default)]
    beam_args: BeamArgs,

    #[clap(flatten)]
    #[serde(rename = "di-calibration")]
    #[serde(default)]
    calibration_args: DiCalCliArgs,
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
    pub(super) fn merge(self) -> Result<DiCalArgs, HyperdriveError> {
        debug!("Merging command-line arguments with the argument file");

        let cli_args = self;

        if let Some(arg_file) = cli_args.args_file {
            // Read in the file arguments. Ensure all of the file args are
            // accounted for by pattern matching.
            let DiCalArgs {
                args_file: _,
                data_args,
                srclist_args,
                model_args,
                beam_args,
                calibration_args,
            } = unpack_arg_file!(arg_file);

            // Merge all the arguments, preferring the CLI args when available.
            Ok(DiCalArgs {
                args_file: None,
                data_args: cli_args.data_args.merge(data_args),
                srclist_args: cli_args.srclist_args.merge(srclist_args),
                model_args: cli_args.model_args.merge(model_args),
                beam_args: cli_args.beam_args.merge(beam_args),
                calibration_args: cli_args.calibration_args.merge(calibration_args),
            })
        } else {
            Ok(cli_args)
        }
    }

    /// Parse the arguments into parameters ready for calibration.
    fn parse(self) -> Result<DiCalParams, HyperdriveError> {
        debug!("{:#?}", self);

        let DiCalArgs {
            args_file: _,
            data_args,
            srclist_args,
            model_args,
            beam_args,
            calibration_args,
        } = self;

        let input_vis_params = data_args.parse("DI calibrating")?;
        let obs_context = input_vis_params.get_obs_context();
        let total_num_tiles = input_vis_params.get_total_num_tiles();

        let beam = beam_args.parse(
            total_num_tiles,
            obs_context.dipole_delays.clone(),
            obs_context.dipole_gains.clone(),
            Some(obs_context.input_data_type),
        )?;
        let modelling_params @ ModellingParams { apply_precession } = model_args.parse();

        let DiCalCliArgs {
            timesteps_per_timeblock,
            uvw_min,
            uvw_max,
            max_iterations,
            stop_threshold,
            min_threshold,
            solutions,
            model_filenames,
            output_model_time_average,
            output_model_freq_average,
            output_smallest_contiguous_band,
        } = calibration_args;

        let LatLngHeight {
            longitude_rad,
            latitude_rad,
            height_metres: _,
        } = obs_context.array_position;
        let precession_info = precess_time(
            longitude_rad,
            latitude_rad,
            obs_context.phase_centre,
            // obs_context.timestamps[*timesteps_to_use.first()],
            input_vis_params.timeblocks.first().median,
            input_vis_params.dut1,
        );
        let (lst_rad, latitude_rad) = if apply_precession {
            (
                precession_info.lmst_j2000,
                precession_info.array_latitude_j2000,
            )
        } else {
            (precession_info.lmst, latitude_rad)
        };

        let source_list = srclist_args.parse(
            obs_context.phase_centre,
            lst_rad,
            latitude_rad,
            &obs_context.get_veto_freqs(),
            &*beam,
        )?;

        // Set up the calibration timeblocks.
        let time_average_factor = parse_time_average_factor(
            Some(input_vis_params.time_res),
            timesteps_per_timeblock.as_deref(),
            NonZeroUsize::new(
                input_vis_params.timeblocks.last().timesteps.last()
                    - input_vis_params.timeblocks.first().timesteps.first()
                    + 1,
            )
            .expect("is not 0"),
        )
        .map_err(|e| match e {
            AverageFactorError::Zero => DiCalArgsError::CalTimeFactorZero,
            AverageFactorError::NotInteger => DiCalArgsError::CalTimeFactorNotInteger,
            AverageFactorError::NotIntegerMultiple { out, inp } => {
                DiCalArgsError::CalTimeResNotMultiple { out, inp }
            }
            AverageFactorError::Parse(e) => DiCalArgsError::ParseCalTimeAverageFactor(e),
        })?;
        let all_selected_timestamps = Vec1::try_from_vec(
            input_vis_params
                .timeblocks
                .iter()
                .flat_map(|t| &t.timestamps)
                .copied()
                .collect(),
        )
        .expect("cannot be empty");
        let cal_timeblocks = timesteps_to_timeblocks(
            &all_selected_timestamps,
            input_vis_params.time_res,
            time_average_factor,
            None,
        );

        let mut cal_printer = InfoPrinter::new("DI calibration set up".into());
        // I'm quite bored right now.
        let timeblock_plural = if input_vis_params.timeblocks.len() > 1 {
            "timeblocks"
        } else {
            "timeblock"
        };
        let chanblock_plural = if input_vis_params.spw.chanblocks.len() > 1 {
            "chanblocks"
        } else {
            "chanblock"
        };
        cal_printer.push_block(vec![
            format!(
                "{} calibration {timeblock_plural}, {} calibration {chanblock_plural}",
                cal_timeblocks.len(),
                input_vis_params.spw.chanblocks.len()
            )
            .into(),
            format!("{time_average_factor} timesteps per timeblock").into(),
            // format!("{freq_average_factor} channels per chanblock").into(), // TODO: Not yet implemented
        ]);

        // Set baseline weights from UVW cuts. Use a lambda from the centroid
        // frequency if UVW cutoffs are specified as wavelengths.
        let freq_centroid = obs_context
            .fine_chan_freqs
            .iter()
            .map(|&u| u as f64)
            .sum::<f64>()
            / obs_context.fine_chan_freqs.len() as f64;
        let lambda = marlu::constants::VEL_C / freq_centroid;
        let (uvw_min, uvw_min_metres) = {
            let (quantity, unit) = parse_wavelength(uvw_min.as_deref().unwrap_or(DEFAULT_UVW_MIN))
                .map_err(DiCalArgsError::ParseUvwMin)?;
            match unit {
                WavelengthUnit::M => ((quantity, unit), quantity),
                WavelengthUnit::L => ((quantity, unit), quantity * lambda),
            }
        };
        let (uvw_max, uvw_max_metres) = match uvw_max {
            None => ((f64::INFINITY, WavelengthUnit::M), f64::INFINITY),
            Some(s) => {
                let (quantity, unit) = parse_wavelength(&s).map_err(DiCalArgsError::ParseUvwMax)?;
                match unit {
                    WavelengthUnit::M => ((quantity, unit), quantity),
                    WavelengthUnit::L => ((quantity, unit), quantity * lambda),
                }
            }
        };

        let unflagged_tile_xyzs: Vec<XyzGeodetic> = obs_context
            .tile_xyzs
            .par_iter()
            .enumerate()
            .filter(|(tile_index, _)| {
                !input_vis_params
                    .tile_baseline_flags
                    .flagged_tiles
                    .contains(tile_index)
            })
            .map(|(_, xyz)| *xyz)
            .collect();

        let (baseline_weights, num_flagged_baselines) = {
            let mut baseline_weights = Vec1::try_from_vec(vec![
                1.0;
                input_vis_params
                    .tile_baseline_flags
                    .unflagged_cross_baseline_to_tile_map
                    .len()
            ])
            .expect("not possible to have no unflagged tiles here");
            let uvws = xyzs_to_cross_uvws(
                &unflagged_tile_xyzs,
                obs_context.phase_centre.to_hadec(lst_rad),
            );
            assert_eq!(baseline_weights.len(), uvws.len());
            let uvw_min = uvw_min_metres.powi(2);
            let uvw_max = uvw_max_metres.powi(2);
            let mut num_flagged_baselines = 0;
            for (uvw, baseline_weight) in uvws.into_iter().zip(baseline_weights.iter_mut()) {
                let uvw_length = uvw.u.powi(2) + uvw.v.powi(2) + uvw.w.powi(2);
                if uvw_length < uvw_min || uvw_length > uvw_max {
                    *baseline_weight = 0.0;
                    num_flagged_baselines += 1;
                }
            }
            (baseline_weights, num_flagged_baselines)
        };
        if num_flagged_baselines == baseline_weights.len() {
            return Err(DiCalArgsError::AllBaselinesFlaggedFromUvwCutoffs.into());
        }

        let mut block = vec![];
        block.push(
            format!(
                "Calibrating with {} of {} baselines",
                baseline_weights.len() - num_flagged_baselines,
                baseline_weights.len()
            )
            .into(),
        );
        match (uvw_min, uvw_min.0.is_infinite()) {
            // Again, bored.
            (_, true) => block.push("Minimum UVW cutoff: ∞".into()),
            ((quantity, WavelengthUnit::M), _) => {
                block.push(format!("Minimum UVW cutoff: {quantity}m").into())
            }
            ((quantity, WavelengthUnit::L), _) => block.push(
                format!(
                    "Minimum UVW cutoff: {quantity}λ ({:.3}m)",
                    quantity * lambda
                )
                .into(),
            ),
        }
        match (uvw_max, uvw_max.0.is_infinite()) {
            (_, true) => block.push("Maximum UVW cutoff: ∞".into()),
            ((quantity, WavelengthUnit::M), _) => {
                block.push(format!("Maximum UVW cutoff: {quantity}m").into())
            }
            ((quantity, WavelengthUnit::L), _) => block.push(
                format!(
                    "Maximum UVW cutoff: {quantity}λ ({:.3}m)",
                    quantity * lambda
                )
                .into(),
            ),
        }
        // Report extra info if we need to use our own lambda (the user
        // specified wavelengths).
        if matches!(uvw_min.1, WavelengthUnit::L) || matches!(uvw_max.1, WavelengthUnit::L) {
            block.push(
                format!(
                    "(Used obs. centroid frequency {} MHz to convert lambdas to metres)",
                    freq_centroid / 1e6
                )
                .into(),
            );
        }
        cal_printer.push_block(block);

        let mut unflagged_fine_chan_freqs = vec![];
        let flagged_fine_chans = &input_vis_params.spw.flagged_chan_indices;
        for (i_chan, &freq) in obs_context.fine_chan_freqs.iter().enumerate() {
            if !flagged_fine_chans.contains(&(i_chan as u16)) {
                unflagged_fine_chan_freqs.push(freq as f64);
            }
        }
        if log_enabled!(Debug) {
            let unflagged_fine_chans: Vec<_> = (0..obs_context.fine_chan_freqs.len())
                .filter(|i_chan| !flagged_fine_chans.contains(&(*i_chan as u16)))
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
        // There should never be any no unflagged channels, because this
        // should've been handled by the input-vis-reading code.
        assert!(!unflagged_fine_chan_freqs.is_empty());

        // Make sure the calibration thresholds are sensible.
        let mut stop_threshold = stop_threshold.unwrap_or(DEFAULT_STOP_THRESHOLD);
        let min_threshold = min_threshold.unwrap_or(DEFAULT_MIN_THRESHOLD);
        if stop_threshold > min_threshold {
            format!("Specified stop threshold ({:e}) is bigger than the min. threshold ({:e}); capping stop threshold.", stop_threshold, min_threshold).warn();
            stop_threshold = min_threshold;
        }
        let max_iterations = max_iterations.unwrap_or(DEFAULT_MAX_ITERATIONS);

        cal_printer.push_block(vec![
            "Chanblocks will stop iterating".into(),
            format!(
                "- when the iteration difference is less than {:e} (stop threshold)",
                stop_threshold
            )
            .into(),
            format!("- or after {} iterations.", max_iterations).into(),
            format!(
                "Chanblocks with an iteration diff. less than {:e} are considered converged (min. threshold)",
                min_threshold
            )
            .into(),
        ]);

        let output_solution_files = {
            match solutions {
                // Defaults.
                None => {
                    let pb = PathBuf::from(DEFAULT_OUTPUT_SOLUTIONS_FILENAME);
                    let sol_type = pb
                        .extension()
                        .and_then(|os_str| os_str.to_str())
                        .and_then(|s| CalSolutionType::from_str(s).ok())
                        // Tests should pick up a bad default filename.
                        .expect("DEFAULT_OUTPUT_SOLUTIONS_FILENAME has an unhandled extension!");
                    vec1![(pb, sol_type)]
                }
                Some(outputs) => {
                    let mut cal_sols = vec![];
                    for file in outputs {
                        // Is the output file type supported?
                        let ext = file.extension().and_then(|os_str| os_str.to_str());
                        match ext.and_then(|s| CalSolutionType::from_str(s).ok()) {
                            Some(sol_type) => {
                                trace!("{} is a solution output", file.display());
                                can_write_to_file(&file)
                                    .map_err(|e| HyperdriveError::Generic(e.to_string()))?;
                                cal_sols.push((file, sol_type));
                            }
                            None => {
                                return Err(DiCalArgsError::CalibrationOutputFile {
                                    ext: ext.unwrap_or("<no extension>").to_string(),
                                }
                                .into())
                            }
                        }
                    }
                    Vec1::try_from_vec(cal_sols).expect("cannot fail")
                }
            }
        };
        if output_solution_files.is_empty() {
            return Err(DiCalArgsError::NoOutput.into());
        }

        cal_printer.push_line(
            format!(
                "Writing calibration solutions to: {}",
                output_solution_files
                    .iter()
                    .map(|(pb, _)| pb.display())
                    .join(", ")
            )
            .into(),
        );

        // Parse the output model vis args like normal output vis args, to
        // re-use existing code (we only make the args distinct to make it clear
        // that these visibilities are not calibrated, just the model vis).
        let output_model_vis_params = match model_filenames {
            None => None,
            Some(model_filenames) => {
                let output_vis_params = OutputVisArgs {
                    outputs: Some(model_filenames),
                    output_vis_time_average: output_model_time_average,
                    output_vis_freq_average: output_model_freq_average,
                    output_autos: input_vis_params.using_autos,
                }
                .parse(
                    input_vis_params.time_res,
                    input_vis_params.spw.freq_res,
                    &input_vis_params.timeblocks.mapped_ref(|tb| tb.median),
                    output_smallest_contiguous_band,
                    "hyp_model.uvfits", // not actually used
                    Some("model"),
                )?;

                Some(output_vis_params)
            }
        };

        cal_printer.display();
        display_warnings();

        Ok(DiCalParams {
            input_vis_params,
            beam,
            source_list,
            cal_timeblocks,
            uvw_min: uvw_min_metres,
            uvw_max: uvw_max_metres,
            freq_centroid,
            baseline_weights,
            max_iterations,
            stop_threshold,
            min_threshold,
            output_solution_files,
            output_model_vis_params,
            modelling_params,
        })
    }

    pub(super) fn run(
        self,
        dry_run: bool,
    ) -> Result<Option<CalibrationSolutions>, HyperdriveError> {
        debug!("Converting arguments into parameters");
        trace!("{:#?}", self);
        let params = self.parse()?;

        if dry_run {
            info!("Dry run -- exiting now.");
            return Ok(None);
        }

        let sols = params.run()?;

        // Write out the solutions.
        let num_solution_files = params.output_solution_files.len();
        for (i, (file, sol_type)) in params.output_solution_files.into_iter().enumerate() {
            match sol_type {
                CalSolutionType::Fits => solutions::hyperdrive::write(&sols, &file)?,
                CalSolutionType::Bin => solutions::ao::write(&sols, &file)?,
            }
            if num_solution_files == 1 {
                info!("Calibration solutions written to {}", file.display());
            } else {
                if i == 0 {
                    info!("Calibration solutions written to:");
                }
                info!("  {}", file.display());
            }
        }

        Ok(Some(sols))
    }
}

/// Errors associated with DI calibration arguments.
#[derive(thiserror::Error, Debug)]
pub(super) enum DiCalArgsError {
    #[error("No calibration output was specified. There must be at least one calibration solution file.")]
    NoOutput,

    #[error(
        "All baselines were flagged due to UVW cutoffs. Try adjusting the UVW min and/or max."
    )]
    AllBaselinesFlaggedFromUvwCutoffs,

    #[error("Cannot write calibration solutions to a file type '{ext}'.\nSupported formats are: {}", *crate::solutions::CAL_SOLUTION_EXTENSIONS)]
    CalibrationOutputFile { ext: String },

    #[error("Error when parsing time average factor: {0}")]
    ParseCalTimeAverageFactor(crate::unit_parsing::UnitParseError),

    #[error("Calibration time average factor isn't an integer")]
    CalTimeFactorNotInteger,

    #[error("Calibration time resolution isn't a multiple of input data's: {out} seconds vs {inp} seconds")]
    CalTimeResNotMultiple { out: f64, inp: f64 },

    #[error("Calibration time average factor cannot be 0")]
    CalTimeFactorZero,

    // #[error("Error when parsing freq. average factor: {0}")]
    // ParseCalFreqAverageFactor(crate::unit_parsing::UnitParseError),

    // #[error("Calibration freq. average factor isn't an integer")]
    // CalFreqFactorNotInteger,

    // #[error("Calibration freq. resolution isn't a multiple of input data's: {out} Hz vs {inp} Hz")]
    // CalFreqResNotMultiple { out: f64, inp: f64 },

    // #[error("Calibration freq. average factor cannot be 0")]
    // CalFreqFactorZero,
    #[error("Error when parsing minimum UVW cutoff: {0}")]
    ParseUvwMin(crate::unit_parsing::UnitParseError),

    #[error("Error when parsing maximum UVW cutoff: {0}")]
    ParseUvwMax(crate::unit_parsing::UnitParseError),

    #[error(transparent)]
    IO(#[from] std::io::Error),
}

impl DiCalCliArgs {
    fn merge(self, other: Self) -> Self {
        Self {
            timesteps_per_timeblock: self
                .timesteps_per_timeblock
                .or(other.timesteps_per_timeblock),
            uvw_min: self.uvw_min.or(other.uvw_min),
            uvw_max: self.uvw_max.or(other.uvw_max),
            max_iterations: self.max_iterations.or(other.max_iterations),
            stop_threshold: self.stop_threshold.or(other.stop_threshold),
            min_threshold: self.min_threshold.or(other.min_threshold),
            solutions: self.solutions.or(other.solutions),
            model_filenames: self.model_filenames.or(other.model_filenames),
            output_model_time_average: self
                .output_model_time_average
                .or(other.output_model_time_average),
            output_model_freq_average: self
                .output_model_freq_average
                .or(other.output_model_freq_average),
            output_smallest_contiguous_band: self.output_smallest_contiguous_band
                || other.output_smallest_contiguous_band,
        }
    }
}
