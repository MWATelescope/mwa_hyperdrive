// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

mod error;
pub(crate) use error::PeelArgsError;

use std::{collections::HashSet, num::NonZeroUsize, path::PathBuf, str::FromStr};

use clap::Parser;
use log::{debug, info, trace};
use marlu::{
    pos::xyz::xyzs_to_cross_uvws, precession::precess_time, LatLngHeight, XyzGeodetic, UVW,
};
use serde::{Deserialize, Serialize};

use super::common::{BeamArgs, InputVisArgs, ModellingArgs, SkyModelWithVetoArgs, ARG_FILE_HELP};
use crate::{
    averaging::{
        channels_to_chanblocks, parse_freq_average_factor, parse_time_average_factor,
        timesteps_to_timeblocks, unflag_spw, AverageFactorError,
    },
    cli::{
        common::{display_warnings, InfoPrinter, OutputVisArgs},
        Warn,
    },
    io::write::{VisOutputType, VIS_OUTPUT_EXTENSIONS},
    math::div_ceil,
    params::{ModellingParams, PeelLoopParams, PeelParams, PeelWeightParams},
    unit_parsing::{parse_wavelength, WavelengthUnit, WAVELENGTH_FORMATS},
    HyperdriveError,
};

#[cfg(test)]
mod tests;

const DEFAULT_OUTPUT_PEEL_FILENAME: &str = "hyperdrive_peeled.uvfits";
const DEFAULT_OUTPUT_IONO_CONSTS: &str = "hyperdrive_iono_consts.json";
#[cfg(not(any(feature = "cuda", feature = "hip")))]
const DEFAULT_NUM_PASSES: usize = 1;
#[cfg(any(feature = "cuda", feature = "hip"))]
const DEFAULT_NUM_PASSES: usize = 3;
const DEFAULT_NUM_LOOPS: usize = 10;
const DEFAULT_IONO_TIME_AVERAGE: &str = "8s";
const DEFAULT_IONO_FREQ_AVERAGE: &str = "1.28MHz";
const DEFAULT_UVW_MIN: &str = "50λ";
const DEFAULT_SHORT_BASELINE_SIGMA: f64 = 40.0;
const DEFAULT_CONVERGENCE: f64 = 1.0;

lazy_static::lazy_static! {
    static ref VIS_OUTPUTS_HELP: String = format!("The paths to the files where the peeled visibilities are written. Supported formats: {}", *VIS_OUTPUT_EXTENSIONS);

    static ref NUM_PASSES_HELP: String = format!("The number of times to iterate over all sources per iono timeblock. Default: {DEFAULT_NUM_PASSES}");

    static ref NUM_LOOPS_HELP: String = format!("The number of times to loop over a single sources per pass. Default: {DEFAULT_NUM_LOOPS}");

    static ref IONO_TIME_AVERAGE_FACTOR_HELP: String = format!("The time resolution of a peeling timeblock, or an averaging factor relative to the input resolution. If this is 0, then all data are averaged together. Default: {DEFAULT_IONO_TIME_AVERAGE}. e.g. If this variable is 4, then peeling is performed with 4 timesteps per timeblock. If the variable is instead 4s, then each timeblock contains up to 4s worth of data.");

    static ref IONO_FREQ_AVERAGE_FACTOR_HELP: String = format!("The number of fine-frequency channels to average together *during* peeling. Also supports a target frequency resolution (e.g. 1.28MHz). Cannot be 0. Default: {DEFAULT_IONO_FREQ_AVERAGE}. e.g. If the input data is in 40kHz resolution and this variable was 2, then we average 80kHz worth of data into a chanblock during peeling. If the variable is instead 1.28MHz, then each chanblock contains 32 fine channels.");

    static ref OUTPUT_TIME_AVERAGE_FACTOR_HELP: String = format!("The number of timeblocks to average together when writing out visibilities. Also supports a target time resolution (e.g. 8s). If this is 0, then all data are averaged together. Defaults to input resolution. e.g. If this variable is 4, then 8 timesteps are averaged together as a timeblock in the output visibilities.");

    static ref OUTPUT_FREQ_AVERAGE_FACTOR_HELP: String = format!("The number of fine-frequency channels to average together when writing out visibilities. Also supports a target time resolution (e.g. 80kHz). If this is 0, then all data are averaged together. Defaults to input resolution. This is multiplicative with the freq average factor; e.g. If this variable is 4, and the freq average factor is 2, then 8 fine-frequency channels are averaged together as a chanblock in the output visibilities.");

    static ref UVW_MIN_HELP: String = format!("The minimum UVW length to use. This value must have a unit annotated. Allowed units: {}. Default: {DEFAULT_UVW_MIN}", *WAVELENGTH_FORMATS);

    static ref UVW_MAX_HELP: String = format!("The maximum UVW length to use. This value must have a unit annotated. Allowed units: {}. No default.", *WAVELENGTH_FORMATS);

    static ref SHORT_BASELINE_SIGMA_HELP: String = format!("taper weights of short baselines with a gaussian of this sigma (wavenumbers). Default: {DEFAULT_SHORT_BASELINE_SIGMA}");

    static ref CONVERGENCE_HELP: String = format!("A value between 0 and 1 that determines the speed that peel converges while solving for alpha and beta. Default: {DEFAULT_CONVERGENCE}");
}

#[derive(Parser, Debug, Clone, Default, Serialize, Deserialize)]
pub(crate) struct PeelCliArgs {
    /// The number of sources to "ionospherically subtract". In contrast to the
    /// rest of the sky model, which is directly subtracted without finding a λ²
    /// dependence source. The default is to ionospherically subtract all
    /// sources in the sky model after vetoing, and restricting to --num-sources
    /// (-n). The number of iono subtract sources cannot be greater than this
    /// default value.
    #[clap(long = "iono-sub", help_heading = "PEELING")]
    pub(super) num_sources_to_iono_subtract: Option<usize>,

    // TODO: peel
    // The number of sources to peel. Peel sources are treated the same as
    // "ionospherically subtracted" sources, except before subtracting, a "DI
    // calibration" is done between the iono-rotated model and the data. This
    // allows for scintillation and any other phase shift to be corrected.
    // #[clap(long = "peel", help_heading = "PEELING")]
    // pub(super) num_sources_to_peel: Option<usize>,
    #[clap(long, help = NUM_PASSES_HELP.as_str(), help_heading = "PEELING")]
    pub(super) num_passes: Option<usize>,

    #[clap(long, help = NUM_LOOPS_HELP.as_str(), help_heading = "PEELING")]
    pub(super) num_loops: Option<usize>,

    #[clap(long, help = IONO_TIME_AVERAGE_FACTOR_HELP.as_str(), help_heading = "PEELING")]
    pub(super) iono_time_average: Option<String>,

    #[clap(long, help = IONO_FREQ_AVERAGE_FACTOR_HELP.as_str(), help_heading = "PEELING")]
    pub(super) iono_freq_average: Option<String>,

    #[clap(long, help = UVW_MIN_HELP.as_str(), help_heading = "PEELING")]
    pub(super) uvw_min: Option<String>,

    #[clap(long, help = UVW_MAX_HELP.as_str(), help_heading = "PEELING")]
    pub(super) uvw_max: Option<String>,

    #[clap(long, help = SHORT_BASELINE_SIGMA_HELP.as_str(), help_heading = "PEELING")]
    pub(super) short_baseline_sigma: Option<f64>,

    #[clap(long, help = CONVERGENCE_HELP.as_str(), help_heading = "PEELING")]
    pub(super) convergence: Option<f64>,

    #[clap(short, long, multiple_values(true), help = VIS_OUTPUTS_HELP.as_str(), help_heading = "OUTPUT FILES")]
    pub(super) outputs: Option<Vec<PathBuf>>,

    #[clap(long, help = OUTPUT_TIME_AVERAGE_FACTOR_HELP.as_str(), help_heading = "OUTPUT FILES")]
    pub(super) output_vis_time_average: Option<String>,

    #[clap(long, help = OUTPUT_FREQ_AVERAGE_FACTOR_HELP.as_str(), help_heading = "OUTPUT FILES")]
    pub(super) output_vis_freq_average: Option<String>,


    /// When writing out visibilities, rather than writing out the entire input
    /// bandwidth, write out only the smallest contiguous band. e.g. Typical 40
    /// kHz MWA data has 768 channels, but the first 2 and last 2 channels are
    /// usually flagged. Turning this option on means that 764 channels would be
    /// written out instead of 768. Note that other flagged channels in the band
    /// are unaffected, because the data written out must be contiguous.
    #[clap(long, help_heading = "OUTPUT FILES")]
    #[serde(default)]
    output_smallest_contiguous_band: bool,
}

#[derive(Parser, Debug, Clone, Default, Serialize, Deserialize)]
pub(super) struct PeelArgs {
    #[clap(name = "ARGUMENTS_FILE", help = ARG_FILE_HELP.as_str(), parse(from_os_str))]
    pub(super) args_file: Option<PathBuf>,

    #[clap(flatten)]
    #[serde(rename = "data")]
    #[serde(default)]
    pub(super) data_args: InputVisArgs,

    #[clap(flatten)]
    #[serde(rename = "sky-model")]
    #[serde(default)]
    pub(super) srclist_args: SkyModelWithVetoArgs,

    #[clap(flatten)]
    #[serde(rename = "model")]
    #[serde(default)]
    pub(super) model_args: ModellingArgs,

    #[clap(flatten)]
    #[serde(rename = "beam")]
    #[serde(default)]
    pub(super) beam_args: BeamArgs,

    #[clap(flatten)]
    #[serde(rename = "peel")]
    #[serde(default)]
    pub(super) peel_args: PeelCliArgs,
}

impl PeelArgs {
    pub(crate) fn merge(self) -> Result<PeelArgs, HyperdriveError> {
        debug!("Merging command-line arguments with the argument file");

        let cli_args = self;

        if let Some(arg_file) = cli_args.args_file {
            // Read in the file arguments. Ensure all of the file args are
            // accounted for by pattern matching.
            let PeelArgs {
                args_file: _,
                data_args,
                srclist_args,
                model_args,
                beam_args,
                peel_args,
            } = unpack_arg_file!(arg_file);

            // Merge all the arguments, preferring the CLI args when available.
            Ok(PeelArgs {
                args_file: None,
                data_args: cli_args.data_args.merge(data_args),
                srclist_args: cli_args.srclist_args.merge(srclist_args),
                model_args: cli_args.model_args.merge(model_args),
                beam_args: cli_args.beam_args.merge(beam_args),
                peel_args: cli_args.peel_args.merge(peel_args),
            })
        } else {
            Ok(cli_args)
        }
    }

    fn parse(self) -> Result<PeelParams, HyperdriveError> {
        debug!("{:#?}", self);

        let Self {
            args_file: _,
            data_args,
            srclist_args,
            model_args,
            beam_args,
            peel_args:
                PeelCliArgs {
                    num_sources_to_iono_subtract,
                    num_passes,
                    num_loops,
                    iono_time_average,
                    iono_freq_average,
                    uvw_min,
                    uvw_max,
                    short_baseline_sigma,
                    convergence,
                    outputs,
                    output_vis_time_average,
                    output_vis_freq_average,
                    output_smallest_contiguous_band,
                },
        } = self;

        // TODO: peel doesn't correctly average channels with flags?
        // data_args.freq_average = Some("1".into());
        let input_vis_params = {
            let mut input_vis_params = data_args.parse("Peeling")?;
            input_vis_params.spw = unflag_spw(input_vis_params.spw);
            input_vis_params
        };

        let obs_context = input_vis_params.get_obs_context();
        let total_num_tiles = input_vis_params.get_total_num_tiles();

        let beam = beam_args.parse(
            total_num_tiles,
            obs_context.dipole_delays.clone(),
            obs_context.dipole_gains.clone(),
            Some(obs_context.input_data_type),
        )?;
        let modelling_params @ ModellingParams { apply_precession } = model_args.parse();

        let LatLngHeight {
            longitude_rad,
            latitude_rad,
            height_metres: _,
        } = obs_context.array_position;
        let (source_list, lmst_rad) = {
            let precession_info = precess_time(
                longitude_rad,
                latitude_rad,
                obs_context.phase_centre,
                input_vis_params.timeblocks.first().median,
                input_vis_params.dut1,
            );
            let (lst_rad, lat_rad) = if apply_precession {
                (
                    precession_info.lmst_j2000,
                    precession_info.array_latitude_j2000,
                )
            } else {
                (precession_info.lmst, latitude_rad)
            };
            let srclist = srclist_args.parse(
                obs_context.phase_centre,
                lst_rad,
                lat_rad,
                &obs_context.get_veto_freqs(),
                &*beam,
            )?;
            (srclist, lst_rad)
        };

        let sky_model_source_count = source_list.len();

        // Check that the number of sources to peel, iono subtract and subtract
        // are sensible.
        if let Some(is) = num_sources_to_iono_subtract {
            if is > sky_model_source_count {
                return Err(PeelArgsError::TooManyIonoSub {
                    total: sky_model_source_count,
                    iono: is,
                }
                .into());
            }
        }

        let num_passes = NonZeroUsize::try_from(num_passes.unwrap_or(DEFAULT_NUM_PASSES))
            .map_err(|_| PeelArgsError::ZeroPasses)?;

        let num_loops = NonZeroUsize::try_from(num_loops.unwrap_or(DEFAULT_NUM_LOOPS))
            .map_err(|_| PeelArgsError::ZeroLoops)?;

        // Set up the iono timeblocks. These break up the input data timesteps
        // (which may be averaged into timeblocks) into groups, each of which
        // will be peeled together.
        let iono_time_average_factor = {
            let default_time_average_factor = parse_time_average_factor(
                Some(input_vis_params.time_res),
                Some(DEFAULT_IONO_TIME_AVERAGE),
                NonZeroUsize::new(1).unwrap(),
            )
            .unwrap_or(NonZeroUsize::new(1).unwrap());

            let f = parse_time_average_factor(
                Some(input_vis_params.time_res),
                iono_time_average.as_deref(),
                default_time_average_factor,
            )
            .map_err(|e| match e {
                AverageFactorError::Zero => PeelArgsError::IonoTimeFactorZero,
                AverageFactorError::NotInteger => PeelArgsError::IonoTimeFactorNotInteger,
                AverageFactorError::NotIntegerMultiple { out, inp } => {
                    PeelArgsError::IonoTimeResNotMultiple { out, inp }
                }
                AverageFactorError::Parse(e) => PeelArgsError::ParseIonoTimeAverageFactor(e),
            })?;

            // Check that the factor is not too big.
            if f.get() > input_vis_params.timeblocks.len() {
                format!(
                    "Cannot average {f} timeblocks; only {} are being used. Capping.",
                    input_vis_params.timeblocks.len()
                )
                .warn();
                NonZeroUsize::new(input_vis_params.timeblocks.len())
                    .expect("timeblocks is Vec1, which cannot be empty")
            } else {
                f
            }
        };
        let iono_timeblocks = timesteps_to_timeblocks(
            &input_vis_params.timeblocks.mapped_ref(|tb| tb.median),
            input_vis_params.time_res,
            iono_time_average_factor,
            None,
        );

        // Set up the chanblocks.
        let iono_freq_average_factor = {
            let default_freq_average_factor = parse_freq_average_factor(
                Some(input_vis_params.spw.freq_res),
                Some(DEFAULT_IONO_FREQ_AVERAGE),
                NonZeroUsize::new(1).unwrap(),
            )
            .unwrap_or(NonZeroUsize::new(1).unwrap());

            parse_freq_average_factor(
                Some(input_vis_params.spw.freq_res),
                iono_freq_average.as_deref(),
                default_freq_average_factor,
            )
            .map_err(|e| match e {
                AverageFactorError::Zero => PeelArgsError::IonoFreqFactorZero,
                AverageFactorError::NotInteger => PeelArgsError::IonoFreqFactorNotInteger,
                AverageFactorError::NotIntegerMultiple { out, inp } => {
                    PeelArgsError::IonoFreqResNotMultiple { out, inp }
                }
                AverageFactorError::Parse(e) => PeelArgsError::ParseIonoFreqAverageFactor(e),
            })?
        };
        // limit iono freq average factor if it is too big.
        let iono_freq_average_factor =
            if iono_freq_average_factor.get() > input_vis_params.spw.chanblocks.len() {
                format!(
                    "Cannot average {} channels; only {} are being used. Capping.",
                    iono_freq_average_factor,
                    input_vis_params.spw.chanblocks.len()
                )
                .warn();
                NonZeroUsize::new(input_vis_params.spw.chanblocks.len())
                    .expect("no channels should've been checked earlier")
            } else {
                iono_freq_average_factor
            };

        let mut low_res_spws = {
            let spw = &input_vis_params.spw;
            let all_freqs = {
                let n = spw.chanblocks.len() + spw.flagged_chanblock_indices.len();
                let mut freqs = Vec::with_capacity(n);
                let first_freq = spw.first_freq.round() as u64;
                let freq_res = spw.freq_res.round() as u64;
                for i in 0..n as u64 {
                    freqs.push(first_freq + freq_res * i);
                }
                freqs
            };

            channels_to_chanblocks(
                &all_freqs,
                spw.freq_res.round() as u64,
                iono_freq_average_factor,
                &HashSet::new(),
            )
        };
        assert_eq!(
            low_res_spws.len(),
            1,
            "There should only be 1 low-res SPW, because there's only 1 high-res SPW"
        );
        let low_res_spw = low_res_spws.swap_remove(0);
        let n_low_freqs = low_res_spw.get_all_freqs().len();
        let n_input_freqs = input_vis_params.spw.get_all_freqs().len();
        assert_eq!(
            n_low_freqs,
            div_ceil(n_input_freqs, iono_freq_average_factor.get()),
            "low chans (flagged+unflagged) {} * iono_freq_average_factor {} != input chans (flagged+unflagged) {}.",
            n_low_freqs,
            iono_freq_average_factor.get(),
            n_input_freqs,
        );

        // Parse vis and iono const outputs.
        let (vis_outputs, iono_outputs) = {
            let mut vis_outputs = vec![];
            let mut iono_outputs = vec![];
            match outputs {
                // Defaults.
                None => {
                    let pb = PathBuf::from(DEFAULT_OUTPUT_PEEL_FILENAME);
                    pb.extension()
                        .and_then(|os_str| os_str.to_str())
                        .and_then(|s| VisOutputType::from_str(s).ok())
                        // Tests should pick up a bad default filename.
                        .expect("DEFAULT_OUTPUT_PEEL_FILENAME has an unhandled extension!");
                    vis_outputs.push(pb);
                    // TODO: Type this and clean up
                    let pb = PathBuf::from(DEFAULT_OUTPUT_IONO_CONSTS);
                    if pb.extension().and_then(|os_str| os_str.to_str()) != Some("json") {
                        // Tests should pick up a bad default filename.
                        panic!("DEFAULT_OUTPUT_IONO_CONSTS has an unhandled extension!");
                    }
                    iono_outputs.push(pb);
                }
                Some(os) => {
                    // Just find the .json files; other code will parse the
                    // visibility outputs.
                    for file in os {
                        let ext = file.extension().and_then(|os_str| os_str.to_str());
                        match ext.map(|s| s == "json") {
                            Some(true) => {
                                iono_outputs.push(file);
                            }
                            _ => {
                                vis_outputs.push(file);
                            }
                        }
                    }
                }
            };
            (vis_outputs, iono_outputs)
        };
        if vis_outputs.len() + iono_outputs.len() == 0 {
            return Err(PeelArgsError::NoOutput.into());
        }

        let output_vis_params = if vis_outputs.is_empty() {
            None
        } else {
            let params = OutputVisArgs {
                outputs: Some(vis_outputs),
                output_vis_time_average,
                output_vis_freq_average,
                output_autos: input_vis_params.using_autos,
            }
            .parse(
                input_vis_params.time_res,
                input_vis_params.spw.freq_res,
                &input_vis_params.timeblocks.mapped_ref(|tb| tb.median),
                output_smallest_contiguous_band,
                DEFAULT_OUTPUT_PEEL_FILENAME,
                Some("peeled"),
            )?;
            Some(params)
        };

        let tile_baseline_flags = &input_vis_params.tile_baseline_flags;
        let flagged_tiles = &tile_baseline_flags.flagged_tiles;

        let unflagged_tile_xyzs: Vec<XyzGeodetic> = obs_context
            .tile_xyzs
            .iter()
            .enumerate()
            .filter(|(tile_index, _)| !flagged_tiles.contains(tile_index))
            .map(|(_, xyz)| *xyz)
            .collect();

        // Set baseline weights from UVW cuts. Use a lambda from the centroid
        // frequency if UVW cutoffs are specified as wavelengths.
        let freq_centroid = obs_context
            .fine_chan_freqs
            .iter()
            .map(|&u| u as f64)
            .sum::<f64>()
            / obs_context.fine_chan_freqs.len() as f64;
        let lambda = marlu::constants::VEL_C / freq_centroid;
        let uvw_min_metres = {
            let (quantity, unit) = parse_wavelength(uvw_min.as_deref().unwrap_or(DEFAULT_UVW_MIN))
                .map_err(PeelArgsError::ParseUvwMin)?;
            match unit {
                WavelengthUnit::M => quantity,
                WavelengthUnit::L => quantity * lambda,
            }
        };
        let uvw_max_metres = match uvw_max {
            None => f64::INFINITY,
            Some(s) => {
                let (quantity, unit) = parse_wavelength(&s).map_err(PeelArgsError::ParseUvwMax)?;
                match unit {
                    WavelengthUnit::M => quantity,
                    WavelengthUnit::L => quantity * lambda,
                }
            }
        };
        let short_baseline_sigma = short_baseline_sigma.unwrap_or(DEFAULT_SHORT_BASELINE_SIGMA);
        let convergence = convergence.unwrap_or(DEFAULT_CONVERGENCE);

        let num_unflagged_baselines = {
            let uvws = xyzs_to_cross_uvws(
                &unflagged_tile_xyzs,
                obs_context.phase_centre.to_hadec(lmst_rad),
            );
            let uvw_min = uvw_min_metres.powi(2);
            let uvw_max = uvw_max_metres.powi(2);
            let mut num_unflagged_baselines = 0;
            for UVW { u, v, w } in uvws {
                let uvw_length = u.powi(2) + v.powi(2) + w.powi(2);
                if uvw_length > uvw_min && uvw_length < uvw_max {
                    num_unflagged_baselines += 1;
                }
            }
            num_unflagged_baselines
        };
        assert!(num_unflagged_baselines > 0, "All baselines were cut off");

        let num_sources_to_iono_subtract =
            num_sources_to_iono_subtract.unwrap_or(source_list.len());

        let mut peel_printer = InfoPrinter::new("Peeling set up".into());
        peel_printer.push_block(vec![
            format!("Subtracting {} sources", source_list.len()).into(),
            format!(
                "Ionospheric subtracting {} sources",
                num_sources_to_iono_subtract
            )
            .into(),
        ]);
        if num_sources_to_iono_subtract > 0 {
            let mut block = vec![];
            block.push("Finding ionospheric offsets with data at:".into());
            if iono_time_average_factor.get() == 1 {
                block.push(format!("- {}", input_vis_params.time_res).into());
            } else {
                block.push(
                    format!(
                        "- {} (averaging {}x)",
                        input_vis_params.time_res * iono_time_average_factor.get() as i64,
                        iono_time_average_factor
                    )
                    .into(),
                );
            }
            if iono_freq_average_factor.get() == 1 {
                block.push(format!("- {} kHz", input_vis_params.spw.freq_res / 1e3).into());
            } else {
                block.push(
                    format!(
                        "- {} kHz (averaging {}x)",
                        input_vis_params.spw.freq_res / 1e3 * iono_freq_average_factor.get() as f64,
                        iono_freq_average_factor
                    )
                    .into(),
                );
            }
            block.push(
                format!(
                    "- {num_passes} passes of {num_loops} loops at {convergence:.2} convergence"
                )
                .into(),
            );
            peel_printer.push_block(block);
        }

        peel_printer.display();
        display_warnings();

        let peel_loop_params = PeelLoopParams {
            num_passes,
            num_loops,
            convergence,
        };

        let peel_weight_params = PeelWeightParams {
            uvw_min_metres,
            uvw_max_metres,
            short_baseline_sigma,
        };

        Ok(PeelParams {
            input_vis_params,
            output_vis_params,
            iono_outputs,
            beam,
            source_list,
            modelling_params,
            // TODO: need both of these?
            iono_timeblocks,
            iono_time_average_factor,
            low_res_spw,
            peel_weight_params,
            peel_loop_params,
            num_sources_to_iono_subtract,
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

impl PeelCliArgs {
    fn merge(self, other: Self) -> Self {
        Self {
            num_sources_to_iono_subtract: self
                .num_sources_to_iono_subtract
                .or(other.num_sources_to_iono_subtract),
            num_passes: self.num_passes.or(other.num_passes),
            num_loops: self.num_loops.or(other.num_loops),
            iono_time_average: self.iono_time_average.or(other.iono_time_average),
            iono_freq_average: self.iono_freq_average.or(other.iono_freq_average),
            uvw_min: self.uvw_min.or(other.uvw_min),
            uvw_max: self.uvw_max.or(other.uvw_max),
            short_baseline_sigma: self.short_baseline_sigma.or(other.short_baseline_sigma),
            convergence: self.convergence.or(other.convergence),
            outputs: self.outputs.or(other.outputs),
            output_vis_time_average: self
                .output_vis_time_average
                .or(other.output_vis_time_average),
            output_vis_freq_average: self
                .output_vis_freq_average
                .or(other.output_vis_freq_average),
            output_smallest_contiguous_band: self.output_smallest_contiguous_band
                || other.output_smallest_contiguous_band,
        }
    }
}
