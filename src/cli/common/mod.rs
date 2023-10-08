// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Common arguments for command-line interfaces. Here, we abstract many aspects
//! of `hyperdrive`, e.g. the `di-calibrate` and `vis-subtract` subcommands both
//! take visibilities as input, so the same vis input arguments are shared
//! between them.

mod beam;
mod input_vis;
mod printers;
#[cfg(test)]
mod tests;

pub(super) use beam::BeamArgs;
pub(super) use input_vis::{InputVisArgs, InputVisArgsError};
pub(super) use printers::InfoPrinter;
pub(crate) use printers::{display_warnings, Warn};

use std::{num::NonZeroUsize, path::PathBuf, str::FromStr};

use clap::Parser;
use hifitime::{Duration, Epoch};
use itertools::Itertools;
use log::{trace, Level::Trace};
use marlu::RADec;
use serde::{Deserialize, Serialize};
use strum::IntoEnumIterator;
use strum_macros::{Display, EnumIter, EnumString};
use vec1::Vec1;

use super::HyperdriveError;
use crate::{
    averaging::{
        parse_freq_average_factor, parse_time_average_factor, timesteps_to_timeblocks,
        AverageFactorError,
    },
    beam::Beam,
    constants::{
        DEFAULT_CUTOFF_DISTANCE, DEFAULT_VETO_THRESHOLD, MWA_HEIGHT_M, MWA_LAT_DEG, MWA_LONG_DEG,
    },
    io::{
        get_single_match_from_glob,
        write::{can_write_to_file, VisOutputType, VIS_OUTPUT_EXTENSIONS},
    },
    model::ModelDevice,
    params::{ModellingParams, OutputVisParams},
    srclist::{
        read::read_source_list_file, veto_sources, ComponentCounts, ReadSourceListError,
        SourceList, SourceListType, SOURCE_LIST_TYPES_COMMA_SEPARATED,
    },
    MODEL_DEVICE,
};

lazy_static::lazy_static! {
    pub(super) static ref ARG_FILE_TYPES_COMMA_SEPARATED: String = ArgFileTypes::iter().join(", ");

    pub(super) static ref ARG_FILE_HELP: String =
        format!("All arguments may be specified in a file. Any CLI arguments override arguments set in the file. Supported formats: {}", *ARG_FILE_TYPES_COMMA_SEPARATED);

    pub(super) static ref ARRAY_POSITION_HELP: String =
        format!("The Earth longitude, latitude, and height of the instrumental array [degrees, degrees, meters]. Default (MWA): ({MWA_LONG_DEG}°, {MWA_LAT_DEG}°, {MWA_HEIGHT_M}m)");

    pub(super) static ref SOURCE_LIST_TYPE_HELP: String =
        format!("The type of sky-model source list. Valid types are: {}. If not specified, all types are attempted", *SOURCE_LIST_TYPES_COMMA_SEPARATED);

    pub(super) static ref SOURCE_DIST_CUTOFF_HELP: String =
        format!("Specifies the maximum distance from the phase centre a source can be [degrees]. Default: {DEFAULT_CUTOFF_DISTANCE}");

    pub(super) static ref VETO_THRESHOLD_HELP: String =
        format!("Specifies the minimum Stokes XX+YY a source must have before it gets vetoed [Jy]. Default: {DEFAULT_VETO_THRESHOLD}");

    pub(super) static ref SOURCE_LIST_INPUT_TYPE_HELP: String =
        format!("Specifies the type of the input source list. Currently supported types: {}", *SOURCE_LIST_TYPES_COMMA_SEPARATED);

    pub(super) static ref SOURCE_LIST_OUTPUT_TYPE_HELP: String =
        format!("Specifies the type of the output source list. May be required depending on the output filename. Currently supported types: {}",
                *SOURCE_LIST_TYPES_COMMA_SEPARATED);

    pub(super) static ref SRCLIST_BY_BEAM_OUTPUT_TYPE_HELP: String =
        format!("Specifies the type of the output source list. If not specified, the input source list type is used. Currently supported types: {}",
                *SOURCE_LIST_TYPES_COMMA_SEPARATED);
}

#[derive(Debug, Display, EnumIter, EnumString)]
pub(super) enum ArgFileTypes {
    #[strum(serialize = "toml")]
    Toml,
    #[strum(serialize = "json")]
    Json,
}

macro_rules! unpack_arg_file {
    ($arg_file:expr) => ({
        use std::{fs::File, io::Read, str::FromStr};

        use crate::cli::common::{ArgFileTypes, ARG_FILE_TYPES_COMMA_SEPARATED};

        debug!("Attempting to parse argument file {}", $arg_file.display());

        let mut contents = String::new();
        let arg_file_type = $arg_file
            .extension()
            .and_then(|e| e.to_str())
            .map(|e| e.to_lowercase())
            .and_then(|e| ArgFileTypes::from_str(&e).ok());

        match arg_file_type {
            Some(ArgFileTypes::Toml) => {
                debug!("Parsing toml file...");
                let mut fh = File::open(&$arg_file)?;
                fh.read_to_string(&mut contents)?;
                match toml::from_str(&contents) {
                    Ok(p) => p,
                    Err(err) => {
                        return Err(HyperdriveError::ArgFile(format!(
                            "Couldn't decode toml structure from {:?}:\n{err}",
                            $arg_file
                        )))
                    }
                }
            }
            Some(ArgFileTypes::Json) => {
                debug!("Parsing json file...");
                let mut fh = File::open(&$arg_file)?;
                fh.read_to_string(&mut contents)?;
                match serde_json::from_str(&contents) {
                    Ok(p) => p,
                    Err(err) => {
                        return Err(HyperdriveError::ArgFile(format!(
                            "Couldn't decode json structure from {:?}:\n{err}",
                            $arg_file
                        )))
                    }
                }
            }

            _ => {
                return Err(HyperdriveError::ArgFile(format!(
                    "Argument file '{:?}' doesn't have a recognised file extension! Valid extensions are: {}", $arg_file, *ARG_FILE_TYPES_COMMA_SEPARATED)
                ))
            }
        }
    });
}

/// Arguments to be parsed for visibility outputs. Unlike other "arg" structs,
/// this one is not parsed by `clap`; this is to allow the help texts for
/// `hyperdrive` subcommands to better details what the output visibilities
/// represent (e.g. di-calibrate outputs model visibilities, whereas
/// vis-subtract outputs subtracted visibilities; attempting to have one set of
/// help text for both of these vis outputs is not as clear as just having the
/// curated help text specified in each of di-calibrate and vis-subtract).
#[derive(Debug, Clone, Default)]
pub(super) struct OutputVisArgs {
    pub(super) outputs: Option<Vec<PathBuf>>,
    pub(super) output_vis_time_average: Option<String>,
    pub(super) output_vis_freq_average: Option<String>,
    pub(super) output_autos: bool,
}

impl OutputVisArgs {
    pub(super) fn parse(
        self,
        input_vis_time_res: Duration,
        input_vis_freq_res_hz: f64,
        timestamps: &Vec1<Epoch>,
        write_smallest_contiguous_band: bool,
        default_output_filename: &str,
        vis_description: Option<&str>,
    ) -> Result<OutputVisParams, HyperdriveError> {
        let OutputVisArgs {
            outputs,
            output_vis_time_average,
            output_vis_freq_average,
            output_autos,
        } = self;

        let (time_average_factor, freq_average_factor) = {
            // Parse and verify user input (specified resolutions must
            // evenly divide the input data's resolutions).
            let time_factor = parse_time_average_factor(
                Some(input_vis_time_res),
                output_vis_time_average.as_deref(),
                NonZeroUsize::new(1).unwrap(),
            )
            .map_err(|e| match e {
                AverageFactorError::Zero => HyperdriveError::Generic(
                    "The output visibility time average factor cannot be 0".to_string(),
                ),
                AverageFactorError::NotInteger => HyperdriveError::Generic(
                    "The output visibility time average factor isn't an integer".to_string(),
                ),
                AverageFactorError::NotIntegerMultiple { out, inp } => HyperdriveError::Generic(format!("The output visibility time resolution isn't a multiple of input data's: {out} seconds vs {inp} seconds")),
                AverageFactorError::Parse(e) => HyperdriveError::Generic(format!("Error when parsing the output visibility time average factor: {e}")),
            })?;
            let freq_factor =
                parse_freq_average_factor(Some(input_vis_freq_res_hz), output_vis_freq_average.as_deref(), NonZeroUsize::new(1).unwrap())
                    .map_err(|e| match e {
                        AverageFactorError::Zero => {
                            HyperdriveError::Generic(
                    "The output visibility freq. average factor cannot be 0".to_string(),
                )                        }
                        AverageFactorError::NotInteger => {
                            HyperdriveError::Generic(
                    "The output visibility freq. average factor isn't an integer".to_string(),
                )                        }
                        AverageFactorError::NotIntegerMultiple { out, inp } => {
                            HyperdriveError::Generic(format!("The output visibility freq. resolution isn't a multiple of input data's: {out} seconds vs {inp} seconds"))
                        }
                        AverageFactorError::Parse(e) => {
                            HyperdriveError::Generic(format!("Error when parsing the output visibility freq. average factor: {e}"))
                        }
                    })?;

            (time_factor, freq_factor)
        };

        let mut vis_printer = if let Some(vis_description) = vis_description {
            InfoPrinter::new(format!("Output {vis_description} vis info").into())
        } else {
            InfoPrinter::new("Output vis info".into())
        };

        let output_files = {
            let outputs = outputs.unwrap_or_else(|| vec![PathBuf::from(default_output_filename)]);
            let mut valid_outputs = Vec::with_capacity(outputs.len());
            for file in outputs {
                // Is the output file type supported?
                let ext = file.extension().and_then(|os_str| os_str.to_str());
                match ext.and_then(|s| VisOutputType::from_str(s).ok()) {
                    Some(t) => {
                        can_write_to_file(&file)?;
                        valid_outputs.push((file, t));
                    }
                    None => {
                        return Err(HyperdriveError::VisWrite(format!(
                            "An invalid output format was specified ({}). Supported:\n{}",
                            ext.unwrap_or("<no extension>"),
                            *VIS_OUTPUT_EXTENSIONS
                        )))
                    }
                }
            }

            Vec1::try_from_vec(valid_outputs).expect("cannot be empty")
        };

        let vis_str = output_files.iter().map(|(pb, _)| pb.display()).join(", ");
        if let Some(vis_description) = vis_description {
            vis_printer
                .push_line(format!("Writing {vis_description} visibilities to: {vis_str}").into());
        } else {
            vis_printer.push_line(format!("Writing visibilities to: {vis_str}").into());
        }

        let mut block = vec![];
        if time_average_factor.get() != 1 || freq_average_factor.get() != 1 {
            block.push(
                format!(
                    "Time averaging  {}x ({}s)",
                    time_average_factor,
                    input_vis_time_res.to_seconds() * time_average_factor.get() as f64
                )
                .into(),
            );

            block.push(
                format!(
                    "Freq. averaging {}x ({}kHz)",
                    freq_average_factor,
                    input_vis_freq_res_hz * freq_average_factor.get() as f64 / 1000.0
                )
                .into(),
            );
        }
        vis_printer.push_block(block);
        if write_smallest_contiguous_band {
            vis_printer.push_line("Writing the smallest possible contiguous band, ignoring any flagged fine channels at the edges of the SPW".into());
        }
        if output_autos {
            vis_printer.push_line("Writing out auto-correlations".into());
        } else {
            vis_printer.push_line("Not writing out auto-correlations".into());
        }
        vis_printer.display();

        let timeblocks =
            timesteps_to_timeblocks(timestamps, input_vis_time_res, time_average_factor, None);

        Ok(OutputVisParams {
            output_files,
            output_time_average_factor: time_average_factor,
            output_freq_average_factor: freq_average_factor,
            output_autos,
            output_timeblocks: timeblocks,
            write_smallest_contiguous_band,
        })
    }
}

#[derive(Parser, Debug, Clone, Default, Serialize, Deserialize)]
pub(super) struct SkyModelWithVetoArgs {
    /// Path to the sky-model source list file.
    #[clap(short, long, help_heading = "SKY MODEL")]
    pub(super) source_list: Option<String>,

    #[clap(long, help = SOURCE_LIST_TYPE_HELP.as_str(), help_heading = "SKY MODEL")]
    pub(super) source_list_type: Option<String>,

    /// The number of sources to use in the source list. The default is to use
    /// them all. Example: If 1000 sources are specified here, then the top 1000
    /// sources are used (based on their flux densities after the beam
    /// attenuation) within the specified source distance cutoff.
    #[clap(short, long, help_heading = "SKY MODEL")]
    pub(super) num_sources: Option<usize>,

    #[clap(long, help = SOURCE_DIST_CUTOFF_HELP.as_str(), help_heading = "SKY MODEL")]
    pub(super) source_dist_cutoff: Option<f64>,

    #[clap(long, help = VETO_THRESHOLD_HELP.as_str(), help_heading = "SKY MODEL")]
    pub(super) veto_threshold: Option<f64>,
}

impl SkyModelWithVetoArgs {
    pub(super) fn merge(self, other: Self) -> Self {
        Self {
            source_list: self.source_list.or(other.source_list),
            source_list_type: self.source_list_type.or(other.source_list_type),
            num_sources: self.num_sources.or(other.num_sources),
            source_dist_cutoff: self.source_dist_cutoff.or(other.source_dist_cutoff),
            veto_threshold: self.veto_threshold.or(other.veto_threshold),
        }
    }

    pub(super) fn parse(
        self,
        phase_centre: RADec,
        lst_rad: f64,
        array_latitude_rad: f64,
        veto_freqs_hz: &[f64],
        beam: &dyn Beam,
    ) -> Result<SourceList, ReadSourceListError> {
        let Self {
            source_list,
            source_list_type,
            num_sources,
            source_dist_cutoff,
            veto_threshold,
        } = self;

        let mut printer = InfoPrinter::new("Sky model info".into());

        // Handle the source list argument.
        let sl_pb: PathBuf = match source_list {
            None => return Err(ReadSourceListError::NoSourceList),
            Some(sl) => {
                // If the specified source list file can't be found, treat
                // it as a glob and expand it to find a match.
                let pb = PathBuf::from(&sl);
                if pb.exists() {
                    pb
                } else {
                    get_single_match_from_glob(&sl)?
                }
            }
        };

        // Read the source list file. If the type was manually specified,
        // use that, otherwise the reading code will try all available
        // kinds.
        let sl_type_not_specified = source_list_type.is_none();
        let sl_type = source_list_type.and_then(|t| SourceListType::from_str(t.as_ref()).ok());
        let (mut sl, sl_type) = read_source_list_file(sl_pb, sl_type)?;

        let ComponentCounts {
            num_points,
            num_gaussians,
            num_shapelets,
            ..
        } = sl.get_counts();
        printer.push_block(vec![
            format!("Source list contains {} sources", sl.len()).into(),
            format!("({} components, {num_points} points, {num_gaussians} Gaussians, {num_shapelets} shapelets)", num_points + num_gaussians + num_shapelets).into()
        ]);

        // If the user didn't specify the source list type, then print out
        // what we found.
        if sl_type_not_specified {
            trace!("Successfully parsed {}-style source list", sl_type);
        }

        trace!("Found {} sources in the source list", sl.len());
        // Veto any sources that may be troublesome, and/or cap the total number
        // of sources. If the user doesn't specify how many source-list sources
        // to use, then all sources are used.
        if num_sources == Some(0) || sl.is_empty() {
            return Err(ReadSourceListError::NoSources);
        }
        veto_sources(
            &mut sl,
            phase_centre,
            lst_rad,
            array_latitude_rad,
            veto_freqs_hz,
            beam,
            num_sources,
            source_dist_cutoff.unwrap_or(DEFAULT_CUTOFF_DISTANCE),
            veto_threshold.unwrap_or(DEFAULT_VETO_THRESHOLD),
        )?;
        if sl.is_empty() {
            return Err(ReadSourceListError::NoSourcesAfterVeto);
        }

        {
            let ComponentCounts {
                num_points,
                num_gaussians,
                num_shapelets,
                ..
            } = sl.get_counts();
            let num_components = num_points + num_gaussians + num_shapelets;
            printer.push_block(vec![
                format!(
                    "Using {} sources with a total of {num_components} components",
                    sl.len()
                )
                .into(),
                format!(
                    "{num_points} points, {num_gaussians} Gaussians, {num_shapelets} shapelets"
                )
                .into(),
            ]);
            if num_components > 10000 {
                "Using more than 10,000 sky model components!".warn();
            }
            if log::log_enabled!(Trace) {
                trace!("Using sources:");
                let mut v = Vec::with_capacity(5);
                for source in sl.keys() {
                    if v.len() == 5 {
                        trace!("  {v:?}");
                        v.clear();
                    }
                    v.push(source);
                }
                if !v.is_empty() {
                    trace!("  {v:?}");
                }
            }
        }

        printer.display();

        Ok(sl)
    }
}

#[derive(Parser, Debug, Clone, Copy, Default, Serialize, Deserialize)]
pub(super) struct ModellingArgs {
    /// If specified, don't precess the array to J2000. We assume that sky-model
    /// sources are specified in the J2000 epoch.
    #[clap(long, help_heading = "MODELLING")]
    #[serde(default)]
    pub(super) no_precession: bool,

    /// Use the CPU for visibility generation. This is deliberately made
    /// non-default because using a GPU is much faster.
    #[cfg(any(feature = "cuda", feature = "hip"))]
    #[clap(long, help_heading = "MODELLING")]
    #[serde(default)]
    pub(super) cpu: bool,
}

impl ModellingArgs {
    pub(super) fn merge(self, other: Self) -> Self {
        Self {
            no_precession: self.no_precession || other.no_precession,
            #[cfg(any(feature = "cuda", feature = "hip"))]
            cpu: self.cpu || other.cpu,
        }
    }

    pub(super) fn parse(self) -> ModellingParams {
        let ModellingArgs {
            no_precession,
            #[cfg(any(feature = "cuda", feature = "hip"))]
            cpu,
        } = self;

        #[cfg(any(feature = "cuda", feature = "hip"))]
        if cpu {
            MODEL_DEVICE.store(ModelDevice::Cpu);
        }

        let d = MODEL_DEVICE.load();
        let mut printer = InfoPrinter::new("Sky- and beam-modelling info".into());
        let mut block = vec![];
        match d {
            ModelDevice::Cpu => {
                block.push(format!("Using CPU with {} precision", d.get_precision()).into());
                block.push(crate::model::get_cpu_info().into());
            }

            #[cfg(any(feature = "cuda", feature = "hip"))]
            ModelDevice::Gpu => {
                block.push(format!("Using GPU with {} precision", d.get_precision()).into());
                let (device_info, driver_info) = match crate::gpu::get_device_info() {
                    Ok(i) => i,
                    Err(e) => {
                        // For some reason, despite hyperdrive being compiled
                        // with the "cuda" or "hip" feature, we failed to get
                        // the device info. Maybe there's no GPU present. Either
                        // way, we cannot continue. I'd rather not have error
                        // handling here because (1) without the "cuda" or "hip"
                        // feature, this function will never fail on the CPU
                        // path, so adding error handling means the caller would
                        // have to handle a `Result` uselessly and (2) if this
                        // "petty" display function fails, then we can't use the
                        // GPU for real work anyway.
                        #[cfg(feature = "cuda")]
                        eprintln!("Couldn't retrieve CUDA device info for device 0, is a device present? {e}");
                        #[cfg(feature = "hip")]
                        eprintln!("Couldn't retrieve HIP device info for device 0, is a device present? {e}");
                        std::process::exit(1);
                    }
                };
                #[cfg(feature = "cuda")]
                let device_type = "CUDA";
                #[cfg(feature = "hip")]
                let device_type = "HIP";
                block.push(
                    format!(
                        "{device_type} device: {} (capability {}, {} MiB)",
                        device_info.name, device_info.capability, device_info.total_global_mem
                    )
                    .into(),
                );
                block.push(
                    format!(
                        "{device_type} driver: {}, runtime: {}",
                        driver_info.driver_version, driver_info.runtime_version
                    )
                    .into(),
                );
            }
        }
        printer.push_block(block);
        printer.display();

        ModellingParams {
            apply_precession: !no_precession,
        }
    }
}
