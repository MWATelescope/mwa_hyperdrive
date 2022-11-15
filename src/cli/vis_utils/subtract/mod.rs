// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Given input data, a sky model and specific sources, subtract those specific
//! sources from the input data and write them out.

mod error;

pub(crate) use error::VisSubtractError;
#[cfg(test)]
mod tests;

use std::{collections::HashSet, ops::Deref, path::PathBuf, str::FromStr, thread};

use clap::Parser;
use crossbeam_channel::{bounded, Receiver, Sender};
use crossbeam_utils::atomic::AtomicCell;
use hifitime::Duration;
use indicatif::{MultiProgress, ProgressBar, ProgressDrawTarget, ProgressStyle};
use itertools::Itertools;
use log::{debug, info, trace, warn};
use marlu::{precession::precess_time, Jones, LatLngHeight, MwaObsContext};
use ndarray::{prelude::*, ArcArray2};
use scopeguard::defer_on_unwind;
use vec1::{vec1, Vec1};

use crate::{
    averaging::{
        parse_freq_average_factor, parse_time_average_factor, timesteps_to_timeblocks,
        AverageFactorError,
    },
    beam::{create_fee_beam_object, create_no_beam_object, Beam, Delays},
    constants::{DEFAULT_CUTOFF_DISTANCE, DEFAULT_VETO_THRESHOLD},
    context::ObsContext,
    filenames::InputDataTypes,
    glob::*,
    help_texts::{
        ARRAY_POSITION_HELP, DIPOLE_DELAYS_HELP, SOURCE_DIST_CUTOFF_HELP as sdc_help,
        SOURCE_LIST_TYPE_HELP, VETO_THRESHOLD_HELP as vt_help,
    },
    math::TileBaselineFlags,
    messages,
    model::ModellerInfo,
    srclist::{read::read_source_list_file, veto_sources, SourceList, SourceListType},
    vis_io::{
        read::{MsReader, UvfitsReader, VisInputType, VisRead},
        write::{can_write_to_file, write_vis, VisOutputType, VisTimestep, VIS_OUTPUT_EXTENSIONS},
    },
    HyperdriveError,
};

pub(crate) const DEFAULT_OUTPUT_VIS_FILENAME: &str = "hyp_subtracted.uvfits";

lazy_static::lazy_static! {
    static ref OUTPUTS_HELP: String =
        format!("Paths to the output visibility files. Supported formats: {}. Default: {}", *VIS_OUTPUT_EXTENSIONS, DEFAULT_OUTPUT_VIS_FILENAME);

    static ref SOURCE_DIST_CUTOFF_HELP: String =
        format!("{}. Only useful if subtraction is inverted.", *sdc_help);

    static ref VETO_THRESHOLD_HELP: String =
        format!("{}. Only useful if subtraction is inverted.", *vt_help);
}

#[derive(Parser, Debug, Default)]
pub struct VisSubtractArgs {
    /// Paths to the input data files to have visibilities subtracted. These can
    /// include a metafits file, a measurement set and/or uvfits files.
    #[clap(short, long, multiple_values(true), help_heading = "INPUT FILES")]
    data: Vec<String>,

    /// Path to the sky-model source list used for simulation.
    #[clap(short, long, help_heading = "INPUT FILES")]
    source_list: String,

    #[clap(long, help = SOURCE_LIST_TYPE_HELP.as_str(), help_heading = "INPUT FILES")]
    source_list_type: Option<String>,

    /// The timesteps to use from the input data. The default is to use all
    /// timesteps, including flagged ones.
    #[clap(long, multiple_values(true), help_heading = "INPUT FILES")]
    timesteps: Option<Vec<usize>>,

    /// Use a DUT1 value of 0 seconds rather than what is in the input data.
    #[clap(long, help_heading = "INPUT FILES")]
    ignore_dut1: bool,

    #[clap(
        short = 'o',
        long,
        multiple_values(true),
        help = OUTPUTS_HELP.as_str(),
        help_heading = "OUTPUT FILES"
    )]
    outputs: Vec<PathBuf>,

    /// When writing out subtracted visibilities, average this many timesteps
    /// together. Also supports a target time resolution (e.g. 8s). The value
    /// must be a multiple of the input data's time resolution. The default is
    /// to preserve the input data's time resolution. e.g. If the input data is
    /// in 0.5s resolution and this variable is 4, then we average 2s worth of
    /// subtracted data together before writing the data out. If the variable is
    /// instead 4s, then 8 subtracted timesteps are averaged together before
    /// writing the data out.
    #[clap(long, help_heading = "OUTPUT FILES")]
    time_average: Option<String>,

    /// When writing out subtracted visibilities, average this many fine freq.
    /// channels together. Also supports a target freq. resolution (e.g. 80kHz).
    /// The value must be a multiple of the input data's freq. resolution. The
    /// default is to preserve the input data's freq. resolution. e.g. If the
    /// input data is in 40kHz resolution and this variable is 4, then we
    /// average 160kHz worth of subtracted data together before writing the data
    /// out. If the variable is instead 80kHz, then 2 subtracted fine freq.
    /// channels are averaged together before writing the data out.
    #[clap(long, help_heading = "OUTPUT FILES")]
    freq_average: Option<String>,

    /// The names of the sources in the sky-model source list that will be
    /// subtracted from the input data.
    #[clap(long, multiple_values(true), help_heading = "SKY-MODEL SOURCES")]
    sources_to_subtract: Vec<String>,

    /// Invert the subtraction; sources *not* specified in sources-to-subtract
    /// will be subtracted from the input data.
    #[clap(short, long, help_heading = "SKY-MODEL SOURCES")]
    invert: bool,

    /// The number of sources to use in the source list. Only useful if
    /// subtraction is inverted. The default is to use all sources in the source
    /// list. Example: If 1000 sources are specified here, then the top 1000
    /// sources *after* removing specified sources are subtracted. Standard veto
    /// rules apply (sources are ranked based on their flux densities after the
    /// beam attenuation, must be within the specified source distance cutoff
    /// and above the horizon).
    #[clap(short, long, help_heading = "SKY-MODEL SOURCES")]
    num_sources: Option<usize>,

    #[clap(long, help = SOURCE_DIST_CUTOFF_HELP.as_str(), help_heading = "SKY-MODEL SOURCES")]
    source_dist_cutoff: Option<f64>,

    #[clap(long, help = VETO_THRESHOLD_HELP.as_str(), help_heading = "SKY-MODEL SOURCES")]
    veto_threshold: Option<f64>,

    /// Should we use a beam? Default is to use the FEE beam.
    #[clap(long, help_heading = "MODEL PARAMETERS")]
    no_beam: bool,

    /// The path to the HDF5 MWA FEE beam file. If not specified, this must be
    /// provided by the MWA_BEAM_FILE environment variable.
    #[clap(long, help_heading = "MODEL PARAMETERS")]
    beam_file: Option<PathBuf>,

    /// Pretend that all MWA dipoles are alive and well, ignoring whatever is in
    /// the metafits file.
    #[clap(long, help_heading = "MODEL PARAMETERS")]
    unity_dipole_gains: bool,

    #[clap(long, multiple_values(true), help = DIPOLE_DELAYS_HELP.as_str(), help_heading = "MODEL PARAMETERS")]
    delays: Option<Vec<u32>>,

    #[clap(
        long, help = ARRAY_POSITION_HELP.as_str(), help_heading = "MODEL PARAMETERS",
        number_of_values = 3,
        allow_hyphen_values = true,
        value_names = &["LONG_RAD", "LAT_RAD", "HEIGHT_M"]
    )]
    array_position: Option<Vec<f64>>,

    /// If specified, don't precess the array to J2000. We assume that sky-model
    /// sources are specified in the J2000 epoch.
    #[clap(long, help_heading = "MODEL PARAMETERS")]
    no_precession: bool,

    /// Use the CPU for visibility generation. This is deliberately made
    /// non-default because using a GPU is much faster.
    #[cfg(feature = "cuda")]
    #[clap(long, help_heading = "MODEL PARAMETERS")]
    cpu: bool,

    /// Don't draw progress bars.
    #[clap(long, help_heading = "USER INTERFACE")]
    no_progress_bars: bool,
}

impl VisSubtractArgs {
    pub fn run(self, dry_run: bool) -> Result<(), HyperdriveError> {
        vis_subtract(self, dry_run)?;
        Ok(())
    }
}

fn vis_subtract(args: VisSubtractArgs, dry_run: bool) -> Result<(), VisSubtractError> {
    debug!("{:#?}", args);

    // Expose all the struct fields to ensure they're all used.
    let VisSubtractArgs {
        data,
        source_list,
        source_list_type,
        timesteps,
        ignore_dut1,
        outputs,
        time_average,
        freq_average,
        sources_to_subtract,
        invert,
        num_sources,
        source_dist_cutoff,
        veto_threshold,
        no_beam,
        beam_file,
        unity_dipole_gains,
        delays,
        array_position,
        no_precession,
        #[cfg(feature = "cuda")]
            cpu: use_cpu_for_modelling,
        no_progress_bars,
    } = args;

    // If we're going to use a GPU for modelling, get the device info so we
    // can ensure a CUDA-capable device is available, and so we can report
    // it to the user later.
    #[cfg(feature = "cuda")]
    let modeller_info = if use_cpu_for_modelling {
        ModellerInfo::Cpu
    } else {
        let (device_info, driver_info) = crate::cuda::get_device_info()?;
        ModellerInfo::Cuda {
            device_info,
            driver_info,
        }
    };
    #[cfg(not(feature = "cuda"))]
    let modeller_info = ModellerInfo::Cpu;

    // If we're not inverted but `sources_to_subtract` is empty, then there's
    // nothing to do.
    if !invert && sources_to_subtract.is_empty() {
        return Err(VisSubtractError::NoSources);
    }

    // Read in the source list and remove all but the specified sources.
    let source_list: SourceList = {
        // If the specified source list file can't be found, treat it as a glob
        // and expand it to find a match.
        let pb = PathBuf::from(&source_list);
        let pb = if pb.exists() {
            pb
        } else {
            get_single_match_from_glob(&source_list)?
        };

        // Read the source list file. If the type was manually specified,
        // use that, otherwise the reading code will try all available
        // kinds.
        let sl_type = source_list_type
            .as_ref()
            .and_then(|t| SourceListType::from_str(t.as_ref()).ok());
        let (sl, _) = match read_source_list_file(pb, sl_type) {
            Ok((sl, sl_type)) => (sl, sl_type),
            Err(e) => return Err(VisSubtractError::from(e)),
        };

        sl
    };
    debug!("Found {} sources in the source list", source_list.len());

    // Ensure that all specified sources are actually in the source list.
    for name in &sources_to_subtract {
        if !source_list.contains_key(name) {
            return Err(VisSubtractError::MissingSource {
                name: name.to_owned(),
            });
        }
    }

    // If the user supplied the array position, unpack it here.
    let array_position = match array_position {
        Some(pos) => {
            if pos.len() != 3 {
                return Err(VisSubtractError::BadArrayPosition { pos });
            }
            Some(LatLngHeight {
                longitude_rad: pos[0].to_radians(),
                latitude_rad: pos[1].to_radians(),
                height_metres: pos[2],
            })
        }
        None => None,
    };

    // Prepare an input data reader.
    let input_data_types = InputDataTypes::new(&data)?;
    let input_data: Box<dyn VisRead> = match (
        input_data_types.metafits,
        input_data_types.gpuboxes,
        input_data_types.mwafs,
        input_data_types.ms,
        input_data_types.uvfits,
    ) {
        // Valid input for reading a measurement set.
        (meta, None, None, Some(ms), None) => {
            // Only one MS is supported at the moment.
            let ms: PathBuf = if ms.len() > 1 {
                return Err(VisSubtractError::MultipleMeasurementSets(ms));
            } else {
                ms.first().clone()
            };

            // Ensure that there's only one metafits.
            let meta: Option<&PathBuf> = match meta.as_ref() {
                None => None,
                Some(m) => {
                    if m.len() > 1 {
                        return Err(VisSubtractError::MultipleMetafits(m.clone()));
                    } else {
                        Some(m.first())
                    }
                }
            };

            let input_data = MsReader::new(ms, meta, array_position)?;
            match input_data.get_obs_context().obsid {
                Some(o) => info!(
                    "Reading obsid {} from measurement set {}",
                    o,
                    input_data.ms.canonicalize()?.display()
                ),
                None => info!(
                    "Reading measurement set {}",
                    input_data.ms.canonicalize()?.display()
                ),
            }
            Box::new(input_data)
        }

        // Valid input for reading uvfits files.
        (meta, None, None, None, Some(uvfits)) => {
            // Only one uvfits is supported at the moment.
            let uvfits: PathBuf = if uvfits.len() > 1 {
                return Err(VisSubtractError::MultipleUvfits(uvfits));
            } else {
                uvfits.first().clone()
            };

            // Ensure that there's only one metafits.
            let meta: Option<&PathBuf> = match meta.as_ref() {
                None => None,
                Some(m) => {
                    if m.len() > 1 {
                        return Err(VisSubtractError::MultipleMetafits(m.clone()));
                    } else {
                        Some(m.first())
                    }
                }
            };

            let input_data = UvfitsReader::new(uvfits, meta)?;
            match input_data.get_obs_context().obsid {
                Some(o) => info!(
                    "Reading obsid {} from uvfits {}",
                    o,
                    input_data.uvfits.canonicalize()?.display()
                ),
                None => info!(
                    "Reading uvfits {}",
                    input_data.uvfits.canonicalize()?.display()
                ),
            }
            Box::new(input_data)
        }

        // The following matches are for invalid combinations of input
        // files. Make an error message for the user.
        (_, Some(_), _, _, _) => {
            let msg = "Received gpubox files, but these are not supported by vis-subtract.";
            return Err(VisSubtractError::InvalidDataInput(msg));
        }
        (_, _, Some(_), _, _) => {
            let msg = "Received mwaf files, but these are not supported by vis-subtract.";
            return Err(VisSubtractError::InvalidDataInput(msg));
        }
        (Some(_), None, None, None, None) => {
            let msg = "Received only a metafits file; a calibrated uvfits file or calibrated measurement set is required.";
            return Err(VisSubtractError::InvalidDataInput(msg));
        }
        (_, _, _, Some(_), Some(_)) => {
            let msg = "Received uvfits and measurement set files; this is not supported.";
            return Err(VisSubtractError::InvalidDataInput(msg));
        }
        (None, None, None, None, None) => return Err(VisSubtractError::NoInputData),
    };

    let obs_context = input_data.get_obs_context();
    let total_num_tiles = obs_context.get_total_num_tiles();
    let num_unflagged_tiles = obs_context.get_num_unflagged_tiles();
    let num_unflagged_cross_baselines = (num_unflagged_tiles * (num_unflagged_tiles - 1)) / 2;
    let flagged_tiles = obs_context
        .get_tile_flags(false, None)
        .expect("can't fail; no additional flags");
    let tile_baseline_flags = TileBaselineFlags::new(total_num_tiles, flagged_tiles);
    let vis_shape = (
        obs_context.fine_chan_freqs.len(),
        num_unflagged_cross_baselines,
    );

    // Set up the beam for modelling.
    let dipole_delays = match delays {
        // We have user-provided delays; check that they're are sensible,
        // regardless of whether we actually need them.
        Some(d) => {
            if d.len() != 16 || d.iter().any(|&v| v > 32) {
                return Err(VisSubtractError::BadDelays);
            }
            Some(Delays::Partial(d))
        }

        // No delays were provided; use whatever was in the input data.
        None => obs_context.dipole_delays.clone(),
    };

    let beam: Box<dyn Beam> = if no_beam {
        create_no_beam_object(obs_context.tile_xyzs.len())
    } else {
        let mut dipole_delays = dipole_delays.ok_or(VisSubtractError::NoDelays)?;
        let dipole_gains = if unity_dipole_gains {
            None
        } else {
            // If we don't have dipole gains from the input data, then
            // we issue a warning that we must assume no dead dipoles.
            if obs_context.dipole_gains.is_none() {
                match input_data.get_input_data_type() {
                    VisInputType::MeasurementSet => {
                        warn!("Measurement sets cannot supply dead dipole information.");
                        warn!("Without a metafits file, we must assume all dipoles are alive.");
                        warn!("This will make beam Jones matrices inaccurate in sky-model generation.");
                    }
                    VisInputType::Uvfits => {
                        warn!("uvfits files cannot supply dead dipole information.");
                        warn!("Without a metafits file, we must assume all dipoles are alive.");
                        warn!("This will make beam Jones matrices inaccurate in sky-model generation.");
                    }
                    VisInputType::Raw => unreachable!(),
                }
            }
            obs_context.dipole_gains.clone()
        };
        if dipole_gains.is_none() {
            // If we don't have dipole gains, we must assume all dipoles are
            // "alive". But, if any dipole delays are 32, then the beam code
            // will still ignore those dipoles. So use ideal dipole delays for
            // all tiles.
            let ideal_delays = dipole_delays.get_ideal_delays();

            // Warn the user if they wanted unity dipole gains but the ideal
            // dipole delays contain 32.
            if unity_dipole_gains && ideal_delays.iter().any(|&v| v == 32) {
                warn!("Some ideal dipole delays are 32; these dipoles will not have unity gains");
            }
            dipole_delays.set_to_ideal_delays();
        }

        create_fee_beam_object(
            beam_file.as_deref(),
            total_num_tiles,
            dipole_delays,
            dipole_gains,
        )?
    };
    let beam_file = beam.get_beam_file();
    debug!("Beam file: {beam_file:?}");

    // If the array position wasn't user defined, try the input data.
    let array_pos = array_position.unwrap_or_else(|| {
        trace!("The array position was not specified in the input data; assuming MWA");
        LatLngHeight::mwa()
    });

    let timesteps = match timesteps {
        None => Vec1::try_from(obs_context.all_timesteps.as_slice()),
        Some(mut ts) => {
            // Make sure there are no duplicates.
            let timesteps_hashset: HashSet<&usize> = ts.iter().collect();
            if timesteps_hashset.len() != ts.len() {
                return Err(VisSubtractError::DuplicateTimesteps);
            }

            // Ensure that all specified timesteps are actually available.
            for &t in &ts {
                if obs_context.timestamps.get(t).is_none() {
                    return Err(VisSubtractError::UnavailableTimestep {
                        got: t,
                        last: obs_context.timestamps.len() - 1,
                    });
                }
            }

            ts.sort_unstable();
            Vec1::try_from_vec(ts)
        }
    }
    .map_err(|_| VisSubtractError::NoTimesteps)?;

    let dut1 = if ignore_dut1 { None } else { obs_context.dut1 };

    let precession_info = precess_time(
        array_pos.longitude_rad,
        array_pos.latitude_rad,
        obs_context.phase_centre,
        obs_context.timestamps[*timesteps.first()],
        dut1.unwrap_or_else(|| Duration::from_total_nanoseconds(0)),
    );
    let (lmst, latitude) = if no_precession {
        (precession_info.lmst, array_pos.latitude_rad)
    } else {
        (
            precession_info.lmst_j2000,
            precession_info.array_latitude_j2000,
        )
    };

    messages::ArrayDetails {
        array_position: Some(array_pos),
        array_latitude_j2000: if no_precession {
            None
        } else {
            Some(precession_info.array_latitude_j2000)
        },
        total_num_tiles,
        num_unflagged_tiles,
        flagged_tiles: &tile_baseline_flags
            .flagged_tiles
            .iter()
            .cloned()
            .sorted()
            .map(|i| (obs_context.tile_names[i].as_str(), i))
            .collect::<Vec<_>>(),
    }
    .print();

    let time_res = obs_context.guess_time_res();
    let freq_res = obs_context.guess_freq_res();

    messages::ObservationDetails {
        dipole_delays: beam.get_ideal_dipole_delays(),
        beam_file,
        num_tiles_with_dead_dipoles: if unity_dipole_gains {
            None
        } else {
            obs_context.dipole_gains.as_ref().map(|array| {
                array
                    .outer_iter()
                    .filter(|tile_dipole_gains| {
                        tile_dipole_gains.iter().any(|g| g.abs() < f64::EPSILON)
                    })
                    .count()
            })
        },
        phase_centre: obs_context.phase_centre,
        pointing_centre: None,
        dut1,
        lmst: Some(precession_info.lmst),
        lmst_j2000: if no_precession {
            None
        } else {
            Some(precession_info.lmst_j2000)
        },
        available_timesteps: Some(obs_context.all_timesteps.as_slice()),
        unflagged_timesteps: Some(obs_context.unflagged_timesteps.as_slice()),
        using_timesteps: Some(timesteps.as_slice()),
        first_timestamp: Some(obs_context.timestamps[*timesteps.first()]),
        last_timestamp: Some(obs_context.timestamps[*timesteps.last()]),
        time_res: Some(time_res),
        total_num_channels: obs_context.fine_chan_freqs.len(),
        num_unflagged_channels: None,
        flagged_chans_per_coarse_chan: None,
        first_freq_hz: Some(*obs_context.fine_chan_freqs.first() as f64),
        last_freq_hz: Some(*obs_context.fine_chan_freqs.last() as f64),
        first_unflagged_freq_hz: None,
        last_unflagged_freq_hz: None,
        freq_res_hz: Some(freq_res),
    }
    .print();

    // Handle the invert option.
    let source_list: SourceList = if invert {
        let mut sl: SourceList = source_list
            .into_iter()
            .filter(|(name, _)| !sources_to_subtract.contains(name))
            .collect();
        if sl.is_empty() {
            // Nothing to do.
            return Err(VisSubtractError::AllSourcesFiltered);
        }
        veto_sources(
            &mut sl,
            obs_context.phase_centre,
            lmst,
            latitude,
            &obs_context.coarse_chan_freqs,
            beam.deref(),
            num_sources,
            source_dist_cutoff.unwrap_or(DEFAULT_CUTOFF_DISTANCE),
            veto_threshold.unwrap_or(DEFAULT_VETO_THRESHOLD),
        )?;
        if sl.is_empty() {
            return Err(VisSubtractError::NoSourcesAfterVeto);
        }
        info!("Subtracting {} sources", sl.len());
        sl
    } else {
        let sl = source_list
            .into_iter()
            .filter(|(name, _)| sources_to_subtract.contains(name))
            .collect();
        info!(
            "Subtracting {} specified sources",
            sources_to_subtract.len()
        );
        sl
    };

    messages::SkyModelDetails {
        source_list: &source_list,
    }
    .print();

    messages::print_modeller_info(&modeller_info);

    // Handle output visibility arguments.
    let (time_average_factor, freq_average_factor) = {
        // Parse and verify user input (specified resolutions must
        // evenly divide the input data's resolutions).
        let time_factor = parse_time_average_factor(
            obs_context.time_res,
            time_average.as_deref(),
            1,
        )
        .map_err(|e| match e {
            AverageFactorError::Zero => VisSubtractError::OutputVisTimeAverageFactorZero,
            AverageFactorError::NotInteger => VisSubtractError::OutputVisTimeFactorNotInteger,
            AverageFactorError::NotIntegerMultiple { out, inp } => {
                VisSubtractError::OutputVisTimeResNotMultiple { out, inp }
            }
            AverageFactorError::Parse(e) => VisSubtractError::ParseOutputVisTimeAverageFactor(e),
        })?;
        let freq_factor = parse_freq_average_factor(
            obs_context.freq_res,
            freq_average.as_deref(),
            1,
        )
        .map_err(|e| match e {
            AverageFactorError::Zero => VisSubtractError::OutputVisFreqAverageFactorZero,
            AverageFactorError::NotInteger => VisSubtractError::OutputVisFreqFactorNotInteger,
            AverageFactorError::NotIntegerMultiple { out, inp } => {
                VisSubtractError::OutputVisFreqResNotMultiple { out, inp }
            }
            AverageFactorError::Parse(e) => VisSubtractError::ParseOutputVisFreqAverageFactor(e),
        })?;

        (time_factor, freq_factor)
    };

    let outputs = {
        if outputs.is_empty() {
            vec1![(
                PathBuf::from(DEFAULT_OUTPUT_VIS_FILENAME),
                VisOutputType::Uvfits
            )]
        } else {
            let mut valid_outputs = Vec::with_capacity(outputs.len());
            for file in outputs {
                // Is the output file type supported?
                let ext = file.extension().and_then(|os_str| os_str.to_str());
                match ext.and_then(|s| VisOutputType::from_str(s).ok()) {
                    Some(t) => {
                        can_write_to_file(&file)?;
                        valid_outputs.push((file.to_owned(), t));
                    }
                    None => return Err(VisSubtractError::InvalidOutputFormat(file.clone())),
                }
            }
            Vec1::try_from_vec(valid_outputs).unwrap()
        }
    };

    messages::OutputFileDetails {
        output_solutions: &[],
        vis_type: "subtracted",
        output_vis: Some(&outputs),
        input_vis_time_res: Some(time_res),
        input_vis_freq_res: Some(freq_res),
        output_vis_time_average_factor: time_average_factor,
        output_vis_freq_average_factor: freq_average_factor,
    }
    .print();

    let timeblocks =
        timesteps_to_timeblocks(&obs_context.timestamps, time_average_factor, &timesteps);

    if dry_run {
        info!("Dry run -- exiting now.");
        return Ok(());
    }

    // Channel for modelling and subtracting.
    let (tx_model, rx_model) = bounded(5);
    // Channel for writing subtracted visibilities.
    let (tx_write, rx_write) = bounded(5);

    // Progress bars.
    let multi_progress = MultiProgress::with_draw_target(if no_progress_bars {
        ProgressDrawTarget::hidden()
    } else {
        ProgressDrawTarget::stdout()
    });
    let read_progress = multi_progress.add(
    ProgressBar::new(timesteps.len() as _)
        .with_style(
            ProgressStyle::default_bar()
                .template("{msg:17}: [{wide_bar:.blue}] {pos:2}/{len:2} timesteps ({elapsed_precise}<{eta_precise})").unwrap()
                .progress_chars("=> "),
        )
        .with_position(0)
        .with_message("Reading data"),
);
    let model_progress = multi_progress.add(
    ProgressBar::new(timesteps.len() as _)
        .with_style(
            ProgressStyle::default_bar()
                .template("{msg:17}: [{wide_bar:.blue}] {pos:2}/{len:2} timesteps ({elapsed_precise}<{eta_precise})").unwrap()
                .progress_chars("=> "),
        )
        .with_position(0)
        .with_message("Sky modelling"),
);
    let write_progress = multi_progress.add(
        ProgressBar::new(timeblocks.len() as _)
            .with_style(
                ProgressStyle::default_bar()
                    .template("{msg:17}: [{wide_bar:.blue}] {pos:2}/{len:2} timeblocks ({elapsed_precise}<{eta_precise})").unwrap()
                    .progress_chars("=> "),
            )
            .with_position(0)
            .with_message("Subtracted writing"),
    );

    // Use a variable to track whether any threads have an issue.
    let error = AtomicCell::new(false);

    info!("Reading input data, sky modelling, and writing");
    let scoped_threads_result = thread::scope(|s| {
        // Input visibility-data reading thread.
        let data_handle = s.spawn(|| {
            // If a panic happens, update our atomic error.
            defer_on_unwind! { error.store(true); }
            read_progress.tick();

            let result = read_vis(
                obs_context,
                &tile_baseline_flags,
                input_data.deref(),
                &timesteps,
                vis_shape,
                tx_model,
                &error,
                read_progress,
            );
            // If the result of reading data was an error, allow the other
            // threads to see this so they can abandon their work early.
            if result.is_err() {
                error.store(true);
            }
            result
        });

        // Sky-model generation and subtraction thread.
        let model_handle = s.spawn(|| {
            defer_on_unwind! { error.store(true); }
            model_progress.tick();

            let result = model_vis_and_subtract(
                beam.deref(),
                &source_list,
                obs_context,
                array_pos,
                vis_shape,
                dut1.unwrap_or_else(|| Duration::from_total_nanoseconds(0)),
                !no_precession,
                rx_model,
                tx_write,
                &error,
                model_progress,
                #[cfg(feature = "cuda")]
                use_cpu_for_modelling,
            );
            if result.is_err() {
                error.store(true);
            }
            result
        });

        // Subtracted vis writing thread.
        let write_handle = s.spawn(|| {
            defer_on_unwind! { error.store(true); }
            write_progress.tick();

            let marlu_mwa_obs_context = input_data.get_metafits_context().map(|c| {
                (
                    MwaObsContext::from_mwalib(c),
                    0..obs_context.coarse_chan_freqs.len(),
                )
            });
            let result = write_vis(
                &outputs,
                array_pos,
                obs_context.phase_centre,
                obs_context.pointing_centre,
                &obs_context.tile_xyzs,
                &obs_context.tile_names,
                obs_context.obsid,
                &obs_context.timestamps,
                &timesteps,
                &timeblocks,
                time_res,
                dut1.unwrap_or_else(|| Duration::from_total_nanoseconds(0)),
                freq_res,
                &obs_context.fine_chan_freqs.mapped_ref(|&f| f as f64),
                &tile_baseline_flags
                    .unflagged_cross_baseline_to_tile_map
                    .values()
                    .copied()
                    .sorted()
                    .collect::<Vec<_>>(),
                // TODO: Provide CLI options
                &HashSet::new(),
                time_average_factor,
                freq_average_factor,
                marlu_mwa_obs_context.as_ref().map(|(c, r)| (c, r)),
                rx_write,
                &error,
                Some(write_progress),
            );
            if result.is_err() {
                error.store(true);
            }
            result
        });

        // Join all thread handles. This propagates any errors and lets us know
        // if any threads panicked, if panics aren't aborting as per the
        // Cargo.toml. (It would be nice to capture the panic information, if
        // it's possible, but I don't know how, so panics are currently
        // aborting.)
        let result = data_handle.join().unwrap();
        let result = result.and_then(|_| model_handle.join().unwrap());
        result.and_then(|_| write_handle.join().unwrap().map_err(VisSubtractError::from))
    });

    // Propagate errors and print out the write message.
    let s = scoped_threads_result?;
    info!("{s}");

    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn read_vis(
    obs_context: &ObsContext,
    tile_baseline_flags: &TileBaselineFlags,
    input_data: &dyn VisRead,
    timesteps: &Vec1<usize>,
    vis_shape: (usize, usize),
    tx: Sender<VisTimestep>,
    error: &AtomicCell<bool>,
    progress_bar: ProgressBar,
) -> Result<(), VisSubtractError> {
    let flagged_fine_chans = HashSet::new();

    // Read data to fill the buffer, pausing when the buffer is full to
    // write it all out.
    for &timestep in timesteps {
        let timestamp = obs_context.timestamps[timestep];
        debug!("Reading timestamp {}", timestamp.to_gpst_seconds());

        let mut cross_data: ArcArray2<Jones<f32>> = ArcArray2::zeros(vis_shape);
        let mut cross_weights: ArcArray2<f32> = ArcArray2::zeros(vis_shape);
        input_data.read_crosses(
            cross_data.view_mut(),
            cross_weights.view_mut(),
            timestep,
            tile_baseline_flags,
            &flagged_fine_chans,
        )?;

        // Should we continue?
        if error.load() {
            return Ok(());
        }

        match tx.send(VisTimestep {
            cross_data,
            cross_weights,
            autos: None,
            timestamp,
        }) {
            Ok(()) => (),
            // If we can't send the message, it's because the channel
            // has been closed on the other side. That should only
            // happen because the writer has exited due to error; in
            // that case, just exit this thread.
            Err(_) => return Ok(()),
        }
        progress_bar.inc(1);
    }
    debug!("Finished reading");
    progress_bar.abandon_with_message("Finished reading visibilities");

    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn model_vis_and_subtract(
    beam: &dyn Beam,
    source_list: &SourceList,
    obs_context: &ObsContext,
    array_pos: LatLngHeight,
    vis_shape: (usize, usize),
    dut1: Duration,
    apply_precession: bool,
    rx: Receiver<VisTimestep>,
    tx: Sender<VisTimestep>,
    error: &AtomicCell<bool>,
    progress_bar: ProgressBar,
    #[cfg(feature = "cuda")] use_cpu_for_modelling: bool,
) -> Result<(), VisSubtractError> {
    let flagged_tiles = obs_context
        .get_tile_flags(false, None)
        .expect("can't fail; no additional flags");
    let unflagged_tile_xyzs = obs_context
        .tile_xyzs
        .iter()
        .enumerate()
        .filter(|(i, _)| !flagged_tiles.contains(i))
        .map(|(_, xyz)| *xyz)
        .collect::<Vec<_>>();
    let freqs = obs_context
        .fine_chan_freqs
        .iter()
        .map(|&i| i as f64)
        .collect::<Vec<_>>();
    let mut modeller = crate::model::new_sky_modeller(
        #[cfg(feature = "cuda")]
        use_cpu_for_modelling,
        beam,
        source_list,
        &unflagged_tile_xyzs,
        &freqs,
        &flagged_tiles,
        obs_context.phase_centre,
        array_pos.longitude_rad,
        array_pos.latitude_rad,
        dut1,
        apply_precession,
    )?;

    // Recycle an array for model visibilities.
    let mut vis_model = Array2::zeros(vis_shape);

    // Iterate over the incoming data.
    for VisTimestep {
        cross_data: mut vis_data,
        cross_weights,
        autos,
        timestamp,
    } in rx.iter()
    {
        debug!("Modelling timestamp {}", timestamp.to_gpst_seconds());
        modeller.model_timestep(vis_model.view_mut(), timestamp)?;
        vis_data
            .iter_mut()
            .zip(vis_model.iter())
            .for_each(|(vis_data, vis_model)| {
                *vis_data =
                    Jones::from(Jones::<f64>::from(*vis_data) - Jones::<f64>::from(*vis_model));
            });
        vis_model.fill(Jones::default());

        // Should we continue?
        if error.load() {
            return Ok(());
        }

        match tx.send(VisTimestep {
            cross_data: vis_data,
            cross_weights,
            autos,
            timestamp,
        }) {
            Ok(()) => (),
            Err(_) => return Ok(()),
        }
        progress_bar.inc(1);
    }
    debug!("Finished modelling");
    progress_bar.abandon_with_message("Finished subtracting sky model");
    Ok(())
}
