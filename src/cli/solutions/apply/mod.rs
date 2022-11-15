// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Given input data and a calibration solutions file, apply the solutions and
//! write out the calibrated visibilities.

mod error;
#[cfg(test)]
mod tests;

pub(crate) use error::SolutionsApplyError;

use std::{collections::HashSet, ops::Deref, path::PathBuf, str::FromStr, thread};

use clap::Parser;
use crossbeam_channel::{bounded, Receiver, Sender};
use crossbeam_utils::atomic::AtomicCell;
use hifitime::Duration;
use indicatif::{MultiProgress, ProgressBar, ProgressDrawTarget, ProgressStyle};
use itertools::Itertools;
use log::{debug, info, log_enabled, trace, warn, Level::Debug};
use marlu::{Jones, LatLngHeight, MwaObsContext};
use ndarray::{prelude::*, ArcArray2};
use scopeguard::defer_on_unwind;
use vec1::{vec1, Vec1};

use crate::{
    averaging::{
        parse_freq_average_factor, parse_time_average_factor, timesteps_to_timeblocks,
        AverageFactorError,
    },
    context::ObsContext,
    filenames::InputDataTypes,
    help_texts::ARRAY_POSITION_HELP,
    math::TileBaselineFlags,
    messages,
    pfb_gains::PfbFlavour,
    solutions::CalibrationSolutions,
    vis_io::{
        read::{MsReader, RawDataReader, UvfitsReader, VisInputType, VisRead},
        write::{can_write_to_file, write_vis, VisOutputType, VisTimestep, VIS_OUTPUT_EXTENSIONS},
    },
    HyperdriveError,
};

pub(crate) const DEFAULT_OUTPUT_VIS_FILENAME: &str = "hyperdrive_calibrated.uvfits";

lazy_static::lazy_static! {
    static ref OUTPUTS_HELP: String =
        format!("Paths to the output calibrated visibility files. Supported formats: {}. Default: {}", *VIS_OUTPUT_EXTENSIONS, DEFAULT_OUTPUT_VIS_FILENAME);

    static ref PFB_FLAVOUR_HELP: String =
        format!("{}. Only useful if the input solutions don't specify that this correction should be applied.", *crate::help_texts::PFB_FLAVOUR_HELP);
}

#[derive(Parser, Debug, Default)]
pub struct SolutionsApplyArgs {
    /// Paths to the input data files to apply solutions to. These can include a
    /// metafits file, a measurement set and/or uvfits files.
    #[clap(short, long, multiple_values(true), help_heading = "INPUT FILES")]
    data: Vec<String>,

    /// Path to the calibration solutions file.
    #[clap(short, long, help_heading = "INPUT FILES")]
    solutions: PathBuf,

    /// The timesteps to use from the input data. The default is to use all
    /// unflagged timesteps.
    #[clap(long, multiple_values(true), help_heading = "INPUT FILES")]
    timesteps: Option<Vec<usize>>,

    /// Use all timesteps in the data, including flagged ones. The default is to
    /// use all unflagged timesteps.
    #[clap(long, conflicts_with("timesteps"), help_heading = "INPUT FILES")]
    use_all_timesteps: bool,

    #[clap(
        long, help = ARRAY_POSITION_HELP.as_str(), help_heading = "INPUT FILES",
        number_of_values = 3,
        allow_hyphen_values = true,
        value_names = &["LONG_RAD", "LAT_RAD", "HEIGHT_M"]
    )]
    array_position: Option<Vec<f64>>,

    /// Use a DUT1 value of 0 seconds rather than what is in the input data.
    #[clap(long, help_heading = "INPUT FILES")]
    ignore_dut1: bool,

    /// Additional tiles to be flagged. These values correspond to either the
    /// values in the "Antenna" column of HDU 2 in the metafits file (e.g. 0 3
    /// 127), or the "TileName" (e.g. Tile011).
    #[clap(long, multiple_values(true), help_heading = "FLAGGING")]
    tile_flags: Option<Vec<String>>,

    /// If specified, pretend that all tiles are unflagged in the input data.
    #[clap(long, help_heading = "FLAGGING")]
    ignore_input_data_tile_flags: bool,

    /// If specified, pretend that all tiles are unflagged in the input
    /// solutions.
    #[clap(long, help_heading = "FLAGGING")]
    ignore_input_solutions_tile_flags: bool,

    #[clap(
        short, long, multiple_values(true), help = OUTPUTS_HELP.as_str(),
        help_heading = "OUTPUT FILES"
    )]
    outputs: Option<Vec<PathBuf>>,

    /// When writing out calibrated visibilities, average this many timesteps
    /// together. Also supports a target time resolution (e.g. 8s). The value
    /// must be a multiple of the input data's time resolution. The default is
    /// to preserve the input data's time resolution. e.g. If the input data is
    /// in 0.5s resolution and this variable is 4, then we average 2s worth of
    /// calibrated data together before writing the data out. If the variable is
    /// instead 4s, then 8 calibrated timesteps are averaged together before
    /// writing the data out.
    #[clap(long, help_heading = "OUTPUT FILES")]
    time_average: Option<String>,

    /// When writing out calibrated visibilities, average this many fine freq.
    /// channels together. Also supports a target freq. resolution (e.g. 80kHz).
    /// The value must be a multiple of the input data's freq. resolution. The
    /// default is to preserve the input data's freq. resolution. e.g. If the
    /// input data is in 40kHz resolution and this variable is 4, then we
    /// average 160kHz worth of calibrated data together before writing the data
    /// out. If the variable is instead 80kHz, then 2 calibrated fine freq.
    /// channels are averaged together before writing the data out.
    #[clap(long, help_heading = "OUTPUT FILES")]
    freq_average: Option<String>,

    /// Don't include autocorrelations in the output visibilities.
    #[clap(long, help_heading = "OUTPUT FILES")]
    no_autos: bool,

    #[clap(long, help = PFB_FLAVOUR_HELP.as_str(), help_heading = "RAW MWA DATA CORRECTIONS")]
    pfb_flavour: Option<String>,

    /// When reading in raw MWA data, don't apply digital gains. Only useful if
    /// the input solutions don't specify that this correction should be
    /// applied.
    #[clap(long, help_heading = "RAW MWA DATA CORRECTIONS")]
    no_digital_gains: bool,

    /// When reading in raw MWA data, don't apply cable length corrections. Only
    /// useful if the input solutions don't specify that this correction should
    /// be applied.
    #[clap(long, help_heading = "RAW MWA DATA CORRECTIONS")]
    no_cable_length_correction: bool,

    /// When reading in raw MWA data, don't apply geometric corrections. Only
    /// useful if the input solutions don't specify that this correction should
    /// be applied.
    #[clap(long, help_heading = "RAW MWA DATA CORRECTIONS")]
    no_geometric_correction: bool,

    /// Don't draw progress bars.
    #[clap(long, help_heading = "USER INTERFACE")]
    no_progress_bars: bool,
}

impl SolutionsApplyArgs {
    pub fn run(self, dry_run: bool) -> Result<(), HyperdriveError> {
        apply_solutions(self, dry_run)?;
        Ok(())
    }
}

fn apply_solutions(args: SolutionsApplyArgs, dry_run: bool) -> Result<(), SolutionsApplyError> {
    debug!("{:#?}", args);

    // Expose all the struct fields to ensure they're all used.
    let SolutionsApplyArgs {
        data,
        solutions,
        timesteps,
        use_all_timesteps,
        array_position,
        ignore_dut1,
        tile_flags,
        ignore_input_data_tile_flags,
        ignore_input_solutions_tile_flags,
        outputs,
        time_average,
        freq_average,
        no_autos,
        pfb_flavour,
        no_digital_gains,
        no_cable_length_correction,
        no_geometric_correction,
        no_progress_bars,
    } = args;

    // Get the input data types.
    let input_data_types = InputDataTypes::new(&data)?;

    // We don't necessarily need a metafits file, but if there's multiple of
    // them, we complain.
    let metafits = {
        if let Some(ms) = input_data_types.metafits.as_ref() {
            if ms.len() > 1 {
                return Err(SolutionsApplyError::MultipleMetafits(ms.clone()));
            }
            Some(ms.first().as_ref())
        } else {
            None
        }
    };

    // Read the solutions before the input data; if something is wrong with
    // them, then we can bail much faster.
    let sols = CalibrationSolutions::read_solutions_from_ext_inner(&solutions, metafits)?;

    messages::CalSolDetails {
        filename: &solutions,
        sols: &sols,
    }
    .print();

    // Use corrections specified by the solutions if they exist. Otherwise, we
    // start with defaults.
    let mut raw_data_corrections = sols.raw_data_corrections.unwrap_or_default();
    if let Some(s) = pfb_flavour {
        let pfb_flavour = PfbFlavour::parse(&s)?;
        raw_data_corrections.pfb_flavour = pfb_flavour;
    };
    if no_digital_gains {
        raw_data_corrections.digital_gains = false;
    }
    if no_cable_length_correction {
        raw_data_corrections.cable_length = false;
    }
    if no_geometric_correction {
        raw_data_corrections.geometric = false;
    }
    debug!("Raw data corrections with user input: {raw_data_corrections:?}");

    // If the user supplied the array position, unpack it here.
    let array_position = match array_position {
        Some(pos) => {
            if pos.len() != 3 {
                return Err(SolutionsApplyError::BadArrayPosition { pos });
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
    let input_data: Box<dyn VisRead> = match (
        input_data_types.metafits,
        input_data_types.gpuboxes,
        input_data_types.mwafs,
        input_data_types.ms,
        input_data_types.uvfits,
    ) {
        // Valid input for reading raw data.
        (Some(meta), Some(gpuboxes), mwafs, None, None) => {
            // Ensure that there's only one metafits.
            let meta = if meta.len() > 1 {
                return Err(SolutionsApplyError::MultipleMetafits(meta));
            } else {
                meta.first()
            };

            debug!("gpubox files: {:?}", &gpuboxes);
            debug!("mwaf files: {:?}", &mwafs);

            let input_data = Box::new(RawDataReader::new(
                meta,
                &gpuboxes,
                mwafs.as_deref(),
                raw_data_corrections,
            )?);

            messages::InputFileDetails::Raw {
                obsid: input_data.get_obs_context().obsid.unwrap(),
                gpubox_count: gpuboxes.len(),
                metafits_file_name: meta.display().to_string(),
                mwaf: input_data.get_flags(),
                raw_data_corrections,
            }
            .print("Applying solutions to"); // Print some high-level information.

            input_data
        }

        // Valid input for reading a measurement set.
        (meta, None, None, Some(ms), None) => {
            // Only one MS is supported at the moment.
            let ms: PathBuf = if ms.len() > 1 {
                return Err(SolutionsApplyError::MultipleMeasurementSets(ms));
            } else {
                ms.first().clone()
            };

            // Ensure that there's only one metafits.
            let meta: Option<&PathBuf> = match meta.as_ref() {
                None => None,
                Some(m) => {
                    if m.len() > 1 {
                        return Err(SolutionsApplyError::MultipleMetafits(m.clone()));
                    } else {
                        Some(m.first())
                    }
                }
            };

            let input_data = MsReader::new(&ms, meta, array_position)?;

            messages::InputFileDetails::MeasurementSet {
                obsid: input_data.get_obs_context().obsid,
                file_name: ms.display().to_string(),
                metafits_file_name: meta.map(|m| m.display().to_string()),
            }
            .print("Applying solutions to");

            Box::new(input_data)
        }

        // Valid input for reading uvfits files.
        (meta, None, None, None, Some(uvfits)) => {
            // Only one uvfits is supported at the moment.
            let uvfits: PathBuf = if uvfits.len() > 1 {
                return Err(SolutionsApplyError::MultipleUvfits(uvfits));
            } else {
                uvfits.first().clone()
            };

            // Ensure that there's only one metafits.
            let meta: Option<&PathBuf> = match meta.as_ref() {
                None => None,
                Some(m) => {
                    if m.len() > 1 {
                        return Err(SolutionsApplyError::MultipleMetafits(m.clone()));
                    } else {
                        Some(m.first())
                    }
                }
            };

            let input_data = UvfitsReader::new(&uvfits, meta)?;

            messages::InputFileDetails::UvfitsFile {
                obsid: input_data.get_obs_context().obsid,
                file_name: uvfits.display().to_string(),
                metafits_file_name: meta.map(|m| m.display().to_string()),
            }
            .print("Applying solutions to");

            Box::new(input_data)
        }

        // The following matches are for invalid combinations of input
        // files. Make an error message for the user.
        (Some(_), _, None, None, None) => {
            let msg = "Received only a metafits file; a uvfits file, a measurement set or gpubox files are required.";
            return Err(SolutionsApplyError::InvalidDataInput(msg));
        }
        (Some(_), _, Some(_), None, None) => {
            let msg = "Received only a metafits file and mwaf files; gpubox files are required.";
            return Err(SolutionsApplyError::InvalidDataInput(msg));
        }
        (None, Some(_), _, None, None) => {
            let msg = "Received gpuboxes without a metafits file; this is not supported.";
            return Err(SolutionsApplyError::InvalidDataInput(msg));
        }
        (None, None, Some(_), None, None) => {
            let msg =
                "Received mwaf files without gpuboxes and a metafits file; this is not supported.";
            return Err(SolutionsApplyError::InvalidDataInput(msg));
        }
        (_, Some(_), _, Some(_), None) => {
            let msg = "Received gpuboxes and measurement set files; this is not supported.";
            return Err(SolutionsApplyError::InvalidDataInput(msg));
        }
        (_, Some(_), _, None, Some(_)) => {
            let msg = "Received gpuboxes and uvfits files; this is not supported.";
            return Err(SolutionsApplyError::InvalidDataInput(msg));
        }
        (_, _, _, Some(_), Some(_)) => {
            let msg = "Received uvfits and measurement set files; this is not supported.";
            return Err(SolutionsApplyError::InvalidDataInput(msg));
        }
        (_, _, Some(_), Some(_), _) => {
            let msg = "Received mwafs and measurement set files; this is not supported.";
            return Err(SolutionsApplyError::InvalidDataInput(msg));
        }
        (_, _, Some(_), _, Some(_)) => {
            let msg = "Received mwafs and uvfits files; this is not supported.";
            return Err(SolutionsApplyError::InvalidDataInput(msg));
        }
        (None, None, None, None, None) => return Err(SolutionsApplyError::NoInputData),
    };

    // Warn the user if we're applying solutions to raw data without corrections.
    if matches!(input_data.get_input_data_type(), VisInputType::Raw)
        && sols.raw_data_corrections.is_none()
    {
        warn!("The calibration solutions do not list raw data corrections.");
        warn!("Defaults and any user inputs are being used.");
    };

    let obs_context = input_data.get_obs_context();
    let no_autos = if !obs_context.autocorrelations_present {
        info!("No auto-correlations in the input data; none will be written out");
        true
    } else if no_autos {
        info!("Ignoring auto-correlations in the input data; none will be written out");
        true
    } else {
        info!("Will write out calibrated cross- and auto-correlations");
        false
    };
    let total_num_tiles = obs_context.get_total_num_tiles();

    // We can't do anything if the number of tiles in the data is different to
    // that of the solutions.
    if total_num_tiles != sols.di_jones.len_of(Axis(1)) {
        return Err(SolutionsApplyError::TileCountMismatch {
            data: total_num_tiles,
            solutions: sols.di_jones.len_of(Axis(1)),
        });
    }

    // Assign the tile flags. The flagged tiles in the solutions are always
    // used.
    let tile_flags = {
        // Need to convert indices into strings to use the `get_tile_flags`
        // method below.
        let mut sol_flags: Vec<String> = if ignore_input_solutions_tile_flags {
            debug!("Including any tiles with only NaN for solutions");
            vec![]
        } else {
            debug!(
                "There are {} tiles with only NaN for solutions; considering them as flagged tiles",
                sols.flagged_tiles.len()
            );
            sols.flagged_tiles.iter().map(|i| format!("{i}")).collect()
        };
        if let Some(user_tile_flags) = tile_flags {
            debug!("Using additional user tile flags: {user_tile_flags:?}");
            sol_flags.extend(user_tile_flags.into_iter());
        }
        if ignore_input_data_tile_flags {
            debug!("Not using flagged tiles in the input data");
        } else {
            debug!(
                "Using input data tile flags: {:?}",
                obs_context.flagged_tiles
            );
        }
        obs_context.get_tile_flags(ignore_input_data_tile_flags, Some(&sol_flags))?
    };
    let num_unflagged_tiles = total_num_tiles - tile_flags.len();
    if log_enabled!(Debug) {
        debug!("Tile indices, names and statuses:");
        obs_context
            .tile_names
            .iter()
            .enumerate()
            .map(|(i, name)| {
                let flagged = tile_flags.contains(&i);
                (i, name, if flagged { "  flagged" } else { "unflagged" })
            })
            .for_each(|(i, name, status)| {
                debug!("    {:3}: {:10}: {}", i, name, status);
            })
    }
    if num_unflagged_tiles == 0 {
        return Err(SolutionsApplyError::NoTiles);
    }
    // If the array position wasn't user defined, try the input data.
    let array_position = array_position.unwrap_or_else(|| {
        trace!("The array position was not specified in the input data; assuming MWA");
        LatLngHeight::mwa()
    });
    messages::ArrayDetails {
        array_position: Some(array_position),
        array_latitude_j2000: None,
        total_num_tiles,
        num_unflagged_tiles,
        flagged_tiles: &tile_flags
            .iter()
            .cloned()
            .sorted()
            .map(|i| (obs_context.tile_names[i].as_str(), i))
            .collect::<Vec<_>>(),
    }
    .print();

    let tile_baseline_flags = TileBaselineFlags::new(total_num_tiles, tile_flags);
    let timesteps = match (use_all_timesteps, timesteps) {
        (true, _) => Vec1::try_from(obs_context.all_timesteps.as_slice()),
        (false, None) => Vec1::try_from(obs_context.unflagged_timesteps.as_slice()),
        (false, Some(mut ts)) => {
            // Make sure there are no duplicates.
            let timesteps_hashset: HashSet<&usize> = ts.iter().collect();
            if timesteps_hashset.len() != ts.len() {
                return Err(SolutionsApplyError::DuplicateTimesteps);
            }

            // Ensure that all specified timesteps are actually available.
            for &t in &ts {
                if obs_context.timestamps.get(t).is_none() {
                    return Err(SolutionsApplyError::UnavailableTimestep {
                        got: t,
                        last: obs_context.timestamps.len() - 1,
                    });
                }
            }

            ts.sort_unstable();
            Vec1::try_from_vec(ts)
        }
    }
    .map_err(|_| SolutionsApplyError::NoTimesteps)?;

    let dut1 = if ignore_dut1 { None } else { obs_context.dut1 };

    messages::ObservationDetails {
        dipole_delays: None,
        beam_file: None,
        num_tiles_with_dead_dipoles: None,
        phase_centre: obs_context.phase_centre,
        pointing_centre: obs_context.pointing_centre,
        dut1,
        lmst: None,
        lmst_j2000: None,
        available_timesteps: Some(&obs_context.all_timesteps),
        unflagged_timesteps: Some(&obs_context.unflagged_timesteps),
        using_timesteps: Some(&timesteps),
        first_timestamp: Some(obs_context.timestamps[*timesteps.first()]),
        last_timestamp: if timesteps.len() > 1 {
            Some(obs_context.timestamps[*timesteps.last()])
        } else {
            None
        },
        time_res: obs_context.time_res,
        total_num_channels: obs_context.fine_chan_freqs.len(),
        num_unflagged_channels: None,
        flagged_chans_per_coarse_chan: None,
        first_freq_hz: Some(*obs_context.fine_chan_freqs.first() as f64),
        last_freq_hz: Some(*obs_context.fine_chan_freqs.last() as f64),
        first_unflagged_freq_hz: None,
        last_unflagged_freq_hz: None,
        freq_res_hz: obs_context.freq_res,
    }
    .print();

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
            AverageFactorError::Zero => SolutionsApplyError::OutputVisTimeAverageFactorZero,
            AverageFactorError::NotInteger => SolutionsApplyError::OutputVisTimeFactorNotInteger,
            AverageFactorError::NotIntegerMultiple { out, inp } => {
                SolutionsApplyError::OutputVisTimeResNotMultiple { out, inp }
            }
            AverageFactorError::Parse(e) => SolutionsApplyError::ParseOutputVisTimeAverageFactor(e),
        })?;
        let freq_factor = parse_freq_average_factor(
            obs_context.freq_res,
            freq_average.as_deref(),
            1,
        )
        .map_err(|e| match e {
            AverageFactorError::Zero => SolutionsApplyError::OutputVisFreqAverageFactorZero,
            AverageFactorError::NotInteger => SolutionsApplyError::OutputVisFreqFactorNotInteger,
            AverageFactorError::NotIntegerMultiple { out, inp } => {
                SolutionsApplyError::OutputVisFreqResNotMultiple { out, inp }
            }
            AverageFactorError::Parse(e) => SolutionsApplyError::ParseOutputVisFreqAverageFactor(e),
        })?;

        (time_factor, freq_factor)
    };

    let outputs = match outputs {
        None => vec1![(
            PathBuf::from("hyp_calibrated.uvfits"),
            VisOutputType::Uvfits
        )],
        Some(v) => {
            let mut outputs = Vec::with_capacity(v.len());
            for file in v {
                // Is the output file type supported?
                let ext = file.extension().and_then(|os_str| os_str.to_str());
                match ext.and_then(|s| VisOutputType::from_str(s).ok()) {
                    Some(vis_type) => {
                        trace!("{} is a visibility output", file.display());
                        can_write_to_file(&file)?;
                        outputs.push((file, vis_type));
                    }
                    None => return Err(SolutionsApplyError::InvalidOutputFormat(file)),
                }
            }
            Vec1::try_from_vec(outputs).map_err(|_| SolutionsApplyError::NoOutput)?
        }
    };

    messages::OutputFileDetails {
        output_solutions: &[],
        vis_type: "calibrated",
        output_vis: Some(&outputs),
        input_vis_time_res: obs_context.time_res,
        input_vis_freq_res: obs_context.freq_res,
        output_vis_time_average_factor: time_average_factor,
        output_vis_freq_average_factor: freq_average_factor,
    }
    .print();

    if dry_run {
        info!("Dry run -- exiting now.");
        return Ok(());
    }

    apply_solutions_inner(
        input_data.deref(),
        &sols,
        &timesteps,
        array_position,
        dut1.unwrap_or_else(|| Duration::from_seconds(0.0)),
        no_autos,
        &tile_baseline_flags,
        // TODO: Provide CLI options
        &HashSet::new(),
        false,
        &outputs,
        time_average_factor,
        freq_average_factor,
        no_progress_bars,
    )
}

#[allow(clippy::too_many_arguments)]
pub(super) fn apply_solutions_inner(
    input_data: &dyn VisRead,
    sols: &CalibrationSolutions,
    timesteps: &Vec1<usize>,
    array_position: LatLngHeight,
    dut1: Duration,
    no_autos: bool,
    tile_baseline_flags: &TileBaselineFlags,
    flagged_fine_chans: &HashSet<usize>,
    ignore_input_data_fine_channel_flags: bool,
    outputs: &Vec1<(PathBuf, VisOutputType)>,
    output_vis_time_average_factor: usize,
    output_vis_freq_average_factor: usize,
    no_progress_bars: bool,
) -> Result<(), SolutionsApplyError> {
    let obs_context = input_data.get_obs_context();
    let fine_chan_flags = {
        let mut flagged_fine_chans = flagged_fine_chans.clone();
        if !ignore_input_data_fine_channel_flags {
            flagged_fine_chans.extend(obs_context.flagged_fine_chans.iter().copied());
        }
        flagged_fine_chans
    };

    let timeblocks = timesteps_to_timeblocks(
        &obs_context.timestamps,
        output_vis_time_average_factor,
        timesteps,
    );

    // Channel for applying solutions.
    let (tx_data, rx_data) = bounded(3);
    // Channel for writing calibrated visibilities.
    let (tx_write, rx_write) = bounded(3);

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
                .template("{msg:18}: [{wide_bar:.blue}] {pos:2}/{len:2} timesteps ({elapsed_precise}<{eta_precise})").unwrap()
                .progress_chars("=> "),
        )
        .with_position(0)
        .with_message("Reading data"),
    );
    let apply_progress = multi_progress.add(
    ProgressBar::new(timesteps.len() as _)
        .with_style(
            ProgressStyle::default_bar()
                .template("{msg:18}: [{wide_bar:.blue}] {pos:2}/{len:2} timesteps ({elapsed_precise}<{eta_precise})").unwrap()
                .progress_chars("=> "),
        )
        .with_position(0)
        .with_message("Applying solutions"),
    );
    let write_progress = multi_progress.add(
        ProgressBar::new(timeblocks.len() as _)
            .with_style(
                ProgressStyle::default_bar()
                    .template("{msg:18}: [{wide_bar:.blue}] {pos:2}/{len:2} timeblocks ({elapsed_precise}<{eta_precise})").unwrap()
                    .progress_chars("=> "),
            )
            .with_position(0)
            .with_message("Writing data"),
    );

    // Use a variable to track whether any threads have an issue.
    let error = AtomicCell::new(false);

    info!("Reading input data, applying, and writing");
    let scoped_threads_result = thread::scope(|s| {
        // Input visibility-data reading thread.
        let data_handle = s.spawn(|| {
            // If a panic happens, update our atomic error.
            defer_on_unwind! { error.store(true); }
            read_progress.tick();

            let result = read_vis(
                obs_context,
                tile_baseline_flags,
                input_data.deref(),
                timesteps,
                no_autos,
                tx_data,
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

        // Solutions applying thread.
        let apply_handle = s.spawn(|| {
            defer_on_unwind! { error.store(true); }
            apply_progress.tick();

            let result = apply_solutions_thread(
                obs_context,
                sols,
                tile_baseline_flags,
                &fine_chan_flags,
                rx_data,
                tx_write,
                &error,
                apply_progress,
            );
            if result.is_err() {
                error.store(true);
            }
            result
        });

        // Calibrated vis writing thread.
        let write_handle = s.spawn(|| {
            defer_on_unwind! { error.store(true); }
            write_progress.tick();

            // If we're not using autos, "disable" the `unflagged_tiles_iter` by
            // making it not iterate over anything.
            let total_num_tiles = if no_autos {
                0
            } else {
                obs_context.get_total_num_tiles()
            };
            let unflagged_tiles_iter = (0..total_num_tiles)
                .filter(|i_tile| !tile_baseline_flags.flagged_tiles.contains(i_tile))
                .map(|i_tile| (i_tile, i_tile));
            // Form (sorted) unflagged baselines from our cross- and
            // auto-correlation baselines.
            let unflagged_cross_and_auto_baseline_tile_pairs = tile_baseline_flags
                .tile_to_unflagged_cross_baseline_map
                .keys()
                .copied()
                .chain(unflagged_tiles_iter)
                .sorted()
                .collect::<Vec<_>>();
            let fine_chan_freqs = obs_context.fine_chan_freqs.mapped_ref(|&f| f as f64);
            let marlu_mwa_obs_context = input_data.get_metafits_context().map(|c| {
                (
                    MwaObsContext::from_mwalib(c),
                    0..obs_context.coarse_chan_freqs.len(),
                )
            });
            let result = write_vis(
                outputs,
                array_position,
                obs_context.phase_centre,
                obs_context.pointing_centre,
                &obs_context.tile_xyzs,
                &obs_context.tile_names,
                obs_context.obsid,
                &obs_context.timestamps,
                timesteps,
                &timeblocks,
                obs_context.guess_time_res(),
                dut1,
                obs_context.guess_freq_res(),
                &fine_chan_freqs,
                &unflagged_cross_and_auto_baseline_tile_pairs,
                &HashSet::new(),
                output_vis_time_average_factor,
                output_vis_freq_average_factor,
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
        let result = result.and_then(|_| apply_handle.join().unwrap());
        result.and_then(|_| {
            write_handle
                .join()
                .unwrap()
                .map_err(SolutionsApplyError::from)
        })
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
    no_autos: bool,
    tx: Sender<VisTimestep>,
    error: &AtomicCell<bool>,
    progress_bar: ProgressBar,
) -> Result<(), SolutionsApplyError> {
    let num_unflagged_tiles = tile_baseline_flags.unflagged_auto_index_to_tile_map.len();
    let num_unflagged_cross_baselines = (num_unflagged_tiles * (num_unflagged_tiles - 1)) / 2;

    let cross_vis_shape = (
        obs_context.fine_chan_freqs.len(),
        num_unflagged_cross_baselines,
    );
    let auto_vis_shape = (obs_context.fine_chan_freqs.len(), num_unflagged_tiles);

    // Send the data as timesteps.
    for &timestep in timesteps {
        let timestamp = obs_context.timestamps[timestep];
        debug!(
            "Reading timestep {timestep} (GPS {})",
            timestamp.to_gpst_seconds()
        );

        let mut cross_data: ArcArray2<Jones<f32>> = ArcArray2::zeros(cross_vis_shape);
        let mut cross_weights: ArcArray2<f32> = ArcArray2::zeros(cross_vis_shape);
        let mut autos = if no_autos {
            None
        } else {
            Some((
                ArcArray::zeros(auto_vis_shape),
                ArcArray::zeros(auto_vis_shape),
            ))
        };

        if let Some((auto_data, auto_weights)) = autos.as_mut() {
            input_data.read_crosses_and_autos(
                cross_data.view_mut(),
                cross_weights.view_mut(),
                auto_data.view_mut(),
                auto_weights.view_mut(),
                timestep,
                tile_baseline_flags,
                // We want to read in all channels, even if they're flagged.
                // Channels will get flagged later based on the calibration
                // solutions, the input data flags and user flags.
                &HashSet::new(),
            )?;
        } else {
            input_data.read_crosses(
                cross_data.view_mut(),
                cross_weights.view_mut(),
                timestep,
                tile_baseline_flags,
                &HashSet::new(),
            )?;
        }

        // Should we continue?
        if error.load() {
            return Ok(());
        }

        match tx.send(VisTimestep {
            cross_data,
            cross_weights,
            autos,
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
fn apply_solutions_thread(
    obs_context: &ObsContext,
    solutions: &CalibrationSolutions,
    tile_baseline_flags: &TileBaselineFlags,
    fine_chan_flags: &HashSet<usize>,
    rx: Receiver<VisTimestep>,
    tx: Sender<VisTimestep>,
    error: &AtomicCell<bool>,
    progress_bar: ProgressBar,
) -> Result<(), SolutionsApplyError> {
    for VisTimestep {
        mut cross_data,
        mut cross_weights,
        mut autos,
        timestamp,
    } in rx.iter()
    {
        // Should we continue?
        if error.load() {
            return Ok(());
        }

        let span = *obs_context.timestamps.last() - *obs_context.timestamps.first();
        let timestamp_fraction = ((timestamp - *obs_context.timestamps.first()).to_seconds()
            / span.to_seconds())
        // Stop stupid values.
        .clamp(0.0, 0.99);

        // Find solutions corresponding to this timestamp.
        let sols = solutions.get_timeblock(timestamp, timestamp_fraction);

        for (i_baseline, (mut vis_data, mut vis_weights)) in cross_data
            .axis_iter_mut(Axis(1))
            .zip_eq(cross_weights.axis_iter_mut(Axis(1)))
            .enumerate()
        {
            let (tile1, tile2) = tile_baseline_flags.unflagged_cross_baseline_to_tile_map
                .get(&i_baseline)
                .copied()
                .unwrap_or_else(|| {
                    panic!("Couldn't find baseline index {i_baseline} in unflagged_cross_baseline_to_tile_map")
                });
            // TODO: Allow solutions to have a different number of channels than
            // the data.

            // Get the solutions for both tiles and apply them.
            let sols_tile1 = sols.slice(s![tile1, ..]);
            let sols_tile2 = sols.slice(s![tile2, ..]);
            vis_data
                .iter_mut()
                .zip_eq(vis_weights.iter_mut())
                .zip_eq(sols_tile1.iter())
                .zip_eq(sols_tile2.iter())
                .enumerate()
                .for_each(|(i_chan, (((vis_data, vis_weight), sol1), sol2))| {
                    // One of the tiles doesn't have a solution; flag.
                    if sol1.any_nan() || sol2.any_nan() {
                        *vis_weight = -vis_weight.abs();
                        *vis_data = Jones::default();
                    } else {
                        if fine_chan_flags.contains(&i_chan) {
                            // The channel is flagged, but we still have a solution for it.
                            *vis_weight = -vis_weight.abs();
                        }
                        // Promote the data before demoting it again.
                        let d: Jones<f64> = Jones::from(*vis_data);
                        *vis_data = Jones::from((*sol1 * d) * sol2.h());
                    }
                });
        }

        if let Some((auto_data, auto_weights)) = autos.as_mut() {
            for (i_tile, (mut vis_data, mut vis_weights)) in auto_data
                .axis_iter_mut(Axis(1))
                .zip_eq(auto_weights.axis_iter_mut(Axis(1)))
                .enumerate()
            {
                let i_tile = tile_baseline_flags
                    .unflagged_auto_index_to_tile_map
                    .get(&i_tile)
                    .copied()
                    .unwrap_or_else(|| {
                        panic!(
                            "Couldn't find auto index {i_tile} in unflagged_auto_index_to_tile_map"
                        )
                    });

                // Get the solutions for the tile and apply it twice.
                let sols = sols.slice(s![i_tile, ..]);
                vis_data
                    .iter_mut()
                    .zip_eq(vis_weights.iter_mut())
                    .zip_eq(sols.iter())
                    .enumerate()
                    .for_each(|(i_chan, ((vis_data, vis_weight), sol))| {
                        // No solution; flag.
                        if sol.any_nan() {
                            *vis_weight = -vis_weight.abs();
                            *vis_data = Jones::default();
                        } else {
                            if fine_chan_flags.contains(&i_chan) {
                                // The channel is flagged, but we still have a solution for it.
                                *vis_weight = -vis_weight.abs();
                            }
                            // Promote the data before demoting it again.
                            let d: Jones<f64> = Jones::from(*vis_data);
                            *vis_data = Jones::from((*sol * d) * sol.h());
                        }
                    });
            }
        }

        // Send the calibrated visibilities to the writer.
        match tx.send(VisTimestep {
            cross_data,
            cross_weights,
            autos,
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
    debug!("Finished applying");
    progress_bar.abandon_with_message("Finished applying solutions");
    Ok(())
}
