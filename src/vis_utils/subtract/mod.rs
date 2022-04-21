// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Given input data, a sky model and specific sources, subtract those specific
//! sources from the input data and write them out.

mod error;
pub use error::VisSubtractError;
#[cfg(test)]
mod tests;

use std::collections::{HashMap, HashSet};
use std::ops::Deref;
use std::path::{Path, PathBuf};
use std::str::FromStr;

use clap::Parser;
use crossbeam_channel::{bounded, Receiver, Sender};
use crossbeam_utils::{atomic::AtomicCell, thread};
use hifitime::{Duration, Epoch, Unit};
use indicatif::{MultiProgress, ProgressBar, ProgressDrawTarget, ProgressStyle};
use itertools::{izip, Itertools};
use log::{debug, info, trace};
use marlu::{
    precession::precess_time, Jones, LatLngHeight, MeasurementSetWriter,
    ObsContext as MarluObsContext, UvfitsWriter, VisContext, VisWritable,
};
use ndarray::prelude::*;
use scopeguard::defer_on_unwind;

use crate::{
    context::ObsContext, data_formats::*, filenames::InputDataTypes, glob::*, help_texts::*,
    math::TileBaselineMaps,
};
use mwa_hyperdrive_beam::{create_fee_beam_object, create_no_beam_object, Beam, Delays};
use mwa_hyperdrive_common::{
    clap, hifitime, indicatif, itertools, lazy_static, log, marlu, ndarray,
};
use mwa_hyperdrive_srclist::{
    veto_sources, SourceList, SourceListType, DEFAULT_CUTOFF_DISTANCE, DEFAULT_VETO_THRESHOLD,
    SOURCE_DIST_CUTOFF_HELP as sdc_help, VETO_THRESHOLD_HELP as vt_help,
};

lazy_static::lazy_static! {
    pub static ref SOURCE_DIST_CUTOFF_HELP: String =
    format!("{}. Only useful if subtraction is inverted.", *sdc_help);

    pub static ref VETO_THRESHOLD_HELP: String =
    format!("{}. Only useful if subtraction is inverted.", *vt_help);
}
#[derive(Parser, Debug, Default)]
pub struct VisSubtractArgs {
    /// Paths to the input data files to have visibilities subtracted. These can
    /// include a metafits file, a measurement set and/or uvfits files.
    #[clap(short, long, multiple_values(true), help_heading = "INPUT AND OUTPUT")]
    data: Vec<String>,

    /// Path to the output visibilities file.
    #[clap(
        short = 'o',
        long,
        default_value = "hyp_subtracted.uvfits",
        help_heading = "INPUT AND OUTPUT"
    )]
    output: PathBuf,

    /// Path to the sky-model source list used for simulation.
    #[clap(short, long, help_heading = "INPUT AND OUTPUT")]
    source_list: String,

    #[clap(long, help = SOURCE_LIST_TYPE_HELP.as_str(), help_heading = "INPUT AND OUTPUT")]
    source_list_type: Option<String>,

    /// The names of the sources in the sky-model source list that will be
    /// subtracted from the input data.
    #[clap(long, multiple_values(true), help_heading = "SKY-MODEL SOURCES")]
    sources_to_subtract: Vec<String>,

    /// Invert the subtraction; sources *not* specified in the sky-model source
    /// list will be subtracted from the input data.
    #[clap(short, long, help_heading = "SKY-MODEL SOURCES")]
    invert: bool,

    /// The number of sources to use in the source list. Only useful if
    /// subtraction is inverted. The default is to use all sources unspecified
    /// sources. Example: If 1000 sources are specified here, then the top 1000
    /// sources are used *after* removing specified sources (based on their flux
    /// densities after the beam attenuation) within the specified source
    /// distance cutoff.
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

    /// Specify the MWA dipoles delays, ignoring whatever is in the metafits
    /// file.
    #[clap(long, multiple_values(true), help_heading = "MODEL PARAMETERS")]
    dipole_delays: Option<Vec<u32>>,

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
    pub fn run(&self, dry_run: bool) -> Result<(), VisSubtractError> {
        vis_subtract(self, dry_run)
    }
}

fn vis_subtract(args: &VisSubtractArgs, dry_run: bool) -> Result<(), VisSubtractError> {
    debug!("{:#?}", &args);

    // Expose all the struct fields to ensure they're all used.
    let VisSubtractArgs {
        data,
        output,
        source_list,
        source_list_type,
        sources_to_subtract,
        invert,
        num_sources,
        source_dist_cutoff,
        veto_threshold,
        no_beam,
        beam_file,
        unity_dipole_gains,
        dipole_delays,
        #[cfg(feature = "cuda")]
            cpu: use_cpu_for_modelling,
        no_progress_bars,
    } = args;

    // If we're not inverted but `sources_to_subtract` is empty, then there's
    // nothing to do.
    if !invert && sources_to_subtract.is_empty() {
        return Err(VisSubtractError::NoSources);
    }

    // Read in the source list and remove all but the specified sources.
    let source_list: SourceList = {
        // If the specified source list file can't be found, treat it as a glob
        // and expand it to find a match.
        let pb = PathBuf::from(source_list);
        let pb = if pb.exists() {
            pb
        } else {
            get_single_match_from_glob(source_list)?
        };

        // Read the source list file. If the type was manually specified,
        // use that, otherwise the reading code will try all available
        // kinds.
        let sl_type = source_list_type
            .as_ref()
            .and_then(|t| SourceListType::from_str(t.as_ref()).ok());
        let (sl, _) = match mwa_hyperdrive_srclist::read::read_source_list_file(&pb, sl_type) {
            Ok((sl, sl_type)) => (sl, sl_type),
            Err(e) => return Err(VisSubtractError::from(e)),
        };

        sl
    };
    debug!("Found {} sources in the source list", source_list.len());

    // Ensure that all specified sources are actually in the source list.
    for name in sources_to_subtract {
        if !source_list.contains_key(name) {
            return Err(VisSubtractError::MissingSource {
                name: name.to_owned(),
            });
        }
    }

    // Prepare an input data reader.
    let input_data_types = InputDataTypes::new(data)?;
    let input_data: Box<dyn InputData> = match (
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

            let input_data = MS::new(&ms, meta)?;
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

            let input_data = UvfitsReader::new(&uvfits, meta)?;
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

        _ => return Err(VisSubtractError::InvalidDataInput),
    };

    let obs_context = input_data.get_obs_context();
    let num_tiles = obs_context.tile_xyzs.len();
    let num_unflagged_tiles = num_tiles - obs_context.flagged_tiles.len();
    let num_unflagged_cross_baselines = (num_unflagged_tiles * (num_unflagged_tiles - 1)) / 2;
    let tile_to_unflagged_baseline_map =
        TileBaselineMaps::new(num_tiles, &obs_context.flagged_tiles)
            .tile_to_unflagged_cross_baseline_map;
    let vis_shape = (
        num_unflagged_cross_baselines,
        obs_context.fine_chan_freqs.len(),
    );

    // Set up the beam for modelling.
    let dipole_delays = match dipole_delays {
        // We have user-provided delays; check that they're are sensible,
        // regardless of whether we actually need them.
        Some(d) => {
            if d.len() != 16 || d.iter().any(|&v| v > 32) {
                return Err(VisSubtractError::BadDelays);
            }
            Some(Delays::Partial(d.clone()))
        }

        // No delays were provided; use whatever was in the input data.
        None => obs_context.dipole_delays.clone(),
    };

    let beam: Box<dyn Beam> = if *no_beam {
        create_no_beam_object(obs_context.tile_xyzs.len())
    } else {
        let dipole_delays = dipole_delays.ok_or(VisSubtractError::NoDelays)?;
        create_fee_beam_object(
            beam_file.as_deref(),
            obs_context.tile_xyzs.len(),
            dipole_delays,
            if *unity_dipole_gains {
                None
            } else {
                obs_context.dipole_gains.clone()
            },
        )?
    };

    // Handle the invert option.
    let source_list: SourceList = if *invert {
        let mut sl: SourceList = source_list
            .into_iter()
            .filter(|(name, _)| !sources_to_subtract.contains(name))
            .collect();
        if sl.is_empty() {
            // Nothing to do.
            return Err(VisSubtractError::AllSourcesFiltered);
        }
        let array_position = obs_context
            .array_position
            .unwrap_or_else(LatLngHeight::new_mwa);
        let precession_info = precess_time(
            obs_context.phase_centre,
            *obs_context.timestamps.first(),
            array_position.longitude_rad,
            array_position.latitude_rad,
        );
        veto_sources(
            &mut sl,
            obs_context.phase_centre,
            precession_info.lmst_j2000,
            precession_info.array_latitude_j2000,
            &obs_context.coarse_chan_freqs,
            beam.deref(),
            *num_sources,
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

    if dry_run {
        info!("Dry run -- exiting now.");
        return Ok(());
    }

    // Channel for modelling and subtracting.
    let (tx_model, rx_model) = bounded(5);
    // Channel for writing subtracted visibilities.
    let (tx_write, rx_write) = bounded(5);

    // Progress bars.
    let multi_progress = MultiProgress::with_draw_target(if *no_progress_bars {
        ProgressDrawTarget::hidden()
    } else {
        ProgressDrawTarget::stdout()
    });
    let read_progress = multi_progress.add(
    ProgressBar::new(obs_context.timestamps.len() as _)
        .with_style(
            ProgressStyle::default_bar()
                .template("{msg:17}: [{wide_bar:.blue}] {pos:2}/{len:2} timesteps ({elapsed_precise}<{eta_precise})")
                .progress_chars("=> "),
        )
        .with_position(0)
        .with_message("Reading data"),
);
    let model_progress = multi_progress.add(
    ProgressBar::new(obs_context.timestamps.len() as _)
        .with_style(
            ProgressStyle::default_bar()
                .template("{msg:17}: [{wide_bar:.blue}] {pos:2}/{len:2} timesteps ({elapsed_precise}<{eta_precise})")
                .progress_chars("=> "),
        )
        .with_position(0)
        .with_message("Sky modelling"),
);
    let write_progress = multi_progress.add(
        ProgressBar::new(obs_context.timestamps.len() as _)
            .with_style(
                ProgressStyle::default_bar()
                    .template("{msg:17}: [{wide_bar:.blue}] {pos:2}/{len:2} timesteps ({elapsed_precise}<{eta_precise})")
                    .progress_chars("=> "),
            )
            .with_position(0)
            .with_message("Model writing"),
    );

    info!(
        "Writing the subtracted visibilities to {}",
        output.display()
    );

    // Draw the progress bars. Not doing this means that the bars aren't
    // rendered until they've progressed.
    read_progress.tick();
    model_progress.tick();
    write_progress.tick();

    // Use a variable to track whether any threads have an issue.
    let error = AtomicCell::new(false);

    info!("Reading input data, sky modelling, and writing");
    let scoped_threads_result = thread::scope(|scope| {
        // Spawn a thread to draw the progress bars.
        scope.spawn(move |_| {
            multi_progress.join().unwrap();
        });

        // Input visibility-data reading thread.
        let data_handle = scope.spawn(|_| {
            // If a panic happens, update our atomic error.
            defer_on_unwind! { error.store(true); }

            let result = read_vis(
                obs_context,
                &tile_to_unflagged_baseline_map,
                input_data.deref(),
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
        let model_handle = scope.spawn(|_| {
            defer_on_unwind! { error.store(true); }

            let result = model_vis_and_subtract(
                beam.deref(),
                &source_list,
                obs_context,
                vis_shape,
                rx_model,
                tx_write,
                &error,
                model_progress,
                #[cfg(feature = "cuda")]
                *use_cpu_for_modelling,
            );
            if result.is_err() {
                error.store(true);
            }
            result
        });

        // Subtracted vis writing thread.
        let writer_handle = scope.spawn(|_| {
            defer_on_unwind! { error.store(true); }

            let result = writer(
                output,
                obs_context,
                tile_to_unflagged_baseline_map.keys().cloned(),
                rx_write,
                &error,
                write_progress,
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
        let result = data_handle.join();
        let result = match result {
            Err(_) | Ok(Err(_)) => result,     // Propagate the previous result
            Ok(Ok(())) => model_handle.join(), // Propagate the model result
        };
        let result = match result {
            Err(_) | Ok(Err(_)) => result,
            Ok(Ok(())) => writer_handle.join(),
        };
        result
    });

    match scoped_threads_result {
        // Propagate anything that didn't panic.
        Ok(Ok(r)) => r?,
        // A panic. This ideally only happens because a programmer made a
        // mistake, but it could happen in drastic situations (e.g. hardware
        // failure).
        Err(_) | Ok(Err(_)) => panic!(
            "A panic occurred; the message should be above. You may need to disable progress bars."
        ),
    }

    info!("Subtracted visibilities written to {}", output.display());

    Ok(())
}

fn read_vis(
    obs_context: &ObsContext,
    tile_to_unflagged_baseline_map: &HashMap<(usize, usize), usize>,
    input_data: &dyn InputData,
    vis_shape: (usize, usize),
    tx: Sender<(Array2<Jones<f32>>, Array2<f32>, Epoch)>,
    error: &AtomicCell<bool>,
    progress_bar: ProgressBar,
) -> Result<(), VisSubtractError> {
    let flagged_fine_chans = HashSet::new();

    // Read data to fill the buffer, pausing when the buffer is full to
    // write it all out.
    for &timestep in obs_context.all_timesteps.iter() {
        let timestamp = obs_context.timestamps[timestep];
        debug!("Reading timestamp {}", timestamp.as_gpst_seconds());

        let mut vis_data: Array2<Jones<f32>> = Array2::zeros(vis_shape);
        let mut vis_weights: Array2<f32> = Array2::zeros(vis_shape);
        input_data.read_crosses(
            vis_data.view_mut(),
            vis_weights.view_mut(),
            timestep,
            tile_to_unflagged_baseline_map,
            &flagged_fine_chans,
        )?;

        // Should we continue?
        if error.load() {
            return Ok(());
        }

        match tx.send((vis_data, vis_weights, timestamp)) {
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
    debug!("Finished reading");

    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn model_vis_and_subtract(
    beam: &dyn Beam,
    source_list: &SourceList,
    obs_context: &ObsContext,
    vis_shape: (usize, usize),
    rx: Receiver<(Array2<Jones<f32>>, Array2<f32>, Epoch)>,
    tx: Sender<(Array2<Jones<f32>>, Array2<f32>, Epoch)>,
    error: &AtomicCell<bool>,
    progress_bar: ProgressBar,
    #[cfg(feature = "cuda")] use_cpu_for_modelling: bool,
) -> Result<(), VisSubtractError> {
    let array_pos = obs_context.array_position.unwrap_or_else(|| {
        trace!("The array position was not specified in the input data; assuming MWA");
        marlu::LatLngHeight::new_mwa()
    });
    let unflagged_tile_xyzs = obs_context
        .tile_xyzs
        .iter()
        .enumerate()
        .filter(|(i, _)| !obs_context.flagged_tiles.contains(i))
        .map(|(_, xyz)| *xyz)
        .collect::<Vec<_>>();
    let freqs = obs_context
        .fine_chan_freqs
        .iter()
        .map(|&i| i as f64)
        .collect::<Vec<_>>();
    let modeller = crate::model::new_sky_modeller(
        #[cfg(feature = "cuda")]
        use_cpu_for_modelling,
        beam,
        source_list,
        &unflagged_tile_xyzs,
        &freqs,
        &obs_context.flagged_tiles,
        obs_context.phase_centre,
        array_pos.longitude_rad,
        array_pos.latitude_rad,
        // TODO: Allow the user to turn off precession.
        true,
    )?;

    // Recycle an array for model visibilities.
    let mut vis_model = Array2::zeros(vis_shape);

    // Iterate over the incoming data.
    for (mut vis_data, vis_weights, timestamp) in rx.iter() {
        debug!("Modelling timestamp {}", timestamp.as_gpst_seconds());
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

        match tx.send((vis_data, vis_weights, timestamp)) {
            Ok(()) => (),
            Err(_) => return Ok(()),
        }
        progress_bar.inc(1);
    }
    debug!("Finished modelling");
    progress_bar.abandon_with_message("Finished subtracting sky model");
    Ok(())
}

fn writer<I>(
    output: &Path,
    obs_context: &ObsContext,
    unflagged_baseline_tile_pairs: I,
    rx: Receiver<(Array2<Jones<f32>>, Array2<f32>, Epoch)>,
    error: &AtomicCell<bool>,
    progress_bar: ProgressBar,
) -> Result<(), VisSubtractError>
where
    I: Iterator<Item = (usize, usize)>,
{
    let start_epoch = *obs_context.timestamps.first();
    let ant_pairs = unflagged_baseline_tile_pairs.sorted().collect();
    let int_time: Duration = Duration::from_f64(
        obs_context.time_res.unwrap_or_else(|| {
            trace!("No integration time specified; assuming 1 second");
            1.
        }),
        Unit::Second,
    );
    let freq_res = obs_context.freq_res.unwrap_or_else(|| {
        trace!("No frequency resolution specified; assuming 10 kHz");
        10_000.
    });
    let array_pos = obs_context
        .array_position
        .unwrap_or_else(marlu::LatLngHeight::new_mwa);

    let vis_ctx = VisContext {
        num_sel_timesteps: obs_context.timestamps.len(),
        start_timestamp: start_epoch,
        int_time,
        num_sel_chans: obs_context.fine_chan_freqs.len(),
        start_freq_hz: *obs_context.fine_chan_freqs.first() as f64,
        freq_resolution_hz: freq_res,
        sel_baselines: ant_pairs,
        avg_time: 1,
        avg_freq: 1,
        num_vis_pols: 4,
    };

    let obs_name = obs_context.obsid.map(|o| format!("{}", o));

    let ext = output.extension().and_then(|os_str| os_str.to_str());
    let mut vis_writer: Box<dyn VisWritable> = match ext
        .and_then(|s| VisOutputType::from_str(s).ok())
    {
        Some(VisOutputType::Uvfits) => {
            let uvfits = UvfitsWriter::from_marlu(
                output,
                &vis_ctx,
                Some(array_pos),
                obs_context.phase_centre,
                obs_name,
            )?;
            Box::new(uvfits)
        }
        Some(VisOutputType::MeasurementSet) => {
            let ms = MeasurementSetWriter::new(output, obs_context.phase_centre, Some(array_pos));

            let sched_start_timestamp = match obs_context.obsid {
                Some(gpst) => Epoch::from_gpst_seconds(gpst as f64),
                None => obs_context.timestamps[*obs_context.all_timesteps.first()],
            };
            let sched_duration = obs_context.timestamps[*obs_context.all_timesteps.last()]
                + int_time
                - sched_start_timestamp;
            let marlu_obs_ctx = MarluObsContext {
                sched_start_timestamp,
                sched_duration,
                name: obs_name,
                phase_centre: obs_context.phase_centre,
                pointing_centre: obs_context.pointing_centre,
                array_pos,
                ant_positions_enh: obs_context
                    .tile_xyzs
                    .iter()
                    .map(|xyz| xyz.to_enh(array_pos.latitude_rad))
                    .collect(),
                ant_names: obs_context.tile_names.iter().cloned().collect(),
                // TODO(dev): is there any value in adding this metadata via hyperdrive obs context?
                field_name: None,
                project_id: None,
                observer: None,
            };
            debug!("Creating measurement set {}", output.display());
            ms.initialize(&vis_ctx, &marlu_obs_ctx)?;
            Box::new(ms)
        }
        _ => {
            return Err(VisSubtractError::InvalidOutputFormat(
                ext.unwrap_or("<no extension>").to_string(),
            ))
        }
    };

    // Receive data to write from the modelling thread.
    for (vis_data, vis_weights, timestamp) in rx.iter() {
        debug!("Writing timestamp {}", timestamp.as_gpst_seconds());

        let chunk_vis_ctx = VisContext {
            start_timestamp: timestamp - int_time / 2.0,
            num_sel_timesteps: 1,
            ..vis_ctx.clone()
        };

        let out_shape = chunk_vis_ctx.avg_dims();
        let mut out_data = Array3::zeros(out_shape);
        let mut out_weights = Array3::from_elem(out_shape, -0.0);

        assert_eq!(vis_data.len_of(Axis(0)), out_shape.2);

        // pad and transpose the data, baselines then channels
        for (mut out_data, mut out_weights, in_data, in_weights) in izip!(
            out_data.axis_iter_mut(Axis(1)),
            out_weights.axis_iter_mut(Axis(1)),
            vis_data.axis_iter(Axis(1)),
            vis_weights.axis_iter(Axis(1)),
        ) {
            // merge frequency axis
            for (out_jones, out_weight, in_jones, in_weight) in izip!(
                out_data.iter_mut(),
                out_weights.iter_mut(),
                in_data.iter(),
                in_weights.iter(),
            ) {
                *out_jones = *in_jones;
                *out_weight = *in_weight;
            }
        }

        vis_writer.write_vis_marlu(
            out_data.view(),
            out_weights.view(),
            &chunk_vis_ctx,
            &obs_context.tile_xyzs,
            false,
        )?;

        // Should we continue?
        if error.load() {
            return Ok(());
        }

        progress_bar.inc(1);
    }

    // If we have to, finish the writer.
    if let Some(VisOutputType::Uvfits) = ext.and_then(|s| VisOutputType::from_str(s).ok()) {
        trace!("Finalising writing of model uvfits file");
        let uvfits_writer =
            unsafe { Box::from_raw(Box::into_raw(vis_writer) as *mut UvfitsWriter) };
        uvfits_writer
            .write_uvfits_antenna_table(&obs_context.tile_names, &obs_context.tile_xyzs)?;
    }
    debug!("Finished writing");
    progress_bar.abandon_with_message("Finished subtracted visibilities");
    Ok(())
}
