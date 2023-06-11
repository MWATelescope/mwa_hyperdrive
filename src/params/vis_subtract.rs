// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Given input data, a sky model and specific sources, subtract those specific
//! sources from the input data and write them out.

use std::thread::{self, ScopedJoinHandle};

use crossbeam_channel::{bounded, Receiver, Sender};
use crossbeam_utils::atomic::AtomicCell;
use indicatif::{MultiProgress, ProgressBar, ProgressDrawTarget, ProgressStyle};
use itertools::Itertools;
use log::{debug, info};
use marlu::Jones;
use ndarray::prelude::*;
use scopeguard::defer_on_unwind;

use super::{InputVisParams, ModellingParams, OutputVisParams};
use crate::{
    beam::Beam,
    io::{
        read::VisReadError,
        write::{write_vis, VisTimestep},
    },
    model::{new_sky_modeller, ModelError},
    srclist::SourceList,
    PROGRESS_BARS,
};

pub(crate) struct VisSubtractParams {
    pub(crate) input_vis_params: InputVisParams,
    pub(crate) output_vis_params: OutputVisParams,
    pub(crate) beam: Box<dyn Beam>,
    pub(crate) source_list: SourceList,
    pub(crate) modelling_params: ModellingParams,
}

impl VisSubtractParams {
    pub(crate) fn run(&self) -> Result<(), VisSubtractError> {
        // Expose all the struct fields to ensure they're all used.
        let VisSubtractParams {
            input_vis_params,
            output_vis_params,
            beam,
            source_list,
            modelling_params: ModellingParams { apply_precession },
        } = self;

        let obs_context = input_vis_params.get_obs_context();
        let num_unflagged_tiles = input_vis_params.get_num_unflagged_tiles();
        let num_unflagged_cross_baselines = (num_unflagged_tiles * (num_unflagged_tiles - 1)) / 2;
        let vis_shape = (
            input_vis_params.spw.chanblocks.len(),
            num_unflagged_cross_baselines,
        );

        // Channel for modelling and subtracting.
        let (tx_model, rx_model) = bounded(5);
        // Channel for writing subtracted visibilities.
        let (tx_write, rx_write) = bounded(5);

        // Progress bars.
        let multi_progress = MultiProgress::with_draw_target(if PROGRESS_BARS.load() {
            ProgressDrawTarget::stdout()
        } else {
            ProgressDrawTarget::hidden()
        });
        let pb = ProgressBar::new(input_vis_params.timeblocks.len() as _)
        .with_style(
            ProgressStyle::default_bar()
                .template("{msg:17}: [{wide_bar:.blue}] {pos:2}/{len:2} timesteps ({elapsed_precise}<{eta_precise})").unwrap()
                .progress_chars("=> "),
        )
        .with_position(0)
        .with_message("Reading data");
        let read_progress = multi_progress.add(pb);
        let pb = ProgressBar::new(input_vis_params.timeblocks.len() as _)
        .with_style(
            ProgressStyle::default_bar()
                .template("{msg:17}: [{wide_bar:.blue}] {pos:2}/{len:2} timesteps ({elapsed_precise}<{eta_precise})").unwrap()
                .progress_chars("=> "),
        )
        .with_position(0)
        .with_message("Sky modelling");
        let model_progress = multi_progress.add(pb);
        let pb = ProgressBar::new(output_vis_params.output_timeblocks.len() as _)
        .with_style(
            ProgressStyle::default_bar()
                .template("{msg:17}: [{wide_bar:.blue}] {pos:2}/{len:2} timeblocks ({elapsed_precise}<{eta_precise})").unwrap()
                .progress_chars("=> "),
        )
        .with_position(0)
        .with_message("Subtracted writing");
        let write_progress = multi_progress.add(pb);

        // Use a variable to track whether any threads have an issue.
        let error = AtomicCell::new(false);

        info!("Reading input data, sky modelling, and writing");
        let scoped_threads_result: Result<String, VisSubtractError> = thread::scope(|scope| {
            // Input visibility-data reading thread.
            let data_handle: thread::ScopedJoinHandle<Result<(), VisReadError>> =
                thread::Builder::new()
                    .name("read".to_string())
                    .spawn_scoped(scope, || {
                        // If a panic happens, update our atomic error.
                        defer_on_unwind! { error.store(true); }
                        read_progress.tick();

                        for timeblock in &input_vis_params.timeblocks {
                            // Read data to fill the buffer, pausing when the buffer is
                            // full to write it all out.
                            let mut cross_data_fb = Array2::zeros(vis_shape);
                            let mut cross_weights_fb = Array2::zeros(vis_shape);

                            let result = self.input_vis_params.read_timeblock(
                                timeblock,
                                cross_data_fb.view_mut(),
                                cross_weights_fb.view_mut(),
                                None,
                                &error,
                            );

                            // If the result of reading data was an error, allow the other
                            // threads to see this so they can abandon their work early.
                            if result.is_err() {
                                error.store(true);
                            }
                            result?;

                            // Should we continue?
                            if error.load() {
                                return Ok(());
                            }

                            match tx_model.send(VisTimestep {
                                cross_data_fb: cross_data_fb.into_shared(),
                                cross_weights_fb: cross_weights_fb.into_shared(),
                                autos: None,
                                timestamp: timeblock.median,
                            }) {
                                Ok(()) => (),
                                // If we can't send the message, it's because the channel
                                // has been closed on the other side. That should only
                                // happen because the writer has exited due to error; in
                                // that case, just exit this thread.
                                Err(_) => return Ok(()),
                            }

                            read_progress.inc(1);
                        }

                        debug!("Finished reading");
                        read_progress.abandon_with_message("Finished reading visibilities");
                        drop(tx_model);
                        Ok(())
                    })
                    .expect("OS can create threads");

            // Sky-model generation and subtraction thread.
            let model_handle: ScopedJoinHandle<Result<(), ModelError>> = thread::Builder::new()
                .name("model".to_string())
                .spawn_scoped(scope, || {
                    defer_on_unwind! { error.store(true); }
                    model_progress.tick();

                    let result = model_thread(
                        &**beam,
                        source_list,
                        input_vis_params,
                        *apply_precession,
                        vis_shape,
                        rx_model,
                        tx_write,
                        &error,
                        model_progress,
                    );
                    if result.is_err() {
                        error.store(true);
                    }
                    result
                })
                .expect("OS can create threads");

            // Subtracted vis writing thread.
            let write_handle = thread::Builder::new()
                .name("write".to_string())
                .spawn_scoped(scope, || {
                    defer_on_unwind! { error.store(true); }
                    write_progress.tick();

                    let result = write_vis(
                        &output_vis_params.output_files,
                        obs_context.array_position,
                        obs_context.phase_centre,
                        obs_context.pointing_centre,
                        &obs_context.tile_xyzs,
                        &obs_context.tile_names,
                        obs_context.obsid,
                        &output_vis_params.output_timeblocks,
                        input_vis_params.time_res,
                        input_vis_params.dut1,
                        &input_vis_params.spw,
                        &input_vis_params
                            .tile_baseline_flags
                            .unflagged_cross_baseline_to_tile_map
                            .values()
                            .copied()
                            .sorted()
                            .collect::<Vec<_>>(),
                        output_vis_params.output_time_average_factor,
                        output_vis_params.output_freq_average_factor,
                        input_vis_params.vis_reader.get_marlu_mwa_info().as_ref(),
                        output_vis_params.write_smallest_contiguous_band,
                        rx_write,
                        &error,
                        Some(write_progress),
                    );
                    if result.is_err() {
                        error.store(true);
                    }
                    result
                })
                .expect("OS can create threads");

            // Join all thread handles. This propagates any errors and lets us know
            // if any threads panicked, if panics aren't aborting as per the
            // Cargo.toml. (It would be nice to capture the panic information, if
            // it's possible, but I don't know how, so panics are currently
            // aborting.)
            data_handle.join().unwrap()?;
            model_handle.join().unwrap()?;
            let write_message = write_handle.join().unwrap()?;
            Ok(write_message)
        });

        // Propagate errors and print out the write message.
        info!("{}", scoped_threads_result?);

        Ok(())
    }
}

#[allow(clippy::too_many_arguments)]
fn model_thread(
    beam: &dyn Beam,
    source_list: &SourceList,
    input_vis_params: &InputVisParams,
    apply_precession: bool,
    vis_shape: (usize, usize),
    rx: Receiver<VisTimestep>,
    tx: Sender<VisTimestep>,
    error: &AtomicCell<bool>,
    progress_bar: ProgressBar,
) -> Result<(), ModelError> {
    let obs_context = input_vis_params.get_obs_context();
    let unflagged_tile_xyzs = obs_context
        .tile_xyzs
        .iter()
        .enumerate()
        .filter(|(i, _)| {
            !input_vis_params
                .tile_baseline_flags
                .flagged_tiles
                .contains(i)
        })
        .map(|(_, xyz)| *xyz)
        .collect::<Vec<_>>();
    let freqs = input_vis_params
        .spw
        .chanblocks
        .iter()
        .map(|c| c.freq)
        .collect::<Vec<_>>();
    let modeller = new_sky_modeller(
        beam,
        source_list,
        obs_context.polarisations,
        &unflagged_tile_xyzs,
        &freqs,
        &input_vis_params.tile_baseline_flags.flagged_tiles,
        obs_context.phase_centre,
        obs_context.array_position.longitude_rad,
        obs_context.array_position.latitude_rad,
        input_vis_params.dut1,
        apply_precession,
    )?;

    // Recycle an array for model visibilities.
    let mut vis_model_fb = Array2::zeros(vis_shape);

    // Iterate over the incoming data.
    for VisTimestep {
        mut cross_data_fb,
        cross_weights_fb,
        autos,
        timestamp,
    } in rx.iter()
    {
        debug!("Modelling timestamp {}", timestamp.to_gpst_seconds());
        modeller.model_timestep_with(timestamp, vis_model_fb.view_mut())?;
        cross_data_fb
            .iter_mut()
            .zip_eq(vis_model_fb.iter())
            .for_each(|(vis_data, vis_model)| {
                *vis_data =
                    Jones::from(Jones::<f64>::from(*vis_data) - Jones::<f64>::from(*vis_model));
            });
        vis_model_fb.fill(Jones::default());

        // Should we continue?
        if error.load() {
            return Ok(());
        }

        match tx.send(VisTimestep {
            cross_data_fb,
            cross_weights_fb,
            autos,
            timestamp,
        }) {
            Ok(()) => (),
            Err(_) => return Ok(()),
        }
        progress_bar.inc(1);
    }

    debug!("Finished modelling");
    progress_bar.abandon_with_message("Finished generating sky model");
    Ok(())
}

#[derive(thiserror::Error, Debug)]
pub(crate) enum VisSubtractError {
    #[error(transparent)]
    VisRead(#[from] crate::io::read::VisReadError),

    #[error(transparent)]
    VisWrite(#[from] crate::io::write::VisWriteError),

    #[error(transparent)]
    Model(#[from] crate::model::ModelError),

    #[error(transparent)]
    IO(#[from] std::io::Error),

    #[cfg(any(feature = "cuda", feature = "hip"))]
    #[error(transparent)]
    Gpu(#[from] crate::gpu::GpuError),
}
