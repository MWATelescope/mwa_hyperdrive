// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

use std::thread::{self, ScopedJoinHandle};

use crossbeam_channel::bounded;
use crossbeam_utils::atomic::AtomicCell;
use indicatif::{MultiProgress, ProgressBar, ProgressDrawTarget, ProgressStyle};
use itertools::Itertools;
use log::{debug, info};
use ndarray::prelude::*;
use scopeguard::defer_on_unwind;

use super::{InputVisParams, OutputVisParams};
use crate::{
    io::{
        read::VisReadError,
        write::{write_vis, VisTimestep},
    },
    PROGRESS_BARS,
};

pub(crate) struct VisConvertParams {
    pub(crate) input_vis_params: InputVisParams,
    pub(crate) output_vis_params: OutputVisParams,
}

impl VisConvertParams {
    pub(crate) fn run(&self) -> Result<(), VisConvertError> {
        let Self {
            input_vis_params,
            output_vis_params,
        } = self;

        Self::run_inner(input_vis_params, output_vis_params)
    }

    // This function does the actual work, and only exists because
    // `SolutionsApplyParams` is doing the exact same thing, but I can't work
    // out how to make a `&VisConvertParams` from `&InputVisParams` and
    // `&OutputVisParams` (if it's possible).
    pub(super) fn run_inner(
        input_vis_params: &InputVisParams,
        output_vis_params: &OutputVisParams,
    ) -> Result<(), VisConvertError> {
        let obs_context = input_vis_params.get_obs_context();

        // Channel for transferring visibilities from the reader to the writer.
        let (tx_data, rx_data) = bounded(3);

        // Progress bars.
        let multi_progress = MultiProgress::with_draw_target(if PROGRESS_BARS.load() {
            ProgressDrawTarget::stdout()
        } else {
            ProgressDrawTarget::hidden()
        });
        let pb = ProgressBar::new(input_vis_params.timeblocks.len() as _)
            .with_style(
                ProgressStyle::default_bar()
                    .template("{msg:18}: [{wide_bar:.blue}] {pos:2}/{len:2} timeblocks ({elapsed_precise}<{eta_precise})").unwrap()
                    .progress_chars("=> "),
            )
            .with_position(0)
            .with_message("Reading data");
        let read_progress = multi_progress.add(pb);
        let pb = ProgressBar::new(output_vis_params.output_timeblocks.len() as _)
            .with_style(
                ProgressStyle::default_bar()
                    .template("{msg:18}: [{wide_bar:.blue}] {pos:2}/{len:2} timeblocks ({elapsed_precise}<{eta_precise})").unwrap()
                    .progress_chars("=> "),
            )
            .with_position(0)
            .with_message("Writing data");
        let write_progress = multi_progress.add(pb);

        // Use a variable to track whether any threads have an issue.
        let error = AtomicCell::new(false);

        info!("Reading input data and writing");
        let scoped_threads_result: Result<String, VisConvertError> = thread::scope(|scope| {
            // Input visibility-data reading thread.
            let data_handle: ScopedJoinHandle<Result<(), VisReadError>> = thread::Builder::new()
                .name("read".to_string())
                .spawn_scoped(scope, || {
                    // If a panic happens, update our atomic error.
                    defer_on_unwind! { error.store(true); }
                    read_progress.tick();

                    let num_unflagged_tiles = input_vis_params.get_num_unflagged_tiles();
                    let num_unflagged_cross_baselines =
                        (num_unflagged_tiles * (num_unflagged_tiles - 1)) / 2;
                    let cross_vis_shape = (
                        input_vis_params.spw.chanblocks.len(),
                        num_unflagged_cross_baselines,
                    );
                    let auto_vis_shape =
                        (input_vis_params.spw.chanblocks.len(), num_unflagged_tiles);

                    for timeblock in &input_vis_params.timeblocks {
                        let mut cross_data_fb = Array2::zeros(cross_vis_shape);
                        let mut cross_weights_fb = Array2::zeros(cross_vis_shape);
                        let mut autos_fb = if input_vis_params.using_autos {
                            Some((Array2::zeros(auto_vis_shape), Array2::zeros(auto_vis_shape)))
                        } else {
                            None
                        };

                        let result = input_vis_params.read_timeblock(
                            timeblock,
                            cross_data_fb.view_mut(),
                            cross_weights_fb.view_mut(),
                            autos_fb.as_mut().map(|(d, w)| (d.view_mut(), w.view_mut())),
                            &error,
                        );
                        // If the result of reading data was an error, allow the
                        // other threads to see this so they can abandon their work
                        // early.
                        if result.is_err() {
                            error.store(true);
                        }
                        result?;

                        // Send the data as timesteps.
                        match tx_data.send(VisTimestep {
                            cross_data_fb: cross_data_fb.into_shared(),
                            cross_weights_fb: cross_weights_fb.into_shared(),
                            autos: autos_fb.map(|(d, w)| (d.into_shared(), w.into_shared())),
                            timestamp: timeblock.median,
                        }) {
                            Ok(()) => (),
                            // If we can't send the message, it's because the
                            // channel has been closed on the other side. That
                            // should only happen because the writer has exited due
                            // to error; in that case, just exit this thread.
                            Err(_) => return Ok(()),
                        }

                        read_progress.inc(1);
                    }

                    drop(tx_data);
                    debug!("Finished reading");
                    read_progress.abandon_with_message("Finished reading visibilities");
                    Ok(())
                })
                .expect("OS can create threads");

            // Calibrated vis writing thread.
            let write_handle = thread::Builder::new()
                .name("write".to_string())
                .spawn_scoped(scope, || {
                    defer_on_unwind! { error.store(true); }
                    write_progress.tick();

                    // If we're not using autos, "disable" the
                    // `unflagged_tiles_iter` by making it not iterate over
                    // anything.
                    let total_num_tiles = if input_vis_params.using_autos {
                        obs_context.get_total_num_tiles()
                    } else {
                        0
                    };
                    let unflagged_tiles_iter = (0..total_num_tiles)
                        .filter(|i_tile| {
                            !input_vis_params
                                .tile_baseline_flags
                                .flagged_tiles
                                .contains(i_tile)
                        })
                        .map(|i_tile| (i_tile, i_tile));
                    // Form (sorted) unflagged baselines from our cross- and
                    // auto-correlation baselines.
                    let unflagged_cross_and_auto_baseline_tile_pairs = input_vis_params
                        .tile_baseline_flags
                        .tile_to_unflagged_cross_baseline_map
                        .keys()
                        .copied()
                        .chain(unflagged_tiles_iter)
                        .sorted()
                        .collect::<Vec<_>>();

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
                        &unflagged_cross_and_auto_baseline_tile_pairs,
                        output_vis_params.output_time_average_factor,
                        output_vis_params.output_freq_average_factor,
                        input_vis_params.vis_reader.get_marlu_mwa_info().as_ref(),
                        output_vis_params.write_smallest_contiguous_band,
                        rx_data,
                        &error,
                        Some(write_progress),
                    );
                    if result.is_err() {
                        error.store(true);
                    }
                    result
                })
                .expect("OS can create threads");

            // Join all thread handles. This propagates any errors and lets us
            // know if any threads panicked, if panics aren't aborting as per
            // the Cargo.toml. (It would be nice to capture the panic
            // information, if it's possible, but I don't know how, so panics
            // are currently aborting.)
            data_handle.join().unwrap()?;
            let write_message = write_handle.join().unwrap()?;
            Ok(write_message)
        });

        // Propagate errors and print out the write message.
        info!("{}", scoped_threads_result?);

        Ok(())
    }
}

#[derive(thiserror::Error, Debug)]
pub(crate) enum VisConvertError {
    #[error(transparent)]
    VisRead(#[from] crate::io::read::VisReadError),

    #[error(transparent)]
    VisWrite(#[from] crate::io::write::VisWriteError),

    #[error(transparent)]
    IO(#[from] std::io::Error),
}
