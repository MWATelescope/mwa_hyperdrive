// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Code to handle direction-independent calibration.
//!
//! This code borrows heavily from Torrance Hodgson's excellent Julia code at
//! https://github.com/torrance/MWAjl

use std::ops::Deref;

use crossbeam_channel::{bounded, unbounded};
use crossbeam_utils::thread::scope;
use indicatif::{MultiProgress, ProgressBar, ProgressDrawTarget, ProgressStyle};
use log::{debug, info, trace};
use ndarray::prelude::*;
use rayon::prelude::*;

use super::{model, solutions::CalibrationSolutions, CalibrateError, CalibrateParams};
use crate::data_formats::uvfits::UvfitsWriter;
use crate::precession::precess_time;
use crate::{math::cross_correlation_baseline_to_tiles, *};
use mwa_hyperdrive_core::xyz;

/// Do all the steps required for direction-independent calibration; read the
/// input data, generate a model against it, and write the solutions out.
pub fn di_cal(params: &CalibrateParams) -> Result<(), CalibrateError> {
    let obs_context = params.input_data.get_obs_context();
    let freq_context = params.input_data.get_freq_context();

    trace!("Allocating memory for input data visibilities and model visibilities");
    let vis_shape = (
        params.timesteps.len(),
        params.unflagged_baseline_to_tile_map.len(),
        params.freq.unflagged_fine_chans.len(),
    );
    let mut vis_data: Array3<Jones<f32>> = Array3::from_elem(vis_shape, Jones::default());
    let mut vis_model: Array3<Jones<f32>> = Array3::from_elem(vis_shape, Jones::default());
    debug!(
        "Shape of data and model arrays: ({} timesteps, {} baselines, {} channels) ({} MiB each)",
        vis_shape.0,
        vis_shape.1,
        vis_shape.2,
        vis_shape.0 * vis_shape.1 * vis_shape.2 * std::mem::size_of_val(&vis_data[[0, 0, 0]])
        // 1024 * 1024 == 1 MiB.
        / 1024 / 1024
    );

    // Set up our producer (IO reading and sending) thread, worker (IO receiving
    // and predicting) thread and model (writes the sky model to a file) thread.
    // By doing things this way, the disk and CPU is fully utilised; the input
    // data and our predicted model is assembled as efficiently as possible.
    info!("Beginning reading of input data and prediction");
    // Data communication channel. The producer might send an error on this
    // channel; it's up to the worker to propagate it.
    let (sx_data, rx_data) = unbounded();
    // Model communication channel. The worker might send an error on this
    // channel.
    let (sx_model, rx_model) = unbounded();
    // Final channel. Used to communicate with the main thread outside the
    // thread scope.
    let (sx_final, rx_final) = bounded(1);

    // Progress bars. Courtesy Dev.
    let multi_progress = MultiProgress::with_draw_target(ProgressDrawTarget::stdout());
    let read_progress = multi_progress.add(
        ProgressBar::new(vis_shape.0 as _)
            .with_style(
                ProgressStyle::default_bar()
                    .template("{msg:17}: [{wide_bar:.blue}] {pos:2}/{len:2} ({elapsed_precise}<{eta_precise})")
                    .progress_chars("=> "),
            )
            .with_position(0)
            .with_message("Reading timesteps"),
    );
    let model_progress = multi_progress.add(
        ProgressBar::new(vis_shape.0 as _)
            .with_style(
                ProgressStyle::default_bar()
                    .template("{msg:17}: [{wide_bar:.blue}] {pos:2}/{len:2} ({elapsed_precise}<{eta_precise})")
                    .progress_chars("=> "),
            )
            .with_position(0)
            .with_message("Sky modelling"),
    );
    // Only add a model writing progress bar if we need it.
    let model_write_progress = params.model_file.clone().map(|_|
        multi_progress.add(
            ProgressBar::new(vis_shape.0 as _)
                .with_style(
                    ProgressStyle::default_bar()
                        .template("{msg:17}: [{wide_bar:.blue}] {pos:2}/{len:2} ({elapsed_precise}<{eta_precise})")
                        .progress_chars("=> "),
                )
                .with_position(0)
                .with_message("Model writing"),
        )
    );

    scope(|scope| {
        // Spawn a thread to draw the progress bars.
        scope.spawn(|_| {
            multi_progress.join().unwrap();
        });

        // Mutable slices of the "global" data and model arrays. These allow
        // threads to mutate the global arrays in parallel (using the
        // Arc<Mutex<_>> pattern would kill performance here).
        let vis_data_slices: Vec<_> = vis_data.outer_iter_mut().collect();
        let vis_model_slices: Vec<_> = vis_model.outer_iter_mut().collect();

        // Producer (input data reading thread).
        scope.spawn(move |_| {
            for ((&timestep, vis_data_slice), vis_model_slice) in params
                .timesteps
                .iter()
                .zip(vis_data_slices)
                .zip(vis_model_slices)
            {
                let read_result = params.input_data.read(
                    vis_data_slice,
                    timestep,
                    &params.tile_to_unflagged_baseline_map,
                    &params.freq.fine_chan_flags,
                );
                let read_failed = read_result.is_err();
                // Send the result of the read to the worker thread.
                let msg = read_result
                    .map(|weights| (timestep, vis_model_slice, weights))
                    .map_err(CalibrateError::from);
                // If we can't send the message, it's because the channel has
                // been closed on the other side. That should only happen
                // because the worker has exited due to error; in that case,
                // just exit this thread.
                match sx_data.send(msg) {
                    Ok(_) => (),
                    Err(_) => break,
                }
                // If the result of the read was erroneous, then exit now.
                if read_failed {
                    break;
                }

                read_progress.inc(1);
            }

            // By dropping the send channel, we signal to the worker thread that
            // there is no more incoming data, and it can stop waiting.
            drop(sx_data);
            read_progress.finish_with_message("Finished reading input data");
        });

        // Worker (predictor thread). Only one thread receives the input data,
        // but it is processed in parallel. This is much more efficient than
        // having slices of the input data being processed serially by
        // individual threads.
        scope.spawn(move |_| {
            // Split the source list by component. This only needs to be done
            // once, so do it outside the loop.
            match model::split_components(
                &params.source_list,
                &params.freq.unflagged_fine_chan_freqs,
                &obs_context.phase_centre,
            ) {
                Err(e) => {
                    sx_model.send(Err(CalibrateError::from(e))).unwrap();
                }
                Ok(split_components) => {
                    // Iterate on the receive channel forever. This terminates when
                    // there is no data in the channel _and_ the sender has been
                    // dropped.
                    for msg in rx_data.iter() {
                        let (timestep, mut vis_model_slice, weights) = match msg {
                            Ok(msg) => msg,
                            Err(e) => {
                                // Propagate the error.
                                sx_model.send(Err(e)).unwrap();
                                break;
                            }
                        };
                        let timestamp = obs_context.timesteps[timestep];

                        // TODO: Allow the user to turn off precession.
                        let precession_info = precess_time(
                            &obs_context.phase_centre,
                            &timestamp,
                            params.array_longitude,
                            params.array_latitude,
                        );
                        // Apply precession to the tile XYZ positions.
                        let precessed_tile_xyzs =
                            precession_info.precess_xyz_parallel(&params.unflagged_tile_xyzs);
                        let uvws = xyz::xyzs_to_uvws(
                            &precessed_tile_xyzs,
                            &obs_context
                                .phase_centre
                                .to_hadec(precession_info.lmst_j2000),
                        );

                        let model_result = model::model_timestep(
                            vis_model_slice.view_mut(),
                            weights.view(),
                            &split_components,
                            params.beam.deref(),
                            precession_info.lmst_j2000,
                            &precessed_tile_xyzs,
                            &uvws,
                            &params.freq.unflagged_fine_chan_freqs,
                        );
                        let model_failed = model_result.is_err();
                        let msg = model_result.map(|_| (vis_model_slice, weights, uvws, timestamp));
                        // If we can't send the message, it's because the
                        // channel has been closed on the other side. That
                        // should only happen because the thread has exited due
                        // to error; in that case, just exit this thread.
                        match sx_model.send(msg) {
                            Ok(_) => (),
                            Err(_) => break,
                        }
                        if model_failed {
                            break;
                        }

                        model_progress.inc(1);
                    }
                    model_progress.finish_with_message("Finished generating sky model");
                }
            };

            drop(sx_model);
        });

        // Model writing thread. If the user hasn't specified to write the model
        // to a file, then this thread just propagates errors.
        scope.spawn(move |_| {
            // If the user wants the sky model written out, create the file
            // here. This can take a good deal of time; by creating the file in
            // a thread, the other threads can do useful work in the meantime.
            let model_writer_result = if let Some(model_pb) = &params.model_file {
                info!("Writing the sky model to {}", model_pb.display());
                let start_epoch = &obs_context.timesteps[params.timesteps[0]];
                let centre_freq =
                    freq_context.fine_chan_freqs[0] + freq_context.total_bandwidth / 2.0;
                let obs_name = obs_context.obsid.map(|o| format!("{}", o));

                let create_result = UvfitsWriter::new(
                    &model_pb,
                    // Don't include flagged timesteps or flagged baselines.
                    vis_shape.0,
                    vis_shape.1,
                    // ... but use all channels (including flagged channels).
                    // fits files expect a neat layout.
                    params.freq.num_fine_chans,
                    start_epoch,
                    freq_context.native_fine_chan_width,
                    centre_freq,
                    params.freq.num_fine_chans / 2,
                    &obs_context.phase_centre,
                    obs_name.as_deref(),
                );
                // Handle any errors during output model file creation.
                match create_result {
                    Err(e) => {
                        sx_final.send(Err(CalibrateError::from(e))).unwrap();
                        // If there was an error, make the code below exit early
                        // so that this thread does no more work. The error has
                        // already been propagated to the main thread.
                        Err(0)
                    }
                    Ok(v) => Ok(Some(v)),
                }
            } else {
                Ok(None)
            };

            match model_writer_result {
                Ok(Some(mut model_writer)) => {
                    for msg in rx_model.iter() {
                        // Handle any errors from the worker thread.
                        let (vis_model_timestep, weights, uvws, epoch) = match msg {
                            Err(e) => {
                                sx_final.send(Err(e)).unwrap();
                                break;
                            }
                            Ok(v) => v,
                        };

                        let write_result: Result<(), CalibrateError> = {
                            model_writer.open().map_err(CalibrateError::from).and_then(
                                |mut uvfits| {
                                    model_writer
                                        .write_from_vis(
                                            &mut uvfits,
                                            vis_model_timestep.view(),
                                            weights.view(),
                                            &uvws,
                                            &epoch,
                                            params.freq.num_fine_chans,
                                            &params.freq.fine_chan_flags,
                                        )
                                        .map_err(CalibrateError::from)
                                },
                            )
                        };
                        match write_result {
                            Err(e) => {
                                sx_final.send(Err(e)).unwrap();
                                break;
                            }
                            Ok(()) => (),
                        };

                        if let Some(pb) = &model_write_progress {
                            pb.inc(1)
                        }
                    }

                    // Send the model writer object out to the main thread.
                    sx_final.send(Ok(Some(model_writer))).unwrap();
                }

                // There's no model to write out, but we still need to handle
                // all of the incoming messages.
                Ok(None) => {
                    for msg in rx_model.iter() {
                        // Handle any errors from the worker thread.
                        if let Err(e) = msg {
                            sx_final.send(Err(e)).unwrap();
                            break;
                        };
                    }
                    // Send the model writer object out to the main thread.
                    sx_final.send(Ok(None)).unwrap();
                }

                // There was an error when creating the model file. Exit now.
                Err(_) => (),
            }

            drop(sx_final);
            if let Some(pb) = model_write_progress {
                pb.finish_with_message("Finished writing sky model");
            }
        });
    })
    .unwrap();

    // Handle messages from the scoped threads.
    for msg in rx_final.iter() {
        match msg {
            // Finalise writing the model file.
            Ok(Some(model_writer)) => {
                trace!("Finalising writing of model uvfits file");
                model_writer.write_uvfits_antenna_table(
                    &params.unflagged_tile_names,
                    &params.unflagged_tile_xyzs,
                )?;
                if let Some(model_pb) = &params.model_file {
                    info!("Finished writing sky model to {}", model_pb.display());
                }
            }

            // We're not writing a model; nothing to do.
            Ok(None) => (),

            // If an error message comes in on this channel, propagate it.
            Err(e) => return Err(e),
        }
    }

    // TODO: Let the user determine this -- using all timesteps at once for now.
    let timeblock_len = params.timesteps.len();
    let num_timeblocks = 1;
    let mut chanblocks = params.freq.unflagged_fine_chans.iter().collect::<Vec<_>>();
    chanblocks.sort_unstable();

    // The shape of the array containing output Jones matrices.
    let total_num_tiles = obs_context.tile_xyzs.len();
    let num_unflagged_tiles = total_num_tiles - params.tile_flags.len();
    let total_num_fine_freq_chans = freq_context.fine_chan_freqs.len();
    let shape = (num_timeblocks, num_unflagged_tiles, chanblocks.len());
    let mut sols = CalibrationSolutions {
        file: params.output_solutions_filename.clone(),
        di_jones: Array3::from_elem(shape, Jones::identity()),
        num_timeblocks,
        total_num_tiles,
        total_num_fine_freq_chans,
        timesteps: params
            .timesteps
            .iter()
            .map(|&ts| obs_context.timesteps[ts])
            .collect(),
        obsid: obs_context.obsid,
        time_res: params.time_res,
    };
    debug!(
        "Shape of DI Jones matrices array: {:?} ({} MiB)",
        shape,
        shape.0 * shape.1 * shape.2 * std::mem::size_of_val(&sols.di_jones[[0, 0, 0]])
        // 1024 * 1024 == 1 MiB.
        / 1024 / 1024
    );

    // For each timeblock, calibrate all chanblocks in parallel.
    for timeblock in (0..num_timeblocks).into_iter() {
        info!("Calibrating timeblock {}", timeblock);
        let mut di_jones_rev = sols
            .di_jones
            .slice_mut(s![timeblock, .., ..])
            .reversed_axes();

        let pb = ProgressBar::new(chanblocks.len() as _)
            .with_style(
                ProgressStyle::default_bar()
                    .template("{msg:17}: [{wide_bar:.blue}] {pos:3}/{len:3} ({elapsed_precise}<{eta_precise})")
                    .progress_chars("=> "),
            );
        pb.set_draw_target(ProgressDrawTarget::stdout());

        let mut timeblock_cal_results = Vec::with_capacity(chanblocks.len());
        chanblocks
            .par_iter()
            .zip(di_jones_rev.outer_iter_mut())
            .enumerate()
            .map(|(chanblock_index, (&&chanblock, di_jones))| {
                let range = s![
                    timeblock * timeblock_len..(timeblock + 1) * timeblock_len,
                    ..,
                    chanblock_index..chanblock_index + 1
                ];
                let mut cal_result = calibrate(
                    vis_data.slice(range),
                    vis_model.slice(range),
                    di_jones,
                    params.max_iterations,
                    params.stop_threshold,
                    params.min_threshold,
                );
                cal_result.chanblock = Some(chanblock);
                cal_result.chanblock_index = Some(chanblock_index);

                let mut status_str = format!("Chanblock {:>3}", chanblock);
                if num_unflagged_tiles - cal_result.num_failed <= 4 {
                    status_str.push_str(&format!(
                        ": failed    ({:>2}): Too many antenna solutions failed ({})",
                        cal_result.num_iterations, cal_result.num_failed
                    ));
                } else if cal_result.max_precision > params.min_threshold {
                    status_str.push_str(&format!(
                        ": failed    ({:>2}): {:.5e} > {:e}",
                        cal_result.num_iterations, cal_result.max_precision, params.min_threshold,
                    ));
                } else if cal_result.max_precision > params.stop_threshold {
                    status_str.push_str(&format!(
                        ": converged ({:>2}): {:e} > {:.5e} > {:e}",
                        cal_result.num_iterations,
                        params.min_threshold,
                        cal_result.max_precision,
                        params.stop_threshold
                    ));
                } else {
                    status_str.push_str(&format!(
                        ": converged ({:>2}): {:e} > {:.5e}",
                        cal_result.num_iterations, params.stop_threshold, cal_result.max_precision
                    ));
                }
                pb.println(status_str);
                pb.inc(1);
                cal_result
            })
            .collect_into_vec(&mut timeblock_cal_results);
        debug_assert_eq!(timeblock_cal_results.len(), chanblocks.len());
        let total_converged = timeblock_cal_results
            .iter()
            .filter(|result| result.converged)
            .count();
        pb.finish_with_message(format!(
            "Timeblock {}: {}/{} chanblocks succeeded",
            timeblock,
            total_converged,
            chanblocks.len()
        ));

        // Attempt to calibrate any chanblocks that failed by taking solutions
        // from nearby chanblocks as starting points.
        if total_converged > 0 && total_converged != chanblocks.len() {
            info!("Attempting to calibrate failed chanblocks");

            // Iterate over all the calibration results until we find one that
            // failed. Then find the next that succeeded. With a converged
            // solution on both sides (or either side) of the failures, use a
            // weighted average for a guess of what the Jones matrices should
            // be, then re-run MitchCal.
            let mut left = None;
            let mut pairs = vec![];
            let mut in_failures = false;
            for cal_result in timeblock_cal_results.iter() {
                match (in_failures, cal_result.converged) {
                    (false, true) => left = Some(cal_result.chanblock_index.unwrap()),
                    (false, false) => in_failures = true,
                    (true, true) => {
                        in_failures = false;
                        pairs.push((left, Some(cal_result.chanblock_index.unwrap())));
                        left = Some(cal_result.chanblock_index.unwrap());
                    }
                    (true, false) => (),
                }
            }

            for p in pairs {
                match p {
                    (Some(l), Some(r)) => {
                        let left_sol = di_jones_rev.slice(s![l, ..]).to_owned();
                        let right_sol = di_jones_rev.slice(s![r, ..]).to_owned();
                        for i in l + 1..r {
                            let left_weight = (r - i) as f64;
                            let right_weight = (i - l) as f64;
                            let weighted_sol = (&left_sol * left_weight
                                + &right_sol * right_weight)
                                / (r - l) as f64;
                            di_jones_rev.slice_mut(s![i, ..]).assign(&weighted_sol);
                        }
                    }
                    (Some(l), None) => {
                        let left_sol = di_jones_rev.slice(s![l, ..]).to_owned();
                        di_jones_rev
                            .slice_mut(s![l + 1..chanblocks.len(), ..])
                            .assign(&left_sol);
                    }
                    (None, Some(r)) => {
                        let right_sol = di_jones_rev.slice(s![r, ..]).to_owned();
                        di_jones_rev.slice_mut(s![0..r, ..]).assign(&right_sol);
                    }
                    (None, None) => unreachable!(),
                }
            }

            // Repeat calibration.
            let num_failures = chanblocks.len() - total_converged;
            let pb = ProgressBar::new(num_failures as _)
                .with_style(
                    ProgressStyle::default_bar()
                        .template("{msg:17}: [{wide_bar:.blue}] {pos:3}/{len:3} ({elapsed_precise}<{eta_precise})")
                        .progress_chars("=> "),
                );
            pb.set_draw_target(ProgressDrawTarget::stdout());

            let new_timeblock_cal_results: Vec<CalibrationResult> = chanblocks
                .par_iter()
                .zip(di_jones_rev.outer_iter_mut())
                .zip(timeblock_cal_results.par_iter())
                .enumerate()
                .filter(|(_, ((_, _), prev_result))| !prev_result.converged)
                .map(|(chanblock_index, ((&&chanblock, di_jones), _))| {
                    let range = s![
                        timeblock * timeblock_len..(timeblock + 1) * timeblock_len,
                        ..,
                        chanblock_index..chanblock_index + 1
                    ];
                    let mut cal_result = calibrate(
                        vis_data.slice(range),
                        vis_model.slice(range),
                        di_jones,
                        params.max_iterations,
                        params.stop_threshold,
                        params.min_threshold,
                    );
                    cal_result.chanblock = Some(chanblock);
                    cal_result.chanblock_index = Some(chanblock_index);

                    let mut status_str = format!("Chanblock {:>3}", chanblock);
                    if num_unflagged_tiles - cal_result.num_failed <= 4 {
                        status_str.push_str(&format!(
                            ": failed    ({:>2}): Too many antenna solutions failed ({})",
                            cal_result.num_iterations, cal_result.num_failed
                        ));
                    } else if cal_result.max_precision > params.min_threshold {
                        status_str.push_str(&format!(
                            ": failed    ({:>2}): {:.5e} > {:e}",
                            cal_result.num_iterations,
                            cal_result.max_precision,
                            params.min_threshold,
                        ));
                    } else if cal_result.max_precision > params.stop_threshold {
                        status_str.push_str(&format!(
                            ": converged ({:>2}): {:e} > {:.5e} > {:e}",
                            cal_result.num_iterations,
                            params.min_threshold,
                            cal_result.max_precision,
                            params.stop_threshold
                        ));
                    } else {
                        status_str.push_str(&format!(
                            ": converged ({:>2}): {:e} > {:.5e}",
                            cal_result.num_iterations,
                            params.stop_threshold,
                            cal_result.max_precision
                        ));
                    }
                    pb.println(status_str);
                    pb.inc(1);
                    cal_result
                })
                .collect();
            debug_assert_eq!(new_timeblock_cal_results.len(), num_failures);
            let new_converged = new_timeblock_cal_results
                .iter()
                .filter(|result| result.converged)
                .count();
            let new_total_converged = total_converged + new_converged;
            pb.finish_with_message(format!(
                "Timeblock {}: {}/{} reattempted chanblocks succeeded",
                timeblock, new_converged, num_failures
            ));
            info!(
                "Timeblock {}: {}/{} chanblocks succeeded",
                timeblock,
                new_total_converged,
                chanblocks.len()
            );
        }
    }

    // Write out the solutions.
    trace!("Writing solutions...");
    sols.write_solutions_from_ext(&params.tile_flags, &params.freq.unflagged_fine_chans)?;
    info!("Calibration solutions written to {}", sols.file.display());

    Ok(())
}

struct CalibrationResult {
    num_iterations: usize,
    converged: bool,
    max_precision: f64,
    num_failed: usize,
    chanblock: Option<usize>,
    chanblock_index: Option<usize>,
}

/// Calibrate the antennas of the array by comparing the observed input data
/// against our generated model. Return the number of iterations this took.
///
/// This function is intended to be run in parallel; for that reason, no
/// parallel code is inside this function.
fn calibrate(
    data: ArrayView3<Jones<f32>>,
    model: ArrayView3<Jones<f32>>,
    mut di_jones: ArrayViewMut1<Jones<f64>>,
    max_iterations: usize,
    stop_threshold: f64,
    min_threshold: f64,
) -> CalibrationResult {
    debug_assert_eq!(data.dim(), model.dim());

    let mut new_jones: Array1<Jones<f64>> = Array::from_elem(di_jones.dim(), Jones::default());
    let mut top: Array1<Jones<f64>> = Array::from_elem(di_jones.dim(), Jones::default());
    let mut bot: Array1<Jones<f64>> = Array::from_elem(di_jones.dim(), Jones::default());
    // The convergence precisions per antenna. They are stored per polarisation
    // for programming convenience, but really only we're interested in the
    // largest value in the entire array.
    let mut precisions: Array2<f64> = Array::zeros((di_jones.len(), 4));
    let mut failed: Array1<bool> = Array1::from_elem(di_jones.len(), false);

    // Shortcuts.
    let num_unflagged_tiles = di_jones.len_of(Axis(0));

    let mut iteration = 0;
    while iteration < max_iterations {
        iteration += 1;
        // Re-initialise top and bot.
        top.fill(Jones::default());
        bot.fill(Jones::default());

        calibration_loop(data, model, di_jones.view(), top.view_mut(), bot.view_mut());

        // Obtain the new DI Jones matrices from "top" and "bot".
        // Tile/antenna axis.
        di_jones
            .outer_iter_mut()
            .zip(new_jones.outer_iter_mut())
            .zip(top.outer_iter())
            .zip(bot.outer_iter())
            .zip(failed.iter_mut())
            .filter(|(_, &mut failed)| !failed)
            .for_each(|((((mut di_jones, mut new_jones), top), bot), failed)| {
                // Unflagged fine-channel axis.
                di_jones
                    .iter_mut()
                    .zip(new_jones.iter_mut())
                    .zip(top.iter())
                    .zip(bot.iter())
                    .for_each(|(((di_jones, new_jones), top), bot)| {
                        *new_jones = top.clone() / bot;
                        if new_jones.iter().any(|f| f.is_nan()) {
                            *failed = true;
                            *di_jones = Jones::default();
                            *new_jones = Jones::default();
                        }
                    });
            });

        // More than 4 antenna need to be present to get a good solution.
        let num_failed = failed.iter().filter(|&&f| f).count();
        if num_unflagged_tiles - num_failed <= 4 {
            break;
        }

        // On every even iteration, we test for convergence and set the new gain
        // solution as the average of the last two, as per Stefcal. This speeds
        // up convergence.
        if iteration % 2 == 0 {
            // Update the DI Jones matrices, and for each pair of Jones matrices
            // in new_jones and di_jones, form a maximum "distance" between
            // elements of the Jones matrices.
            di_jones
                .outer_iter_mut()
                .zip(new_jones.outer_iter())
                .zip(precisions.outer_iter_mut())
                .zip(failed.iter())
                .filter(|(_, &failed)| !failed)
                .for_each(|(((mut di_jones, new_jones), mut antenna_precision), _)| {
                    // antenna_precision = sum(norm_sqr(new_jones - di_jones)) / num_freqs
                    let jones_diff_sum = (&new_jones - &di_jones).into_iter().fold(
                        Array1::zeros(4),
                        |acc, diff_jones| {
                            let norm = diff_jones.norm_sqr();
                            acc + array![norm[0], norm[1], norm[2], norm[3]]
                        },
                    );
                    antenna_precision.assign(&(jones_diff_sum / di_jones.len() as f64));

                    // di_jones = 0.5 * (di_jones + new_jones)
                    di_jones += &new_jones;
                    di_jones.mapv_inplace(|v| v * 0.5);
                });

            // Stop iterating if we have reached the stop threshold.
            if precisions.iter().all(|&v| v < stop_threshold) {
                break;
            }
        } else {
            // On odd iterations, we simply update the DI Jones matrices with
            // the new ones.
            di_jones.assign(&new_jones);
        }
    }

    // Set failed antennas to NaN.
    di_jones
        .outer_iter_mut()
        .zip(failed.iter())
        .filter(|(_, &failed)| failed)
        .for_each(|(mut di_jones, _)| {
            di_jones.fill(Jones::nan());
        });

    // max_precision = maximum(distances)
    let max_precision: f64 = precisions
        .outer_iter()
        .zip(failed.iter())
        .filter(|(_, &failed)| !failed)
        .fold(0.0, |acc, (antenna_precision, _)| {
            // Rust really doesn't want you to use the .max() iterator method on
            // floats...
            acc.max(antenna_precision[[0]])
                .max(antenna_precision[[1]])
                .max(antenna_precision[[2]])
                .max(antenna_precision[[3]])
        });

    let num_failed = failed.iter().filter(|&&f| f).count();
    let converged = {
        // If only 4 or fewer antennas remain, or we never reached the minimum
        // threshold level, mark the solution as failed.
        if num_unflagged_tiles - num_failed <= 4 || max_precision > min_threshold {
            di_jones.fill(Jones::nan());
            false
        }
        // If converged within the minimum threshold, signal success.
        else {
            true
        }
    };

    CalibrationResult {
        num_iterations: iteration,
        converged,
        max_precision,
        num_failed,
        chanblock: None,
        chanblock_index: None,
    }
}

/// "MitchCal".
fn calibration_loop(
    data: ArrayView3<Jones<f32>>,
    model: ArrayView3<Jones<f32>>,
    di_jones: ArrayView1<Jones<f64>>,
    mut top: ArrayViewMut1<Jones<f64>>,
    mut bot: ArrayViewMut1<Jones<f64>>,
) {
    let num_unflagged_tiles = di_jones.len_of(Axis(0));

    // Time axis.
    data.outer_iter()
        .zip(model.outer_iter())
        .for_each(|(data_time, model_time)| {
            // Unflagged baseline axis.
            data_time
                .outer_iter()
                .zip(model_time.outer_iter())
                .enumerate()
                .for_each(|(unflagged_bl_index, (data_bl, model_bl))| {
                    let (tile1, tile2) = cross_correlation_baseline_to_tiles(
                        num_unflagged_tiles,
                        unflagged_bl_index,
                    );

                    // Unflagged frequency chan axis.
                    data_bl
                        .iter()
                        .zip(model_bl.iter())
                        .for_each(|(j_data, j_model)| {
                            // Copy and promote the data and model Jones
                            // matrices.
                            let j_data: Jones<f64> = j_data.into();
                            let j_model: Jones<f64> = j_model.into();

                            // Suppress boundary checks for maximum performance!
                            unsafe {
                                let j_t1 = di_jones.uget(tile1);
                                let j_t2 = di_jones.uget(tile2);

                                let top_t1 = top.uget_mut(tile1);
                                let bot_t1 = bot.uget_mut(tile1);

                                // André's calibrate: ( D J M^H ) / ( M J^H J M^H )
                                // J M^H
                                let z = Jones::axbh(j_t2, &j_model);
                                // D (J M^H)
                                Jones::plus_axb(top_t1, &j_data, &z);
                                // (J M^H)^H (J M^H)
                                Jones::plus_ahxb(bot_t1, &z, &z);

                                let top_t2 = top.uget_mut(tile2);
                                let bot_t2 = bot.uget_mut(tile2);

                                // J (M^H)^H
                                let z = Jones::axb(j_t1, &j_model);
                                // D^H (J M^H)^H
                                Jones::plus_ahxb(top_t2, &j_data, &z);
                                // (J M^H) (J M^H)
                                Jones::plus_ahxb(bot_t2, &z, &z);
                            }
                        })
                })
        });
}
