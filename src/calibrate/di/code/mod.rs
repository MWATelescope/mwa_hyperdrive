// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Code that *actually* does the calibration. This module exists mostly to keep
//! some things private so that they aren't misused.

#[cfg(test)]
mod tests;

use std::ops::Deref;

use crossbeam_channel::{bounded, unbounded};
use crossbeam_utils::thread::scope;
use hifitime::Epoch;
use indicatif::{MultiProgress, ProgressBar, ProgressDrawTarget, ProgressStyle};
use itertools::Either;
use log::{debug, info, trace};
use marlu::{
    c64,
    math::{cross_correlation_baseline_to_tiles, num_tiles_from_num_cross_correlation_baselines},
    pos::xyz::{xyzs_to_cross_uvws_parallel, XyzGeodetic},
    precession::precess_time,
    Jones,
};
use ndarray::prelude::*;
use rayon::prelude::*;

use crate::{
    calibrate::{
        params::CalibrateParams, solutions::CalibrationSolutions, CalibrateError, Chanblock,
        Timeblock,
    },
    data_formats::UvfitsWriter,
    model,
};
use mwa_hyperdrive_common::{cfg_if, hifitime, indicatif, itertools, log, marlu, ndarray, rayon};
use mwa_hyperdrive_srclist::ComponentList;

cfg_if::cfg_if! {
    if #[cfg(feature = "cuda")] {
        use mwa_hyperdrive_cuda as cuda;
        use cuda::modeller::SkyModellerCuda;
    }
}

pub(super) struct CalVis {
    /// Visibilites read from input data.
    pub(super) vis_data: Array3<Jones<f32>>,

    /// The weights on the visibilites read from input data.
    pub(super) vis_weights: Array3<f32>,

    /// Visibilites generated from the sky-model source list.
    pub(super) vis_model: Array3<Jones<f32>>,
}

/// For calibration, read in unflagged visibilities and generate sky-model
/// visibilities.
pub(super) fn get_cal_vis(
    params: &CalibrateParams,
    draw_progress_bar: bool,
) -> Result<CalVis, CalibrateError> {
    // Witchcraft to allow this code to be used with or without CUDA support
    // compiled.
    #[cfg(not(feature = "cuda"))]
    let use_cpu_for_modelling = true;
    #[cfg(feature = "cuda")]
    let use_cpu_for_modelling = params.use_cpu_for_modelling;

    if use_cpu_for_modelling {
        info!("Generating sky model visibilities on the CPU");
    } else {
        // TODO: Display GPU info.
        #[cfg(not(feature = "cuda-single"))]
        info!("Generating sky model visibilities on the GPU (double precision)");
        #[cfg(feature = "cuda-single")]
        info!("Generating sky model visibilities on the GPU (single precision)");
    }

    let obs_context = params.input_data.get_obs_context();
    // TODO: Use all fences, not just the first.
    let fence = params.fences.first();

    let vis_shape = (
        params.get_num_timesteps(),
        params.get_num_unflagged_cross_baselines(),
        fence.chanblocks.len(),
    );
    let num_elems = vis_shape.0 * vis_shape.1 * vis_shape.2;
    let size = num_elems * std::mem::size_of::<Jones<f32>>();
    let (size_unit, size) = if size < 1024_usize.pow(3) {
        ("MiB", size as f64 / (1024.0_f64.powi(2)))
    } else {
        ("GiB", size as f64 / (1024.0_f64.powi(3)))
    };
    debug!(
        "Shape of data and model arrays: ({} timesteps, {} baselines, {} channels; {:.2} {} each)",
        vis_shape.0, vis_shape.1, vis_shape.2, size, size_unit
    );

    // We need this many gibibytes to do calibration (two visibility arrays and
    // one weights array).
    let need_gib = (num_elems
        * (2 * std::mem::size_of::<Jones<f32>>() + std::mem::size_of::<f32>()))
        / 1024_usize.pow(3);
    let fallible_jones_allocator =
        |shape: (usize, usize, usize)| -> Result<Array3<Jones<f32>>, CalibrateError> {
            let mut v = Vec::new();
            let num_elems = shape.0 * shape.1 * shape.2;
            match v.try_reserve_exact(num_elems) {
                Ok(()) => {
                    v.resize(num_elems, Jones::default());
                    Ok(Array3::from_shape_vec(shape, v).unwrap())
                }
                Err(_) => Err(CalibrateError::InsufficientMemory {
                    // Instead of erroring out with how many GiB we need for *this*
                    // array, error out with how many we need total.
                    need_gib,
                }),
            }
        };
    let fallible_f32_allocator =
        |shape: (usize, usize, usize)| -> Result<Array3<f32>, CalibrateError> {
            let mut v = Vec::new();
            let num_elems = shape.0 * shape.1 * shape.2;
            match v.try_reserve_exact(num_elems) {
                Ok(()) => {
                    v.resize(num_elems, 0.0);
                    Ok(Array3::from_shape_vec(shape, v).unwrap())
                }
                Err(_) => Err(CalibrateError::InsufficientMemory { need_gib }),
            }
        };

    debug!("Allocating memory for input data visibilities and model visibilities");
    let mut vis_data: Array3<Jones<f32>> = fallible_jones_allocator(vis_shape)?;
    let mut vis_model: Array3<Jones<f32>> = fallible_jones_allocator(vis_shape)?;
    let mut vis_weights: Array3<f32> = fallible_f32_allocator(vis_shape)?;

    // Set up our producer (IO reading and sending) thread, sky-modelling
    // thread and model-writer (writes the sky model to a file) thread. By doing
    // things this way, the disk and CPU/GPU is fully utilised; the input data
    // and our sky model is assembled as efficiently as possible.
    info!("Reading input data and sky modelling");
    // Data communication channel. The producer might send an error on this
    // channel; it's up to the worker to propagate it.
    let (tx_data, rx_data) = unbounded();
    // Model communication channel. The worker might send an error on this
    // channel.
    let (tx_model, rx_model) = unbounded();
    // Final channel. Used to communicate with the main thread outside the
    // thread scope.
    let (tx_final, rx_final) = bounded(1);

    // Progress bars. Courtesy Dev.
    let multi_progress = MultiProgress::with_draw_target(if draw_progress_bar {
        ProgressDrawTarget::stdout()
    } else {
        ProgressDrawTarget::hidden()
    });
    let read_progress = multi_progress.add(
        ProgressBar::new(vis_shape.0 as _)
            .with_style(
                ProgressStyle::default_bar()
                    .template("{msg:17}: [{wide_bar:.blue}] {pos:2}/{len:2} timesteps ({elapsed_precise}<{eta_precise})")
                    .progress_chars("=> "),
            )
            .with_position(0)
            .with_message("Reading data"),
    );
    let model_progress = multi_progress.add(
        ProgressBar::new(vis_shape.0 as _)
            .with_style(
                ProgressStyle::default_bar()
                    .template("{msg:17}: [{wide_bar:.blue}] {pos:2}/{len:2} timesteps ({elapsed_precise}<{eta_precise})")
                    .progress_chars("=> "),
            )
            .with_position(0)
            .with_message("Sky modelling"),
    );
    // Only add a model writing progress bar if we need it.
    let model_write_progress = params.model_file.clone().map(|model_pb| {
        info!("Writing the sky model to {}", model_pb.display());
        multi_progress.add(
            ProgressBar::new(vis_shape.0 as _)
                .with_style(
                    ProgressStyle::default_bar()
                        .template("{msg:17}: [{wide_bar:.blue}] {pos:2}/{len:2} timesteps ({elapsed_precise}<{eta_precise})")
                        .progress_chars("=> "),
                )
                .with_position(0)
                .with_message("Model writing"),
        )
    });

    // Draw the progress bars.
    read_progress.tick();
    model_progress.tick();
    if let Some(pb) = &model_write_progress {
        pb.tick();
    }

    scope(|scope| {
        // Spawn a thread to draw the progress bars.
        scope.spawn(|_| {
            multi_progress.join().unwrap();
        });

        // Mutable slices of the "global" arrays. These allow threads to mutate
        // the global arrays in parallel (using the Arc<Mutex<_>> pattern would
        // kill performance here).
        let vis_data_slices = vis_data.outer_iter_mut();
        let vis_model_slices = vis_model.outer_iter_mut();
        let vis_weight_slices = vis_weights.outer_iter_mut();

        // Input visibility-data reading thread.
        scope.spawn(move |_| {
            for (((&timestep, vis_data_slice), vis_model_slice), mut vis_weight_slice) in params
                .timesteps
                .iter()
                .zip(vis_data_slices)
                .zip(vis_model_slices)
                .zip(vis_weight_slices)
            {
                let read_result =
                    params.read_crosses(vis_data_slice, vis_weight_slice.view_mut(), timestep);
                let read_failed = read_result.is_err();
                // Send the result of the read to the worker thread.
                let msg = read_result
                    .map(|()| (timestep, vis_model_slice, vis_weight_slice))
                    .map_err(CalibrateError::from);
                // If we can't send the message, it's because the channel has
                // been closed on the other side. That should only happen
                // because the worker has exited due to error; in that case,
                // just exit this thread.
                match tx_data.send(msg) {
                    Ok(()) => (),
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
            drop(tx_data);
            read_progress.abandon_with_message("Finished reading input data");
        });

        // Sky-model generation thread. Only one thread receives the input data,
        // but it is processed in parallel. This is much more efficient than
        // having slices of the input data being processed serially by
        // individual threads.
        scope.spawn(move |_| {
            // Split the sky-model components (for efficiency). Use an Either to hold
            // one type or the other (the types differ between CPU and GPU code).
            let modeller = if use_cpu_for_modelling {
                Either::Left(ComponentList::new(
                    &params.source_list,
                    &params.unflagged_fine_chan_freqs,
                    obs_context.phase_centre,
                ))
            } else {
                #[cfg(feature = "cuda")]
                unsafe {
                    Either::Right(
                        SkyModellerCuda::new(
                            params.beam.deref(),
                            &params.source_list,
                            &params.unflagged_fine_chan_freqs,
                            &params.unflagged_tile_xyzs,
                            &params.flagged_tiles,
                            obs_context.phase_centre,
                            params.array_latitude,
                            &crate::shapelets::SHAPELET_BASIS_VALUES,
                            crate::shapelets::SBF_L,
                            crate::shapelets::SBF_N,
                            crate::shapelets::SBF_C,
                            crate::shapelets::SBF_DX,
                        )
                        .unwrap(),
                    )
                }

                // It doesn't matter what goes in Right when we're not using CUDA.
                #[cfg(not(feature = "cuda"))]
                Either::Right(0)
            };

            // Iterate on the receive channel forever. This terminates when
            // there is no data in the channel _and_ the sender has been
            // dropped.
            for msg in rx_data.iter() {
                let (timestep, mut vis_model_slice, weights) = match msg {
                    Ok(msg) => msg,
                    Err(e) => {
                        // Propagate the error.
                        tx_model.send(Err(e)).unwrap();
                        break;
                    }
                };
                debug_assert_eq!(
                    vis_model_slice.dim(),
                    weights.dim(),
                    "vis_model_slice.dim() != weights.dim()"
                );

                let timestamp = obs_context.timestamps[timestep];

                // TODO: Allow the user to turn off precession.
                let precession_info = precess_time(
                    obs_context.phase_centre,
                    timestamp,
                    params.array_longitude,
                    params.array_latitude,
                );
                // Apply precession to the tile XYZ positions.
                let precessed_tile_xyzs =
                    precession_info.precess_xyz_parallel(&params.unflagged_tile_xyzs);
                let uvws = xyzs_to_cross_uvws_parallel(
                    &precessed_tile_xyzs,
                    obs_context
                        .phase_centre
                        .to_hadec(precession_info.lmst_j2000),
                );

                let model_result = if use_cpu_for_modelling {
                    model::model_timestep(
                        vis_model_slice.view_mut(),
                        modeller.as_ref().unwrap_left(),
                        params.beam.deref(),
                        precession_info.lmst_j2000,
                        &precessed_tile_xyzs,
                        &uvws,
                        &params.unflagged_fine_chan_freqs,
                        &params.unflagged_cross_baseline_to_tile_map,
                    )
                    .map_err(CalibrateError::from)
                } else {
                    #[cfg(feature = "cuda")]
                    unsafe {
                        modeller
                            .as_ref()
                            .unwrap_right()
                            .model_timestep(
                                vis_model_slice.view_mut(),
                                precession_info.lmst_j2000,
                                &uvws,
                            )
                            .unwrap();
                    }
                    Ok(())
                };
                let model_failed = model_result.is_err();

                let msg = model_result.map(|()| (vis_model_slice, weights, uvws, timestamp));
                // If we can't send the message, it's because the
                // channel has been closed on the other side. That
                // should only happen because the thread has exited due
                // to error; in that case, just exit this thread.
                match tx_model.send(msg) {
                    Ok(_) => (),
                    Err(_) => break,
                }
                if model_failed {
                    break;
                }

                model_progress.inc(1);
            }
            model_progress.abandon_with_message("Finished generating sky model");

            drop(tx_model);
        });

        // Model writing thread. If the user hasn't specified to write the model
        // to a file, then this thread just propagates errors.
        scope.spawn(move |_| {
            // If the user wants the sky model written out, create the file
            // here. This can take a good deal of time; by creating the file in
            // a thread, the other threads can do useful work in parallel.
            let model_writer_result = if let Some(model_pb) = &params.model_file {
                let start_epoch = params.timeblocks.first().start;
                let obs_name = obs_context.obsid.map(|o| format!("{}", o));

                let create_result = UvfitsWriter::new(
                    model_pb,
                    // Don't include flagged timesteps or flagged baselines.
                    vis_shape.0,
                    vis_shape.1,
                    // ... but use all channels (including flagged channels).
                    // fits files expect a neat layout.
                    fence.get_total_num_chanblocks(),
                    false,
                    start_epoch,
                    fence.freq_res,
                    fence.get_centre_freq(),
                    obs_context.phase_centre,
                    obs_name.as_deref(),
                    &params.unflagged_cross_baseline_to_tile_map,
                    &params.flagged_fine_chans,
                );
                // Handle any errors during output model file creation.
                match create_result {
                    Err(e) => {
                        tx_final.send(Err(CalibrateError::from(e))).unwrap();
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
                                tx_final.send(Err(e)).unwrap();
                                drop(rx_model);
                                break;
                            }
                            Ok(v) => v,
                        };

                        let write_result: Result<(), CalibrateError> = {
                            model_writer
                                .write_cross_timestep_vis(
                                    vis_model_timestep.view(),
                                    weights.view(),
                                    &uvws,
                                    epoch,
                                )
                                .map_err(CalibrateError::from)
                        };
                        match write_result {
                            Err(e) => {
                                tx_final.send(Err(e)).unwrap();
                                break;
                            }
                            Ok(()) => (),
                        };

                        if let Some(pb) = &model_write_progress {
                            pb.inc(1)
                        }
                    }

                    // Send the model writer object out to the main thread.
                    tx_final.send(Ok(Some(model_writer))).unwrap();
                }

                // There's no model to write out, but we still need to handle
                // all of the incoming messages.
                Ok(None) => {
                    for msg in rx_model.iter() {
                        match msg {
                            // Handle any errors from the worker thread.
                            Err(e) => {
                                tx_final.send(Err(e)).unwrap();
                                break;
                            }
                            Ok(v) => v,
                        };
                    }

                    // Send the model writer object out to the main thread.
                    tx_final.send(Ok(None)).unwrap();
                }

                // There was an error when creating the model file. Exit now.
                Err(_) => (),
            }

            drop(tx_final);
            if let Some(pb) = model_write_progress {
                pb.abandon_with_message("Finished writing sky model");
            }
        });
    })
    .unwrap();
    info!("Finished reading input data and sky modelling");

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

    Ok(CalVis {
        vis_data,
        vis_weights,
        vis_model,
    })
}

/// (Possibly) incomplete calibration solutions.
///
/// hyperdrive only reads in the data it needs for DI calibration; it ignores
/// any flagged tiles and channels in the input data. Consequentially, when DI
/// solutions are made from only unflagged data, these incomplete solutions need
/// to be "padded" with NaNs such that the complete calibration solutions can be
/// saved to disk or applied to an observation.
pub(super) struct IncompleteSolutions<'a> {
    /// Direction-independent calibration solutions *for only unflagged data*.
    /// The first dimension is timeblock, the second is unflagged tile, the
    /// third is unflagged chanblock.
    pub(super) di_jones: Array3<Jones<f64>>,

    /// The timeblocks used in calibration.
    timeblocks: &'a [Timeblock],

    /// The unflagged chanblocks used in calibration.
    chanblocks: &'a [Chanblock],

    /// The baseline weights used during calibration.
    baseline_weights: &'a [f64],

    /// The maximum allowed number of iterations during calibration.
    max_iterations: usize,

    /// The stop threshold used during calibration.
    stop_threshold: f64,

    /// The minimum threshold used during calibration.    
    min_threshold: f64,
}

impl<'a> IncompleteSolutions<'a> {
    /// Convert these [IncompleteSolutions] into "padded"
    /// [CalibrationSolutions].
    ///
    /// `all_tile_positions` includes flagged tiles, and its length is used as
    /// the total number of tiles. In conjunction with `tile_flags`, the
    /// solutions are "padded" for the flagged tiles.
    ///
    /// `all_chanblocks` would be the same as `self.chanblocks` if all channels
    /// were unflagged. Comparing these two collections lets us pad the flagged
    /// chanblocks.
    pub(super) fn into_cal_sols(
        self,
        all_tile_positions: &[XyzGeodetic],
        flagged_tiles: &[usize],
        flagged_chanblock_indices: &[u16],
        obsid: Option<u32>,
    ) -> CalibrationSolutions {
        let (num_timeblocks, num_unflagged_tiles, num_unflagged_chanblocks) = self.di_jones.dim();
        let total_num_tiles = all_tile_positions.len();
        let total_num_chanblocks = self.chanblocks.len() + flagged_chanblock_indices.len();

        // These things should always be true; if they aren't, it's a
        // programmer's fault.
        assert_eq!(num_timeblocks, self.timeblocks.len());
        assert!(num_unflagged_tiles <= total_num_tiles);
        assert_eq!(num_unflagged_tiles + flagged_tiles.len(), total_num_tiles);
        assert_eq!(num_unflagged_chanblocks, self.chanblocks.len());
        assert_eq!(
            num_unflagged_chanblocks + flagged_chanblock_indices.len(),
            total_num_chanblocks
        );

        // `out_di_jones` will contain the "complete" calibration solutions. The
        // data is the same as `self.di_jones`, but NaNs will be placed anywhere
        // a tile or chanblock was flagged. The "chanblock" terminology is
        // deliberate here; the amount of frequency/channel averaging on `self`
        // must propagate to `out_di_jones`.
        let mut out_di_jones = Array3::from_elem(
            (num_timeblocks, total_num_tiles, total_num_chanblocks),
            Jones::from([c64::new(f64::NAN, f64::NAN); 4]),
        );

        // Populate out_di_jones. The timeblocks are always 1-to-1.
        out_di_jones
            .outer_iter_mut()
            .into_par_iter()
            .zip(self.di_jones.outer_iter())
            .for_each(|(mut out_di_jones, di_jones)| {
                // Iterate over the tiles.
                let mut i_unflagged_tile = 0;
                out_di_jones
                    .outer_iter_mut()
                    .enumerate()
                    .for_each(|(i_tile, mut out_di_jones)| {
                        // Nothing needs to be done if this tile is flagged.
                        if !flagged_tiles.contains(&i_tile) {
                            // Iterate over the chanblocks.
                            let mut i_unflagged_chanblock = 0;
                            out_di_jones.iter_mut().enumerate().for_each(
                                |(i_chanblock, out_di_jones)| {
                                    // Nothing needs to be done if this
                                    // chanblock is flagged.
                                    if !flagged_chanblock_indices.contains(&(i_chanblock as u16)) {
                                        // The incomplete solutions aren't
                                        // inverted (i.e. they go from model to
                                        // data, but we want to store data to
                                        // model).
                                        *out_di_jones = di_jones
                                            [(i_unflagged_tile, i_unflagged_chanblock)]
                                            .inv();
                                        i_unflagged_chanblock += 1;
                                    }
                                },
                            );

                            i_unflagged_tile += 1;
                        }
                    });
            });

        CalibrationSolutions {
            di_jones: out_di_jones,
            flagged_tiles: flagged_tiles.to_vec(),
            flagged_chanblocks: flagged_chanblock_indices.to_vec(),
            average_timestamps: self.timeblocks.iter().map(|tb| tb.average).collect(),
            start_timestamps: self.timeblocks.iter().map(|tb| tb.start).collect(),
            end_timestamps: self.timeblocks.iter().map(|tb| tb.end).collect(),
            obsid,
        }
    }
}

/// Perform DI calibration on the data and model. Incomplete DI solutions are
/// returned; these need to be "padded" with NaNs by `into_cal_sols` before they
/// can be saved to disk or applied to an observation's visibilities.
///
/// This function basically wraps `calibrate_timeblock`, which does work in
/// parallel. For this reason, `calibrate_timeblocks` does nothing in parallel.
///
/// The way this code is currently structured mandates that all timetimes are
/// calibrated together (as if they all belonged to a single timeblock) before
/// any timeblocks are individually calibrated. This decision can be revisited.
#[allow(clippy::too_many_arguments)]
pub(super) fn calibrate_timeblocks<'a>(
    vis_data: ArrayView3<Jones<f32>>,
    vis_model: ArrayView3<Jones<f32>>,
    timeblocks: &'a [Timeblock],
    chanblocks: &'a [Chanblock],
    baseline_weights: &'a [f64],
    max_iterations: usize,
    stop_threshold: f64,
    min_threshold: f64,
    draw_progress_bar: bool,
) -> (IncompleteSolutions<'a>, Array2<CalibrationResult>) {
    let num_unflagged_tiles = num_tiles_from_num_cross_correlation_baselines(vis_data.dim().1);
    let num_timeblocks = timeblocks.len();
    let num_chanblocks = chanblocks.len();
    let shape = (num_timeblocks, num_unflagged_tiles, num_chanblocks);
    let mut di_jones = Array3::from_elem(shape, Jones::identity());

    let cal_results = if num_timeblocks == 1 {
        // Calibrate all timesteps together.
        let pb = make_calibration_progress_bar(
            num_chanblocks,
            "Calibrating".to_string(),
            draw_progress_bar,
        );
        let cal_results = calibrate_timeblock(
            vis_data.view(),
            vis_model.view(),
            di_jones.view_mut(),
            timeblocks.first().unwrap(),
            chanblocks,
            baseline_weights,
            max_iterations,
            stop_threshold,
            min_threshold,
            pb,
        );
        let total_converged_count = cal_results.iter().filter(|r| r.converged).count();
        info!(
            "All timesteps: {}/{} ({}%) chanblocks converged",
            total_converged_count,
            num_chanblocks,
            ((total_converged_count as f64 / num_chanblocks as f64) * 100.0).round()
        );
        Array2::from_shape_vec((num_timeblocks, num_chanblocks), cal_results).unwrap()
    } else {
        // Calibrate all timesteps together to get a good initial guess at what
        // the solutions for each timeblock should be.
        let pb = make_calibration_progress_bar(
            num_chanblocks,
            "Calibrating all timeblocks together".to_string(),
            draw_progress_bar,
        );
        // This timeblock represents all timeblocks.
        let timeblock = timeblocks.iter().fold(
            Timeblock {
                index: 0,
                range: 0..0,
                // The timestamps don't matter for calibration, but attempt to set
                // them correctly just in case.
                start: Epoch::from_gpst_seconds(2e10),
                end: Epoch::from_gpst_seconds(0.0),
                average: Epoch::from_gpst_seconds(0.0),
            },
            |acc, tb| Timeblock {
                index: 0,
                range: acc.range.start..tb.range.end,
                start: if acc.start < tb.start {
                    tb.start
                } else {
                    acc.start
                },
                end: if acc.end < tb.end { tb.end } else { acc.end },
                average: acc.average,
            },
        );
        let cal_results = calibrate_timeblock(
            vis_data.view(),
            vis_model.view(),
            di_jones.view_mut(),
            &timeblock,
            chanblocks,
            baseline_weights,
            max_iterations,
            stop_threshold,
            min_threshold,
            pb,
        );
        let total_converged_count = cal_results.into_iter().filter(|r| r.converged).count();
        info!(
            "All timesteps for initial guesses: {}/{} ({}%) chanblocks converged",
            total_converged_count,
            num_chanblocks,
            ((total_converged_count as f64 / num_chanblocks as f64) * 100.0).round()
        );

        // Calibrate each timeblock individually.
        let mut all_cal_results = vec![];
        for (i_timeblock, timeblock) in timeblocks.iter().enumerate() {
            // Set all solutions to be that of the averaged solutions.
            di_jones.accumulate_axis_inplace(Axis(1), |&prev, curr| *curr = prev);

            let pb = make_calibration_progress_bar(
                num_chanblocks,
                format!(
                    "Calibrating timeblock {}/{}",
                    i_timeblock + 1,
                    num_timeblocks
                ),
                draw_progress_bar,
            );
            let mut cal_results = calibrate_timeblock(
                vis_data.view(),
                vis_model.view(),
                di_jones.view_mut(),
                timeblock,
                chanblocks,
                baseline_weights,
                max_iterations,
                stop_threshold,
                min_threshold,
                pb,
            );
            let total_converged_count = cal_results.iter().filter(|r| r.converged).count();
            info!(
                "Timeblock {}: {}/{} ({}%) chanblocks converged",
                i_timeblock + 1,
                total_converged_count,
                num_chanblocks,
                (total_converged_count as f64 / num_chanblocks as f64 * 100.0).round()
            );
            all_cal_results.append(&mut cal_results);
        }
        Array2::from_shape_vec((num_timeblocks, num_chanblocks), all_cal_results).unwrap()
    };

    (
        IncompleteSolutions {
            di_jones,
            timeblocks,
            chanblocks,
            baseline_weights,
            max_iterations,
            stop_threshold,
            min_threshold,
        },
        cal_results,
    )
}

/// Convenience function to make a progress bar while calibrating. `draw`
/// determines if the progress bar is actually displayed.
fn make_calibration_progress_bar(
    num_chanblocks: usize,
    message: String,
    draw: bool,
) -> ProgressBar {
    ProgressBar::with_draw_target(
        num_chanblocks as _,
        if draw {
            // Use stdout, not stderr, because the messages printed by the
            // progress bar are valuable.
            ProgressDrawTarget::stdout()
        } else {
            ProgressDrawTarget::hidden()
        },
    )
    .with_style(
        ProgressStyle::default_bar()
            .template("{msg}: [{wide_bar:.blue}] {pos:3}/{len:3} ({elapsed_precise}<{eta_precise})")
            .progress_chars("=> "),
    )
    .with_position(0)
    .with_message(message)
}

/// Worker function to do calibration.
#[allow(clippy::too_many_arguments)]
fn calibrate_timeblock(
    vis_data: ArrayView3<Jones<f32>>,
    vis_model: ArrayView3<Jones<f32>>,
    mut di_jones: ArrayViewMut3<Jones<f64>>,
    timeblock: &Timeblock,
    chanblocks: &[Chanblock],
    baseline_weights: &[f64],
    max_iterations: usize,
    stop_threshold: f64,
    min_threshold: f64,
    progress_bar: ProgressBar,
) -> Vec<CalibrationResult> {
    let (_, num_unflagged_tiles, num_chanblocks) = di_jones.dim();

    let mut di_jones_rev = di_jones
        .slice_mut(s![timeblock.index, .., ..])
        .reversed_axes();

    let mut timeblock_cal_results: Vec<CalibrationResult> = chanblocks
        .par_iter()
        .zip(di_jones_rev.outer_iter_mut())
        .map(|(chanblock, di_jones)| {
            let i_chanblock = chanblock.unflagged_index as usize;
            let range = s![
                timeblock.range.clone(),
                ..,
                // We use a range because `calibrate` and `calibration_loop`
                // expect visibility arrays with potentially multiple
                // chanblocks. It may be worth enforcing only single chanblocks.
                i_chanblock..i_chanblock + 1
            ];
            let mut cal_result = calibrate(
                vis_data.slice(range),
                vis_model.slice(range),
                di_jones,
                baseline_weights,
                max_iterations,
                stop_threshold,
                min_threshold,
            );
            cal_result.chanblock = Some(chanblock.chanblock_index as usize);
            cal_result.i_chanblock = Some(chanblock.unflagged_index as usize);

            let mut status_str = format!("Chanblock {:>3}", chanblock.chanblock_index);
            if num_unflagged_tiles - cal_result.num_failed <= 4 {
                status_str.push_str(&format!(
                    ": failed    ({:>2}): Too many antenna solutions failed ({})",
                    cal_result.num_iterations, cal_result.num_failed
                ));
            } else if cal_result.max_precision > min_threshold {
                status_str.push_str(&format!(
                    ": failed    ({:>2}): {:.5e} > {:e}",
                    cal_result.num_iterations, cal_result.max_precision, min_threshold,
                ));
            } else if cal_result.max_precision > stop_threshold {
                status_str.push_str(&format!(
                    ": converged ({:>2}): {:e} > {:.5e} > {:e}",
                    cal_result.num_iterations,
                    min_threshold,
                    cal_result.max_precision,
                    stop_threshold
                ));
            } else {
                status_str.push_str(&format!(
                    ": converged ({:>2}): {:e} > {:.5e}",
                    cal_result.num_iterations, stop_threshold, cal_result.max_precision
                ));
            }
            progress_bar.inc(1);
            if progress_bar.is_hidden() {
                println!("{status_str}");
            } else {
                progress_bar.println(status_str);
            }
            cal_result
        })
        .collect();
    debug_assert_eq!(timeblock_cal_results.len(), num_chanblocks);
    let mut total_converged_count = timeblock_cal_results
        .iter()
        .filter(|result| result.converged)
        .count();

    // Attempt to calibrate any chanblocks that failed by taking solutions
    // from nearby chanblocks as starting points.
    if total_converged_count > 0 && total_converged_count != num_chanblocks {
        let mut new_converged_count = 1;

        let mut retry_iter = 0;
        while new_converged_count > 0 && total_converged_count != num_chanblocks {
            retry_iter += 1;
            progress_bar.println(format!(
                "*** Re-calibrating failed chanblocks iteration {} ***",
                retry_iter
            ));

            // Iterate over all the calibration results until we find one
            // that failed. Then find the next that succeeded. With a
            // converged solution on both sides (or either side) of the
            // failures, use a weighted average for a guess of what the
            // Jones matrices should be, then re-run MitchCal.
            let mut left = None;
            let mut pairs = vec![];
            let mut in_failures = false;
            for cal_result in timeblock_cal_results.iter() {
                match (in_failures, cal_result.converged) {
                    (false, true) => left = Some(cal_result.i_chanblock.unwrap()),
                    (false, false) => in_failures = true,
                    (true, true) => {
                        in_failures = false;
                        pairs.push((left, Some(cal_result.i_chanblock.unwrap())));
                        left = Some(cal_result.i_chanblock.unwrap());
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
                            .slice_mut(s![l + 1..num_chanblocks, ..])
                            .assign(&left_sol);
                    }
                    (None, Some(r)) => {
                        let right_sol = di_jones_rev.slice(s![r, ..]).to_owned();
                        di_jones_rev.slice_mut(s![0..r, ..]).assign(&right_sol);
                    }
                    (None, None) => unreachable!(),
                }
            }

            // Repeat calibration on failed chanblocks. Iterate over all
            // failed chanblocks again if any chanblocks managed to
            // converge.
            timeblock_cal_results
                .par_iter_mut()
                .zip(di_jones_rev.outer_iter_mut())
                .for_each(|(old_cal_result, di_jones)| {
                    if !old_cal_result.converged {
                        let chanblock = old_cal_result.chanblock.unwrap();
                        let i_chanblock = old_cal_result.i_chanblock.unwrap();
                        let range = s![timeblock.range.clone(), .., i_chanblock..i_chanblock + 1];
                        let mut new_cal_result = calibrate(
                            vis_data.slice(range),
                            vis_model.slice(range),
                            di_jones,
                            baseline_weights,
                            max_iterations,
                            stop_threshold,
                            min_threshold,
                        );
                        new_cal_result.chanblock = Some(chanblock);
                        new_cal_result.i_chanblock = Some(i_chanblock);

                        let mut status_str = format!("Chanblock {:>3}", chanblock);
                        if num_unflagged_tiles - new_cal_result.num_failed <= 4 {
                            status_str.push_str(&format!(
                                ": failed    ({:>2}): Too many antenna solutions failed ({})",
                                new_cal_result.num_iterations, new_cal_result.num_failed
                            ));
                        } else if new_cal_result.max_precision > min_threshold {
                            status_str.push_str(&format!(
                                ": failed    ({:>2}): {:.5e} > {:e}",
                                new_cal_result.num_iterations,
                                new_cal_result.max_precision,
                                min_threshold,
                            ));
                        } else if new_cal_result.max_precision > stop_threshold {
                            status_str.push_str(&format!(
                                ": converged ({:>2}): {:e} > {:.5e} > {:e}",
                                new_cal_result.num_iterations,
                                min_threshold,
                                new_cal_result.max_precision,
                                stop_threshold
                            ));
                        } else {
                            status_str.push_str(&format!(
                                ": converged ({:>2}): {:e} > {:.5e}",
                                new_cal_result.num_iterations,
                                stop_threshold,
                                new_cal_result.max_precision
                            ));
                        }
                        progress_bar.println(status_str);
                        *old_cal_result = new_cal_result;
                    }
                });
            let tmp = timeblock_cal_results
                .iter()
                .filter(|result| result.converged)
                .count();
            new_converged_count = tmp - total_converged_count;
            total_converged_count = tmp;
        }
    }
    progress_bar.abandon();

    timeblock_cal_results
}

#[derive(Debug)]
pub(super) struct CalibrationResult {
    pub(super) num_iterations: usize,
    pub(super) converged: bool,
    pub(super) max_precision: f64,
    pub(super) num_failed: usize,
    pub(super) chanblock: Option<usize>,

    /// The unflagged index of the chanblock. e.g. If there are 10 chanblocks
    /// that *could* be calibrated, but we calibrate only 2-9 (i.e. 0 and 1 are
    /// flagged), then the first chanblock index is 2, but its i_chanblock is 0.
    pub(super) i_chanblock: Option<usize>,
}

/// Calibrate the antennas of the array by comparing the observed input data
/// against our generated model. Return information on this process in a
/// [CalibrationResult].
///
/// This function is intended to be run in parallel; for that reason, no
/// parallel code is inside this function.
pub(super) fn calibrate(
    data: ArrayView3<Jones<f32>>,
    model: ArrayView3<Jones<f32>>,
    mut di_jones: ArrayViewMut1<Jones<f64>>,
    baseline_weights: &[f64],
    max_iterations: usize,
    stop_threshold: f64,
    min_threshold: f64,
) -> CalibrationResult {
    debug_assert_eq!(data.dim(), model.dim());

    let mut new_jones: Array1<Jones<f64>> = Array::zeros(di_jones.dim());
    let mut top: Array1<Jones<f64>> = Array::zeros(di_jones.dim());
    let mut bot: Array1<Jones<f64>> = Array::zeros(di_jones.dim());
    // The convergence precisions per antenna. They are stored per polarisation
    // for programming convenience, but really only we're interested in the
    // largest value in the entire array.
    let mut precisions: Array2<f64> = Array::zeros((di_jones.len(), 4));
    let mut failed: Array1<bool> = Array1::from_elem(di_jones.len(), false);

    // Shortcuts.
    let num_tiles = di_jones.len_of(Axis(0));

    let mut iteration = 0;
    while iteration < max_iterations {
        iteration += 1;
        // Re-initialise top and bot.
        top.fill(Jones::default());
        bot.fill(Jones::default());

        calibration_loop(
            data,
            model,
            baseline_weights,
            di_jones.view(),
            top.view_mut(),
            bot.view_mut(),
        );

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
                        let div = *top / bot;
                        if div.any_nan() {
                            *failed = true;
                            *di_jones = Jones::default();
                            *new_jones = Jones::default();
                        } else {
                            *new_jones = div;
                        }
                    });
            });

        // More than 4 antenna need to be present to get a good solution.
        let num_failed = failed.iter().filter(|&&f| f).count();
        if num_tiles - num_failed <= 4 {
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
                        [0.0, 0.0, 0.0, 0.0],
                        |acc, diff_jones| {
                            [
                                acc[0] + diff_jones[0].norm_sqr(),
                                acc[1] + diff_jones[1].norm_sqr(),
                                acc[2] + diff_jones[2].norm_sqr(),
                                acc[3] + diff_jones[3].norm_sqr(),
                            ]
                        },
                    );
                    let len = di_jones.len() as f64;
                    antenna_precision
                        .iter_mut()
                        .zip(jones_diff_sum.into_iter())
                        .for_each(|(a, d)| {
                            *a = d / len;
                        });

                    // di_jones = 0.5 * (di_jones + new_jones)
                    di_jones += &new_jones;
                    di_jones.mapv_inplace(|v| v * 0.5);
                });

            // Stop iterating if we have reached the stop threshold.
            if precisions.iter().all(|&v| v < stop_threshold) {
                break;
            }
        } else {
            // On odd iterations, update the DI Jones matrices with the new
            // ones.
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
        if num_tiles - num_failed <= 4 || max_precision > min_threshold {
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
        i_chanblock: None,
    }
}

/// "MitchCal".
fn calibration_loop(
    data: ArrayView3<Jones<f32>>,
    model: ArrayView3<Jones<f32>>,
    baseline_weights: &[f64],
    di_jones: ArrayView1<Jones<f64>>,
    mut top: ArrayViewMut1<Jones<f64>>,
    mut bot: ArrayViewMut1<Jones<f64>>,
) {
    let num_tiles = di_jones.len_of(Axis(0));

    // Time axis.
    data.outer_iter()
        .zip(model.outer_iter())
        .for_each(|(data_time, model_time)| {
            // Unflagged baseline axis.
            data_time
                .outer_iter()
                .zip(model_time.outer_iter())
                .zip(baseline_weights.iter())
                .enumerate()
                .for_each(|(i_baseline, ((data_bl, model_bl), &baseline_weight))| {
                    // Don't do anything if the baseline weight is 0.
                    if baseline_weight.abs() > f64::EPSILON {
                        let (tile1, tile2) =
                            cross_correlation_baseline_to_tiles(num_tiles, i_baseline);

                        // Unflagged frequency chan axis.
                        data_bl
                            .iter()
                            .zip(model_bl.iter())
                            .for_each(|(j_data, j_model)| {
                                // Copy and promote the data and model Jones
                                // matrices.
                                let j_data: Jones<f64> = Jones::from(j_data) * baseline_weight;
                                let j_model: Jones<f64> = Jones::from(j_model) * baseline_weight;

                                // Suppress boundary checks for maximum performance!
                                unsafe {
                                    let j_t1 = di_jones.uget(tile1);
                                    let j_t2 = di_jones.uget(tile2);

                                    let top_t1 = top.uget_mut(tile1);
                                    let bot_t1 = bot.uget_mut(tile1);

                                    // André's calibrate: ( D J M^H ) / ( M J^H J M^H )
                                    // J M^H
                                    let z = *j_t2 * j_model.h();
                                    // D (J M^H)
                                    *top_t1 += j_data * z;
                                    // (J M^H)^H (J M^H)
                                    *bot_t1 += z.h() * z;

                                    let top_t2 = top.uget_mut(tile2);
                                    let bot_t2 = bot.uget_mut(tile2);

                                    // André's calibrate: ( D J M^H ) / ( M J^H J M^H )
                                    // J (M^H)^H
                                    let z = *j_t1 * j_model;
                                    // D^H (J M^H)^H
                                    *top_t2 += j_data.h() * z;
                                    // (J M^H) (J M^H)
                                    *bot_t2 += z.h() * z;
                                }
                            });
                    }
                })
        });
}
