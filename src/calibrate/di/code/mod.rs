// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Code that *actually* does the calibration. This module exists mostly to keep
//! some things private so that they aren't misused.

#[cfg(test)]
pub(crate) mod tests;

use std::ops::Deref;

use birli::marlu::{VisContext, VisWritable};
use crossbeam_channel::{unbounded, Receiver, Sender};
use crossbeam_utils::{atomic::AtomicCell, thread};
use hifitime::Epoch;
use indicatif::{MultiProgress, ProgressBar, ProgressDrawTarget, ProgressStyle};
use log::{debug, info, trace};
use marlu::{
    c64,
    math::{cross_correlation_baseline_to_tiles, num_tiles_from_num_cross_correlation_baselines},
    Jones, UVW,
};
use ndarray::{iter::AxisIterMut, prelude::*};
use rayon::prelude::*;
use scopeguard::defer_on_unwind;

use crate::{
    calibrate::{
        params::CalibrateParams, solutions::CalibrationSolutions, CalibrateError, Chanblock, Fence,
        Timeblock,
    },
    model,
};
use mwa_hyperdrive_common::{
    cfg_if,
    hifitime::{self, Duration, Unit},
    indicatif,
    itertools::izip,
    log, marlu,
    marlu::UvfitsWriter,
    ndarray,
    num_traits::Zero,
    rayon,
};

pub(crate) struct CalVis {
    /// Visibilites read from input data.
    pub(crate) vis_data: Array3<Jones<f32>>,

    /// The weights on the visibilites read from input data.
    pub(crate) vis_weights: Array3<f32>,

    /// Visibilites generated from the sky-model source list.
    pub(crate) vis_model: Array3<Jones<f32>>,
}

/// For calibration, read in unflagged visibilities and generate sky-model
/// visibilities.
pub(crate) fn get_cal_vis(
    params: &CalibrateParams,
    draw_progress_bar: bool,
) -> Result<CalVis, CalibrateError> {
    // TODO: Display GPU info.
    cfg_if::cfg_if! {
        if #[cfg(feature = "cuda-single")] {
            if params.use_cpu_for_modelling {
                info!("Generating sky model visibilities on the CPU");
            } else {
                info!("Generating sky model visibilities on the GPU (single precision)");
            }
        } else if #[cfg(feature = "cuda")] {
            if params.use_cpu_for_modelling {
                info!("Generating sky model visibilities on the CPU");
            } else {
                info!("Generating sky model visibilities on the GPU (double precision)");
            }
        } else {
            info!("Generating sky model visibilities on the CPU");
        }
    }

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

    // Sky-modelling communicate channel. Used to tell the model writer when
    // visibilities have been generated and they're ready to be written.
    let (tx_model, rx_model) = unbounded();

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

    // Draw the progress bars. Not doing this means that the bars aren't
    // rendered until they've progressed.
    read_progress.tick();
    model_progress.tick();
    if let Some(pb) = &model_write_progress {
        pb.tick();
    }

    // Use a variable to track whether any threads have an issue.
    let error = AtomicCell::new(false);
    info!("Reading input data and sky modelling");
    let scoped_threads_result = thread::scope(|scope| {
        // Spawn a thread to draw the progress bars.
        scope.spawn(move |_| {
            multi_progress.join().unwrap();
        });

        // Mutable slices of the "global" arrays. These allow threads to mutate
        // the global arrays in parallel (using the Arc<Mutex<_>> pattern would
        // kill performance here).
        let vis_data_slices = vis_data.outer_iter_mut();
        let vis_model_slices = vis_model.outer_iter_mut();
        let vis_weight_slices = vis_weights.outer_iter_mut();

        // Input visibility-data reading thread.
        let data_handle = scope.spawn(|_| {
            // If a panic happens, update our atomic error.
            defer_on_unwind! { error.store(true); }

            let result = read_vis_data(
                params,
                vis_data_slices,
                vis_weight_slices,
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

        // Sky-model generation thread.
        let model_handle = scope.spawn(|_| {
            defer_on_unwind! { error.store(true); }

            let result = model_vis(
                params,
                vis_model_slices,
                tx_model,
                &error,
                model_progress,
                #[cfg(feature = "cuda")]
                params.use_cpu_for_modelling,
            );
            if result.is_err() {
                error.store(true);
            }
            result
        });

        // Model writing thread. If the user hasn't specified to write the model
        // to a file, then this thread just consumes messages from the modeller.
        let writer_handle = scope.spawn(|_| {
            defer_on_unwind! { error.store(true); }

            let result = model_write(params, fence, rx_model, &error, model_write_progress);
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

    info!("Finished reading input data and sky modelling");

    Ok(CalVis {
        vis_data,
        vis_weights,
        vis_model,
    })
}

fn read_vis_data(
    params: &CalibrateParams,
    vis_data_slices: AxisIterMut<Jones<f32>, Dim<[usize; 2]>>,
    vis_weight_slices: AxisIterMut<f32, Dim<[usize; 2]>>,
    error: &AtomicCell<bool>,
    progress_bar: ProgressBar,
) -> Result<(), CalibrateError> {
    for ((&timestep, vis_data_slice), vis_weight_slice) in params
        .timesteps
        .iter()
        .zip(vis_data_slices)
        .zip(vis_weight_slices)
    {
        params.read_crosses(vis_data_slice, vis_weight_slice, timestep)?;

        // Should we continue?
        if error.load() {
            return Ok(());
        }

        progress_bar.inc(1);
    }

    progress_bar.abandon_with_message("Finished reading input data");
    Ok(())
}

fn model_vis<'a>(
    params: &CalibrateParams,
    vis_model_slices: AxisIterMut<'a, Jones<f32>, Dim<[usize; 2]>>,
    tx_model: Sender<(ArrayViewMut2<'a, Jones<f32>>, Vec<UVW>, Epoch)>,
    error: &AtomicCell<bool>,
    progress_bar: ProgressBar,
    #[cfg(feature = "cuda")] use_cpu_for_modelling: bool,
) -> Result<(), CalibrateError> {
    let obs_context = params.get_obs_context();
    let modeller = model::new_sky_modeller(
        #[cfg(feature = "cuda")]
        use_cpu_for_modelling,
        params.beam.deref(),
        &params.source_list,
        &params.unflagged_tile_xyzs,
        &params.unflagged_fine_chan_freqs,
        &params.flagged_tiles,
        obs_context.phase_centre,
        params.array_longitude,
        params.array_latitude,
        // TODO: Allow the user to turn off precession.
        true,
    )?;

    // Iterate over all calibration timesteps and write to the model slices.
    for (&timestep, mut vis_model_slice) in params.timesteps.iter().zip(vis_model_slices) {
        // If for some reason the timestamp isn't there for this timestep, a
        // programmer stuffed up, but emit a decent error message.
        let timestamp = obs_context
            .timestamps
            .get(timestep)
            .ok_or(CalibrateError::TimestepUnavailable { timestep })?;
        match modeller.model_timestep(vis_model_slice.view_mut(), *timestamp) {
            // Send the model information to the writer.
            Ok(uvws) => match tx_model.send((vis_model_slice, uvws, *timestamp)) {
                Ok(()) => (),
                // If we can't send the message, it's because the channel has
                // been closed on the other side. That should only happen
                // because the writer has exited due to error; in that case,
                // just exit this thread.
                Err(_) => return Ok(()),
            },
            Err(e) => return Err(CalibrateError::from(e)),
        }

        // Should we continue?
        if error.load() {
            return Ok(());
        }

        progress_bar.inc(1);
    }

    progress_bar.abandon_with_message("Finished generating sky model");
    Ok(())
}

fn model_write(
    params: &CalibrateParams,
    fence: &Fence,
    rx_model: Receiver<(ArrayViewMut2<Jones<f32>>, Vec<UVW>, Epoch)>,
    error: &AtomicCell<bool>,
    progress_bar: Option<ProgressBar>,
) -> Result<(), CalibrateError> {
    // If the user wants the sky model written out, create the file here. This
    // can take a good deal of time; by creating the file in a thread, the other
    // threads can do useful work in parallel.
    if let Some(model_pb) = &params.model_file {
        let start_epoch = params.timeblocks.first().start;
        let obs_context = params.get_obs_context();
        let ant_pairs: Vec<(usize, usize)> = params.get_ant_pairs();
        let int_time: Duration = Duration::from_f64(obs_context.time_res.unwrap(), Unit::Second);

        let vis_ctx = VisContext {
            num_sel_timesteps: params.get_num_timesteps(),
            start_timestamp: start_epoch,
            int_time,
            // num_sel_chans: obs_context.fine_chan_freqs.len(),
            num_sel_chans: fence.get_total_num_chanblocks(),
            // start_freq_hz: obs_context.fine_chan_freqs[0] as f64,
            start_freq_hz: fence.first_freq,
            // freq_resolution_hz: obs_context.freq_res.unwrap(),
            freq_resolution_hz: fence.freq_res.unwrap(),
            sel_baselines: ant_pairs,
            avg_time: params.output_vis_time_average_factor,
            avg_freq: params.freq_average_factor,
            num_vis_pols: 4,
        };

        let obs_name = obs_context.obsid.map(|o| format!("{}", o));
        let array_pos = obs_context.get_array_pos()?;

        let mut model_writer = UvfitsWriter::from_marlu(
            &model_pb,
            &vis_ctx,
            Some(array_pos),
            obs_context.phase_centre,
            obs_name,
        )?;

        let weight_factor = vis_ctx.weight_factor() as f32;

        // Receiver model information from the modelling thread.
        for (vis_model_timestep, _, timestamp) in rx_model.iter() {
            let chunk_vis_ctx = VisContext {
                start_timestamp: timestamp - int_time * 0.5_f64,
                num_sel_timesteps: 1,
                ..vis_ctx.clone()
            };

            let out_shape = chunk_vis_ctx.avg_dims();
            let mut out_data = Array3::from_elem(out_shape, Jones::zero());
            let mut out_weights = Array3::from_elem(out_shape, -0.0);

            assert_eq!(vis_model_timestep.len_of(Axis(0)), out_shape.2);
            assert_eq!(
                vis_model_timestep.len_of(Axis(1)) + params.flagged_fine_chans.len(),
                out_shape.1
            );

            // pad and transpose the data, baselines then channels
            for (mut out_data, mut out_weights, in_data) in izip!(
                out_data.axis_iter_mut(Axis(1)),
                out_weights.axis_iter_mut(Axis(1)),
                vis_model_timestep.axis_iter(Axis(1)),
            ) {
                // merge frequency axis
                for ((_, out_jones, out_weight), in_jones) in izip!(
                    izip!(0.., out_data.iter_mut(), out_weights.iter_mut(),)
                        .filter(|(chan_idx, _, _)| !params.flagged_fine_chans.contains(chan_idx)),
                    in_data.iter(),
                ) {
                    *out_jones = *in_jones;
                    *out_weight = weight_factor;
                }
            }

            model_writer.write_vis_marlu(
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

            if let Some(pb) = &progress_bar {
                pb.inc(1)
            }
        }

        // If we have to, finish the writer.
        trace!("Finalising writing of model uvfits file");
        model_writer.write_uvfits_antenna_table(
            &params.unflagged_tile_names,
            &params.unflagged_tile_xyzs,
        )?;
        if let Some(pb) = progress_bar {
            pb.abandon_with_message("Finished writing sky model");
        }
    } else {
        // There's no model to write out, but we still need to handle all of the
        // incoming messages.
        for _ in rx_model.iter() {}
    };

    Ok(())
}

/// (Possibly) incomplete calibration solutions.
///
/// hyperdrive only reads in the data it needs for DI calibration; it ignores
/// any flagged tiles and channels in the input data. Consequentially, when DI
/// solutions are made from only unflagged data, these incomplete solutions need
/// to be "padded" with NaNs such that the complete calibration solutions can be
/// saved to disk or applied to an observation.
pub struct IncompleteSolutions<'a> {
    /// Direction-independent calibration solutions *for only unflagged data*.
    /// The first dimension is timeblock, the second is unflagged tile, the
    /// third is unflagged chanblock.
    pub di_jones: Array3<Jones<f64>>,

    /// The timeblocks used in calibration.
    timeblocks: &'a [Timeblock],

    /// The unflagged chanblocks used in calibration.
    chanblocks: &'a [Chanblock],
    // TODO: Capture and use more calibration information when writing
    // solutions. This will clarify how the solutions were made and aid
    // reproducibility.
    /// The baseline weights used during calibration.
    _baseline_weights: &'a [f64],

    /// The maximum allowed number of iterations during calibration.
    _max_iterations: usize,

    /// The stop threshold used during calibration.
    _stop_threshold: f64,

    /// The minimum threshold used during calibration.
    _min_threshold: f64,
}

impl<'a> IncompleteSolutions<'a> {
    /// Convert these [IncompleteSolutions] into "padded"
    /// [CalibrationSolutions].
    ///
    /// `total_num_tiles` is the total number of tiles (including flagged
    /// tiles).
    ///
    /// `tile_flags` and `flagged_chanblock_indices` are the flagged tile and
    /// chanblock indices, respectively.
    ///
    /// `obsid` is the observation ID.
    pub fn into_cal_sols(
        self,
        total_num_tiles: usize,
        flagged_tiles: &[usize],
        flagged_chanblock_indices: &[u16],
        obsid: Option<u32>,
    ) -> CalibrationSolutions {
        let (num_timeblocks, num_unflagged_tiles, num_unflagged_chanblocks) = self.di_jones.dim();
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
            obsid,
            start_timestamps: self.timeblocks.iter().map(|tb| tb.start).collect(),
            end_timestamps: self.timeblocks.iter().map(|tb| tb.end).collect(),
            average_timestamps: self.timeblocks.iter().map(|tb| tb.average).collect(),
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
pub fn calibrate_timeblocks<'a>(
    vis_data: ArrayView3<Jones<f32>>,
    vis_weights: ArrayView3<f32>,
    vis_model: ArrayView3<Jones<f32>>,
    timeblocks: &'a [Timeblock],
    chanblocks: &'a [Chanblock],
    baseline_weights: &'a [f64],
    max_iterations: usize,
    stop_threshold: f64,
    min_threshold: f64,
    draw_progress_bar: bool,
    print_convergence_messages: bool,
) -> (IncompleteSolutions<'a>, Array2<CalibrationResult>) {
    // Multiply the baseline weights against the visibility weights. Then, only
    // the visibility weights need to be multiplied against the data and model
    // visibilities.
    assert_eq!(vis_weights.len_of(Axis(1)), baseline_weights.len());
    let mut vis_weights = vis_weights.to_owned();
    vis_weights
        .axis_iter_mut(Axis(1))
        .into_par_iter()
        .zip(baseline_weights)
        .for_each(|(mut vis_weights, &baseline_weight)| {
            vis_weights.iter_mut().for_each(|vis_weight| {
                *vis_weight = (*vis_weight as f64 * baseline_weight) as f32;
            })
        });

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
            vis_weights.view(),
            vis_model.view(),
            di_jones.view_mut(),
            timeblocks.first().unwrap(),
            chanblocks,
            max_iterations,
            stop_threshold,
            min_threshold,
            pb,
            print_convergence_messages,
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
            vis_weights.view(),
            vis_model.view(),
            di_jones.view_mut(),
            &timeblock,
            chanblocks,
            max_iterations,
            stop_threshold,
            min_threshold,
            pb,
            print_convergence_messages,
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
                vis_weights.view(),
                vis_model.view(),
                di_jones.view_mut(),
                timeblock,
                chanblocks,
                max_iterations,
                stop_threshold,
                min_threshold,
                pb,
                print_convergence_messages,
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
            _baseline_weights: baseline_weights,
            _max_iterations: max_iterations,
            _stop_threshold: stop_threshold,
            _min_threshold: min_threshold,
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
    vis_weights: ArrayView3<f32>,
    vis_model: ArrayView3<Jones<f32>>,
    mut di_jones: ArrayViewMut3<Jones<f64>>,
    timeblock: &Timeblock,
    chanblocks: &[Chanblock],
    max_iterations: usize,
    stop_threshold: f64,
    min_threshold: f64,
    progress_bar: ProgressBar,
    print_convergence_messages: bool,
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
                vis_weights.slice(range),
                vis_model.slice(range),
                di_jones,
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
            if print_convergence_messages {
                if progress_bar.is_hidden() {
                    println!("{status_str}");
                } else {
                    progress_bar.println(status_str);
                }
            }
            cal_result
        })
        .collect();
    assert_eq!(timeblock_cal_results.len(), num_chanblocks);
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
                            vis_weights.slice(range),
                            vis_model.slice(range),
                            di_jones,
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
pub struct CalibrationResult {
    pub num_iterations: usize,
    pub converged: bool,
    pub max_precision: f64,
    pub num_failed: usize,
    pub chanblock: Option<usize>,

    /// The unflagged index of the chanblock. e.g. If there are 10 chanblocks
    /// that *could* be calibrated, but we calibrate only 2-9 (i.e. 0 and 1 are
    /// flagged), then the first chanblock index is 2, but its i_chanblock is 0.
    pub i_chanblock: Option<usize>,
}

/// Calibrate the antennas of the array by comparing the observed input data
/// against our generated model. Return information on this process in a
/// [CalibrationResult].
///
/// This function is intended to be run in parallel; for that reason, no
/// parallel code is inside this function.
pub(super) fn calibrate(
    data: ArrayView3<Jones<f32>>,
    weights: ArrayView3<f32>,
    model: ArrayView3<Jones<f32>>,
    mut di_jones: ArrayViewMut1<Jones<f64>>,
    max_iterations: usize,
    stop_threshold: f64,
    min_threshold: f64,
) -> CalibrationResult {
    assert_eq!(data.dim(), model.dim());

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
            weights,
            model,
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
    weights: ArrayView3<f32>,
    model: ArrayView3<Jones<f32>>,
    di_jones: ArrayView1<Jones<f64>>,
    mut top: ArrayViewMut1<Jones<f64>>,
    mut bot: ArrayViewMut1<Jones<f64>>,
) {
    let num_tiles = di_jones.len_of(Axis(0));

    // Time axis.
    data.outer_iter()
        .zip(weights.outer_iter())
        .zip(model.outer_iter())
        .for_each(|((data, weights), model)| {
            // Unflagged baseline axis.
            data.outer_iter()
                .zip(weights.outer_iter())
                .zip(model.outer_iter())
                .enumerate()
                .for_each(|(i_baseline, ((data, weights), model))| {
                    let (tile1, tile2) = cross_correlation_baseline_to_tiles(num_tiles, i_baseline);

                    // Unflagged frequency chan axis.
                    data.iter()
                        .zip(weights)
                        .zip(model)
                        // Don't do anything if the weight is flagged.
                        .filter(|((_, weight), _)| **weight > 0.0)
                        .for_each(|((j_data, weight), j_model)| {
                            // Copy and promote the data and model Jones
                            // matrices.
                            let weight = *weight as f64;
                            let j_data: Jones<f64> = Jones::from(j_data) * weight;
                            let j_model: Jones<f64> = Jones::from(j_model) * weight;

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
                })
        });
}
