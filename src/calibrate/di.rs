// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

/*!
Code to handle direction-independent calibration.

This code borrows heavily from Torrance Hodgson's excellent Julia code at
https://github.com/torrance/MWAjl
*/

use crossbeam_channel::{bounded, unbounded};
use crossbeam_utils::thread::scope;
use log::{debug, info, trace};
use ndarray::prelude::*;
use rayon::prelude::*;

use super::{predict, solutions::write_solutions, CalibrateError, CalibrateParams};
use crate::{math::cross_correlation_baseline_to_tiles, *};

/// Do all the steps required for direction-independent calibration; read the
/// input data, predict a model against it, and write the solutions out.
pub fn di_cal(params: &CalibrateParams) -> Result<(), CalibrateError> {
    // The LMN coordinates of each source component.
    let lmns = params
        .source_list
        .get_lmns_parallel(&params.input_data.get_obs_context().pointing);

    // Get the instrumental flux densities for each component at each frequency.
    // These don't change with time, so we can save a lot of computation by just
    // doing this once.
    // The calculation is currently done in serial. This shouldn't be a problem
    // unless the size of the source list is huge.
    // TODO: Don't do a transpose and make this run in parallel.
    trace!("Estimating flux densities for sky-model components at all frequencies");
    let flux_densities = {
        let mut fds = Array2::from_elem(
            (
                params.num_components,
                params.freq.unflagged_fine_chan_freqs.len(),
            ),
            // [0.0; 4],
            Jones::default(),
        );
        let components = params
            .source_list
            .iter()
            .map(|(_, src)| &src.components)
            .flatten();
        for (mut comp_axis, comp) in fds.outer_iter_mut().zip(components) {
            for (comp_fd, freq) in comp_axis
                .iter_mut()
                .zip(params.freq.unflagged_fine_chan_freqs.iter())
            {
                *comp_fd = comp.estimate_at_freq(*freq)?.into();
            }
        }
        // Flip the array axes; this makes things simpler later.
        fds.t().to_owned()
    };
    // let inst_flux_densities = {
    //     let mut fds = Array2::from_elem(
    //         (
    //             params.num_components,
    //             params.freq.unflagged_fine_chan_freqs.len(),
    //         ),
    //         InstrumentalStokes::default(),
    //     );
    //     let components = params
    //         .source_list
    //         .iter()
    //         .map(|(_, src)| &src.components)
    //         .flatten();
    //     for (mut comp_axis, comp) in fds.outer_iter_mut().zip(components) {
    //         for (comp_fd, freq) in comp_axis
    //             .iter_mut()
    //             .zip(params.freq.unflagged_fine_chan_freqs.iter())
    //         {
    //             *comp_fd = comp.estimate_at_freq(*freq)?.into();
    //         }
    //     }
    //     // Flip the array axes; this makes things simpler later.
    //     fds.t().to_owned()
    // };

    trace!("Allocating memory for input data visibilities and model visibilities");
    // TODO: Use params' timesteps.
    let timesteps = &params.input_data.get_obs_context().timestep_indices;
    let vis_shape = (
        timesteps.end - timesteps.start,
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
        vis_shape.0 * vis_shape.1 * vis_shape.2 * std::mem::size_of::<Jones<f32>>()
        // 1024 * 1024 == 1 MiB.
        / 1024 / 1024
    );

    // As most of the tiles likely have the same configuration (all the same
    // delays and amps), we can be much more efficient with computation by
    // computing over only unique tile configurations (that is, unique
    // combinations of amplitudes/delays).
    // let mut tile_configs: HashMap<u64, TileConfig> = HashMap::new();
    // for tile in params
    //     .get_obs_context()
    //     .
    //     .rf_inputs
    //     .iter()
    //     .filter(|&rf| !params.tile_flags.contains(&(rf.ant as _)))
    //     .filter(|&rf| rf.pol == mwa_hyperdrive_core::mwalib::Pol::Y)
    // {
    //     let h = TileConfig::hash(&tile.dipole_delays, &tile.dipole_gains);
    //     match tile_configs.get_mut(&h) {
    //         None => {
    //             tile_configs.insert(
    //                 h,
    //                 TileConfig::new(tile.ant, &tile.dipole_delays, &tile.dipole_gains),
    //             );
    //         }
    //         Some(c) => {
    //             c.antennas.push(tile.ant as _);
    //         }
    //     };
    // }

    // Set up our producer (IO reading and sending) thread and worker (IO
    // receiving and predicting) thread. By doing things this way, the disk and
    // CPU is fully utilised; the input data and our predicted model is
    // assembled as efficiently as possible.
    info!("Beginning reading of input data and prediction");
    // Data communication channel. The producer might send an error on this
    // channel; it's up to the worker to propagate it.
    let (sx_data, rx_data) = unbounded();
    // Error channel. This allows the worker to send an error out to the
    // main thread.
    let (sx_error, rx_error) = bounded(1);
    scope(|scope| {
        // Mutable slices of the "global" data and model arrays. These allow
        // threads to mutate the global arrays in parallel (using the
        // Arc<Mutex<_>> pattern would kill performance here).
        let vis_data_slices: Vec<_> = vis_data.outer_iter_mut().collect();
        let vis_model_slices: Vec<_> = vis_model.outer_iter_mut().collect();

        // Producer (input data reading thread).
        scope.spawn(move |_| {
            for ((timestep, vis_data_slice), vis_model_slice) in
                timesteps.clone().zip(vis_data_slices).zip(vis_model_slices)
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
                    .map(|(uvws, weights)| (timestep, uvws, vis_model_slice, weights))
                    .map_err(CalibrateError::from);
                // If we can't send the message, it's because the channel has
                // been closed on the other side. That should only happen
                // because the worker has exited due to error; in that casea,
                // just exit this thread.
                match sx_data.send(msg) {
                    Ok(_) => (),
                    Err(_) => break,
                }
                // If the result of the read was erroneous, then exit now.
                if read_failed {
                    break;
                }
            }

            // By dropping the send channel, we signal to the worker thread that
            // there is no more incoming data, and it can stop waiting.
            drop(sx_data);
            trace!("Producer finished reading input data");
        });

        // Worker (predictor thread). Only one thread receives the input data,
        // but it is processed in parallel. This is much more efficient than
        // having slices of the input data being processed serially by
        // individual threads.
        scope.spawn(move |_| {
            let obs_context = params.input_data.get_obs_context();

            // Make collections of references to each component type.
            let mut point_comps: Vec<&SourceComponent> = vec![];
            let mut gaussian_comps: Vec<&SourceComponent> = vec![];
            let mut shapelet_comps: Vec<&SourceComponent> = vec![];
            for comp in params
                .source_list
                .iter()
                .flat_map(|(_, src)| &src.components)
            {
                match comp.comp_type {
                    ComponentType::Point => point_comps.push(comp),
                    ComponentType::Gaussian { .. } => gaussian_comps.push(comp),
                    ComponentType::Shapelet { .. } => shapelet_comps.push(comp),
                }
            }

            // Iterate on the receive channel forever. This terminates when
            // there is no data in the channel _and_ the sender has been
            // dropped.
            for msg in rx_data.iter() {
                let (timestep, uvws, mut vis_model_slice, weights) = match msg {
                    Ok(msg) => msg,
                    Err(e) => {
                        sx_error.send(e).unwrap();
                        break;
                    }
                };
                trace!("Predicting timestep {}", timestep);

                // For this time, get the AzEl coordinates of the sky-model
                // components.
                let lst = obs_context.lst_from_timestep(timestep);
                let azels = params.source_list.get_azel_mwa_parallel(lst);
                let hadecs: Vec<_> = params
                    .source_list
                    .iter()
                    .flat_map(|(_, src)| &src.components)
                    .map(|comp| comp.radec.to_hadec(lst))
                    .collect();

                // TODO: Use a Jones matrix cache.
                let fds_result = predict::beam_correct_flux_densities(
                    flux_densities.view(),
                    &azels,
                    &hadecs,
                    &params.beam,
                    &obs_context.delays,
                    &[1.0; 16],
                    &params.freq.unflagged_fine_chan_freqs,
                );
                // If we encountered an error, we need to handle it on the main
                // thread (send it out).
                let fds = match fds_result {
                    Err(e) => {
                        sx_error.send(e).unwrap();
                        break;
                    }
                    // Otherwise, we can continue.
                    Ok(fds) => fds,
                };

                predict::predict_model_points(
                    vis_model_slice.view_mut(),
                    weights.view(),
                    fds.view(),
                    &point_comps,
                    &lmns,
                    &uvws,
                    &params.freq.unflagged_fine_chan_freqs,
                );
                predict::predict_model_gaussians(
                    vis_model_slice.view_mut(),
                    weights.view(),
                    fds.view(),
                    &gaussian_comps,
                    &lmns,
                    &uvws,
                    &params.freq.unflagged_fine_chan_freqs,
                );
                // Shapelets need their own special kind of UVW coordinates.
                let mut shapelet_uvws: Array2<UVW> = Array2::from_elem(
                    (shapelet_comps.len(), params.unflagged_baseline_xyz.len()),
                    UVW::default(),
                );
                shapelet_uvws
                    .outer_iter_mut()
                    .into_par_iter()
                    .zip(shapelet_comps.par_iter())
                    .for_each(|(mut array, comp)| {
                        let hadec = comp.radec.to_hadec(lst);
                        let shapelet_uvws =
                            UVW::get_baselines(&params.unflagged_baseline_xyz, &hadec);
                        array.assign(&Array1::from(shapelet_uvws));
                    });
                // To ensure that `shapelet_uvws` is being strided efficiently,
                // invert the axes here.
                let shapelet_uvws = shapelet_uvws.t().to_owned();
                predict::predict_model_shapelets(
                    vis_model_slice.view_mut(),
                    weights.view(),
                    fds.view(),
                    &shapelet_comps,
                    shapelet_uvws.view(),
                    &lmns,
                    &uvws,
                    &params.freq.unflagged_fine_chan_freqs,
                );
            }
            drop(sx_error);
        });
    })
    .unwrap();

    // If an error message comes in on this channel, propagate it.
    for err_msg in rx_error.iter() {
        return Err(err_msg);
    }

    info!("Finished reading data and predicting a model against it.");

    let timeblock_len = timesteps.end - timesteps.start;
    // TODO: Let the user determine this -- using all timesteps at once for now.
    let num_timeblocks = 1;
    let mut chanblocks = params.freq.unflagged_fine_chans.iter().collect::<Vec<_>>();
    chanblocks.sort_unstable();

    // The shape of the array containing output Jones matrices.
    let obs_context = params.input_data.get_obs_context();
    let total_num_tiles = obs_context.tile_xyz.len();
    let num_unflagged_tiles = obs_context.num_unflagged_tiles;
    let shape = (num_timeblocks, num_unflagged_tiles, chanblocks.len());
    debug!(
        "Shape of DI Jones matrices array: {:?} ({} MiB)",
        shape,
        shape.0 * shape.1 * shape.2 * std::mem::size_of::<Jones<f32>>()
        // 1024 * 1024 == 1 MiB.
        / 1024 / 1024
    );
    // The output DI Jones matrices.
    let mut di_jones = Array3::from_elem(shape, Jones::identity());
    let mut converged = Array2::from_elem(
        (
            timesteps.end - timesteps.start,
            params.freq.num_unflagged_fine_chans,
        ),
        false,
    );

    // For each timeblock, calibrate all chanblocks in parallel.
    (0..num_timeblocks)
        .into_iter()
        .zip(di_jones.outer_iter_mut())
        .for_each(|(timeblock, di_jones)| {
            info!("Calibrating timeblock {}", timeblock);
            let mut di_jones_rev = di_jones.reversed_axes();
            chanblocks
                .par_iter()
                .zip(di_jones_rev.outer_iter_mut())
                .enumerate()
                .for_each(|(chanblock_index, (&&chanblock, di_jones))| {
                    let cal_result = calibrate(
                        vis_data.slice(s![
                            timeblock * timeblock_len..(timeblock + 1) * timeblock_len,
                            ..,
                            chanblock_index..chanblock_index + 1
                        ]),
                        vis_model.slice(s![
                            timeblock * timeblock_len..(timeblock + 1) * timeblock_len,
                            ..,
                            chanblock_index..chanblock_index + 1
                        ]),
                        di_jones,
                        params.max_iterations,
                        params.stop_threshold,
                        params.min_threshold,
                    );

                    let start_str = format!("chanblock {:>3}", chanblock);
                    if num_unflagged_tiles - cal_result.num_failed <= 4 {
                        info!(
                            "{}: failed    ({:>2}): Too many antenna solutions failed ({})",
                            start_str, cal_result.num_iterations, cal_result.num_failed
                        );
                    } else if cal_result.max_precision > params.min_threshold {
                        info!(
                            "{}: failed    ({:>2}): {:.5e} > {:e}",
                            start_str,
                            cal_result.num_iterations,
                            cal_result.max_precision,
                            params.min_threshold,
                        );
                    } else if cal_result.max_precision > params.stop_threshold {
                        info!(
                            "{}: converged ({:>2}): {:e} > {:.5e} > {:e}",
                            start_str,
                            cal_result.num_iterations,
                            params.min_threshold,
                            cal_result.max_precision,
                            params.stop_threshold
                        );
                    } else {
                        info!(
                            "{}: converged ({:>2}): {:e} > {:.5e}",
                            start_str,
                            cal_result.num_iterations,
                            params.stop_threshold,
                            cal_result.max_precision
                        );
                    }
                });
        });

    // Write out the solutions.
    let num_fine_freq_chans = params.input_data.get_freq_context().fine_chan_freqs.len();
    trace!("Writing solutions...");
    write_solutions(
        &params.output_solutions_filename,
        di_jones.view(),
        num_timeblocks,
        total_num_tiles,
        num_fine_freq_chans,
        &params.tile_flags,
        &params.freq.unflagged_fine_chans,
    )?;
    info!(
        "Calibration solutions written to {}",
        &params.output_solutions_filename.display()
    );

    Ok(())
}

struct CalibrationResult {
    num_iterations: usize,
    converged: bool,
    max_precision: f32,
    num_failed: usize,
}

/// Calibrate the antennas of the array by comparing the observed input data
/// against our predicted model. Return the number of iterations this took.
///
/// This function is intended to be run in parallel; for that reason, no
/// parallel code is inside this function.
fn calibrate(
    data: ArrayView3<Jones<f32>>,
    model: ArrayView3<Jones<f32>>,
    mut di_jones: ArrayViewMut1<Jones<f32>>,
    max_iterations: usize,
    stop_threshold: f32,
    min_threshold: f32,
) -> CalibrationResult {
    let mut new_jones: Array1<Jones<f32>> = Array::from_elem(di_jones.dim(), Jones::default());
    let mut top: Array1<Jones<f32>> = Array::from_elem(di_jones.dim(), Jones::default());
    let mut bot: Array1<Jones<f32>> = Array::from_elem(di_jones.dim(), Jones::default());
    // The convergence precisions per antenna. They are stored per polarisation
    // for programming convenience, but really only we're interested in the
    // largest value in the entire array.
    let mut precisions: Array2<f32> = Array::from_elem((di_jones.len(), 4), f32::default());
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
                        *new_jones = top.div(&bot);
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
                    antenna_precision.assign(&(jones_diff_sum / di_jones.len() as f32));

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
    let max_precision: f32 = precisions
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
    }
}

fn calibration_loop(
    data: ArrayView3<Jones<f32>>,
    model: ArrayView3<Jones<f32>>,
    di_jones: ArrayView1<Jones<f32>>,
    mut top: ArrayViewMut1<Jones<f32>>,
    mut bot: ArrayViewMut1<Jones<f32>>,
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
                            // Suppress boundary checks for maximum performance!
                            unsafe {
                                let j_t1 = di_jones.uget(tile1);
                                let j_t2 = di_jones.uget(tile2);

                                // André's calibrate: ( D J M^H ) / ( M J^H J M^H )
                                let top_t1 = top.uget_mut(tile1);
                                let bot_t1 = bot.uget_mut(tile1);

                                // J M^H
                                let z = Jones::axbh(j_t2, &j_model);
                                // D (J M^H)
                                Jones::plus_axb(top_t1, &j_data, &z);
                                // (J M^H)^H (J M^H)
                                Jones::plus_ahxb(bot_t1, &z, &z);

                                // Release the mutable references on `top` and
                                // `bot` so we can make new ones.
                                drop(top_t1);
                                drop(bot_t1);

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

// #[derive(Debug)]
// pub(crate) struct TileConfig<'a> {
//     /// The tile antenna numbers that this configuration applies to.
//     pub(crate) antennas: Vec<usize>,

//     /// The delays of this configuration.
//     pub(crate) delays: &'a [u32],

//     /// The amps of this configuration.
//     pub(crate) amps: &'a [f64],
// }

// impl<'a> TileConfig<'a> {
//     /// Make a new `TileConfig`.
//     pub(crate) fn new(antenna: u32, delays: &'a [u32], amps: &'a [f64]) -> Self {
//         Self {
//             antennas: vec![antenna as _],
//             delays,
//             amps,
//         }
//     }

//     /// From tile delays and amplitudes, generate a hash. Useful to identify if
//     /// this `TileConfig` matches another.
//     pub(crate) fn hash(delays: &[u32], amps: &[f64]) -> u64 {
//         let mut hasher = DefaultHasher::new();
//         delays.hash(&mut hasher);
//         // We can't hash f64 values, so convert them to ints. Multiply by a big
//         // number to get away from integer rounding.
//         let to_int = |x: f64| (x * 1e8) as u32;
//         for &a in amps {
//             to_int(a).hash(&mut hasher);
//         }
//         hasher.finish()
//     }
// }
