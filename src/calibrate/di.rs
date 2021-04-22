// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

/*!
Code to handle direction-independent calibration.

This code borrows heavily from Torrance Hodgson's excellent Julia code at
https://github.com/torrance/MWAjl
*/

use std::collections::HashMap;
use std::fs::File;
use std::io::{BufWriter, Write};

use byteorder::{ByteOrder, LittleEndian, WriteBytesExt};
use crossbeam_channel::{bounded, unbounded};
use crossbeam_utils::thread::scope;
use log::{debug, info, trace};
use mwalib::fitsio::{
    images::{ImageDescription, ImageType},
    FitsFile,
};
use ndarray::prelude::*;
use num::Complex;
use rayon::prelude::*;

use super::{predict, CalibrateError, CalibrateParams};
use crate::data_formats::*;
use crate::*;

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
            InstrumentalStokes::default(),
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
                // let fd = comp.estimate_at_freq(*freq)?;
                //             comp_fd[0] = fd.i;
                //             comp_fd[1] = fd.q;
                //             comp_fd[2] = fd.u;
                //             comp_fd[3] = fd.v;
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
        params.baseline_to_tile_map.len(),
        params.freq.unflagged_fine_chans.len(),
    );
    let mut vis_data: Array3<Vis<f32>> = Array3::from_elem(vis_shape, Vis::default());
    let mut vis_model: Array3<Vis<f32>> = Array3::from_elem(vis_shape, Vis::default());

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
            // for ((timestep, vis_data_slice), vis_model_slice) in (2..4)
            //     .into_iter()
            for ((timestep, vis_data_slice), vis_model_slice) in
                timesteps.clone().zip(vis_data_slices).zip(vis_model_slices)
            {
                let read_result = params.input_data.read(
                    vis_data_slice,
                    timestep,
                    &params.tile_to_baseline_map,
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
            // Iterate on the receive channel forever. This terminates when
            // there is no data in the channel _and_ the sender has been
            // dropped.
            for msg in rx_data.iter() {
                let (timestep, uvws, vis_model_slice, weights) = match msg {
                    Ok(msg) => msg,
                    Err(e) => {
                        sx_error.send(e).unwrap();
                        break;
                    }
                };
                trace!("Predicting timestep {}", timestep);

                // For this time, get the AzEl coordinates of the sky-model
                // components.
                // TODO: Verify LSTs are sensible. Write tests.
                let lst = obs_context.lst_from_timestep(timestep);
                let azels = params.source_list.get_azel_mwa_parallel(lst);
                let hadecs: Vec<_> = params
                    .source_list
                    .iter()
                    .map(|(_, src)| &src.components)
                    .flatten()
                    .map(|comp| comp.radec.to_hadec(lst))
                    .collect();

                // TODO: Use a Jones matrix cache.
                let fds = predict::beam_correct_flux_densities(
                    flux_densities.view(),
                    &azels,
                    &hadecs,
                    &params.beam,
                    &obs_context.delays,
                    &[1.0; 16],
                    &params.freq.unflagged_fine_chan_freqs,
                );
                // let fds = predict::beam_correct_flux_densities(
                //     flux_densities.view(),
                //     &azels,
                //     &params.beam,
                //     &obs_context.delays,
                //     &[1.0; 16],
                //     &params.freq.unflagged_fine_chan_freqs,
                // );
                let model_result = fds.map(|fds| {
                    predict::predict_model(
                        vis_model_slice,
                        weights.view(),
                        fds.view(),
                        &params.source_list,
                        &lmns,
                        &uvws,
                        &params.freq.unflagged_fine_chan_freqs,
                    )
                });
                // If we encountered an error, we need to handle it on the main
                // thread (send it out).
                if let Err(e) = model_result {
                    sx_error.send(e).unwrap();
                    break;
                }
            }
            drop(sx_error);
        });
    })
    .unwrap();

    // If an error message comes in on this channel, propagate it.
    for err_msg in rx_error.iter() {
        return Err(err_msg);
    }

    // let mut asdf = Array3::from_elem((uvwss.len(), uvwss[0].len(), 3), 0.0);
    // for (axis0, mut asdf) in uvwss.iter().zip(asdf.outer_iter_mut()) {
    //     for (axis1, mut asdf) in axis0.iter().zip(asdf.outer_iter_mut()) {
    //         asdf[[0]] = axis1.u;
    //         asdf[[1]] = axis1.v;
    //         asdf[[2]] = axis1.w;
    //     }
    // }
    // ndarray_npy::write_npy("/tmp/weights.npy", &asdf).unwrap();
    // std::process::exit(1);

    info!("Finished reading data and predicting a model against it.");
    // ndarray_npy::write_npy("/tmp/model_xx_re.npy", &vis_model.mapv(|v| v.xx.re)).unwrap();
    // ndarray_npy::write_npy("/tmp/model_xx_im.npy", &vis_model.mapv(|v| v.xx.im)).unwrap();
    // ndarray_npy::write_npy("/tmp/model_xy_re.npy", &vis_model.mapv(|v| v.xy.re)).unwrap();
    // ndarray_npy::write_npy("/tmp/model_xy_im.npy", &vis_model.mapv(|v| v.xy.im)).unwrap();
    // ndarray_npy::write_npy("/tmp/model_yx_re.npy", &vis_model.mapv(|v| v.yx.re)).unwrap();
    // ndarray_npy::write_npy("/tmp/model_yx_im.npy", &vis_model.mapv(|v| v.yx.im)).unwrap();
    // ndarray_npy::write_npy("/tmp/model_yy_re.npy", &vis_model.mapv(|v| v.yy.re)).unwrap();
    // ndarray_npy::write_npy("/tmp/model_yy_im.npy", &vis_model.mapv(|v| v.yy.im)).unwrap();
    // ndarray_npy::write_npy("/tmp/data_xx_re.npy", &vis_data.mapv(|v| v.xx.re)).unwrap();
    // ndarray_npy::write_npy("/tmp/data_xx_im.npy", &vis_data.mapv(|v| v.xx.im)).unwrap();
    // ndarray_npy::write_npy("/tmp/data_xy_re.npy", &vis_data.mapv(|v| v.xy.re)).unwrap();
    // ndarray_npy::write_npy("/tmp/data_xy_im.npy", &vis_data.mapv(|v| v.xy.im)).unwrap();
    // ndarray_npy::write_npy("/tmp/data_yx_re.npy", &vis_data.mapv(|v| v.yx.re)).unwrap();
    // ndarray_npy::write_npy("/tmp/data_yx_im.npy", &vis_data.mapv(|v| v.yx.im)).unwrap();
    // ndarray_npy::write_npy("/tmp/data_yy_re.npy", &vis_data.mapv(|v| v.yy.re)).unwrap();
    // ndarray_npy::write_npy("/tmp/data_yy_im.npy", &vis_data.mapv(|v| v.yy.im)).unwrap();
    // std::process::exit(1);

    // TODO: Split the data and model arrays by frequency and send them to
    // workers.

    let timeblock_len = timesteps.end - timesteps.start;
    let num_timeblocks = 1;

    // The shape of the array containing output Jones matrices.
    let shape = (
        // TODO: Let the user determine this -- using all timesteps at once for
        // now.
        num_timeblocks,
        params.unflagged_tiles.len(),
        params.freq.unflagged_fine_chans.len(),
    );
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

    // TODO: Implement timeblocks and chanblocks. For now, there's only one
    // timeblock which spans all available times, and chanblocks == fine chans.
    // for timeblock in 0..1 {
    // for chanblock in 0..params.freq.unflagged_fine_chans.len() {
    // let di_jones_slices: Vec<_> = (0..1)
    //     .into_iter()
    //     .zip(0..params.freq.unflagged_fine_chans.len())
    //     .into_iter()
    //     .map(|(timeblock, chanblock)| {
    //         &mut di_jones.slice_mut(s![timeblock, .., chanblock..chanblock + 1])
    //     })
    //     .collect();
    // di_jones_slices
    // .iter()
    // .enumerate()
    // .map(|(timeblock, di_jones)| {
    (0..num_timeblocks)
        .into_iter()
        .zip(di_jones.outer_iter_mut())
        .for_each(|(timeblock, mut di_jones)| {
            let di_jones_slices: Vec<_> = (0..params.freq.unflagged_fine_chans.len())
                .into_iter()
                .map(|chanblock| di_jones.slice_mut(s![.., chanblock..chanblock + 1]))
                .collect();
            (0..params.freq.unflagged_fine_chans.len())
                .into_par_iter()
                .zip(di_jones_slices.into_par_iter())
                .for_each(|(chanblock, di_jones)| {
                    trace!("timeblock {}, chanblock {}", timeblock, chanblock);
                    calibrate(
                        vis_data.slice(s![
                            timeblock * timeblock_len..(timeblock + 1) * timeblock_len,
                            ..,
                            chanblock..chanblock + 1
                        ]),
                        vis_model.slice(s![
                            timeblock * timeblock_len..(timeblock + 1) * timeblock_len,
                            ..,
                            chanblock..chanblock + 1
                        ]),
                        // di_jones.slice_mut(s![.., chanblock..chanblock + 1]),
                        di_jones,
                        &params.baseline_to_tile_map,
                    );
                });
        });
    // for timeblock in 0..1 {
    //     (0..params.freq.unflagged_fine_chans.len())
    //         .into_iter()
    //         .for_each(|chanblock| {
    //             trace!("timeblock {}, chanblock {}", timeblock, chanblock);
    //             calibrate(
    //                 vis_data.slice(s![.., .., chanblock..chanblock + 1]),
    //                 vis_model.slice(s![.., .., chanblock..chanblock + 1]),
    //                 di_jones.slice_mut(s![timeblock, .., chanblock..chanblock + 1]),
    //                 &params.baseline_to_tile_map,
    //             );
    //         });
    // }

    // Write out the solutions.
    let total_num_tiles = params.input_data.get_obs_context().tile_xyz.len();
    let num_fine_freq_chans = params.input_data.get_freq_context().fine_chan_freqs.len();

    let fits_file = std::path::PathBuf::from("hyperdrive_solutions.fits");
    if fits_file.exists() {
        std::fs::remove_file(&fits_file)?;
    }
    let mut fptr = FitsFile::create(&fits_file).open()?;
    // Four elements for each Jones matrix, and we need to double the last axis,
    // because we can't write complex numbers directly to FITS files; instead,
    // we write each real and imag float as individual floats.
    let dim = [1, total_num_tiles, num_fine_freq_chans, 4 * 2];
    let image_description = ImageDescription {
        data_type: ImageType::Float,
        dimensions: &dim,
    };
    let hdu = fptr.create_image("SOLUTIONS".to_string(), &image_description)?;

    // Fill the fits file with NaN before overwriting with our solved solutions.
    // We have to be tricky with what gets written out, because `di_jones`
    // doesn't necessarily have the same shape as the output.
    let mut fits_image_data = vec![f32::NAN; dim.iter().product()];
    let mut bin_file = BufWriter::new(File::create("hyperdrive_solutions.bin")?);
    // 8 floats, 8 bytes per float.
    let mut buf = [0; 8 * 8];
    bin_file.write_all(b"MWAOCAL")?;
    bin_file.write_u8(0)?;
    bin_file.write_i32::<LittleEndian>(0)?;
    bin_file.write_i32::<LittleEndian>(0)?;
    bin_file.write_i32::<LittleEndian>(di_jones.len_of(Axis(0)) as _)?;
    bin_file.write_i32::<LittleEndian>(total_num_tiles as _)?;
    bin_file.write_i32::<LittleEndian>(num_fine_freq_chans as _)?;
    bin_file.write_i32::<LittleEndian>(4)?;
    // TODO: Use real timestamps.
    bin_file.write_f64::<LittleEndian>(0.0)?;
    bin_file.write_f64::<LittleEndian>(1.0)?;

    for (timestep, di_jones_per_time) in di_jones.outer_iter().enumerate() {
        let mut unflagged_tile_index = 0;
        for tile in 0..total_num_tiles {
            let mut unflagged_chan_index = 0;
            for chan in 0..num_fine_freq_chans {
                if params.freq.unflagged_fine_chans.contains(&chan) {
                    let one_dim_index = timestep * dim[1] * dim[2] * dim[3]
                        + tile * dim[2] * dim[3]
                        + chan * dim[3];
                    // Invert the Jones matrices so that they can be applied as
                    // J D J^H
                    let j = di_jones_per_time[[unflagged_tile_index, unflagged_chan_index]].inv();
                    fits_image_data[one_dim_index + 0] = j[0].re;
                    fits_image_data[one_dim_index + 1] = j[0].im;
                    fits_image_data[one_dim_index + 2] = j[1].re;
                    fits_image_data[one_dim_index + 3] = j[1].im;
                    fits_image_data[one_dim_index + 4] = j[2].re;
                    fits_image_data[one_dim_index + 5] = j[2].im;
                    fits_image_data[one_dim_index + 6] = j[3].re;
                    fits_image_data[one_dim_index + 7] = j[3].im;

                    LittleEndian::write_f64_into(
                        &[
                            j[0].re as _,
                            j[0].im as _,
                            j[1].re as _,
                            j[1].im as _,
                            j[2].re as _,
                            j[2].im as _,
                            j[3].re as _,
                            j[3].im as _,
                        ],
                        &mut buf,
                    );
                    bin_file.write_all(&buf)?;

                    unflagged_chan_index += 1;
                } else {
                    LittleEndian::write_f64_into(&[f64::NAN; 8], &mut buf);
                    bin_file.write_all(&buf)?;
                }
            }
            unflagged_tile_index += 1;
        }
    }
    hdu.write_image(&mut fptr, &fits_image_data)?;

    Ok(())
}

/// Calibrate the antennas of the array by comparing the observed input data
/// against our predicted model. Return the number of iterations this took.
///
/// This function is intended to be run in parallel; for that reason, no
/// parallel code is inside this function.
fn calibrate(
    data: ArrayView3<Vis<f32>>,
    model: ArrayView3<Vis<f32>>,
    mut di_jones: ArrayViewMut2<Jones<f32>>,
    baseline_to_tile_map: &HashMap<usize, (usize, usize)>,
) -> (usize, bool) {
    let mut new_jones: Array2<Jones<f32>> = Array::from_elem(di_jones.dim(), Jones::default());
    let mut top: Array2<Jones<f32>> = Array::from_elem(di_jones.dim(), Jones::default());
    let mut bot: Array2<Jones<f32>> = Array::from_elem(di_jones.dim(), Jones::default());
    // The convergence precisions per antenna. They are stored per polarisation
    // for programming convenience, but really only we're interested in the
    // largest value in the entire array.
    let mut precisions: Array2<f32> =
        Array::from_elem((di_jones.len_of(Axis(0)), 4), f32::default());
    let mut failed: Array1<bool> = Array1::from_elem(di_jones.len_of(Axis(0)), false);

    // Shortcuts.
    let num_unflagged_tiles = di_jones.len_of(Axis(0));

    // TODO: Don't hard code!
    let tols = (1e-5, 1e-8);
    let max_iterations = 50;
    // let max_iterations = 100;

    let mut iteration = 0;
    while iteration < max_iterations {
        iteration += 1;
        // Re-initialise top and bot.
        top.fill(Jones::default());
        bot.fill(Jones::default());

        calibration_loop(
            data,
            model,
            di_jones.view(),
            baseline_to_tile_map,
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
                    .enumerate()
                    .for_each(|(freq, (((di_jones, new_jones), top), bot))| {
                        *new_jones = top.div(&bot);
                        if new_jones.iter().any(|f| f.is_nan()) {
                            dbg!(&top, &bot, &top.div(&bot), freq, iteration);
                            std::process::exit(1);
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
            // let distances = (&new_jones - &di_jones)
            //     .mapv(|v| v.norm_sqr().into_iter().sum() / 4.0)
            //     .mean_axis(Axis(0));
            // // di_jones = 0.5 * (di_jones + new_jones)
            // di_jones += &new_jones;
            // di_jones *= 0.5;

            // let mut biggest_distance: f64 = 0.0;
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
            if precisions.iter().all(|&v| v < tols.1) {
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
            di_jones.fill(Jones::from([Complex::new(f32::NAN, 0.0); 4]));
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
    // let max_distance: f64 = distances
    // .iter()
    // .zip(failed.iter())
    // .filter(|(_, &failed)| !failed)
    // .fold(0.0, |acc, (&antenna_precision, _)| {
    //     acc.max(antenna_precision)
    // });
    // dbg!(&distances, max_distance);
    // dbg!(max_distance);
    // std::process::exit(1);
    let num_failed = failed.iter().filter(|&&f| f).count();
    let converged = {
        // First, if only 4 or fewer antennas remain, mark the solution as
        // trash.
        if num_unflagged_tiles - num_failed <= 4 {
            info!("Too many antenna solutions failed ({}) after {} iterations, setting solution block as failed", num_failed, iteration);
            di_jones.fill(Jones::from([Complex::new(f32::NAN, 0.0); 4]));
            false
        }
        // Second, if we never reached the minimum threshold level, mark the entire solution as failed
        else if max_precision > tols.0 {
            info!("Solution block failed to converge after {} iterations, setting as failed for all antennas (precision = {:+e})", iteration, max_precision);
            di_jones.fill(Jones::from([Complex::new(f32::NAN, 0.0); 4]));
            false
        }
        // Third, we exceeded the minimum threshold level, but not the maximum (ie. we didn't break early)
        else if max_precision > tols.1 {
            info!("Solution block converged but did not meet {:+e} threshold after {} iterations (precision = {:+e})", tols.1, iteration, max_precision);
            true
        }
        // Finally, we exceeded the maximum threshold level and broke the iterations early
        else {
            info!(
                "Solution block converged after {} iterations (precision = {:+e})",
                iteration, max_precision
            );
            true
        }
    };

    (iteration, converged)
}

fn calibration_loop(
    data: ArrayView3<Vis<f32>>,
    model: ArrayView3<Vis<f32>>,
    di_jones: ArrayView2<Jones<f32>>,
    baseline_to_tile_map: &HashMap<usize, (usize, usize)>,
    mut top: ArrayViewMut2<Jones<f32>>,
    mut bot: ArrayViewMut2<Jones<f32>>,
) {
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
                    let (tile1, tile2) = baseline_to_tile_map[&unflagged_bl_index];

                    // Unflagged frequency chan axis.
                    data_bl.iter().zip(model_bl.iter()).enumerate().for_each(
                        |(fine_chan, (data_vis, model_vis))| {
                            // Suppress boundary checks for maximum performance!
                            unsafe {
                                let j_t1 = di_jones.uget((tile1, fine_chan));
                                let j_t2 = di_jones.uget((tile2, fine_chan));
                                let j_data: Jones<f32> = Jones::from([
                                    data_vis.xx,
                                    data_vis.xy,
                                    data_vis.yx,
                                    data_vis.yy,
                                ]);
                                let j_model: Jones<f32> = Jones::from([
                                    model_vis.xx,
                                    model_vis.xy,
                                    model_vis.yx,
                                    model_vis.yy,
                                ]);

                                // André's calibrate: ( D J M^H ) / ( M J^H J M^H )
                                {
                                    let top_t1 = top.uget_mut((tile1, fine_chan));
                                    let bot_t1 = bot.uget_mut((tile1, fine_chan));

                                    // J M^H
                                    // let z = j_t2.mul_hermitian(&j_model);
                                    // // D (J M^H)
                                    // *top_t1 += j_data.clone() * &z;
                                    // // (J M^H)^H (J M^H)
                                    // *bot_t1 += z.h() * z;

                                    // J M^H
                                    let z = Jones::axbh(j_t2, &j_model);
                                    // D (J M^H)
                                    Jones::plus_axb(top_t1, &j_data, &z);
                                    // (J M^H)^H (J M^H)
                                    Jones::plus_ahxb(bot_t1, &z, &z);

                                    // *bot_t1 += z.h() * &z;
                                    // dbg!(j_t1, j_t2, j_data, j_model, &z, top_t1, bot_t1);
                                    // std::process::exit(1);
                                }
                                {
                                    let top_t2 = top.uget_mut((tile2, fine_chan));
                                    let bot_t2 = bot.uget_mut((tile2, fine_chan));

                                    // // J (M^H)^H
                                    // let z = j_t1.clone() * j_model;
                                    // // D^H (J M^H)^H
                                    // *top_t2 += j_data.h() * &z;
                                    // // (J M^H) (J M^H)
                                    // *bot_t2 += z.h() * z;

                                    // J (M^H)^H
                                    let z = Jones::axb(j_t1, &j_model);
                                    // D^H (J M^H)^H
                                    Jones::plus_ahxb(top_t2, &j_data, &z);
                                    // (J M^H) (J M^H)
                                    Jones::plus_ahxb(bot_t2, &z, &z);
                                }
                            }
                        },
                    )
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
