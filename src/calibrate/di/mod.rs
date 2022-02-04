// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Code to handle direction-independent calibration.
//!
//! This code borrows heavily from Torrance Hodgson's excellent Julia code at
//! https://github.com/torrance/MWAjl

mod code;

use code::*;

use hifitime::{Duration, Epoch, TimeUnit};
use log::{debug, info, log_enabled, trace, Level::Debug};
use marlu::{
    math::cross_correlation_baseline_to_tiles, pos::xyz::xyzs_to_cross_uvws_parallel,
    precession::precess_time, Jones,
};
use ndarray::prelude::*;
use rayon::prelude::*;

use super::{params::CalibrateParams, solutions::CalibrationSolutions, CalibrateError};
use crate::data_formats::{UvfitsWriter, VisOutputType};
use mwa_hyperdrive_common::{hifitime, log, marlu, ndarray, rayon};

/// Do all the steps required for direction-independent calibration; read the
/// input data, generate a model against it, and write the solutions out.
pub(crate) fn di_calibrate(
    params: &CalibrateParams,
) -> Result<CalibrationSolutions, CalibrateError> {
    let CalVis {
        mut vis_data,
        vis_weights,
        vis_model,
    } = get_cal_vis(params, !params.no_progress_bars)?;

    let obs_context = params.input_data.get_obs_context();
    let freq_context = params.input_data.get_freq_context();

    // The shape of the array containing output Jones matrices.
    let num_timeblocks = params.timeblocks.len();
    let num_chanblocks = params.chanblocks.len();
    let total_num_tiles = obs_context.tile_xyzs.len();
    let num_unflagged_tiles = total_num_tiles - params.flagged_tiles.len();

    if log_enabled!(Debug) {
        let shape = (num_timeblocks, num_unflagged_tiles, num_chanblocks);
        debug!(
            "Shape of DI Jones matrices array: ({} timeblocks, {} tiles, {} chanblocks; {} MiB)",
            shape.0,
            shape.1,
            shape.2,
            shape.0 * shape.1 * shape.2 * std::mem::size_of::<Jones<f64>>()
        // 1024 * 1024 == 1 MiB.
        / 1024 / 1024
        );
    }

    let (sols, _) = calibrate_timeblocks(
        vis_data.view(),
        vis_model.view(),
        &params.timeblocks,
        &params.chanblocks,
        &params.baseline_weights,
        params.max_iterations,
        params.stop_threshold,
        params.min_threshold,
        true,
    );

    // The model visibilities are no longer needed.
    drop(vis_model);

    // Apply the solutions to the input data.
    trace!("Applying solutions");
    vis_data
        .outer_iter_mut()
        .into_par_iter()
        .zip(vis_weights.outer_iter())
        .for_each(|(mut vis_data, vis_weights)| {
            vis_data
                .outer_iter_mut()
                .zip(vis_weights.outer_iter())
                .enumerate()
                .for_each(|(i_baseline, (mut vis_data, vis_weights))| {
                    let (tile1, tile2) = cross_correlation_baseline_to_tiles(
                        params.get_num_unflagged_tiles(),
                        i_baseline,
                    );
                    // TODO: This assumes one timeblock
                    // TODO: This assumes #freqs == #chanblocks
                    let sols_tile1 = sols.di_jones.slice(s![0_usize, tile1, ..]);
                    let sols_tile2 = sols.di_jones.slice(s![0_usize, tile2, ..]);

                    vis_data
                        .iter_mut()
                        .zip(vis_weights.iter())
                        .zip(sols_tile1.iter())
                        .zip(sols_tile2.iter())
                        .for_each(|(((vis_data, vis_weight), sol1), sol2)| {
                            // Promote the data before demoting it again.
                            let mut d: Jones<f64> = Jones::from(*vis_data);
                            // Divide by the weight.
                            d /= *vis_weight as f64;
                            // Solutions need to be inverted, because they're
                            // currently stored as something that goes from
                            // model to data, not data to model.
                            // J1 * D * J2^H
                            d = sol1.inv() * d;
                            d *= sol2.inv().h();
                            *vis_data = Jones::from(d);
                        });
                });
        });

    // "Complete" the solutions.
    let sols = sols.into_cal_sols(
        &obs_context.tile_xyzs,
        &params.flagged_tiles,
        &params.flagged_chanblocks,
        obs_context.obsid,
    );

    // Write out the solutions.
    if !params.output_solutions_filenames.is_empty() {
        info!("Writing solutions...");
    }
    for (sol_type, file) in &params.output_solutions_filenames {
        sols.write_solutions_from_ext(file)?;
        info!("Calibration solutions written to {}", file.display());
    }

    // Write out calibrated visibilities.
    if !params.output_vis_filenames.is_empty() {
        info!("Writing visibilities...");
        // Average the visibilities up.
        let out_timesteps: Vec<Vec<(usize, usize)>> = {
            // Make a new set of timesteps from the specified timesteps. Each
            // bunch of timesteps can only be as big as the time average factor.
            let mut out_timesteps = vec![];
            let mut first = (params.timesteps[0], 0);
            let mut current_timesteps = vec![first];
            for (i, &ts) in params.timesteps[1..].iter().enumerate() {
                if ts - first.0 > params.output_vis_time_average_factor - 1 {
                    out_timesteps.push(current_timesteps.clone());
                    current_timesteps.clear();
                    first = (ts, i + 1);
                }
                current_timesteps.push((ts, i + 1));
            }
            out_timesteps.push(current_timesteps);
            out_timesteps
        };

        let out_timestamps: Vec<Epoch> = out_timesteps
            .iter()
            .map(|timesteps| {
                let average_duration = timesteps
                    .iter()
                    .fold(Duration::from_f64(0.0, TimeUnit::Second), |acc, t| {
                        acc + obs_context.timestamps[t.0].as_et_duration()
                    });
                Epoch::from_et_seconds(average_duration.in_seconds() / timesteps.len() as f64)
            })
            .collect();

        let (vis_data, vis_weights) = if params.output_vis_time_average_factor > 1 {
            trace!("Averaging visibilities in time");
            let mut out_vis_data: Array3<Jones<f32>> =
                Array3::zeros((out_timestamps.len(), vis_data.dim().1, vis_data.dim().2));
            let mut out_vis_weights: Array3<f32> =
                Array3::zeros((out_timestamps.len(), vis_data.dim().1, vis_data.dim().2));
            // `num_merged` is used to track how to average the visibilities
            // after accumulating them.
            let mut num_merged: Array2<u32> =
                Array2::from_elem((out_vis_data.dim().1, out_vis_data.dim().2), 0);

            for (i_ts_block, out_timestep_block) in out_timesteps.iter().enumerate() {
                let mut vis_new_timestep = out_vis_data.slice_mut(s![i_ts_block, .., ..]);
                let mut weights_new_timestep = out_vis_weights.slice_mut(s![i_ts_block, .., ..]);
                num_merged.fill(0);

                for (i_ts, &(_, i_data)) in out_timestep_block.iter().enumerate() {
                    let vis_to_be_merged = vis_data.slice(s![i_data, .., ..]);
                    let weights_to_be_merged = vis_weights.slice(s![i_data, .., ..]);

                    if i_ts == 0 {
                        vis_new_timestep.assign(&vis_to_be_merged);
                        weights_new_timestep.assign(&weights_to_be_merged);
                        // If the visibilities aren't NaN, then increment
                        // `num_merged`.
                        vis_new_timestep
                            .iter()
                            .zip(weights_new_timestep.iter())
                            .zip(num_merged.iter_mut())
                            .for_each(|((vis, &weight), num_merged)| {
                                *num_merged = match (vis.any_nan(), weight.abs() < f32::EPSILON) {
                                    (false, false) => 1,
                                    _ => 0,
                                };
                            })
                    } else {
                        vis_new_timestep
                            .iter_mut()
                            .zip(vis_to_be_merged.iter())
                            .zip(weights_new_timestep.iter_mut())
                            .zip(weights_to_be_merged.iter())
                            .zip(num_merged.iter_mut())
                            .for_each(
                                |(
                                    (((vis_new, &vis_merge), weight_new), &weight_merge),
                                    num_merged,
                                )| {
                                    let bad_vis =
                                        vis_merge.any_nan() || weight_merge.abs() < f32::EPSILON;
                                    if !bad_vis {
                                        // Accumulate in double precision.
                                        let (v, w): (Jones<f64>, f64) = if vis_new.any_nan() {
                                            (Jones::from(vis_merge), weight_merge as f64)
                                        } else {
                                            let mut v: Jones<f64> = Jones::from(*vis_new);
                                            v += Jones::from(vis_merge);
                                            let mut w = *weight_new as f64;
                                            w += weight_merge as f64;
                                            *num_merged += 1;
                                            (v, w)
                                        };
                                        *vis_new = Jones::from(v);
                                        *weight_new = w as f32;
                                    }
                                },
                            );
                    }
                }

                // Divide by `num_merged`.
                vis_new_timestep.iter_mut().zip(num_merged.iter()).for_each(
                    |(vis, &num_merged)| {
                        if num_merged > 0 {
                            let v: Jones<f64> = Jones::from(*vis) / num_merged as f64;
                            *vis = Jones::from(v);
                        }
                    },
                );
            }
            (out_vis_data, out_vis_weights)
        } else {
            trace!("Not averaging visibilities in time");
            (vis_data, vis_weights)
        };

        let out_chans: Vec<Vec<usize>> = {
            // Similar to that above.
            let mut out_chans = vec![];
            let mut first = 0;
            let mut current_chans = vec![0];
            for i in 1..freq_context.fine_chan_freqs.len() {
                if i - first > params.output_vis_freq_average_factor - 1 {
                    out_chans.push(current_chans.clone());
                    current_chans.clear();
                    first = i;
                }
                current_chans.push(i);
            }
            out_chans.push(current_chans);
            out_chans
        };

        let (vis_data, vis_weights) = {
            trace!("Averaging visibilities in frequency");
            let mut out_vis_data: Array3<Jones<f32>> =
                Array3::zeros((vis_data.dim().0, vis_data.dim().1, out_chans.len()));
            let mut out_vis_weights: Array3<f32> =
                Array3::zeros((vis_data.dim().0, vis_data.dim().1, out_chans.len()));

            // Time axis.
            vis_data
                .outer_iter()
                .zip(out_vis_data.outer_iter_mut())
                .zip(vis_weights.outer_iter())
                .zip(out_vis_weights.outer_iter_mut())
                .for_each(
                    |(((vis_data, mut out_vis_data), vis_weights), mut out_vis_weights)| {
                        // Baseline axis.
                        vis_data
                            .outer_iter()
                            .zip(out_vis_data.outer_iter_mut())
                            .zip(vis_weights.outer_iter())
                            .zip(out_vis_weights.outer_iter_mut())
                            .for_each(
                                |(
                                    ((vis_data, mut out_vis_data), vis_weights),
                                    mut out_vis_weights,
                                )| {
                                    // Frequency axis.
                                    let mut i_unflagged_chan = 0;
                                    out_vis_data
                                        .iter_mut()
                                        .zip(out_vis_weights.iter_mut())
                                        .zip(out_chans.iter())
                                        .for_each(
                                            |((out_vis_data, out_vis_weight), out_chan_block)| {
                                                let mut num_to_merge = 0;
                                                let mut merged_vis: Jones<f64> = Jones::default();
                                                let mut merged_weight: f64 = 0.0;
                                                for i_chan_to_be_merged in out_chan_block {
                                                    if !params
                                                        .flagged_fine_chans
                                                        .contains(i_chan_to_be_merged)
                                                    {
                                                        let (v, w) = unsafe {
                                                            (
                                                                vis_data.uget(i_unflagged_chan),
                                                                vis_weights.uget(i_unflagged_chan),
                                                            )
                                                        };
                                                        merged_vis += Jones::from(v);
                                                        merged_weight += *w as f64;
                                                        num_to_merge += 1;
                                                        i_unflagged_chan += 1;
                                                    }
                                                }
                                                if num_to_merge > 0 {
                                                    merged_vis /= num_to_merge as f64;
                                                    *out_vis_data = Jones::from(merged_vis);
                                                    *out_vis_weight = merged_weight as f32;
                                                }
                                            },
                                        );
                                },
                            );
                    },
                );

            (out_vis_data, out_vis_weights)
        };

        for (vis_type, file) in &params.output_vis_filenames {
            match vis_type {
                // TODO: Make this an obs_context method?
                VisOutputType::Uvfits => {
                    trace!("Writing to output uvfits");

                    // We don't want to use `params.freq.fine_chan_flags` because
                    // we're handling the flagging ourselves here.
                    let flagged_fine_chans = std::collections::HashSet::new();
                    let obs_name = obs_context.obsid.map(|o| format!("MWA obsid {}", o));
                    let num_fine_chans = out_chans.len();
                    let centre_freq_hz = out_chans[num_fine_chans / 2]
                        .iter()
                        .map(|&i| freq_context.fine_chan_freqs[i])
                        .sum::<f64>()
                        / params.output_vis_freq_average_factor as f64;
                    let mut writer = UvfitsWriter::new(
                        &file,
                        out_timestamps.len(),
                        params.get_num_unflagged_baselines(),
                        num_fine_chans,
                        params.using_autos,
                        out_timestamps[0],
                        freq_context
                            .freq_res
                            .map(|w| w * params.output_vis_freq_average_factor as f64),
                        centre_freq_hz,
                        num_fine_chans / 2,
                        obs_context.phase_centre,
                        obs_name.as_deref(),
                        &params.unflagged_cross_baseline_to_tile_map,
                        &flagged_fine_chans,
                    )?;

                    let num_unflagged_tiles = params.unflagged_tile_names.len();
                    for (i_timestep, &timestamp) in out_timestamps.iter().enumerate() {
                        trace!("Writing timestep {i_timestep}");

                        let cross_vis_proper = vis_data.slice(s![i_timestep, .., ..]);
                        let cross_weights_proper = vis_weights.slice(s![i_timestep, .., ..]);

                        // Write out the visibilities.
                        let precession_info = precess_time(
                            obs_context.phase_centre,
                            timestamp,
                            params.array_longitude,
                            params.array_latitude,
                        );
                        let precessed_tile_xyzs =
                            precession_info.precess_xyz_parallel(&params.unflagged_tile_xyzs);
                        let uvws = xyzs_to_cross_uvws_parallel(
                            &precessed_tile_xyzs,
                            obs_context
                                .phase_centre
                                .to_hadec(precession_info.lmst_j2000),
                        );

                        if params.using_autos {
                            writer.write_cross_and_auto_timestep_vis(
                                cross_vis_proper.view(),
                                cross_weights_proper.view(),
                                Array2::zeros((num_unflagged_tiles, num_fine_chans)).view(),
                                Array2::ones((num_unflagged_tiles, num_fine_chans)).view(),
                                &uvws,
                                timestamp,
                            )?;
                        } else {
                            writer.write_cross_timestep_vis(
                                cross_vis_proper.view(),
                                cross_weights_proper.view(),
                                &uvws,
                                timestamp,
                            )?;
                        }
                    }

                    writer.write_uvfits_antenna_table(
                        &obs_context.tile_names,
                        &obs_context.tile_xyzs,
                    )?;
                }
            }
            info!("Calibrated visibilities written to {}", file.display());
        }
    }

    Ok(sols)
}
