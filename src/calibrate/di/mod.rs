// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Code to handle direction-independent calibration.
//!
//! This code borrows heavily from Torrance Hodgson's excellent Julia code at
//! https://github.com/torrance/MWAjl

pub(crate) mod code;

pub use code::calibrate_timeblocks;
use code::*;

use itertools::izip;
use log::{debug, info, log_enabled, trace, Level::Debug};
use marlu::{
    math::cross_correlation_baseline_to_tiles, Jones, MeasurementSetWriter,
    ObsContext as MarluObsContext, UvfitsWriter, VisContext, VisWritable,
};
use ndarray::prelude::*;
use rayon::prelude::*;

use super::{params::CalibrateParams, solutions::CalibrationSolutions, CalibrateError};
use crate::data_formats::VisOutputType;
use mwa_hyperdrive_common::{
    hifitime::Epoch, itertools, log, marlu, ndarray, num_traits::Zero, rayon,
};

/// Do all the steps required for direction-independent calibration; read the
/// input data, generate a model against it, and write the solutions out.
pub(crate) fn di_calibrate(
    params: &CalibrateParams,
) -> Result<CalibrationSolutions, CalibrateError> {
    // TODO: Fix.
    if params.freq_average_factor > 1 {
        panic!("Frequency averaging isn't working right now. Sorry!");
    }

    let CalVis {
        mut vis_data,
        vis_weights,
        vis_model,
    } = get_cal_vis(params, !params.no_progress_bars)?;
    assert_eq!(vis_weights.len_of(Axis(1)), params.baseline_weights.len());

    let obs_context = params.input_data.get_obs_context();

    // The shape of the array containing output Jones matrices.
    let num_timeblocks = params.timeblocks.len();
    let num_chanblocks = params.fences.first().chanblocks.len();
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
        &params.fences.first().chanblocks,
        &params.baseline_weights,
        params.max_iterations,
        params.stop_threshold,
        params.min_threshold,
        !params.no_progress_bars,
        true,
    );

    // The model visibilities are no longer needed.
    drop(vis_model);

    // Apply the solutions to the input data.
    trace!("Applying solutions");
    vis_data
        .outer_iter_mut()
        .into_par_iter()
        .for_each(|mut vis_data| {
            vis_data
                .outer_iter_mut()
                .enumerate()
                .for_each(|(i_baseline, mut vis_data)| {
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
                        .zip(sols_tile1.iter())
                        .zip(sols_tile2.iter())
                        .for_each(|((vis_data, sol1), sol2)| {
                            // Promote the data before demoting it again.
                            let mut d: Jones<f64> = Jones::from(*vis_data);
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
        obs_context.tile_xyzs.len(),
        &params.flagged_tiles,
        &params.fences.first().flagged_chanblock_indices,
        obs_context.obsid,
    );

    // Write out the solutions.
    if !params.output_solutions_filenames.is_empty() {
        info!("Writing solutions...");
    }
    for (_, file) in &params.output_solutions_filenames {
        // TODO: Provide a path to the metafits file. This is kinda redundant
        // because only RTS solutions need metafits, and hyperdrive *will not*
        // write RTS solutions out directly from calibration; they're only
        // written out when converting from another format.
        let metafits: Option<&str> = None;
        sols.write_solutions_from_ext(file, metafits)?;
        info!("Calibration solutions written to {}", file.display());
    }

    // Write out calibrated visibilities.
    if !params.output_vis_filenames.is_empty() {
        info!("Writing visibilities...");

        debug!("Dividing visibilities by weights");
        // Divide the visibilities by the weights (undoing the multiplication earlier).
        vis_data
            .outer_iter_mut()
            .into_par_iter()
            .zip(vis_weights.outer_iter())
            .for_each(|(mut vis_data, vis_weights)| {
                vis_data
                    .outer_iter_mut()
                    .zip(vis_weights.outer_iter())
                    .zip(params.baseline_weights.iter())
                    .for_each(|((mut vis_data, vis_weights), &baseline_weight)| {
                        vis_data.iter_mut().zip(vis_weights.iter()).for_each(
                            |(vis_data, &vis_weight)| {
                                let weight = f64::from(vis_weight) * baseline_weight;
                                *vis_data =
                                    Jones::<f32>::from(Jones::<f64>::from(*vis_data) / weight);
                            },
                        );
                    });
            });

        // TODO(dev): support and test time averaging for calibrated vis
        if params.output_vis_time_average_factor > 1 {
            panic!("time averaging for calibrated vis not supported");
        }

        let ant_pairs: Vec<(usize, usize)> = params.get_ant_pairs();
        let int_time = obs_context.guess_time_res();

        let start_timestamp = obs_context.timestamps[params.timesteps[0]];

        // XXX(Dev): VisContext does not support sparse timesteps, but in this case it doesn't matter
        let vis_ctx = VisContext {
            num_sel_timesteps: params.timesteps.len(),
            start_timestamp,
            int_time,
            num_sel_chans: obs_context.fine_chan_freqs.len(),
            start_freq_hz: obs_context.fine_chan_freqs[0] as f64,
            freq_resolution_hz: obs_context.guess_freq_res(),
            sel_baselines: ant_pairs,
            avg_time: params.output_vis_time_average_factor,
            avg_freq: params.output_vis_freq_average_factor,
            num_vis_pols: 4,
        };

        let obs_name = obs_context.obsid.map(|o| format!("MWA obsid {}", o));

        // shape of entire output [time, freq, baseline]. in data is [time, baseline, freq]
        let out_shape = vis_ctx.sel_dims();

        assert_eq!(vis_weights.dim(), vis_data.dim());
        // time
        assert_eq!(vis_data.len_of(Axis(0)), out_shape.0);
        // baseline
        assert_eq!(vis_data.len_of(Axis(1)), out_shape.2);
        // freq
        assert_eq!(
            vis_data.len_of(Axis(2)) + params.flagged_fine_chans.len(),
            out_shape.1
        );

        // re-use output arrays each timestep chunk
        let out_shape_timestep = (1, out_shape.1, out_shape.2);
        let mut tmp_out_data = Array3::from_elem(out_shape_timestep, Jones::zero());
        let mut tmp_out_weights = Array3::from_elem(out_shape_timestep, -0.0);

        // create a VisWritable for each output vis filename
        let mut out_writers: Vec<(VisOutputType, Box<dyn VisWritable>)> = vec![];
        for (vis_type, file) in params.output_vis_filenames.iter() {
            match vis_type {
                VisOutputType::Uvfits => {
                    trace!(" - to uvfits {}", file.display());

                    let writer = UvfitsWriter::from_marlu(
                        &file,
                        &vis_ctx,
                        Some(params.array_position),
                        obs_context.phase_centre,
                        obs_name.clone(),
                    )?;

                    out_writers.push((VisOutputType::Uvfits, Box::new(writer)));
                }
                VisOutputType::MeasurementSet => {
                    trace!(" - to measurement set {}", file.display());
                    let writer = MeasurementSetWriter::new(
                        &file,
                        obs_context.phase_centre,
                        Some(params.array_position),
                    );

                    let sched_start_timestamp = match obs_context.obsid {
                        Some(gpst) => Epoch::from_gpst_seconds(gpst as f64),
                        None => start_timestamp,
                    };
                    let sched_duration = *obs_context.timestamps.last() - sched_start_timestamp;

                    let marlu_obs_ctx = MarluObsContext {
                        sched_start_timestamp,
                        sched_duration,
                        name: obs_name.clone(),
                        phase_centre: obs_context.phase_centre,
                        pointing_centre: obs_context.pointing_centre,
                        array_pos: params.array_position,
                        ant_positions_enh: obs_context
                            .tile_xyzs
                            .iter()
                            .map(|xyz| xyz.to_enh(params.array_position.latitude_rad))
                            .collect(),
                        ant_names: obs_context.tile_names.iter().cloned().collect(),
                        // TODO(dev): is there any value in adding this metadata via hyperdrive obs context?
                        field_name: None,
                        project_id: None,
                        observer: None,
                    };

                    writer.initialize(&vis_ctx, &marlu_obs_ctx)?;
                    out_writers.push((VisOutputType::MeasurementSet, Box::new(writer)));
                }
            };
        }

        // zip over time axis;
        for (&timestep, vis_data, vis_weights) in izip!(
            params.timesteps.iter(),
            vis_data.outer_iter(),
            vis_weights.outer_iter(),
        ) {
            let chunk_vis_ctx = VisContext {
                start_timestamp: obs_context.timestamps[timestep],
                ..vis_ctx.clone()
            };
            tmp_out_data.fill(Jones::zero());
            tmp_out_weights.fill(-0.0);

            // zip over baseline axis
            for (mut tmp_out_data, mut tmp_out_weights, vis_data, vis_weights) in izip!(
                tmp_out_data.axis_iter_mut(Axis(1)),
                tmp_out_weights.axis_iter_mut(Axis(1)),
                vis_data.axis_iter(Axis(0)),
                vis_weights.axis_iter(Axis(0))
            ) {
                // merge frequency axis
                for ((_, out_jones, out_weight), in_jones, in_weight) in izip!(
                    izip!(0.., tmp_out_data.iter_mut(), tmp_out_weights.iter_mut(),)
                        .filter(|(chan_idx, _, _)| !params.flagged_fine_chans.contains(chan_idx)),
                    vis_data.iter(),
                    vis_weights.iter()
                ) {
                    *out_jones = *in_jones;
                    *out_weight = *in_weight;
                }
            }

            for (_, writer) in out_writers.iter_mut() {
                writer.write_vis_marlu(
                    tmp_out_data.view(),
                    tmp_out_weights.view(),
                    &chunk_vis_ctx,
                    &obs_context.tile_xyzs,
                    false,
                )?;
            }
        }

        // finalize writing uvfits
        for (vis_type, writer) in out_writers.into_iter() {
            if matches!(vis_type, VisOutputType::Uvfits) {
                let uvfits_writer =
                    unsafe { Box::from_raw(Box::into_raw(writer) as *mut UvfitsWriter) };
                uvfits_writer
                    .write_uvfits_antenna_table(&obs_context.tile_names, &obs_context.tile_xyzs)?;
            }
        }
    }

    Ok(sols)
}
