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

use hifitime::{Duration, Unit};
use itertools::{izip, Itertools};
use log::{debug, info, log_enabled, trace, Level::Debug};
use marlu::{
    math::cross_correlation_baseline_to_tiles, Jones, UvfitsWriter, VisContext, VisWritable,
};
use ndarray::prelude::*;
use rayon::prelude::*;

use super::{params::CalibrateParams, solutions::CalibrationSolutions, CalibrateError};
use crate::data_formats::VisOutputType;
use mwa_hyperdrive_common::{hifitime, itertools, log, marlu, ndarray, rayon};

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
        vis_weights.view(),
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

        // TODO(dev): support and test autos
        if params.using_autos {
            panic!("not supperted yet... or are they?");
        }

        let ant_pairs: Vec<(usize, usize)> = params.get_ant_pairs();
        let int_time: Duration = Duration::from_f64(obs_context.time_res.unwrap(), Unit::Second);

        // TODO(dev): support sparse timesteps by chunking over time
        for (&past, &future) in params.timesteps.iter().tuple_windows() {
            assert!(future > past);
            assert!(future - past == 1, "assuming contiguous timesteps");
        }

        let vis_ctx = VisContext {
            num_sel_timesteps: params.timesteps.len(),
            start_timestamp: obs_context.timestamps[params.timesteps[0]],
            int_time,
            num_sel_chans: obs_context.fine_chan_freqs.len(),
            start_freq_hz: obs_context.fine_chan_freqs[0] as f64,
            freq_resolution_hz: obs_context.freq_res.unwrap(),
            sel_baselines: ant_pairs,
            avg_time: params.output_vis_time_average_factor,
            avg_freq: params.output_vis_freq_average_factor,
            num_vis_pols: 4,
        };

        // pad and transpose the data
        // TODO(dev): unify unpacking

        // out data is [time, freq, baseline], in data is [time, baseline, freq]
        let out_shape = vis_ctx.sel_dims();
        let mut out_data = Array3::zeros(out_shape);
        let mut out_weights = Array3::from_elem(out_shape, -0.0);

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

        // zip over time axis;
        for (mut out_data, mut out_weights, vis_data, vis_weights) in izip!(
            out_data.outer_iter_mut(),
            out_weights.outer_iter_mut(),
            vis_data.outer_iter(),
            vis_weights.outer_iter(),
        ) {
            // zip over baseline axis
            for (mut out_data, mut out_weights, vis_data, vis_weights) in izip!(
                out_data.axis_iter_mut(Axis(1)),
                out_weights.axis_iter_mut(Axis(1)),
                vis_data.axis_iter(Axis(0)),
                vis_weights.axis_iter(Axis(0))
            ) {
                // merge frequency axis
                for ((_, out_jones, out_weight), in_jones, in_weight) in izip!(
                    izip!(0.., out_data.iter_mut(), out_weights.iter_mut(),)
                        .filter(|(chan_idx, _, _)| !params.flagged_fine_chans.contains(chan_idx)),
                    vis_data.iter(),
                    vis_weights.iter()
                ) {
                    *out_jones = *in_jones;
                    *out_weight = *in_weight;
                }
            }
        }

        let array_pos = obs_context.get_array_pos()?;
        let obs_name = obs_context.obsid.map(|o| format!("MWA obsid {}", o));

        for (vis_type, file) in &params.output_vis_filenames {
            match vis_type {
                // TODO: Make this an obs_context method?
                VisOutputType::Uvfits => {
                    trace!("Writing to output uvfits");

                    let mut writer = UvfitsWriter::from_marlu(
                        &file,
                        &vis_ctx,
                        Some(array_pos),
                        obs_context.phase_centre,
                        obs_name.clone(),
                    )?;

                    writer.write_vis_marlu(
                        out_data.view(),
                        out_weights.view(),
                        &vis_ctx,
                        &obs_context.tile_xyzs,
                        false,
                    )?;

                    writer.write_uvfits_antenna_table(
                        &obs_context.tile_names,
                        &obs_context.tile_xyzs,
                    )?;
                } // TODO(dev): Other formats
            }
            info!("Calibrated visibilities written to {}", file.display());
        }
    }

    Ok(sols)
}
