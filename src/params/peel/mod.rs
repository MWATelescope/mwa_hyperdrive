// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

#[cfg(test)]
mod tests;

use std::{
    borrow::Cow,
    cmp::Ordering,
    f64::consts::TAU,
    io::Write,
    num::NonZeroUsize,
    ops::{Div, Neg, Sub},
    path::PathBuf,
    thread::{self, ScopedJoinHandle},
};

use crossbeam_channel::{bounded, unbounded, Receiver, Sender};
use crossbeam_utils::atomic::AtomicCell;
use hifitime::{Duration, Epoch};
use indexmap::IndexMap;
use indicatif::{MultiProgress, ProgressBar, ProgressDrawTarget, ProgressStyle};
use itertools::{izip, Itertools};
use log::{debug, info, trace, warn};
use marlu::{
    constants::VEL_C,
    pos::xyz::xyzs_to_cross_uvws,
    precession::{get_lmst, precess_time},
    HADec, Jones, LatLngHeight, RADec, XyzGeodetic, UVW,
};
use ndarray::{prelude::*, Zip};
use num_complex::Complex;
use num_traits::Zero;
use rayon::prelude::*;
use scopeguard::defer_on_unwind;
use serde::{Deserialize, Serialize};
use vec1::Vec1;

use crate::{
    averaging::{Spw, Timeblock},
    beam::Beam,
    context::ObsContext,
    di_calibrate::calibrate_timeblock,
    io::{
        read::VisReadError,
        write::{write_vis, VisTimestep},
    },
    math::div_ceil,
    model::{ModelDevice, ModelError, SkyModeller, SkyModellerCpu},
    srclist::SourceList,
    Chanblock, TileBaselineFlags, MODEL_DEVICE, PROGRESS_BARS,
};
#[cfg(any(feature = "cuda", feature = "hip"))]
use crate::{
    gpu::{self, gpu_kernel_call, DevicePointer, GpuError, GpuFloat},
    model::SkyModellerGpu,
};

use super::{InputVisParams, ModellingParams, OutputVisParams};

#[derive(Debug, Clone, Copy)]
pub(crate) struct IonoConsts {
    alpha: f64,
    beta: f64,
    gain: f64,
}

impl Default for IonoConsts {
    fn default() -> Self {
        IonoConsts {
            alpha: 0.0,
            beta: 0.0,
            gain: 1.0,
        }
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub(crate) struct SourceIonoConsts {
    pub(crate) alphas: Vec<f64>,
    pub(crate) betas: Vec<f64>,
    pub(crate) gains: Vec<f64>,
    pub(crate) weighted_catalogue_pos_j2000: RADec,
    // pub(crate) centroid_timestamps: Vec<Epoch>,
}

#[derive(Debug, PartialEq)] //, Serialize, Deserialize)]
pub(crate) struct BadSource {
    // timeblock: Timeblock,
    pub(crate) gpstime: f64,
    pub(crate) pass: usize,
    // pub(crate) gsttime: Epoch,
    // pub(crate) times: Vec<Epoch>,
    // source: Source,
    pub(crate) i_source: usize,
    pub(crate) source_name: String,
    // pub(crate) weighted_catalogue_pos_j2000: RADec,
    // iono_consts: IonoConsts,
    pub(crate) alpha: f64,
    pub(crate) beta: f64,
    pub(crate) gain: f64,
    // pub(crate) alphas: Vec<f64>,
    // pub(crate) betas: Vec<f64>,
    // pub(crate) gains: Vec<f64>,
    pub(crate) residuals_i: Vec<Complex<f64>>,
    pub(crate) residuals_q: Vec<Complex<f64>>,
    pub(crate) residuals_u: Vec<Complex<f64>>,
    pub(crate) residuals_v: Vec<Complex<f64>>,
}

// custom sorting implementations
impl PartialOrd for BadSource {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        match self.source_name.partial_cmp(&other.source_name) {
            // Some(Ordering::Equal) => match self.gpstime.partial_cmp(&other.gpstime) {
            Some(Ordering::Equal) => match self.pass.partial_cmp(&other.pass) {
                Some(Ordering::Less) => Some(Ordering::Greater),
                Some(Ordering::Greater) => Some(Ordering::Less),
                other => other,
            },
            //     other => other,
            // },
            other => other,
        }
    }
}

pub(crate) struct PeelParams {
    pub(crate) input_vis_params: InputVisParams,
    pub(crate) output_vis_params: Option<OutputVisParams>,
    pub(crate) iono_outputs: Vec<PathBuf>,
    pub(crate) beam: Box<dyn Beam>,
    pub(crate) source_list: SourceList,
    pub(crate) modelling_params: ModellingParams,
    pub(crate) iono_timeblocks: Vec1<Timeblock>,
    pub(crate) iono_time_average_factor: NonZeroUsize,
    pub(crate) low_res_spw: Spw,
    pub(crate) uvw_min_metres: f64,
    pub(crate) uvw_max_metres: f64,
    pub(crate) short_baseline_sigma: f64,
    pub(crate) convergence: f64,
    pub(crate) num_sources_to_iono_subtract: usize,
    pub(crate) num_passes: NonZeroUsize,
    pub(crate) num_loops: NonZeroUsize,
}

impl PeelParams {
    pub(crate) fn run(&self) -> Result<(), PeelError> {
        // Expose all the struct fields to ensure they're all used.
        let PeelParams {
            input_vis_params,
            output_vis_params,
            iono_outputs,
            beam,
            source_list,
            modelling_params: ModellingParams { apply_precession },
            iono_timeblocks,
            iono_time_average_factor,
            low_res_spw,
            uvw_min_metres,
            uvw_max_metres,
            short_baseline_sigma,
            convergence,
            num_sources_to_iono_subtract,
            num_passes,
            num_loops,
        } = self;

        let obs_context = input_vis_params.get_obs_context();
        let num_unflagged_tiles = input_vis_params.get_num_unflagged_tiles();
        let num_unflagged_cross_baselines = (num_unflagged_tiles * (num_unflagged_tiles - 1)) / 2;
        let array_position = obs_context.array_position;
        let tile_baseline_flags = &input_vis_params.tile_baseline_flags;
        let flagged_tiles = &tile_baseline_flags.flagged_tiles;

        let unflagged_tile_xyzs: Vec<XyzGeodetic> = obs_context
            .tile_xyzs
            .par_iter()
            .enumerate()
            .filter(|(tile_index, _)| !flagged_tiles.contains(tile_index))
            .map(|(_, xyz)| *xyz)
            .collect();

        let spw = &input_vis_params.spw;
        let all_fine_chan_freqs_hz =
            Vec1::try_from_vec(spw.chanblocks.iter().map(|c| c.freq).collect()).unwrap();
        let all_fine_chan_lambdas_m = all_fine_chan_freqs_hz.mapped_ref(|f| VEL_C / *f);
        let (low_res_freqs_hz, low_res_lambdas_m): (Vec<_>, Vec<_>) = low_res_spw
            .chanblocks
            .iter()
            .map(|c| {
                let f = c.freq;
                (f, VEL_C / f)
            })
            .unzip();

        assert!(all_fine_chan_lambdas_m.len() % low_res_lambdas_m.len() == 0);

        // Finding the Stokes-I-weighted `RADec` of each source.
        let source_weighted_positions = {
            let mut component_radecs = vec![];
            let mut component_stokes_is = vec![];
            let mut source_weighted_positions = Vec::with_capacity(source_list.len());
            for source in source_list.values() {
                component_radecs.clear();
                component_stokes_is.clear();
                for comp in source.components.iter() {
                    component_radecs.push(comp.radec);
                    // TODO: Do this properly.
                    component_stokes_is.push(1.0);
                }

                source_weighted_positions.push(
                    RADec::weighted_average(&component_radecs, &component_stokes_is)
                        .expect("component RAs aren't too far apart from one another"),
                );
            }
            source_weighted_positions
        };

        assert!(
            div_ceil(
                input_vis_params.timeblocks.len(),
                iono_time_average_factor.get()
            ) == iono_timeblocks.len(),
            "num_read_times {} != num_iono_times {} * iono_time_average_factor {}",
            input_vis_params.timeblocks.len(),
            iono_timeblocks.len(),
            iono_time_average_factor.get(),
        );

        let error = AtomicCell::new(false);
        let (tx_data, rx_data) = bounded(2);
        let (tx_residual, rx_residual) = bounded(2);
        let (tx_full_residual, rx_full_residual) = bounded(iono_time_average_factor.get());
        let (tx_write, rx_write) = bounded(2);
        let (tx_iono_consts, rx_iono_consts) = unbounded();

        // Progress bars. Courtesy Dev.
        let multi_progress = MultiProgress::with_draw_target(if PROGRESS_BARS.load() {
            ProgressDrawTarget::stdout()
        } else {
            ProgressDrawTarget::hidden()
        });
        let pb = ProgressBar::new(input_vis_params.timeblocks.len() as _)
            .with_style(
                ProgressStyle::default_bar()
                    .template("{msg:17}: [{wide_bar:.blue}] {pos:2}/{len:2} timeblocks ({elapsed_precise}<{eta_precise})").unwrap()
                    .progress_chars("=> "),
            )
            .with_position(0)
            .with_message("Reading data");
        let read_progress = multi_progress.add(pb);
        let pb = ProgressBar::new(input_vis_params.timeblocks.len() as _)
            .with_style(
                ProgressStyle::default_bar()
                    .template("{msg:17}: [{wide_bar:.blue}] {pos:2}/{len:2} timeblocks ({elapsed_precise}<{eta_precise})").unwrap()
                    .progress_chars("=> "),
            )
            .with_position(0)
            .with_message("Sky subtracting");
        let model_progress = multi_progress.add(pb);
        let pb = ProgressBar::new(source_list.len() as _)
            .with_style(
                ProgressStyle::default_bar()
                    .template("{msg:17}: [{wide_bar:.blue}] {pos:2}/{len:2} sources ({elapsed_precise}<{eta_precise})").unwrap()
                    .progress_chars("=> "),
            )
            .with_position(0)
            .with_message("Subtracting timeblock");
        let sub_progress = multi_progress.add(pb);
        let pb = ProgressBar::new(iono_timeblocks.len() as _)
            .with_style(
                ProgressStyle::default_bar()
                    .template("{msg:17}: [{wide_bar:.blue}] {pos:2}/{len:2} timeblocks ({elapsed_precise}<{eta_precise})").unwrap()
                    .progress_chars("=> "),
            )
            .with_position(0)
            .with_message("Peeling timeblocks");
        let overall_peel_progress = multi_progress.add(pb);
        let write_progress = if let Some(output_vis_params) = output_vis_params {
            let pb = ProgressBar::new(output_vis_params.output_timeblocks.len() as _)
                .with_style(
                    ProgressStyle::default_bar()
                        .template("{msg:17}: [{wide_bar:.blue}] {pos:2}/{len:2} timeblocks ({elapsed_precise}<{eta_precise})").unwrap()
                        .progress_chars("=> "),
                )
                .with_position(0)
                .with_message("Writing visibilities");
            Some(multi_progress.add(pb))
        } else {
            None
        };

        thread::scope(|scope| -> Result<(), PeelError> {
            // Input visibility-data reading thread.
            let read_handle: ScopedJoinHandle<Result<(), VisReadError>> = thread::Builder::new()
                .name("read".to_string())
                .spawn_scoped(scope, || {
                    // If a panic happens, update our atomic error.
                    defer_on_unwind! { error.store(true); }
                    read_progress.tick();

                    // If the result of reading data was an error, allow
                    // the other threads to see this so they can abandon
                    // their work early.
                    let result = read_thread(input_vis_params, tx_data, &error, &read_progress);
                    if result.is_err() {
                        error.store(true);
                    }
                    result?;

                    Ok(())
                })
                .expect("OS can create threads");

            let model_handle: ScopedJoinHandle<Result<(), ModelError>> = thread::Builder::new()
                .name("model".to_string())
                .spawn_scoped(scope, || {
                    defer_on_unwind! { error.store(true); }
                    model_progress.tick();

                    let result = subtract_thread(
                        &**beam,
                        source_list,
                        obs_context,
                        &unflagged_tile_xyzs,
                        tile_baseline_flags,
                        array_position,
                        input_vis_params.dut1,
                        &all_fine_chan_freqs_hz,
                        *apply_precession,
                        rx_data,
                        tx_residual,
                        &error,
                        &model_progress,
                        &sub_progress,
                    );
                    if result.is_err() {
                        error.store(true);
                    }
                    result?;

                    Ok(())
                })
                .expect("OS can create threads");

            let joiner_handle: ScopedJoinHandle<Result<(), PeelError>> = thread::Builder::new()
                .name("joiner".to_string())
                .spawn_scoped(scope, || {
                    defer_on_unwind! { error.store(true); }

                    joiner_thread(
                        iono_timeblocks,
                        spw,
                        num_unflagged_cross_baselines,
                        rx_residual,
                        tx_full_residual,
                        &error,
                    );

                    Ok(())
                })
                .expect("OS can create threads");

            let peel_handle: ScopedJoinHandle<Result<(), PeelError>> = thread::Builder::new()
                .name("peel".to_string())
                .spawn_scoped(scope, || {
                    defer_on_unwind! { error.store(true); }
                    overall_peel_progress.tick();

                    let result = peel_thread(
                        &**beam,
                        source_list,
                        &source_weighted_positions,
                        *num_sources_to_iono_subtract,
                        *num_passes,
                        *num_loops,
                        obs_context,
                        &unflagged_tile_xyzs,
                        *uvw_min_metres,
                        *uvw_max_metres,
                        *short_baseline_sigma,
                        *convergence,
                        tile_baseline_flags,
                        array_position,
                        input_vis_params.dut1,
                        &spw.chanblocks,
                        &all_fine_chan_lambdas_m,
                        &low_res_freqs_hz,
                        &low_res_lambdas_m,
                        *apply_precession,
                        output_vis_params.as_ref(),
                        rx_full_residual,
                        tx_write,
                        tx_iono_consts,
                        &error,
                        &multi_progress,
                        &overall_peel_progress,
                    );
                    if result.is_err() {
                        error.store(true);
                    }
                    result?;

                    Ok(())
                })
                .expect("OS can create threads");

            let write_handle = thread::Builder::new()
                .name("write".to_string())
                .spawn_scoped(scope, || {
                    defer_on_unwind! { error.store(true); }

                    if let Some((output_vis_params, write_progress_ref)) =
                        output_vis_params.as_ref().zip(write_progress.as_ref())
                    {
                        write_progress_ref.tick();

                        let result = write_vis(
                            &output_vis_params.output_files,
                            array_position,
                            obs_context.phase_centre,
                            obs_context.pointing_centre,
                            &obs_context.tile_xyzs,
                            &obs_context.tile_names,
                            obs_context.obsid,
                            &output_vis_params.output_timeblocks,
                            input_vis_params.time_res,
                            input_vis_params.dut1,
                            spw,
                            &tile_baseline_flags
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
                            write_progress,
                        );
                        match result {
                            Ok(m) => info!("{m}"),
                            Err(e) => {
                                drop(rx_iono_consts);
                                error.store(true);
                                return Err(e);
                            }
                        }
                    }

                    if !iono_outputs.is_empty() {
                        // Write out the iono consts. First, allocate a space
                        // for all the results. We use an IndexMap to keep the
                        // order of the sources preserved while also being able
                        // to write out a "HashMap-style" json.
                        let mut output_iono_consts: IndexMap<&str, SourceIonoConsts> = source_list
                            .iter()
                            .take(*num_sources_to_iono_subtract)
                            .zip(source_weighted_positions.iter().copied())
                            .map(|((name, _src), weighted_pos)| {
                                (
                                    name.as_str(),
                                    SourceIonoConsts {
                                        alphas: vec![],
                                        betas: vec![],
                                        gains: vec![],
                                        weighted_catalogue_pos_j2000: weighted_pos,
                                    },
                                )
                            })
                            .collect();

                        // Store the results as they are received on the
                        // channel.
                        while let Ok(incoming_iono_consts) = rx_iono_consts.recv() {
                            incoming_iono_consts
                                .into_iter()
                                .zip_eq(output_iono_consts.iter_mut())
                                .for_each(|(iono_consts, (_src_name, src_iono_consts))| {
                                    src_iono_consts.alphas.push(iono_consts.alpha);
                                    src_iono_consts.betas.push(iono_consts.beta);
                                    src_iono_consts.gains.push(iono_consts.gain);
                                });
                        }

                        // The channel has stopped sending results; write them
                        // out to a file.
                        let output_json_string =
                            serde_json::to_string_pretty(&output_iono_consts).unwrap();
                        for iono_output in iono_outputs {
                            let mut file = std::fs::File::create(iono_output)?;
                            file.write_all(output_json_string.as_bytes())?;
                        }
                    }

                    Ok(())
                })
                .expect("OS can create threads");

            read_handle.join().unwrap()?;
            model_handle.join().unwrap()?;
            joiner_handle.join().unwrap()?;
            peel_handle.join().unwrap()?;
            write_handle.join().unwrap()?;
            Ok(())
        })?;

        Ok(())
    }
}

fn get_weights_rts(tile_uvs: ArrayView2<UV>, lambdas_m: &[f64], short_sigma: f64) -> Array3<f32> {
    let (num_timesteps, num_tiles) = tile_uvs.dim();
    let num_cross_baselines = (num_tiles * (num_tiles - 1)) / 2;

    let mut weights = Array3::zeros((num_timesteps, lambdas_m.len(), num_cross_baselines));
    weights
        .outer_iter_mut()
        .into_par_iter()
        .zip_eq(tile_uvs.outer_iter())
        .for_each(|(mut weights, tile_uvs)| {
            let mut i_tile1 = 0;
            let mut i_tile2 = 0;
            let mut tile1_uv = tile_uvs[i_tile1];
            let mut tile2_uv = tile_uvs[i_tile2];
            weights.axis_iter_mut(Axis(1)).for_each(|mut weights| {
                i_tile2 += 1;
                if i_tile2 == num_tiles {
                    i_tile1 += 1;
                    i_tile2 = i_tile1 + 1;
                    tile1_uv = tile_uvs[i_tile1];
                }
                tile2_uv = tile_uvs[i_tile2];
                let uv = tile1_uv - tile2_uv;

                weights
                    .iter_mut()
                    .zip_eq(lambdas_m)
                    .for_each(|(weight, lambda_m)| {
                        let UV { u, v } = uv / *lambda_m;
                        // 1 - exp(-(u*u+v*v)/(2*sig^2))
                        let uv_sq = u * u + v * v;
                        let exp = (-uv_sq / (2.0 * short_sigma * short_sigma)).exp();
                        *weight = (1.0 - exp) as f32;
                    });
            });
        });
    weights
}

/// Like `vis_weight_average_tfb`, but for when we don't need to keep the low-res weights
/// Average "high-res" vis and weights to "low-res" vis (no low-res weights)
/// arguments are all 3D arrays with axes (time, freq, baseline).
/// assumes weights are capped to 0
// TODO (Dev): rename to vis_average_tfb
fn vis_average2(
    jones_from_tfb: ArrayView3<Jones<f32>>,
    mut jones_to_tfb: ArrayViewMut3<Jones<f32>>,
    weight_tfb: ArrayView3<f32>,
) {
    let from_dims = jones_from_tfb.dim();
    let (time_axis, freq_axis, baseline_axis) = (Axis(0), Axis(1), Axis(2));
    let avg_time = div_ceil(
        jones_from_tfb.len_of(time_axis),
        jones_to_tfb.len_of(time_axis),
    );
    let avg_freq = div_ceil(
        jones_from_tfb.len_of(freq_axis),
        jones_to_tfb.len_of(freq_axis),
    );

    assert_eq!(from_dims, weight_tfb.dim());
    let to_dims = jones_to_tfb.dim();
    assert_eq!(
        to_dims,
        (
            div_ceil(from_dims.0, avg_time),
            div_ceil(from_dims.1, avg_freq),
            from_dims.2,
        )
    );

    // iterate along time axis in chunks of avg_time
    for (jones_chunk_tfb, weight_chunk_tfb, mut jones_to_fb) in izip!(
        jones_from_tfb.axis_chunks_iter(time_axis, avg_time),
        weight_tfb.axis_chunks_iter(time_axis, avg_time),
        jones_to_tfb.outer_iter_mut()
    ) {
        for (jones_chunk_tfb, weight_chunk_tfb, mut jones_to_b) in izip!(
            jones_chunk_tfb.axis_chunks_iter(freq_axis, avg_freq),
            weight_chunk_tfb.axis_chunks_iter(freq_axis, avg_freq),
            jones_to_fb.outer_iter_mut()
        ) {
            // iterate along baseline axis
            for (jones_chunk_tf, weight_chunk_tf, jones_to) in izip!(
                jones_chunk_tfb.axis_iter(baseline_axis),
                weight_chunk_tfb.axis_iter(baseline_axis),
                jones_to_b.iter_mut()
            ) {
                let mut jones_weighted_sum = Jones::zero();
                let mut weight_sum: f64 = 0.0;
                for (&jones, &weight) in jones_chunk_tf.iter().zip_eq(weight_chunk_tf.iter()) {
                    // assumes weights are capped to 0. otherwise we would need to check weight >= 0
                    debug_assert!(weight >= 0.0, "weight was not capped to zero: {}", weight);
                    jones_weighted_sum += Jones::<f64>::from(jones) * weight as f64;
                    weight_sum += weight as f64;
                }

                if weight_sum > 0.0 {
                    *jones_to = Jones::from(jones_weighted_sum / weight_sum);
                }
            }
        }
    }
}

fn weights_average(weight_tfb: ArrayView3<f32>, mut weight_avg_tfb: ArrayViewMut3<f32>) {
    let from_dims = weight_tfb.dim();
    let (time_axis, freq_axis, baseline_axis) = (Axis(0), Axis(1), Axis(2));
    let avg_time = div_ceil(
        weight_tfb.len_of(time_axis),
        weight_avg_tfb.len_of(time_axis),
    );
    let avg_freq = div_ceil(
        weight_tfb.len_of(freq_axis),
        weight_avg_tfb.len_of(freq_axis),
    );

    let to_dims = weight_avg_tfb.dim();
    assert_eq!(
        to_dims,
        (
            div_ceil(from_dims.0, avg_time),
            div_ceil(from_dims.1, avg_freq),
            from_dims.2,
        )
    );

    // iterate along time axis in chunks of avg_time
    for (weight_chunk_tfb, mut weight_avg_fb) in izip!(
        weight_tfb.axis_chunks_iter(time_axis, avg_time),
        weight_avg_tfb.outer_iter_mut()
    ) {
        // iterate along frequency axis in chunks of avg_freq
        for (weight_chunk_tfb, mut weight_avg_b) in izip!(
            weight_chunk_tfb.axis_chunks_iter(freq_axis, avg_freq),
            weight_avg_fb.outer_iter_mut()
        ) {
            // iterate along baseline axis
            for (weight_chunk_tf, weight_avg) in izip!(
                weight_chunk_tfb.axis_iter(baseline_axis),
                weight_avg_b.iter_mut()
            ) {
                let mut weight_sum: f64 = 0.0;
                for &weight in weight_chunk_tf.iter() {
                    weight_sum += weight as f64;
                }

                *weight_avg = (weight_sum as f32).max(0.);
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
// TODO: rename vis_rotate_tfb
fn vis_rotate2(
    vis_tfb: ArrayView3<Jones<f32>>,
    mut vis_rot_tfb: ArrayViewMut3<Jones<f32>>,
    tile_ws_from: ArrayView2<W>,
    tile_ws_to: ArrayView2<W>,
    lambdas_m: &[f64],
) {
    // iterate along time axis in chunks of avg_time
    vis_tfb
        .outer_iter()
        .into_par_iter()
        .zip_eq(vis_rot_tfb.outer_iter_mut())
        .zip_eq(tile_ws_from.outer_iter())
        .zip_eq(tile_ws_to.outer_iter())
        .for_each(|(((vis_tfb, vis_rot_tfb), tile_ws_from), tile_ws_to)| {
            vis_rotate_fb(
                vis_tfb,
                vis_rot_tfb,
                tile_ws_from.as_slice().unwrap(),
                tile_ws_to.as_slice().unwrap(),
                lambdas_m,
            );
        });
}

fn vis_rotate_fb(
    vis_fb: ArrayView2<Jones<f32>>,
    mut vis_rot_fb: ArrayViewMut2<Jones<f32>>,
    tile_ws_from: &[W],
    tile_ws_to: &[W],
    // TODO(Dev): rename lambdas_m
    fine_chan_lambdas_m: &[f64],
) {
    let num_tiles = tile_ws_from.len();
    assert_eq!(num_tiles, tile_ws_to.len());
    let mut i_tile1 = 0;
    let mut i_tile2 = 0;
    let mut tile1_w_from = tile_ws_from[i_tile1];
    let mut tile2_w_from = tile_ws_from[i_tile2];
    let mut tile1_w_to = tile_ws_to[i_tile1];
    let mut tile2_w_to = tile_ws_to[i_tile2];
    // iterate along baseline axis
    vis_fb
        .axis_iter(Axis(1))
        .zip_eq(vis_rot_fb.axis_iter_mut(Axis(1)))
        .for_each(|(vis_f, mut vis_rot_f)| {
            i_tile2 += 1;
            if i_tile2 == num_tiles {
                i_tile1 += 1;
                i_tile2 = i_tile1 + 1;
                tile1_w_from = tile_ws_from[i_tile1];
                tile1_w_to = tile_ws_to[i_tile1];
            }
            tile2_w_from = tile_ws_from[i_tile2];
            tile2_w_to = tile_ws_to[i_tile2];

            let w_diff = (tile1_w_to - tile2_w_to) - (tile1_w_from - tile2_w_from);
            let arg = -TAU * w_diff;
            // iterate along frequency axis
            vis_f
                .iter()
                .zip_eq(vis_rot_f.iter_mut())
                .zip_eq(fine_chan_lambdas_m.iter())
                .for_each(|((jones, jones_rot), lambda_m)| {
                    let rotation = Complex::cis(arg / *lambda_m);
                    *jones_rot = Jones::<f32>::from(Jones::<f64>::from(*jones) * rotation);
                });
        });
}

/// Rotate the supplied visibilities (3D: time, freq, bl) according to the `λ²` constants of
/// proportionality with `exp(-2πi(αu+βv)λ²)`.
/// TODO(Dev): rename apply_iono_tfb
fn apply_iono2(
    vis_tfb: ArrayView3<Jones<f32>>,
    mut vis_iono_tfb: ArrayViewMut3<Jones<f32>>,
    tile_uvs: ArrayView2<UV>,
    iono_consts: IonoConsts,
    lambdas_m: &[f64],
) {
    // iterate along time axis
    vis_tfb
        .outer_iter()
        .zip_eq(vis_iono_tfb.outer_iter_mut())
        .zip_eq(tile_uvs.outer_iter())
        .for_each(|((vis_fb, vis_iono_fb), tile_uvs)| {
            apply_iono_fb(
                vis_fb,
                vis_iono_fb,
                tile_uvs.as_slice().unwrap(),
                iono_consts,
                lambdas_m,
            );
        });
}

/// Rotate the supplied visibilities (2d: freq, bl) according to the `λ²` constants of
/// proportionality with `exp(-2πi(αu+βv)λ²)`.
fn apply_iono_fb(
    vis_fb: ArrayView2<Jones<f32>>,
    mut vis_iono_fb: ArrayViewMut2<Jones<f32>>,
    tile_uvs: &[UV],
    iono_consts: IonoConsts,
    lambdas_m: &[f64],
) {
    let num_tiles = tile_uvs.len();

    // iterate along baseline axis
    let mut i_tile1 = 0;
    let mut i_tile2 = 0;
    vis_fb
        .axis_iter(Axis(1))
        .zip_eq(vis_iono_fb.axis_iter_mut(Axis(1)))
        .for_each(|(vis_f, mut vis_iono_f)| {
            i_tile2 += 1;
            if i_tile2 == num_tiles {
                i_tile1 += 1;
                i_tile2 = i_tile1 + 1;
            }

            let UV { u, v } = tile_uvs[i_tile1] - tile_uvs[i_tile2];
            let arg = -TAU * (u * iono_consts.alpha + v * iono_consts.beta);
            // iterate along frequency axis
            vis_f
                .iter()
                .zip_eq(vis_iono_f.iter_mut())
                .zip_eq(lambdas_m.iter())
                .for_each(|((jones, jones_iono), lambda_m)| {
                    let j = Jones::<f64>::from(*jones);
                    // The baseline UV is in units of metres, so we need
                    // to divide by λ to use it in an exponential. But
                    // we're also multiplying by λ², so just multiply by
                    // λ.
                    let rotation = Complex::cis(arg * *lambda_m) * iono_consts.gain;
                    *jones_iono = Jones::from(j * rotation);
                });
        });
}

/// unpeel model, peel iono model
/// this is useful when vis_model has already been subtraced from vis_residual
/// TODO (Dev): rename to unpeel_model
fn apply_iono3(
    vis_model: ArrayView3<Jones<f32>>,
    mut vis_residual: ArrayViewMut3<Jones<f32>>,
    tile_uvs: ArrayView2<UV>,
    iono_consts: IonoConsts,
    old_iono_consts: IonoConsts,
    lambdas_m: &[f64],
) {
    let num_tiles = tile_uvs.len_of(Axis(1));

    // iterate along time axis
    vis_model
        .outer_iter()
        .into_par_iter()
        .zip_eq(vis_residual.outer_iter_mut())
        .zip_eq(tile_uvs.outer_iter())
        .for_each(|((vis_model, mut vis_residual), tile_uvs)| {
            // Just in case the compiler can't understand how an ndarray is laid
            // out.
            assert_eq!(tile_uvs.len(), num_tiles);

            // iterate along baseline axis
            let mut i_tile1 = 0;
            let mut i_tile2 = 0;
            vis_model
                .axis_iter(Axis(1))
                .zip_eq(vis_residual.axis_iter_mut(Axis(1)))
                .for_each(|(vis_model, mut vis_residual)| {
                    i_tile2 += 1;
                    if i_tile2 == num_tiles {
                        i_tile1 += 1;
                        i_tile2 = i_tile1 + 1;
                    }

                    let UV { u, v } = tile_uvs[i_tile1] - tile_uvs[i_tile2];
                    let arg = -TAU * (u * iono_consts.alpha + v * iono_consts.beta);
                    let old_arg = -TAU * (u * old_iono_consts.alpha + v * old_iono_consts.beta);
                    // iterate along frequency axis
                    vis_model
                        .iter()
                        .zip_eq(vis_residual.iter_mut())
                        .zip_eq(lambdas_m.iter())
                        .for_each(|((vis_model, vis_residual), lambda_m)| {
                            let mut j = Jones::<f64>::from(*vis_residual);
                            let m = Jones::<f64>::from(*vis_model);
                            // The baseline UV is in units of metres, so we need
                            // to divide by λ to use it in an exponential. But
                            // we're also multiplying by λ², so just multiply by
                            // λ.
                            let old_rotation =
                                Complex::cis(old_arg * *lambda_m) * old_iono_consts.gain;
                            j += m * old_rotation;

                            let rotation = Complex::cis(arg * *lambda_m) * iono_consts.gain;
                            j -= m * rotation;
                            *vis_residual = Jones::from(j);
                        });
                });
        });
}

// the offsets as defined by the RTS code
// TODO: Assume there's only 1 timestep, because this is low res data?
fn iono_fit(
    residual: ArrayView3<Jones<f32>>,
    weights: ArrayView3<f32>,
    model: ArrayView3<Jones<f32>>,
    lambdas_m: &[f64],
    tile_uvs_low_res: ArrayView2<UV>,
) -> [f64; 4] {
    let num_tiles = tile_uvs_low_res.len_of(Axis(1));

    // a-terms used in least-squares estimator
    let (mut a_uu, mut a_uv, mut a_vv) = (0.0, 0.0, 0.0);
    // A-terms used in least-squares estimator
    #[allow(non_snake_case)]
    let (mut A_u, mut A_v) = (0.0, 0.0);
    // Excess amplitude in the visibilities (V) over the models (M)
    let (mut s_vm, mut s_mm) = (0.0, 0.0);

    // iterate over time
    residual
        .outer_iter()
        .zip_eq(weights.outer_iter())
        .zip_eq(model.outer_iter())
        .zip_eq(tile_uvs_low_res.outer_iter())
        .for_each(|(((residual, weights), model), tile_uvs_low_res)| {
            // iterate over frequency
            residual
                .outer_iter()
                .zip_eq(weights.outer_iter())
                .zip_eq(model.outer_iter())
                .zip_eq(lambdas_m.iter())
                .for_each(|(((residual, weights), model), &lambda)| {
                    let lambda_2 = lambda * lambda;

                    let mut i_tile1 = 0;
                    let mut i_tile2 = 0;
                    let mut uv_tile1 = tile_uvs_low_res[i_tile1];
                    let mut uv_tile2 = tile_uvs_low_res[i_tile2];

                    let mut a_uu_bl = 0.0;
                    let mut a_uv_bl = 0.0;
                    let mut a_vv_bl = 0.0;
                    let mut aa_u_bl = 0.0;
                    let mut aa_v_bl = 0.0;
                    let mut s_vm_bl = 0.0;
                    let mut s_mm_bl = 0.0;

                    // iterate over baseline
                    residual
                        .iter()
                        .zip_eq(weights.iter())
                        .zip_eq(model.iter())
                        .for_each(|((residual, weight), model)| {
                            i_tile2 += 1;
                            if i_tile2 == num_tiles {
                                i_tile1 += 1;
                                i_tile2 = i_tile1 + 1;
                                uv_tile1 = tile_uvs_low_res[i_tile1];
                            }

                            if *weight > 0.0 {
                                uv_tile2 = tile_uvs_low_res[i_tile2];
                                // Normally, we would divide by λ to get
                                // dimensionless UV. However, UV are only used
                                // to determine a_uu, a_uv, a_vv, which are also
                                // scaled by lambda. So... don't divide by λ.
                                let UV { u, v } = uv_tile1 - uv_tile2;

                                // Stokes I of the residual visibilities and
                                // model visibilities. It doesn't matter if the
                                // convention is to divide by 2 or not; the
                                // algorithm's result is algebraically the same.
                                let residual_i = residual[0] + residual[3];
                                let model_i = model[0] + model[3];

                                let model_i_re = model_i.re as f64;
                                let mr = model_i_re * (residual_i.im as f64 - model_i.im as f64);
                                let mm = model_i_re * model_i_re;
                                let s_vm = model_i_re * residual_i.re as f64;
                                let s_mm = mm;
                                let weight = *weight as f64;

                                // To avoid accumulating floating-point errors
                                // (and save some multiplies), multiplications
                                // with powers of lambda are done outside the
                                // loop.
                                a_uu_bl += weight * mm * u * u;
                                a_uv_bl += weight * mm * u * v;
                                a_vv_bl += weight * mm * v * v;
                                aa_u_bl += weight * mr * u;
                                aa_v_bl += weight * mr * v;
                                s_vm_bl += weight * s_vm;
                                s_mm_bl += weight * s_mm;
                            }
                        });

                    // As above, we didn't divide UV by lambda, so below we use
                    // λ² for λ⁴, and λ for λ².
                    a_uu += a_uu_bl * lambda_2;
                    a_uv += a_uv_bl * lambda_2;
                    a_vv += a_vv_bl * lambda_2;
                    A_u += aa_u_bl * -lambda;
                    A_v += aa_v_bl * -lambda;
                    s_vm += s_vm_bl;
                    s_mm += s_mm_bl;
                });
        });

    let denom = TAU * (a_uu * a_vv - a_uv * a_uv);
    let alpha = (A_u * a_vv - A_v * a_uv) / denom;
    let beta = (A_v * a_uu - A_u * a_uv) / denom;
    // #[cfg(test)]
    // {
    //     let gain = s_vm / s_mm;
    //     println!("a_uu {a_uu:6.4e} a_uv {a_uv:6.4e} a_vv {a_vv:6.4e} A_u {A_u:6.4e} A_v {A_v:6.4e} denom {denom:6.4e} s_vm {s_vm:6.4e}, s_mm {s_mm:6.4e} s_vm/s_mm {gain:6.4e}");
    // }
    [alpha, beta, s_vm, s_mm]
}

#[cfg(test)]
fn setup_ws(tile_ws: &mut [W], tile_xyzs: &[XyzGeodetic], phase_centre: HADec) {
    // assert_eq!(tile_ws.len(), tile_xyzs.len());
    let (s_ha, c_ha) = phase_centre.ha.sin_cos();
    let (s_dec, c_dec) = phase_centre.dec.sin_cos();
    tile_ws
        .iter_mut()
        .zip_eq(tile_xyzs.iter().copied())
        .for_each(|(tile_w, tile_xyz)| {
            *tile_w = W::_from_xyz(tile_xyz, s_ha, c_ha, s_dec, c_dec);
        });
}

fn setup_uvs(tile_uvs: &mut [UV], tile_xyzs: &[XyzGeodetic], phase_centre: HADec) {
    // assert_eq!(tile_uvs.len(), tile_xyzs.len());
    let (s_ha, c_ha) = phase_centre.ha.sin_cos();
    let (s_dec, c_dec) = phase_centre.dec.sin_cos();
    tile_uvs
        .iter_mut()
        .zip_eq(tile_xyzs.iter().copied())
        .for_each(|(tile_uv, tile_xyz)| {
            *tile_uv = UV::from_xyz(tile_xyz, s_ha, c_ha, s_dec, c_dec);
        });
}

#[cfg(test)]
fn model_timesteps(
    modeller: &dyn SkyModeller,
    timestamps: &[Epoch],
    mut vis_result_tfb: ArrayViewMut3<Jones<f32>>,
) -> Result<(), ModelError> {
    vis_result_tfb
        .outer_iter_mut()
        .zip(timestamps.iter())
        .try_for_each(|(mut vis_result, epoch)| {
            modeller
                .model_timestep_with(*epoch, vis_result.view_mut())
                .map(|_| ())
        })
}

// TODO (Dev): make this take a single timeblock
#[allow(clippy::too_many_arguments)]
fn peel_cpu(
    // TODO (Dev): I would name this resid_hi_obs_tfb
    mut vis_residual: ArrayViewMut3<Jones<f32>>,
    // TODO (Dev): I would name this vis_weights_tfb
    vis_weights: ArrayView3<f32>,
    timeblock: &Timeblock,
    source_list: &SourceList,
    iono_consts: &mut [IonoConsts],
    source_weighted_positions: &[RADec],
    num_passes: usize,
    num_loops: usize,
    short_baseline_sigma: f64,
    convergence: f64,
    // TODO (dev): Why do we need both this and low_res_lambdas_m? it's not even used
    _low_res_freqs_hz: &[f64],
    all_fine_chan_lambdas_m: &[f64],
    low_res_lambdas_m: &[f64],
    obs_context: &ObsContext,
    // TODO (dev): array_position is available from obs_context
    array_position: LatLngHeight,
    // TODO (dev): unflagged_tile_xyzs is available from obs_context
    unflagged_tile_xyzs: &[XyzGeodetic],
    high_res_modeller: &mut dyn SkyModeller,
    // TODO (dev): dut1 is available from obs_context
    dut1: Duration,
    no_precession: bool,
    multi_progress_bar: &MultiProgress,
) -> Result<(), PeelError> {
    // TODO: Do we allow multiple timesteps in the low-res data?

    let timestamps = &timeblock.timestamps;
    let num_timestamps_high_res = timestamps.len();
    let num_timestamps_low_res = 1;

    let num_tiles = unflagged_tile_xyzs.len();
    let num_cross_baselines = (num_tiles * (num_tiles - 1)) / 2;

    let num_freqs_high_res = all_fine_chan_lambdas_m.len();
    let num_freqs_low_res = low_res_lambdas_m.len();

    let num_sources = source_list.len();
    let num_sources_to_iono_subtract = iono_consts.len();

    // TODO: these assertions should be actual errors.
    let (time_axis, freq_axis, baseline_axis) = (Axis(0), Axis(1), Axis(2));

    assert_eq!(vis_residual.len_of(time_axis), num_timestamps_high_res);
    assert_eq!(vis_weights.len_of(time_axis), num_timestamps_high_res);

    assert_eq!(vis_residual.len_of(baseline_axis), num_cross_baselines);
    assert_eq!(vis_weights.len_of(baseline_axis), num_cross_baselines);

    assert_eq!(vis_residual.len_of(freq_axis), num_freqs_high_res);
    assert_eq!(vis_weights.len_of(freq_axis), num_freqs_high_res);

    assert_eq!(iono_consts.len(), num_sources_to_iono_subtract);
    assert!(num_sources_to_iono_subtract <= num_sources);

    let peel_progress = multi_progress_bar.add(
        ProgressBar::new(num_sources_to_iono_subtract as _)
            .with_style(
                ProgressStyle::default_bar()
                    .template("{msg:17}: [{wide_bar:.blue}] {pos:2}/{len:2} sources ({elapsed_precise}<{eta_precise})").unwrap()
                    .progress_chars("=> "),
            )
            .with_position(0)
            .with_message(format!("Peeling timeblock {}", timeblock.index + 1)),
    );
    peel_progress.tick();

    // observation phase center
    // TODO(Dev): rename tile_uvs_hi_obs
    let mut tile_uvs_high_res = Array2::<UV>::default((timestamps.len(), num_tiles));
    // TODO(Dev): rename tile_ws_hi_obs
    let mut tile_ws_from = Array2::<W>::default((timestamps.len(), num_tiles));
    // source phase center
    // TODO (Dev): rename tile_uvs_hi_src
    let mut tile_uvs_high_res_rot = tile_uvs_high_res.clone();
    // TODO (Dev): rename tile_ws_hi_src
    let mut tile_ws_to = tile_ws_from.clone();
    // TODO (Dev): rename tile_uvs_lo_src
    let mut tile_uvs_low_res = Array2::<UV>::default((num_timestamps_low_res, num_tiles));

    // Pre-compute high-res tile UVs and Ws at observation phase centre.
    for (&time, mut tile_uvs, mut tile_ws) in izip!(
        timestamps.iter(),
        tile_uvs_high_res.outer_iter_mut(),
        tile_ws_from.outer_iter_mut(),
    ) {
        let (lmst, precessed_xyzs) = if !no_precession {
            let precession_info = precess_time(
                array_position.longitude_rad,
                array_position.latitude_rad,
                obs_context.phase_centre,
                time,
                dut1,
            );
            let precessed_xyzs = precession_info.precess_xyz(unflagged_tile_xyzs);
            (precession_info.lmst_j2000, precessed_xyzs)
        } else {
            let lmst = get_lmst(array_position.longitude_rad, time, dut1);
            (lmst, unflagged_tile_xyzs.into())
        };
        let hadec_phase = obs_context.phase_centre.to_hadec(lmst);
        let (s_ha, c_ha) = hadec_phase.ha.sin_cos();
        let (s_dec, c_dec) = hadec_phase.dec.sin_cos();
        for (tile_uv, tile_w, &precessed_xyzs) in izip!(
            tile_uvs.iter_mut(),
            tile_ws.iter_mut(),
            precessed_xyzs.iter(),
        ) {
            let uvw = UVW::from_xyz_inner(precessed_xyzs, s_ha, c_ha, s_dec, c_dec);
            *tile_uv = UV { u: uvw.u, v: uvw.v };
            *tile_w = W(uvw.w);
        }
    }

    let (average_lmst, _average_latitude, average_tile_xyzs) = if no_precession {
        let average_timestamp = timeblock.median;
        let average_tile_xyzs =
            ArrayView2::from_shape((1, num_tiles), unflagged_tile_xyzs).expect("correct shape");
        (
            get_lmst(array_position.longitude_rad, average_timestamp, dut1),
            array_position.latitude_rad,
            CowArray::from(average_tile_xyzs),
        )
    } else {
        let average_timestamp = timeblock.median;
        let average_precession_info = precess_time(
            array_position.longitude_rad,
            array_position.latitude_rad,
            obs_context.phase_centre,
            average_timestamp,
            dut1,
        );
        let average_precessed_tile_xyzs = Array2::from_shape_vec(
            (1, num_tiles),
            average_precession_info.precess_xyz(unflagged_tile_xyzs),
        )
        .expect("correct shape");

        (
            average_precession_info.lmst_j2000,
            average_precession_info.array_latitude_j2000,
            CowArray::from(average_precessed_tile_xyzs),
        )
    };

    // TODO (Dev): iono_taper_weights could be supplied to peel
    // use the baseline taper from the RTS, 1-exp(-(u*u+v*v)/(2*sig^2));
    // TODO: Do we care about weights changing over time?
    let weights = {
        let mut iono_taper = get_weights_rts(
            tile_uvs_high_res.view(),
            all_fine_chan_lambdas_m,
            short_baseline_sigma,
        );
        iono_taper *= &vis_weights;
        iono_taper
    };

    // Temporary visibility array, re-used for each timestep
    // TODO (Dev): rename resid_hi_src_tfb
    let mut vis_residual_tmp = vis_residual.to_owned();
    let high_res_vis_dims = vis_residual.dim();
    // TODO (Dev): rename model_hi_obs_tfb
    let mut vis_model_high_res = Array3::default(high_res_vis_dims);
    let mut model_hi_src_tfb = Array3::default(high_res_vis_dims);
    let mut model_hi_src_iono_tfb = Array3::default(high_res_vis_dims);

    // temporary arrays for accumulation
    // TODO: Do a stocktake of arrays that are lying around!
    // TODO (Dev): rename resid_lo_src_tfb
    let mut vis_residual_low_res: Array3<Jones<f32>> = Array3::zeros((
        num_timestamps_low_res,
        num_freqs_low_res,
        num_cross_baselines,
    ));
    // let mut model_lo_obs_tfb = resid_lo_src_tfb.clone();
    // let mut model_lo_src_tfb = resid_lo_src_tfb.clone();
    // TODO(Dev): rename model_lo_src_iono_tfb
    let mut vis_model_low_res_tmp = vis_residual_low_res.clone();
    // TODO(Dev): rename weights_lo
    let mut vis_weights_low_res: Array3<f32> = Array3::zeros(vis_residual_low_res.dim());

    // The low-res weights only need to be populated once.
    weights_average(weights.view(), vis_weights_low_res.view_mut());

    for pass in 0..num_passes {
        for (((source_name, source), iono_consts), source_pos) in source_list
            .iter()
            .take(num_sources_to_iono_subtract)
            .zip_eq(iono_consts.iter_mut())
            .zip_eq(source_weighted_positions.iter().copied())
        {
            multi_progress_bar.suspend(|| {
                debug!("peel loop {pass}: {source_name} at {source_pos} (has iono {iono_consts:?})")
            });
            let start = std::time::Instant::now();
            let old_iono_consts = *iono_consts;

            high_res_modeller.update_with_a_source(source, obs_context.phase_centre)?;
            // high_res_modeller.update_with_a_source(source, source_pos)?;
            // this is only necessary for cpu modeller.
            vis_model_high_res.fill(Jones::zero());

            multi_progress_bar.suspend(|| trace!("{:?}: initialise modellers", start.elapsed()));
            // iterate along high res times:
            // - calculate high-res uvws in source phase centre
            // - model high res visibilities in source phase centre
            // - calculate low-res uvws in source phase centre
            // iterate along high res times
            for (&time, mut model_hi_obs_fb, mut tile_uvs_src, mut tile_ws_src) in izip!(
                timestamps,
                vis_model_high_res.outer_iter_mut(),
                tile_uvs_high_res_rot.outer_iter_mut(),
                tile_ws_to.outer_iter_mut(),
            ) {
                let (lmst, precessed_xyzs) = if !no_precession {
                    let precession_info = precess_time(
                        array_position.longitude_rad,
                        array_position.latitude_rad,
                        obs_context.phase_centre,
                        time,
                        dut1,
                    );
                    let precessed_xyzs = precession_info.precess_xyz(unflagged_tile_xyzs);
                    (precession_info.lmst_j2000, precessed_xyzs)
                } else {
                    let lmst = get_lmst(array_position.longitude_rad, time, dut1);
                    (lmst, unflagged_tile_xyzs.into())
                };
                let hadec_source = source_pos.to_hadec(lmst);
                let (s_ha, c_ha) = hadec_source.ha.sin_cos();
                let (s_dec, c_dec) = hadec_source.dec.sin_cos();
                for (tile_uv, tile_w, &precessed_xyz) in izip!(
                    tile_uvs_src.iter_mut(),
                    tile_ws_src.iter_mut(),
                    precessed_xyzs.iter(),
                ) {
                    let UVW { u, v, w } =
                        UVW::from_xyz_inner(precessed_xyz, s_ha, c_ha, s_dec, c_dec);
                    *tile_uv = UV { u, v };
                    *tile_w = W(w);
                }

                multi_progress_bar.suspend(|| trace!("{:?}: high res model", start.elapsed()));
                high_res_modeller.model_timestep_with(time, model_hi_obs_fb.view_mut())?;
                multi_progress_bar.suspend(|| trace!("{:?}: low-res uvws", start.elapsed()));
            }

            let hadec_source = source_pos.to_hadec(average_lmst);
            setup_uvs(
                tile_uvs_low_res.as_slice_mut().unwrap(),
                average_tile_xyzs.as_slice().unwrap(),
                hadec_source,
            );

            // rotate residuals to source phase centre

            multi_progress_bar
                .suspend(|| trace!("{:?}: high-res residual rotate", start.elapsed()));
            vis_rotate2(
                vis_residual.view(),
                vis_residual_tmp.view_mut(),
                tile_ws_from.view(),
                tile_ws_to.view(),
                all_fine_chan_lambdas_m,
            );

            multi_progress_bar.suspend(|| trace!("{:?}: high-res model rotate", start.elapsed()));
            // TODO: just model in src pc
            vis_rotate2(
                vis_model_high_res.view(),
                model_hi_src_tfb.view_mut(),
                tile_ws_from.view(),
                tile_ws_to.view(),
                all_fine_chan_lambdas_m,
            );

            multi_progress_bar.suspend(|| trace!("{:?}: high-res model iono", start.elapsed()));
            apply_iono2(
                model_hi_src_tfb.view(),
                model_hi_src_iono_tfb.view_mut(),
                tile_uvs_high_res_rot.view(),
                *iono_consts,
                all_fine_chan_lambdas_m,
            );

            // Add the high-res model to the residuals.
            multi_progress_bar.suspend(|| trace!("{:?}: add low-res model", start.elapsed()));
            Zip::from(&mut vis_residual_tmp)
                .and(&model_hi_src_iono_tfb)
                .for_each(|r, m| {
                    *r += *m;
                });

            multi_progress_bar.suspend(|| trace!("{:?}: vis_average", start.elapsed()));
            vis_average2(
                vis_residual_tmp.view(),
                vis_residual_low_res.view_mut(),
                weights.view(),
            );

            multi_progress_bar.suspend(|| trace!("{:?}: alpha/beta loop", start.elapsed()));
            // let mut gain_update = 1.0;
            let mut iteration = 0;
            while iteration != num_loops {
                iteration += 1;
                multi_progress_bar.suspend(|| debug!("iter {iteration}, consts: {iono_consts:?}"));

                // iono rotate model using existing iono consts
                apply_iono2(
                    model_hi_src_tfb.view(),
                    model_hi_src_iono_tfb.view_mut(),
                    tile_uvs_high_res_rot.view(),
                    *iono_consts,
                    all_fine_chan_lambdas_m,
                );

                vis_average2(
                    model_hi_src_iono_tfb.view(),
                    vis_model_low_res_tmp.view_mut(),
                    weights.view(),
                );

                let iono_fits = iono_fit(
                    vis_residual_low_res.view(),
                    vis_weights_low_res.view(),
                    vis_model_low_res_tmp.view(),
                    low_res_lambdas_m,
                    tile_uvs_low_res.view(),
                );
                multi_progress_bar.suspend(|| trace!("iono_fits: {iono_fits:?}"));
                let da = iono_fits[0];
                let db = iono_fits[1];
                let dg = iono_fits[2] / iono_fits[3];
                iono_consts.alpha += convergence * da;
                iono_consts.beta += convergence * db;
                iono_consts.gain *= 1. + convergence * (dg - 1.);

                // if the offset is small, we've converged.
                if (da.powf(2.) + db.powf(2.) + (dg - 1.).powf(2.)).sqrt() < 1e-8 {
                    break;
                }
            }

            // multi_progress_bar.suspend(|| trace!("{:?}: high res model", start.elapsed()));
            // model_hi_obs_tfb.fill(Jones::default());
            // model_timesteps(
            //     high_res_modeller,
            //     timestamps,
            //     model_hi_obs_tfb.view_mut(),
            // )?;

            multi_progress_bar.suspend(|| trace!("{:?}: apply_iono3", start.elapsed()));
            // add the model to residual, and subtract the iono rotated model
            apply_iono3(
                vis_model_high_res.view(),
                vis_residual.view_mut(),
                // tile_uvs_high_res.view(),
                tile_uvs_high_res_rot.view(),
                *iono_consts,
                old_iono_consts,
                all_fine_chan_lambdas_m,
            );

            multi_progress_bar.suspend(|| {
                debug!(
                    "peel loop finished: {source_name} at {source_pos} (has iono {iono_consts:?})"
                )
            });
            peel_progress.inc(1);
        }
    }

    Ok(())
}

#[cfg(any(feature = "cuda", feature = "hip"))]
#[allow(clippy::too_many_arguments)]
fn peel_gpu(
    mut vis_residual_tfb: ArrayViewMut3<Jones<f32>>,
    vis_weights_tfb: ArrayView3<f32>,
    timeblock: &Timeblock,
    source_list: &SourceList,
    iono_consts: &mut [IonoConsts],
    source_weighted_positions: &[RADec],
    num_passes: usize,
    num_loops: usize,
    // TODO (Dev): bake this into weights
    short_baseline_sigma: f64,
    convergence: f64,
    // TODO (Dev): rename chanblocks
    high_res_chanblocks: &[Chanblock],
    // TODO (Dev): derive from high_res_chanblocks
    all_fine_chan_lambdas_m: &[f64],
    low_res_lambdas_m: &[f64],
    obs_context: &ObsContext,
    array_position: LatLngHeight,
    unflagged_tile_xyzs: &[XyzGeodetic],
    // TODO (Dev): bake this into weights
    baseline_weights: &[f64],
    high_res_modeller: &mut SkyModellerGpu,
    dut1: Duration,
    no_precession: bool,
    multi_progress_bar: &MultiProgress,
) -> Result<(), PeelError> {
    use std::collections::{HashMap, HashSet};

    use crate::srclist::{ComponentType, FluxDensity, FluxDensityType};

    let timestamps = &timeblock.timestamps;

    let num_timesteps = vis_residual_tfb.len_of(Axis(0));
    let num_tiles = unflagged_tile_xyzs.len();
    let num_cross_baselines = (num_tiles * (num_tiles - 1)) / 2;
    let num_high_res_chans = all_fine_chan_lambdas_m.len();
    let num_high_res_chans_spw = high_res_chanblocks.len();
    assert_eq!(
        num_high_res_chans, num_high_res_chans_spw,
        "chans from fine_chan_lambdas {} != chans in high_res_chanblocks (flagged+unflagged) {}",
        num_high_res_chans, num_high_res_chans_spw
    );

    let num_low_res_chans = low_res_lambdas_m.len();
    assert!(
        num_high_res_chans % num_low_res_chans == 0,
        "TODO: averaging can't deal with non-integer ratios. channels high {} low {}",
        num_high_res_chans,
        num_low_res_chans
    );

    let num_timesteps_i32: i32 = num_timesteps.try_into().expect("smaller than i32::MAX");
    let num_tiles_i32: i32 = num_tiles.try_into().expect("smaller than i32::MAX");
    let num_cross_baselines_i32: i32 = num_cross_baselines
        .try_into()
        .expect("smaller than i32::MAX");
    let num_high_res_chans_i32 = num_high_res_chans
        .try_into()
        .expect("smaller than i32::MAX");
    let num_low_res_chans_i32 = num_low_res_chans.try_into().expect("smaller than i32::MAX");

    let num_sources_to_iono_subtract = iono_consts.len();

    let (time_axis, freq_axis, baseline_axis) = (Axis(0), Axis(1), Axis(2));

    assert_eq!(vis_residual_tfb.len_of(time_axis), timestamps.len());
    assert_eq!(vis_weights_tfb.len_of(time_axis), timestamps.len());

    assert_eq!(vis_residual_tfb.len_of(baseline_axis), num_cross_baselines);
    assert_eq!(vis_weights_tfb.len_of(baseline_axis), num_cross_baselines);

    assert_eq!(vis_residual_tfb.len_of(freq_axis), num_high_res_chans);
    assert_eq!(vis_weights_tfb.len_of(freq_axis), num_high_res_chans);

    assert!(num_sources_to_iono_subtract <= source_list.len());

    if num_sources_to_iono_subtract == 0 {
        return Ok(());
    }

    let timestamps = &timeblock.timestamps;
    let peel_progress = multi_progress_bar.add(
        ProgressBar::new(num_sources_to_iono_subtract as _)
            .with_style(
                ProgressStyle::default_bar()
                    .template("{msg:17}: [{wide_bar:.blue}] {pos:2}/{len:2} sources ({elapsed_precise}<{eta_precise})").unwrap()
                    .progress_chars("=> "),
            )
            .with_position(0)
    );
    peel_progress.tick();

    macro_rules! pb_warn { ($($arg:tt)+) => (multi_progress_bar.suspend(|| warn!($($arg)+))) }
    macro_rules! pb_info { ($($arg:tt)+) => (multi_progress_bar.suspend(|| info!($($arg)+))) }
    macro_rules! pb_debug { ($($arg:tt)+) => (multi_progress_bar.suspend(|| debug!($($arg)+))) }
    macro_rules! pb_trace { ($($arg:tt)+) => (multi_progress_bar.suspend(|| trace!($($arg)+))) }

    let mut lmsts = vec![0.; timestamps.len()];
    let mut latitudes = vec![0.; timestamps.len()];
    let mut tile_xyzs_high_res = Array2::<XyzGeodetic>::default((timestamps.len(), num_tiles));
    let mut high_res_uvws = Array2::default((timestamps.len(), num_cross_baselines));
    let mut tile_uvs_high_res = Array2::<UV>::default((timestamps.len(), num_tiles));
    let mut tile_ws_high_res = Array2::<W>::default((timestamps.len(), num_tiles));

    // Pre-compute high-res tile UVs and Ws at observation phase centre.
    for (
        &time,
        lmst,
        latitude,
        mut tile_xyzs_high_res,
        mut high_res_uvws,
        mut tile_uvs_high_res,
        mut tile_ws_high_res,
    ) in izip!(
        timestamps.iter(),
        lmsts.iter_mut(),
        latitudes.iter_mut(),
        tile_xyzs_high_res.outer_iter_mut(),
        high_res_uvws.outer_iter_mut(),
        tile_uvs_high_res.outer_iter_mut(),
        tile_ws_high_res.outer_iter_mut(),
    ) {
        if !no_precession {
            let precession_info = precess_time(
                array_position.longitude_rad,
                array_position.latitude_rad,
                obs_context.phase_centre,
                time,
                dut1,
            );
            tile_xyzs_high_res
                .iter_mut()
                .zip_eq(&precession_info.precess_xyz(unflagged_tile_xyzs))
                .for_each(|(a, b)| *a = *b);
            *lmst = precession_info.lmst_j2000;
            *latitude = precession_info.array_latitude_j2000;
        } else {
            tile_xyzs_high_res
                .iter_mut()
                .zip_eq(unflagged_tile_xyzs)
                .for_each(|(a, b)| *a = *b);
            *lmst = get_lmst(array_position.longitude_rad, time, dut1);
            *latitude = array_position.latitude_rad;
        };
        let hadec_phase = obs_context.phase_centre.to_hadec(*lmst);
        let (s_ha, c_ha) = hadec_phase.ha.sin_cos();
        let (s_dec, c_dec) = hadec_phase.dec.sin_cos();
        let mut tile_uvws_high_res = vec![UVW::default(); num_tiles];
        for (tile_uvw, tile_uv, tile_w, &tile_xyz) in izip!(
            tile_uvws_high_res.iter_mut(),
            tile_uvs_high_res.iter_mut(),
            tile_ws_high_res.iter_mut(),
            tile_xyzs_high_res.iter(),
        ) {
            let uvw = UVW::from_xyz_inner(tile_xyz, s_ha, c_ha, s_dec, c_dec);
            *tile_uvw = uvw;
            *tile_uv = UV { u: uvw.u, v: uvw.v };
            *tile_w = W(uvw.w);
        }

        // The UVWs for every timestep will be the same (because the phase
        // centres are always the same). Make these ahead of time for
        // efficiency.
        let mut count = 0;
        for (i, t1) in tile_uvws_high_res.iter().enumerate() {
            for t2 in tile_uvws_high_res.iter().skip(i + 1) {
                high_res_uvws[count] = *t1 - *t2;
                count += 1;
            }
        }
    }

    // /////// //
    // WEIGHTS //
    // /////// //

    // // use the baseline taper from the RTS, 1-exp(-(u*u+v*v)/(2*sig^2));
    let vis_weights_tfb = {
        let mut iono_taper = get_weights_rts(
            tile_uvs_high_res.view(),
            all_fine_chan_lambdas_m,
            short_baseline_sigma,
        );
        iono_taper *= &vis_weights_tfb;
        iono_taper
    };
    let vis_weights_tfb = {
        assert_eq!(baseline_weights.len(), vis_weights_tfb.len_of(Axis(2)));
        let mut vis_weights_tfb = vis_weights_tfb.to_owned();
        for (vis_weight, bl_weight) in vis_weights_tfb
            .iter_mut()
            .zip(baseline_weights.iter().cycle())
        {
            *vis_weight = (*vis_weight as f64 * *bl_weight) as f32;
        }
        vis_weights_tfb
    };

    // ////////////////// //
    // LOW RES PRECESSION //
    // ////////////////// //

    let (average_lmst, _average_latitude, average_tile_xyzs) = if no_precession {
        let average_timestamp = timeblock.median;
        let average_tile_xyzs =
            ArrayView2::from_shape((1, num_tiles), unflagged_tile_xyzs).expect("correct shape");
        (
            get_lmst(array_position.longitude_rad, average_timestamp, dut1),
            array_position.latitude_rad,
            CowArray::from(average_tile_xyzs),
        )
    } else {
        let average_timestamp = timeblock.median;
        let average_precession_info = precess_time(
            array_position.longitude_rad,
            array_position.latitude_rad,
            obs_context.phase_centre,
            average_timestamp,
            dut1,
        );
        let average_precessed_tile_xyzs = Array2::from_shape_vec(
            (1, num_tiles),
            average_precession_info.precess_xyz(unflagged_tile_xyzs),
        )
        .expect("correct shape");

        (
            average_precession_info.lmst_j2000,
            average_precession_info.array_latitude_j2000,
            CowArray::from(average_precessed_tile_xyzs),
        )
    };

    let gpu_xyzs_high_res: Vec<_> = tile_xyzs_high_res
        .iter()
        .copied()
        .map(|XyzGeodetic { x, y, z }| gpu::XYZ {
            x: x as GpuFloat,
            y: y as GpuFloat,
            z: z as GpuFloat,
        })
        .collect();
    let d_xyzs = DevicePointer::copy_to_device(&gpu_xyzs_high_res)?;

    // temporary arrays for accumulation
    // TODO: Do a stocktake of arrays that are lying around!
    let vis_residual_low_res_fb: Array3<Jones<f32>> =
        Array3::zeros((1, num_low_res_chans, num_cross_baselines));
    let mut vis_weights_low_res_fb: Array3<f32> = Array3::zeros(vis_residual_low_res_fb.raw_dim());

    // The low-res weights only need to be populated once.
    weights_average(vis_weights_tfb.view(), vis_weights_low_res_fb.view_mut());

    let freq_average_factor: i32 = (all_fine_chan_lambdas_m.len() / num_low_res_chans)
        .try_into()
        .expect("smaller than i32::MAX");

    let mut bad_sources = Vec::<BadSource>::new();

    unsafe {
        let mut gpu_uvws: ArrayBase<ndarray::OwnedRepr<gpu::UVW>, Dim<[usize; 2]>> =
            Array2::default((num_timesteps, num_cross_baselines));
        gpu_uvws
            .outer_iter_mut()
            .zip(tile_xyzs_high_res.outer_iter())
            .zip(lmsts.iter())
            .for_each(|((mut gpu_uvws, xyzs), lmst)| {
                let phase_centre = obs_context.phase_centre.to_hadec(*lmst);
                let v = xyzs_to_cross_uvws(xyzs.as_slice().unwrap(), phase_centre)
                    .into_iter()
                    .map(|uvw| gpu::UVW {
                        u: uvw.u as GpuFloat,
                        v: uvw.v as GpuFloat,
                        w: uvw.w as GpuFloat,
                    })
                    .collect::<Vec<_>>();
                gpu_uvws.assign(&ArrayView1::from(&v));
            });
        let mut d_uvws_from = DevicePointer::copy_to_device(gpu_uvws.as_slice().unwrap())?;
        let mut d_uvws_to =
            DevicePointer::malloc(gpu_uvws.len() * std::mem::size_of::<gpu::UVW>())?;

        let gpu_lmsts: Vec<GpuFloat> = lmsts.iter().map(|l| *l as GpuFloat).collect();
        let d_lmsts = DevicePointer::copy_to_device(&gpu_lmsts)?;

        let gpu_lambdas: Vec<GpuFloat> = all_fine_chan_lambdas_m
            .iter()
            .map(|l| *l as GpuFloat)
            .collect();
        let d_lambdas = DevicePointer::copy_to_device(&gpu_lambdas)?;

        let gpu_xyzs_low_res: Vec<_> = average_tile_xyzs
            .iter()
            .copied()
            .map(|XyzGeodetic { x, y, z }| gpu::XYZ {
                x: x as GpuFloat,
                y: y as GpuFloat,
                z: z as GpuFloat,
            })
            .collect();
        let d_xyzs_low_res = DevicePointer::copy_to_device(&gpu_xyzs_low_res)?;

        let gpu_low_res_lambdas: Vec<GpuFloat> =
            low_res_lambdas_m.iter().map(|l| *l as GpuFloat).collect();
        let d_low_res_lambdas = DevicePointer::copy_to_device(&gpu_low_res_lambdas)?;

        let d_average_lmsts = DevicePointer::copy_to_device(&[average_lmst as GpuFloat])?;
        let mut d_low_res_uvws: DevicePointer<gpu::UVW> =
            DevicePointer::malloc(gpu_uvws.len() * std::mem::size_of::<gpu::UVW>())?;
        // Make the amount of elements in `d_iono_fits` a power of 2, for
        // efficiency.
        let mut d_iono_fits = {
            let min_size =
                num_cross_baselines * num_low_res_chans * std::mem::size_of::<Jones<f64>>();
            let n = (min_size as f64).log2().ceil() as u32;
            let size = 2_usize.pow(n);
            let mut d: DevicePointer<Jones<f64>> = DevicePointer::malloc(size).unwrap();
            d.clear();
            d
        };

        // let mut d_low_res_vis = DevicePointer::malloc(
        //     num_cross_baselines * low_res_freqs_hz.len() * std::mem::size_of::<Jones<GpuFloat>>(),
        // );
        // let mut d_low_res_weights = DevicePointer::malloc(
        //     num_cross_baselines * low_res_freqs_hz.len() * std::mem::size_of::<GpuFloat>(),
        // );

        let mut d_high_res_vis_tfb =
            DevicePointer::copy_to_device(vis_residual_tfb.as_slice().unwrap())?;
        // TODO: rename d_high_res_resid_tfb
        let mut d_high_res_vis2_tfb =
            DevicePointer::copy_to_device(vis_residual_tfb.as_slice().unwrap())?;
        let d_high_res_weights_tfb =
            DevicePointer::copy_to_device(vis_weights_tfb.as_slice().unwrap())?;

        // TODO: rename d_low_res_resid_fb
        let mut d_low_res_vis_fb =
            DevicePointer::copy_to_device(vis_residual_low_res_fb.as_slice().unwrap())?;
        let d_low_res_weights_fb =
            DevicePointer::copy_to_device(vis_weights_low_res_fb.as_slice().unwrap())?;

        let mut d_high_res_model_tfb: DevicePointer<Jones<f32>> = DevicePointer::malloc(
            timestamps.len()
                * num_cross_baselines
                * all_fine_chan_lambdas_m.len()
                * std::mem::size_of::<Jones<f32>>(),
        )?;
        let mut d_low_res_model_fb =
            DevicePointer::copy_to_device(vis_residual_low_res_fb.as_slice().unwrap())?;
        let mut d_low_res_model_rotated =
            DevicePointer::copy_to_device(vis_residual_low_res_fb.as_slice().unwrap())?;

        // One pointer per timestep.
        let mut d_uvws = Vec::with_capacity(high_res_uvws.len_of(Axis(0)));
        // Temp vector to store results.
        let mut gpu_uvws = vec![gpu::UVW::default(); high_res_uvws.len_of(Axis(1))];
        for uvws in high_res_uvws.outer_iter() {
            // Convert the type and push the results to the device,
            // saving the resulting pointer.
            uvws.iter()
                .zip_eq(gpu_uvws.iter_mut())
                .for_each(|(&UVW { u, v, w }, gpu_uvw)| {
                    *gpu_uvw = gpu::UVW {
                        u: u as GpuFloat,
                        v: v as GpuFloat,
                        w: w as GpuFloat,
                    }
                });
            d_uvws.push(DevicePointer::copy_to_device(&gpu_uvws)?);
        }
        let mut d_beam_jones = DevicePointer::default();

        for pass in 0..num_passes {
            peel_progress.reset();
            peel_progress.set_message(format!(
                "Peeling timeblock {}, pass {}",
                timeblock.index + 1,
                pass + 1
            ));

            let mut pass_issues = 0;
            let mut pass_alpha_mag = 0.;
            let mut pass_bega_mag = 0.;
            let mut pass_gain_mag = 0.;

            // this needs to be inside the pass loop, because d_uvws_from gets swapped with d_uvws_to
            gpu_kernel_call!(
                gpu::xyzs_to_uvws,
                d_xyzs.get(),
                d_lmsts.get(),
                d_uvws_from.get_mut(),
                gpu::RADec {
                    ra: obs_context.phase_centre.ra as GpuFloat,
                    dec: obs_context.phase_centre.dec as GpuFloat,
                },
                num_tiles_i32,
                num_cross_baselines_i32,
                num_timesteps_i32
            )?;

            for (i_source, (((source_name, source), iono_consts), source_pos)) in source_list
                .iter()
                .take(num_sources_to_iono_subtract)
                .zip_eq(iono_consts.iter_mut())
                .zip(source_weighted_positions.iter().copied())
                .enumerate()
            {
                let start = std::time::Instant::now();
                // pb_debug!(
                //     "peel loop {pass}: {source_name} at {source_pos} (has iono {iono_consts:?})"
                // );

                // let old_iono_consts = *iono_consts;
                let old_iono_consts = IonoConsts {
                    alpha: iono_consts.alpha,
                    beta: iono_consts.beta,
                    gain: iono_consts.gain,
                };
                let gpu_old_iono_consts = gpu::IonoConsts {
                    alpha: old_iono_consts.alpha,
                    beta: old_iono_consts.beta,
                    gain: old_iono_consts.gain,
                };

                gpu_kernel_call!(
                    gpu::xyzs_to_uvws,
                    d_xyzs.get(),
                    d_lmsts.get(),
                    d_uvws_to.get_mut(),
                    gpu::RADec {
                        ra: source_pos.ra as GpuFloat,
                        dec: source_pos.dec as GpuFloat,
                    },
                    num_tiles_i32,
                    num_cross_baselines_i32,
                    num_timesteps_i32,
                )?;
                pb_trace!("{:?}: xyzs_to_uvws", start.elapsed());

                // rotate d_high_res_vis_tfb in place
                gpu_kernel_call!(
                    gpu::rotate,
                    d_high_res_vis_tfb.get_mut().cast(),
                    num_timesteps_i32,
                    num_cross_baselines_i32,
                    num_high_res_chans_i32,
                    d_uvws_from.get(),
                    d_uvws_to.get(),
                    d_lambdas.get()
                )?;
                pb_trace!("{:?}: rotate", start.elapsed());

                // there's a bug in the modeller where it ignores --no-precession
                // high_res_modeller.update_with_a_source(source, obs_context.phase_centre)?;
                high_res_modeller.update_with_a_source(source, source_pos)?;
                // Clear the old memory before reusing the buffer.
                d_high_res_model_tfb.clear();
                for (i_time, (lmst, latitude)) in lmsts.iter().zip(latitudes.iter()).enumerate() {
                    let original_model_ptr = d_high_res_model_tfb.ptr;
                    d_high_res_model_tfb.ptr = d_high_res_model_tfb
                        .ptr
                        .add(i_time * num_cross_baselines * all_fine_chan_lambdas_m.len());
                    let original_uvw_ptr = d_uvws_to.ptr;
                    d_uvws_to.ptr = d_uvws_to.ptr.add(i_time * num_cross_baselines);
                    high_res_modeller.model_timestep_with(
                        *lmst,
                        *latitude,
                        &d_uvws_to,
                        &mut d_beam_jones,
                        &mut d_high_res_model_tfb,
                    )?;
                    d_high_res_model_tfb.ptr = original_model_ptr;
                    d_uvws_to.ptr = original_uvw_ptr;
                }
                pb_trace!("{:?}: high res model", start.elapsed());

                // d_high_res_vis2_tfb = residuals@src.
                d_high_res_vis_tfb.copy_to(&mut d_high_res_vis2_tfb)?;
                // add iono@src to residuals@src
                gpu_kernel_call!(
                    gpu::add_model,
                    d_high_res_vis2_tfb.get_mut().cast(),
                    d_high_res_model_tfb.get().cast(),
                    gpu_old_iono_consts,
                    d_lambdas.get(),
                    d_uvws_to.get(),
                    num_timesteps_i32,
                    num_high_res_chans_i32,
                    num_cross_baselines_i32,
                )?;
                pb_trace!("{:?}: add_model", start.elapsed());

                // *** UGLY HACK ***
                // d_high_res_model_rotated = iono@src
                let mut d_high_res_model_rotated: DevicePointer<Jones<f32>> =
                    DevicePointer::malloc(d_high_res_model_tfb.get_size())?;
                d_high_res_model_rotated.clear();
                gpu_kernel_call!(
                    gpu::add_model,
                    d_high_res_model_rotated.get_mut().cast(),
                    d_high_res_model_tfb.get().cast(),
                    gpu_old_iono_consts,
                    d_lambdas.get(),
                    d_uvws_to.get(),
                    num_timesteps_i32,
                    num_high_res_chans_i32,
                    num_cross_baselines_i32,
                )?;
                pb_trace!("{:?}: add_model", start.elapsed());

                gpu_kernel_call!(
                    gpu::average,
                    d_high_res_vis2_tfb.get().cast(),
                    d_high_res_weights_tfb.get(),
                    d_low_res_vis_fb.get_mut().cast(),
                    num_timesteps_i32,
                    num_cross_baselines_i32,
                    num_high_res_chans_i32,
                    freq_average_factor
                )?;
                pb_trace!("{:?}: average high res vis", start.elapsed());

                gpu_kernel_call!(
                    gpu::average,
                    d_high_res_model_rotated.get().cast(),
                    d_high_res_weights_tfb.get(),
                    d_low_res_model_fb.get_mut().cast(),
                    num_timesteps_i32,
                    num_cross_baselines_i32,
                    num_high_res_chans_i32,
                    freq_average_factor
                )?;
                pb_trace!("{:?}: average high res model", start.elapsed());

                gpu_kernel_call!(
                    gpu::xyzs_to_uvws,
                    d_xyzs_low_res.get(),
                    d_average_lmsts.get(),
                    d_low_res_uvws.get_mut(),
                    gpu::RADec {
                        ra: source_pos.ra as GpuFloat,
                        dec: source_pos.dec as GpuFloat,
                    },
                    num_tiles_i32,
                    num_cross_baselines_i32,
                    1,
                )?;
                pb_trace!("{:?}: low res xyzs_to_uvws", start.elapsed());

                // !!!!
                // comment me out
                // !!!!
                // we have low res residual and model, now print what iono fit will see.
                // {
                //     use marlu::math::cross_correlation_baseline_to_tiles;
                //     let vis_residual_low_res_fb = d_low_res_vis_fb.copy_from_device_new()?;
                //     dbg!(vis_residual_low_res_fb.len(), num_low_res_chans, num_cross_baselines);
                //     let vis_residual_low_res_fb = Array2::from_shape_vec(
                //         (num_low_res_chans, num_cross_baselines),
                //         vis_residual_low_res_fb,
                //     )
                //     .unwrap();
                //     let vis_model_low_res_fb = d_low_res_model_fb.copy_from_device_new()?;
                //     let vis_model_low_res_fb = Array2::<Jones<f32>>::from_shape_vec(
                //         (num_low_res_chans, num_cross_baselines),
                //         vis_model_low_res_fb,
                //     )
                //     .unwrap();
                //     let vis_weights_low_res_fb = d_low_res_weights_fb.copy_from_device_new()?;
                //     let vis_weights_low_res_fb = Array2::<f32>::from_shape_vec(
                //         (num_low_res_chans, num_cross_baselines),
                //         vis_weights_low_res_fb,
                //     )
                //     .unwrap();
                //     let ant_pairs = (0..num_cross_baselines)
                //         .map(|bl_idx| cross_correlation_baseline_to_tiles(num_tiles, bl_idx))
                //         .collect_vec();
                //     let uvws_low_res = d_low_res_uvws.copy_from_device_new()?;
                //     for (ch_idx, (vis_residual_low_res_b, vis_model_low_res_b, vis_weights_low_res_b, &lambda)) in izip!(
                //         vis_residual_low_res_fb.outer_iter(),
                //         vis_model_low_res_fb.outer_iter(),
                //         vis_weights_low_res_fb.outer_iter(),
                //         low_res_lambdas_m,
                //     ).enumerate() {
                //         // dbg!(&ch_idx);
                //         if ch_idx > 1 {
                //             continue;
                //         }
                //         for (residual, model, weight, &gpu::UVW { u, v, w: _ }, &(ant1, ant2)) in izip!(
                //             vis_residual_low_res_b.iter(),
                //             vis_model_low_res_b.iter(),
                //             vis_weights_low_res_b.iter(),
                //             &uvws_low_res,
                //             ant_pairs.iter(),
                //         ) {
                //             // dbg!(&ant1, &ant2);
                //             if ant1 != 0 || (ant2 >= 16 && ant2 < num_tiles_i32 as usize - 16) {
                //                 continue;
                //             }
                //             let residual_i = residual[0] + residual[3];
                //             let model_i = model[0] + model[3];
                //             let u = u as GpuFloat;
                //             let v = v as GpuFloat;
                //             println!("uv {ant1:3} {ant2:3} ({u:+9.3}, {v:+9.3}) l{lambda:+7.5} wt{weight:+3.1} | RI {:+11.7} @{:+5.3}pi | MI {:+11.7} @{:+5.3}pi", residual_i.norm(), residual_i.arg(), model_i.norm(), model_i.arg());
                //         }
                //     }
                // }

                let mut gpu_iono_consts = gpu::IonoConsts {
                    alpha: 0.0,
                    beta: 0.0,
                    gain: 1.0,
                };
                // get size of device ptr
                let lrblch = (num_cross_baselines_i32 * num_low_res_chans_i32) as f64;
                pb_trace!("before iono_loop nt{:?} nxbl{:?} nlrch{:?} = lrxblch{:?}; lrvfb{:?} lrwfb{:?} lrmfb{:?} lrmrfb{:?}",
                    num_tiles_i32,
                    num_cross_baselines_i32,
                    num_low_res_chans_i32,
                    lrblch,
                    d_low_res_vis_fb.get_size() as f64 / lrblch,
                    d_low_res_weights_fb.get_size() as f64 / lrblch,
                    d_low_res_model_fb.get_size() as f64 / lrblch,
                    d_low_res_model_rotated.get_size() as f64 / lrblch,
                );
                gpu_kernel_call!(
                    gpu::iono_loop,
                    d_low_res_vis_fb.get().cast(),
                    d_low_res_weights_fb.get(),
                    d_low_res_model_fb.get().cast(),
                    d_low_res_model_rotated.get_mut().cast(),
                    d_iono_fits.get_mut().cast(),
                    &mut gpu_iono_consts,
                    num_cross_baselines_i32,
                    num_low_res_chans_i32,
                    num_loops as i32,
                    d_low_res_uvws.get(),
                    d_low_res_lambdas.get(),
                    convergence as GpuFloat,
                )?;
                pb_trace!("{:?}: iono_loop", start.elapsed());

                iono_consts.alpha = old_iono_consts.alpha + gpu_iono_consts.alpha;
                iono_consts.beta = old_iono_consts.beta + gpu_iono_consts.beta;
                iono_consts.gain = old_iono_consts.gain * gpu_iono_consts.gain;

                #[rustfmt::skip]
                let issues = format!(
                    "{}{}{}",
                    if iono_consts.alpha.abs() > 1e-3 {
                        if iono_consts.alpha > 0.0 { "A" } else { "a" }
                    } else {
                        ""
                    },
                    if iono_consts.beta.abs() > 1e-3 {
                        if iono_consts.beta > 0.0 { "B" } else { "b" }
                    } else {
                        ""
                    },
                    if iono_consts.gain < 0.0 {
                        "g"
                    } else if iono_consts.gain > 1.5 {
                        "G"
                    } else {
                        ""
                    },
                );
                let message = format!(
                    // "t{:03} p{pass} s{i_source:6}|{source_name:16} @ ra {:+7.2} d {:+7.2} | a {:+7.2e} b {:+7.2e} g {:+3.2} | da {:+8.2e} db {:+8.2e} dg {:+3.2} | {}",
                    "t{:3} pass {:2} s{i_source:6}|{source_name:16} @ ra {:+7.2} d {:+7.2} | a {:+8.6} b {:+8.6} g {:+3.2} | da {:+8.6} db {:+8.6} dg {:+3.2} | {}",
                    timeblock.index,
                    pass+1,
                    (source_pos.ra.to_degrees() + 180.) % 360. - 180.,
                    source_pos.dec.to_degrees(),
                    iono_consts.alpha,
                    iono_consts.beta,
                    iono_consts.gain,
                    iono_consts.alpha - old_iono_consts.alpha,
                    iono_consts.beta - old_iono_consts.beta,
                    iono_consts.gain - old_iono_consts.gain,
                    issues,
                );
                if issues.is_empty() {
                    pb_debug!("[peel_gpu] {}", message);
                    pass_alpha_mag += (iono_consts.alpha - old_iono_consts.alpha).abs();
                    pass_bega_mag += (iono_consts.beta - old_iono_consts.beta).abs();
                    pass_gain_mag += iono_consts.gain - old_iono_consts.gain;
                } else {
                    pb_debug!(
                        "[peel_gpu] {} (reverting to a {:+8.6} b {:+8.6} g {:+3.2})",
                        message,
                        old_iono_consts.alpha,
                        old_iono_consts.beta,
                        old_iono_consts.gain,
                    );
                    bad_sources.push(BadSource {
                        gpstime: timeblock.median.to_gpst_seconds(),
                        pass,
                        i_source,
                        source_name: source_name.to_string(),
                        // weighted_catalogue_pos_j2000: source_pos,
                        alpha: iono_consts.alpha,
                        beta: iono_consts.beta,
                        gain: iono_consts.gain,
                        residuals_i: Vec::default(), // todo!(),
                        residuals_q: Vec::default(), // todo!(),
                        residuals_u: Vec::default(), // todo!(),
                        residuals_v: Vec::default(), // todo!(),
                    });

                    iono_consts.alpha = old_iono_consts.alpha;
                    iono_consts.beta = old_iono_consts.beta;
                    iono_consts.gain = old_iono_consts.gain;
                    pass_issues += 1;
                }

                let gpu_iono_consts = gpu::IonoConsts {
                    alpha: iono_consts.alpha,
                    beta: iono_consts.beta,
                    gain: iono_consts.gain,
                };

                pb_trace!("subtracting {gpu_old_iono_consts:?} adding {gpu_iono_consts:?}");

                // vis += iono(model, old_consts) - iono(model, consts)
                gpu_kernel_call!(
                    gpu::subtract_iono,
                    d_high_res_vis_tfb.get_mut().cast(),
                    d_high_res_model_tfb.get().cast(),
                    gpu_iono_consts,
                    gpu_old_iono_consts,
                    d_uvws_to.get(),
                    d_lambdas.get(),
                    num_timesteps_i32,
                    num_cross_baselines_i32,
                    num_high_res_chans_i32,
                )?;
                pb_trace!("{:?}: subtract_iono", start.elapsed());

                // Peel?
                let num_sources_to_peel = 0;
                if pass == num_passes - 1 && i_source < num_sources_to_peel {
                    // We currently can only do DI calibration on the CPU. Copy the visibilities back to the host.
                    let vis = d_high_res_vis_tfb.copy_from_device_new()?;

                    // *** UGLY HACK ***
                    let mut d_high_res_model_rotated: DevicePointer<Jones<f32>> =
                        DevicePointer::malloc(d_high_res_model_tfb.get_size())?;
                    d_high_res_model_rotated.clear();
                    gpu_kernel_call!(
                        gpu::add_model,
                        d_high_res_model_rotated.get_mut().cast(),
                        d_high_res_model_tfb.get().cast(),
                        //iono_consts.0 as GpuFloat,
                        //iono_consts.1 as GpuFloat,
                        gpu_iono_consts,
                        d_lambdas.get(),
                        d_uvws_to.get(),
                        num_timesteps_i32,
                        num_high_res_chans_i32,
                        num_cross_baselines_i32,
                    )?;
                    let model = d_high_res_model_rotated.copy_from_device_new()?;

                    let mut di_jones =
                        Array3::from_elem((1, num_tiles, num_high_res_chans), Jones::identity());
                    let shape = (num_timesteps, num_high_res_chans, num_cross_baselines);
                    let pb = ProgressBar::hidden();
                    let di_cal_results = calibrate_timeblock(
                        ArrayView3::from_shape(shape, &vis).expect("correct shape"),
                        ArrayView3::from_shape(shape, &model).expect("correct shape"),
                        di_jones.view_mut(),
                        timeblock,
                        high_res_chanblocks,
                        50,
                        1e-8,
                        1e-4,
                        obs_context.polarisations,
                        pb,
                        true,
                    );
                    if di_cal_results.into_iter().all(|r| r.converged) {
                        // Apply.
                    }
                }

                // The new phase centre becomes the old one.
                std::mem::swap(&mut d_uvws_from, &mut d_uvws_to);

                peel_progress.inc(1);
            }

            let num_good_sources = (num_sources_to_iono_subtract - pass_issues) as f64;
            if num_good_sources > 0. {
                let msg = format!(
                    "t{:3} pass {:2} ma {:+7.2e} mb {:+7.2e} mg {:+7.2e}",
                    timeblock.index,
                    pass + 1,
                    pass_alpha_mag / num_good_sources,
                    pass_bega_mag / num_good_sources,
                    pass_gain_mag / num_good_sources
                );
                if pass_issues > 0 {
                    pb_warn!("[peel_gpu] {} ({} issues)", msg, pass_issues);
                } else {
                    pb_info!("[peel_gpu] {} (no issues)", msg);
                }
            } else {
                pb_warn!(
                    "[peel_gpu] t{:03} pass {:2} all sources had issues",
                    timeblock.index,
                    pass + 1
                );
            }

            // Rotate back to the phase centre.
            gpu_kernel_call!(
                gpu::xyzs_to_uvws,
                d_xyzs.get(),
                d_lmsts.get(),
                d_uvws_to.get_mut(),
                gpu::RADec {
                    ra: obs_context.phase_centre.ra as GpuFloat,
                    dec: obs_context.phase_centre.dec as GpuFloat,
                },
                num_tiles_i32,
                num_cross_baselines_i32,
                num_timesteps_i32
            )?;

            gpu_kernel_call!(
                gpu::rotate,
                d_high_res_vis_tfb.get_mut().cast(),
                num_timesteps_i32,
                num_cross_baselines_i32,
                num_high_res_chans_i32,
                d_uvws_from.get(),
                d_uvws_to.get(),
                d_lambdas.get()
            )?;
        }

        // copy results back to host
        d_high_res_vis_tfb.copy_from_device(vis_residual_tfb.as_slice_mut().unwrap())?;
    }

    let mut pass_counts = HashMap::<String, usize>::new();
    for bad_source in bad_sources.iter() {
        *pass_counts
            .entry(bad_source.source_name.clone())
            .or_default() += 1;
    }
    let mut printed = HashSet::<String>::new();
    bad_sources.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
    for bad_source in bad_sources.into_iter() {
        let BadSource {
            gpstime,
            // pass,
            i_source,
            source_name,
            alpha,
            beta,
            gain,
            ..
        } = bad_source;
        // if source_name is in printed
        if printed.contains(&source_name) {
            continue;
        } else {
            printed.insert(source_name.clone());
        }
        let passes = pass_counts[&source_name];

        let pos = source_weighted_positions[i_source];
        let RADec { ra, dec } = pos;
        pb_warn!(
            "[peel_gpu] Bad source: {:.2} p{:2} {} at radec({:+7.2}, {:+7.2}) iono({:+8.6},{:+8.6},{:+3.2})",
            gpstime,
            passes,
            source_name,
            (ra + 180.) % 360. - 180.,
            dec,
            alpha,
            beta,
            gain
        );
        let matches = source_list.search_asec(pos, 300.);
        for (sep, src_name, idx, comp) in matches {
            let compstr = match comp.comp_type {
                ComponentType::Gaussian { maj, min, pa } => format!(
                    "G {:6.1}as {:6.1}as {:+6.1}d",
                    maj.to_degrees() * 3600.,
                    min.to_degrees() * 3600.,
                    pa.to_degrees()
                ),
                ComponentType::Point => format!("P {:8} {:8} {:7}", "", "", ""),
                _ => format!("? {:8} {:8} {:7}", "", "", ""),
            };
            let fluxstr = match comp.flux_type {
                FluxDensityType::CurvedPowerLaw {
                    si,
                    fd: FluxDensity { freq, i, .. },
                    q,
                } => format!(
                    "cpl S={:+6.2}(νn)^{:+5.2} exp[{:+5.2}ln(νn)]; @{:.1}MHz",
                    i,
                    si,
                    q,
                    freq / 1e6
                ),
                FluxDensityType::PowerLaw {
                    si,
                    fd: FluxDensity { freq, i, .. },
                } => format!("pl  S={:+6.2}(νn)^{:+5.2}; @{:.1}MHz", i, si, freq / 1e6),
                FluxDensityType::List(fds) => {
                    let FluxDensity { i, freq, .. } = fds[0];
                    format!("lst S={:+6.2} @{:.1}MHz", i, freq / 1e6)
                }
            };
            pb_warn!(
                "[peel_gpu]  {sep:5.1} {src_name:16} c{idx:2} at radec({:+7.2},{:+7.2}) comp({}) flx ({})",
                (comp.radec.ra + 180.) % 360. - 180.,
                comp.radec.dec,
                compstr,
                fluxstr,
            );
        }
    }

    Ok(())
}

/// Just the W terms of [`UVW`] coordinates.
#[derive(Clone, Copy, Default, PartialEq, Debug)]
struct W(f64);

impl W {
    fn _from_xyz(xyz: XyzGeodetic, s_ha: f64, c_ha: f64, s_dec: f64, c_dec: f64) -> W {
        W(c_dec * c_ha * xyz.x - c_dec * s_ha * xyz.y + s_dec * xyz.z)
    }
}

impl Sub for W {
    type Output = f64;

    fn sub(self, rhs: Self) -> Self::Output {
        self.0 - rhs.0
    }
}

impl Neg for W {
    type Output = Self;

    fn neg(self) -> Self::Output {
        W(-self.0)
    }
}

#[cfg(test)]
impl approx::AbsDiffEq for W {
    type Epsilon = f64;

    fn default_epsilon() -> Self::Epsilon {
        f64::EPSILON
    }

    fn abs_diff_eq(&self, other: &Self, epsilon: Self::Epsilon) -> bool {
        f64::abs_diff_eq(&self.0, &other.0, epsilon)
    }
}

/// Just the U and V terms of [`UVW`] coordinates.
#[derive(Clone, Copy, Default, PartialEq, Debug)]
struct UV {
    u: f64,
    v: f64,
}

impl UV {
    fn from_xyz(xyz: XyzGeodetic, s_ha: f64, c_ha: f64, s_dec: f64, c_dec: f64) -> UV {
        UV {
            u: s_ha * xyz.x + c_ha * xyz.y,
            v: -s_dec * c_ha * xyz.x + s_dec * s_ha * xyz.y + c_dec * xyz.z,
        }
    }
}

impl Sub for UV {
    type Output = UV;

    fn sub(self, rhs: Self) -> Self::Output {
        UV {
            u: self.u - rhs.u,
            v: self.v - rhs.v,
        }
    }
}

impl Div<f64> for UV {
    type Output = UV;

    fn div(self, rhs: f64) -> Self::Output {
        UV {
            u: self.u / rhs,
            v: self.v / rhs,
        }
    }
}

#[cfg(test)]
impl approx::AbsDiffEq for UV {
    type Epsilon = f64;

    fn default_epsilon() -> Self::Epsilon {
        f64::EPSILON
    }

    fn abs_diff_eq(&self, other: &Self, epsilon: Self::Epsilon) -> bool {
        f64::abs_diff_eq(&self.u, &other.u, epsilon) && f64::abs_diff_eq(&self.v, &other.v, epsilon)
    }
}

#[allow(clippy::too_many_arguments)]
fn read_thread(
    input_vis_params: &InputVisParams,
    tx_data: Sender<(Array2<Jones<f32>>, Array2<f32>, Epoch)>,
    error: &AtomicCell<bool>,
    read_progress: &ProgressBar,
) -> Result<(), VisReadError> {
    let num_unflagged_tiles = input_vis_params.get_num_unflagged_tiles();
    let num_unflagged_cross_baselines = (num_unflagged_tiles * (num_unflagged_tiles - 1)) / 2;

    for timeblock in &input_vis_params.timeblocks {
        // Make a new block of data to be passed along.
        let mut vis_data_fb = Array2::zeros((
            input_vis_params.spw.chanblocks.len(),
            num_unflagged_cross_baselines,
        ));
        let mut vis_weights_fb = Array2::zeros(vis_data_fb.raw_dim());

        input_vis_params.read_timeblock(
            timeblock,
            vis_data_fb.view_mut(),
            vis_weights_fb.view_mut(),
            None,
            error,
        )?;

        // Should we continue?
        if error.load() {
            return Ok(());
        }

        debug!(
            "[read] vdfb shp={:?} sum={:?} vwfb shp={:?} sum={:?}",
            vis_data_fb.shape(),
            vis_data_fb.sum(),
            vis_weights_fb.shape(),
            vis_weights_fb.sum(),
        );

        match tx_data.send((vis_data_fb, vis_weights_fb, timeblock.median)) {
            Ok(()) => (),
            // If we can't send the message, it's because the
            // channel has been closed on the other side. That
            // should only happen because the writer has exited
            // due to error; in that case, just exit this
            // thread.
            Err(_) => return Ok(()),
        }

        read_progress.inc(1);
    }

    read_progress.abandon_with_message("Finished reading input data");
    Ok(())
}

// subtract the model from the visibilities to get residuals
// acts on a stream of 2D visibilities and weights [chan, baseline]
#[allow(clippy::too_many_arguments)]
fn subtract_thread(
    beam: &dyn Beam,
    source_list: &SourceList,
    obs_context: &ObsContext,
    unflagged_tile_xyzs: &[XyzGeodetic],
    tile_baseline_flags: &TileBaselineFlags,
    array_position: LatLngHeight,
    dut1: Duration,
    all_fine_chan_freqs_hz: &[f64],
    apply_precession: bool,
    rx_data: Receiver<(Array2<Jones<f32>>, Array2<f32>, Epoch)>,
    tx_residual: Sender<(Array2<Jones<f32>>, Array2<f32>, Epoch)>,
    error: &AtomicCell<bool>,
    model_progress: &ProgressBar,
    sub_progress: &ProgressBar,
) -> Result<(), ModelError> {
    let mut cpu_modeller = if matches!(MODEL_DEVICE.load(), ModelDevice::Cpu) {
        Some(SkyModellerCpu::new(
            beam,
            // &SourceList::new(),
            source_list,
            obs_context.polarisations,
            unflagged_tile_xyzs,
            all_fine_chan_freqs_hz,
            &tile_baseline_flags.flagged_tiles,
            obs_context.phase_centre,
            array_position.longitude_rad,
            array_position.latitude_rad,
            dut1,
            apply_precession,
        ))
    } else {
        None
    };

    #[cfg(any(feature = "cuda", feature = "hip"))]
    let mut gpu_modeller = if matches!(MODEL_DEVICE.load(), ModelDevice::Gpu) {
        let modeller = SkyModellerGpu::new(
            beam,
            //&SourceList::new(),
            source_list,
            obs_context.polarisations,
            unflagged_tile_xyzs,
            all_fine_chan_freqs_hz,
            &tile_baseline_flags.flagged_tiles,
            obs_context.phase_centre,
            array_position.longitude_rad,
            array_position.latitude_rad,
            dut1,
            apply_precession,
        )?;
        Some(modeller)
    } else {
        None
    };

    for (mut vis_data_fb, vis_weights_fb, timestamp) in rx_data.iter() {
        // Should we continue?
        if error.load() {
            return Ok(());
        }
        sub_progress.reset();

        // Here, we make the data negative, and as we iterate
        // over all sources, they get modelled and added to the
        // negative data. Once we're done with the sources,
        // we'll turn negate this array and then we have
        // residuals.
        vis_data_fb.iter_mut().for_each(|j| *j *= -1.0);

        // let (lst, xyzs, latitude) =
        if apply_precession {
            let precession_info = precess_time(
                array_position.longitude_rad,
                array_position.latitude_rad,
                obs_context.phase_centre,
                timestamp,
                dut1,
            );
            // Apply precession to the tile XYZ positions.
            let precessed_tile_xyzs = precession_info.precess_xyz(unflagged_tile_xyzs);
            debug!(
                "Modelling GPS timestamp {}, LMST {}°, J2000 LMST {}°",
                timestamp.to_gpst_seconds(),
                precession_info.lmst.to_degrees(),
                precession_info.lmst_j2000.to_degrees()
            );
            (
                precession_info.lmst_j2000,
                Cow::from(precessed_tile_xyzs),
                precession_info.array_latitude_j2000,
            )
        } else {
            let lst = get_lmst(array_position.longitude_rad, timestamp, dut1);
            debug!(
                "Modelling GPS timestamp {}, LMST {}°",
                timestamp.to_gpst_seconds(),
                lst.to_degrees()
            );
            (
                lst,
                Cow::from(unflagged_tile_xyzs),
                array_position.latitude_rad,
            )
        };

        sub_progress.tick();

        if let Some(modeller) = cpu_modeller.as_mut() {
            modeller.model_timestep_with(timestamp, vis_data_fb.view_mut())?;
        }

        #[cfg(any(feature = "cuda", feature = "hip"))]
        if let Some(modeller) = gpu_modeller.as_mut() {
            let (model_vis, _) = modeller.model_timestep(timestamp)?;
            vis_data_fb += &model_vis;
        }

        vis_data_fb.iter_mut().for_each(|j| *j *= -1.0);
        match tx_residual.send((vis_data_fb, vis_weights_fb, timestamp)) {
            Ok(()) => (),
            Err(_) => return Ok(()),
        }

        model_progress.inc(1);
    }

    model_progress.abandon_with_message("Finished generating residuals");
    sub_progress.finish_and_clear();
    Ok(())
}

/// reshapes residuals for peel.
/// receives a stream of 2D residuals and weights [chan, baseline]
/// joins them into a 3D array [time, chan, baseline] whose time axis
/// is determined by iono_timeblocks.
///
/// # Warning
/// Each time rx_residual.iter() is called, it will
/// consume the stream. If called multiple times it will skip items.
fn joiner_thread<'a>(
    iono_timeblocks: &'a [Timeblock],
    spw: &Spw,
    num_unflagged_cross_baselines: usize,
    rx_residual: Receiver<(Array2<Jones<f32>>, Array2<f32>, Epoch)>,
    tx_full_residual: Sender<(Array3<Jones<f32>>, Array3<f32>, &'a Timeblock)>,
    error: &AtomicCell<bool>,
) {
    for timeblock in iono_timeblocks {
        let mut vis_residual_tfb = Array3::zeros((
            timeblock.timestamps.len(),
            spw.chanblocks.len(),
            num_unflagged_cross_baselines,
        ));
        let mut vis_weights_tfb = Array3::zeros(vis_residual_tfb.raw_dim());

        let timestamps = &timeblock.timestamps;
        trace!("[joiner] timestamps={timestamps:?}");

        for (mut full_residual_fb, mut full_weights_fb) in izip!(
            vis_residual_tfb.outer_iter_mut(),
            vis_weights_tfb.outer_iter_mut()
        ) {
            let (vis_residual_fb, mut vis_weights_fb, timestamp) = rx_residual.recv().unwrap();
            assert!(timestamps.contains(&timestamp));

            // Should we continue?
            if error.load() {
                return;
            }

            // Cap negative weights to 0.
            vis_weights_fb.iter_mut().for_each(|w| {
                if *w <= 0.0 {
                    *w = -0.0;
                }
            });

            full_residual_fb.assign(&vis_residual_fb);
            full_weights_fb.assign(&vis_weights_fb);
        }

        if vis_weights_tfb.sum() < 0.0 {
            warn!("[joiner] all flagged: timestamps={timestamps:?}")
        }

        match tx_full_residual.send((vis_residual_tfb, vis_weights_tfb, timeblock)) {
            Ok(()) => (),
            Err(_) => return,
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn peel_thread(
    beam: &dyn Beam,
    source_list: &SourceList,
    source_weighted_positions: &[RADec],
    num_sources_to_iono_subtract: usize,
    num_passes: NonZeroUsize,
    num_loops: NonZeroUsize,
    obs_context: &ObsContext,
    unflagged_tile_xyzs: &[XyzGeodetic],
    uvw_min_metres: f64,
    uvw_max_metres: f64,
    short_baseline_sigma: f64,
    convergence: f64,
    tile_baseline_flags: &TileBaselineFlags,
    array_position: LatLngHeight,
    dut1: Duration,
    chanblocks: &[Chanblock],
    all_fine_chan_lambdas_m: &[f64],
    low_res_freqs_hz: &[f64],
    low_res_lambdas_m: &[f64],
    apply_precession: bool,
    output_vis_params: Option<&OutputVisParams>,
    rx_full_residual: Receiver<(Array3<Jones<f32>>, Array3<f32>, &Timeblock)>,
    tx_write: Sender<VisTimestep>,
    tx_iono_consts: Sender<Vec<IonoConsts>>,
    error: &AtomicCell<bool>,
    multi_progress: &MultiProgress,
    overall_peel_progress: &ProgressBar,
) -> Result<(), PeelError> {
    let mut baseline_weights = None;

    for (i, (mut vis_residual_tfb, vis_weights_tfb, timeblock)) in
        rx_full_residual.iter().enumerate()
    {
        // Should we continue?
        if error.load() {
            return Ok(());
        }

        let baseline_weights = baseline_weights.get_or_insert_with(|| {
            let mut baseline_weights = Vec1::try_from_vec(vec![
                1.0;
                tile_baseline_flags
                    .unflagged_cross_baseline_to_tile_map
                    .len()
            ])
            .expect("not possible to have no unflagged tiles here");
            let uvws = xyzs_to_cross_uvws(
                unflagged_tile_xyzs,
                obs_context.phase_centre.to_hadec(get_lmst(
                    array_position.longitude_rad,
                    *timeblock.timestamps.first(),
                    dut1,
                )),
            );
            assert_eq!(baseline_weights.len(), uvws.len());
            for (UVW { u, v, w }, baseline_weight) in
                uvws.into_iter().zip(baseline_weights.iter_mut())
            {
                let uvw_length = (u.powi(2) + v.powi(2) + w.powi(2)).sqrt();
                if uvw_length < uvw_min_metres || uvw_length > uvw_max_metres {
                    *baseline_weight = 0.0;
                }
            }

            baseline_weights
        });

        let all_fine_chan_freqs_hz = chanblocks.iter().map(|c| c.freq).collect::<Vec<_>>();
        let mut iono_consts = vec![IonoConsts::default(); num_sources_to_iono_subtract];
        if num_sources_to_iono_subtract > 0 {
            if matches!(MODEL_DEVICE.load(), ModelDevice::Cpu) {
                let mut high_res_modeller = SkyModellerCpu::new(
                    beam,
                    &SourceList::new(),
                    obs_context.polarisations,
                    unflagged_tile_xyzs,
                    &all_fine_chan_freqs_hz,
                    &tile_baseline_flags.flagged_tiles,
                    RADec::default(),
                    array_position.longitude_rad,
                    array_position.latitude_rad,
                    dut1,
                    apply_precession,
                );

                peel_cpu(
                    vis_residual_tfb.view_mut(),
                    vis_weights_tfb.view(),
                    timeblock,
                    source_list,
                    &mut iono_consts,
                    source_weighted_positions,
                    num_passes.get(),
                    num_loops.get(),
                    short_baseline_sigma,
                    convergence,
                    low_res_freqs_hz,
                    all_fine_chan_lambdas_m,
                    low_res_lambdas_m,
                    obs_context,
                    array_position,
                    unflagged_tile_xyzs,
                    &mut high_res_modeller,
                    dut1,
                    !apply_precession,
                    multi_progress,
                )?;
            }

            #[cfg(any(feature = "cuda", feature = "hip"))]
            if matches!(MODEL_DEVICE.load(), ModelDevice::Gpu) {
                let mut high_res_modeller = SkyModellerGpu::new(
                    beam,
                    &SourceList::new(),
                    obs_context.polarisations,
                    unflagged_tile_xyzs,
                    &all_fine_chan_freqs_hz,
                    &tile_baseline_flags.flagged_tiles,
                    RADec::default(),
                    array_position.longitude_rad,
                    array_position.latitude_rad,
                    dut1,
                    apply_precession,
                )?;
                peel_gpu(
                    vis_residual_tfb.view_mut(),
                    vis_weights_tfb.view(),
                    timeblock,
                    source_list,
                    &mut iono_consts,
                    source_weighted_positions,
                    num_passes.get(),
                    num_loops.get(),
                    short_baseline_sigma,
                    convergence,
                    chanblocks,
                    all_fine_chan_lambdas_m,
                    low_res_lambdas_m,
                    obs_context,
                    array_position,
                    unflagged_tile_xyzs,
                    baseline_weights,
                    &mut high_res_modeller,
                    dut1,
                    !apply_precession,
                    multi_progress,
                )?;
            }

            // dev: what's with this?
            if i == 0 {
                source_list
                    .iter()
                    .take(10)
                    .zip(iono_consts.iter())
                    .for_each(|((name, src), iono_consts)| {
                        multi_progress
                            .println(format!(
                                "{name} ({:>7.3}°, {:>7.3}°): {:+.5e} {:+.5e} {:.3}",
                                src.components[0].radec.ra.to_degrees(),
                                src.components[0].radec.dec.to_degrees(),
                                iono_consts.alpha,
                                iono_consts.beta,
                                iono_consts.gain,
                            ))
                            .unwrap();
                    });
            }
        }

        match tx_iono_consts.send(iono_consts) {
            Ok(()) => (),
            Err(_) => return Ok(()),
        }

        for ((cross_data_fb, cross_weights_fb), timestamp) in vis_residual_tfb
            .outer_iter()
            .zip(vis_weights_tfb.outer_iter())
            .zip(timeblock.timestamps.iter())
        {
            // TODO: Puke.
            let cross_data_fb = cross_data_fb.to_shared();
            let cross_weights_fb = cross_weights_fb.to_shared();
            if output_vis_params.is_some() {
                match tx_write.send(VisTimestep {
                    cross_data_fb,
                    cross_weights_fb,
                    autos: None,
                    timestamp: *timestamp,
                }) {
                    Ok(()) => (),
                    Err(_) => return Ok(()),
                }
            }
        }

        overall_peel_progress.inc(1);
    }

    overall_peel_progress.abandon_with_message("Finished peeling");
    Ok(())
}

#[derive(thiserror::Error, Debug)]
pub(crate) enum PeelError {
    #[error(transparent)]
    VisRead(#[from] crate::io::read::VisReadError),

    #[error(transparent)]
    VisWrite(#[from] crate::io::write::VisWriteError),

    #[error(transparent)]
    FileWrite(#[from] crate::io::write::FileWriteError),

    #[error(transparent)]
    Beam(#[from] crate::beam::BeamError),

    #[error(transparent)]
    Model(#[from] crate::model::ModelError),

    #[error(transparent)]
    IO(#[from] std::io::Error),

    #[cfg(any(feature = "cuda", feature = "hip"))]
    #[error(transparent)]
    Gpu(#[from] crate::gpu::GpuError),
}
