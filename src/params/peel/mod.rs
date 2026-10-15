// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

#[cfg(test)]
mod tests;

use std::{
    borrow::Cow,
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
    io::{
        read::VisReadError,
        write::{write_vis, VisTimestep},
    },
    math::div_ceil,
    model::{ModelDevice, ModelError, SkyModeller, SkyModellerCpu},
    srclist::SourceList,
    Chanblock, TileBaselineFlags, MODEL_DEVICE, PROGRESS_BARS,
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

/// parameters relating to weighting of visibilities
pub(crate) struct PeelWeightParams {
    /// minimum uvw cutoff in metres
    pub(crate) uvw_min_metres: f64,
    /// maximum uvw cutoff in metres
    pub(crate) uvw_max_metres: f64,
    /// sigma for RTS baseline taper, 1-exp(-(u²+v²)/(2*σ²));
    pub(crate) short_baseline_sigma: f64,
}

impl PeelWeightParams {
    /// Applies the baseline weights to the visibilities.
    pub(crate) fn apply_tfb(
        &self,
        mut vis_weights_tfb: ArrayViewMut3<f32>,
        obs_context: &ObsContext,
        timeblock: &Timeblock,
        apply_precession: bool,
        chanblocks: &[Chanblock],
        tile_baseline_flags: &TileBaselineFlags,
    ) {
        let array_position = obs_context.array_position;
        let dut1 = obs_context.dut1.unwrap_or_default();

        let all_fine_chan_lambdas_m = chanblocks
            .iter()
            .map(|c| VEL_C / c.freq)
            .collect::<Vec<_>>();

        let flagged_tiles = &tile_baseline_flags.flagged_tiles;

        let unflagged_tile_xyzs: Vec<XyzGeodetic> = obs_context
            .tile_xyzs
            .par_iter()
            .enumerate()
            .filter(|(tile_index, _)| !flagged_tiles.contains(tile_index))
            .map(|(_, xyz)| *xyz)
            .collect();

        let num_tiles = unflagged_tile_xyzs.len();

        let mut baseline_weights = Vec1::try_from_vec(vec![
            1.0;
            tile_baseline_flags
                .unflagged_cross_baseline_to_tile_map
                .len()
        ])
        .expect("not possible to have no unflagged tiles here");

        let average_timestamp = timeblock.median;

        let (tile_xyzs, lmst) = if apply_precession {
            let precession_info = precess_time(
                array_position.longitude_rad,
                array_position.latitude_rad,
                obs_context.phase_centre,
                average_timestamp,
                dut1,
            );
            let precessed_tile_xyzs = precession_info.precess_xyz(&unflagged_tile_xyzs);
            (precessed_tile_xyzs, precession_info.lmst_j2000)
        } else {
            (
                unflagged_tile_xyzs.clone(),
                get_lmst(array_position.longitude_rad, average_timestamp, dut1),
            )
        };
        let uvws = xyzs_to_cross_uvws(&tile_xyzs, obs_context.phase_centre.to_hadec(lmst));

        assert_eq!(baseline_weights.len(), uvws.len());
        for (UVW { u, v, w }, baseline_weight) in uvws.into_iter().zip(baseline_weights.iter_mut())
        {
            let uvw_length = (u.powi(2) + v.powi(2) + w.powi(2)).sqrt();
            if uvw_length < self.uvw_min_metres || uvw_length > self.uvw_max_metres {
                *baseline_weight = 0.0;
            }
        }

        let timestamps = &timeblock.timestamps;
        let mut tile_uvs_high_res = Array2::<UV>::default((timestamps.len(), num_tiles));
        // Pre-compute high-res tile UVs and Ws at observation phase centre.
        for (&time, mut tile_uvs_high_res) in
            izip!(timestamps.iter(), tile_uvs_high_res.outer_iter_mut(),)
        {
            let (tile_xyzs, lmst) = if apply_precession {
                let precession_info = precess_time(
                    array_position.longitude_rad,
                    array_position.latitude_rad,
                    obs_context.phase_centre,
                    time,
                    dut1,
                );
                let precessed_tile_xyzs = precession_info.precess_xyz(&unflagged_tile_xyzs);
                (precessed_tile_xyzs, precession_info.lmst_j2000)
            } else {
                (
                    unflagged_tile_xyzs.to_owned(),
                    get_lmst(array_position.longitude_rad, time, dut1),
                )
            };
            let hadec_phase = obs_context.phase_centre.to_hadec(lmst);
            let (s_ha, c_ha) = hadec_phase.ha.sin_cos();
            let (s_dec, c_dec) = hadec_phase.dec.sin_cos();
            for (tile_uv, &tile_xyz) in izip!(tile_uvs_high_res.iter_mut(), tile_xyzs.iter(),) {
                let uvw = UVW::from_xyz_inner(tile_xyz, s_ha, c_ha, s_dec, c_dec);
                *tile_uv = UV { u: uvw.u, v: uvw.v };
            }
        }

        // use the baseline taper from the RTS, 1-exp(-(u*u+v*v)/(2*sig^2));
        let iono_taper = get_weights_rts(
            tile_uvs_high_res.view(),
            &all_fine_chan_lambdas_m,
            self.short_baseline_sigma,
        );
        vis_weights_tfb *= &iono_taper;
        assert_eq!(baseline_weights.len(), vis_weights_tfb.len_of(Axis(2)));
        for (vis_weight, bl_weight) in vis_weights_tfb
            .iter_mut()
            .zip(baseline_weights.iter().cycle())
        {
            *vis_weight = (*vis_weight as f64 * *bl_weight) as f32;
        }
    }
}

/// parameters relating to peel loop control
pub(crate) struct PeelLoopParams {
    /// number of outer loops over all sources
    pub(crate) num_passes: NonZeroUsize,
    /// number of loops per source
    pub(crate) num_loops: NonZeroUsize,
    /// convergence factor, determines how fast the loop converges
    pub(crate) convergence: f64,
}

impl PeelLoopParams {
    pub(crate) fn get(&self) -> (usize, usize, f64) {
        (
            self.num_passes.get(),
            self.num_loops.get(),
            self.convergence,
        )
    }
}

#[cfg(any(feature = "cuda", feature = "hip"))]
mod gpu;
#[cfg(any(feature = "cuda", feature = "hip"))]
use gpu::peel_gpu;

pub(crate) struct PeelParams {
    pub(crate) input_vis_params: InputVisParams,
    pub(crate) output_vis_params: Option<OutputVisParams>,
    pub(crate) iono_outputs: Vec<PathBuf>,
    pub(crate) di_per_source_dir: Option<PathBuf>,
    pub(crate) beam: Box<dyn Beam>,
    pub(crate) source_list: SourceList,
    pub(crate) modelling_params: ModellingParams,
    pub(crate) iono_timeblocks: Vec1<Timeblock>,
    pub(crate) iono_time_average_factor: NonZeroUsize,
    pub(crate) low_res_spw: Spw,
    pub(crate) peel_weight_params: PeelWeightParams,
    pub(crate) peel_loop_params: PeelLoopParams,
    pub(crate) num_sources_to_iono_subtract: usize,
    pub(crate) num_sources_to_peel: usize,
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
            peel_weight_params,
            peel_loop_params,
            num_sources_to_iono_subtract,
            ..
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
        let (_low_res_freqs_hz, low_res_lambdas_m): (Vec<_>, Vec<_>) = low_res_spw
            .chanblocks
            .iter()
            .map(|c| {
                let f = c.freq;
                (f, VEL_C / f)
            })
            .unzip();

        assert!(all_fine_chan_lambdas_m.len() % low_res_lambdas_m.len() == 0);

        // Finding the Stokes-I-weighted `RADec` of each ionosub source.
        let source_weighted_positions = {
            let mut component_radecs = vec![];
            let mut component_stokes_is = vec![];
            let mut source_weighted_positions = Vec::with_capacity(*num_sources_to_iono_subtract);
            for source in source_list.values().take(*num_sources_to_iono_subtract) {
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
                        self.num_sources_to_peel,
                        peel_loop_params,
                        obs_context,
                        &unflagged_tile_xyzs,
                        peel_weight_params,
                        tile_baseline_flags,
                        &spw.chanblocks,
                        &low_res_lambdas_m,
                        *apply_precession,
                        output_vis_params.as_ref(),
                        self.di_per_source_dir.as_ref(),
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

                    if !iono_outputs.is_empty() || self.di_per_source_dir.is_some() {
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
                        // Also buffer optional per-source DI solutions (single timeblock)
                        let mut per_source_di: Option<Vec<Array3<Jones<f64>>>> = self
                            .di_per_source_dir
                            .as_ref()
                            .map(|_| {
                                (0..*num_sources_to_iono_subtract)
                                    .map(|_| Array3::from_elem((1, 0, 0), Jones::identity()))
                                    .collect::<Vec<_>>()
                            });

                        while let Ok((incoming_iono_consts, incoming_di)) = rx_iono_consts.recv() {
                            incoming_iono_consts
                                .into_iter()
                                .zip_eq(output_iono_consts.iter_mut())
                                .for_each(|(iono_consts, (_src_name, src_iono_consts))| {
                                    src_iono_consts.alphas.push(iono_consts.alpha);
                                    src_iono_consts.betas.push(iono_consts.beta);
                                    src_iono_consts.gains.push(iono_consts.gain);
                                });
                            if let (Some(buf), Some(di_vec)) = (per_source_di.as_mut(), incoming_di)
                            {
                                for (i, di) in di_vec.into_iter().enumerate() {
                                    // replace storage with incoming di if non-empty
                                    if di.len() > 0 {
                                        if i < buf.len() {
                                            buf[i] = di;
                                        }
                                    }
                                }
                            }
                        }

                        // Channel done; write output(s).
                        if !iono_outputs.is_empty() {
                            let output_json_string =
                                serde_json::to_string_pretty(&output_iono_consts).unwrap();
                            for iono_output in iono_outputs {
                                let mut file = std::fs::File::create(iono_output)?;
                                file.write_all(output_json_string.as_bytes())?;
                            }
                        }

                        // Write per-source DI solutions if requested and available.
                        if let (Some(dir), Some(per_src)) =
                            (self.di_per_source_dir.as_ref(), per_source_di.as_ref())
                        {
                            std::fs::create_dir_all(dir)?;
                            for (((name, _src), _src_iono), di) in source_list
                                .iter()
                                .take(*num_sources_to_iono_subtract)
                                .zip(output_iono_consts.iter())
                                .zip(per_src.iter())
                            {
                                // Only write if di has correct shape
                                if di.len_of(Axis(0)) == 1 && di.len_of(Axis(1)) > 0 && di.len_of(Axis(2)) > 0 {
                                    let mut sols = crate::solutions::CalibrationSolutions::default();
                                    sols.di_jones = di.clone();
                                    let mut path = dir.clone();
                                    let fname = format!("{}.fits", name.replace('/', "_"));
                                    path.push(fname);
                                    let _ = sols.write_solutions_from_ext(&path);
                                }
                            }
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
fn vis_average_tfb(
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
fn vis_rotate_tfb(
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
    lambdas_m: &[f64],
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
                .zip_eq(lambdas_m.iter())
                .for_each(|((jones, jones_rot), lambda_m)| {
                    let rotation = Complex::cis(arg / *lambda_m);
                    *jones_rot = Jones::<f32>::from(Jones::<f64>::from(*jones) * rotation);
                });
        });
}

/// Rotate the supplied visibilities (3D: time, freq, bl) according to the `λ²` constants of
/// proportionality with `exp(-2πi(αu+βv)λ²)`.
fn apply_iono_tfb(
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
/// this is useful when vis_model has already been subtracted from vis_residual
fn unpeel_model(
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
    mut resid_hi_obs_tfb: ArrayViewMut3<Jones<f32>>,
    vis_weights_tfb: ArrayView3<f32>,
    timeblock: &Timeblock,
    source_list: &SourceList,
    iono_consts: &mut [IonoConsts],
    source_weighted_positions: &[RADec],
    peel_loop_params: &PeelLoopParams,
    chanblocks: &[Chanblock],
    low_res_lambdas_m: &[f64],
    obs_context: &ObsContext,
    tile_baseline_flags: &TileBaselineFlags,
    high_res_modeller: &mut dyn SkyModeller,
    no_precession: bool,
    num_sources_to_peel: usize,
    di_per_source_dir: Option<&std::path::PathBuf>,
    multi_progress_bar: &MultiProgress,
) -> Result<(), PeelError> {
    // TODO: Do we allow multiple timesteps in the low-res data?
    let (num_loops, num_passes, convergence) = peel_loop_params.get();

    let all_fine_chan_lambdas_m = chanblocks
        .iter()
        .map(|c| VEL_C / c.freq)
        .collect::<Vec<_>>();

    let flagged_tiles = &tile_baseline_flags.flagged_tiles;
    let unflagged_tile_xyzs: Vec<XyzGeodetic> = obs_context
        .tile_xyzs
        .par_iter()
        .enumerate()
        .filter(|(tile_index, _)| !flagged_tiles.contains(tile_index))
        .map(|(_, xyz)| *xyz)
        .collect();

    let array_position = obs_context.array_position;
    let dut1 = obs_context.dut1.unwrap_or_default();

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
    assert_eq!(resid_hi_obs_tfb.len_of(time_axis), num_timestamps_high_res);
    assert_eq!(vis_weights_tfb.len_of(time_axis), num_timestamps_high_res);
    assert_eq!(resid_hi_obs_tfb.len_of(baseline_axis), num_cross_baselines);
    assert_eq!(vis_weights_tfb.len_of(baseline_axis), num_cross_baselines);
    assert_eq!(resid_hi_obs_tfb.len_of(freq_axis), num_freqs_high_res);
    assert_eq!(vis_weights_tfb.len_of(freq_axis), num_freqs_high_res);

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
    let mut tile_uvs_hi_obs = Array2::<UV>::default((timestamps.len(), num_tiles));
    let mut tile_ws_hi_obs = Array2::<W>::default((timestamps.len(), num_tiles));
    // source phase center
    let mut tile_uvs_hi_src = tile_uvs_hi_obs.clone();
    let mut tile_ws_hi_src = tile_ws_hi_obs.clone();
    let mut tile_uvs_lo_src = Array2::<UV>::default((num_timestamps_low_res, num_tiles));

    // Pre-compute high-res tile UVs and Ws at observation phase centre.
    for (&time, mut tile_uvs, mut tile_ws) in izip!(
        timestamps.iter(),
        tile_uvs_hi_obs.outer_iter_mut(),
        tile_ws_hi_obs.outer_iter_mut(),
    ) {
        let (lmst, precessed_xyzs) = if !no_precession {
            let precession_info = precess_time(
                array_position.longitude_rad,
                array_position.latitude_rad,
                obs_context.phase_centre,
                time,
                dut1,
            );
            let precessed_xyzs = precession_info.precess_xyz(&unflagged_tile_xyzs);
            (precession_info.lmst_j2000, precessed_xyzs)
        } else {
            let lmst = get_lmst(array_position.longitude_rad, time, dut1);
            (lmst, unflagged_tile_xyzs.clone())
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
            ArrayView2::from_shape((1, num_tiles), &unflagged_tile_xyzs).expect("correct shape");
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
            average_precession_info.precess_xyz(&unflagged_tile_xyzs),
        )
        .expect("correct shape");

        (
            average_precession_info.lmst_j2000,
            average_precession_info.array_latitude_j2000,
            CowArray::from(average_precessed_tile_xyzs),
        )
    };

    // Temporary visibility array, re-used for each timestep
    let mut resid_hi_src_tfb = resid_hi_obs_tfb.to_owned();
    let high_res_vis_dims = resid_hi_obs_tfb.dim();
    let mut model_hi_obs_tfb = Array3::default(high_res_vis_dims);
    let mut model_hi_src_tfb = Array3::default(high_res_vis_dims);
    let mut model_hi_src_iono_tfb = Array3::default(high_res_vis_dims);

    // temporary arrays for accumulation
    let mut resid_lo_src_tfb: Array3<Jones<f32>> = Array3::zeros((
        num_timestamps_low_res,
        num_freqs_low_res,
        num_cross_baselines,
    ));
    let mut model_lo_src_iono_tfb = resid_lo_src_tfb.clone();
    let mut weights_lo: Array3<f32> = Array3::zeros(resid_lo_src_tfb.dim());

    // The low-res weights only need to be populated once.
    weights_average(vis_weights_tfb.view(), weights_lo.view_mut());

    for pass in 0..num_passes {
        for (i_source, (((source_name, source), iono_consts), source_pos)) in source_list
            .iter()
            .take(num_sources_to_iono_subtract)
            .zip_eq(iono_consts.iter_mut())
            .zip_eq(source_weighted_positions.iter().copied())
            .enumerate()
        {
            multi_progress_bar.suspend(|| {
                debug!("peel loop {pass}: {source_name} at {source_pos} (has iono {iono_consts:?})")
            });
            let start = std::time::Instant::now();
            let old_iono_consts = *iono_consts;

            high_res_modeller.update_with_a_source(source, obs_context.phase_centre)?;
            // high_res_modeller.update_with_a_source(source, source_pos)?;
            // this is only necessary for cpu modeller.
            model_hi_obs_tfb.fill(Jones::zero());

            multi_progress_bar.suspend(|| trace!("{:?}: initialise modellers", start.elapsed()));
            // iterate along high res times:
            // - calculate high-res uvws in source phase centre
            // - model high res visibilities in source phase centre
            // - calculate low-res uvws in source phase centre
            // iterate along high res times
            for (&time, mut model_hi_obs_fb, mut tile_uvs_src, mut tile_ws_src) in izip!(
                timestamps,
                model_hi_obs_tfb.outer_iter_mut(),
                tile_uvs_hi_src.outer_iter_mut(),
                tile_ws_hi_src.outer_iter_mut(),
            ) {
                let (lmst, precessed_xyzs) = if !no_precession {
                    let precession_info = precess_time(
                        array_position.longitude_rad,
                        array_position.latitude_rad,
                        obs_context.phase_centre,
                        time,
                        dut1,
                    );
                    let precessed_xyzs = precession_info.precess_xyz(&unflagged_tile_xyzs);
                    (precession_info.lmst_j2000, precessed_xyzs)
                } else {
                    let lmst = get_lmst(array_position.longitude_rad, time, dut1);
                    (lmst, unflagged_tile_xyzs.clone())
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
                tile_uvs_lo_src.as_slice_mut().unwrap(),
                average_tile_xyzs.as_slice().unwrap(),
                hadec_source,
            );

            // rotate residuals to source phase centre

            multi_progress_bar
                .suspend(|| trace!("{:?}: high-res residual rotate", start.elapsed()));
            vis_rotate_tfb(
                resid_hi_obs_tfb.view(),
                resid_hi_src_tfb.view_mut(),
                tile_ws_hi_obs.view(),
                tile_ws_hi_src.view(),
                &all_fine_chan_lambdas_m,
            );

            multi_progress_bar.suspend(|| trace!("{:?}: high-res model rotate", start.elapsed()));
            // TODO: just model in src pc
            vis_rotate_tfb(
                model_hi_obs_tfb.view(),
                model_hi_src_tfb.view_mut(),
                tile_ws_hi_obs.view(),
                tile_ws_hi_src.view(),
                &all_fine_chan_lambdas_m,
            );

            multi_progress_bar.suspend(|| trace!("{:?}: high-res model iono", start.elapsed()));
            apply_iono_tfb(
                model_hi_src_tfb.view(),
                model_hi_src_iono_tfb.view_mut(),
                tile_uvs_hi_src.view(),
                *iono_consts,
                &all_fine_chan_lambdas_m,
            );

            // Add the high-res model to the residuals.
            multi_progress_bar.suspend(|| trace!("{:?}: add low-res model", start.elapsed()));
            Zip::from(&mut resid_hi_src_tfb)
                .and(&model_hi_src_iono_tfb)
                .for_each(|r, m| {
                    *r += *m;
                });

            multi_progress_bar.suspend(|| trace!("{:?}: vis_average", start.elapsed()));
            vis_average_tfb(
                resid_hi_src_tfb.view(),
                resid_lo_src_tfb.view_mut(),
                vis_weights_tfb.view(),
            );

            multi_progress_bar.suspend(|| trace!("{:?}: alpha/beta loop", start.elapsed()));
            // let mut gain_update = 1.0;
            let mut iteration = 0;
            while iteration != num_loops {
                iteration += 1;
                multi_progress_bar.suspend(|| debug!("iter {iteration}, consts: {iono_consts:?}"));

                // iono rotate model using existing iono consts
                apply_iono_tfb(
                    model_hi_src_tfb.view(),
                    model_hi_src_iono_tfb.view_mut(),
                    tile_uvs_hi_src.view(),
                    *iono_consts,
                    &all_fine_chan_lambdas_m,
                );

                vis_average_tfb(
                    model_hi_src_iono_tfb.view(),
                    model_lo_src_iono_tfb.view_mut(),
                    vis_weights_tfb.view(),
                );

                let iono_fits = iono_fit(
                    resid_lo_src_tfb.view(),
                    weights_lo.view(),
                    model_lo_src_iono_tfb.view(),
                    low_res_lambdas_m,
                    tile_uvs_lo_src.view(),
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

            multi_progress_bar.suspend(|| trace!("{:?}: unpeel_model", start.elapsed()));
            // add the model to residual, and subtract the iono rotated model
            unpeel_model(
                model_hi_obs_tfb.view(),
                resid_hi_obs_tfb.view_mut(),
                tile_uvs_hi_src.view(),
                *iono_consts,
                old_iono_consts,
                &all_fine_chan_lambdas_m,
            );

            // If peeling requested for this source (last pass), do DI cal and subtract
            if pass == num_passes - 1 && num_sources_to_peel > 0 {
                if i_source < num_sources_to_peel {
                    // Re-rotate updated residuals to source phase centre
                    vis_rotate_tfb(
                        resid_hi_obs_tfb.view(),
                        resid_hi_src_tfb.view_mut(),
                        tile_ws_hi_obs.view(),
                        tile_ws_hi_src.view(),
                        &all_fine_chan_lambdas_m,
                    );
                    // Build model at source phase centre with iono applied (already in model_hi_src_iono_tfb)
                    // Run DI calibration comparing resid_hi_src_tfb vs model_hi_src_iono_tfb in source phase centre
                    let mut di_jones =
                        Array3::from_elem((1, num_tiles, num_freqs_high_res), Jones::identity());
                    let shape = (
                        num_timestamps_high_res,
                        num_freqs_high_res,
                        num_cross_baselines,
                    );
                    let pb = ProgressBar::hidden();
                    // Build a local timeblock with index 0 and full range
                    let cal_tb = Timeblock {
                        index: 0,
                        range: 0..num_timestamps_high_res,
                        timestamps: timeblock.timestamps.clone(),
                        timesteps: timeblock.timesteps.clone(),
                        median: timeblock.median,
                    };
                    let results = crate::di_calibrate::calibrate_timeblock(
                        ArrayView3::from_shape(shape, resid_hi_src_tfb.as_slice().unwrap())
                            .expect("correct shape"),
                        ArrayView3::from_shape(shape, model_hi_src_iono_tfb.as_slice().unwrap())
                            .expect("correct shape"),
                        di_jones.view_mut(),
                        &cal_tb,
                        chanblocks,
                        50,
                        1e-8,
                        1e-4,
                        obs_context.polarisations,
                        pb,
                        true,
                    );
                    if results.iter().all(|r| r.converged) {
                        // Subtract DI-calibrated model from residuals at source phase centre
                        // resid_hi_src_tfb -= J_i * model_hi_src_tfb * J_j^H (per baseline)
                        resid_hi_src_tfb
                            .outer_iter_mut()
                            .zip(model_hi_src_tfb.outer_iter())
                            .for_each(|(mut res_fb, model_fb)| {
                                // di_jones shape: (1, tiles, chans) -> take [0, .., ..]
                                let j_tiles = di_jones.index_axis(Axis(0), 0);
                                let mut i_tile1 = 0usize;
                                let mut i_tile2 = 1usize;
                                res_fb
                                    .axis_iter_mut(Axis(1))
                                    .zip(model_fb.axis_iter(Axis(1)))
                                    .for_each(|(mut res_f, model_f)| {
                                        let j1 = j_tiles.index_axis(Axis(0), i_tile1);
                                        let j2 = j_tiles.index_axis(Axis(0), i_tile2);
                                        res_f.iter_mut().zip(model_f.iter()).enumerate().for_each(
                                            |(i_chan, (res, m))| {
                                                let g1 = j1[i_chan];
                                                let g2 = j2[i_chan];
                                                let m64: Jones<f64> = Jones::from(*m);
                                                let sub: Jones<f64> = (g1 * m64) * g2.h();
                                                let mut r64: Jones<f64> = Jones::from(*res);
                                                r64 -= sub;
                                                *res = Jones::from(r64);
                                            },
                                        );
                                        // advance baseline tile indices
                                        i_tile2 += 1;
                                        if i_tile2 == num_tiles {
                                            i_tile1 += 1;
                                            i_tile2 = i_tile1 + 1;
                                        }
                                    });
                            });
                        // Rotate residuals back to observation phase centre for subsequent sources
                        vis_rotate_tfb(
                            resid_hi_src_tfb.view(),
                            resid_hi_obs_tfb.view_mut(),
                            tile_ws_hi_src.view(),
                            tile_ws_hi_obs.view(),
                            &all_fine_chan_lambdas_m,
                        );

                        // Optionally write per-source DI solutions
                        if let Some(dir) = di_per_source_dir {
                            std::fs::create_dir_all(dir)?;
                            let mut sols = crate::solutions::CalibrationSolutions::default();
                            // One timeblock; take di_jones for current source
                            let _num_tiles = di_jones.len_of(Axis(1));
                            let _num_chans = di_jones.len_of(Axis(2));
                            sols.di_jones = di_jones.clone();
                            let mut path = dir.clone();
                            let fname = format!("{}.fits", source_name.replace('/', "_"));
                            path.push(fname);
                            let _ = sols.write_solutions_from_ext(&path);
                        }
                    }
                }
            }

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

// type for passing full residuals from joiner thread
type FullResidual<'a> = (Array3<Jones<f32>>, Array3<f32>, &'a Timeblock);

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
    tx_full_residual: Sender<FullResidual<'a>>,
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

#[cfg(any(feature = "cuda", feature = "hip"))]
use crate::model::SkyModellerGpu;
// use SkyModellerGpu

#[allow(clippy::too_many_arguments)]
fn peel_thread(
    beam: &dyn Beam,
    source_list: &SourceList,
    source_weighted_positions: &[RADec],
    num_sources_to_iono_subtract: usize,
    num_sources_to_peel: usize,
    peel_loop_params: &PeelLoopParams,
    obs_context: &ObsContext,
    unflagged_tile_xyzs: &[XyzGeodetic],
    peel_weight_params: &PeelWeightParams,
    tile_baseline_flags: &TileBaselineFlags,
    chanblocks: &[Chanblock],
    low_res_lambdas_m: &[f64],
    apply_precession: bool,
    output_vis_params: Option<&OutputVisParams>,
    di_per_source_dir: Option<&PathBuf>,
    rx_full_residual: Receiver<FullResidual>,
    tx_write: Sender<VisTimestep>,
    // Send iono constants and optional per-source DI solutions (shape 1xTilesxChans)
    tx_iono_consts: Sender<(Vec<IonoConsts>, Option<Vec<Array3<Jones<f64>>>>)>,
    error: &AtomicCell<bool>,
    multi_progress: &MultiProgress,
    overall_peel_progress: &ProgressBar,
) -> Result<(), PeelError> {
    let array_position = obs_context.array_position;
    let dut1 = obs_context.dut1.unwrap_or_default();

    for (i, (mut vis_residual_tfb, vis_weights_tfb, timeblock)) in
        rx_full_residual.iter().enumerate()
    {
        // Should we continue?
        if error.load() {
            return Ok(());
        }

        // /////// //
        // WEIGHTS //
        // /////// //

        // copy weights to a new array for tapering, but keep originals for writing
        let mut tapered_weights_tfb = vis_weights_tfb.clone();
        peel_weight_params.apply_tfb(
            tapered_weights_tfb.view_mut(),
            obs_context,
            timeblock,
            apply_precession,
            chanblocks,
            tile_baseline_flags,
        );

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
                    tapered_weights_tfb.view(),
                    timeblock,
                    source_list,
                    &mut iono_consts,
                    source_weighted_positions,
                    peel_loop_params,
                    chanblocks,
                    low_res_lambdas_m,
                    obs_context,
                    tile_baseline_flags,
                    &mut high_res_modeller,
                    !apply_precession,
                    num_sources_to_peel,
                    di_per_source_dir,
                    multi_progress,
                )?;
            }

            let _sent_iono = false;
            #[cfg(any(feature = "cuda", feature = "hip"))]
            {
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

                    // Prepare optional DI storage when requested, one timeblock per source
                    let mut di_storage_opt: Option<Vec<Array3<Jones<f64>>>> = if di_per_source_dir
                        .is_some()
                        && num_sources_to_peel > 0
                    {
                        Some(
                            (0..num_sources_to_iono_subtract)
                                .map(|_| {
                                    Array3::from_elem((1, unflagged_tile_xyzs.len(), chanblocks.len()), Jones::identity())
                                })
                                .collect(),
                        )
                    } else {
                        None
                    };

                    peel_gpu(
                        vis_residual_tfb.view_mut(),
                        tapered_weights_tfb.view(),
                        timeblock,
                        source_list,
                        &mut iono_consts,
                        source_weighted_positions,
                        peel_loop_params,
                        chanblocks,
                        low_res_lambdas_m,
                        obs_context,
                        tile_baseline_flags,
                        &mut high_res_modeller,
                        !apply_precession,
                        num_sources_to_peel,
                        multi_progress,
                        di_storage_opt.as_mut().map(|v| v.as_mut_slice()),
                    )?;

                    // Send iono consts and optional DI solutions
                    let di_payload = di_storage_opt.take();
                    match tx_iono_consts.send((iono_consts.clone(), di_payload)) {
                        Ok(()) => (),
                        Err(_) => return Ok(()),
                    }
                    sent_iono = true;
                }
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

        #[allow(unused_variables)]
        if !{
            #[cfg(any(feature = "cuda", feature = "hip"))]
            { sent_iono }
            #[cfg(not(any(feature = "cuda", feature = "hip")))]
            { false }
        } {
            match tx_iono_consts.send((iono_consts, None)) {
                Ok(()) => (),
                Err(_) => return Ok(()),
            }
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
