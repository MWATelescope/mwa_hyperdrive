// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Generate sky-model visibilities from a sky-model source list.

use std::{
    collections::HashSet,
    num::NonZeroUsize,
    thread::{self, ScopedJoinHandle},
};

use crossbeam_channel::{bounded, Sender};
use crossbeam_utils::atomic::AtomicCell;
use hifitime::{Duration, Epoch};
use indicatif::{MultiProgress, ProgressBar, ProgressDrawTarget, ProgressStyle};
use log::info;
use marlu::{
    constants::{FREQ_WEIGHT_FACTOR, TIME_WEIGHT_FACTOR},
    LatLngHeight, MwaObsContext, RADec, XyzGeodetic,
};
use mwalib::MetafitsContext;
use ndarray::ArcArray2;
use scopeguard::defer_on_unwind;
use thiserror::Error;
use vec1::Vec1;

use crate::{
    averaging::channels_to_chanblocks,
    beam::Beam,
    context::Polarisations,
    io::write::{write_vis, VisTimestep, VisWriteError},
    math::TileBaselineFlags,
    model::{self, ModelError},
    params::{ModellingParams, OutputVisParams},
    srclist::SourceList,
    PROGRESS_BARS,
};

/// Parameters needed to do sky-model visibility simulation.
pub(crate) struct VisSimulateParams {
    /// Sky-model source list.
    pub(crate) source_list: SourceList,

    /// mwalib metafits context
    pub(crate) metafits: MetafitsContext,

    /// The output visibility files.
    pub(crate) output_vis_params: OutputVisParams,

    /// The phase centre.
    pub(crate) phase_centre: RADec,

    /// The fine channel frequencies \[Hz\].
    pub(crate) fine_chan_freqs: Vec1<f64>,

    /// The frequency resolution of the fine channels.
    pub(crate) freq_res_hz: f64,

    /// The [`XyzGeodetic`] positions of the tiles.
    pub(crate) tile_xyzs: Vec<XyzGeodetic>,

    /// The names of the tiles.
    pub(crate) tile_names: Vec<String>,

    /// Information on flagged tiles, baselines and mapping between indices.
    pub(crate) tile_baseline_flags: TileBaselineFlags,

    /// Timestamps to be simulated.
    pub(crate) timestamps: Vec1<Epoch>,

    pub(crate) time_res: Duration,

    /// Interface to beam code.
    pub(crate) beam: Box<dyn Beam>,

    /// The Earth position of the interferometer.
    pub(crate) array_position: LatLngHeight,

    /// UT1 - UTC.
    pub(crate) dut1: Duration,

    /// Should we be precessing?
    pub(crate) modelling_params: ModellingParams,
}

impl VisSimulateParams {
    pub(crate) fn run(&self) -> Result<(), VisSimulateError> {
        let VisSimulateParams {
            source_list,
            metafits,
            output_vis_params:
                OutputVisParams {
                    output_files,
                    output_time_average_factor,
                    output_freq_average_factor,
                    output_autos,
                    output_timeblocks,
                    write_smallest_contiguous_band,
                },
            phase_centre,
            fine_chan_freqs,
            freq_res_hz,
            tile_xyzs,
            tile_names,
            tile_baseline_flags,
            timestamps,
            time_res,
            beam,
            array_position,
            dut1,
            modelling_params: ModellingParams { apply_precession },
        } = self;

        // Channel for writing simulated visibilities.
        let (tx_model, rx_model) = bounded(5);

        // Progress bar.
        let multi_progress = MultiProgress::with_draw_target(if PROGRESS_BARS.load() {
            ProgressDrawTarget::stdout()
        } else {
            ProgressDrawTarget::hidden()
        });
        let model_progress = multi_progress.add(
            ProgressBar::new(timestamps.len() as _)
                    .with_style(
                        ProgressStyle::default_bar()
                            .template("{msg:17}: [{wide_bar:.blue}] {pos:2}/{len:2} timesteps ({elapsed_precise}<{eta_precise})").unwrap()
                            .progress_chars("=> "),
                    )
                    .with_position(0)
                    .with_message("Sky modelling"),
        );
        let write_progress = multi_progress.add(
            ProgressBar::new(output_timeblocks.len() as _)
                    .with_style(
                        ProgressStyle::default_bar()
                            .template("{msg:17}: [{wide_bar:.blue}] {pos:2}/{len:2} timeblocks ({elapsed_precise}<{eta_precise})").unwrap()
                            .progress_chars("=> "),
                    )
                    .with_position(0)
                    .with_message("Model writing"),
        );

        // Generate the visibilities and write them out asynchronously.
        let error = AtomicCell::new(false);
        let scoped_threads_result: Result<String, VisSimulateError> = thread::scope(|scope| {
            // Modelling thread.
            let model_handle: ScopedJoinHandle<Result<(), ModelError>> = thread::Builder::new()
                .name("model".to_string())
                .spawn_scoped(scope, || {
                    defer_on_unwind! { error.store(true); }
                    model_progress.tick();

                    let weight_factor = (freq_res_hz / FREQ_WEIGHT_FACTOR)
                        * (time_res.to_seconds() / TIME_WEIGHT_FACTOR);
                    let result = model_thread(
                        &**beam,
                        source_list,
                        tile_xyzs,
                        tile_baseline_flags,
                        timestamps,
                        fine_chan_freqs,
                        *phase_centre,
                        *array_position,
                        *dut1,
                        *apply_precession,
                        *output_autos,
                        weight_factor,
                        tx_model,
                        &error,
                        model_progress,
                    );
                    if result.is_err() {
                        error.store(true);
                    }
                    result
                })
                .expect("OS can create threads");

            // Writing thread.
            let write_handle: ScopedJoinHandle<Result<String, VisWriteError>> =
                thread::Builder::new()
                    .name("write".to_string())
                    .spawn_scoped(scope, || {
                        defer_on_unwind! { error.store(true); }
                        write_progress.tick();

                        // Form sorted unflagged tile pairs from our
                        // cross-correlation baselines (and maybe autos too).
                        let unflagged_baseline_tile_pairs: Vec<_> = if *output_autos {
                            tile_baseline_flags
                                .get_unflagged_baseline_tile_pairs()
                                .collect()
                        } else {
                            tile_baseline_flags
                                .get_unflagged_cross_baseline_tile_pairs()
                                .collect()
                        };
                        let spw = &channels_to_chanblocks(
                            &fine_chan_freqs.mapped_ref(|f| *f as u64),
                            freq_res_hz.round() as u64,
                            NonZeroUsize::new(1).unwrap(),
                            &HashSet::new(),
                        )[0];
                        let result = write_vis(
                            output_files,
                            *array_position,
                            *phase_centre,
                            None,
                            tile_xyzs,
                            tile_names,
                            Some(metafits.obs_id),
                            output_timeblocks,
                            *time_res,
                            *dut1,
                            spw,
                            &unflagged_baseline_tile_pairs,
                            *output_time_average_factor,
                            *output_freq_average_factor,
                            Some(&MwaObsContext::from_mwalib(metafits)),
                            *write_smallest_contiguous_band,
                            rx_model,
                            &error,
                            Some(write_progress),
                        );
                        if result.is_err() {
                            error.store(true);
                        }
                        result
                    })
                    .expect("OS can create threads");

            // Join all thread handles. This propagates any errors and lets us
            // know if any threads panicked, if panics aren't aborting as per
            // the Cargo.toml. (It would be nice to capture the panic
            // information, if it's possible, but I don't know how, so panics
            // are currently aborting.)
            model_handle.join().unwrap()?;
            let write_message = write_handle.join().unwrap()?;
            Ok(write_message)
        });

        // Propagate errors and print out the write message.
        info!("{}", scoped_threads_result?);

        Ok(())
    }
}

#[allow(clippy::too_many_arguments)]
fn model_thread(
    beam: &dyn Beam,
    source_list: &SourceList,
    unflagged_tile_xyzs: &[XyzGeodetic],
    tile_baseline_flags: &TileBaselineFlags,
    timestamps: &[Epoch],
    fine_chan_freqs: &[f64],
    phase_centre: RADec,
    array_position: LatLngHeight,
    dut1: Duration,
    apply_precession: bool,
    model_autos: bool,
    weight_factor: f64,
    tx: Sender<VisTimestep>,
    error: &AtomicCell<bool>,
    progress_bar: ProgressBar,
) -> Result<(), ModelError> {
    let modeller = model::new_sky_modeller(
        beam,
        source_list,
        Polarisations::XX_XY_YX_YY,
        unflagged_tile_xyzs,
        fine_chan_freqs,
        &tile_baseline_flags.flagged_tiles,
        phase_centre,
        array_position.longitude_rad,
        array_position.latitude_rad,
        dut1,
        apply_precession,
    )?;
    let num_tiles = unflagged_tile_xyzs.len();
    let num_cross_baselines = (num_tiles * (num_tiles - 1)) / 2;
    let cross_vis_shape = (fine_chan_freqs.len(), num_cross_baselines);
    let auto_vis_shape = (fine_chan_freqs.len(), num_tiles);

    for &timestamp in timestamps {
        let mut cross_data_fb = ArcArray2::zeros(cross_vis_shape);
        modeller.model_timestep_with(timestamp, cross_data_fb.view_mut())?;
        let auto_data_fb = if model_autos {
            let mut auto_data_fb = ArcArray2::zeros(auto_vis_shape);
            modeller.model_timestep_autos_with(timestamp, auto_data_fb.view_mut())?;
            Some(auto_data_fb)
        } else {
            None
        };

        // Should we continue?
        if error.load() {
            return Ok(());
        }

        match tx.send(VisTimestep {
            cross_data_fb,
            cross_weights_fb: ArcArray2::from_elem(cross_vis_shape, weight_factor as f32),
            autos: auto_data_fb.map(|d| {
                (
                    d,
                    ArcArray2::from_elem(auto_vis_shape, weight_factor as f32),
                )
            }),
            timestamp,
        }) {
            Ok(()) => (),
            // If we can't send the message, it's because the channel has been
            // closed on the other side. That should only happen because the
            // writer has exited due to error; in that case, just exit this
            // thread.
            Err(_) => return Ok(()),
        }

        progress_bar.inc(1);
    }

    progress_bar.abandon_with_message("Finished generating sky model");
    Ok(())
}

#[derive(Error, Debug)]
pub(crate) enum VisSimulateError {
    #[error(transparent)]
    VisWrite(#[from] crate::io::write::VisWriteError),

    #[error(transparent)]
    Model(#[from] crate::model::ModelError),

    #[error(transparent)]
    IO(#[from] std::io::Error),
}
