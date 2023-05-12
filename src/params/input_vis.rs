// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Parameters for input data. The main struct here ([`InputVisParams`])
//! includes a method (`read_timeblock`) for reading in averaged and/or
//! calibrated visibilities, depending on whether averaging is requested and
//! whether calibration solutions have been supplied. Other info like
//! chanblocks, tile and channel flags etc. also live here.

use std::collections::HashSet;

use crossbeam_utils::atomic::AtomicCell;
use hifitime::{Duration, Epoch};
use itertools::{izip, Itertools};
use log::debug;
use marlu::Jones;
use ndarray::prelude::*;
use vec1::Vec1;

use crate::{
    averaging::{vis_average, Spw, Timeblock},
    context::ObsContext,
    io::read::{VisRead, VisReadError},
    math::TileBaselineFlags,
    CalibrationSolutions,
};

pub(crate) struct InputVisParams {
    /// The object to read visibility data.
    pub(crate) vis_reader: Box<dyn VisRead>,

    /// Calibration solutions. If available, these are automatically applied
    /// when `InputVisParams::read_timeblock` is called.
    pub(crate) solutions: Option<CalibrationSolutions>,

    /// The timeblocks to be used from the averaged data. If there is no
    /// averaging to be done, then these are the same as the timesteps to be
    /// read from the data.
    pub(crate) timeblocks: Vec1<Timeblock>,

    /// The time resolution of the data *after* averaging (i.e. when using
    /// `InputVisParams::read_timeblock`).
    pub(crate) time_res: Duration,

    /// Channel and frequency information. Note that this is a single contiguous
    /// spectral window, not multiple spectral windows (a.k.a. picket fence).
    pub(crate) spw: Spw,

    pub(crate) tile_baseline_flags: TileBaselineFlags,

    /// Are autocorrelations to be read?
    pub(crate) using_autos: bool,

    /// Are we ignoring weights?
    pub(crate) ignore_weights: bool,

    /// The UT1 - UTC offset. If this is 0, effectively UT1 == UTC, which is a
    /// wrong assumption by up to 0.9s. We assume the this value does not change
    /// over the timestamps used in this [`InputVisParams`].
    ///
    /// Note that this need not be the same DUT1 in the input data's
    /// [`ObsContext`]; the user may choose to suppress that DUT1 or supply
    /// their own.
    pub(crate) dut1: Duration,
}

impl InputVisParams {
    pub(crate) fn get_obs_context(&self) -> &ObsContext {
        self.vis_reader.get_obs_context()
    }

    pub(crate) fn get_total_num_tiles(&self) -> usize {
        self.get_obs_context().get_total_num_tiles()
    }

    pub(crate) fn get_num_unflagged_tiles(&self) -> usize {
        self.tile_baseline_flags
            .unflagged_auto_index_to_tile_map
            .len()
    }

    /// Read the cross-correlation visibilities out of the input data, averaged
    /// to the target resolution. If calibration solutions were supplied, then
    /// these are applied before averaging.
    pub(crate) fn read_timeblock(
        &self,
        timeblock: &Timeblock,
        mut cross_data_fb: ArrayViewMut2<Jones<f32>>,
        mut cross_weights_fb: ArrayViewMut2<f32>,
        mut autos_fb: Option<(ArrayViewMut2<Jones<f32>>, ArrayViewMut2<f32>)>,
        error: &AtomicCell<bool>,
    ) -> Result<(), VisReadError> {
        let obs_context = self.get_obs_context();
        let num_unflagged_tiles = self.get_num_unflagged_tiles();
        let num_unflagged_cross_baselines = (num_unflagged_tiles * (num_unflagged_tiles - 1)) / 2;
        let avg_cross_vis_shape = (self.spw.chanblocks.len(), num_unflagged_cross_baselines);
        let avg_auto_vis_shape = (self.spw.chanblocks.len(), num_unflagged_tiles);
        assert_eq!(cross_data_fb.dim(), avg_cross_vis_shape);
        assert_eq!(cross_weights_fb.dim(), avg_cross_vis_shape);
        if let Some((auto_data_fb, auto_weights_fb)) = autos_fb.as_ref() {
            assert_eq!(auto_data_fb.dim(), avg_auto_vis_shape);
            assert_eq!(auto_weights_fb.dim(), avg_auto_vis_shape);
        }

        let averaging = timeblock.timestamps.len() > 1 || self.spw.chans_per_chanblock.get() > 1;

        if averaging {
            let cross_vis_shape = (
                timeblock.timestamps.len(),
                obs_context.fine_chan_freqs.len(),
                num_unflagged_cross_baselines,
            );
            let mut unaveraged_cross_data_tfb = Array3::zeros(cross_vis_shape);
            let mut unaveraged_cross_weights_tfb = Array3::zeros(cross_vis_shape);
            // If the user has supplied arrays for autos and the input data has
            // autos, read those out.
            let mut unaveraged_autos =
                match (autos_fb.as_ref(), obs_context.autocorrelations_present) {
                    (Some(_), true) => {
                        let auto_vis_shape = (
                            timeblock.timestamps.len(),
                            obs_context.fine_chan_freqs.len(),
                            num_unflagged_tiles,
                        );
                        let unaveraged_auto_data_tfb = Array3::zeros(auto_vis_shape);
                        let unaveraged_auto_weights_tfb = Array3::zeros(auto_vis_shape);
                        Some((unaveraged_auto_data_tfb, unaveraged_auto_weights_tfb))
                    }

                    _ => None,
                };

            if let Some((unaveraged_auto_data_tfb, unaveraged_auto_weights_tfb)) =
                unaveraged_autos.as_mut()
            {
                for (
                    &timestamp,
                    &timestep,
                    unaveraged_cross_data_fb,
                    unaveraged_cross_weights_fb,
                    unaveraged_auto_data_fb,
                    unaveraged_auto_weights_fb,
                ) in izip!(
                    timeblock.timestamps.iter(),
                    timeblock.timesteps.iter(),
                    unaveraged_cross_data_tfb.outer_iter_mut(),
                    unaveraged_cross_weights_tfb.outer_iter_mut(),
                    unaveraged_auto_data_tfb.outer_iter_mut(),
                    unaveraged_auto_weights_tfb.outer_iter_mut()
                ) {
                    debug!("Reading timestamp {}", timestamp.to_gpst_seconds());

                    self.read_timestep(
                        timestep,
                        unaveraged_cross_data_fb,
                        unaveraged_cross_weights_fb,
                        Some((unaveraged_auto_data_fb, unaveraged_auto_weights_fb)),
                        &HashSet::new(),
                    )?;

                    // Should we continue?
                    if error.load() {
                        return Ok(());
                    }
                }
            } else {
                for (
                    &timestamp,
                    &timestep,
                    unaveraged_cross_data_fb,
                    unaveraged_cross_weights_fb,
                ) in izip!(
                    timeblock.timestamps.iter(),
                    timeblock.timesteps.iter(),
                    unaveraged_cross_data_tfb.outer_iter_mut(),
                    unaveraged_cross_weights_tfb.outer_iter_mut()
                ) {
                    debug!("Reading timestamp {}", timestamp.to_gpst_seconds());

                    self.read_timestep(
                        timestep,
                        unaveraged_cross_data_fb,
                        unaveraged_cross_weights_fb,
                        None,
                        &HashSet::new(),
                    )?;

                    // Should we continue?
                    if error.load() {
                        return Ok(());
                    }
                }
            };

            // Apply flagged channels.
            for i_chan in &self.spw.flagged_chan_indices {
                let i_chan = usize::from(*i_chan);
                unaveraged_cross_weights_tfb
                    .slice_mut(s![.., i_chan, ..])
                    .mapv_inplace(|w| -w.abs());
                unaveraged_cross_weights_tfb
                    .slice_mut(s![.., i_chan, ..])
                    .mapv_inplace(|w| -w.abs());
                if let Some((_, unaveraged_auto_weights_tfb)) = unaveraged_autos.as_mut() {
                    unaveraged_auto_weights_tfb
                        .slice_mut(s![.., i_chan, ..])
                        .mapv_inplace(|w| -w.abs());
                }
            }

            // We've now read in all of the timesteps for this timeblock. If
            // there are calibration solutions, these now need to be applied.
            if self.solutions.is_some() {
                debug!(
                    "Applying calibration solutions to input data from timeblock {}",
                    timeblock.index
                );

                let chan_freqs = obs_context.fine_chan_freqs.mapped_ref(|f| *f as f64);
                if let Some((unaveraged_auto_data_tfb, unaveraged_auto_weights_tfb)) =
                    unaveraged_autos.as_mut()
                {
                    for (
                        &timestamp,
                        cross_data_fb,
                        cross_weights_fb,
                        auto_data_fb,
                        auto_weights_fb,
                    ) in izip!(
                        timeblock.timestamps.iter(),
                        unaveraged_cross_data_tfb.outer_iter_mut(),
                        unaveraged_cross_weights_tfb.outer_iter_mut(),
                        unaveraged_auto_data_tfb.outer_iter_mut(),
                        unaveraged_auto_weights_tfb.outer_iter_mut()
                    ) {
                        self.apply_solutions(
                            timestamp,
                            cross_data_fb,
                            cross_weights_fb,
                            Some((auto_data_fb, auto_weights_fb)),
                            &chan_freqs,
                        );
                    }
                } else {
                    {
                        for (&timestamp, cross_data_fb, cross_weights_fb) in izip!(
                            timeblock.timestamps.iter(),
                            unaveraged_cross_data_tfb.outer_iter_mut(),
                            unaveraged_cross_weights_tfb.outer_iter_mut(),
                        ) {
                            self.apply_solutions(
                                timestamp,
                                cross_data_fb,
                                cross_weights_fb,
                                None,
                                &chan_freqs,
                            );
                        }
                    }
                }
            }

            // Now that solutions have been applied, we can average the data
            // into the supplied arrays.
            debug!("Averaging input data from timeblock {}", timeblock.index);
            vis_average(
                unaveraged_cross_data_tfb.view(),
                cross_data_fb,
                unaveraged_cross_weights_tfb.view(),
                cross_weights_fb,
                &self.spw.flagged_chanblock_indices,
            );
            if let (
                Some((mut auto_data_fb, mut auto_weights_fb)),
                Some((unaveraged_auto_data_tfb, unaveraged_auto_weights_tfb)),
            ) = (autos_fb, unaveraged_autos)
            {
                vis_average(
                    unaveraged_auto_data_tfb.view(),
                    auto_data_fb.view_mut(),
                    unaveraged_auto_weights_tfb.view(),
                    auto_weights_fb.view_mut(),
                    &self.spw.flagged_chanblock_indices,
                );
            };
        } else {
            // Not averaging; read the data directly into the supplied arrays.
            let timestamp = *timeblock.timestamps.first();
            let timestep = *timeblock.timesteps.first();
            debug!("Reading timestamp {}", timestamp.to_gpst_seconds());
            self.read_timestep(
                timestep,
                cross_data_fb.view_mut(),
                cross_weights_fb.view_mut(),
                autos_fb.as_mut().map(|(auto_data_fb, auto_weights_fb)| {
                    (auto_data_fb.view_mut(), auto_weights_fb.view_mut())
                }),
                &self.spw.flagged_chan_indices,
            )?;

            // Should we continue?
            if error.load() {
                return Ok(());
            }

            // Apply calibration solutions, if they're supplied.
            if self.solutions.is_some() {
                debug!("Applying calibration solutions to input data from timestep {timestep}");
                self.apply_solutions(
                    timestamp,
                    cross_data_fb,
                    cross_weights_fb.view_mut(),
                    autos_fb,
                    &self
                        .spw
                        .chanblocks
                        .iter()
                        .map(|c| c.freq)
                        .collect::<Vec<_>>(),
                );
            }
        }

        // Should we continue?
        if error.load() {
            return Ok(());
        }

        Ok(())
    }

    fn read_timestep(
        &self,
        timestep: usize,
        mut cross_data_fb: ArrayViewMut2<Jones<f32>>,
        mut cross_weights_fb: ArrayViewMut2<f32>,
        autos_fb: Option<(ArrayViewMut2<Jones<f32>>, ArrayViewMut2<f32>)>,
        flagged_channels: &HashSet<u16>,
    ) -> Result<(), VisReadError> {
        let obs_context = self.get_obs_context();

        match (autos_fb, obs_context.autocorrelations_present) {
            (Some((mut auto_data_fb, mut auto_weights_fb)), true) => {
                debug!("Reading crosses and autos for timestep {timestep}");

                self.vis_reader.read_crosses_and_autos(
                    cross_data_fb.view_mut(),
                    cross_weights_fb.view_mut(),
                    auto_data_fb.view_mut(),
                    auto_weights_fb.view_mut(),
                    timestep,
                    &self.tile_baseline_flags,
                    flagged_channels,
                )?;

                if self.ignore_weights {
                    cross_weights_fb.fill(1.0);
                    auto_weights_fb.fill(1.0);
                }
            }

            // Otherwise, just read the crosses.
            _ => {
                debug!("Reading crosses for timestep {timestep}");

                self.vis_reader.read_crosses(
                    cross_data_fb.view_mut(),
                    cross_weights_fb.view_mut(),
                    timestep,
                    &self.tile_baseline_flags,
                    flagged_channels,
                )?;

                if self.ignore_weights {
                    cross_weights_fb.fill(1.0);
                }
            }
        }

        Ok(())
    }

    fn apply_solutions(
        &self,
        timestamp: Epoch,
        mut cross_data_fb: ArrayViewMut2<Jones<f32>>,
        mut cross_weights_fb: ArrayViewMut2<f32>,
        mut autos_fb: Option<(ArrayViewMut2<Jones<f32>>, ArrayViewMut2<f32>)>,
        chan_freqs: &[f64],
    ) {
        assert_eq!(cross_data_fb.dim(), cross_weights_fb.dim());
        assert_eq!(cross_data_fb.len_of(Axis(0)), chan_freqs.len());
        let solutions = match self.solutions.as_ref() {
            Some(s) => s,
            None => return,
        };
        let obs_context = self.get_obs_context();
        let solution_freqs = solutions.chanblock_freqs.as_ref();
        // If there aren't any solution frequencies, we can only apply solutions
        // to equally-sized arrays (i.e. if the incoming data and the solutions
        // have 768 channels, then we're OK, otherwise we don't know how to map
        // the solutions). Note that in this scenario, this assumes that the
        // frequencies corresponding to the solutions are the same as what's in
        // the data, but there's no way of checking.
        if solution_freqs.is_none()
            && cross_data_fb.len_of(Axis(0)) + self.spw.flagged_chanblock_indices.len()
                != solutions.di_jones.len_of(Axis(2))
        {
            panic!("Cannot apply calibration solutions to unequal sized data");
        }

        let timestamps = &obs_context.timestamps;
        let span = *timestamps.last() - *timestamps.first();
        let timestamp_fraction = ((timestamp - *timestamps.first()).to_seconds()
            / span.to_seconds())
        // Stop stupid values.
        .clamp(0.0, 0.99);

        // Find solutions corresponding to this timestamp.
        let sols = solutions.get_timeblock(timestamp, timestamp_fraction);
        // Now make a lookup vector for the channels. This is better than
        // searching for the right solution channel for each channel below (we
        // use more memory but avoid a quadratic-complexity algorithm).
        let solution_freq_indices: Option<Vec<usize>> =
            solution_freqs.as_ref().map(|solution_freqs| {
                chan_freqs
                    .iter()
                    .map(|freq| {
                        // Find the nearest solution freq to our data freq.
                        let mut best = f64::INFINITY;
                        let mut i_sol_freq = 0;
                        for (i, &sol_freq) in solution_freqs.iter().enumerate() {
                            let this_diff = (sol_freq - freq).abs();
                            if this_diff < best {
                                best = this_diff;
                                i_sol_freq = i;
                            } else {
                                // Because the frequencies are always
                                // ascendingly sorted, if the frequency
                                // difference is getting bigger, we can break
                                // early.
                                break;
                            }
                        }
                        i_sol_freq
                    })
                    .collect()
            });

        for (i_baseline, (mut cross_data_f, mut cross_weights_f)) in cross_data_fb
            .axis_iter_mut(Axis(1))
            .zip_eq(cross_weights_fb.axis_iter_mut(Axis(1)))
            .enumerate()
        {
            let (tile1, tile2) = self.tile_baseline_flags.unflagged_cross_baseline_to_tile_map
                .get(&i_baseline)
                .copied()
                .unwrap_or_else(|| {
                    panic!("Couldn't find baseline index {i_baseline} in unflagged_cross_baseline_to_tile_map")
                });

            if let Some(solution_freq_indices) = solution_freq_indices.as_ref() {
                cross_data_f
                    .iter_mut()
                    .zip_eq(cross_weights_f.iter_mut())
                    .zip_eq(solution_freq_indices.iter().copied())
                    .for_each(|((vis_data, vis_weight), i_sol_freq)| {
                        // Get the solutions for both tiles and apply them.
                        let sol1 = sols[(tile1, i_sol_freq)];
                        let sol2 = sols[(tile2, i_sol_freq)];

                        // One of the tiles doesn't have a solution; flag.
                        if sol1.any_nan() || sol2.any_nan() {
                            *vis_weight = -vis_weight.abs();
                            *vis_data = Jones::default();
                        } else {
                            // Promote the data before demoting it again.
                            let d: Jones<f64> = Jones::from(*vis_data);
                            *vis_data = Jones::from((sol1 * d) * sol2.h());
                        }
                    });
            } else {
                // Get the solutions for both tiles and apply them.
                let sols_tile1 = sols.slice(s![tile1, ..]);
                let sols_tile2 = sols.slice(s![tile2, ..]);
                izip!(
                    (0..),
                    cross_data_f.iter_mut(),
                    cross_weights_f.iter_mut(),
                    sols_tile1.iter(),
                    sols_tile2.iter()
                )
                .for_each(|(i_chan, vis_data, vis_weight, sol1, sol2)| {
                    // One of the tiles doesn't have a solution; flag.
                    if sol1.any_nan() || sol2.any_nan() {
                        *vis_weight = -vis_weight.abs();
                        *vis_data = Jones::default();
                    } else {
                        if self.spw.flagged_chan_indices.contains(&i_chan) {
                            // The channel is flagged, but we still have a solution for it.
                            *vis_weight = -vis_weight.abs();
                        }
                        // Promote the data before demoting it again.
                        let d: Jones<f64> = Jones::from(*vis_data);
                        *vis_data = Jones::from((*sol1 * d) * sol2.h());
                    }
                });
            }
        }

        if let Some((auto_data_fb, auto_weights_fb)) = autos_fb.as_mut() {
            for (i_tile, (mut auto_data_f, mut auto_weights_f)) in auto_data_fb
                .axis_iter_mut(Axis(1))
                .zip_eq(auto_weights_fb.axis_iter_mut(Axis(1)))
                .enumerate()
            {
                let i_tile = self
                    .tile_baseline_flags
                    .unflagged_auto_index_to_tile_map
                    .get(&i_tile)
                    .copied()
                    .unwrap_or_else(|| {
                        panic!(
                            "Couldn't find auto index {i_tile} in unflagged_auto_index_to_tile_map"
                        )
                    });

                if let Some(solution_freq_indices) = solution_freq_indices.as_ref() {
                    auto_data_f
                        .iter_mut()
                        .zip_eq(auto_weights_f.iter_mut())
                        .zip_eq(solution_freq_indices.iter().copied())
                        .for_each(|((vis_data, vis_weight), i_sol_freq)| {
                            // Get the solutions for the tile and apply it twice.
                            let sol = sols[(i_tile, i_sol_freq)];

                            // No solution; flag.
                            if sol.any_nan() {
                                *vis_weight = -vis_weight.abs();
                                *vis_data = Jones::default();
                            } else {
                                // Promote the data before demoting it again.
                                let d: Jones<f64> = Jones::from(*vis_data);
                                *vis_data = Jones::from((sol * d) * sol.h());
                            }
                        });
                } else {
                    // Get the solutions for the tile and apply it twice.
                    let sols = sols.slice(s![i_tile, ..]);
                    izip!(
                        (0..),
                        auto_data_f.iter_mut(),
                        auto_weights_f.iter_mut(),
                        sols.iter()
                    )
                    .for_each(|(i_chan, vis_data, vis_weight, sol)| {
                        // No solution; flag.
                        if sol.any_nan() {
                            *vis_weight = -vis_weight.abs();
                            *vis_data = Jones::default();
                        } else {
                            if self.spw.flagged_chan_indices.contains(&i_chan) {
                                // The channel is flagged, but we still have a solution for it.
                                *vis_weight = -vis_weight.abs();
                            }
                            // Promote the data before demoting it again.
                            let d: Jones<f64> = Jones::from(*vis_data);
                            *vis_data = Jones::from((*sol * d) * sol.h());
                        }
                    });
                }
            }
        }

        debug!("Finished applying solutions");
    }
}
