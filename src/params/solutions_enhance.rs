// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Parameters for (and the running of) solutions-enhance: taking the fine
//! frequency structure of calibration solutions from the auto-correlations of
//! the data they belong to.

use std::{borrow::Cow, collections::HashSet, path::PathBuf};

use itertools::Itertools;
use log::{debug, info};
use marlu::Jones;
use ndarray::prelude::*;
use vec1::Vec1;

use super::InputVisParams;
use crate::{
    cli::Warn,
    io::read::VisReadError,
    solutions::{
        enhance::{enhance_timeblock, nan_median, EnhanceOptions, EnhanceSummary, MIN_CHANS},
        CalibrationSolutions, SolutionsWriteError,
    },
};

pub(crate) struct SolutionsEnhanceParams {
    /// The data to read auto-correlations from. The solutions in here aren't
    /// applied to anything.
    pub(crate) input_vis_params: InputVisParams,

    /// The solutions to be enhanced.
    pub(crate) solutions: CalibrationSolutions,

    /// Where the enhanced solutions are written.
    pub(crate) output_solution_files: Vec1<PathBuf>,

    pub(crate) options: EnhanceOptions,
}

impl SolutionsEnhanceParams {
    /// Enhance the solutions, write them out and return them.
    pub(crate) fn run(self) -> Result<CalibrationSolutions, SolutionsEnhanceError> {
        let Self {
            input_vis_params,
            mut solutions,
            output_solution_files,
            options,
        } = self;

        let obs_context = input_vis_params.get_obs_context();
        if !obs_context.autocorrelations_present {
            return Err(SolutionsEnhanceError::NoAutos);
        }
        let (num_timeblocks, num_tiles, num_chanblocks) = solutions.di_jones.dim();
        let data_freqs = obs_context.fine_chan_freqs.mapped_ref(|&f| f as f64);

        // The frequencies of the solutions.
        let solution_freqs = match solutions.chanblock_freqs.as_ref() {
            Some(f) => f.to_vec(),
            // Without frequencies, all that can be done is to assume they're
            // those of the data.
            None => {
                if data_freqs.len() != num_chanblocks {
                    return Err(SolutionsEnhanceError::ChannelCountMismatch {
                        data: data_freqs.len(),
                        solutions: num_chanblocks,
                    });
                }
                data_freqs.to_vec()
            }
        };
        if num_chanblocks < MIN_CHANS {
            return Err(SolutionsEnhanceError::TooFewChannels {
                got: num_chanblocks,
                need: MIN_CHANS,
            });
        }
        let solution_freq_res = solution_freqs
            .windows(2)
            .map(|w| (w[1] - w[0]).abs())
            .fold(f64::INFINITY, f64::min);
        // Which chanblock does each channel of the data go into? Channels
        // outside of the solutions' band aren't used.
        let chan_to_chanblock = data_freqs
            .iter()
            .map(|&freq| {
                solution_freqs
                    .iter()
                    .enumerate()
                    .map(|(i, &sol_freq)| ((sol_freq - freq).abs(), i))
                    .min_by(|a, b| a.0.total_cmp(&b.0))
                    .filter(|(diff, _)| *diff <= 0.505 * solution_freq_res)
                    .map(|(_, i)| i)
            })
            .collect::<Vec<_>>();

        // Read the auto-correlation powers (XX and YY) of every timestep, and
        // note which timeblock of the solutions each timestep belongs to.
        let tile_map = &input_vis_params
            .tile_baseline_flags
            .unflagged_auto_index_to_tile_map;
        let num_unflagged_tiles = input_vis_params.get_num_unflagged_tiles();
        let vis_shape = (data_freqs.len(), num_unflagged_tiles);
        let mut vis_fb = Array2::from_elem(vis_shape, Jones::<f32>::default());
        let mut weights_fb = Array2::<f32>::zeros(vis_shape);
        let no_flagged_chans = HashSet::new();
        let first_timestamp = *obs_context.timestamps.first();
        let span = (*obs_context.timestamps.last() - first_timestamp).to_seconds();
        // (index of the solutions timeblock, powers [pol][tile][chan])
        let mut powers: Vec<(usize, Array3<f32>)> = vec![];
        for timeblock in &input_vis_params.timeblocks {
            for (&timestamp, &timestep) in timeblock.timestamps.iter().zip(&timeblock.timesteps) {
                debug!("Reading autos of timestamp {}", timestamp.to_gpst_seconds());
                vis_fb.fill(Jones::default());
                weights_fb.fill(0.0);
                input_vis_params.vis_reader.read_autos(
                    vis_fb.view_mut(),
                    weights_fb.view_mut(),
                    timestep,
                    &input_vis_params.tile_baseline_flags,
                    &no_flagged_chans,
                )?;

                let fraction = if span > 0.0 {
                    ((timestamp - first_timestamp).to_seconds() / span).clamp(0.0, 0.99)
                } else {
                    0.0
                };
                let i_timeblock = solutions.get_timeblock_index(timestamp, fraction);
                let mut power = Array3::from_elem((2, num_unflagged_tiles, vis_shape.0), f32::NAN);
                for (((i_chan, i_tile), vis), weight) in
                    vis_fb.indexed_iter().zip(weights_fb.iter())
                {
                    if *weight > 0.0 {
                        power[(0, i_tile, i_chan)] = vis[0].re;
                        power[(1, i_tile, i_chan)] = vis[3].re;
                    }
                }
                powers.push((i_timeblock, power));
            }
        }

        let tile_names = solutions
            .tile_names
            .as_ref()
            .map(|n| n.as_slice())
            .unwrap_or(obs_context.tile_names.as_slice());
        for i_timeblock in 0..num_timeblocks {
            // Use the timesteps that belong to this timeblock; if there aren't
            // any, use all of them.
            let mut timesteps = powers
                .iter()
                .filter(|(i, _)| *i == i_timeblock)
                .map(|(_, p)| p)
                .collect::<Vec<_>>();
            if timesteps.is_empty() {
                timesteps = powers.iter().map(|(_, p)| p).collect();
            }

            // The median over time for each channel of the data, then the mean
            // over the channels of a chanblock.
            let mut autos_ptf = Array3::from_elem((2, num_tiles, num_chanblocks), f64::NAN);
            let mut values = Vec::with_capacity(timesteps.len());
            for i_pol in 0..2 {
                for (&i_unflagged_tile, &i_tile) in tile_map {
                    let mut sums = vec![(0.0, 0); num_chanblocks];
                    for (i_chan, i_chanblock) in chan_to_chanblock.iter().enumerate() {
                        let Some(i_chanblock) = *i_chanblock else {
                            continue;
                        };
                        values.clear();
                        values.extend(
                            timesteps
                                .iter()
                                .map(|p| f64::from(p[(i_pol, i_unflagged_tile, i_chan)])),
                        );
                        let median = nan_median(&mut values);
                        if median > 0.0 {
                            sums[i_chanblock].0 += median;
                            sums[i_chanblock].1 += 1;
                        }
                    }
                    for (auto, (sum, n)) in autos_ptf
                        .slice_mut(s![i_pol, i_tile, ..])
                        .iter_mut()
                        .zip(sums)
                    {
                        if n > 0 {
                            *auto = sum / f64::from(n);
                        }
                    }
                }
            }

            let summaries = enhance_timeblock(
                solutions.di_jones.index_axis_mut(Axis(0), i_timeblock),
                autos_ptf.view(),
                &solution_freqs,
                &options,
            );
            for (pol, summary) in ["XX", "YY"].into_iter().zip(summaries) {
                report(i_timeblock, pol, &summary, tile_names, &options);
            }
        }

        for file in &output_solution_files {
            CalibrationSolutions::write_solutions_from_ext_inner(&solutions, file)?;
            info!(
                "Enhanced calibration solutions written to {}",
                file.display()
            );
        }

        Ok(solutions)
    }
}

/// Tell the user what happened to a polarisation of a timeblock.
fn report(
    i_timeblock: usize,
    pol: &str,
    summary: &EnhanceSummary,
    tile_names: &[String],
    options: &EnhanceOptions,
) {
    let EnhanceSummary {
        num_fitted,
        kept_tiles,
        phase_slope,
        auto_fine_amp,
        cal_minus_auto_amp,
        cal_minus_auto_phase,
    } = summary;

    info!("Timeblock {i_timeblock} {pol}: {num_fitted} tiles enhanced");
    if !kept_tiles.is_empty() {
        info!(
            "    {} tiles keep their solutions (their autos don't match them): {}",
            kept_tiles.len(),
            kept_tiles
                .iter()
                .map(|&i| tile_names
                    .get(i)
                    .map(|n| Cow::from(n.as_str()))
                    .unwrap_or_else(|| Cow::from(i.to_string())))
                .join(", ")
        );
    }
    if *num_fitted == 0 {
        format!("Timeblock {i_timeblock} {pol}: no solutions could be enhanced").warn();
        return;
    }
    info!(
        "    Fine structure in the autos: {:.2}%; in the solutions but not the autos: {:.2}%, {:.2}°",
        auto_fine_amp * 100.0,
        cal_minus_auto_amp * 100.0,
        cal_minus_auto_phase.to_degrees()
    );
    info!("    Phase of the solutions against that derived from the autos: slope {phase_slope:.2}");
    // The phase structure of the solutions should follow that derived from the
    // autos (a slope of 1 for noiseless solutions).
    if !options.amplitude_only && *phase_slope < 0.0 {
        [
            format!("Timeblock {i_timeblock} {pol}: the phase structure of the solutions is opposite to that derived from the autos (slope {phase_slope:.2}).").into(),
            "The solutions may not be for this data. Consider --amplitude-only.".into(),
        ]
        .warn();
    }
}

#[derive(thiserror::Error, Debug)]
pub(crate) enum SolutionsEnhanceError {
    #[error("The input data has no auto-correlations; these are needed to enhance solutions")]
    NoAutos,

    #[error("The solutions don't list their frequencies and have a different number of channels to the data (data: {data}, solutions: {solutions}); cannot continue")]
    ChannelCountMismatch { data: usize, solutions: usize },

    #[error("The solutions have {got} channels, but at least {need} are needed to enhance them")]
    TooFewChannels { got: usize, need: usize },

    #[error(transparent)]
    VisRead(#[from] VisReadError),

    #[error(transparent)]
    SolutionsWrite(#[from] SolutionsWriteError),
}
