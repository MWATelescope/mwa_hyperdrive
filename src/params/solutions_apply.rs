// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

use std::collections::HashSet;

use crossbeam_utils::atomic::AtomicCell;
use log::info;
use ndarray::prelude::*;

use super::{InputVisParams, OutputVisParams, VisConvertError, VisConvertParams};
use crate::io::read::{ColumnTimestep, VisReadError};

/// The column of a measurement set that calibrated visibilities are written
/// to.
pub(crate) const CORRECTED_DATA_COL_NAME: &str = "CORRECTED_DATA";

pub(crate) struct SolutionsApplyParams {
    pub(crate) input_vis_params: InputVisParams,

    /// The visibility files to write, if any.
    pub(crate) output_vis_params: Option<OutputVisParams>,

    /// Write the calibrated visibilities to the `CORRECTED_DATA` column of
    /// the input measurement set?
    pub(crate) ms_corrected_data: bool,
}

impl SolutionsApplyParams {
    pub(crate) fn run(&self) -> Result<(), VisConvertError> {
        let Self {
            input_vis_params,
            output_vis_params,
            ms_corrected_data,
        } = self;

        assert!(
            input_vis_params.solutions.is_some(),
            "No calibration solutions are in the input vis params; this shouldn't be possible"
        );

        if *ms_corrected_data {
            write_corrected_data(input_vis_params)?;
        }
        if let Some(output_vis_params) = output_vis_params {
            VisConvertParams::run_inner(input_vis_params, output_vis_params)?;
        }
        Ok(())
    }
}

/// Write calibrated visibilities to the `CORRECTED_DATA` column of the input
/// measurement set. Each timestep is read, calibrated and written back to the
/// rows it came from.
///
/// Every cell of the column is written. Those that hyperdrive didn't
/// calibrate (flagged tiles and channels, timesteps that aren't being used,
/// and visibilities without solutions) get zeros and are flagged in the `FLAG`
/// column, so that nothing downstream mistakes them for calibrated data.
fn write_corrected_data(input_vis_params: &InputVisParams) -> Result<(), VisReadError> {
    let ms_reader = input_vis_params
        .vis_reader
        .as_ms_reader()
        .expect("checked when the arguments were parsed");
    let obs_context = input_vis_params.get_obs_context();
    let tile_baseline_flags = &input_vis_params.tile_baseline_flags;
    let flagged_fine_chans = &input_vis_params.spw.flagged_chan_indices;
    assert_eq!(
        input_vis_params.spw.chans_per_chanblock.get(),
        1,
        "checked when the arguments were parsed"
    );

    info!(
        "Writing calibrated visibilities to the {CORRECTED_DATA_COL_NAME} column of {}",
        ms_reader.path().display()
    );
    ms_reader.prepare_column(CORRECTED_DATA_COL_NAME)?;

    let num_unflagged_tiles = input_vis_params.get_num_unflagged_tiles();
    let num_unflagged_cross_baselines = (num_unflagged_tiles * (num_unflagged_tiles - 1)) / 2;
    let num_chans = input_vis_params.spw.chanblocks.len();
    let cross_vis_shape = (num_chans, num_unflagged_cross_baselines);
    let auto_vis_shape = (num_chans, num_unflagged_tiles);
    let mut cross_data_fb = Array2::default(cross_vis_shape);
    let mut cross_weights_fb = Array2::zeros(cross_vis_shape);
    // Calibrate the auto-correlations if they're there, whether or not the
    // user asked for them; otherwise their rows would be flagged.
    let mut autos_fb = obs_context.autocorrelations_present.then(|| {
        (
            Array2::default(auto_vis_shape),
            Array2::zeros(auto_vis_shape),
        )
    });

    let error = AtomicCell::new(false);
    let mut written_timesteps = HashSet::new();
    for timeblock in &input_vis_params.timeblocks {
        assert_eq!(
            timeblock.timesteps.len(),
            1,
            "checked when the arguments were parsed"
        );
        let timestep = *timeblock.timesteps.first();

        cross_data_fb.fill(Default::default());
        cross_weights_fb.fill(0.0);
        if let Some((auto_data_fb, auto_weights_fb)) = autos_fb.as_mut() {
            auto_data_fb.fill(Default::default());
            auto_weights_fb.fill(0.0);
        }
        input_vis_params.read_timeblock(
            timeblock,
            cross_data_fb.view_mut(),
            cross_weights_fb.view_mut(),
            autos_fb.as_mut().map(|(d, w)| (d.view_mut(), w.view_mut())),
            &error,
        )?;

        ms_reader.write_column_timestep(
            CORRECTED_DATA_COL_NAME,
            timestep,
            Some(ColumnTimestep {
                cross_vis_fb: cross_data_fb.view(),
                cross_weights_fb: cross_weights_fb.view(),
                autos_fb: autos_fb.as_ref().map(|(d, w)| (d.view(), w.view())),
            }),
            tile_baseline_flags,
            flagged_fine_chans,
            true,
        )?;
        written_timesteps.insert(timestep);
    }

    // The timesteps that weren't used have no calibrated visibilities.
    for &timestep in &obs_context.all_timesteps {
        if !written_timesteps.contains(&timestep) {
            ms_reader.write_column_timestep(
                CORRECTED_DATA_COL_NAME,
                timestep,
                None,
                tile_baseline_flags,
                flagged_fine_chans,
                true,
            )?;
        }
    }

    ms_reader.finish_writing();
    info!(
        "Finished writing {CORRECTED_DATA_COL_NAME} ({} of {} timesteps calibrated)",
        written_timesteps.len(),
        obs_context.all_timesteps.len()
    );
    Ok(())
}
