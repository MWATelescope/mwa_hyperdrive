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

use log::{debug, log_enabled, Level::Debug};
use marlu::Jones;
use ndarray::prelude::*;

use super::{params::CalibrateParams, CalibrateError};
use crate::solutions::CalibrationSolutions;
use mwa_hyperdrive_common::{log, marlu, ndarray};

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
        vis_data,
        vis_weights,
        vis_model,
    } = get_cal_vis(params, !params.no_progress_bars)?;
    assert_eq!(vis_weights.len_of(Axis(1)), params.baseline_weights.len());

    // The shape of the array containing output Jones matrices.
    let num_timeblocks = params.timeblocks.len();
    let num_chanblocks = params.fences.first().chanblocks.len();
    let num_unflagged_tiles = params.get_num_unflagged_tiles();

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

    let (sols, results) = calibrate_timeblocks(
        vis_data.view(),
        vis_model.view(),
        &params.timeblocks,
        // TODO: Picket fences.
        &params.fences.first().chanblocks,
        params.max_iterations,
        params.stop_threshold,
        params.min_threshold,
        !params.no_progress_bars,
        true,
    );

    // "Complete" the solutions.
    let sols = sols.into_cal_sols(params, Some(results.map(|r| r.max_precision)));

    Ok(sols)
}
