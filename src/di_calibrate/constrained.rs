// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Calibration of a timeblock with all chanblocks solved in lock step so that
//! constraints spanning frequency (phase models, smoothness) can be applied
//! between iterations, as DP3's DDECal does.

use indicatif::ProgressBar;
use log::debug;
use marlu::{c64, Jones};
use ndarray::{prelude::*, Zip};

use super::{
    calibration_loop,
    constraints::{Constraints, PhaseFit},
    CalibrationResult,
};
use crate::{
    averaging::{Chanblock, Timeblock},
    context::Polarisations,
};

/// Describe a chanblock's result the same way the unconstrained path does.
pub(super) fn status_string(
    result: &CalibrationResult,
    chanblock_index: usize,
    num_unflagged_tiles: usize,
    stop_threshold: f64,
    min_threshold: f64,
) -> String {
    let mut status_str = format!("Chanblock {chanblock_index:>3}");
    if num_unflagged_tiles - result.num_failed <= 4 {
        status_str.push_str(&format!(
            ": failed    ({:>2}): Too many antenna solutions failed ({})",
            result.num_iterations, result.num_failed
        ));
    } else if result.max_precision > min_threshold {
        status_str.push_str(&format!(
            ": failed    ({:>2}): {:.5e} > {:e}",
            result.num_iterations, result.max_precision, min_threshold,
        ));
    } else if result.max_precision > stop_threshold {
        status_str.push_str(&format!(
            ": converged ({:>2}): {:e} > {:.5e} > {:e}",
            result.num_iterations, min_threshold, result.max_precision, stop_threshold
        ));
    } else {
        status_str.push_str(&format!(
            ": converged ({:>2}): {:e} > {:.5e}",
            result.num_iterations, stop_threshold, result.max_precision
        ));
    }
    status_str
}

/// Jones "division" that respects single-polarisation data.
pub(super) fn jones_div(pols: Polarisations, top: &Jones<f64>, bot: &Jones<f64>) -> Jones<f64> {
    match pols {
        Polarisations::XX => Jones::from([
            top[0] / bot[0],
            c64::default(),
            c64::default(),
            c64::default(),
        ]),
        Polarisations::YY => Jones::from([
            c64::default(),
            c64::default(),
            c64::default(),
            top[3] / bot[3],
        ]),
        Polarisations::XX_XY_YX_YY | Polarisations::XX_YY | Polarisations::XX_YY_XY => *top / bot,
    }
}

/// Like `calibrate_timeblock`, but every iteration updates all chanblocks
/// before the constraints are applied across them; iteration continues until
/// every chanblock has converged (or `max_iterations`). Chanblocks with no
/// data are marked failed on the first iteration, exactly as in the
/// unconstrained path. The constraints are applied once more at the end so
/// the returned solutions satisfy them.
#[allow(clippy::too_many_arguments)]
pub(super) fn calibrate_timeblock_constrained(
    vis_data_tfb: ArrayView3<Jones<f32>>,
    vis_model_tfb: ArrayView3<Jones<f32>>,
    mut di_jones: ArrayViewMut3<Jones<f64>>,
    timeblock: &Timeblock,
    chanblocks: &[Chanblock],
    max_iterations: u32,
    stop_threshold: f64,
    min_threshold: f64,
    pols: Polarisations,
    constraints: &Constraints,
    progress_bar: ProgressBar,
    print_convergence_messages: bool,
) -> Vec<CalibrationResult> {
    let (_, num_tiles, num_chanblocks) = di_jones.dim();
    assert_eq!(chanblocks.len(), num_chanblocks);
    // (tiles, chanblocks) for this timeblock.
    let mut jones = di_jones.slice_mut(s![timeblock.index, .., ..]);
    let mut old_jones: Array2<Jones<f64>> =
        Array2::from_elem((num_tiles, num_chanblocks), Jones::default());
    let mut failed: Array2<bool> = Array2::from_elem((num_tiles, num_chanblocks), false);
    // Chanblocks that stop iterating early (no data, or too few tiles).
    let mut settled: Array1<bool> = Array1::from_elem(num_chanblocks, false);
    let mut max_precisions: Array1<f64> = Array1::zeros(num_chanblocks);
    let unflagged_indices: Array1<usize> =
        Array1::from_iter(chanblocks.iter().map(|c| c.unflagged_index as usize));

    progress_bar.set_length(max_iterations as u64);
    let mut iteration = 0;
    let mut last_fits: Option<Vec<PhaseFit>> = None;
    while iteration < max_iterations {
        iteration += 1;

        // One StefCal update per chanblock.
        Zip::from(jones.axis_iter_mut(Axis(1)))
            .and(old_jones.axis_iter_mut(Axis(1)))
            .and(failed.axis_iter_mut(Axis(1)))
            .and(&mut settled)
            .and(&unflagged_indices)
            .par_for_each(
                |mut jones, mut old_jones, mut failed, settled, &i_chanblock| {
                    if *settled {
                        return;
                    }
                    let range = s![timeblock.range.clone(), i_chanblock..i_chanblock + 1, ..];
                    let mut top: Array1<Jones<f64>> =
                        Array1::from_elem(num_tiles, Jones::default());
                    let mut bot: Array1<Jones<f64>> =
                        Array1::from_elem(num_tiles, Jones::default());
                    calibration_loop(
                        vis_data_tfb.slice(range),
                        vis_model_tfb.slice(range),
                        jones.view(),
                        top.view_mut(),
                        bot.view_mut(),
                    );

                    if iteration == 1 {
                        let mut equal = true;
                        let mut zeros = true;
                        for (top, bot) in top.iter().zip(bot.iter()) {
                            let diff = *top - bot;
                            if diff
                                .to_float_array()
                                .iter()
                                .any(|d| d.abs() > stop_threshold)
                            {
                                equal = false;
                                zeros = false;
                                break;
                            }
                            if zeros
                                && (top.to_float_array().iter().any(|v| v.abs() > f64::EPSILON)
                                    || bot.to_float_array().iter().any(|v| v.abs() > f64::EPSILON))
                            {
                                zeros = false;
                            }
                        }
                        if equal {
                            if zeros {
                                failed.fill(true);
                            }
                            *settled = true;
                            return;
                        }
                    }

                    for (((j, old), failed), (top, bot)) in jones
                        .iter_mut()
                        .zip(old_jones.iter_mut())
                        .zip(failed.iter_mut())
                        .zip(top.iter().zip(bot.iter()))
                    {
                        if *failed {
                            continue;
                        }
                        let div = jones_div(pols, top, bot);
                        if div.any_nan() {
                            *failed = true;
                            *j = Jones::default();
                            *old = Jones::default();
                        } else {
                            *j = div;
                        }
                    }
                },
            );

        // Constrain across chanblocks. Settled chanblocks are masked out.
        let mask = masked(&failed, &settled);
        last_fits = constraints.apply(jones.view_mut(), mask.view());

        // Convergence bookkeeping, per chanblock.
        let mut all_converged = true;
        Zip::from(jones.axis_iter_mut(Axis(1)))
            .and(old_jones.axis_iter_mut(Axis(1)))
            .and(failed.axis_iter(Axis(1)))
            .and(&mut settled)
            .and(&mut max_precisions)
            .par_map_collect(|mut jones, mut old_jones, failed, settled, max_precision| {
                if *settled {
                    return true;
                }
                let num_failed = failed.iter().filter(|&&f| f).count();
                if num_tiles - num_failed <= 4 {
                    *settled = true;
                    return true;
                }
                let mut converged = false;
                if iteration % 2 == 0 {
                    let mut max_p: f64 = 0.0;
                    for ((j, old), _) in jones
                        .iter_mut()
                        .zip(old_jones.iter())
                        .zip(failed.iter())
                        .filter(|(_, &f)| !f)
                    {
                        let diff = *j - old;
                        for p in [
                            diff[0].norm_sqr(),
                            diff[1].norm_sqr(),
                            diff[2].norm_sqr(),
                            diff[3].norm_sqr(),
                        ] {
                            max_p = max_p.max(p);
                        }
                        *j = (*j + *old) * 0.5;
                    }
                    *max_precision = max_p;
                    converged = max_p < stop_threshold;
                }
                old_jones.assign(&jones);
                converged
            })
            .iter()
            .for_each(|&c| all_converged &= c);

        progress_bar.inc(1);
        if all_converged {
            break;
        }
    }

    // Leave the solutions on the constraint.
    let mask = masked(&failed, &settled);
    if let Some(fits) = constraints.apply(jones.view_mut(), mask.view()) {
        last_fits = Some(fits);
    }
    if let Some(fits) = &last_fits {
        let ref_freq = constraints.ref_freq();
        for (i, fit) in fits.iter().enumerate() {
            let (tec, phase, delay) = fit.to_physical(ref_freq);
            debug!(
                "Timeblock {}: unflagged tile {i:>3}: TEC {tec:+.4e} TECU, phase {phase:+.3} rad, delay {:+.3e} ns",
                timeblock.index,
                delay * 1e9
            );
        }
    }

    // Set failed solutions to NaN and summarise each chanblock.
    jones
        .iter_mut()
        .zip(failed.iter())
        .filter(|(_, &f)| f)
        .for_each(|(j, _)| *j = Jones::nan());
    let results: Vec<CalibrationResult> = chanblocks
        .iter()
        .zip(failed.axis_iter(Axis(1)))
        .zip(max_precisions.iter())
        .zip(jones.axis_iter_mut(Axis(1)))
        .map(|(((chanblock, failed), &max_precision), mut jones)| {
            let num_failed = failed.iter().filter(|&&f| f).count();
            let max_precision = if max_precision.abs() > 1e100 {
                f64::NAN
            } else {
                max_precision
            };
            let converged = if num_tiles - num_failed <= 4 || max_precision > min_threshold {
                jones.fill(Jones::nan());
                false
            } else {
                true
            };
            let result = CalibrationResult {
                num_iterations: iteration,
                converged,
                max_precision,
                num_failed,
                chanblock: Some(chanblock.chanblock_index as usize),
                i_chanblock: Some(chanblock.unflagged_index as usize),
            };
            if print_convergence_messages {
                let s = status_string(
                    &result,
                    chanblock.chanblock_index as usize,
                    num_tiles,
                    stop_threshold,
                    min_threshold,
                );
                if progress_bar.is_hidden() {
                    println!("{s}");
                } else {
                    progress_bar.println(s);
                }
            }
            result
        })
        .collect();
    progress_bar.abandon();
    results
}

fn masked(failed: &Array2<bool>, settled: &Array1<bool>) -> Array2<bool> {
    let mut mask = failed.clone();
    for (mut col, &s) in mask.axis_iter_mut(Axis(1)).zip(settled.iter()) {
        if s {
            col.fill(true);
        }
    }
    mask
}
