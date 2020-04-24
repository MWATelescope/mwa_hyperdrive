// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Code to plot calibration solutions.

use std::path::{Path, PathBuf};

use log::warn;
use plotters::{
    coord::Shift,
    prelude::*,
    style::{Color, RGBAColor},
};
use thiserror::Error;

use super::*;
use mwa_hyperdrive_common::{lazy_static, log, thiserror};

const X_PIXELS: u32 = 3200;
const Y_PIXELS: u32 = 1800;

lazy_static::lazy_static! {
    static ref CLEAR: RGBAColor = WHITE.mix(0.0);

    static ref POLS: [(&'static str, RGBAColor); 4] = [
        ("XX", BLUE.mix(1.0)),
        ("XY", BLUE.mix(0.2)),
        ("YX", RED.mix(0.2)),
        ("YY", RED.mix(1.0)),
    ];
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn plot_sols<T: AsRef<Path>, S: AsRef<str>>(
    sols: &CalibrationSolutions,
    filename_base: T,
    obs_name: &str,
    ref_tile: Option<usize>,
    no_ref_tile: bool,
    tile_names: Option<&[S]>,
    ignore_cross_pols: bool,
    min_amp: Option<f64>,
    max_amp: Option<f64>,
) -> Result<Vec<String>, Box<dyn std::error::Error>> {
    let (num_timeblocks, total_num_tiles, _) = sols.di_jones.dim();

    // How should the plot be split up to distribute the tiles?
    let split = match total_num_tiles {
        0..=128 => (8, total_num_tiles / 8),
        _ => (16, total_num_tiles / 16),
    };
    let title_style = ("sans-serif", 60).into_font();

    let mut amps = Array2::from_elem(
        (sols.di_jones.dim().1, sols.di_jones.dim().2),
        [0.0, 0.0, 0.0, 0.0],
    );
    let mut phases = Array2::from_elem(
        (sols.di_jones.dim().1, sols.di_jones.dim().2),
        [0.0, 0.0, 0.0, 0.0],
    );

    let ref_tile = match (no_ref_tile, ref_tile) {
        (true, _) => None,
        (_, Some(r)) => Some(r),
        // If the reference tile wasn't defined, use the first valid one from
        // the end.
        (_, None) => {
            let possibly_good = sols
                .di_jones
                .slice(s![0_usize, .., ..])
                .outer_iter()
                .enumerate()
                .filter(|(_, j)| {
                    !j.iter()
                        .all(|f| f[0].is_nan() || f[1].is_nan() || f[2].is_nan() || f[3].is_nan())
                })
                .map(|(i, _)| i)
                .next();
            // If the search for a valid tile didn't find anything, all
            // solutions must be NaN. In this case, it doesn't matter what the
            // reference is.
            possibly_good.map(|g| total_num_tiles - 1 - g)
        }
    };

    let mut output_filenames = vec![];
    for timeblock in 0..num_timeblocks {
        let mut output_amps = PathBuf::new();
        let mut output_phases = PathBuf::new();

        if num_timeblocks > 1 {
            let filename = format!(
                "{}_amps_{:03}.png",
                filename_base.as_ref().display(),
                timeblock
            );
            output_amps.set_file_name(&filename);
            output_filenames.push(filename);

            let filename = format!(
                "{}_phases_{:03}.png",
                filename_base.as_ref().display(),
                timeblock
            );
            output_phases.set_file_name(&filename);
            output_filenames.push(filename);
        } else {
            let filename = format!("{}_amps.png", filename_base.as_ref().display());
            output_amps.set_file_name(&filename);
            output_filenames.push(filename);

            let filename = format!("{}_phases.png", filename_base.as_ref().display());
            output_phases.set_file_name(&filename);
            output_filenames.push(filename);
        }

        let amps_root_area =
            BitMapBackend::new(&output_amps, (X_PIXELS, Y_PIXELS)).into_drawing_area();
        let phases_root_area =
            BitMapBackend::new(&output_phases, (X_PIXELS, Y_PIXELS)).into_drawing_area();
        amps_root_area.fill(&WHITE)?;
        phases_root_area.fill(&WHITE)?;
        // Draw the coloured text for each polarisation.
        for (i, (pol, colour)) in POLS.iter().enumerate() {
            if ignore_cross_pols && [1, 2].contains(&i) {
                continue;
            }
            amps_root_area.draw_text(
                *pol,
                &("sans-serif", 55).into_font().color(&colour),
                (X_PIXELS as i32 - 500 + 80 * i as i32, 10),
            )?;
            phases_root_area.draw_text(
                *pol,
                &("sans-serif", 55).into_font().color(&colour),
                (X_PIXELS as i32 - 500 + 80 * i as i32, 10),
            )?;
        }
        // Also draw the reference tile number and the GPS times for this
        // timeblock.
        let mut meta_str = match ref_tile {
            Some(ref_tile) => format!("Ref. tile {ref_tile}"),
            None => String::new(),
        };
        let time_str = match (
            sols.start_timestamps.get(timeblock),
            sols.end_timestamps.get(timeblock),
            sols.average_timestamps.get(timeblock),
        ) {
            (Some(s), Some(e), Some(a)) => {
                format!(
                    "GPS start {}, end {}, average {}",
                    s.as_gpst_seconds(),
                    e.as_gpst_seconds(),
                    a.as_gpst_seconds()
                )
            }
            (Some(s), Some(e), None) => format!(
                "GPS start {}, end {}",
                s.as_gpst_seconds(),
                e.as_gpst_seconds()
            ),
            (Some(s), None, None) => format!("GPS start {}, end unknown", s.as_gpst_seconds()),
            (None, Some(e), None) => format!("GPS start unknown, end {}", e.as_gpst_seconds()),
            (Some(s), None, Some(a)) => format!(
                "GPS start {}, end unknown, average {}",
                s.as_gpst_seconds(),
                a.as_gpst_seconds()
            ),
            (None, Some(e), Some(a)) => format!(
                "GPS start unknown, end {}, average {}",
                e.as_gpst_seconds(),
                a.as_gpst_seconds()
            ),
            (None, None, Some(a)) => format!(
                "GPS start unknown, end unknown, average {}",
                a.as_gpst_seconds()
            ),
            (None, None, None) => String::new(),
        };
        if !meta_str.is_empty() && !time_str.is_empty() {
            meta_str.push_str(", ");
        }
        meta_str.push_str(&time_str);
        amps_root_area.draw_text(
            &meta_str,
            &("sans-serif", 38).into_font().color(&BLACK),
            (10, 10),
        )?;
        phases_root_area.draw_text(
            &meta_str,
            &("sans-serif", 38).into_font().color(&BLACK),
            (10, 10),
        )?;

        let amps_root_area = amps_root_area
            .shrink((15, 0), (X_PIXELS - 15, Y_PIXELS))
            .titled(&format!("Amps for {}", obs_name), title_style.clone())?;
        let amps_tile_plots = amps_root_area.split_evenly(split);

        let phases_root_area =
            phases_root_area.titled(&format!("Phases for {}", obs_name), title_style.clone())?;
        let phase_tile_plots = phases_root_area.split_evenly(split);

        let ones = vec![Jones::identity(); sols.di_jones.dim().2];
        let ref_jones = if let Some(ref_tile) = ref_tile {
            sols.di_jones.slice(s![timeblock, ref_tile, ..])
        } else {
            ArrayView1::from_shape(sols.di_jones.dim().2, &ones)?
        };
        amps.outer_iter_mut()
            .zip(phases.outer_iter_mut())
            .zip(sols.di_jones.slice(s![timeblock, .., ..]).outer_iter())
            .for_each(|((mut a, mut p), s)| {
                a.iter_mut()
                    .zip(p.iter_mut())
                    .zip(s.iter())
                    .zip(ref_jones.iter())
                    .for_each(|(((a, p), s), r)| {
                        let div = *s / r;
                        a[0] = div[0].norm();
                        a[1] = div[1].norm();
                        a[2] = div[2].norm();
                        a[3] = div[3].norm();
                        p[0] = div[0].arg();
                        p[1] = div[1].arg();
                        p[2] = div[2].arg();
                        p[3] = div[3].arg();
                    });
            });

        let (min_amp, max_amp) = match (min_amp, max_amp) {
            (Some(user_min), Some(user_max)) => (user_min, user_max),
            _ => {
                // We need to work out the min and max ourselves.
                let (data_min, data_max) = amps.iter().flatten().filter(|a| !a.is_nan()).fold(
                    (f64::INFINITY, 0.0),
                    |(acc_min, acc_max), &a| {
                        let acc_min = if a < acc_min { a } else { acc_min };
                        let acc_max = if a > acc_max { a } else { acc_max };
                        (acc_min, acc_max)
                    },
                );

                // Check any user-specified limits. Are they sensible relative
                // to the data?
                let min_amp = match min_amp {
                    Some(user_min_amp) => {
                        if user_min_amp > data_max {
                            warn!("User-specified plot minimum {user_min_amp} is larger than all data; ignoring");
                            data_min
                        } else {
                            user_min_amp
                        }
                    }
                    None => data_min,
                };
                let max_amp = match max_amp {
                    Some(user_max_amp) => {
                        if user_max_amp < data_min {
                            warn!("User-specified plot maximum {user_max_amp} is smaller than all data; ignoring");
                            data_max
                        } else {
                            user_max_amp
                        }
                    }
                    None => data_max,
                };
                (min_amp, max_amp)
            }
        };

        for (i_tile, (amps, amp_tile_plot)) in amps
            .outer_iter()
            .zip(amps_tile_plots.into_iter())
            .enumerate()
        {
            let tile_name = match tile_names {
                Some(names) => format!("{}: {}", i_tile, names[i_tile].as_ref()),
                None => format!("{}", i_tile),
            };
            plot_amps(
                &amp_tile_plot,
                amps.view(),
                min_amp,
                max_amp,
                &tile_name,
                (i_tile / split.0, i_tile % split.1),
                ignore_cross_pols,
            )?;
        }
        for (i_tile, (phases, phase_tile_plot)) in phases
            .outer_iter()
            .zip(phase_tile_plots.into_iter())
            .enumerate()
        {
            let tile_name = match tile_names {
                Some(names) => format!("{}: {}", i_tile, names[i_tile].as_ref()),
                None => format!("{}", i_tile),
            };
            plot_phases(
                &phase_tile_plot,
                phases.view(),
                &tile_name,
                (i_tile / split.0, i_tile % split.1),
                ignore_cross_pols,
            )?;
        }

        // Finalise the plots.
        amps_root_area.present()?;
        phases_root_area.present()?;
    }

    Ok(output_filenames)
}

/// For a single drawing area, plot gains.
fn plot_amps<DB: DrawingBackend, S: AsRef<str>>(
    drawing_area: &DrawingArea<DB, Shift>,
    amps: ArrayView1<[f64; 4]>,
    min_amp: f64,
    max_amp: f64,
    tile_name: S,
    tile_plot_indices: (usize, usize),
    ignore_cross_pols: bool,
) -> Result<(), DrawError> {
    let x_axis = (0..amps.len()).step(1);
    let y_label_area_size = if tile_plot_indices.1 == 0 { 20 } else { 0 };
    let mut cc = ChartBuilder::on(drawing_area)
        .caption(&tile_name, ("sans-serif", 30))
        .top_x_label_area_size(15)
        .y_label_area_size(y_label_area_size)
        .build_cartesian_2d(0..amps.len(), min_amp..max_amp)
        .map_err(|e| DrawError::Amps(e.to_string()))?;

    cc.configure_mesh()
        .light_line_style(&WHITE)
        .draw()
        .map_err(|e| DrawError::Amps(e.to_string()))?;

    if amps
        .iter()
        .all(|f| f[0].is_nan() || f[1].is_nan() || f[2].is_nan() || f[3].is_nan())
    {
        cc.plotting_area()
            .fill(&RGBColor(220, 220, 220))
            .map_err(|e| DrawError::Amps(e.to_string()))?;
        return Ok(());
    }

    for (pol_index, (_, colour)) in POLS.iter().enumerate() {
        cc.draw_series(PointSeries::of_element(
            x_axis
                .values()
                .zip(amps.iter().map(|g| g[pol_index]))
                .filter(|(_, y)| !y.is_nan())
                .map(|(x, y)| (x, y)),
            1,
            if [0, 3].contains(&pol_index) {
                ShapeStyle::from(&colour).filled()
            } else if ignore_cross_pols {
                ShapeStyle::from(*CLEAR)
            } else {
                ShapeStyle::from(&colour)
            },
            &|coord, size, style| EmptyElement::at(coord) + Circle::new((0, 0), size, style),
        ))
        .map_err(|e| DrawError::Amps(e.to_string()))?;
    }

    Ok(())
}

/// For a single drawing area, plot phases.
fn plot_phases<DB: DrawingBackend, S: AsRef<str>>(
    drawing_area: &DrawingArea<DB, Shift>,
    phases: ArrayView1<[f64; 4]>,
    tile_name: S,
    tile_plot_indices: (usize, usize),
    ignore_cross_pols: bool,
) -> Result<(), DrawError> {
    let x_axis = (0..phases.len()).step(1);
    let y_label_area_size = if tile_plot_indices.1 == 0 { 45 } else { 0 };
    let mut cc = ChartBuilder::on(drawing_area)
        .caption(&tile_name, ("sans-serif", 30))
        .top_x_label_area_size(15)
        .y_label_area_size(y_label_area_size)
        .build_cartesian_2d(0..phases.len(), -180.0..180.0)
        .map_err(|e| DrawError::Phases(e.to_string()))?;

    cc.configure_mesh()
        .light_line_style(&WHITE)
        .draw()
        .map_err(|e| DrawError::Phases(e.to_string()))?;

    if phases
        .iter()
        .all(|f| f[0].is_nan() || f[1].is_nan() || f[2].is_nan() || f[3].is_nan())
    {
        cc.plotting_area()
            .fill(&RGBColor(220, 220, 220))
            .map_err(|e| DrawError::Phases(e.to_string()))?;
        return Ok(());
    }

    for (pol_index, (_, colour)) in POLS.iter().enumerate() {
        cc.draw_series(PointSeries::of_element(
            x_axis
                .values()
                .zip(phases.iter().map(|g| g[pol_index]))
                .filter(|(_, y)| !y.is_nan())
                .map(|(x, y)| {
                    let mut y_val = y.to_degrees();
                    if y_val < -180.0 {
                        y_val += 360.0
                    } else if y_val > 180.0 {
                        y_val -= 360.0
                    }
                    (x, y_val)
                }),
            1,
            if [0, 3].contains(&pol_index) {
                ShapeStyle::from(&colour).filled()
            } else if ignore_cross_pols {
                ShapeStyle::from(*CLEAR)
            } else {
                ShapeStyle::from(&colour)
            },
            &|coord, size, style| EmptyElement::at(coord) + Circle::new((0, 0), size, style),
        ))
        .map_err(|e| DrawError::Phases(e.to_string()))?;
    }

    Ok(())
}

#[derive(Error, Debug)]
pub enum DrawError {
    #[error("While plotting amps: {0}")]
    Amps(String),

    #[error("While plotting phases: {0}")]
    Phases(String),
}
