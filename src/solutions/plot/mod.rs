// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Code to plot calibration solutions.

mod error;

pub(crate) use error::SolutionsPlotError;

use std::path::PathBuf;

use clap::Parser;

use crate::HyperdriveError;
use mwa_hyperdrive_common::clap;

#[derive(Parser, Debug, Default)]
pub struct SolutionsPlotArgs {
    #[clap(name = "SOLUTIONS_FILES", parse(from_os_str))]
    files: Vec<PathBuf>,

    /// The reference tile to use. If this isn't specified, the best one
    /// from the end is used.
    #[clap(short, long)]
    ref_tile: Option<usize>,

    /// Don't use a reference tile. Using this will ignore any input for
    /// `ref_tile`.
    #[clap(short, long)]
    no_ref_tile: bool,

    /// Don't plot XY and YX polarisations.
    #[clap(long)]
    ignore_cross_pols: bool,

    /// The minimum y-range value on the amplitude gain plots.
    #[clap(long)]
    min_amp: Option<f64>,

    /// The maximum y-range value on the amplitude gain plots.
    #[clap(long)]
    max_amp: Option<f64>,

    /// The metafits file associated with the solutions. This provides
    /// additional information on the plots, like the tile names.
    #[clap(short, long, parse(from_str))]
    metafits: Option<PathBuf>,
}

impl SolutionsPlotArgs {
    #[cfg(not(feature = "plotting"))]
    pub fn run(self) -> Result<(), HyperdriveError> {
        // Plotting is an optional feature. This is because it doesn't look
        // possible to statically compile the C dependencies needed for
        // plotting. If the "plotting" feature isn't available, warn the user
        // that they'll need to compile hyperdrive from source.
        return Err(HyperdriveError::from(SolutionsPlotError::NoPlottingFeature));
    }

    #[cfg(feature = "plotting")]
    pub fn run(self) -> Result<(), HyperdriveError> {
        plotting::plot_all_sol_files(self)?;
        Ok(())
    }
}

#[cfg(feature = "plotting")]
mod plotting {
    use std::str::FromStr;

    use log::{debug, info, trace, warn};
    use marlu::Jones;
    use ndarray::prelude::*;
    use plotters::{
        coord::Shift,
        prelude::*,
        style::{Color, RGBAColor},
    };
    use thiserror::Error;
    use vec1::Vec1;

    use super::*;
    use crate::solutions::{ao, hyperdrive, CalSolutionType, CalibrationSolutions};
    use mwa_hyperdrive_common::{lazy_static, log, marlu, mwalib, ndarray, thiserror, vec1};

    /// The number of X pixels on the plots.
    const X_PIXELS: u32 = 3200;
    /// The number of Y pixels on the plots.
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

    pub(crate) fn plot_all_sol_files(args: SolutionsPlotArgs) -> Result<(), SolutionsPlotError> {
        if args.files.is_empty() {
            return Err(SolutionsPlotError::NoInputs);
        }

        let mwalib_context = match args.metafits.as_deref() {
            Some(m) => Some(mwalib::MetafitsContext::new(&m, None)?),
            None => None,
        };
        let mwalib_tile_names = match mwalib_context.as_ref() {
            Some(c) => {
                // TODO: Make mwalib provide SoA, not AoS
                let names = c
                    .antennas
                    .iter()
                    .map(|a| a.tile_name.clone())
                    .collect::<Vec<String>>();
                Some(
                    Vec1::try_from_vec(names)
                        .map_err(|_| SolutionsPlotError::MetafitsNoAntennaNames)?,
                )
            }
            None => None,
        };

        // Have we warned the user that tile names won't be on the plots?
        let mut warned_no_tile_names = false;

        for solutions_file in &args.files {
            debug!("Plotting solutions for '{}'", solutions_file.display());
            let solutions_file = solutions_file.canonicalize()?;
            let ext = solutions_file
                .extension()
                .and_then(|os_str| os_str.to_str());
            let solutions_type = match ext.and_then(|s| CalSolutionType::from_str(s).ok()) {
                Some(sol_type) => {
                    trace!("{} is a visibility output", solutions_file.display());
                    sol_type
                }
                None => return Err(SolutionsPlotError::InvalidSolsFormat(solutions_file)),
            };
            let base = solutions_file
                .file_stem()
                .and_then(|os_str| os_str.to_str())
                .unwrap_or_else(|| {
                    panic!(
                        "Calibration solutions filename '{}' contains invalid UTF-8",
                        solutions_file.display()
                    )
                });

            let sols = match solutions_type {
                CalSolutionType::Fits => hyperdrive::read(&solutions_file)?,
                CalSolutionType::Bin => ao::read(&solutions_file)?,
            };
            let plot_title = format!(
                "obsid {}",
                sols.obsid
                    .or_else(|| mwalib_context.as_ref().map(|m| m.obs_id))
                    .map(|o| o.to_string())
                    .unwrap_or_else(|| "<unknown>".to_string())
            );
            let tile_names = sols.tile_names.as_ref().or(mwalib_tile_names.as_ref());
            if tile_names.is_none() && !warned_no_tile_names {
                warn!("No metafits supplied; the obsid and tile names won't be on the plots");
                warned_no_tile_names = true;
            }

            let plot_files = plotting::plot_sols(
                &sols,
                base,
                &plot_title,
                args.ref_tile,
                args.no_ref_tile,
                tile_names,
                args.ignore_cross_pols,
                args.min_amp,
                args.max_amp,
            )?;
            info!("Wrote {:?}", plot_files);
        }

        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn plot_sols(
        sols: &CalibrationSolutions,
        filename_base: &str,
        obs_name: &str,
        ref_tile: Option<usize>,
        no_ref_tile: bool,
        tile_names: Option<&Vec1<String>>,
        ignore_cross_pols: bool,
        min_amp: Option<f64>,
        max_amp: Option<f64>,
    ) -> Result<Vec<String>, DrawError> {
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
                    // Search only in the first timeblock
                    .outer_iter()
                    // Search by tile from the end
                    .rev()
                    .enumerate()
                    // Keep only those that aren't all NaN
                    .filter(|(_, j)| !j.iter().all(|f| f.any_nan()))
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
                let filename = format!("{}_amps_{:03}.png", filename_base, timeblock);
                output_amps.set_file_name(&filename);
                output_filenames.push(filename);

                let filename = format!("{}_phases_{:03}.png", filename_base, timeblock);
                output_phases.set_file_name(&filename);
                output_filenames.push(filename);
            } else {
                let filename = format!("{}_amps.png", filename_base);
                output_amps.set_file_name(&filename);
                output_filenames.push(filename);

                let filename = format!("{}_phases.png", filename_base);
                output_phases.set_file_name(&filename);
                output_filenames.push(filename);
            }

            let amps_root_area =
                BitMapBackend::new(&output_amps, (X_PIXELS, Y_PIXELS)).into_drawing_area();
            let phases_root_area =
                BitMapBackend::new(&output_phases, (X_PIXELS, Y_PIXELS)).into_drawing_area();
            amps_root_area
                .fill(&WHITE)
                .map_err(|e| DrawError::Plotters(Box::new(e)))?;
            phases_root_area
                .fill(&WHITE)
                .map_err(|e| DrawError::Plotters(Box::new(e)))?;
            // Draw the coloured text for each polarisation.
            for (i, (pol, colour)) in POLS.iter().enumerate() {
                if ignore_cross_pols && [1, 2].contains(&i) {
                    continue;
                }
                amps_root_area
                    .draw_text(
                        *pol,
                        &("sans-serif", 55).into_font().color(&colour),
                        (X_PIXELS as i32 - 500 + 80 * i as i32, 10),
                    )
                    .map_err(|e| DrawError::Plotters(Box::new(e)))?;
                phases_root_area
                    .draw_text(
                        *pol,
                        &("sans-serif", 55).into_font().color(&colour),
                        (X_PIXELS as i32 - 500 + 80 * i as i32, 10),
                    )
                    .map_err(|e| DrawError::Plotters(Box::new(e)))?;
            }
            // Also draw the reference tile number and the GPS times for this
            // timeblock.
            let mut meta_str = match ref_tile {
                Some(ref_tile) => format!("Ref. tile {ref_tile}"),
                None => String::new(),
            };
            let time_str = match (
                sols.start_timestamps
                    .as_ref()
                    .and_then(|t| t.get(timeblock)),
                sols.end_timestamps.as_ref().and_then(|t| t.get(timeblock)),
                sols.average_timestamps
                    .as_ref()
                    .and_then(|t| t.get(timeblock)),
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
            amps_root_area
                .draw_text(
                    &meta_str,
                    &("sans-serif", 38).into_font().color(&BLACK),
                    (10, 10),
                )
                .map_err(|e| DrawError::Plotters(Box::new(e)))?;
            phases_root_area
                .draw_text(
                    &meta_str,
                    &("sans-serif", 38).into_font().color(&BLACK),
                    (10, 10),
                )
                .map_err(|e| DrawError::Plotters(Box::new(e)))?;

            let amps_root_area = amps_root_area
                .shrink((15, 0), (X_PIXELS - 15, Y_PIXELS))
                .titled(&format!("Amps for {}", obs_name), title_style.clone())
                .map_err(|e| DrawError::Plotters(Box::new(e)))?;
            let amps_tile_plots = amps_root_area.split_evenly(split);

            let phases_root_area = phases_root_area
                .titled(&format!("Phases for {}", obs_name), title_style.clone())
                .map_err(|e| DrawError::Plotters(Box::new(e)))?;
            let phase_tile_plots = phases_root_area.split_evenly(split);

            let ones = Array1::from_elem(sols.di_jones.dim().2, Jones::identity());
            let ref_jones = if let Some(ref_tile) = ref_tile {
                sols.di_jones.slice(s![timeblock, ref_tile, ..])
            } else {
                ones.view()
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

                    // Failing all else, make sure the limits are sensible.
                    let min_amp = if min_amp.is_infinite() { 0.0 } else { min_amp };
                    let max_amp = if max_amp.abs() < f64::EPSILON {
                        1.0
                    } else {
                        max_amp
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
                    Some(names) => format!("{}: {}", i_tile, names[i_tile]),
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
                    Some(names) => format!("{}: {}", i_tile, names[i_tile]),
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
            amps_root_area
                .present()
                .map_err(|e| DrawError::Plotters(Box::new(e)))?;
            phases_root_area
                .present()
                .map_err(|e| DrawError::Plotters(Box::new(e)))?;
        }

        Ok(output_filenames)
    }

    /// For a single drawing area, plot gains.
    fn plot_amps<DB: DrawingBackend>(
        drawing_area: &DrawingArea<DB, Shift>,
        amps: ArrayView1<[f64; 4]>,
        min_amp: f64,
        max_amp: f64,
        tile_name: &str,
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
    fn plot_phases<DB: DrawingBackend>(
        drawing_area: &DrawingArea<DB, Shift>,
        phases: ArrayView1<[f64; 4]>,
        tile_name: &str,
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

        #[error("Error from the plotters library: {0}")]
        Plotters(Box<dyn std::error::Error>),
    }
}
