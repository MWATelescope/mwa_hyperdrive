// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Code to plot calibration solutions.

mod error;

pub(crate) use error::SolutionsPlotError;

use std::path::PathBuf;

use clap::Parser;

use crate::HyperdriveError;

#[derive(Parser, Debug, Default)]
pub(crate) struct SolutionsPlotArgs {
    #[clap(value_name = "SOLUTIONS_FILES")]
    files: Vec<PathBuf>,

    /// The reference tile to use. If this isn't specified, the best one
    /// from the end is used.
    #[clap(short, long)]
    ref_tile: Option<usize>,

    /// Don't use a reference tile. Using this will ignore any input for
    /// `ref_tile`.
    #[clap(short, long)]
    no_ref_tile: bool,

    /// Don't plot the leakage polarisations (D_x and D_y).
    #[clap(long)]
    ignore_cross_pols: bool,

    /// The minimum y-range value on the amplitude gain plots.
    #[clap(long)]
    min_amp: Option<f64>,

    /// The maximum y-range value on the amplitude gain plots.
    #[clap(long)]
    max_amp: Option<f64>,

    /// The number of rows to use in the plots. The default is determined based
    /// off of the number of tiles in the solutions.
    #[clap(long)]
    num_rows: Option<usize>,

    /// The number of columns to use in the plots. The default is determined
    /// based off of the number of tiles in the solutions.
    #[clap(long)]
    num_cols: Option<usize>,

    /// The directory to write the plots into. If this doesn't exist, then the
    /// relevant directories will be created. The filenames are based off of the
    /// input files, just as they would without specifying the output directory.
    #[clap(short, long)]
    output_directory: Option<String>,

    /// The metafits file associated with the solutions. This provides
    /// additional information on the plots, like the tile names.
    #[clap(short, long)]
    metafits: Option<PathBuf>,
}

impl SolutionsPlotArgs {
    pub(crate) fn run(self) -> Result<(), HyperdriveError> {
        plotting::plot_all_sol_files(self)?;
        Ok(())
    }
}

mod plotting {
    use std::str::FromStr;

    use log::{debug, info, warn};
    use marlu::Jones;
    use ndarray::prelude::*;
    use rizzma::{
        artist::Rgba,
        axis::ticker::{FormatStrFormatter, MaxNLocator, NBins},
        Axes, Figure, GridSpec, RcParams,
    };
    use vec1::Vec1;

    use super::*;
    use crate::solutions::{ao, hyperdrive, CalSolutionType, CalibrationSolutions};

    /// The plots are 3200x1800 pixels.
    const WIDTH_INCHES: f64 = 16.0;
    const HEIGHT_INCHES: f64 = 9.0;
    const DPI: f64 = 200.0;

    const POLS: [(&str, Rgba); 4] = [
        ("$g_X$", Rgba::BLUE),
        (
            "$D_X$",
            Rgba {
                a: 0.2,
                ..Rgba::BLUE
            },
        ),
        (
            "$D_Y$",
            Rgba {
                a: 0.2,
                ..Rgba::RED
            },
        ),
        ("$g_Y$", Rgba::RED),
    ];
    const FLAGGED: Rgba = Rgba {
        r: 220.0 / 255.0,
        g: 220.0 / 255.0,
        b: 220.0 / 255.0,
        a: 1.0,
    };
    /// The grid plotters drew at each tick: black at 50/255.
    const GRID: Rgba = Rgba {
        a: 50.0 / 255.0,
        ..Rgba::BLACK
    };

    /// rizzma sizes text, ticks and pads in units that scale with the DPI
    /// (one unit per pixel at 100 DPI); this converts output pixels to them.
    const PX: f64 = 100.0 / DPI;
    /// One output pixel as a line width, which rizzma takes in points.
    const LINE: f64 = 72.0 / DPI;
    /// The plotters layout, in output pixels.
    const TICK_LENGTH: f64 = 5.0;
    const TICK_LABEL_SIZE: f64 = 9.6;

    pub(crate) fn plot_all_sol_files(args: SolutionsPlotArgs) -> Result<(), SolutionsPlotError> {
        let SolutionsPlotArgs {
            files,
            ref_tile,
            no_ref_tile,
            ignore_cross_pols,
            min_amp,
            max_amp,
            num_rows,
            num_cols,
            output_directory,
            metafits,
        } = args;

        if files.is_empty() {
            return Err(SolutionsPlotError::NoInputs);
        }

        let mwalib_context = match metafits.as_deref() {
            Some(m) => Some(mwalib::MetafitsContext::new(m, None)?),
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

        for solutions_file in &files {
            debug!("Plotting solutions for '{}'", solutions_file.display());
            let solutions_file = solutions_file.canonicalize()?;
            let solutions_type = match solutions_file
                .extension()
                .and_then(|os_str| os_str.to_str())
                .and_then(|s| CalSolutionType::from_str(s).ok())
            {
                Some(sol_type) => sol_type,
                None => return Err(SolutionsPlotError::InvalidSolsFormat(solutions_file)),
            };
            let base = solutions_file
                .file_stem()
                .unwrap_or_else(|| {
                    panic!(
                        "Calibration solutions filename '{}' has no file stem",
                        solutions_file.display()
                    );
                })
                .to_str()
                .unwrap_or_else(|| {
                    panic!(
                        "Calibration solutions filename '{}' contains invalid UTF-8",
                        solutions_file.display()
                    )
                });
            let base = if let Some(o) = output_directory.as_deref() {
                let pb = PathBuf::from(o);
                if !pb.exists() {
                    std::fs::create_dir_all(&pb)?;
                }
                pb.join(base)
                    .to_str()
                    .expect("only contains valid UTF-8, as this has been checked above")
                    .to_string()
            } else {
                base.to_string()
            };

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
                // N.B. Not using `crate::cli::Warn` here because multiple
                // calibration solutions may be plotted, and we want the user to
                // see the warnings for each file.
                warn!("No metafits supplied; the obsid and tile names won't be on the plots");
                warned_no_tile_names = true;
            }

            // How should the plot be split up to distribute the tiles?
            let (auto_num_rows, auto_num_cols, tile_name_font_size) = {
                let total_num_tiles = sols.di_jones.len_of(Axis(1));
                let (num_rows, tile_name_font_size) = match total_num_tiles {
                    0..=128 => (8, 30),
                    129..=256 => (10, 24),
                    _ => (16, 18),
                };
                let num_cols = (total_num_tiles as f64 / num_rows as f64).ceil() as usize;
                (num_rows, num_cols, tile_name_font_size)
            };
            let plot_files = plotting::plot_sols(
                &sols,
                &base,
                &plot_title,
                ref_tile,
                no_ref_tile,
                tile_names,
                ignore_cross_pols,
                min_amp,
                max_amp,
                num_rows.unwrap_or(auto_num_rows),
                num_cols.unwrap_or(auto_num_cols),
                tile_name_font_size,
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
        num_rows: usize,
        num_cols: usize,
        tile_name_font_size: i32,
    ) -> Result<Vec<String>, rizzma::skia::PngError> {
        let (num_timeblocks, total_num_tiles, _) = sols.di_jones.dim();

        let mut amps = Array2::from_elem(
            (sols.di_jones.dim().1, sols.di_jones.dim().2),
            [0.0, 0.0, 0.0, 0.0],
        );
        let mut phases = Array2::from_elem(
            (sols.di_jones.dim().1, sols.di_jones.dim().2),
            [0.0, 0.0, 0.0, 0.0],
        );

        let ref_tile = match (no_ref_tile, ref_tile) {
            (true, _) => {
                debug!("Not using a reference tile");
                None
            }
            (_, Some(r)) => {
                debug!("Using user-specified reference tile: {r}");
                Some(r)
            }
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
                    // Include solutions for tiles that (1) aren't all NaN and
                    // (2) aren't singular (this can happen when dealing with
                    // single-pol data).
                    .filter(|(_, j)| !j.iter().all(|f| f.any_nan() || f.inv().any_nan()))
                    .map(|(i, _)| i)
                    .next();
                // If the search for a valid tile didn't find anything, all
                // solutions must be NaN. In this case, it doesn't matter what the
                // reference is.
                let r = possibly_good.map(|g| total_num_tiles - 1 - g);
                debug!("Automatically determined reference tile: {r:?}");
                r
            }
        };

        let mut output_filenames = vec![];
        for timeblock in 0..num_timeblocks {
            let (output_amps, output_phases) = if num_timeblocks > 1 {
                (
                    format!("{filename_base}_amps_{timeblock:03}.png"),
                    format!("{filename_base}_phases_{timeblock:03}.png"),
                )
            } else {
                (
                    format!("{filename_base}_amps.png"),
                    format!("{filename_base}_phases.png"),
                )
            };

            // Draw the reference tile number and the GPS times for this
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
                        s.to_gpst_seconds(),
                        e.to_gpst_seconds(),
                        a.to_gpst_seconds()
                    )
                }
                (Some(s), Some(e), None) => format!(
                    "GPS start {}, end {}",
                    s.to_gpst_seconds(),
                    e.to_gpst_seconds()
                ),
                (Some(s), None, None) => format!("GPS start {}, end unknown", s.to_gpst_seconds()),
                (None, Some(e), None) => format!("GPS start unknown, end {}", e.to_gpst_seconds()),
                (Some(s), None, Some(a)) => format!(
                    "GPS start {}, end unknown, average {}",
                    s.to_gpst_seconds(),
                    a.to_gpst_seconds()
                ),
                (None, Some(e), Some(a)) => format!(
                    "GPS start unknown, end {}, average {}",
                    e.to_gpst_seconds(),
                    a.to_gpst_seconds()
                ),
                (None, None, Some(a)) => format!(
                    "GPS start unknown, end unknown, average {}",
                    a.to_gpst_seconds()
                ),
                (None, None, None) => String::new(),
            };
            if !meta_str.is_empty() && !time_str.is_empty() {
                meta_str.push_str(", ");
            }
            meta_str.push_str(&time_str);

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
                            p[0] = div[0].arg().to_degrees();
                            p[1] = div[1].arg().to_degrees();
                            p[2] = div[2].arg().to_degrees();
                            p[3] = div[3].arg().to_degrees();
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

            let mut amps_fig = new_figure(
                &format!("Amps for {obs_name}"),
                &meta_str,
                tile_name_font_size,
                ignore_cross_pols,
            );
            let mut phases_fig = new_figure(
                &format!("Phases for {obs_name}"),
                &meta_str,
                tile_name_font_size,
                ignore_cross_pols,
            );
            let grid = GridSpec::new(num_rows, num_cols)
                .with_margins(0.025, 0.995, 0.005, 0.93)
                .with_spacing(0.03, 0.4);
            let num_plotted = total_num_tiles.min(num_rows * num_cols);
            for (i_tile, (amps, phases)) in amps
                .outer_iter()
                .zip(phases.outer_iter())
                .take(num_plotted)
                .enumerate()
            {
                let tile_name = match tile_names {
                    Some(names) => format!("{}: {}", i_tile, names[i_tile]),
                    None => format!("{i_tile}"),
                };
                // Every tile labels its channels along the top; only the first
                // column labels the y axis.
                let labels = (true, i_tile % num_cols == 0);
                let cell = grid
                    .subplot(i_tile / num_cols, i_tile % num_cols)
                    .get_position(&grid);
                let rect = (cell.x0, cell.y0, cell.width(), cell.height());
                plot_tile(
                    amps_fig.add_axes(rect.0, rect.1, rect.2, rect.3),
                    amps,
                    (min_amp, max_amp),
                    &tile_name,
                    labels,
                    ignore_cross_pols,
                );
                plot_tile(
                    phases_fig.add_axes(rect.0, rect.1, rect.2, rect.3),
                    phases,
                    (-180.0, 180.0),
                    &tile_name,
                    labels,
                    ignore_cross_pols,
                );
            }
            amps_fig.save_png(&output_amps)?;
            phases_fig.save_png(&output_phases)?;
            output_filenames.push(output_amps);
            output_filenames.push(output_phases);
        }

        Ok(output_filenames)
    }

    /// A blank figure with the title, the timeblock metadata in the top left
    /// and the polarisation colour key in the top right.
    fn new_figure(
        title: &str,
        meta: &str,
        tile_name_font_size: i32,
        ignore_cross_pols: bool,
    ) -> Figure {
        // plotters drew a caption of font size N at 0.8 N pixels.
        let mut fig = Figure::new(WIDTH_INCHES, HEIGHT_INCHES)
            .with_dpi(DPI)
            .with_rcparams(RcParams {
                axes_titlesize: 0.8 * f64::from(tile_name_font_size) * PX,
                axes_titlepad: 2.0 * PX,
                xtick_labelsize: TICK_LABEL_SIZE * PX,
                ytick_labelsize: TICK_LABEL_SIZE * PX,
                xtick_major_size: TICK_LENGTH * PX,
                ytick_major_size: TICK_LENGTH * PX,
                xtick_major_pad: 5.0 * PX,
                ytick_major_pad: 3.0 * PX,
                ..RcParams::default()
            });
        fig.suptitle(title);
        let header = fig.add_axes(0.0, 0.0, 1.0, 1.0);
        header.set_axis_off().set_xlim(0.0, 1.0).set_ylim(0.0, 1.0);
        header.text(0.005, 0.975, meta);
        for (i, (label, colour)) in POLS.iter().enumerate() {
            if ignore_cross_pols && [1, 2].contains(&i) {
                continue;
            }
            header.text_with_color(0.86 + 0.03 * i as f64, 0.975, *label, *colour);
        }
        fig
    }

    /// Scatter one tile's four polarisations against channel, or grey the
    /// tile out if every channel is flagged.
    fn plot_tile(
        ax: &mut Axes,
        values: ArrayView1<[f64; 4]>,
        (y_min, y_max): (f64, f64),
        tile_name: &str,
        (x_labels, y_labels): (bool, bool),
        ignore_cross_pols: bool,
    ) {
        // As plotters did: up to 10 round ticks per axis, labelled along the
        // top and left with a thin grid through each, and no frame box.
        let num_chans = values.len();
        let key_points = |integer| {
            Box::new(MaxNLocator::with_steps(
                NBins::Fixed(10),
                &[1.0, 2.0, 5.0, 10.0],
                integer,
                false,
                2,
            ))
        };
        ax.set_title(tile_name)
            .set_xlim(0.0, num_chans as f64)
            .set_ylim(y_min, y_max)
            .set_frame_on(false)
            .grid_with(GRID, LINE, 1.0);
        ax.xaxis_mut()
            .tick_top()
            .set_locator(key_points(true))
            .set_tick_width(LINE)
            .set_tick_labels_visible(x_labels);
        ax.yaxis_mut()
            .set_locator(key_points(false))
            .set_formatter(Box::new(FormatStrFormatter::new("%.1f")))
            .set_tick_width(LINE)
            .set_tick_labels_visible(y_labels);

        if values.iter().all(|v| v.iter().any(|f| f.is_nan())) {
            ax.set_facecolor(FLAGGED);
            return;
        }

        for (pol_index, (_, colour)) in POLS.iter().enumerate() {
            let cross_pol = [1, 2].contains(&pol_index);
            if cross_pol && ignore_cross_pols {
                continue;
            }
            let (x, y): (Vec<f64>, Vec<f64>) = values
                .iter()
                .enumerate()
                .map(|(i, v)| (i as f64, v[pol_index]))
                .filter(|(_, y)| !y.is_nan())
                .unzip();
            // Gains are filled dots, leakages hollow rings, both 3 pixels
            // across as plotters drew them (marker sizes are points squared).
            let points = ax.scatter(&x, &y);
            *points = if cross_pol {
                points
                    .clone()
                    .with_facecolors(vec![Rgba { a: 0.0, ..*colour }])
                    .with_edgecolors(vec![*colour])
                    .linewidth(LINE)
            } else {
                points.clone().with_facecolors(vec![*colour])
            }
            .with_sizes(vec![(3.0 * LINE).powi(2)]);
        }
    }
}
