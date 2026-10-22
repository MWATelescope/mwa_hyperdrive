// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! The crosstalk outlier figures.
//!
//! Every per-tile axis is in the order given by [`Tiles::order`], so that
//! tiles sharing a receiver sit together and a crosstalk hub shows up as one
//! bright row or column. Points and strips are coloured by cable flavour.

use log::info;
use rizzma::{
    artist::{Collection, Patch, Rgba},
    Axes, Figure, RcParams,
};

use super::{
    data::{self, BaselineSummary, ObsScores, POLS},
    tiles::{GroupBy, GroupSpan, Tiles},
    CrosstalkPlotError,
};

/// rizzma sizes text in pixels at 100 DPI, so these are output pixels.
const DPI: f64 = 100.0;
const TITLE_SIZE: f64 = 22.0;
const LABEL_SIZE: f64 = 13.0;
const TICK_SIZE: f64 = 10.0;
/// One output pixel as a line width, which rizzma takes in points.
const LINE: f64 = 72.0 / DPI;

const GREY: Rgba = Rgba {
    r: 0.6,
    g: 0.6,
    b: 0.6,
    a: 0.25,
};
const SEPARATOR: Rgba = Rgba {
    r: 0.0,
    g: 0.0,
    b: 0.0,
    a: 0.6,
};
/// Mixed-flavour baselines, which have no single flavour colour.
const MIXED: Rgba = Rgba {
    r: 0.15,
    g: 0.15,
    b: 0.15,
    a: 0.8,
};
/// The outline of each receiver's own block on the baseline matrix.
const BLOCK: Rgba = Rgba {
    r: 0.0,
    g: 0.75,
    b: 0.75,
    a: 1.0,
};
const THRESHOLD: Rgba = Rgba {
    r: 0.85,
    g: 0.1,
    b: 0.1,
    a: 0.9,
};
const CMAP: &str = "magma";

/// A night of crosstalk scores, arranged for plotting.
pub(super) struct Night<'a> {
    night: &'a [ObsScores],
    tiles: &'a Tiles,
    group_by: GroupBy,
    /// Tile indices in plotting order.
    order: Vec<usize>,
    /// Plotting position of each tile index.
    position: Vec<usize>,
    spans: Vec<GroupSpan>,
    /// Flagged baselines per tile per observation, `[pol][tile][obs]`.
    counts: Vec<Vec<Vec<f64>>>,
    /// `[pol]`
    baselines: Vec<Vec<BaselineSummary>>,
    z_min: f64,
    num_outliers: usize,
}

impl<'a> Night<'a> {
    pub(super) fn new(
        night: &'a [ObsScores],
        tiles: &'a Tiles,
        group_by: GroupBy,
        z_min: f64,
        num_outliers: usize,
    ) -> Night<'a> {
        let (order, spans) = tiles.order(group_by);
        let mut position = vec![0; order.len()];
        for (pos, &tile) in order.iter().enumerate() {
            position[tile] = pos;
        }
        Night {
            night,
            tiles,
            group_by,
            order,
            position,
            spans,
            counts: (0..POLS.len())
                .map(|pol| data::tile_flag_counts(night, tiles.len(), pol))
                .collect(),
            baselines: (0..POLS.len())
                .map(|pol| data::baseline_summaries(night, pol))
                .collect(),
            z_min,
            num_outliers,
        }
    }

    pub(super) fn num_scored_tiles(&self) -> usize {
        (0..self.tiles.len())
            .filter(|&t| {
                self.counts
                    .iter()
                    .any(|pol| pol[t].iter().any(|c| !c.is_nan()))
            })
            .count()
    }

    /// The night's median flag count of a tile in a polarisation.
    fn median_count(&self, pol: usize, tile: usize) -> f64 {
        data::median(self.counts[pol][tile].iter().copied())
    }

    /// The tiles with the largest median flag counts, worst first.
    fn top_tiles(&self, pol: usize) -> Vec<(usize, f64)> {
        let mut tiles: Vec<(usize, f64)> = (0..self.tiles.len())
            .map(|t| (t, self.median_count(pol, t)))
            .filter(|(_, m)| *m > 0.0)
            .collect();
        tiles.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
        tiles.truncate(self.num_outliers);
        tiles
    }

    /// The top tiles worth a label on a plot: those with at least half the
    /// worst tile's median, so that a crowd of tiles with one flag each
    /// doesn't bury the outliers in text.
    fn labelled_tiles(&self, pol: usize) -> Vec<(usize, f64)> {
        let top = self.top_tiles(pol);
        let worst = top.first().map(|t| t.1).unwrap_or(0.0);
        top.into_iter()
            .filter(|(_, m)| *m >= (0.5 * worst).max(1.0))
            .collect()
    }

    /// The most persistently flagged baselines, then by median z. Only
    /// baselines flagged in at least half of the observations count.
    fn top_baselines(&self, pol: usize) -> Vec<&BaselineSummary> {
        let mut b: Vec<&BaselineSummary> = self.baselines[pol]
            .iter()
            .filter(|b| b.num_flagged > 0 && b.persistence() >= 0.5)
            .collect();
        b.sort_by(|a, b| {
            b.persistence()
                .total_cmp(&a.persistence())
                .then(b.median_z.total_cmp(&a.median_z))
        });
        b.truncate(self.num_outliers);
        b
    }

    fn tile_name(&self, tile: usize) -> &str {
        &self.tiles.tiles[tile].name
    }

    /// Log the outlier tiles and baselines.
    pub(super) fn report(&self) {
        info!(
            "{} observations, {} tiles with scored baselines",
            self.night.len(),
            self.num_scored_tiles()
        );
        for (pol, pol_name) in POLS.iter().enumerate() {
            for (tile, median) in self.top_tiles(pol) {
                let t = &self.tiles.tiles[tile];
                info!(
                    "{pol_name} tile {:<8} Rx{:02} {:<10} median {median} flagged baselines per obs",
                    t.name, t.rx, t.flavour
                );
            }
            for b in self.top_baselines(pol) {
                info!(
                    "{pol_name} baseline {}-{} ({:.1} m): flagged in {}/{} obs, median z {:.1}",
                    self.tile_name(b.tile1),
                    self.tile_name(b.tile2),
                    b.length_m,
                    b.num_flagged,
                    b.num_obs,
                    b.median_z
                );
            }
        }
    }

    /// Write every figure, returning the filenames.
    pub(super) fn plot_all(&self, base: &str) -> Result<Vec<String>, CrosstalkPlotError> {
        let figures = [
            ("tile_obs", self.tile_obs()),
            ("tile_spread", self.tile_spread()),
            ("baseline_matrix", self.baseline_matrix()),
            ("z_vs_length", self.z_vs_length()),
            ("peak_delay", self.peak_delay()),
        ];
        let mut files = vec![];
        for (name, fig) in figures {
            let file = format!("{base}_{name}.png");
            // Encode before writing so that a file-system failure is an IO
            // error rather than a PNG one.
            std::fs::write(&file, fig.encode_png()?)?;
            files.push(file);
        }
        Ok(files)
    }

    fn group_description(&self) -> &'static str {
        match self.group_by {
            GroupBy::ReceiverFlavour => "tiles grouped by receiver, then cable flavour",
            GroupBy::Receiver => "tiles grouped by receiver",
            GroupBy::Flavour => "tiles grouped by cable flavour",
        }
    }

    /// The observations the plots cover.
    fn obs_range(&self) -> String {
        let first = self.night.first().map(|o| o.label.as_str()).unwrap_or("");
        let last = self.night.last().map(|o| o.label.as_str()).unwrap_or("");
        format!("{} obs ({first} to {last})", self.night.len())
    }

    /// The subtitle of the per-tile plots.
    fn subtitle(&self) -> String {
        format!(
            "{}; {}; colour = cable flavour",
            self.obs_range(),
            self.group_description()
        )
    }

    /// The colour of a baseline: its flavour if both tiles share one,
    /// otherwise the mixed colour.
    fn pair_colour(&self, tile1: usize, tile2: usize) -> Rgba {
        let (a, b) = (&self.tiles.tiles[tile1], &self.tiles.tiles[tile2]);
        if a.flavour == b.flavour {
            self.tiles.flavour_colour(&a.flavour)
        } else {
            MIXED
        }
    }

    fn pair_key(&self) -> Vec<(Rgba, String)> {
        let mut key: Vec<(Rgba, String)> = self
            .tiles
            .flavour_key()
            .into_iter()
            .map(|(c, f)| (c, format!("{f} x {f}")))
            .collect();
        key.push((MIXED, "mixed flavours".to_string()));
        key.push((GREY, "not flagged".to_string()));
        key
    }

    /// Fig 1: flagged baselines touching each tile, through the night. A tile
    /// that is a crosstalk source is a bright row across every observation.
    fn tile_obs(&self) -> Figure {
        let num_tiles = self.order.len();
        let num_obs = self.night.len();
        let mut fig = new_figure(
            20.0,
            18.0,
            "Flagged baselines per tile per observation",
            &self.subtitle(),
        );
        let vmax = self
            .counts
            .iter()
            .flatten()
            .flatten()
            .copied()
            .filter(|c| c.is_finite())
            .fold(1.0, f64::max);
        for (pol, pol_name) in POLS.iter().enumerate() {
            let left = 0.07 + 0.46 * pol as f64;
            let ax = fig.add_axes(left, 0.06, 0.38, 0.86);
            let image: Vec<f64> = self
                .order
                .iter()
                .flat_map(|&t| self.counts[pol][t].iter().copied())
                .collect();
            ax.imshow(&image, num_tiles, num_obs)
                .cmap(CMAP)
                .vmin(0.0)
                .vmax(vmax);
            ax.set_title(*pol_name);
            self.flavour_strip_rows(
                ax,
                -0.04 * num_obs as f64 - 0.5,
                0.03 * num_obs as f64 + 0.3,
            );
            ax.set_xlim(-0.05 * num_obs as f64 - 0.6, num_obs as f64)
                .set_ylim(0.0, num_tiles as f64);
            for span in self.spans.iter().skip(1) {
                ax.axhline_with((num_tiles - span.start) as f64, SEPARATOR, 1.5 * LINE);
            }
            self.tile_ticks_y(ax, num_tiles);
            let step = num_obs.div_ceil(8).max(1);
            let ticks: Vec<usize> = (0..num_obs).step_by(step).collect();
            ax.set_xticks(&ticks.iter().map(|&i| i as f64 + 0.5).collect::<Vec<_>>())
                .set_xticklabels(
                    &ticks
                        .iter()
                        .map(|&i| self.night[i].label.as_str())
                        .collect::<Vec<_>>(),
                );
            ax.xaxis_mut().set_tick_label_size(TICK_SIZE);
            ax.set_xlabel("observation");
        }
        fig.colorbar_at((0.93, 0.06, 0.015, 0.86), CMAP, 0.0, vmax);
        fig
    }

    /// Fig 2: each tile's flag count in every observation, with the night's
    /// median, so outliers stand out from the rest of their receiver.
    fn tile_spread(&self) -> Figure {
        let num_tiles = self.order.len();
        let mut fig = new_figure(
            22.0,
            12.0,
            "Flagged baselines per tile, all observations",
            &self.subtitle(),
        );
        for (pol, pol_name) in POLS.iter().enumerate() {
            let bottom = 0.53 - 0.46 * pol as f64;
            let ax = fig.add_axes(0.05, bottom, 0.82, 0.37);
            let mut x = vec![];
            let mut y = vec![];
            let mut colours = vec![];
            let mut median_x = vec![];
            let mut median_y = vec![];
            for (pos, &tile) in self.order.iter().enumerate() {
                for &c in self.counts[pol][tile].iter().filter(|c| c.is_finite()) {
                    x.push(pos as f64);
                    y.push(c);
                    colours.push(Rgba {
                        a: 0.5,
                        ..self.tiles.tile_colour(tile)
                    });
                }
                let m = self.median_count(pol, tile);
                if m.is_finite() {
                    median_x.push(pos as f64);
                    median_y.push(m);
                }
            }
            let y_max = y.iter().copied().fold(1.0, f64::max);
            scatter(ax, &x, &y, colours, 4.0);
            scatter(ax, &median_x, &median_y, vec![Rgba::BLACK], 3.5);
            ax.set_xlim(-1.0, num_tiles as f64)
                .set_ylim(-0.5, y_max * 1.15 + 0.5)
                .set_title(*pol_name)
                .set_ylabel("flagged baselines");
            for span in self.spans.iter().skip(1) {
                ax.axvline_with(span.start as f64 - 0.5, SEPARATOR, LINE);
            }
            ax.set_text_size(TICK_SIZE);
            for (tile, median) in self.labelled_tiles(pol) {
                let pos = self.position[tile] as f64;
                ax.text(pos + 0.6, median + 0.02 * y_max, self.tile_name(tile));
            }
            self.group_ticks_x(ax);
            ax.legend(self.tiles.flavour_key());
        }
        fig
    }

    /// Fig 3: the night's median z of every baseline, XX below the diagonal
    /// and YY above. Hot cells inside a teal block are pairs on the same
    /// receiver; hot cells outside cross receivers.
    fn baseline_matrix(&self) -> Figure {
        let n = self.order.len();
        let mut fig = new_figure(
            16.0,
            15.0,
            "Median z per baseline: XX lower left, YY upper right",
            &self.subtitle(),
        );
        let mut image = vec![f64::NAN; n * n];
        for (pol, baselines) in self.baselines.iter().enumerate() {
            for b in baselines {
                let (p1, p2) = (self.position[b.tile1], self.position[b.tile2]);
                let (row, col) = if (pol == 0) == (p1 > p2) {
                    (p1, p2)
                } else {
                    (p2, p1)
                };
                image[row * n + col] = b.median_z;
            }
        }
        let vmax = (2.0 * self.z_min).max(1.0);
        let ax = fig.add_axes(0.08, 0.05, 0.80, 0.85);
        ax.imshow(&image, n, n).cmap(CMAP).vmin(0.0).vmax(vmax);
        for span in &self.spans {
            let len = (span.end - span.start) as f64;
            ax.add_patch(
                Patch::rectangle(span.start as f64, (n - span.end) as f64, len, len)
                    .facecolor(None)
                    .edgecolor(Some(BLOCK))
                    .linewidth(1.5 * LINE),
            );
        }
        let pad = 0.03 * n as f64;
        self.flavour_strip_rows(ax, -pad - 0.5, pad);
        for (pos, &tile) in self.order.iter().enumerate() {
            ax.add_patch(
                Patch::rectangle(pos as f64, n as f64 + 0.5, 1.0, pad)
                    .facecolor(Some(self.tiles.tile_colour(tile)))
                    .edgecolor(None),
            );
        }
        ax.set_xlim(-pad - 0.6, n as f64)
            .set_ylim(0.0, n as f64 + pad + 0.6);
        self.group_ticks_x(ax);
        let mids: Vec<f64> = self
            .spans
            .iter()
            .map(|s| n as f64 - 0.5 * (s.start + s.end) as f64)
            .collect();
        let labels: Vec<&str> = self.spans.iter().map(|s| s.label.as_str()).collect();
        ax.set_yticks(&mids).set_yticklabels(&labels);
        ax.yaxis_mut().set_tick_label_size(TICK_SIZE * 0.8);
        fig.colorbar_at((0.91, 0.05, 0.015, 0.85), CMAP, 0.0, vmax);
        fig
    }

    /// Fig 4: z against baseline length for every baseline in every
    /// observation. Persistent crosstalk is a vertical stripe of flagged
    /// points at one length.
    fn z_vs_length(&self) -> Figure {
        let mut fig = new_figure(
            18.0,
            13.0,
            "Crosstalk z against baseline length",
            &format!(
                "{}; flagged points coloured by flavour pair; labels: baselines flagged in at least half the obs; red line z = {}",
                self.obs_range(),
                self.z_min
            ),
        );
        for (pol, pol_name) in POLS.iter().enumerate() {
            let bottom = 0.53 - 0.46 * pol as f64;
            let ax = fig.add_axes(0.07, bottom, 0.88, 0.37);
            let (mut x, mut y) = (vec![], vec![]);
            let (mut fx, mut fy, mut fc) = (vec![], vec![], vec![]);
            for s in self
                .night
                .iter()
                .flat_map(|o| o.scores.iter())
                .filter(|s| s.pol == pol && s.z.is_finite())
            {
                if s.flag {
                    fx.push(s.length_m);
                    fy.push(s.z);
                    fc.push(self.pair_colour(s.tile1, s.tile2));
                } else {
                    x.push(s.length_m);
                    y.push(s.z);
                }
            }
            scatter(ax, &x, &y, vec![GREY], 2.0);
            scatter(ax, &fx, &fy, fc, 4.0);
            ax.axhline_with(self.z_min, THRESHOLD, 1.5 * LINE);
            ax.set_title(*pol_name)
                .set_xlabel("baseline length (m)")
                .set_ylabel("z");
            ax.set_text_size(TICK_SIZE);
            for b in self.top_baselines(pol) {
                ax.text(
                    b.length_m,
                    b.median_z,
                    format!("{}-{}", self.tile_name(b.tile1), self.tile_name(b.tile2)),
                );
            }
            ax.legend(self.pair_key());
        }
        fig
    }

    /// Fig 5: peak delay against z for the flagged baselines. Crosstalk
    /// through a fixed signal path keeps the same delay all night.
    fn peak_delay(&self) -> Figure {
        let mut fig = new_figure(
            18.0,
            13.0,
            "Peak delay of flagged baselines",
            &format!("{}; coloured by flavour pair", self.obs_range()),
        );
        for (pol, pol_name) in POLS.iter().enumerate() {
            let bottom = 0.53 - 0.46 * pol as f64;
            let ax = fig.add_axes(0.07, bottom, 0.88, 0.37);
            let (mut x, mut y, mut c) = (vec![], vec![], vec![]);
            for s in self
                .night
                .iter()
                .flat_map(|o| o.scores.iter())
                .filter(|s| s.pol == pol && s.flag && s.tau_peak_ns.is_finite())
            {
                x.push(s.tau_peak_ns);
                y.push(s.z);
                c.push(self.pair_colour(s.tile1, s.tile2));
            }
            scatter(ax, &x, &y, c, 4.0);
            ax.axvline_with(0.0, SEPARATOR, LINE);
            ax.set_title(*pol_name)
                .set_xlabel("peak delay (ns), ant1 to ant2")
                .set_ylabel("z");
            let mut key = self.pair_key();
            key.pop();
            ax.legend(key);
        }
        fig
    }

    /// A column of cable-flavour cells beside the rows of a per-tile image
    /// whose row 0 (first in plotting order) is at the top.
    fn flavour_strip_rows(&self, ax: &mut Axes, x: f64, width: f64) {
        let n = self.order.len();
        for (pos, &tile) in self.order.iter().enumerate() {
            ax.add_patch(
                Patch::rectangle(x, (n - pos - 1) as f64, width, 1.0)
                    .facecolor(Some(self.tiles.tile_colour(tile)))
                    .edgecolor(None),
            );
        }
    }

    /// Tile names down the y axis of a per-tile image, with each group's
    /// label at its first tile.
    fn tile_ticks_y(&self, ax: &mut Axes, n: usize) {
        let ticks: Vec<f64> = (0..n).map(|pos| (n - pos) as f64 - 0.5).collect();
        let labels: Vec<String> = self
            .order
            .iter()
            .enumerate()
            .map(
                |(pos, &tile)| match self.spans.iter().find(|s| s.start == pos) {
                    Some(span) => format!("{} {}", span.label, self.tile_name(tile)),
                    None => self.tile_name(tile).to_string(),
                },
            )
            .collect();
        ax.set_yticks(&ticks).set_yticklabels(&labels);
        let size = (1100.0 / n.max(1) as f64).clamp(4.0, TICK_SIZE);
        ax.yaxis_mut().set_tick_label_size(size);
    }

    /// Group labels along the x axis of a per-tile axis.
    fn group_ticks_x(&self, ax: &mut Axes) {
        let mids: Vec<f64> = self
            .spans
            .iter()
            .map(|s| 0.5 * (s.start + s.end) as f64 - 0.5)
            .collect();
        let labels: Vec<&str> = self.spans.iter().map(|s| s.label.as_str()).collect();
        ax.set_xticks(&mids).set_xticklabels(&labels);
        let size = (1600.0 / (6 * labels.len()).max(1) as f64).clamp(5.0, TICK_SIZE);
        ax.xaxis_mut().set_tick_label_size(size);
    }
}

fn new_figure(width_in: f64, height_in: f64, title: &str, subtitle: &str) -> Figure {
    let mut fig = Figure::new(width_in, height_in)
        .with_dpi(DPI)
        .with_rcparams(RcParams {
            axes_titlesize: LABEL_SIZE * 1.2,
            axes_labelsize: LABEL_SIZE,
            xtick_labelsize: TICK_SIZE,
            ytick_labelsize: TICK_SIZE,
            ..RcParams::default()
        });
    fig.suptitle(title).set_suptitle_size(TITLE_SIZE);
    let header = fig.add_axes(0.0, 0.94, 1.0, 0.03);
    header
        .set_axis_off()
        .set_facecolor(Rgba::TRANSPARENT)
        .set_xlim(0.0, 1.0)
        .set_ylim(0.0, 1.0);
    header.set_text_size(LABEL_SIZE).text(0.01, 0.2, subtitle);
    fig
}

/// Scatter points with per-point (or one shared) fill colours and a diameter
/// in output pixels.
fn scatter(ax: &mut Axes, x: &[f64], y: &[f64], colours: Vec<Rgba>, size_px: f64) {
    if x.is_empty() {
        return;
    }
    let points = ax.scatter(x, y);
    let styled = std::mem::replace(points, Collection::scatter(Vec::new()));
    *points = styled
        .with_facecolors(colours)
        .with_sizes(vec![size_px * LINE]);
}
