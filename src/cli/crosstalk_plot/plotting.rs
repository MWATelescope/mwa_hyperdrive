// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! The crosstalk outlier figures.
//!
//! Every per-tile axis is in the order given by [`Tiles::order`], so that
//! tiles sharing a receiver sit together and a crosstalk hub shows up as one
//! row or column. Tiles are coloured by cable flavour, and baselines by their
//! pair of flavours.

use std::collections::BTreeMap;

use log::info;
use rizzma::{
    artist::{Collection, MarkerStyle, Patch, Rgba},
    axis::axis::{Axis, AxisSide},
    core::color::{magma, Colormap},
    Axes, Figure, RcParams,
};

use super::{
    data::{self, BaselineSummary, ObsScores, Score, SearchParams, POLS},
    tiles::{GroupBy, GroupSpan, Tiles},
    CrosstalkPlotError,
};

/// rizzma sizes text in pixels at 100 DPI, so these are output pixels.
const DPI: f64 = 100.0;
const TITLE_SIZE: f64 = 28.0;
const LABEL_SIZE: f64 = 17.0;
const TICK_SIZE: f64 = 13.0;
/// One output pixel as a line width, which rizzma takes in points.
const LINE: f64 = 72.0 / DPI;

const fn grey(level: f64, a: f64) -> Rgba {
    Rgba {
        r: level,
        g: level,
        b: level,
        a,
    }
}

/// Points that weren't flagged.
const UNFLAGGED: Rgba = grey(0.55, 0.2);
/// Tiles or baselines with no scored data.
const NO_DATA: Rgba = grey(0.8, 1.0);
/// A tile with scored baselines, none of them flagged.
const ZERO: Rgba = Rgba {
    r: 0.97,
    g: 0.96,
    b: 0.92,
    a: 1.0,
};
const SEPARATOR: Rgba = grey(0.0, 0.7);
/// The outline of each receiver's own block on the baseline matrix.
const BLOCK: Rgba = Rgba {
    r: 0.0,
    g: 0.6,
    b: 0.65,
    a: 1.0,
};
const THRESHOLD: Rgba = Rgba {
    r: 0.85,
    g: 0.1,
    b: 0.1,
    a: 0.9,
};
/// Colours for baselines between two flavours, assigned in sorted order of
/// the pairs present, skipping any that a flavour already uses.
const MIXED_COLOURS: [&str; 12] = [
    "#e87ba4", "#4a3aa7", "#e34948", "#008300", "#7f3c8d", "#11a5a5", "#8c564b", "#a5aa99",
    "#cf1c90", "#3969ac", "#bcbd22", "#17becf",
];
/// The bins of the flag-count heatmap: a bin holds counts up to and including
/// its value. Counts are small integers with a long tail, so a linear colour
/// scale spends all its range on the one worst tile.
const COUNT_BINS: [(f64, &str); 7] = [
    (0.0, "0"),
    (1.0, "1"),
    (2.0, "2"),
    (4.0, "3-4"),
    (8.0, "5-8"),
    (16.0, "9-16"),
    (f64::INFINITY, "17+"),
];
/// Tiles on the spread plot are labelled if their median number of flagged
/// baselines is at least this.
const MIN_LABELLED_MEDIAN: f64 = 3.0;
/// The matrix's colour map: light at low z, so that only hot cells stand out.
const MATRIX_CMAP: &str = "magma_r";

/// The colour of a flag-count bin.
fn count_bin_colour(bin: usize) -> Rgba {
    if bin == 0 {
        return ZERO;
    }
    // From light orange (1) to dark purple (17+).
    let t = 0.9 - 0.85 * (bin - 1) as f64 / (COUNT_BINS.len() - 2) as f64;
    magma().sample(t)
}

fn count_bin(count: f64) -> Option<usize> {
    if !count.is_finite() {
        return None;
    }
    COUNT_BINS.iter().position(|(upper, _)| count <= *upper)
}

/// A pair of cable flavours, sorted so that the order of the tiles doesn't
/// matter.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct FlavourPair(String, String);

impl FlavourPair {
    fn new(a: &str, b: &str) -> FlavourPair {
        if a <= b {
            FlavourPair(a.to_string(), b.to_string())
        } else {
            FlavourPair(b.to_string(), a.to_string())
        }
    }

    fn is_mixed(&self) -> bool {
        self.0 != self.1
    }

    fn label(&self) -> String {
        format!("{} x {}", self.0, self.1)
    }
}

/// How a class of baselines is drawn.
#[derive(Debug, Clone, Copy)]
struct PairStyle {
    colour: Rgba,
    /// Same-flavour pairs are circles, mixed pairs triangles.
    marker: char,
}

/// A night of crosstalk scores, arranged for plotting.
pub(super) struct Night<'a> {
    night: &'a [ObsScores],
    tiles: &'a Tiles,
    group_by: GroupBy,
    /// Tiles with at least one scored baseline, in plotting order.
    order: Vec<usize>,
    /// Plotting position of each tile index, if it's plotted.
    position: Vec<Option<usize>>,
    spans: Vec<GroupSpan>,
    /// Flagged baselines per tile per observation, `[pol][tile][obs]`.
    counts: Vec<Vec<Vec<f64>>>,
    /// `[pol]`
    baselines: Vec<Vec<BaselineSummary>>,
    /// The style of every flavour pair that has a flagged baseline.
    pair_styles: BTreeMap<FlavourPair, PairStyle>,
    params: SearchParams,
    num_outliers: usize,
}

impl<'a> Night<'a> {
    pub(super) fn new(
        night: &'a [ObsScores],
        tiles: &'a Tiles,
        group_by: GroupBy,
        params: SearchParams,
        num_outliers: usize,
    ) -> Night<'a> {
        let counts: Vec<Vec<Vec<f64>>> = (0..POLS.len())
            .map(|pol| data::tile_flag_counts(night, tiles.len(), pol))
            .collect();
        let scored = |t: usize| counts.iter().any(|pol| pol[t].iter().any(|c| !c.is_nan()));
        let (order, spans) = tiles.order(group_by, scored);
        let mut position = vec![None; tiles.len()];
        for (pos, &tile) in order.iter().enumerate() {
            position[tile] = Some(pos);
        }

        let mut pairs: Vec<FlavourPair> = night
            .iter()
            .flat_map(|o| o.scores.iter())
            .filter(|s| s.flag)
            .map(|s| FlavourPair::new(&tiles.tiles[s.tile1].flavour, &tiles.tiles[s.tile2].flavour))
            .collect();
        pairs.sort();
        pairs.dedup();
        let flavour_colours: Vec<Rgba> = tiles.flavour_key().into_iter().map(|(c, _)| c).collect();
        let mut mixed_colours: Vec<Rgba> = MIXED_COLOURS
            .iter()
            .map(|hex| Rgba::from_hex(hex).expect("the mixed colours are valid hex"))
            .filter(|c| !flavour_colours.contains(c))
            .collect();
        if mixed_colours.is_empty() {
            mixed_colours.push(grey(0.2, 1.0));
        }
        let mut num_mixed = 0;
        let pair_styles = pairs
            .into_iter()
            .map(|pair| {
                let style = if pair.is_mixed() {
                    let colour = mixed_colours[num_mixed % mixed_colours.len()];
                    num_mixed += 1;
                    PairStyle {
                        colour,
                        marker: '^',
                    }
                } else {
                    PairStyle {
                        colour: tiles.flavour_colour(&pair.0),
                        marker: 'o',
                    }
                };
                (pair, style)
            })
            .collect();

        Night {
            night,
            tiles,
            group_by,
            order,
            position,
            spans,
            counts,
            baselines: (0..POLS.len())
                .map(|pol| data::baseline_summaries(night, pol))
                .collect(),
            pair_styles,
            params,
            num_outliers,
        }
    }

    pub(super) fn num_scored_tiles(&self) -> usize {
        self.order.len()
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

    fn baseline_name(&self, tile1: usize, tile2: usize) -> String {
        format!("{}-{}", self.tile_name(tile1), self.tile_name(tile2))
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
                    "{pol_name} baseline {} ({:.1} m): flagged in {}/{} obs, median z {:.1}",
                    self.baseline_name(b.tile1, b.tile2),
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
            "{}; {}; tile colour = cable flavour",
            self.obs_range(),
            self.group_description()
        )
    }

    /// The key of the flavours of the plotted tiles.
    fn flavour_key(&self, swatch: Swatch) -> Vec<KeyEntry> {
        self.tiles
            .flavour_key()
            .into_iter()
            .filter(|(_, f)| {
                self.order
                    .iter()
                    .any(|&t| &self.tiles.tiles[t].flavour == f)
            })
            .map(|(c, f)| (c, f, swatch))
            .collect()
    }

    /// Fig 1: flagged baselines touching each tile, through the night. A tile
    /// that is a crosstalk source is a dark row across every observation.
    fn tile_obs(&self) -> Figure {
        let num_tiles = self.order.len();
        let num_obs = self.night.len();
        let mut fig = new_figure(
            22.0,
            20.0,
            "Flagged baselines per tile per observation",
            &self.subtitle(),
        );
        // Each panel has, left to right: receiver labels, the flavour strip
        // and the heatmap, all in data units of observations.
        let strip = 0.03 * num_obs as f64 + 0.4;
        let rx_column = 0.08 * num_obs as f64 + 1.5;
        for (pol, pol_name) in POLS.iter().enumerate() {
            let left = 0.07 + 0.42 * pol as f64;
            let ax = data_axes(&mut fig, (left, 0.05, 0.34, 0.87));
            ax.set_facecolor(Rgba::WHITE);
            for (pos, &tile) in self.order.iter().enumerate() {
                let y = (num_tiles - pos - 1) as f64;
                // Runs of equal bins are drawn as one rectangle.
                let bins: Vec<Option<usize>> = self.counts[pol][tile]
                    .iter()
                    .map(|&c| count_bin(c))
                    .collect();
                let mut start = 0;
                while start < num_obs {
                    let mut end = start + 1;
                    while end < num_obs && bins[end] == bins[start] {
                        end += 1;
                    }
                    let colour = bins[start].map(count_bin_colour).unwrap_or(NO_DATA);
                    ax.add_patch(
                        Patch::rectangle(start as f64, y, (end - start) as f64, 1.0)
                            .facecolor(Some(colour))
                            .edgecolor(None),
                    );
                    start = end;
                }
                ax.add_patch(
                    Patch::rectangle(-strip, y, strip, 1.0)
                        .facecolor(Some(self.tiles.tile_colour(tile)))
                        .edgecolor(None),
                );
            }
            ax.set_xlim(-strip - rx_column, num_obs as f64)
                .set_ylim(0.0, num_tiles as f64)
                .set_title(*pol_name);
            ax.set_text_size(TICK_SIZE);
            for span in &self.spans {
                let top = (num_tiles - span.start) as f64;
                if span.start > 0 {
                    ax.add_line(
                        rizzma::artist::Line2D::new(
                            vec![-strip - rx_column, num_obs as f64],
                            vec![top, top],
                        )
                        .with_color(SEPARATOR)
                        .with_linewidth(1.5 * LINE),
                    );
                }
                let mid = (num_tiles as f64) - 0.5 * (span.start + span.end) as f64;
                ax.text(-strip - rx_column + 0.3, mid - 0.4, &span.label);
            }
            self.tile_ticks_y(ax, num_tiles);
            let step = num_obs.div_ceil(6).max(1);
            let ticks: Vec<usize> = (0..num_obs).step_by(step).collect();
            ax.set_xticks(&ticks.iter().map(|&i| i as f64 + 0.5).collect::<Vec<_>>())
                .set_xticklabels(
                    &ticks
                        .iter()
                        .map(|&i| self.night[i].label.as_str())
                        .collect::<Vec<_>>(),
                );
            ax.set_xlabel("observation");
        }
        let mut key: Vec<KeyEntry> = vec![(
            Rgba::BLACK,
            "Flagged baselines".to_string(),
            Swatch::Heading,
        )];
        key.extend(
            COUNT_BINS
                .iter()
                .enumerate()
                .map(|(i, (_, label))| (count_bin_colour(i), (*label).to_string(), Swatch::Square)),
        );
        key.push((NO_DATA, "no scored baselines".to_string(), Swatch::Square));
        // A gap, then the flavours under their own heading, so that a flavour
        // colour can't be read as a count.
        key.push((Rgba::BLACK, String::new(), Swatch::Heading));
        key.push((
            Rgba::BLACK,
            "Cable flavour (strip)".to_string(),
            Swatch::Heading,
        ));
        key.extend(self.flavour_key(Swatch::Square));
        legend_panel(&mut fig, (0.88, 0.5, 0.11, 0.42), key);
        fig
    }

    /// Fig 2: each tile's flag count in every observation, with the night's
    /// median, so outliers stand out from the rest of their receiver.
    fn tile_spread(&self) -> Figure {
        let num_tiles = self.order.len();
        let mut fig = new_figure(
            24.0,
            13.0,
            "Flagged baselines per tile, all observations",
            &format!(
                "{}; points jittered; black bars = median over the night",
                self.subtitle()
            ),
        );
        for (pol, pol_name) in POLS.iter().enumerate() {
            let bottom = 0.52 - 0.47 * pol as f64;
            let ax = data_axes(&mut fig, (0.05, bottom, 0.84, 0.38));
            let mut x = vec![];
            let mut y = vec![];
            let mut colours = vec![];
            for (pos, &tile) in self.order.iter().enumerate() {
                for (i_obs, &c) in self.counts[pol][tile].iter().enumerate() {
                    if !c.is_finite() {
                        continue;
                    }
                    let (jx, jy) = jitter(pos, i_obs);
                    x.push(pos as f64 + 0.7 * jx);
                    y.push(c + 0.25 * jy);
                    colours.push(Rgba {
                        a: 0.45,
                        ..self.tiles.tile_colour(tile)
                    });
                }
            }
            let y_max = y.iter().copied().fold(1.0, f64::max) + 0.5;
            scatter(ax, &x, &y, colours, 5.0, 'o');
            for (pos, &tile) in self.order.iter().enumerate() {
                let m = self.median_count(pol, tile);
                if m.is_finite() {
                    ax.add_line(
                        rizzma::artist::Line2D::new(
                            vec![pos as f64 - 0.45, pos as f64 + 0.45],
                            vec![m, m],
                        )
                        .with_color(Rgba::BLACK)
                        .with_linewidth(2.5 * LINE),
                    );
                }
            }
            let y_top = y_max * 1.12;
            ax.set_xlim(-1.0, num_tiles as f64)
                .set_ylim(-0.6, y_top)
                .set_title(*pol_name)
                .set_ylabel("flagged baselines");
            for span in self.spans.iter().skip(1) {
                ax.axvline_with(span.start as f64 - 0.5, SEPARATOR, LINE);
            }
            // Labelling every tile at a common median would pick among
            // ties arbitrarily, so only clear outliers are named.
            let labels = self
                .top_tiles(pol)
                .into_iter()
                .filter(|(_, median)| *median >= MIN_LABELLED_MEDIAN)
                .filter_map(|(tile, median)| {
                    let pos = self.position[tile]? as f64;
                    let top = self.counts[pol][tile]
                        .iter()
                        .copied()
                        .filter(|c| c.is_finite())
                        .fold(median, f64::max);
                    Some((
                        pos,
                        top + 0.3,
                        format!("{} ({median})", self.tile_name(tile)),
                    ))
                })
                .collect();
            place_labels(ax, labels, (-1.0, num_tiles as f64), (-0.6, y_top));
            self.group_ticks_x(ax);
            if pol == 0 {
                let mut key = self.flavour_key(Swatch::Marker('o'));
                key.push((Rgba::BLACK, "median".to_string(), Swatch::Line));
                legend_panel(&mut fig, (0.9, 0.55, 0.09, 0.3), key);
            }
        }
        fig
    }

    /// Fig 3: the night's median z of every baseline, XX below the diagonal
    /// and YY above. Hot cells inside an outlined block are pairs on the same
    /// receiver; hot cells outside cross receivers.
    fn baseline_matrix(&self) -> Figure {
        let n = self.order.len();
        let mut fig = new_figure(
            18.0,
            17.0,
            "Median z per baseline over the night",
            &format!(
                "{}; {}; outlined blocks = same {}; grey = not scored",
                self.obs_range(),
                self.group_description(),
                match self.group_by {
                    GroupBy::Flavour => "cable flavour",
                    _ => "receiver",
                }
            ),
        );
        let mut image = vec![f64::NAN; n * n];
        for (pol, baselines) in self.baselines.iter().enumerate() {
            for b in baselines {
                let (Some(p1), Some(p2)) = (self.position[b.tile1], self.position[b.tile2]) else {
                    continue;
                };
                // XX below the diagonal (row > column), YY above.
                let (row, col) = if (pol == 0) == (p1 > p2) {
                    (p1, p2)
                } else {
                    (p2, p1)
                };
                image[row * n + col] = b.median_z;
            }
        }
        let vmax = (2.0 * self.params.z_min).max(1.0);
        let ax = data_axes(&mut fig, (0.1, 0.04, 0.78, 0.86));
        ax.set_facecolor(NO_DATA);
        ax.imshow(&image, n, n)
            .cmap(MATRIX_CMAP)
            .vmin(0.0)
            .vmax(vmax);
        for span in &self.spans {
            let len = (span.end - span.start) as f64;
            ax.add_patch(
                Patch::rectangle(span.start as f64, (n - span.end) as f64, len, len)
                    .facecolor(None)
                    .edgecolor(Some(BLOCK))
                    .linewidth(2.0 * LINE),
            );
        }
        let pad = 0.025 * n as f64 + 0.5;
        for (pos, &tile) in self.order.iter().enumerate() {
            let colour = self.tiles.tile_colour(tile);
            ax.add_patch(
                Patch::rectangle(-pad, (n - pos - 1) as f64, pad, 1.0)
                    .facecolor(Some(colour))
                    .edgecolor(None),
            );
            ax.add_patch(
                Patch::rectangle(pos as f64, n as f64, 1.0, pad)
                    .facecolor(Some(colour))
                    .edgecolor(None),
            );
        }
        // The corner the two strips leave empty says which half is which.
        ax.add_patch(
            Patch::rectangle(-pad, n as f64, pad, pad)
                .facecolor(Some(Rgba::WHITE))
                .edgecolor(None),
        );
        ax.set_xlim(-pad, n as f64).set_ylim(0.0, n as f64 + pad);
        ax.set_text_size(LABEL_SIZE * 1.4);
        ax.text(0.03 * n as f64, 0.03 * n as f64, "XX");
        ax.text(0.9 * n as f64, 0.92 * n as f64, "YY");
        self.group_ticks_x(ax);
        self.tile_ticks_y(ax, n);
        fig.colorbar_at((0.9, 0.04, 0.015, 0.86), MATRIX_CMAP, 0.0, vmax);
        let label = fig.add_axes(0.88, 0.905, 0.1, 0.02);
        label
            .set_axis_off()
            .set_facecolor(Rgba::TRANSPARENT)
            .set_xlim(0.0, 1.0)
            .set_ylim(0.0, 1.0);
        label.set_text_size(LABEL_SIZE).text(0.05, 0.2, "median z");
        fig
    }

    /// Fig 4: z against baseline length for every baseline in every
    /// observation. Persistent crosstalk is a vertical stripe of flagged
    /// points at one length.
    fn z_vs_length(&self) -> Figure {
        let num_no_peak = self
            .night
            .iter()
            .flat_map(|o| o.scores.iter())
            .filter(|s| s.t == 0.0 || !s.tau_peak_ns.is_finite())
            .count();
        let mut fig = new_figure(
            20.0,
            14.0,
            "Crosstalk z against baseline length",
            &format!(
                "{}; red line z = {}; not shown: {num_no_peak} rows with no delay peak (t = 0), whose z is not meaningful; labelled: baselines flagged in at least half the obs",
                self.obs_range(),
                self.params.z_min
            ),
        );
        let has_peak = |s: &&Score| s.z.is_finite() && s.t > 0.0 && s.tau_peak_ns.is_finite();
        let all: Vec<&Score> = self
            .night
            .iter()
            .flat_map(|o| o.scores.iter())
            .filter(has_peak)
            .collect();
        let x_max = all.iter().map(|s| s.length_m).fold(1.0, f64::max) * 1.03;
        let (z_lo, z_hi) = all
            .iter()
            .fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), s| {
                (lo.min(s.z), hi.max(s.z))
            });
        let (z_lo, z_hi) = if z_lo <= z_hi {
            (z_lo - 1.0, z_hi + 1.5)
        } else {
            (-1.0, 1.0)
        };
        for (pol, pol_name) in POLS.iter().enumerate() {
            let bottom = 0.52 - 0.47 * pol as f64;
            let ax = data_axes(&mut fig, (0.06, bottom, 0.74, 0.38));
            let scores: Vec<&Score> = all.iter().copied().filter(|s| s.pol == pol).collect();
            let key = self.scatter_pairs(ax, &scores, |s| s.length_m, true);
            ax.axhline_with(self.params.z_min, THRESHOLD, 1.5 * LINE);
            ax.set_xlim(0.0, x_max)
                .set_ylim(z_lo, z_hi)
                .set_title(*pol_name)
                .set_xlabel("baseline length (m)")
                .set_ylabel("z");
            let labels = self
                .top_baselines(pol)
                .into_iter()
                .map(|b| (b.length_m, b.median_z, self.baseline_name(b.tile1, b.tile2)))
                .collect();
            place_labels(ax, labels, (0.0, x_max), (z_lo, z_hi));
            legend_panel(&mut fig, (0.81, bottom, 0.18, 0.38), key);
        }
        fig
    }

    /// Fig 5: where the peak of each flagged baseline is, against its z.
    /// Crosstalk through a fixed signal path keeps the same delay all night;
    /// a peak just past the foreground cut may be leakage from the cut.
    fn peak_delay(&self) -> Figure {
        let buffer = self.params.buffer_ns;
        let mut fig = new_figure(
            22.0,
            14.0,
            "Peak delay of flagged baselines",
            &format!(
                "{}; right: delay past the foreground cut |b|/c + {buffer} ns",
                self.obs_range()
            ),
        );
        let flagged: Vec<&Score> = self
            .night
            .iter()
            .flat_map(|o| o.scores.iter())
            .filter(|s| s.flag && s.tau_peak_ns.is_finite() && s.z.is_finite())
            .collect();
        let past_cut = |s: &Score| s.tau_peak_ns.abs() - (data::horizon_ns(s.length_m) + buffer);
        let range = |f: &dyn Fn(&Score) -> f64| {
            let (lo, hi) = flagged
                .iter()
                .fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), s| {
                    (lo.min(f(s)), hi.max(f(s)))
                });
            if lo <= hi {
                let pad = 0.03 * (hi - lo).max(1.0);
                (lo - pad, hi + pad)
            } else {
                (-1.0, 1.0)
            }
        };
        let tau_range = range(&|s| s.tau_peak_ns);
        let cut_range = range(&|s| past_cut(s));
        let z_range = range(&|s| s.z);
        for (pol, pol_name) in POLS.iter().enumerate() {
            let bottom = 0.52 - 0.47 * pol as f64;
            let scores: Vec<&Score> = flagged.iter().copied().filter(|s| s.pol == pol).collect();

            let ax = data_axes(&mut fig, (0.05, bottom, 0.4, 0.38));
            let key = self.scatter_pairs(ax, &scores, |s| s.tau_peak_ns, false);
            ax.axvline_with(0.0, SEPARATOR, LINE);
            ax.set_xlim(tau_range.0, tau_range.1)
                .set_ylim(z_range.0, z_range.1)
                .set_title(format!("{pol_name}: peak delay"))
                .set_xlabel("peak delay (ns), ant1 to ant2")
                .set_ylabel("z");

            let ax = data_axes(&mut fig, (0.49, bottom, 0.33, 0.38));
            self.scatter_pairs(ax, &scores, past_cut, false);
            ax.axvline_with(0.0, THRESHOLD, 1.5 * LINE);
            ax.set_xlim(cut_range.0.min(-10.0), cut_range.1)
                .set_ylim(z_range.0, z_range.1)
                .set_title(format!("{pol_name}: distance past the foreground cut"))
                .set_xlabel("|peak delay| - (|b|/c + buffer) (ns)");

            legend_panel(&mut fig, (0.83, bottom, 0.16, 0.38), key);
        }
        fig
    }

    /// Scatter scores against `x`, with flagged points styled by flavour pair
    /// and, if `with_unflagged`, the rest in grey underneath. Returns the key,
    /// with the number of flagged points of each pair.
    fn scatter_pairs(
        &self,
        ax: &mut Axes,
        scores: &[&Score],
        x: impl Fn(&Score) -> f64,
        with_unflagged: bool,
    ) -> Vec<KeyEntry> {
        let mut key = vec![];
        if with_unflagged {
            let (ux, uy): (Vec<f64>, Vec<f64>) = scores
                .iter()
                .filter(|s| !s.flag)
                .map(|s| (x(s), s.z))
                .unzip();
            scatter(ax, &ux, &uy, vec![UNFLAGGED], 2.5, 'o');
            key.push((
                grey(0.55, 1.0),
                format!("not flagged ({})", ux.len()),
                Swatch::Marker('o'),
            ));
        }
        // Draw the commonest pairs first so that rare ones stay visible.
        let mut groups: BTreeMap<&FlavourPair, (Vec<f64>, Vec<f64>)> = BTreeMap::new();
        for s in scores.iter().filter(|s| s.flag) {
            let pair = FlavourPair::new(
                &self.tiles.tiles[s.tile1].flavour,
                &self.tiles.tiles[s.tile2].flavour,
            );
            let (pair, _) = self
                .pair_styles
                .get_key_value(&pair)
                .expect("every flagged pair has a style");
            let g = groups.entry(pair).or_default();
            g.0.push(x(s));
            g.1.push(s.z);
        }
        let mut groups: Vec<_> = groups.into_iter().collect();
        groups.sort_by_key(|(_, (gx, _))| std::cmp::Reverse(gx.len()));
        for (pair, (gx, gy)) in &groups {
            let style = self.pair_styles[*pair];
            scatter(
                ax,
                gx,
                gy,
                vec![Rgba {
                    a: 0.85,
                    ..style.colour
                }],
                7.0,
                style.marker,
            );
        }
        groups.sort_by(|a, b| a.0.cmp(b.0));
        for (pair, (gx, _)) in groups {
            let style = self.pair_styles[pair];
            key.push((
                style.colour,
                format!("{} ({})", pair.label(), gx.len()),
                Swatch::Marker(style.marker),
            ));
        }
        key
    }

    /// Tile names down the y axis of a per-tile image whose first tile (in
    /// plotting order) is at the top.
    fn tile_ticks_y(&self, ax: &mut Axes, n: usize) {
        let ticks: Vec<f64> = (0..n).map(|pos| (n - pos) as f64 - 0.5).collect();
        let labels: Vec<&str> = self.order.iter().map(|&t| self.tile_name(t)).collect();
        ax.set_yticks(&ticks).set_yticklabels(&labels);
        let size = (1250.0 / n.max(1) as f64).clamp(5.0, TICK_SIZE);
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
        let size = (1800.0 / (5 * labels.len()).max(1) as f64).clamp(6.0, TICK_SIZE);
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
    let header = fig.add_axes(0.0, 0.94, 1.0, 0.025);
    header
        .set_axis_off()
        .set_facecolor(Rgba::TRANSPARENT)
        .set_xlim(0.0, 1.0)
        .set_ylim(0.0, 1.0);
    header.set_text_size(LABEL_SIZE).text(0.01, 0.2, subtitle);
    fig
}

/// Axes for data, with axis labels at the figure's label size.
fn data_axes(fig: &mut Figure, (left, bottom, width, height): (f64, f64, f64, f64)) -> &mut Axes {
    let ax = fig.add_axes(left, bottom, width, height);
    let x = std::mem::replace(ax.xaxis_mut(), Axis::new(AxisSide::Bottom));
    *ax.xaxis_mut() = x.with_axis_label_size(LABEL_SIZE);
    let y = std::mem::replace(ax.yaxis_mut(), Axis::new(AxisSide::Left));
    *ax.yaxis_mut() = y.with_axis_label_size(LABEL_SIZE);
    ax
}

/// What a legend entry's swatch looks like.
#[derive(Debug, Clone, Copy, PartialEq)]
enum Swatch {
    /// A filled square, for areas.
    Square,
    /// A scatter marker, by its character.
    Marker(char),
    /// A short thick line.
    Line,
    /// No swatch: the entry is a heading for the entries below it.
    Heading,
}

/// A legend entry.
type KeyEntry = (Rgba, String, Swatch);

/// A legend drawn in its own invisible axes, so that it never covers data,
/// with swatches that look like what they stand for and text at the
/// figure's label size. Entries run down from the top of `rect`.
fn legend_panel(fig: &mut Figure, rect: (f64, f64, f64, f64), key: Vec<KeyEntry>) {
    let (_, height_px) = fig.size_px();
    let ax = fig.add_axes(rect.0, rect.1, rect.2, rect.3);
    // Data units are pixels down from the top-left of the panel.
    let panel_height = rect.3 * height_px / (DPI / 100.0);
    ax.set_axis_off()
        .set_facecolor(Rgba::TRANSPARENT)
        .set_xlim(0.0, 100.0)
        .set_ylim(panel_height, 0.0);
    let row = 1.6 * LABEL_SIZE;
    ax.set_text_size(LABEL_SIZE);
    for (i, (colour, label, swatch)) in key.into_iter().enumerate() {
        let y = (i as f64 + 0.5) * row;
        match swatch {
            Swatch::Square => {
                ax.add_patch(
                    Patch::rectangle(0.0, y - 0.35 * row, 8.0, 0.7 * row)
                        .facecolor(Some(colour))
                        .edgecolor(Some(grey(0.5, 1.0)))
                        .linewidth(0.5 * LINE),
                );
            }
            Swatch::Marker(m) => scatter(ax, &[4.0], &[y], vec![colour], 10.0, m),
            Swatch::Heading => {
                ax.text(0.0, y + 0.3 * LABEL_SIZE, label);
                continue;
            }
            Swatch::Line => {
                ax.add_line(
                    rizzma::artist::Line2D::new(vec![0.0, 8.0], vec![y, y])
                        .with_color(colour)
                        .with_linewidth(3.0 * LINE),
                );
            }
        }
        ax.text(11.0, y + 0.3 * LABEL_SIZE, label);
    }
}

/// Scatter points with per-point (or one shared) fill colours, a marker
/// character and a diameter in output pixels.
fn scatter(ax: &mut Axes, x: &[f64], y: &[f64], colours: Vec<Rgba>, size_px: f64, marker: char) {
    if x.is_empty() {
        return;
    }
    let marker = MarkerStyle::from_char(marker)
        .unwrap_or_else(|| MarkerStyle::from_char('o').expect("'o' is a marker"));
    let points = ax.scatter(x, y);
    let styled = std::mem::replace(points, Collection::scatter(Vec::new()));
    *points = styled
        .with_marker(marker.path().clone())
        .with_facecolors(colours)
        .with_sizes(vec![size_px * LINE]);
}

/// A repeatable jitter in [-0.5, 0.5) for a point, so that integer counts
/// don't sit on top of each other.
fn jitter(a: usize, b: usize) -> (f64, f64) {
    let mut h = (a as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)
        ^ (b as u64).wrapping_add(0x632B_E59B_D9B4_E019);
    h ^= h >> 33;
    h = h.wrapping_mul(0xFF51_AFD7_ED55_8CCD);
    h ^= h >> 33;
    let x = (h & 0xFFFF) as f64 / 65536.0 - 0.5;
    let y = ((h >> 16) & 0xFFFF) as f64 / 65536.0 - 0.5;
    (x, y)
}

/// Draw text labels at points, moving any that would overlap an earlier one
/// upward and joining a moved label to its point with a thin line. Ranges are
/// the axes limits, used to judge overlap in screen proportions.
fn place_labels(
    ax: &mut Axes,
    mut labels: Vec<(f64, f64, String)>,
    (x0, x1): (f64, f64),
    (y0, y1): (f64, f64),
) {
    let (xr, yr) = (
        (x1 - x0).abs().max(f64::EPSILON),
        (y1 - y0).abs().max(f64::EPSILON),
    );
    labels.sort_by(|a, b| b.1.total_cmp(&a.1));
    let mut placed: Vec<(f64, f64, f64)> = vec![];
    ax.set_text_size(TICK_SIZE);
    for (x, y, text) in labels {
        // Text is about 0.6 of its size wide per character, in pixels; the
        // axes are about 1500 pixels wide, so this is a fraction of the range.
        let width = text.len() as f64 * 0.6 * TICK_SIZE / 1500.0 * xr;
        let height = 0.035 * yr;
        let lx = x + 0.004 * xr;
        let mut ly = y;
        while placed
            .iter()
            .any(|&(px, py, pw)| lx < px + pw && px < lx + width && (ly - py).abs() < height)
        {
            ly += height;
        }
        if (ly - y).abs() > f64::EPSILON {
            ax.add_line(
                rizzma::artist::Line2D::new(vec![x, lx], vec![y, ly])
                    .with_color(grey(0.3, 0.8))
                    .with_linewidth(LINE),
            );
        }
        ax.text(lx, ly, text);
        placed.push((lx, ly, width));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn counts_fall_in_the_right_bins() {
        assert_eq!(count_bin(0.0), Some(0));
        assert_eq!(count_bin(1.0), Some(1));
        assert_eq!(count_bin(3.0), Some(3));
        assert_eq!(count_bin(4.0), Some(3));
        assert_eq!(count_bin(5.0), Some(4));
        assert_eq!(count_bin(16.0), Some(5));
        assert_eq!(count_bin(40.0), Some(6));
        assert_eq!(count_bin(f64::NAN), None);
    }

    #[test]
    fn flavour_pairs_ignore_order() {
        let a = FlavourPair::new("RG6_90", "RG6_150");
        assert_eq!(a, FlavourPair::new("RG6_150", "RG6_90"));
        assert!(a.is_mixed());
        assert_eq!(a.label(), "RG6_150 x RG6_90");
        assert!(!FlavourPair::new("RG6_90", "RG6_90").is_mixed());
    }

    #[test]
    fn jitter_is_bounded_and_repeatable() {
        for a in 0..50 {
            for b in 0..50 {
                let (x, y) = jitter(a, b);
                assert!((-0.5..0.5).contains(&x) && (-0.5..0.5).contains(&y));
                assert_eq!(jitter(a, b), (x, y));
            }
        }
    }
}
