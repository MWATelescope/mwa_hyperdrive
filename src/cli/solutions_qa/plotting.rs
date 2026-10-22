// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! The calibration QA figures. Per-tile axes use the same order, bands and
//! colours as crosstalk-plot: receivers, then cable flavour, then antenna
//! number, with tiles coloured by flavour.

use log::info;
use rizzma::{
    artist::{Line2D, Patch, Rgba},
    axis::axis::{Axis, AxisSide},
    Axes, Figure,
};

use super::{
    metrics::{median, reflection_delay_ns, ObsQa, POLS, Z_CUT},
    SolutionsQaError,
};
use crate::cli::crosstalk_plot::{
    plotting::{
        data_axes, grey, jitter, legend_panel, new_figure, place_labels, scatter, KeyEntry, Swatch,
        LABEL_SIZE, LINE, NO_DATA, SEPARATOR, THRESHOLD, TICK_SIZE,
    },
    tiles::{GroupBy, GroupSpan, Tiles},
};

/// The colour scale of the normalised amplitude heatmap runs 1 -/+ this.
const AMP_RANGE: f64 = 0.2;
const AMP_CMAP: &str = "RdBu_r";
const DELAY_CMAP: &str = "magma";
const PRECISION_CMAP: &str = "viridis";
/// z-scores are drawn on a symlog scale, linear out to this, so that one
/// very bad tile doesn't flatten the rest.
const Z_LINEAR: f64 = 10.0;
/// The period of the coarse channels' structure in the delay spectra (ns):
/// one over the 1.28 MHz coarse-channel width.
const COARSE_CHANNEL_DELAY_NS: f64 = 1e9 / 1.28e6;

/// The figures for a set of solutions.
pub(super) struct QaSet<'a> {
    obs: &'a [ObsQa],
    tiles: &'a Tiles,
    group_by: GroupBy,
    max_delay_ns: f64,
    /// Plotted tiles, in plotting order, and their groups.
    order: Vec<usize>,
    spans: Vec<GroupSpan>,
    /// Each tile's position in `order`.
    position: Vec<Option<usize>>,
}

impl<'a> QaSet<'a> {
    pub(super) fn new(
        obs: &'a [ObsQa],
        tiles: &'a Tiles,
        group_by: GroupBy,
        max_delay_ns: f64,
    ) -> QaSet<'a> {
        // Flagged tiles are kept, as grey rows, so that it's clear they were
        // flagged rather than missing.
        let has_data = |tile: usize| obs.iter().any(|o| o.in_solutions[tile]);
        let (order, spans) = tiles.order(group_by, has_data);
        let mut position = vec![None; tiles.len()];
        for (pos, &tile) in order.iter().enumerate() {
            position[tile] = Some(pos);
        }
        QaSet {
            obs,
            tiles,
            group_by,
            max_delay_ns,
            order,
            spans,
            position,
        }
    }

    pub(super) fn num_tiles(&self) -> usize {
        self.order.len()
    }

    fn tile_name(&self, tile: usize) -> &str {
        &self.tiles.tiles[tile].name
    }

    /// How many obs each tile is an amplitude outlier in, and its median z.
    fn amp_outliers(&self, pol: usize) -> Vec<(usize, usize, f64)> {
        let mut out: Vec<(usize, usize, f64)> = self
            .order
            .iter()
            .map(|&tile| {
                let zs: Vec<f64> = self.obs.iter().map(|o| o.amp_z[pol][tile]).collect();
                let n = zs.iter().filter(|z| z.abs() > Z_CUT).count();
                (tile, n, median(zs))
            })
            .filter(|(_, n, _)| 2 * n >= self.obs.len() && *n > 0)
            .collect();
        out.sort_by(|a, b| b.1.cmp(&a.1).then(b.2.total_cmp(&a.2)));
        out
    }

    /// Log the persistent outliers.
    pub(super) fn report(&self) {
        info!(
            "{} solutions files, {} tiles with solutions",
            self.obs.len(),
            self.num_tiles()
        );
        for (pol, pol_name) in POLS.iter().enumerate() {
            for (tile, n, z) in self.amp_outliers(pol) {
                let t = &self.tiles.tiles[tile];
                info!(
                    "{pol_name} tile {:<8} Rx{:02} {:<10} amplitude outlier in {n}/{} obs, median z {z:.1}",
                    t.name,
                    t.rx,
                    t.flavour,
                    self.obs.len()
                );
            }
        }
    }

    /// Write every figure, returning the filenames.
    pub(super) fn plot_all(&self, base: &str) -> Result<Vec<String>, SolutionsQaError> {
        let mut figures: Vec<(String, Figure)> = vec![];
        for o in self.obs {
            figures.push((format!("amp_{}", o.label), self.amp(o)));
            if !o.delays_ns.is_empty() {
                figures.push((format!("delay_{}", o.label), self.delay(o)));
            }
        }
        figures.push(("amp_rms".to_string(), self.amp_rms()));
        if self.obs.iter().any(|o| o.precision.is_some()) {
            figures.push(("convergence".to_string(), self.convergence()));
        }
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

    fn obs_range(&self) -> String {
        let first = self.obs.first().map(|o| o.label.as_str()).unwrap_or("");
        let last = self.obs.last().map(|o| o.label.as_str()).unwrap_or("");
        format!("{} obs ({first} to {last})", self.obs.len())
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

    /// A tile x `x` heatmap of `rows` (`[tile][column]`, in metafits order),
    /// with a flavour strip and receiver labels to the left of `x0`.
    fn tile_heatmap(
        &self,
        ax: &mut Axes,
        rows: &[Vec<f64>],
        (x0, x1): (f64, f64),
        (cmap, vmin, vmax): (&str, f64, f64),
    ) {
        let n = self.order.len();
        let ncols = rows.first().map(|r| r.len()).unwrap_or(0);
        let mut image = Vec::with_capacity(n * ncols);
        for &tile in &self.order {
            match rows.get(tile) {
                Some(r) if r.len() == ncols => image.extend_from_slice(r),
                _ => image.extend(std::iter::repeat_n(f64::NAN, ncols)),
            }
        }
        ax.set_facecolor(NO_DATA);
        if ncols > 0 {
            ax.imshow(&image, n, ncols)
                .cmap(cmap)
                .vmin(vmin)
                .vmax(vmax)
                .set_extent([x0, x1, 0.0, n as f64]);
        }
        let width = x1 - x0;
        let strip = 0.02 * width;
        let rx_column = 0.07 * width;
        for (pos, &tile) in self.order.iter().enumerate() {
            ax.add_patch(
                Patch::rectangle(x0 - strip, (n - pos - 1) as f64, strip, 1.0)
                    .facecolor(Some(self.tiles.tile_colour(tile)))
                    .edgecolor(None),
            );
        }
        ax.set_text_size(TICK_SIZE);
        for span in &self.spans {
            let top = (n - span.start) as f64;
            if span.start > 0 {
                ax.add_line(
                    Line2D::new(vec![x0 - strip - rx_column, x1], vec![top, top])
                        .with_color(SEPARATOR)
                        .with_linewidth(1.5 * LINE),
                );
            }
            let mid = n as f64 - 0.5 * (span.start + span.end) as f64;
            ax.text(
                x0 - strip - rx_column + 0.005 * width,
                mid - 0.4,
                &span.label,
            );
        }
        ax.set_xlim(x0 - strip - rx_column, x1)
            .set_ylim(0.0, n as f64);
        let ticks: Vec<f64> = (0..n).map(|pos| (n - pos) as f64 - 0.5).collect();
        let labels: Vec<&str> = self.order.iter().map(|&t| self.tile_name(t)).collect();
        ax.set_yticks(&ticks).set_yticklabels(&labels);
        let size = (1250.0 / n.max(1) as f64).clamp(5.0, TICK_SIZE);
        ax.yaxis_mut().set_tick_label_size(size);
    }

    /// The frequency range (MHz) of an obs's chanblocks, as image edges, or
    /// chanblock indices if the frequencies are unknown.
    fn freq_extent(o: &ObsQa) -> ((f64, f64), &'static str) {
        let n = o.num_chanblocks();
        let (first, last) = (o.freqs_hz.first(), o.freqs_hz.last());
        match (first, last) {
            (Some(&f0), Some(&f1)) if f0.is_finite() && f1.is_finite() && n > 1 => {
                let half = 0.5 * (f1 - f0) / (n - 1) as f64;
                (((f0 - half) / 1e6, (f1 + half) / 1e6), "frequency (MHz)")
            }
            _ => ((0.0, n as f64), "chanblock"),
        }
    }

    /// Fig a: each tile's gain amplitude relative to tiles of the same
    /// flavour, per chanblock. A bad tile is a row of colour; a bad channel
    /// is a column.
    fn amp(&self, o: &ObsQa) -> Figure {
        let mut fig = new_figure(
            24.0,
            20.0,
            &format!("Gain amplitude relative to same-flavour tiles: {}", o.label),
            &format!(
                "|g| / median |g| of same-flavour tiles per chanblock / tile median; {}; grey = no solution or flagged tile",
                self.group_description()
            ),
        );
        let (extent, xlabel) = Self::freq_extent(o);
        for (pol, pol_name) in POLS.iter().enumerate() {
            let ax = data_axes(&mut fig, (0.07 + 0.43 * pol as f64, 0.05, 0.36, 0.86));
            self.tile_heatmap(
                ax,
                &o.norm_amp[pol],
                extent,
                (AMP_CMAP, 1.0 - AMP_RANGE, 1.0 + AMP_RANGE),
            );
            ax.set_title(*pol_name).set_xlabel(xlabel);
        }
        colorbar(
            &mut fig,
            (0.9, 0.5, 0.015, 0.41),
            (AMP_CMAP, 1.0 - AMP_RANGE, 1.0 + AMP_RANGE),
            "relative amplitude",
        );
        let mut key = vec![(
            Rgba::BLACK,
            "Cable flavour (strip)".to_string(),
            Swatch::Heading,
        )];
        key.extend(self.flavour_key(Swatch::Square));
        legend_panel(&mut fig, (0.88, 0.1, 0.11, 0.35), key);
        fig
    }

    /// Fig b: the delay spectrum of each tile's detrended gains. A cable
    /// reflection is a peak at a delay set by the cable length; the median of
    /// each flavour is drawn above, with its expected reflection delay.
    fn delay(&self, o: &ObsQa) -> Figure {
        let mut fig = new_figure(
            24.0,
            20.0,
            &format!("Delay spectrum of the gains: {}", o.label),
            &format!(
                "gains divided by a degree-4 fit to their amplitude and phase, Blackman-Harris window; triangles = expected cable reflection (v_f = 0.85); dotted = coarse-channel harmonics; {}",
                self.group_description()
            ),
        );
        let num_delays = o
            .delays_ns
            .iter()
            .take_while(|&&d| d <= self.max_delay_ns)
            .count()
            .max(2);
        let step = o.delays_ns.get(1).copied().unwrap_or(1.0);
        let extent = (-0.5 * step, (num_delays as f64 - 0.5) * step);
        let cropped = |pol: usize| -> Vec<Vec<f64>> {
            o.delay_db[pol]
                .iter()
                .map(|r| r.iter().take(num_delays).copied().collect())
                .collect()
        };
        // One colour scale for both pols, from the plotted tiles' values
        // past the first few delays (which the detrending leaves at the
        // floor).
        let mut values: Vec<f64> = vec![];
        for rows in &o.delay_db {
            for &t in &self.order {
                if let Some(r) = rows.get(t) {
                    values.extend(
                        r.iter()
                            .take(num_delays)
                            .skip(3)
                            .copied()
                            .filter(|v| v.is_finite()),
                    );
                }
            }
        }
        values.sort_unstable_by(f64::total_cmp);
        let quantile = |q: f64| {
            values
                .get(((values.len() as f64 - 1.0) * q).round() as usize)
                .copied()
        };
        let (vmin, vmax) = match (quantile(0.05), quantile(0.995)) {
            (Some(lo), Some(hi)) if hi > lo => (lo, hi),
            _ => (-60.0, 0.0),
        };
        let flavours: Vec<(Rgba, String, Swatch)> = self.flavour_key(Swatch::Line);
        for (pol, pol_name) in POLS.iter().enumerate() {
            let left = 0.07 + 0.43 * pol as f64;
            let rows = cropped(pol);

            // The median spectrum of each flavour.
            let top = data_axes(&mut fig, (left, 0.68, 0.36, 0.23));
            let x: Vec<f64> = o.delays_ns.iter().take(num_delays).copied().collect();
            let (mut lo, mut hi) = (f64::INFINITY, f64::NEG_INFINITY);
            let mut markers = vec![];
            for (colour, flavour, _) in &flavours {
                let members: Vec<usize> = self
                    .order
                    .iter()
                    .copied()
                    .filter(|&t| &self.tiles.tiles[t].flavour == flavour)
                    .collect();
                let y: Vec<f64> = (0..num_delays)
                    .map(|k| median(members.iter().map(|&t| rows[t][k])))
                    .collect();
                for v in y.iter().skip(3).filter(|v| v.is_finite()) {
                    lo = lo.min(*v);
                    hi = hi.max(*v);
                }
                top.add_line(
                    Line2D::new(x.clone(), y)
                        .with_color(*colour)
                        .with_linewidth(2.0 * LINE),
                );
                if let Some(d) = reflection_delay_ns(flavour) {
                    if d <= self.max_delay_ns {
                        markers.push((d, *colour));
                    }
                }
            }
            let (lo, hi) = if lo < hi {
                (lo - 2.0, hi + 4.0)
            } else {
                (vmin, vmax)
            };
            let mut k = 1.0;
            while k * COARSE_CHANNEL_DELAY_NS <= self.max_delay_ns {
                top.add_line(
                    Line2D::new(vec![k * COARSE_CHANNEL_DELAY_NS; 2], vec![lo, hi])
                        .with_color(grey(0.4, 0.8))
                        .with_linewidth(LINE)
                        .with_dashes(Some((0.0, vec![2.0, 3.0]))),
                );
                k += 1.0;
            }
            for (d, colour) in markers {
                scatter(top, &[d], &[hi - 1.5], vec![colour], 14.0, 'v');
            }
            // The heatmap's axes below include the strip and receiver labels
            // to the left of zero delay, so line this panel up with them.
            let width = extent.1 - extent.0;
            top.set_xlim(extent.0 - 0.09 * width, extent.1)
                .set_ylim(lo, hi)
                .set_title(format!("{pol_name}: median per flavour"))
                .set_ylabel("power (dB)");

            let ax = data_axes(&mut fig, (left, 0.05, 0.36, 0.58));
            self.tile_heatmap(ax, &rows, extent, (DELAY_CMAP, vmin, vmax));
            ax.set_xlabel("delay (ns)");
        }
        colorbar(
            &mut fig,
            (0.9, 0.05, 0.015, 0.4),
            (DELAY_CMAP, vmin, vmax),
            "power (dB)",
        );
        let mut key = vec![(Rgba::BLACK, "Cable flavour".to_string(), Swatch::Heading)];
        key.extend(flavours);
        legend_panel(&mut fig, (0.88, 0.55, 0.11, 0.36), key);
        fig
    }

    /// Fig c: each tile's amplitude RMS as a z-score within its flavour, in
    /// every obs. A tile that is bad in most obs is labelled.
    fn amp_rms(&self) -> Figure {
        let num_tiles = self.order.len();
        let mut fig = new_figure(
            24.0,
            13.0,
            "Gain amplitude outliers, all solutions",
            &format!(
                "{}; RMS of the normalised amplitude as a modified z-score within each flavour; one point per obs, bars = median; red = |z| > {Z_CUT}; labelled: outliers in at least half the obs; {}",
                self.obs_range(),
                self.group_description()
            ),
        );
        for (pol, pol_name) in POLS.iter().enumerate() {
            let bottom = 0.52 - 0.47 * pol as f64;
            let ax = data_axes(&mut fig, (0.05, bottom, 0.84, 0.38));
            let (mut x, mut y, mut colours) = (vec![], vec![], vec![]);
            for (pos, &tile) in self.order.iter().enumerate() {
                for (i_obs, o) in self.obs.iter().enumerate() {
                    let z = o.amp_z[pol][tile];
                    if !z.is_finite() {
                        continue;
                    }
                    let (jx, _) = jitter(pos, i_obs);
                    x.push(pos as f64 + 0.6 * jx);
                    y.push(z);
                    colours.push(Rgba {
                        a: 0.55,
                        ..self.tiles.tile_colour(tile)
                    });
                }
            }
            let y_lo = y.iter().copied().fold(-Z_CUT, f64::min) - 1.0;
            let y_hi = y.iter().copied().fold(Z_CUT, f64::max) * 1.5 + 1.0;
            scatter(ax, &x, &y, colours, 6.0, 'o');
            for (pos, &tile) in self.order.iter().enumerate() {
                let m = median(self.obs.iter().map(|o| o.amp_z[pol][tile]));
                if m.is_finite() {
                    ax.add_line(
                        Line2D::new(vec![pos as f64 - 0.45, pos as f64 + 0.45], vec![m, m])
                            .with_color(Rgba::BLACK)
                            .with_linewidth(2.5 * LINE),
                    );
                }
            }
            ax.axhline_with(Z_CUT, THRESHOLD, 1.5 * LINE);
            ax.axhline_with(-Z_CUT, THRESHOLD, 1.5 * LINE);
            ax.set_yscale_symlog(10.0, Z_LINEAR);
            ax.set_xlim(-1.0, num_tiles as f64)
                .set_ylim(y_lo, y_hi)
                .set_title(*pol_name)
                .set_ylabel(format!("modified z (log beyond ±{Z_LINEAR})"));
            for span in self.spans.iter().skip(1) {
                ax.axvline_with(span.start as f64 - 0.5, SEPARATOR, LINE);
            }
            let labels = self
                .amp_outliers(pol)
                .into_iter()
                .filter_map(|(tile, n, _)| {
                    let pos = self.position[tile]? as f64;
                    let top = self
                        .obs
                        .iter()
                        .map(|o| o.amp_z[pol][tile])
                        .filter(|z| z.is_finite())
                        .fold(f64::NEG_INFINITY, f64::max);
                    Some((
                        pos,
                        top + 0.3,
                        format!("{} ({n}/{})", self.tile_name(tile), self.obs.len()),
                    ))
                })
                .collect();
            place_labels(ax, labels, (-1.0, num_tiles as f64), (y_lo, y_hi));
            let mids: Vec<f64> = self
                .spans
                .iter()
                .map(|s| 0.5 * (s.start + s.end) as f64 - 0.5)
                .collect();
            let names: Vec<&str> = self.spans.iter().map(|s| s.label.as_str()).collect();
            ax.set_xticks(&mids).set_xticklabels(&names);
            let size = (1800.0 / (5 * names.len()).max(1) as f64).clamp(6.0, TICK_SIZE);
            ax.xaxis_mut().set_tick_label_size(size);
            if pol == 0 {
                let mut key = self.flavour_key(Swatch::Marker('o'));
                key.push((Rgba::BLACK, "median".to_string(), Swatch::Line));
                legend_panel(&mut fig, (0.9, 0.55, 0.09, 0.3), key);
            }
        }
        fig
    }

    /// Fig d: the solver's final precision in each chanblock of each obs,
    /// and its median over the obs against the stop and minimum thresholds.
    fn convergence(&self) -> Figure {
        let num_obs = self.obs.len();
        let num_chans = self
            .obs
            .iter()
            .map(|o| o.num_chanblocks())
            .max()
            .unwrap_or(0);
        let stop = self.obs.iter().find_map(|o| o.stop_threshold);
        let min = self.obs.iter().find_map(|o| o.min_threshold);
        let mut fig = new_figure(
            22.0,
            16.0,
            "Calibration convergence",
            &format!(
                "{}; log10 of the final precision per chanblock; grey = no solution{}",
                self.obs_range(),
                match (stop, min) {
                    (Some(s), Some(m)) => format!("; stop threshold {s:e}, minimum {m:e}"),
                    _ => String::new(),
                }
            ),
        );
        let log = |p: f64| if p > 0.0 { p.log10() } else { f64::NAN };
        let mut image = Vec::with_capacity(num_obs * num_chans);
        for o in self.obs {
            let row = o.precision.as_deref().unwrap_or(&[]);
            image.extend((0..num_chans).map(|c| row.get(c).copied().map(log).unwrap_or(f64::NAN)));
        }
        let vmax = min.map(log).unwrap_or(-4.0) + 1.0;
        let vmin = stop.map(log).unwrap_or(-8.0) - 1.0;
        // Frequencies from the first obs with the most chanblocks, if all
        // obs agree on them.
        let freqs_agree = self.obs.windows(2).all(|w| w[0].freqs_hz == w[1].freqs_hz);
        let reference = self.obs.iter().find(|o| o.num_chanblocks() == num_chans);
        let ((x0, x1), xlabel) = match reference {
            Some(o) if freqs_agree => Self::freq_extent(o),
            _ => ((0.0, num_chans as f64), "chanblock"),
        };
        let xs: Vec<f64> = (0..num_chans)
            .map(|c| x0 + (c as f64 + 0.5) * (x1 - x0) / num_chans.max(1) as f64)
            .collect();

        let ax = data_axes(&mut fig, (0.1, 0.38, 0.78, 0.53));
        ax.set_facecolor(NO_DATA);
        if num_chans > 0 {
            ax.imshow(&image, num_obs, num_chans)
                .cmap(PRECISION_CMAP)
                .vmin(vmin)
                .vmax(vmax)
                .set_extent([x0, x1, 0.0, num_obs as f64]);
        }
        ax.set_xlim(x0, x1).set_ylim(0.0, num_obs as f64);
        let ticks: Vec<f64> = (0..num_obs).map(|i| (num_obs - i) as f64 - 0.5).collect();
        let labels: Vec<&str> = self.obs.iter().map(|o| o.label.as_str()).collect();
        ax.set_yticks(&ticks).set_yticklabels(&labels);
        let size = (900.0 / num_obs.max(1) as f64).clamp(5.0, TICK_SIZE);
        ax.yaxis_mut().set_tick_label_size(size);
        ax.set_ylabel("observation");

        let ax = data_axes(&mut fig, (0.1, 0.06, 0.78, 0.25));
        let med: Vec<f64> = (0..num_chans)
            .map(|c| median((0..num_obs).map(|o| image[o * num_chans + c])))
            .collect();
        let worst: Vec<f64> = (0..num_chans)
            .map(|c| {
                (0..num_obs)
                    .map(|o| image[o * num_chans + c])
                    .filter(|v| v.is_finite())
                    .fold(f64::NAN, f64::max)
            })
            .collect();
        ax.add_line(
            Line2D::new(xs.clone(), worst)
                .with_color(grey(0.5, 1.0))
                .with_linewidth(LINE),
        );
        ax.add_line(
            Line2D::new(xs, med)
                .with_color(Rgba::BLACK)
                .with_linewidth(2.0 * LINE),
        );
        if let Some(s) = stop {
            ax.axhline_with(log(s), Rgba::BLUE, 1.5 * LINE);
        }
        if let Some(m) = min {
            ax.axhline_with(log(m), THRESHOLD, 1.5 * LINE);
        }
        ax.set_xlim(x0, x1)
            .set_ylim(vmin - 1.0, vmax + 1.0)
            .set_xlabel(xlabel)
            .set_ylabel("log10 precision");

        colorbar(
            &mut fig,
            (0.9, 0.38, 0.015, 0.53),
            (PRECISION_CMAP, vmin, vmax),
            "log10 precision",
        );
        let mut key = vec![
            (Rgba::BLACK, "median over obs".to_string(), Swatch::Line),
            (grey(0.5, 1.0), "worst obs".to_string(), Swatch::Line),
        ];
        if stop.is_some() {
            key.push((Rgba::BLUE, "stop threshold".to_string(), Swatch::Line));
        }
        if min.is_some() {
            key.push((THRESHOLD, "minimum threshold".to_string(), Swatch::Line));
        }
        key.push((NO_DATA, "no solution".to_string(), Swatch::Square));
        legend_panel(&mut fig, (0.89, 0.06, 0.1, 0.25), key);
        fig
    }
}

/// A vertical colour bar in its own axes, so that its tick labels are at the
/// figure's tick size (rizzma's own colour bars use a fixed small font), with
/// a title above it.
fn colorbar(
    fig: &mut Figure,
    rect: (f64, f64, f64, f64),
    (cmap, vmin, vmax): (&str, f64, f64),
    title: &str,
) {
    const STEPS: usize = 256;
    let gradient: Vec<f64> = (0..STEPS)
        .map(|i| vmax - (vmax - vmin) * (i as f64 + 0.5) / STEPS as f64)
        .collect();
    let ax = data_axes(fig, rect);
    *ax.yaxis_mut() = Axis::new(AxisSide::Right).with_axis_label_size(LABEL_SIZE);
    ax.imshow(&gradient, STEPS, 1)
        .cmap(cmap)
        .vmin(vmin)
        .vmax(vmax)
        .set_extent([0.0, 1.0, vmin, vmax]);
    ax.set_xlim(0.0, 1.0).set_ylim(vmin, vmax).set_xticks(&[]);
    ax.yaxis_mut().set_tick_label_size(TICK_SIZE);
    ax.set_title_size(LABEL_SIZE).set_title(title);
}
