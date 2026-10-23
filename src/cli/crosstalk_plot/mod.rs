// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Plot per-baseline crosstalk scores over many observations, so that tiles
//! and baselines that are persistent crosstalk outliers stand out.

mod data;
mod error;
mod plotting;
pub(crate) mod tiles;

pub(crate) use error::CrosstalkPlotError;

use std::path::PathBuf;

use clap::Parser;
use log::{info, warn};

use crate::HyperdriveError;

#[derive(Parser, Debug, Default)]
pub(crate) struct CrosstalkPlotArgs {
    /// Per-baseline crosstalk score CSVs, one per observation, as written by
    /// crosstalk-qa. Each needs pol, ant1, ant2, length_m, tau_peak_ns, t, z
    /// and flag columns, with ant1 and ant2 as tile names.
    #[clap(value_name = "CROSSTALK_CSVS")]
    files: Vec<PathBuf>,

    /// A metafits file for the array the scores are from. Tile names in the
    /// CSVs are matched to its tiles to find each tile's receiver and cable
    /// flavour.
    #[clap(short, long)]
    metafits: PathBuf,

    /// How tiles are ordered and banded on the per-tile plots: "receiver+flavour"
    /// (receivers, with tiles sorted by cable flavour inside each), "receiver",
    /// or "flavour". Tiles are always coloured by cable flavour.
    #[clap(long, default_value = "receiver+flavour")]
    group_by: tiles::GroupBy,

    /// The z threshold to mark on the plots. By default this is read from
    /// the JSON written next to the first CSV, falling back to 5.
    #[clap(long)]
    z_min: Option<f64>,

    /// The number of outlier tiles and baselines to label and report.
    #[clap(long, default_value = "10")]
    num_outliers: usize,

    /// The directory to write the plots into. If this doesn't exist, then it
    /// is created.
    #[clap(short, long)]
    output_directory: Option<PathBuf>,

    /// The start of the plot filenames.
    #[clap(long, default_value = "crosstalk")]
    prefix: String,
}

impl CrosstalkPlotArgs {
    pub(crate) fn run(self) -> Result<(), HyperdriveError> {
        let files = plot(self)?;
        info!("Wrote {files:?}");
        Ok(())
    }
}

fn plot(args: CrosstalkPlotArgs) -> Result<Vec<String>, CrosstalkPlotError> {
    let CrosstalkPlotArgs {
        mut files,
        metafits,
        group_by,
        z_min,
        num_outliers,
        output_directory,
        prefix,
    } = args;
    if files.is_empty() {
        return Err(CrosstalkPlotError::NoInputs);
    }
    files.sort();

    let context = mwalib::MetafitsContext::new(&metafits, None)?;
    let tiles = tiles::Tiles::from_metafits(&context);
    let night = data::read_all(&files, &tiles)?;
    if night.iter().all(|o| o.scores.is_empty()) {
        return Err(CrosstalkPlotError::NoScores);
    }
    let mut params = data::read_search_params(&files[0]).unwrap_or_default();
    if let Some(z_min) = z_min {
        params.z_min = z_min;
    }

    let base = match output_directory {
        Some(dir) => {
            if !dir.exists() {
                std::fs::create_dir_all(&dir)?;
            }
            dir.join(&prefix)
        }
        None => PathBuf::from(&prefix),
    };
    let base = base.display().to_string();

    let summary = plotting::Night::new(&night, &tiles, group_by, params, num_outliers);
    summary.report();
    if summary.num_scored_tiles() == 0 {
        warn!("No tile has a scored baseline; the plots will be empty");
    }
    summary.plot_all(&base)
}
