// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Plot QA metrics of calibration solutions over many observations: gain
//! amplitudes relative to tiles of the same cable flavour, delay spectra of
//! the gains, persistent amplitude outliers and solver convergence. These
//! replace mwa_qa's `calqa` and `caljson` plots.

mod error;
mod metrics;
mod plotting;

pub(crate) use error::SolutionsQaError;

use std::path::PathBuf;

use clap::Parser;
use log::info;

use super::crosstalk_plot::tiles::{GroupBy, Tiles};
use crate::HyperdriveError;

#[derive(Parser, Debug, Default)]
pub(crate) struct SolutionsQaArgs {
    /// Calibration solutions files, one per observation. Only the first
    /// timeblock of each is used.
    #[clap(value_name = "SOLUTIONS_FILES")]
    files: Vec<PathBuf>,

    /// A metafits file for the array the solutions are from. Solution tiles
    /// are matched to its tiles by name (or by index, if the solutions have no
    /// names) to find each tile's receiver and cable flavour.
    #[clap(short, long)]
    metafits: PathBuf,

    /// How tiles are ordered and banded on the per-tile plots: "receiver+flavour"
    /// (receivers, with tiles sorted by cable flavour inside each), "receiver",
    /// or "flavour". Tiles are always coloured by cable flavour.
    #[clap(long, default_value = "receiver+flavour")]
    group_by: GroupBy,

    /// The largest delay to show on the delay-spectrum plots [ns].
    #[clap(long, default_value = "4000")]
    max_delay_ns: f64,

    /// The directory to write the plots into. If this doesn't exist, then it
    /// is created.
    #[clap(short, long)]
    output_directory: Option<PathBuf>,

    /// The start of the plot filenames.
    #[clap(long, default_value = "solutions_qa")]
    prefix: String,
}

impl SolutionsQaArgs {
    pub(crate) fn run(self) -> Result<(), HyperdriveError> {
        let files = plot(self)?;
        info!("Wrote {files:?}");
        Ok(())
    }
}

fn plot(args: SolutionsQaArgs) -> Result<Vec<String>, SolutionsQaError> {
    let SolutionsQaArgs {
        mut files,
        metafits,
        group_by,
        max_delay_ns,
        output_directory,
        prefix,
    } = args;
    if files.is_empty() {
        return Err(SolutionsQaError::NoInputs);
    }
    files.sort();

    let context = mwalib::MetafitsContext::new(&metafits, None)?;
    let tiles = Tiles::from_metafits(&context);
    let mut obs = files
        .iter()
        .map(|f| metrics::read(f, &tiles))
        .collect::<Result<Vec<_>, _>>()?;
    obs.sort_by(|a, b| a.label.cmp(&b.label));

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

    let set = plotting::QaSet::new(&obs, &tiles, group_by, max_delay_ns);
    if set.num_tiles() == 0 {
        return Err(SolutionsQaError::NoTiles);
    }
    set.report();
    set.plot_all(&base)
}
