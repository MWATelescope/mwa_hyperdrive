// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Code to verify sky-model source list files.

use std::path::{Path, PathBuf};

use log::info;
use structopt::StructOpt;

use crate::{read::read_source_list_file, SrclistError};
use mwa_hyperdrive_common::log;

/// Verify that sky-model source lists can be read by hyperdrive.
///
/// See for more info:
/// https://github.com/MWATelescope/mwa_hyperdrive/wiki/Source-lists
#[derive(StructOpt, Debug)]
pub struct VerifyArgs {
    /// Path to the source list(s) to be verified.
    #[structopt(name = "SOURCE_LISTS", parse(from_os_str))]
    pub source_lists: Vec<PathBuf>,

    /// The verbosity of the program. The default is to print high-level
    /// information.
    #[structopt(short, long, parse(from_occurrences))]
    pub verbosity: u8,
}

impl VerifyArgs {
    /// Run [verify] with these arguments.
    pub fn run(&self) -> Result<(), SrclistError> {
        verify(&self.source_lists)
    }
}

/// Read and print stats out for each input source list. If a source list
/// couldn't be read, print the error, and continue trying to read the other
/// source lists.
///
/// If the source list type is provided, then assume that all source lists have
/// that type.
pub fn verify<T: AsRef<Path>>(source_lists: &[T]) -> Result<(), SrclistError> {
    if source_lists.is_empty() {
        info!("No source lists were supplied!");
        std::process::exit(1);
    }

    for source_list in source_lists {
        info!("{}:", source_list.as_ref().display());

        let (sl, sl_type) = match read_source_list_file(&source_list, None) {
            Ok(sl) => sl,
            Err(e) => {
                info!("{}", e);
                info!("");
                continue;
            }
        };
        info!("    {}-style source list", sl_type);
        let (points, gaussians, shapelets) = sl.get_counts();
        let num_components = points + gaussians + shapelets;
        info!(
            "    {} sources, {} components ({} points, {} gaussians, {} shapelets)",
            sl.len(),
            num_components,
            points,
            gaussians,
            shapelets
        );
        info!("");
    }

    Ok(())
}
