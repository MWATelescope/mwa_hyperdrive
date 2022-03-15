// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Code to verify sky-model source list files.

use std::fs::File;
use std::path::{Path, PathBuf};
use std::str::FromStr;

use clap::Parser;
use log::info;

use crate::{
    ao, hyperdrive, read::read_source_list_file, rts, woden, SourceListType, SrclistError,
    SOURCE_LIST_INPUT_TYPE_HELP,
};
use mwa_hyperdrive_common::{clap, log};

/// Verify that sky-model source lists can be read by hyperdrive.
///
/// See for more info:
/// https://github.com/MWATelescope/mwa_hyperdrive/wiki/Source-lists
#[derive(Parser, Debug)]
pub struct VerifyArgs {
    /// Path to the source list(s) to be verified.
    #[clap(name = "SOURCE_LISTS", parse(from_os_str))]
    pub source_lists: Vec<PathBuf>,

    #[clap(short = 'i', long, parse(from_str), help = SOURCE_LIST_INPUT_TYPE_HELP.as_str())]
    pub input_type: Option<String>,

    /// The verbosity of the program. The default is to print high-level
    /// information.
    #[clap(short, long, parse(from_occurrences))]
    pub verbosity: u8,
}

impl VerifyArgs {
    /// Run [verify] with these arguments. If the `input_type` is given, then
    /// all provided source lists are assumed to be of that type, otherwise each
    /// source list's type is guessed.
    pub fn run(&self) -> Result<(), SrclistError> {
        verify(&self.source_lists, self.input_type.as_ref())
    }
}

/// Read and print stats out for each input source list. If a source list
/// couldn't be read, print the error, and continue trying to read the other
/// source lists.
///
/// If the source list type is provided, then assume that all source lists have
/// that type.
pub fn verify<P: AsRef<Path>>(
    source_lists: &[P],
    input_type: Option<&String>,
) -> Result<(), SrclistError> {
    if source_lists.is_empty() {
        info!("No source lists were supplied!");
        std::process::exit(1);
    }
    // Map the input_type to a proper type.
    let input_type = input_type.and_then(|i| SourceListType::from_str(i).ok());

    for source_list in source_lists {
        info!("{}:", source_list.as_ref().display());

        let (sl, sl_type) = if let Some(input_type) = input_type {
            let mut buf = std::io::BufReader::new(File::open(source_list)?);
            let result = match input_type {
                SourceListType::Hyperdrive => hyperdrive::source_list_from_yaml(&mut buf),
                SourceListType::AO => ao::parse_source_list(&mut buf),
                SourceListType::Rts => rts::parse_source_list(&mut buf),
                SourceListType::Woden => woden::parse_source_list(&mut buf),
            };
            match result {
                Ok(sl) => (sl, input_type),
                Err(e) => {
                    info!("{}", e);
                    info!("");
                    continue;
                }
            }
        } else {
            match read_source_list_file(&source_list, None) {
                Ok(sl) => sl,
                Err(e) => {
                    info!("{}", e);
                    info!("");
                    continue;
                }
            }
        };
        info!("    {}-style source list", sl_type);
        let crate::ComponentCounts {
            num_points,
            num_gaussians,
            num_shapelets,
            num_power_laws,
            num_curved_power_laws,
            num_lists,
        } = sl.get_counts();
        let num_components = num_points + num_gaussians + num_shapelets;
        info!(
            "    {} sources, {num_components} components ({num_points} points, {num_gaussians} gaussians, {num_shapelets} shapelets)",
            sl.len()
        );
        info!(
            "    Flux-density types: {num_power_laws} power laws, {num_curved_power_laws} curved power laws, {num_lists} lists"
        );
        info!("");
    }

    Ok(())
}
