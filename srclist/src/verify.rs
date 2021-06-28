// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Code to verify sky-model source list files.

use std::path::Path;

use log::info;

use crate::{read::read_source_list_file, SrclistError};

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
        info!(
            "    {} sources, {} components",
            sl.len(),
            sl.iter().flat_map(|(_, s)| &s.components).count()
        );
        info!("");
    }

    Ok(())
}
