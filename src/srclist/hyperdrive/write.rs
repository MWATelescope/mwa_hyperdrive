// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Code to write out hyperdrive source lists.

use std::collections::HashMap;

use indexmap::IndexMap;

use crate::{
    cli::Warn,
    srclist::{error::WriteSourceListError, SourceList},
};

/// Write a [`SourceList`] to a yaml file.
pub(crate) fn source_list_to_yaml<T: std::io::Write>(
    mut buf: &mut T,
    sl: &SourceList,
    num_sources: Option<usize>,
) -> Result<(), WriteSourceListError> {
    if let Some(num_sources) = num_sources {
        let mut map = HashMap::with_capacity(1);
        for (name, src) in sl.iter().take(num_sources) {
            map.insert(name.as_str(), serde_yaml::to_value(src)?);
            serde_yaml::to_writer(&mut buf, &map)?;
            map.clear();
        }

        if num_sources > sl.len() {
            format!(
                "Couldn't write the requested number of sources ({num_sources}): wrote {}",
                sl.len()
            )
            .warn()
        };
    } else {
        serde_yaml::to_writer(buf, &sl)?;
    }
    Ok(())
}

/// Write a [`SourceList`] to a json file.
pub(crate) fn source_list_to_json<T: std::io::Write>(
    buf: &mut T,
    sl: &SourceList,
    num_sources: Option<usize>,
) -> Result<(), WriteSourceListError> {
    if let Some(num_sources) = num_sources {
        let mut map = IndexMap::with_capacity(sl.len().min(num_sources));
        for (name, src) in sl.iter().take(num_sources) {
            map.insert(name.as_str(), serde_json::to_value(src)?);
        }
        serde_json::to_writer_pretty(buf, &map)?;

        if num_sources > sl.len() {
            format!(
                "Couldn't write the requested number of sources ({num_sources}): wrote {}",
                sl.len()
            )
            .warn()
        };
    } else {
        serde_json::to_writer_pretty(buf, &sl)?;
    }
    Ok(())
}
