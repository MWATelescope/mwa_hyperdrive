// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Code to write out hyperdrive source lists.

use crate::srclist::{error::WriteSourceListError, SourceList};

/// Write a `SourceList` to a yaml file.
pub(crate) fn source_list_to_yaml<T: std::io::Write>(
    buf: &mut T,
    sl: &SourceList,
    _num_sources: Option<usize>,
) -> Result<(), WriteSourceListError> {
    serde_yaml::to_writer(buf, &sl)?;
    Ok(())
}

/// Write a `SourceList` to a json file.
pub(crate) fn source_list_to_json<T: std::io::Write>(
    buf: &mut T,
    sl: &SourceList,
    _num_sources: Option<usize>,
) -> Result<(), WriteSourceListError> {
    serde_json::to_writer_pretty(buf, &sl)?;
    Ok(())
}
