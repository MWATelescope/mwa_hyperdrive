// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Code to read in hyperdrive source lists.

use crate::srclist::{error::ReadSourceListError, SourceList};

/// Convert a yaml file to a [SourceList].
pub(crate) fn source_list_from_yaml<T: std::io::BufRead>(
    buf: &mut T,
) -> Result<SourceList, ReadSourceListError> {
    let sl = serde_yaml::from_reader(buf)?;
    Ok(sl)
}

/// Convert a json file to a [SourceList].
pub(crate) fn source_list_from_json<T: std::io::BufRead>(
    buf: &mut T,
) -> Result<SourceList, ReadSourceListError> {
    let sl = serde_json::from_reader(buf)?;
    Ok(sl)
}
