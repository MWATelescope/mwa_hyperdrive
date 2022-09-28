// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Common code for reading sky-model source list files.

use std::fs::File;
use std::path::Path;

use log::trace;

use super::{error::ReadSourceListError, SourceList, SourceListType};
use crate::srclist::{ao, hyperdrive, rts, woden};

/// Given the path to a sky-model source list file (and optionally its type,
/// e.g. "RTS style"), return a [SourceList] object. The [SourceListType] is
/// also returned in case that's interesting to the caller.
pub(crate) fn read_source_list_file<P: AsRef<Path>>(
    path: P,
    sl_type: Option<SourceListType>,
) -> Result<(SourceList, SourceListType), ReadSourceListError> {
    fn inner(
        path: &Path,
        sl_type: Option<SourceListType>,
    ) -> Result<(SourceList, SourceListType), ReadSourceListError> {
        trace!("Attempting to read source list");
        let mut f = std::io::BufReader::new(File::open(path)?);

        // If the file extension corresponds to YAML or JSON, we know what to
        // target.
        let ext = path
            .extension()
            .and_then(|os_str| os_str.to_str())
            .map(|s| s.to_lowercase());
        match ext.as_deref() {
            Some("yaml" | "yml") => {
                return hyperdrive::source_list_from_yaml(&mut f)
                    .map(|r| (r, SourceListType::Hyperdrive))
            }
            Some("json") => {
                return hyperdrive::source_list_from_json(&mut f)
                    .map(|r| (r, SourceListType::Hyperdrive))
            }
            _ => (),
        }

        // We're guessing what the format is here.
        match sl_type {
            Some(SourceListType::Hyperdrive) => {
                // Can be either yaml or json.
                let yaml_err = match hyperdrive::source_list_from_yaml(&mut f) {
                    Ok(sl) => return Ok((sl, SourceListType::Hyperdrive)),
                    // If there was an error in parsing, then pass the parse
                    // error along and try JSON.
                    Err(e @ ReadSourceListError::Yaml(_)) => e.to_string(),
                    Err(e) => return Err(e),
                };
                let json_err = match hyperdrive::source_list_from_json(&mut f) {
                    Ok(sl) => return Ok((sl, SourceListType::Hyperdrive)),
                    Err(e @ ReadSourceListError::Json(_)) => e.to_string(),
                    Err(e) => return Err(e),
                };
                Err(ReadSourceListError::FailedToDeserialise { yaml_err, json_err })
            }

            Some(SourceListType::Rts) => match rts::parse_source_list(&mut f) {
                Ok(sl) => Ok((sl, SourceListType::Rts)),
                Err(e) => Err(e),
            },

            Some(SourceListType::AO) => match ao::parse_source_list(&mut f) {
                Ok(sl) => Ok((sl, SourceListType::AO)),
                Err(e) => Err(e),
            },

            Some(SourceListType::Woden) => match woden::parse_source_list(&mut f) {
                Ok(sl) => Ok((sl, SourceListType::Woden)),
                Err(e) => Err(e),
            },

            None => {
                // Try all kinds.
                match hyperdrive::source_list_from_yaml(&mut f) {
                    Ok(sl) => return Ok((sl, SourceListType::Hyperdrive)),
                    Err(_) => {
                        trace!("Failed to read source list as hyperdrive-style yaml");
                        // Even a failed attempt to read the file alters the buffer. Open it
                        // again.
                        f = std::io::BufReader::new(File::open(path)?);
                    }
                }
                match hyperdrive::source_list_from_json(&mut f) {
                    Ok(sl) => return Ok((sl, SourceListType::Hyperdrive)),
                    Err(_) => {
                        trace!("Failed to read source list as hyperdrive-style json");
                        f = std::io::BufReader::new(File::open(path)?);
                    }
                }
                match rts::parse_source_list(&mut f) {
                    Ok(sl) => return Ok((sl, SourceListType::Rts)),
                    Err(_) => {
                        trace!("Failed to read source list as rts-style");
                        f = std::io::BufReader::new(File::open(path)?);
                    }
                }
                match ao::parse_source_list(&mut f) {
                    Ok(sl) => return Ok((sl, SourceListType::AO)),
                    Err(_) => {
                        trace!("Failed to read source list as ao-style");
                        f = std::io::BufReader::new(File::open(path)?);
                    }
                }
                match woden::parse_source_list(&mut f) {
                    Ok(sl) => return Ok((sl, SourceListType::Woden)),
                    Err(_) => {
                        trace!("Failed to read source list as woden-style");
                    }
                }
                Err(ReadSourceListError::FailedToReadAsAnyType)
            }
        }
    }
    inner(path.as_ref(), sl_type)
}
