// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

/*!
Code to read sky-model source list files.
 */

use std::fs::File;
use std::path::Path;

use error::ReadSourceListError;
use log::trace;
use thiserror::Error;

use crate::*;

pub fn parse_file_type(file: &Path) -> Option<SourceListFileType> {
    let ext = file.extension().and_then(|e| e.to_str());
    match ext {
        Some("json") => Some(SourceListFileType::Json),
        Some("yaml") => Some(SourceListFileType::Yaml),
        Some("txt") => Some(SourceListFileType::Txt),
        _ => None,
    }
}

pub fn parse_source_list_type(s: &str) -> Result<SourceListType, SourceListError> {
    match s {
        "hyperdrive" => Ok(SourceListType::Hyperdrive),
        "rts" => Ok(SourceListType::Rts),
        "woden" => Ok(SourceListType::Woden),
        "ao" => Ok(SourceListType::AO),
        _ => Err(SourceListError::InvalidSourceListType),
    }
}

/// Given the path to a sky-model source list file (and optionally its type,
/// e.g. "RTS style"), return a [SourceList] object. The [SourceListType] is
/// also returned in case that's interesting to the caller.
pub fn read_source_list_file<T: AsRef<Path>>(
    file: T,
    sl_type: Option<SourceListType>,
) -> Result<(SourceList, SourceListType), SourceListError> {
    trace!("Attempting to read source list");
    let mut f = std::io::BufReader::new(File::open(&file)?);

    match sl_type {
        Some(SourceListType::Hyperdrive) => {
            // Can be either yaml or json.
            let yaml_err = match hyperdrive::source_list_from_yaml(&mut f) {
                Ok(sl) => return Ok((sl, SourceListType::Hyperdrive)),
                Err(e) => e.to_string(),
            };
            let json_err = match hyperdrive::source_list_from_json(&mut f) {
                Ok(sl) => return Ok((sl, SourceListType::Hyperdrive)),
                Err(e) => e.to_string(),
            };
            Err(ReadSourceListError::FailedToDeserialise { yaml_err, json_err }.into())
        }

        Some(SourceListType::Rts) => match rts::parse_source_list(&mut f) {
            Ok(sl) => Ok((sl, SourceListType::Rts)),
            Err(e) => Err(e.into()),
        },

        Some(SourceListType::AO) => match ao::parse_source_list(&mut f) {
            Ok(sl) => Ok((sl, SourceListType::AO)),
            Err(e) => Err(e.into()),
        },

        Some(SourceListType::Woden) => match woden::parse_source_list(&mut f) {
            Ok(sl) => Ok((sl, SourceListType::Woden)),
            Err(e) => Err(e.into()),
        },

        Some(SourceListType::Unspecified) | None => {
            // Try all kinds.
            match hyperdrive::source_list_from_yaml(&mut f) {
                Ok(sl) => return Ok((sl, SourceListType::Hyperdrive)),
                Err(_) => {
                    trace!("Failed to read source list as hyperdrive-style yaml");
                    // Even a failed attempt to read the file alters the buffer. Open it
                    // again.
                    f = std::io::BufReader::new(File::open(&file)?);
                }
            }
            match hyperdrive::source_list_from_json(&mut f) {
                Ok(sl) => return Ok((sl, SourceListType::Hyperdrive)),
                Err(_) => {
                    trace!("Failed to read source list as hyperdrive-style json");
                    f = std::io::BufReader::new(File::open(&file)?);
                }
            }
            match rts::parse_source_list(&mut f) {
                Ok(sl) => return Ok((sl, SourceListType::Rts)),
                Err(_) => {
                    trace!("Failed to read source list as rts-style");
                    f = std::io::BufReader::new(File::open(&file)?);
                }
            }
            match ao::parse_source_list(&mut f) {
                Ok(sl) => return Ok((sl, SourceListType::AO)),
                Err(_) => {
                    trace!("Failed to read source list as ao-style");
                    f = std::io::BufReader::new(File::open(&file)?);
                }
            }
            match woden::parse_source_list(&mut f) {
                Ok(sl) => return Ok((sl, SourceListType::Woden)),
                Err(_) => {
                    trace!("Failed to read source list as woden-style");
                    // The following should be uncommented if new source list
                    // types are enumerated below!
                    // f = std::io::BufReader::new(File::open(&file)?);
                }
            }
            return Err(ReadSourceListError::FailedToReadAsAnyType.into());
        }
    }
}

/// Errors associated with reading in any kind of source list.
#[derive(Error, Debug)]
pub enum SourceListError {
    #[error("Unrecognised source list type. Valid types are: {}", *SOURCE_LIST_TYPES_COMMA_SEPARATED)]
    InvalidSourceListType,

    #[error("Unrecognised source list file type. Valid types are: {}", *SOURCE_LIST_FILE_TYPES_COMMA_SEPARATED)]
    InvalidSourceListFileType,

    #[error(
        "Attempted to read a {sl_type}-style source list from a {sl_file_type} file, but that is not handled"
    )]
    InvalidFormat {
        sl_type: String,
        sl_file_type: String,
    },

    #[error("{0}")]
    ReadError(#[from] ReadSourceListError),

    #[error("IO error: {0}")]
    IO(#[from] std::io::Error),
}
