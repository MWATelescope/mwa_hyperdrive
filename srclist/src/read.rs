// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

/*!
Code to read sky-model source list files.
 */

use std::fs::File;
use std::path::Path;

use log::debug;
use thiserror::Error;

use crate::*;

pub fn parse_file_type<T: AsRef<Path>>(file: T) -> Result<SourceListFileType, SourceListError> {
    let file_pb = file.as_ref().to_path_buf();
    let ext = file_pb.extension().and_then(|e| e.to_str());
    match ext {
        Some("json") => Ok(SourceListFileType::Json),
        Some("yaml") => Ok(SourceListFileType::Yaml),
        Some("txt") => Ok(SourceListFileType::Txt),
        _ => Err(SourceListError::InvalidSourceListFileType),
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

pub fn read_source_list_file<T: AsRef<Path>>(
    file: T,
    sl_type: SourceListType,
) -> Result<SourceList, SourceListError> {
    debug!("Attempting to read source list");
    let sl_file_type = parse_file_type(&file)?;
    let mut f = std::io::BufReader::new(File::open(&file)?);

    match sl_type {
        SourceListType::Hyperdrive => match sl_file_type {
            SourceListFileType::Json => {
                hyperdrive::source_list_from_json(&mut f).map_err(|e| e.into())
            }
            SourceListFileType::Yaml => {
                hyperdrive::source_list_from_yaml(&mut f).map_err(|e| e.into())
            }
            _ => {
                // This is necessary, because .to_string() on a enum type
                // appends a newline character. Why isn't there a mutating
                // .trim_end()?
                let sl_type_str = sl_type.to_string().trim_end().to_string();

                Err(SourceListError::InvalidFormat {
                    sl_type: sl_type_str,
                    sl_file_type: sl_file_type.to_string(),
                })
            }
        },

        SourceListType::Rts => rts::parse_source_list(&mut f).map_err(|e| e.into()),

        SourceListType::Woden => woden::parse_source_list(&mut f).map_err(|e| e.into()),

        SourceListType::AO => ao::parse_source_list(&mut f).map_err(|e| e.into()),
    }
}

/// Errors associated with reading in any kind of source list.
#[derive(Error, Debug)]
pub enum SourceListError {
    #[error("Unrecognised source list file type. Valid types are: {}", *SOURCE_LIST_FILE_TYPES_COMMA_SEPARATED)]
    InvalidSourceListFileType,

    #[error("Unrecognised source list type. Valid types are: {}", *SOURCE_LIST_TYPES_COMMA_SEPARATED)]
    InvalidSourceListType,

    #[error(
        "Attempted to read a {sl_type}-style source list from a {sl_file_type} file, but that is not handled"
    )]
    InvalidFormat {
        sl_type: String,
        sl_file_type: String,
    },

    #[error("{0}")]
    ReadError(#[from] crate::error::ReadSourceListError),

    #[error("IO error: {0}")]
    IO(#[from] std::io::Error),
}
