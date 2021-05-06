// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

/*!
Code to handle sky-model source lists.
 */

pub mod ao;
pub mod error;
pub mod hyperdrive;
pub mod read;
pub mod rts;
pub mod woden;

use strum::IntoEnumIterator;
use strum_macros::EnumIter;

#[derive(Debug, EnumIter)]
pub enum SourceListFileType {
    Json,
    Yaml,
    Txt,
}

#[derive(Debug, Clone, Copy, EnumIter, PartialEq)]
pub enum SourceListType {
    Hyperdrive,
    Rts,
    Woden,
    AO,
    Unspecified,
}

lazy_static::lazy_static! {
    pub static ref SOURCE_LIST_TYPES_COMMA_SEPARATED: String = {
        // Iterate over all of the enum variants for SourceListType and join
        // them in a comma-separated string. Ignore the "Unspecified" enum
        // variant.
        SourceListType::iter()
            .filter(|&ft| ft != SourceListType::Unspecified)
            .map(|ft| ft.to_string())
            .collect::<Vec<_>>()
            .join(", ")
    };

    pub static ref SOURCE_LIST_FILE_TYPES_COMMA_SEPARATED: String = {
        // Iterate over all of the enum variants for SourceListFileType and join
        // them in a comma-separated string.
        SourceListFileType::iter().map(|ft| ft.to_string()).collect::<Vec<_>>().join(", ")
    };
}

impl std::fmt::Display for SourceListFileType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}",
            match &self {
                SourceListFileType::Json => "json",
                SourceListFileType::Yaml => "yaml",
                SourceListFileType::Txt => "txt",
            }
        )
    }
}

impl std::fmt::Display for SourceListType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}",
            match &self {
                SourceListType::Hyperdrive => "hyperdrive",
                SourceListType::Rts => "rts",
                SourceListType::Woden => "woden",
                SourceListType::AO => "ao",
                SourceListType::Unspecified => "unspecified",
            }
        )
    }
}

// Convenience imports.
use mwa_hyperdrive_core::*;

// External re-exports.
pub use mwa_hyperdrive_core;
pub use mwa_hyperdrive_core::SourceList;
