// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

/*!
Code to handle sky-model source lists.
 */

pub mod ao;
pub mod error;
pub mod hyperdrive;
pub mod rts;
pub mod woden;

use strum_macros::EnumIter;

// Convenience re-exports.
use std::collections::BTreeMap;

use mwa_hyperdrive_core::constants::*;

// Re-exports.
pub use mwa_hyperdrive_core::*;

#[derive(Debug, EnumIter)]
pub enum SourceListFileType {
    Json,
    Yaml,
    Txt,
}

#[derive(Debug, Clone, EnumIter)]
pub enum SourceListType {
    Hyperdrive,
    Rts,
    Woden,
    AO,
}

impl std::fmt::Display for SourceListFileType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(
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
        writeln!(
            f,
            "{}",
            match &self {
                SourceListType::Hyperdrive => "hyperdrive",
                SourceListType::Rts => "rts",
                SourceListType::Woden => "woden",
                SourceListType::AO => "ao",
            }
        )
    }
}
