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

#[derive(Debug, Clone, EnumIter)]
pub enum SourceListType {
    Hyperdrive,
    Rts,
    Woden,
    AO,
}

lazy_static::lazy_static! {
    pub static ref SOURCE_LIST_FILE_TYPES_COMMA_SEPARATED: String = {
        let mut variants: Vec<String> = vec![];
        // Iterate over all of the enum variants for SourceListType.
        for variant in SourceListFileType::iter() {
            let s = format!("{}", variant);
            // Each string has a trailing newline character.
            variants.push(s.strip_suffix("\n").unwrap().to_string());
        }
        variants.join(", ")
    };

    pub static ref SOURCE_LIST_TYPES_COMMA_SEPARATED: String = {
        let mut variants: Vec<String> = vec![];
        // Iterate over all of the enum variants for SourceListType.
        for variant in SourceListType::iter() {
            let s = format!("{}", variant);
            // Each string has a trailing newline character.
            variants.push(s.strip_suffix("\n").unwrap().to_string());
        }
        variants.join(", ")
    };
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

// Convenience imports.
use mwa_hyperdrive_core::*;

// External re-exports.
pub use mwa_hyperdrive_core;
