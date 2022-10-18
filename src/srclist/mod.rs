// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Code for sky-model source lists. See for more info:
//! <https://mwatelescope.github.io/mwa_hyperdrive/defs/source_lists.html>

pub(crate) mod ao;
pub(crate) mod hyperdrive;
pub(crate) mod read;
pub(crate) mod rts;
pub(crate) mod types;
pub(crate) mod woden;

mod error;
#[cfg(test)]
mod general_tests;
mod veto;

pub(crate) use error::*;
pub(crate) use types::*;
pub(crate) use veto::*;

use itertools::Itertools;
use strum::IntoEnumIterator;

/// All of the possible sky-model sources list types.
#[derive(
    Debug,
    Clone,
    Copy,
    PartialEq,
    Eq,
    strum_macros::Display,
    strum_macros::EnumIter,
    strum_macros::EnumString,
)]
pub(crate) enum SourceListType {
    #[strum(serialize = "hyperdrive")]
    Hyperdrive,

    #[strum(serialize = "rts")]
    Rts,

    #[strum(serialize = "woden")]
    Woden,

    #[strum(serialize = "ao")]
    AO,
}

/// All of the possible file extensions that a hyperdrive-style sky-model source
/// list can have.
#[derive(
    Debug, Clone, Copy, strum_macros::Display, strum_macros::EnumIter, strum_macros::EnumString,
)]
pub(crate) enum HyperdriveFileType {
    #[strum(serialize = "yaml")]
    Yaml,

    #[strum(serialize = "json")]
    Json,
}

lazy_static::lazy_static! {
    pub(crate) static ref SOURCE_LIST_TYPES_COMMA_SEPARATED: String = SourceListType::iter().join(", ");

    pub(crate) static ref HYPERDRIVE_SOURCE_LIST_FILE_TYPES_COMMA_SEPARATED: String = HyperdriveFileType::iter().join(", ");
}
