// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Code and utilities for sky-model source lists. See for more info:
//! <https://mwatelescope.github.io/mwa_hyperdrive/defs/source_lists.html>

pub mod ao;
pub mod constants;
pub mod hyperdrive;
pub mod read;
pub mod rts;
pub mod types;
pub mod utilities;
pub mod woden;

mod error;
#[cfg(test)]
mod general_tests;
mod veto;

pub use error::*;
pub use types::*;
pub use utilities::*;
pub use veto::*;

use itertools::Itertools;
use strum::IntoEnumIterator;
use strum_macros::{Display, EnumIter, EnumString};

use mwa_hyperdrive_common::{itertools, lazy_static};

/// All of the possible sky-model sources list types.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Display, EnumIter, EnumString)]
pub enum SourceListType {
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
#[derive(Debug, Display, Clone, Copy, EnumIter, EnumString)]
pub enum HyperdriveFileType {
    #[strum(serialize = "yaml")]
    Yaml,

    #[strum(serialize = "json")]
    Json,
}

lazy_static::lazy_static! {
    pub static ref SOURCE_LIST_TYPES_COMMA_SEPARATED: String = SourceListType::iter().join(", ");

    pub static ref HYPERDRIVE_SOURCE_LIST_FILE_TYPES_COMMA_SEPARATED: String = HyperdriveFileType::iter().join(", ");

    pub static ref SRCLIST_BY_BEAM_OUTPUT_TYPE_HELP: String =
    format!("Specifies the type of the output source list. If not specified, the input source list type is used. Currently supported types: {}",
            *SOURCE_LIST_TYPES_COMMA_SEPARATED);

    pub static ref SOURCE_DIST_CUTOFF_HELP: String =
    format!("Specifies the maximum distance from the phase centre a source can be [degrees]. Default: {}",
            DEFAULT_CUTOFF_DISTANCE);

    pub static ref VETO_THRESHOLD_HELP: String =
    format!("Specifies the minimum Stokes XX+YY a source must have before it gets vetoed [Jy]. Default: {}",
            DEFAULT_VETO_THRESHOLD);

    pub static ref SOURCE_LIST_INPUT_TYPE_HELP: String =
    format!("Specifies the type of the input source list. Currently supported types: {}",
                *SOURCE_LIST_TYPES_COMMA_SEPARATED);

    pub static ref SOURCE_LIST_OUTPUT_TYPE_HELP: String =
    format!("Specifies the type of the output source list. May be required depending on the output filename. Currently supported types: {}",
            *SOURCE_LIST_TYPES_COMMA_SEPARATED);
}

// External re-exports.
pub use mwa_hyperdrive_common::{marlu, ndarray, rayon};
