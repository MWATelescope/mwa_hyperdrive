// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Code to handle "André Offringa"-style source list files.
//!
//! See for more info:
//! <https://github.com/MWATelescope/mwa_hyperdrive/wiki/Source-lists>

mod read;
mod write;

use super::error::*;
use crate::*;

// Re-exports.
pub use read::parse_source_list;
pub use write::write_source_list;
