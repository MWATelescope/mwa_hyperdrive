// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Code to handle hyperdrive source list files.
//!
//! See for more info:
//! <https://mwatelescope.github.io/mwa_hyperdrive/defs/source_list_hyperdrive.html>

mod read;
mod write;

// Re-exports.
pub(crate) use read::{source_list_from_json, source_list_from_yaml};
pub(crate) use write::{source_list_to_json, source_list_to_yaml};
