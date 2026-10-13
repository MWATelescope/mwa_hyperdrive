// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Code to handle DP3/BBS-style ("makesourcedb") text sky models, as used by
//! DP3 and WSClean.
//!
//! See for more info:
//! <https://mwatelescope.github.io/mwa_hyperdrive/defs/source_list_dp3.html>

mod read;

// Re-exports.
pub(crate) use read::{looks_like_dp3, parse_source_list};
