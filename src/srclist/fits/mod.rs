// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Code to handle FITS source list files.

// The reference frequency of the power laws.
const REF_FREQ_HZ: f64 = 200e6;

mod read;
mod write;

// Re-exports.
pub(crate) use read::parse_source_list;
pub(crate) use write::write_source_list_jack;
