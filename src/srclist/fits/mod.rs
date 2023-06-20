// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Code to handle FITS source list files.

mod read;
// mod write;

// Re-exports.
pub(crate) use read::parse_source_list;
// pub(crate) use write::write_source_list;
