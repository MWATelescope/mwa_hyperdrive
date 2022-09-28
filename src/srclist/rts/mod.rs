// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Code to handle RTS source list files.
//!
//! All frequencies are in Hz. All flux densities are in Jy. All Gaussian and
//! Shapelet sizes are in arcmin, but their position angles are in degrees.
//!
//! RA is in decimal hours (0 to 24) and Dec are in degrees in the J2000 epoch.

mod read;
mod write;

// Re-exports.
pub(crate) use read::parse_source_list;
pub(crate) use write::write_source_list;
pub(crate) use write::write_source_list_with_order;
