// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Code to handle "André Offringa"-style source list files.
//!
//! All frequencies have their units annotated (although this appears to only be
//! MHz). All flux densities are in Jy. All Gaussian and Shapelet sizes are in
//! arcsec, but their position angles are in degrees.
//!
//! RA is in decimal hours (0 to 24) and Dec is in degrees in the J2000 epoch,
//! but sexagesimal formatted.
//!
//! Source names are allowed to have spaces inside them, because the names are
//! surrounded in quotes. This is fine for reading, but when writing one of
//! these sources to another format, the spaces need to be translated to
//! underscores.

pub mod read;
pub mod write;

use super::error::*;
use crate::*;

// Re-exports.
pub use read::parse_source_list;
pub use write::write_source_list;
