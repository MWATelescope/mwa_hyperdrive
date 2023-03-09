// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! File stuff (input/output, reading/writing, globs), for visibilities and
//! others.

mod glob;
pub(crate) mod read;
pub(crate) mod write;

pub(crate) use self::glob::{get_all_matches_from_glob, get_single_match_from_glob, GlobError};
