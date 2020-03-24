// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

pub(crate) mod simulate_vis;
pub(crate) mod verify_srclist;

// Re-exports.
use std::fs::File;
use std::io::Read;
use std::path::PathBuf;

use anyhow::bail;

pub(crate) use simulate_vis::*;
pub(crate) use verify_srclist::*;
