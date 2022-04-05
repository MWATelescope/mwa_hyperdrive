// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Code to handle reading from and writing to uvfits files.

mod error;
mod read;

#[cfg(test)]
mod tests;

pub(crate) use error::*;
pub(crate) use read::*;

use hifitime::Epoch;

use super::*;
use mwa_hyperdrive_common::hifitime;
