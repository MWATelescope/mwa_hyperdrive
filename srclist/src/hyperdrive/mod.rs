// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Code to handle hyperdrive source list files.
//!
//! See for more info:
//! <https://github.com/MWATelescope/mwa_hyperdrive/wiki/Source-lists>

mod read;
mod write;

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use super::error::*;
use crate::*;

pub(super) type TmpSourceList = BTreeMap<String, Vec<TmpComponent>>;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub(super) struct TmpComponent {
    /// \[degrees\]
    ra: f64,
    /// \[degrees\]
    dec: f64,
    comp_type: ComponentType,
    flux_type: TmpFluxDensityType,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub(super) enum TmpFluxDensityType {
    #[serde(rename = "list")]
    List(Vec<FluxDensity>),

    #[serde(rename = "power_law")]
    PowerLaw { si: f64, fd: FluxDensity },

    #[serde(rename = "curved_power_law")]
    CurvedPowerLaw { si: f64, fd: FluxDensity, q: f64 },
}

// Re-exports.
pub use read::{source_list_from_json, source_list_from_yaml};
pub use write::{source_list_to_json, source_list_to_yaml};
