// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Code to handle hyperdrive source list files.
//!
//! See for more info:
//! <https://mwatelescope.github.io/mwa_hyperdrive/defs/source_list_hyperdrive.html>

mod read;
mod write;

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use super::{ComponentType, FluxDensity};

pub(super) type TmpSourceList = BTreeMap<String, Vec<TmpComponent>>;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub(super) struct TmpComponent {
    /// \[degrees\]
    pub(super) ra: f64,
    /// \[degrees\]
    pub(super) dec: f64,
    pub(super) comp_type: ComponentType,
    pub(super) flux_type: TmpFluxDensityType,
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
pub(crate) use read::{source_list_from_json, source_list_from_yaml};
pub(crate) use write::{source_list_to_json, source_list_to_yaml};
