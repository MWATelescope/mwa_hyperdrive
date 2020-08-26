// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

/*!
Code to handle hyperdrive source list files.

All frequencies are in Hz. All flux densities are in Jy. All Gaussian and
Shapelet sizes are in arcsec, but their position angles are in degrees.

RA and Dec are in degrees in the J2000 epoch.
*/

pub mod read;
pub mod write;

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use super::error::*;
use crate::*;

pub(super) type TmpSourceList = BTreeMap<String, Vec<TmpComponent>>;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub(super) struct TmpComponent {
    /// Degrees
    ra: f64,
    /// Degrees
    dec: f64,
    comp_type: ComponentType,
    flux_type: TmpFluxDensityType,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum TmpFluxDensityType {
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
