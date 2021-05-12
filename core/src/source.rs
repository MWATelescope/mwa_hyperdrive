// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

/*!
Structures to describe sky-model sources and their components.
 */

use serde::{Deserialize, Serialize};

use crate::flux_density::EstimateError;
use crate::*;

use rayon::prelude::*;

#[derive(Clone, Debug, PartialEq)]
/// A collection of components.
pub struct Source {
    /// The components associated with the source.
    pub components: Vec<SourceComponent>,
}

impl Source {
    /// Calculate the (l,m,n) coordinates of each component's (RA,Dec).
    pub fn get_lmn(&self, pointing: &RADec) -> Vec<LMN> {
        self.components
            .iter()
            .map(|comp| comp.radec.to_lmn(&pointing))
            .collect()
    }

    /// Calculate the (l,m,n) coordinates of each component's (RA,Dec). The
    /// calculation is done in parallel.
    pub fn get_lmn_parallel(&self, pointing: &RADec) -> Vec<LMN> {
        self.components
            .par_iter()
            .map(|comp| comp.radec.to_lmn(&pointing))
            .collect()
    }

    /// Estimate the flux densities for each of a source's components given a
    /// frequency.
    pub fn get_flux_estimates(&self, freq_hz: f64) -> Result<Vec<FluxDensity>, EstimateError> {
        self.components
            .iter()
            .map(|comp| comp.flux_type.estimate_at_freq(freq_hz))
            .collect()
    }

    /// Estimate the flux densities for each of a source's components given a
    /// frequency. The calculation is done in parallel.
    pub fn get_flux_estimates_parallel(
        &self,
        freq_hz: f64,
    ) -> Result<Vec<FluxDensity>, EstimateError> {
        self.components
            .par_iter()
            .map(|comp| comp.flux_type.estimate_at_freq(freq_hz))
            .collect()
    }
}

#[derive(Clone, Debug, PartialEq)]
/// Information on a source's component.
pub struct SourceComponent {
    /// Coordinates struct associated with the component.
    pub radec: RADec,
    /// The type of component.
    pub comp_type: ComponentType,
    /// The flux densities associated with this component.
    pub flux_type: FluxDensityType,
}

impl SourceComponent {
    /// Estimate the flux density of this component at a frequency.
    pub fn estimate_at_freq(&self, freq_hz: f64) -> Result<FluxDensity, EstimateError> {
        self.flux_type.estimate_at_freq(freq_hz)
    }

    /// Is this component a point source?
    pub fn is_point(&self) -> bool {
        match self.comp_type {
            ComponentType::Point => true,
            _ => false,
        }
    }

    /// Is this component a gaussian source?
    pub fn is_gaussian(&self) -> bool {
        match self.comp_type {
            ComponentType::Gaussian { .. } => true,
            _ => false,
        }
    }

    /// Is this component a shapelet source?
    pub fn is_shapelet(&self) -> bool {
        match self.comp_type {
            ComponentType::Shapelet { .. } => true,
            _ => false,
        }
    }
}

/// Source types supported by hyperdrive.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum ComponentType {
    #[serde(rename = "point")]
    Point,

    #[serde(rename = "gaussian")]
    Gaussian {
        /// Major axis size \[radians\]
        maj: f64,
        /// Minor axis size \[radians\]
        min: f64,
        /// Position angle \[radians\]
        pa: f64,
    },

    #[serde(rename = "shapelet")]
    Shapelet {
        /// Major axis size \[radians\]
        maj: f64,
        /// Minor axis size \[radians\]
        min: f64,
        /// Position angle \[radians\]
        pa: f64,
        /// Shapelet coefficients
        coeffs: Vec<ShapeletCoeff>,
    },
}

impl ComponentType {
    // The following functions save the caller from using pattern matching to
    // determine the enum variant.

    /// Is this a point source?
    pub fn is_point(&self) -> bool {
        matches!(self, Self::Point)
    }

    /// Is this a gaussian source?
    pub fn is_gaussian(&self) -> bool {
        matches!(self, Self::Gaussian { .. })
    }

    /// Is this a shapelet source?
    pub fn is_shapelet(&self) -> bool {
        matches!(self, Self::Shapelet { .. })
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ShapeletCoeff {
    pub n1: usize,
    pub n2: usize,
    pub coeff: f64,
}
