// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Structures to describe sky-model sources and their components.

#[cfg(test)]
mod tests;

use marlu::{RADec, LMN};
use rayon::prelude::*;

use super::SourceComponent;
use crate::FluxDensity;
use mwa_hyperdrive_common::{marlu, rayon};

/// A collection of components.
#[derive(Clone, Debug, PartialEq)]
pub struct Source {
    /// The components associated with the source.
    pub components: Vec<SourceComponent>,
}

impl Source {
    /// Calculate the [LMN] coordinates of each component's [RADec].
    pub fn get_lmn(&self, phase_centre: RADec) -> Vec<LMN> {
        self.components
            .iter()
            .map(|comp| comp.radec.to_lmn(phase_centre))
            .collect()
    }

    /// Calculate the [LMN] coordinates of each component's [RADec]. The
    /// calculation is done in parallel.
    pub fn get_lmn_parallel(&self, phase_centre: RADec) -> Vec<LMN> {
        self.components
            .par_iter()
            .map(|comp| comp.radec.to_lmn(phase_centre))
            .collect()
    }

    /// Estimate the flux densities for each of a source's components given a
    /// frequency.
    pub fn get_flux_estimates(&self, freq_hz: f64) -> Vec<FluxDensity> {
        self.components
            .iter()
            .map(|comp| comp.flux_type.estimate_at_freq(freq_hz))
            .collect()
    }

    /// Estimate the flux densities for each of a source's components given a
    /// frequency. The calculation is done in parallel.
    pub fn get_flux_estimates_parallel(&self, freq_hz: f64) -> Vec<FluxDensity> {
        self.components
            .par_iter()
            .map(|comp| comp.flux_type.estimate_at_freq(freq_hz))
            .collect()
    }
}
