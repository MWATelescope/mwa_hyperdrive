// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Structures to describe sky-model sources and their components.

#[cfg(test)]
mod tests;

use serde::{Deserialize, Serialize};

use super::{FluxDensity, SourceComponent};

/// A collection of components.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(transparent)]
pub struct Source {
    /// The components associated with the source.
    #[serde(with = "serde_yaml::with::singleton_map_recursive")]
    pub components: Box<[SourceComponent]>,
}

impl Source {
    /// Estimate the flux densities for each of a source's components given a
    /// frequency.
    pub(crate) fn get_flux_estimates(&self, freq_hz: f64) -> Vec<FluxDensity> {
        self.components
            .iter()
            .map(|comp| comp.flux_type.estimate_at_freq(freq_hz))
            .collect()
    }
}

#[cfg(test)]
impl approx::AbsDiffEq for Source {
    type Epsilon = f64;

    fn default_epsilon() -> Self::Epsilon {
        f64::EPSILON
    }

    fn abs_diff_eq(&self, other: &Self, epsilon: Self::Epsilon) -> bool {
        approx::abs_diff_eq!(*self.components, *other.components, epsilon = epsilon)
    }
}
