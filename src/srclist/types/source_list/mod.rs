// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Code surrounding the [`IndexMap`] used to contain all sky-model sources and
//! their components.

#[cfg(test)]
mod tests;

use std::ops::{Deref, DerefMut};

use indexmap::IndexMap;
use serde::{Deserialize, Serialize};

use super::*;

/// A [`IndexMap`] of source names for keys and [`Source`] structs for values.
///
/// By making [`SourceList`] a new type (specifically, an anonymous struct),
/// useful methods can be put onto it.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(transparent)]
pub struct SourceList(IndexMap<String, Source>);

impl SourceList {
    /// Create an empty [`SourceList`].
    pub(crate) fn new() -> Self {
        Self::default()
    }

    /// Get counts of each of the component types and flux-density types.
    pub(crate) fn get_counts(&self) -> ComponentCounts {
        let mut counts = ComponentCounts::default();
        self.iter()
            .flat_map(|(_, src)| src.components.iter())
            .for_each(|c| {
                match (c.is_point(), c.is_gaussian(), c.is_shapelet()) {
                    (true, false, false) => counts.num_points += 1,
                    (false, true, false) => counts.num_gaussians += 1,
                    (false, false, true) => counts.num_shapelets += 1,
                    _ => {
                        unreachable!();
                    }
                }
                match (
                    matches!(c.flux_type, FluxDensityType::PowerLaw { .. }),
                    matches!(c.flux_type, FluxDensityType::CurvedPowerLaw { .. }),
                    matches!(c.flux_type, FluxDensityType::List { .. }),
                ) {
                    (true, false, false) => counts.num_power_laws += 1,
                    (false, true, false) => counts.num_curved_power_laws += 1,
                    (false, false, true) => counts.num_lists += 1,
                    _ => {
                        unreachable!();
                    }
                }
            });
        counts
    }

    /// Filter component types from one [`SourceList`] and return a new one.
    pub(crate) fn filter(
        self,
        filter_points: bool,
        filter_gaussians: bool,
        filter_shapelets: bool,
    ) -> SourceList {
        let sl: IndexMap<_, _> = self
            .0
            .into_iter()
            // Filter sources containing any of the rejected types.
            .filter_map(|(name, src)| {
                if !(filter_points && src.components.iter().any(|c| c.comp_type.is_point())
                    || filter_gaussians && src.components.iter().any(|c| c.comp_type.is_gaussian())
                    || filter_shapelets && src.components.iter().any(|c| c.comp_type.is_shapelet()))
                {
                    Some((name, src))
                } else {
                    None
                }
            })
            .collect();
        SourceList(sl)
    }
}

impl From<IndexMap<String, Source>> for SourceList {
    fn from(sl: IndexMap<String, Source>) -> Self {
        Self(sl)
    }
}

impl<const N: usize> From<[(String, Source); N]> for SourceList {
    fn from(value: [(String, Source); N]) -> Self {
        Self(IndexMap::from(value))
    }
}

impl Deref for SourceList {
    type Target = IndexMap<String, Source>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl DerefMut for SourceList {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl FromIterator<(String, Source)> for SourceList {
    fn from_iter<I: IntoIterator<Item = (String, Source)>>(iter: I) -> Self {
        let mut c = Self::new();
        for i in iter {
            c.insert(i.0, i.1);
        }
        c
    }
}

impl IntoIterator for SourceList {
    type Item = (String, Source);
    type IntoIter = indexmap::map::IntoIter<String, Source>;

    fn into_iter(self) -> indexmap::map::IntoIter<String, Source> {
        self.0.into_iter()
    }
}

#[derive(Debug, Default)]
pub(crate) struct ComponentCounts {
    pub(crate) num_points: usize,
    pub(crate) num_gaussians: usize,
    pub(crate) num_shapelets: usize,
    pub(crate) num_power_laws: usize,
    pub(crate) num_curved_power_laws: usize,
    pub(crate) num_lists: usize,
}
