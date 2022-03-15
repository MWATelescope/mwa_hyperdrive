// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Code surrounding the [BTreeMap] used to contain all sky-model sources and
//! their components.

#[cfg(test)]
mod tests;

use std::collections::BTreeMap;
use std::ops::{Deref, DerefMut};

use marlu::{constants::MWA_LAT_RAD, AzEl, RADec, LMN};
use rayon::prelude::*;

use super::*;
use mwa_hyperdrive_common::{marlu, rayon};

/// A [BTreeMap] of source names for keys and [Source] structs for values.
///
/// By making [SourceList] a new type (specifically, an anonymous struct),
/// useful methods can be put onto it.
#[derive(Debug, Clone, Default)]
pub struct SourceList(BTreeMap<String, Source>);

impl SourceList {
    /// Create an empty [SourceList].
    pub fn new() -> Self {
        Self::default()
    }

    /// Get counts of each of the component types and flux-density types.
    pub fn get_counts(&self) -> ComponentCounts {
        let mut counts = ComponentCounts::default();
        self.iter()
            .flat_map(|(_, src)| &src.components)
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

    /// Filter component types from one [SourceList] and return a new one.
    pub fn filter(
        self,
        filter_points: bool,
        filter_gaussians: bool,
        filter_shapelets: bool,
    ) -> SourceList {
        let sl: BTreeMap<_, _> = self
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

    /// Get azimuth and elevation coordinates for all components of all sources.
    /// Useful for interfacing with beam code.
    ///
    /// Because [SourceList] is a [BTreeMap], the order of the sources is always
    /// the same, so the [AzEl] coordinates returned from this function are 1:1
    /// with sources and their components.
    pub fn get_azel(&self, lst_rad: f64, latitude_rad: f64) -> Vec<AzEl> {
        self.iter()
            // For each source, get all of its component's (Az, El) coordinates.
            .flat_map(|(_, src)| &src.components)
            .map(|comp| comp.radec.to_hadec(lst_rad).to_azel(latitude_rad))
            .collect()
    }

    /// Get azimuth and elevation coordinates for all components of all sources,
    /// assuming that the latitude is the MWA's latitude. See the documentation
    /// for `SourceList::get_azel` for more details.
    pub fn get_azel_mwa(&self, lst_rad: f64) -> Vec<AzEl> {
        self.get_azel(lst_rad, MWA_LAT_RAD)
    }

    /// Get azimuth and elevation coordinates for all components of all sources.
    /// Useful for interfacing with beam code. The sources are iterated in
    /// parallel.
    ///
    /// Because [SourceList] is a [BTreeMap], the order of the sources is always
    /// the same, so the [AzEl] coordinates returned from this function are 1:1
    /// with sources and their components.
    pub fn get_azel_parallel(&self, lst_rad: f64, latitude_rad: f64) -> Vec<AzEl> {
        self.par_iter()
            .flat_map(|(_, src)| src.components.as_slice())
            .map(|comp| comp.radec.to_hadec(lst_rad).to_azel(latitude_rad))
            .collect()
    }

    /// Get azimuth and elevation coordinates for all components of all sources,
    /// assuming that the latitude is the MWA's latitude. See the documentation
    /// for `SourceList::get_azel` for more details. The sources are iterated in
    /// parallel.
    pub fn get_azel_mwa_parallel(&self, lst_rad: f64) -> Vec<AzEl> {
        self.get_azel_parallel(lst_rad, MWA_LAT_RAD)
    }

    /// Get the LMN coordinates for all components of all sources.
    pub fn get_lmns(&self, phase_centre: RADec) -> Vec<LMN> {
        self.iter()
            .flat_map(|(_, src)| &src.components)
            .map(|comp| comp.radec.to_lmn(phase_centre))
            .collect()
    }

    /// Get the LMN coordinates for all components of all sources. The sources
    /// are iterated in parallel.
    pub fn get_lmns_parallel(&self, phase_centre: RADec) -> Vec<LMN> {
        self.par_iter()
            .flat_map(|(_, src)| src.components.as_slice())
            .map(|comp| comp.radec.to_lmn(phase_centre))
            .collect()
    }
}

impl From<BTreeMap<String, Source>> for SourceList {
    fn from(sl: BTreeMap<String, Source>) -> Self {
        Self(sl)
    }
}

impl Deref for SourceList {
    type Target = BTreeMap<String, Source>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl DerefMut for SourceList {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl IntoIterator for SourceList {
    type Item = (String, Source);
    type IntoIter = std::collections::btree_map::IntoIter<String, Source>;

    fn into_iter(self) -> std::collections::btree_map::IntoIter<String, Source> {
        self.0.into_iter()
    }
}

#[derive(Debug, Default)]
pub struct ComponentCounts {
    pub num_points: usize,
    pub num_gaussians: usize,
    pub num_shapelets: usize,
    pub num_power_laws: usize,
    pub num_curved_power_laws: usize,
    pub num_lists: usize,
}
