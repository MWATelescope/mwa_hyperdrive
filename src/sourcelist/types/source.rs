// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

use super::flux_density::FluxDensity;
use crate::coord::*;

use rayon::prelude::*;

#[derive(Clone, Debug)]
pub struct Source {
    /// Short name of the source.
    pub name: String,
    /// The components associated with the source.
    pub components: Vec<SourceComponent>,
}

impl Source {
    /// Calculate the (l,m,n) coordinates of each component's (RA,Dec). The
    /// calculation is done in parallel.
    pub fn get_lmn(&self, pc: &PointingCentre) -> Vec<LMN> {
        self.components
            .par_iter()
            .map(|comp| comp.radec.to_lmn(&pc))
            .collect()
    }
}

#[derive(Clone, Debug)]
pub struct SourceComponent {
    /// Coordinates struct associated with the component.
    pub radec: RADec,
    /// Extra parameters for a source type.
    ///
    /// Unfortunately "type" is reserved word.
    pub ctype: ComponentType,
    /// The flux densities associated with this component.
    pub flux_densities: Vec<FluxDensity>,
}

/// Parameters describing a Gaussian sky-model source.
pub struct Gaussian {
    pub major: f64,
}

/// Which component types does the hyperdrive allow?
#[derive(Clone, Copy, Debug)]
pub enum ComponentType {
    Point,
    Gaussian,
    Shapelet,
}
