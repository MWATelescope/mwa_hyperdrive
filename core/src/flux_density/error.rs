// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

/*!
Errors associated with flux density calculations.
 */

use thiserror::Error;

use super::FluxDensity;

#[derive(Error, Debug, PartialEq)]
pub enum EstimateError {
    #[error("Tried to estimate a flux density for a component, but it had no flux densities")]
    NoFluxDensities,

    #[error("The list of flux densities used for estimation were not sorted:\n{0:#?}")]
    FluxDensityListNotSorted(Vec<FluxDensity>),
}
