// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Useful constants.

/// When a spectral index must be assumed, this value is used.
pub(crate) const DEFAULT_SPEC_INDEX: f64 = -0.8;

/// The smallest spectral index before we report that things look fishy.
pub(crate) const SPEC_INDEX_CAP: f64 = -2.0;

/// The minimum Stokes XX+YY a source must have before it gets vetoed \[Jy\].
/// Sources with beam-attenuated flux densities less than this value are
/// discarded from sky-model source lists.
pub(crate) const DEFAULT_VETO_THRESHOLD: f64 = 0.01;

/// Sources with elevations less than this value are discarded from sky-model
/// source lists \[degrees\].
pub(crate) const ELEVATION_LIMIT: f64 = 0.0;

/// Sources that are separated by more than this value from the phase centre are
/// discarded from sky-model source lists \[degrees\].
pub(crate) const DEFAULT_CUTOFF_DISTANCE: f64 = 50.0;

// sqrt(pi^2 / (2 ln(2)))
pub(crate) const SQRT_FRAC_PI_SQ_2_LN_2: f64 = 2.6682231283184983;

pub(crate) use marlu::constants::*;

/// The default column to use when reading visibilities from a measurement set.
pub(crate) const DEFAULT_MS_DATA_COL_NAME: &str = "DATA";
