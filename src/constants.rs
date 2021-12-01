// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Useful constants.

/// The maximum number of times to iterate when performing "MitchCal" in
/// direction-independent calibration.
pub(crate) const DEFAULT_MAX_ITERATIONS: usize = 50;

/// The threshold to satisfy convergence when performing "MitchCal" in
/// direction-independent calibration.
pub(crate) const DEFAULT_STOP_THRESHOLD: f64 = 1e-8;

/// The minimum threshold to satisfy convergence when performing "MitchCal" in
/// direction-independent calibration. Reaching this threshold counts as
/// "converged", but it's not as good as the stop threshold.
pub(crate) const DEFAULT_MIN_THRESHOLD: f64 = 1e-4;

/// The default calibration solutions filename to use.
pub(crate) const DEFAULT_OUTPUT_SOLUTIONS_FILENAME: &str = "hyperdrive_solutions.bin";

// sqrt(pi^2 / (2 ln(2)))
pub(crate) const SQRT_FRAC_PI_SQ_2_LN_2: f64 = 2.6682231283184983;

pub(crate) use mwa_rust_core::constants::*;
