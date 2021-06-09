// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Useful constants.

/// Sources with beam-attenuated flux densities less than this value are
/// discarded from sky-model source lists.
pub(crate) const DEFAULT_VETO_THRESHOLD: f64 = 0.01;

/// Sources with elevations less than this value are discarded from sky-model
/// source lists.
pub(crate) const ELEVATION_LIMIT: f64 = 0.0;

/// Sources that are separated by more than this value (degrees) from the
/// pointing are discarded from sky-model source lists.
pub(crate) const CUTOFF_DISTANCE: f64 = 30.0;

/// The maximum number of times to iterate when performing "MitchCal" in
/// direction-independent calibration.
pub(crate) const DEFAULT_MAX_ITERATIONS: usize = 50;

/// The threshold to satisfy convergence when performing "MitchCal" in
/// direction-independent calibration.
pub(crate) const DEFAULT_STOP_THRESHOLD: f32 = 1e-8;

/// The minimum threshold to satisfy convergence when performing "MitchCal" in
/// direction-independent calibration. Reaching this threshold counts as
/// "converged", but it's not as good as the stop threshold.
pub(crate) const DEFAULT_MIN_THRESHOLD: f32 = 1e-5;

/// The default calibration solutions filename to use.
pub(crate) const DEFAULT_OUTPUT_SOLUTIONS_FILENAME: &str = "hyperdrive_solutions.bin";

/// Alan Levine's gains from PFB simulations. Taken from RTS source code.
pub(crate) const LEVINE_GAINS_40KHZ: [f64; 32] = [
    0.5173531193404733,
    0.5925143901943901,
    0.7069509925949563,
    0.8246794181334419,
    0.9174323810107883,
    0.9739924923371597,
    0.9988235178442829,
    1.0041872682882493,
    1.0021295484391897,
    1.0000974383045906,
    1.0004197495080835,
    1.002092702099684,
    1.003201858357689,
    1.0027668031914465,
    1.001305418352239,
    1.0001674256814668,
    1.0003506058381628,
    1.001696297529349,
    1.0030147335641364,
    1.0030573420014388,
    1.0016582119173054,
    1.0001394672444315,
    1.0004004241051296,
    1.002837790192105,
    1.0039523509152424,
    0.9949679743767017,
    0.9632053940967067,
    0.8975113804877556,
    0.7967436134595853,
    0.6766433460480191,
    0.5686988482410316,
    0.5082890508180502,
];

/// Gains from empirical averaging of RTS BP solution points using "Anish" PFB
/// gains for 1062363808 and backing out corrections to flatten average coarse
/// channel.
pub(crate) const EMPIRICAL_GAINS_40KHZ: [f64; 32] = [
    0.5, 0.5, 0.67874855, 0.83576969, 0.95187049, 1.0229769, 1.05711736, 1.06407012, 1.06311151,
    1.06089592, 1.0593481, 1.06025714, 1.06110822, 1.05893943, 1.05765503, 1.05601938, 0.5,
    1.05697461, 1.05691842, 1.05688129, 1.05623901, 1.05272663, 1.05272112, 1.05551337, 1.05724941,
    1.0519857, 1.02483081, 0.96454596, 0.86071928, 0.71382954, 0.5, 0.5,
];

/// This is the number of seconds from 1900 Jan 1 and 1980 Jan 5. The GPS epoch
/// is 1980 Jan 5, but hifitime uses 1900 for everything; subtracting this
/// number from the result of hifitime::Epoch::as_gpst_seconds gives the
/// expected GPS time.
pub(crate) const HIFITIME_GPS_FACTOR: f64 =
    hifitime::SECONDS_PER_YEAR * 80.0 + hifitime::SECONDS_PER_DAY * 4.0;

/// The number of seconds between 1858-11-17T00:00:00 (MJD epoch, used by
/// casacore) and 1900-01-01T00:00:00 (TAI epoch) is 1297728000. I'm using the
/// TAI epoch because that's well supported by hifitime, and hifitime converts an
/// epoch to many formats including JD, and accounts for leap seconds.
pub(crate) const MJD_TAI_EPOCH_DIFF: f64 = 1297728000.0;

// sqrt(pi^2 / (2 ln(2)))
pub(crate) const SQRT_FRAC_PI_SQ_2_LN_2: f64 = 2.6682231283184983;

pub(crate) use mwa_hyperdrive_core::constants::*;
