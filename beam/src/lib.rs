// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Code to abstract beam calculations.
//!
//! [Beam] is a trait detailing how to perform various beam-related tasks. By
//! making this trait, we can neatly abstract over multiple beam codes,
//! including a simple [NoBeam] type (which just returns identity matrices).
//!
//! Note that (where applicable) `norm_to_zenith` is always true; the
//! implication being that a sky-model source's brightness is always assumed to
//! be correct when at zenith.

mod cache;
mod error;
mod fee;
pub use error::*;
pub use fee::*;

use mwa_rust_core::{AzEl, Jones};
use ndarray::prelude::*;

#[cfg(feature = "cuda")]
pub use mwa_hyperbeam_cuda::fee::write_fee_cuda_file;

/// Supported beam types.
pub enum BeamType {
    /// Fully-embedded element beam.
    FEE,

    /// a.k.a. `NoBeam`. Only returns identity matrices.
    None,
}

/// A trait abstracting beam code functions.
pub trait Beam: Sync + Send {
    /// Calculate the Jones matrices for an [AzEl] direction. The pointing
    /// information is not needed because it was provided when `self` was
    /// created.
    fn calc_jones(
        &self,
        azel: AzEl,
        freq_hz: f64,
        tile_index: usize,
    ) -> Result<Jones<f64>, BeamError>;

    /// Get the type of beam.
    fn get_beam_type(&self) -> BeamType;

    /// Given a frequency in Hz, find the closest frequency that the beam code
    /// is defined for. An example of when this is important is with the FEE
    /// beam code, which can only give beam responses at specific frequencies.
    /// On the other hand, the analytic beam can be used at any frequency.
    fn find_closest_freq(&self, desired_freq_hz: f64) -> f64;

    /// Get the size of the [Jones] cache associated with this [Beam].
    fn len(&self) -> usize;

    /// Is the [Jones] cache empty?
    fn is_empty(&self) -> bool;

    /// Empty the [Jones] cache.
    fn empty_cache(&self);

    /// If this [Beam] supports it, empty the coefficient cache.
    fn empty_coeff_cache(&self);

    #[cfg(feature = "cuda")]
    fn get_device_pointers(
        &self,
        freqs_hz: &[u32],
    ) -> Option<mwa_hyperbeam_cuda::fee::FEEDeviceCoeffPointers>;
}

/// An enum to track whether MWA dipole delays are provided and/or necessary.
#[derive(Debug)]
pub enum Delays {
    /// Delays are fully specified.
    Full(Array2<u32>),

    /// Delays are specified for a single tile. If this can't be refined, then
    /// we must assume that these dipoles apply to all tiles.
    Partial(Vec<u32>),

    /// Delays have not been provided, but are necessary for calibration.
    None,

    /// Delays are not necessary, probably because no beam code is being used.
    NotNecessary,
}

/// A beam implementation that returns only identity Jones matrices for all beam
/// calculations.
pub struct NoBeam;

impl Beam for NoBeam {
    fn calc_jones(
        &self,
        _azel: AzEl,
        _freq_hz: f64,
        _tile_index: usize,
    ) -> Result<Jones<f64>, BeamError> {
        Ok(Jones::identity())
    }

    fn find_closest_freq(&self, desired_freq_hz: f64) -> f64 {
        desired_freq_hz
    }

    fn get_beam_type(&self) -> BeamType {
        BeamType::None
    }

    // No caches associated with `NoBeam`.
    fn len(&self) -> usize {
        0
    }
    fn is_empty(&self) -> bool {
        true
    }
    fn empty_cache(&self) {}
    fn empty_coeff_cache(&self) {}

    #[cfg(feature = "cuda")]
    fn get_device_pointers(
        &self,
        _freqs_hz: &[u32],
    ) -> Option<mwa_hyperbeam_cuda::fee::FEEDeviceCoeffPointers> {
        None
    }
}

/// Create a "no beam" object.
pub fn create_no_beam_object() -> Box<dyn Beam> {
    Box::new(NoBeam)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn no_beam_means_no_beam() {
        let azels = [
            AzEl { az: 0.0, el: 0.0 },
            AzEl { az: 1.0, el: 0.1 },
            AzEl { az: -1.0, el: 0.2 },
        ];
        let beam = NoBeam;
        for azel in azels {
            let j = beam.calc_jones(azel, 150e6 as _, 0).unwrap();
            assert_abs_diff_eq!(j, Jones::identity());
        }
    }
}
