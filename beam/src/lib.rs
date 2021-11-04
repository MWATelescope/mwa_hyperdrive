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

// Set a compile-time variable type.
cfg_if::cfg_if! {
    if #[cfg(feature = "cuda-single")] {
        pub use mwa_hyperbeam::cuda::*;
        pub type CudaFloat = f32;
    } else if #[cfg(all(feature = "cuda", not(feature = "cuda-single")))] {
        pub use mwa_hyperbeam::cuda::*;
        pub type CudaFloat = f64;
    }
}

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
    /// Using the tile information from this [Beam] and frequencies to be used,
    /// return a [BeamCUDA]. This object only needs pointings to calculate beam
    /// response [Jones] matrices.
    unsafe fn prepare_cuda_beam(&self, freqs_hz: &[u32]) -> Result<Box<dyn BeamCUDA>, BeamError>;
}

/// A trait abstracting beam code functions on a CUDA-capable device.
#[cfg(feature = "cuda")]
pub trait BeamCUDA {
    /// Calculate the Jones matrices for an [AzEl] direction. The pointing
    /// information is not needed because it was provided when `self` was
    /// created.
    ///
    /// # Safety
    ///
    /// This function interfaces directly with the CUDA API. Rust errors attempt
    /// to catch problems but there are no guarantees.
    unsafe fn calc_jones(
        &self,
        azels: &[AzEl],
    ) -> Result<DevicePointer<Jones<CudaFloat>>, BeamError>;

    /// Get the type of beam.
    fn get_beam_type(&self) -> BeamType;

    /// Get a pointer to the device beam Jones map. This is necessary to access
    /// de-duplicated beam Jones matrices on the device.
    fn get_beam_jones_map(&self) -> *const u64;

    /// Get the number of de-duplicated frequencies associated with this
    /// [BeamCUDA].
    fn get_num_unique_freqs(&self) -> i32;
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
pub struct NoBeam {
    /// This is needed for the CUDA "beam jones map".
    #[cfg(feature = "cuda")]
    num_tiles: usize,
}

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
    unsafe fn prepare_cuda_beam(&self, freqs_hz: &[u32]) -> Result<Box<dyn BeamCUDA>, BeamError> {
        let obj = NoBeamCUDA {
            beam_jones_map: DevicePointer::copy_to_device(&vec![
                0;
                self.num_tiles * freqs_hz.len()
            ])?,
        };
        Ok(Box::new(obj))
    }
}

/// A beam implementation that returns only identity Jones matrices for all beam
/// calculations.
#[cfg(feature = "cuda")]
pub struct NoBeamCUDA {
    beam_jones_map: DevicePointer<u64>,
}

#[cfg(feature = "cuda")]
impl BeamCUDA for NoBeamCUDA {
    unsafe fn calc_jones(
        &self,
        azels: &[AzEl],
    ) -> Result<DevicePointer<Jones<CudaFloat>>, BeamError> {
        let identities: Array3<Jones<CudaFloat>> =
            Array3::from_elem((1, 1, azels.len()), Jones::identity());
        DevicePointer::copy_to_device(identities.as_slice().unwrap()).map_err(BeamError::from)
    }

    fn get_beam_type(&self) -> BeamType {
        BeamType::None
    }

    fn get_beam_jones_map(&self) -> *const u64 {
        self.beam_jones_map.get()
    }

    fn get_num_unique_freqs(&self) -> i32 {
        1
    }
}

/// Create a "no beam" object.
pub fn create_no_beam_object(num_tiles: usize) -> Box<dyn Beam> {
    cfg_if::cfg_if! {
        if #[cfg(feature = "cuda")] {
            Box::new(NoBeam {
                num_tiles
            })
        } else {
            Box::new(NoBeam {})
        }
    }
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
        let beam = create_no_beam_object(1);
        for azel in azels {
            let j = beam.calc_jones(azel, 150e6, 0).unwrap();
            assert_abs_diff_eq!(j, Jones::identity());
        }
    }
}
