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

mod error;
mod fee;
#[cfg(test)]
mod jones_test;
#[cfg(test)]
mod tests;

pub use error::*;
pub use fee::*;

use marlu::{AzEl, Jones};
use ndarray::prelude::*;

pub use mwa_hyperbeam;
use mwa_hyperdrive_common::{cfg_if, marlu, ndarray};

// Set a compile-time variable type.
cfg_if::cfg_if! {
    if #[cfg(feature = "cuda-single")] {
        pub use marlu::cuda::*;
        /// f32 (using the "cuda-single" feature)
        pub type CudaFloat = f32;
    } else if #[cfg(all(feature = "cuda", not(feature = "cuda-single")))] {
        pub use marlu::cuda::*;
        /// f64 (using the "cuda" feature and not "cuda-single")
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
    /// Get the type of beam.
    fn get_beam_type(&self) -> BeamType;

    /// Get the number of tiles associated with this beam. This is determined by
    /// how many delays have been provided.
    fn get_num_tiles(&self) -> usize;

    /// Calculate the beam-response Jones matrix for an [AzEl] direction. The
    /// pointing information is not needed because it was provided when `self`
    /// was created.
    fn calc_jones(
        &self,
        azel: AzEl,
        freq_hz: f64,
        tile_index: usize,
    ) -> Result<Jones<f64>, BeamError>;

    /// Calculate the beam-response Jones matrices for multiple [AzEl]
    /// directions. The pointing information is not needed because it was
    /// provided when `self` was created.
    fn calc_jones_array(
        &self,
        azels: &[AzEl],
        freq_hz: f64,
        tile_index: usize,
    ) -> Result<Vec<Jones<f64>>, BeamError>;

    /// Given a frequency in Hz, find the closest frequency that the beam code
    /// is defined for. An example of when this is important is with the FEE
    /// beam code, which can only give beam responses at specific frequencies.
    /// On the other hand, the analytic beam can be used at any frequency.
    fn find_closest_freq(&self, desired_freq_hz: f64) -> f64;

    /// If this [Beam] supports it, empty the coefficient cache.
    fn empty_coeff_cache(&self);

    #[cfg(feature = "cuda")]
    /// Using the tile information from this [Beam] and frequencies to be used,
    /// return a [BeamCUDA]. This object only needs pointings to calculate beam
    /// response [Jones] matrices.
    ///
    /// # Safety
    ///
    /// This function interfaces directly with the CUDA API. Rust errors attempt
    /// to catch problems but there are no guarantees.
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

    /// Get a pointer to the device tile map. This is necessary to access
    /// de-duplicated beam Jones matrices on the device.
    fn get_tile_map(&self) -> *const i32;

    /// Get a pointer to the device freq map. This is necessary to access
    /// de-duplicated beam Jones matrices on the device.
    fn get_freq_map(&self) -> *const i32;

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
    num_tiles: usize,
}

impl Beam for NoBeam {
    fn get_beam_type(&self) -> BeamType {
        BeamType::None
    }

    fn get_num_tiles(&self) -> usize {
        self.num_tiles
    }

    fn calc_jones(
        &self,
        _azel: AzEl,
        _freq_hz: f64,
        _tile_index: usize,
    ) -> Result<Jones<f64>, BeamError> {
        Ok(Jones::identity())
    }

    fn calc_jones_array(
        &self,
        azels: &[AzEl],
        _freq_hz: f64,
        _tile_index: usize,
    ) -> Result<Vec<Jones<f64>>, BeamError> {
        Ok(vec![Jones::identity(); azels.len()])
    }

    fn find_closest_freq(&self, desired_freq_hz: f64) -> f64 {
        desired_freq_hz
    }

    fn empty_coeff_cache(&self) {}

    #[cfg(feature = "cuda")]
    unsafe fn prepare_cuda_beam(&self, freqs_hz: &[u32]) -> Result<Box<dyn BeamCUDA>, BeamError> {
        let obj = NoBeamCUDA {
            tile_map: DevicePointer::copy_to_device(&vec![0; self.num_tiles])?,
            freq_map: DevicePointer::copy_to_device(&vec![0; freqs_hz.len()])?,
        };
        Ok(Box::new(obj))
    }
}

/// A beam implementation that returns only identity Jones matrices for all beam
/// calculations.
#[cfg(feature = "cuda")]
pub struct NoBeamCUDA {
    tile_map: DevicePointer<i32>,
    freq_map: DevicePointer<i32>,
}

#[cfg(feature = "cuda")]
impl BeamCUDA for NoBeamCUDA {
    unsafe fn calc_jones(
        &self,
        azels: &[AzEl],
    ) -> Result<DevicePointer<Jones<CudaFloat>>, BeamError> {
        let identities: Vec<Jones<CudaFloat>> = vec![Jones::identity(); azels.len()];
        DevicePointer::copy_to_device(&identities).map_err(BeamError::from)
    }

    fn get_beam_type(&self) -> BeamType {
        BeamType::None
    }

    fn get_tile_map(&self) -> *const i32 {
        self.tile_map.get()
    }

    fn get_freq_map(&self) -> *const i32 {
        self.freq_map.get()
    }

    fn get_num_unique_freqs(&self) -> i32 {
        1
    }
}

/// Create a "no beam" object.
pub fn create_no_beam_object(num_tiles: usize) -> Box<dyn Beam> {
    Box::new(NoBeam { num_tiles })
}
