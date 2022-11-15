// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Code to abstract beam calculations.
//!
//! [`Beam`] is a trait detailing how to perform various beam-related tasks. By
//! making this trait, we can neatly abstract over multiple beam codes,
//! including a simple [`NoBeam`] type (which just returns identity matrices).
//!
//! Note that (where applicable) `norm_to_zenith` is always true; the
//! implication being that a sky-model source's brightness is always assumed to
//! be correct when at zenith.

mod error;
mod fee;
#[cfg(test)]
mod tests;

pub(crate) use error::BeamError;
pub use fee::create_fee_beam_object;

use std::path::Path;

use marlu::{AzEl, Jones};
use ndarray::prelude::*;

#[cfg(feature = "cuda")]
use crate::cuda::{CudaFloat, DevicePointer};

/// Supported beam types.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(clippy::upper_case_acronyms)]
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

    /// Get the dipole delays associated with this beam.
    fn get_dipole_delays(&self) -> Option<ArcArray<u32, Dim<[usize; 2]>>>;

    /// Get the ideal dipole delays associated with this beam.
    fn get_ideal_dipole_delays(&self) -> Option<[u32; 16]>;

    /// Get the dipole gains used in this beam object. The rows correspond to
    /// tiles and there are 32 columns, one for each dipole. The first 16 values
    /// are for X dipoles, the second 16 are for Y dipoles.
    fn get_dipole_gains(&self) -> ArcArray<f64, Dim<[usize; 2]>>;

    /// Get the beam file associated with this beam, if there is one.
    fn get_beam_file(&self) -> Option<&Path>;

    /// Calculate the beam-response Jones matrix for an [`AzEl`] direction. The
    /// pointing information is not needed because it was provided when `self`
    /// was created.
    fn calc_jones(
        &self,
        azel: AzEl,
        freq_hz: f64,
        tile_index: Option<usize>,
        latitude_rad: f64,
    ) -> Result<Jones<f64>, BeamError>;

    /// Calculate the beam-response Jones matrices for multiple [`AzEl`]
    /// directions. The pointing information is not needed because it was
    /// provided when `self` was created.
    fn calc_jones_array(
        &self,
        azels: &[AzEl],
        freq_hz: f64,
        tile_index: Option<usize>,
        latitude_rad: f64,
    ) -> Result<Vec<Jones<f64>>, BeamError>;

    /// Calculate the beam-response Jones matrices for multiple [`AzEl`]
    /// directions, saving the results into the supplied slice. The slice must
    /// have the same length as `azels`. The pointing information is not needed
    /// because it was provided when `self` was created.
    fn calc_jones_array_inner(
        &self,
        azels: &[AzEl],
        freq_hz: f64,
        tile_index: Option<usize>,
        latitude_rad: f64,
        results: &mut [Jones<f64>],
    ) -> Result<(), BeamError>;

    /// Given a frequency in Hz, find the closest frequency that the beam code
    /// is defined for. An example of when this is important is with the FEE
    /// beam code, which can only give beam responses at specific frequencies.
    /// On the other hand, the analytic beam can be used at any frequency.
    fn find_closest_freq(&self, desired_freq_hz: f64) -> f64;

    /// If this [`Beam`] supports it, empty the coefficient cache.
    fn empty_coeff_cache(&self);

    #[cfg(feature = "cuda")]
    /// Using the tile information from this [`Beam`] and frequencies to be
    /// used, return a [`BeamCUDA`]. This object only needs pointings to
    /// calculate beam response [`Jones`] matrices.
    fn prepare_cuda_beam(&self, freqs_hz: &[u32]) -> Result<Box<dyn BeamCUDA>, BeamError>;
}

/// A trait abstracting beam code functions on a CUDA-capable device.
#[cfg(feature = "cuda")]
pub trait BeamCUDA {
    /// Calculate the Jones matrices for each `az` and `za` direction. The
    /// pointing information is not needed because it was provided when `self`
    /// was created.
    ///
    /// # Safety
    ///
    /// This function interfaces directly with the CUDA API. Rust errors attempt
    /// to catch problems but there are no guarantees.
    unsafe fn calc_jones_pair(
        &self,
        az_rad: &[CudaFloat],
        za_rad: &[CudaFloat],
        latitude_rad: f64,
        d_jones: *mut std::ffi::c_void,
    ) -> Result<(), BeamError>;

    /// Get the type of beam.
    fn get_beam_type(&self) -> BeamType;

    /// Get a pointer to the device tile map. This is necessary to access
    /// de-duplicated beam Jones matrices on the device.
    fn get_tile_map(&self) -> *const i32;

    /// Get a pointer to the device freq map. This is necessary to access
    /// de-duplicated beam Jones matrices on the device.
    fn get_freq_map(&self) -> *const i32;

    /// Get the number of de-duplicated tiles associated with this [`BeamCUDA`].
    fn get_num_unique_tiles(&self) -> i32;

    /// Get the number of de-duplicated frequencies associated with this
    /// [`BeamCUDA`].
    fn get_num_unique_freqs(&self) -> i32;
}

/// An enum to track whether MWA dipole delays are provided and/or necessary.
#[derive(Debug, Clone)]
pub enum Delays {
    /// Delays are fully specified.
    Full(Array2<u32>),

    /// Delays are specified for a single tile. If this can't be refined, then
    /// we must assume that these dipoles apply to all tiles.
    Partial(Vec<u32>),
}

impl Delays {
    /// The delays of some tiles could contain 32 (which means that that
    /// particular dipole is "dead"). It is sometimes useful to get the "ideal"
    /// dipole delays; i.e. what the delays for each tile would be if all
    /// dipoles were alive.
    pub(crate) fn get_ideal_delays(&self) -> [u32; 16] {
        let mut ideal_delays = [32; 16];
        match self {
            Delays::Partial(v) => {
                // There may be 32 elements per row - 16 for X dipoles, 16 for
                // Y. We only want 16, take the mod of the column index.
                v.iter().enumerate().for_each(|(i, &elem)| {
                    ideal_delays[i % 16] = elem;
                });
            }
            Delays::Full(a) => {
                // Iterate over all rows until none of the delays are 32.
                for row in a.outer_iter() {
                    row.iter().enumerate().for_each(|(i, &col)| {
                        let ideal_delay = ideal_delays.get_mut(i % 16).unwrap();

                        // The delays should be the same, modulo some being
                        // 32 (i.e. that dipole's component is dead). This
                        // code will pick the smaller delay of the two
                        // (delays are always <=32). If both are 32, there's
                        // nothing else that can be done.
                        *ideal_delay = (*ideal_delay).min(col);
                    });
                    if ideal_delays.iter().all(|&e| e < 32) {
                        break;
                    }
                }
            }
        }
        ideal_delays
    }

    /// Some tiles' delays might contain 32s (i.e. dead dipoles), and we might
    /// want to ignore that. Take the ideal delays and replace all tiles' delays
    /// with them.
    pub(crate) fn set_to_ideal_delays(&mut self) {
        let ideal_delays = self.get_ideal_delays();
        match self {
            // In this case, the delays are the ideal delays.
            Delays::Full(a) => {
                let ideal_delays = ArrayView1::from(&ideal_delays);
                a.outer_iter_mut().for_each(|mut r| r.assign(&ideal_delays));
            }

            // In this case, no meaningful change can be made.
            Delays::Partial { .. } => (),
        }
    }
}

/// A beam implementation that returns only identity Jones matrices for all beam
/// calculations.
pub(crate) struct NoBeam {
    num_tiles: usize,
}

impl Beam for NoBeam {
    fn get_beam_type(&self) -> BeamType {
        BeamType::None
    }

    fn get_num_tiles(&self) -> usize {
        self.num_tiles
    }

    fn get_ideal_dipole_delays(&self) -> Option<[u32; 16]> {
        None
    }

    fn get_dipole_delays(&self) -> Option<ArcArray<u32, Dim<[usize; 2]>>> {
        None
    }

    fn get_dipole_gains(&self) -> ArcArray<f64, Dim<[usize; 2]>> {
        Array2::ones((self.num_tiles, 32)).into_shared()
    }

    fn get_beam_file(&self) -> Option<&Path> {
        None
    }

    fn calc_jones(
        &self,
        _azel: AzEl,
        _freq_hz: f64,
        _tile_index: Option<usize>,
        _latitude_rad: f64,
    ) -> Result<Jones<f64>, BeamError> {
        Ok(Jones::identity())
    }

    fn calc_jones_array(
        &self,
        azels: &[AzEl],
        _freq_hz: f64,
        _tile_index: Option<usize>,
        _latitude_rad: f64,
    ) -> Result<Vec<Jones<f64>>, BeamError> {
        Ok(vec![Jones::identity(); azels.len()])
    }

    fn calc_jones_array_inner(
        &self,
        _azels: &[AzEl],
        _freq_hz: f64,
        _tile_index: Option<usize>,
        _latitude_rad: f64,
        results: &mut [Jones<f64>],
    ) -> Result<(), BeamError> {
        results.fill(Jones::identity());
        Ok(())
    }

    fn find_closest_freq(&self, desired_freq_hz: f64) -> f64 {
        desired_freq_hz
    }

    fn empty_coeff_cache(&self) {}

    #[cfg(feature = "cuda")]
    fn prepare_cuda_beam(&self, freqs_hz: &[u32]) -> Result<Box<dyn BeamCUDA>, BeamError> {
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
pub(crate) struct NoBeamCUDA {
    tile_map: DevicePointer<i32>,
    freq_map: DevicePointer<i32>,
}

#[cfg(feature = "cuda")]
impl BeamCUDA for NoBeamCUDA {
    unsafe fn calc_jones_pair(
        &self,
        az_rad: &[CudaFloat],
        _za_rad: &[CudaFloat],
        _latitude_rad: f64,
        d_jones: *mut std::ffi::c_void,
    ) -> Result<(), BeamError> {
        let identities: Vec<Jones<CudaFloat>> = vec![Jones::identity(); az_rad.len()];
        cuda_runtime_sys::cudaMemcpy(
            d_jones,
            identities.as_ptr().cast(),
            identities.len() * std::mem::size_of::<Jones<CudaFloat>>(),
            cuda_runtime_sys::cudaMemcpyKind::cudaMemcpyHostToDevice,
        );
        Ok(())
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

    fn get_num_unique_tiles(&self) -> i32 {
        1
    }

    fn get_num_unique_freqs(&self) -> i32 {
        1
    }
}

/// Create a "no beam" object.
pub fn create_no_beam_object(num_tiles: usize) -> Box<dyn Beam> {
    Box::new(NoBeam { num_tiles })
}
