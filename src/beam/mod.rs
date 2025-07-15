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
pub(crate) use fee::FEEBeam;

use std::{path::Path, str::FromStr};

use itertools::Itertools;
use log::debug;
use marlu::{AzEl, Jones};
use ndarray::prelude::*;
use strum::IntoEnumIterator;

#[cfg(any(feature = "cuda", feature = "hip"))]
use crate::gpu::{DevicePointer, GpuFloat};

/// Supported beam types.
#[derive(
    Debug,
    Clone,
    Copy,
    PartialEq,
    Eq,
    strum_macros::Display,
    strum_macros::EnumIter,
    strum_macros::EnumString,
)]
#[allow(clippy::upper_case_acronyms)]
pub enum BeamType {
    /// Fully-embedded element beam.
    #[strum(serialize = "fee")]
    FEE,

    /// a.k.a. [`NoBeam`]. Only returns identity matrices.
    #[strum(serialize = "none")]
    None,
}

impl Default for BeamType {
    fn default() -> Self {
        Self::FEE
    }
}

lazy_static::lazy_static! {
    pub(crate) static ref BEAM_TYPES_COMMA_SEPARATED: String = BeamType::iter().map(|s| s.to_string().to_lowercase()).join(", ");
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
    fn get_dipole_gains(&self) -> Option<ArcArray<f64, Dim<[usize; 2]>>>;

    /// Get the beam file associated with this beam, if there is one.
    fn get_beam_file(&self) -> Option<&Path>;

    /// Calculate the beam-response Jones matrix for an [`AzEl`] direction. The
    /// delays and gains that will used depend on `tile_index`; if not supplied,
    /// ideal dipole delays and gains are used, otherwise `tile_index` accesses
    /// the information provided when this [`Beam`] was created.
    fn calc_jones(
        &self,
        azel: AzEl,
        freq_hz: f64,
        tile_index: Option<usize>,
        latitude_rad: f64,
    ) -> Result<Jones<f64>, BeamError>;

    /// Calculate the beam-response Jones matrices for multiple [`AzEl`]
    /// directions. The delays and gains that will used depend on `tile_index`;
    /// if not supplied, ideal dipole delays and gains are used, otherwise
    /// `tile_index` accesses the information provided when this [`Beam`] was
    /// created.
    fn calc_jones_array(
        &self,
        azels: &[AzEl],
        freq_hz: f64,
        tile_index: Option<usize>,
        latitude_rad: f64,
    ) -> Result<Vec<Jones<f64>>, BeamError>;

    /// Calculate the beam-response Jones matrices for multiple [`AzEl`]
    /// directions, saving the results into the supplied slice. The slice must
    /// have the same length as `azels`. The delays and gains that will used
    /// depend on `tile_index`; if not supplied, ideal dipole delays and gains
    /// are used, otherwise `tile_index` accesses the information provided when
    /// this [`Beam`] was created.
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

    #[cfg(any(feature = "cuda", feature = "hip"))]
    /// Using the tile information from this [`Beam`] and frequencies to be
    /// used, return a [`BeamGpu`]. This object only needs frequencies to
    /// calculate beam response [`Jones`] matrices.
    fn prepare_gpu_beam(&self, freqs_hz: &[u32]) -> Result<Box<dyn BeamGpu>, BeamError>;
}

/// A trait abstracting beam code functions on a GPU.
#[cfg(any(feature = "cuda", feature = "hip"))]
pub trait BeamGpu {
    /// Calculate the Jones matrices for each `az` and `za` direction and
    /// frequency (these were defined when the [`BeamCUDA`] was created). The
    /// results are ordered tile, frequency, direction, slowest to fastest.
    ///
    /// # Safety
    ///
    /// This function interfaces directly with the CUDA/HIP API. Rust errors
    /// attempt to catch problems but there are no guarantees.
    unsafe fn calc_jones_pair(
        &self,
        az_rad: &[GpuFloat],
        za_rad: &[GpuFloat],
        latitude_rad: f64,
        d_jones: *mut std::ffi::c_void,
    ) -> Result<(), BeamError>;

    /// Get the type of beam used to create this [`BeamGpu`].
    fn get_beam_type(&self) -> BeamType;

    /// Get a pointer to the device tile map. This is necessary to access
    /// de-duplicated beam Jones matrices on the device.
    fn get_tile_map(&self) -> *const i32;

    /// Get a pointer to the device freq map. This is necessary to access
    /// de-duplicated beam Jones matrices on the device.
    fn get_freq_map(&self) -> *const i32;

    /// Get the number of de-duplicated tiles associated with this [`BeamGpu`].
    fn get_num_unique_tiles(&self) -> i32;

    /// Get the number of de-duplicated frequencies associated with this
    /// [`BeamGpu`].
    fn get_num_unique_freqs(&self) -> i32;
}

/// An enum to track whether MWA dipole delays are provided and/or necessary.
#[derive(Debug, Clone)]
pub enum Delays {
    /// Delays are fully specified.
    Full(Array2<u32>),

    /// Delays are specified for a single tile. We must assume that these
    /// dipoles apply to all tiles.
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

    /// Parse user-provided dipole delays.
    pub(crate) fn parse(delays: Vec<u32>) -> Result<Delays, BeamError> {
        if delays.len() != 16 || delays.iter().any(|&v| v > 32) {
            return Err(BeamError::BadDelays);
        }
        Ok(Delays::Partial(delays))
    }
}

/// A beam implementation that returns only identity Jones matrices for all beam
/// calculations.
pub(crate) struct NoBeam {
    pub(crate) num_tiles: usize,
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

    fn get_dipole_gains(&self) -> Option<ArcArray<f64, Dim<[usize; 2]>>> {
        None
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

    #[cfg(any(feature = "cuda", feature = "hip"))]
    fn prepare_gpu_beam(&self, freqs_hz: &[u32]) -> Result<Box<dyn BeamGpu>, BeamError> {
        let obj = NoBeamGpu {
            tile_map: DevicePointer::copy_to_device(&vec![0; self.num_tiles])?,
            freq_map: DevicePointer::copy_to_device(&vec![0; freqs_hz.len()])?,
        };
        Ok(Box::new(obj))
    }
}

/// A beam implementation that returns only identity Jones matrices for all beam
/// calculations.
#[cfg(any(feature = "cuda", feature = "hip"))]
pub(crate) struct NoBeamGpu {
    tile_map: DevicePointer<i32>,
    freq_map: DevicePointer<i32>,
}

#[cfg(any(feature = "cuda", feature = "hip"))]
impl BeamGpu for NoBeamGpu {
    unsafe fn calc_jones_pair(
        &self,
        az_rad: &[GpuFloat],
        _za_rad: &[GpuFloat],
        _latitude_rad: f64,
        d_jones: *mut std::ffi::c_void,
    ) -> Result<(), BeamError> {
        #[cfg(feature = "cuda")]
        use cuda_runtime_sys::{
            cudaMemcpy as gpuMemcpy,
            cudaMemcpyKind::cudaMemcpyHostToDevice as gpuMemcpyHostToDevice,
        };
        #[cfg(feature = "hip")]
        use hip_sys::hiprt::{
            hipMemcpy as gpuMemcpy, hipMemcpyKind::hipMemcpyHostToDevice as gpuMemcpyHostToDevice,
        };

        let identities: Vec<Jones<GpuFloat>> = vec![Jones::identity(); az_rad.len()];
        gpuMemcpy(
            d_jones,
            identities.as_ptr().cast(),
            identities.len() * std::mem::size_of::<Jones<GpuFloat>>(),
            gpuMemcpyHostToDevice,
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

pub fn create_beam_object(
    beam_type: Option<&str>,
    num_tiles: usize,
    dipole_delays: Delays,
) -> Result<Box<dyn Beam>, BeamError> {
    let beam_type = match (
        beam_type,
        beam_type.and_then(|b| BeamType::from_str(b).ok()),
    ) {
        (None, _) => BeamType::default(),
        (Some(_), Some(b)) => b,
        (Some(s), None) => return Err(BeamError::Unrecognised(s.to_string())),
    };

    match beam_type {
        BeamType::None => {
            debug!("Setting up a \"NoBeam\" object");
            Ok(Box::new(NoBeam { num_tiles }))
        }

        BeamType::FEE => {
            debug!("Setting up a FEE beam object");

            // Check that the delays are sensible.
            validate_delays(&dipole_delays, num_tiles)?;

            // Set up the FEE beam struct from the `MWA_BEAM_FILE` environment
            // variable.
            Ok(Box::new(FEEBeam::new_from_env(
                num_tiles,
                dipole_delays,
                None,
            )?))
        }
    }
}

/// Assume that the dipole delays for all tiles is the same as the delays for
/// one tile.
fn partial_to_full(delays: Vec<u32>, num_tiles: usize) -> Array2<u32> {
    let mut out = Array2::zeros((num_tiles, 16));
    let d = Array1::from(delays);
    out.outer_iter_mut().for_each(|mut tile_delays| {
        tile_delays.assign(&d);
    });
    out
}

fn validate_delays(delays: &Delays, num_tiles: usize) -> Result<(), BeamError> {
    match delays {
        Delays::Partial(v) => {
            if v.len() != 16 || v.iter().any(|&v| v > 32) {
                return Err(BeamError::BadDelays);
            }
        }

        Delays::Full(a) => {
            if a.len_of(Axis(1)) != 16 || a.iter().any(|&v| v > 32) {
                return Err(BeamError::BadDelays);
            }
            if a.len_of(Axis(0)) != num_tiles {
                return Err(BeamError::InconsistentDelays {
                    num_rows: a.len_of(Axis(0)),
                    num_tiles,
                });
            }
        }
    }

    Ok(())
}
