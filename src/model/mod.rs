// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Code to generate sky-model visibilities.

mod cpu;
mod error;
#[cfg(any(feature = "cuda", feature = "hip"))]
mod gpu;
pub(crate) mod shapelets;
#[cfg(test)]
mod tests;

pub use cpu::SkyModellerCpu;
pub(crate) use error::ModelError;
#[cfg(any(feature = "cuda", feature = "hip"))]
pub use gpu::SkyModellerGpu;

use std::collections::HashSet;

use hifitime::{Duration, Epoch};
use marlu::{c32, Jones, RADec, XyzGeodetic, UVW};
use ndarray::{Array2, ArrayViewMut2};

use crate::{beam::Beam, context::Polarisations, srclist::SourceList, MODEL_DEVICE};

#[derive(Debug, Clone, Copy)]
pub enum ModelDevice {
    /// The CPU is used for modelling. This always uses double-precision floats
    /// when modelling.
    Cpu,

    /// A CUDA- or HIP-capable device is used for modelling. The precision
    /// depends on the compile features used.
    #[cfg(any(feature = "cuda", feature = "hip"))]
    Gpu,
}

impl ModelDevice {
    pub(crate) fn get_precision(self) -> &'static str {
        match self {
            ModelDevice::Cpu => "double",

            #[cfg(feature = "gpu-single")]
            ModelDevice::Gpu => "single",

            #[cfg(all(any(feature = "cuda", feature = "hip"), not(feature = "gpu-single")))]
            ModelDevice::Gpu => "double",
        }
    }

    /// Get a formatted string with information on the device used for
    /// modelling.
    pub(crate) fn get_device_info(self) -> Result<String, DeviceError> {
        match self {
            ModelDevice::Cpu => Ok(get_cpu_info()),

            #[cfg(any(feature = "cuda", feature = "hip"))]
            ModelDevice::Gpu => {
                let (device_info, driver_info) = crate::gpu::get_device_info()?;
                #[cfg(feature = "cuda")]
                let device_type = "CUDA";
                #[cfg(feature = "hip")]
                let device_type = "HIP";
                Ok(format!(
                    "{} (capability {}, {} MiB), {device_type} driver {}, runtime {}",
                    device_info.name,
                    device_info.capability,
                    device_info.total_global_mem,
                    driver_info.driver_version,
                    driver_info.runtime_version
                ))
            }
        }
    }
}

#[derive(thiserror::Error, Debug)]
pub(crate) enum DeviceError {
    #[cfg(any(feature = "cuda", feature = "hip"))]
    #[error(transparent)]
    Gpu(#[from] crate::gpu::GpuError),
}

/// Get a formatted string with information on the device used for modelling.
// TODO: Is there a way to get the name of the CPU without some crazy
// dependencies?
pub(crate) fn get_cpu_info() -> String {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        // Non-exhaustive but perhaps most-interesting CPU features.
        let avx = std::arch::is_x86_feature_detected!("avx");
        let avx2 = std::arch::is_x86_feature_detected!("avx2");
        let avx512 = std::arch::is_x86_feature_detected!("avx512f");

        match (avx512, avx2, avx) {
            (true, _, _) => {
                format!("{} CPU (AVX512 available)", std::env::consts::ARCH)
            }
            (false, true, _) => {
                format!("{} CPU (AVX2 available)", std::env::consts::ARCH)
            }
            (false, false, true) => {
                format!("{} CPU (AVX available)", std::env::consts::ARCH)
            }
            (false, false, false) => {
                format!("{} CPU (AVX unavailable!)", std::env::consts::ARCH)
            }
        }
    }

    #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
    Ok(format!("{} CPU", std::env::consts::ARCH));
}

/// An object that simulates sky-model visibilities.
pub trait SkyModeller<'a> {
    /// Generate sky-model visibilities for a single timestep. The visibilities
    /// as well as the [`UVW`] coordinates used in generating the visibilities
    /// are returned. The visibilities are ordered by frequency and then
    /// baseline.
    ///
    /// This function is not as efficient as
    /// [`SkyModeller::model_timestep_with`], because that function does not
    /// need to allocate its own buffer for visibilities.
    ///
    /// `timestamp`: The [`hifitime::Epoch`] struct used to determine what this
    /// timestep corresponds to.
    ///
    /// # Errors
    ///
    /// This function will return an error if there was a problem with
    /// beam-response calculation or a CUDA/HIP error (if using CUDA/HIP
    /// functionality).
    fn model_timestep(
        &self,
        timestamp: Epoch,
    ) -> Result<(Array2<Jones<f32>>, Vec<UVW>), ModelError>;

    /// Generate sky-model visibilities for a single timestep. The [`UVW`]
    /// coordinates used in generating the visibilities are returned, but the
    /// new visibilities are added to `vis_fb`. The visibilities are ordered by
    /// frequency and then baseline.
    ///
    /// `timestamp`: The [`hifitime::Epoch`] struct used to determine what this
    /// timestep corresponds to.
    ///
    /// `vis_fb`: A mutable view into an `ndarray`. The view *must* be
    /// contiguous in memory. Rather than returning an array from this function,
    /// modelled visibilities are added to this array. *The view is not cleared
    /// as part of this function.* This view *must* have dimensions `[n1][n2]`,
    /// where `n1` is number of unflagged frequencies and `n2` is the number of
    /// unflagged cross correlation baselines.
    ///
    /// # Errors
    ///
    /// This function will return an error if there was a problem with
    /// beam-response calculation or a CUDA/HIP error (if using CUDA/HIP
    /// functionality).
    fn model_timestep_with(
        &self,
        timestamp: Epoch,
        vis_fb: ArrayViewMut2<Jones<f32>>,
    ) -> Result<Vec<UVW>, ModelError>;
}

/// Create a [`SkyModeller`] trait object that generates sky-model visibilities
/// on the CPU, a CUDA-compatible GPU or a HIP-compatible GPU, depending on the
/// value of [`MODEL_DEVICE`].
///
/// # Errors
///
/// This function will return an error if GPU mallocs and copies can't be
/// executed, or if there was a problem in setting up a `BeamGpu`.
#[allow(clippy::too_many_arguments)]
pub fn new_sky_modeller<'a>(
    beam: &'a dyn Beam,
    source_list: &SourceList,
    pols: Polarisations,
    unflagged_tile_xyzs: &'a [XyzGeodetic],
    unflagged_fine_chan_freqs: &'a [f64],
    flagged_tiles: &'a HashSet<usize>,
    phase_centre: RADec,
    array_longitude_rad: f64,
    array_latitude_rad: f64,
    dut1: Duration,
    apply_precession: bool,
) -> Result<Box<dyn SkyModeller<'a> + 'a>, ModelError> {
    match MODEL_DEVICE.load() {
        ModelDevice::Cpu => Ok(Box::new(SkyModellerCpu::new(
            beam,
            source_list,
            pols,
            unflagged_tile_xyzs,
            unflagged_fine_chan_freqs,
            flagged_tiles,
            phase_centre,
            array_longitude_rad,
            array_latitude_rad,
            dut1,
            apply_precession,
        ))),

        #[cfg(any(feature = "cuda", feature = "hip"))]
        ModelDevice::Gpu => {
            let modeller = SkyModellerGpu::new(
                beam,
                source_list,
                pols,
                unflagged_tile_xyzs,
                unflagged_fine_chan_freqs,
                flagged_tiles,
                phase_centre,
                array_longitude_rad,
                array_latitude_rad,
                dut1,
                apply_precession,
            )?;
            Ok(Box::new(modeller))
        }
    }
}

/// Set any unavailable polarisations to zero.
fn mask_pols(mut vis: ArrayViewMut2<Jones<f32>>, pols: Polarisations) {
    // Don't do anything if all pols are available.
    if matches!(pols, Polarisations::XX_XY_YX_YY) {
        return;
    }

    let func = |j: &mut Jones<f32>| {
        *j = match pols {
            Polarisations::XX_XY_YX_YY => *j,
            Polarisations::XX => {
                Jones::from([j[0], c32::default(), c32::default(), c32::default()])
            }
            Polarisations::YY => {
                Jones::from([c32::default(), c32::default(), c32::default(), j[3]])
            }
            Polarisations::XX_YY => Jones::from([j[0], c32::default(), c32::default(), j[3]]),
            Polarisations::XX_YY_XY => Jones::from([j[0], j[1], c32::default(), j[3]]),
        }
    };

    vis.iter_mut().for_each(func);
}
