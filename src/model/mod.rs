// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Code to generate sky-model visibilities.

mod cpu;
#[cfg(feature = "cuda")]
mod cuda;
mod error;
#[cfg(test)]
mod integration_tests;
pub(crate) mod shapelets;
#[cfg(test)]
mod tests;

use cpu::SkyModellerCpu;
#[cfg(feature = "cuda")]
use cuda::SkyModellerCuda;
pub(crate) use error::ModelError;

use std::collections::HashSet;

use hifitime::{Duration, Epoch};
use marlu::{c32, Jones, RADec, XyzGeodetic, UVW};
use ndarray::{Array2, ArrayViewMut2};

use crate::{beam::Beam, srclist::SourceList, Polarisations};

#[derive(Debug, Clone)]
pub(crate) enum ModellerInfo {
    /// The CPU is used for modelling. This always uses double-precision floats
    /// when modelling.
    Cpu,

    /// A CUDA-capable device is used for modelling. The precision depends on
    /// the compile features used.
    #[cfg(feature = "cuda")]
    Cuda {
        device_info: crate::cuda::CudaDeviceInfo,
        driver_info: crate::cuda::CudaDriverInfo,
    },
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
    /// beam-response calculation or a CUDA error (if using CUDA functionality).
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
    /// beam-response calculation or a CUDA error (if using CUDA functionality).
    fn model_timestep_with(
        &self,
        timestamp: Epoch,
        vis_fb: ArrayViewMut2<Jones<f32>>,
    ) -> Result<Vec<UVW>, ModelError>;
}

/// Create a [`SkyModeller`] trait object that generates sky-model visibilities
/// on either the CPU or a CUDA-compatible GPU. This function conveniently
/// provides either a [`SkyModellerCpu`] or [`SkyModellerCuda`] depending on how
/// `hyperdrive` was compiled and the `use_cpu_for_modelling` flag.
///
/// # Errors
///
/// This function will return an error if CUDA mallocs and copies can't be
/// executed, or if there was a problem in setting up a `BeamCUDA`.
#[allow(clippy::too_many_arguments)]
pub fn new_sky_modeller<'a>(
    #[cfg(feature = "cuda")] use_cpu_for_modelling: bool,
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
    cfg_if::cfg_if! {
        if #[cfg(feature = "cuda")] {
            if use_cpu_for_modelling {
                Ok(Box::new(SkyModellerCpu::new(
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
                )))
            } else {
                let modeller = SkyModellerCuda::new(
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
        } else {
            Ok(Box::new(SkyModellerCpu::new(
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
            )))
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
