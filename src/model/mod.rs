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
#[cfg(test)]
mod tests;

pub(crate) use cpu::SkyModellerCpu;
#[cfg(feature = "cuda")]
pub(crate) use cuda::SkyModellerCuda;
pub(crate) use error::ModelError;

use std::collections::HashSet;

use hifitime::{Duration, Epoch};
use indexmap::{indexmap, IndexMap};
use marlu::{Jones, RADec, XyzGeodetic, UVW};
use ndarray::ArrayViewMut2;

use crate::{
    beam::Beam,
    cli::peel::SourceIonoConsts,
    srclist::{ComponentList, Source, SourceList},
};

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
pub trait SkyModeller<'a>: Send {
    /// Update the sky model associated with the `SkyModeller`. All old source
    /// information is destroyed.
    /// TODO(dev): something like impl IntoIterator<Item = &SourceComponent>
    fn update_source_list(
        &mut self,
        source_list: &SourceList,
        phase_centre: RADec,
    ) -> Result<(), ModelError>;

    /// Update the sky model associated with the `SkyModeller`, with only a
    /// single source. All old source information is destroyed. This is mostly
    /// useful for peeling.
    fn update_with_a_source(
        &mut self,
        source: &Source,
        phase_centre: RADec,
    ) -> Result<(), ModelError> {
        let source_list = SourceList::from(indexmap! {
            "source".into() => source.clone(),
        });
        self.update_source_list(&source_list, phase_centre)
    }

    /// Generate sky-model visibilities for a single timestep. The [`UVW`]
    /// coordinates used in generating the visibilities are returned.
    ///
    /// `vis_model_slice`: A mutable view into an `ndarray`. Rather than
    /// returning an array from this function, modelled visibilities are written
    /// into this array. This slice *must* have dimensions `[n1][n2]`, where
    /// `n1` is number of unflagged frequencies and `n2` is the number of
    /// unflagged cross correlation baselines.
    ///
    /// `timestamp`: The [`hifitime::Epoch`] struct used to determine what this
    /// timestep corresponds to.
    ///
    /// # Errors
    ///
    /// This function will return an error if there was a problem with
    /// beam-response calculation or a CUDA error (if using CUDA functionality).
    fn model_timestep(
        &mut self,
        vis_model_slice: ArrayViewMut2<Jones<f32>>,
        timestamp: Epoch,
    ) -> Result<Vec<UVW>, ModelError>;
}

/// Create a [`SkyModeller`] trait object that generates sky-model visibilities
/// on either the CPU or a CUDA-compatible GPU. This function conveniently uses
/// either [new_cpu_sky_modeller] or [new_cuda_sky_modeller] depending on how
/// `hyperdrive` was compiled and the `use_cpu_for_modelling` flag.
///
/// # Errors
///
/// This function will return an error if CUDA mallocs and copies can't be
/// executed, or if there was a problem in setting up a `BeamCUDA`.
///
/// # Safety
///
/// If not using the CPU modeller, this function wraps an `unsafe` call, which
/// interfaces directly with the CUDA API. Rust errors attempt to catch problems
/// but there are no guarantees.
#[allow(clippy::too_many_arguments)]
pub fn new_sky_modeller<'a>(
    #[cfg(feature = "cuda")] use_cpu_for_modelling: bool,
    beam: &'a dyn Beam,
    source_list: &SourceList,
    unflagged_tile_xyzs: &'a [XyzGeodetic],
    unflagged_fine_chan_freqs: &'a [f64],
    flagged_tiles: &'a HashSet<usize>,
    phase_centre: RADec,
    array_longitude_rad: f64,
    array_latitude_rad: f64,
    dut1: Duration,
    apply_precession: bool,
    iono_consts: &'a IndexMap<String, SourceIonoConsts>,
) -> Result<Box<dyn SkyModeller<'a> + 'a>, ModelError> {
    cfg_if::cfg_if! {
        if #[cfg(feature = "cuda")] {
            if use_cpu_for_modelling {
                let components = ComponentList::new(source_list, unflagged_fine_chan_freqs, phase_centre);
                let maps = crate::math::TileBaselineFlags::new(
                    unflagged_tile_xyzs.len() + flagged_tiles.len(),
                    flagged_tiles.clone(),
                );

                Ok(Box::new(SkyModellerCpu {
                    beam,
                    phase_centre,
                    array_longitude: array_longitude_rad,
                    array_latitude: array_latitude_rad,
                    dut1,
                    precess: apply_precession,
                    unflagged_fine_chan_freqs,
                    unflagged_tile_xyzs,
                    flagged_tiles,
                    unflagged_baseline_to_tile_map: maps.unflagged_cross_baseline_to_tile_map,
                    components,
                }))
            } else {
                unsafe {
                    let modeller = SkyModellerCuda::new(
                        beam,
                        source_list,
                        unflagged_tile_xyzs,
                        unflagged_fine_chan_freqs,
                        flagged_tiles,
                        phase_centre,
                        array_longitude_rad,
                        array_latitude_rad,
                        dut1,
                        apply_precession,
                        iono_consts,
                    )?;
                    Ok(Box::new(modeller))
                }
            }
        } else {
            let components = ComponentList::new(source_list, unflagged_fine_chan_freqs, phase_centre);
            let maps = crate::math::TileBaselineFlags::new(
                unflagged_tile_xyzs.len() + flagged_tiles.len(),
                flagged_tiles.clone(),
            );

            Ok(Box::new(SkyModellerCpu {
                beam,
                phase_centre,
                array_longitude: array_longitude_rad,
                array_latitude: array_latitude_rad,
                dut1,
                precess: apply_precession,
                unflagged_fine_chan_freqs,
                unflagged_tile_xyzs,
                flagged_tiles,
                unflagged_baseline_to_tile_map: maps.unflagged_cross_baseline_to_tile_map,
                components,
            }))
        }
    }
}
